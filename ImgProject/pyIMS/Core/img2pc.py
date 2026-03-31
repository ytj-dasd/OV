from typing import List, Dict
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
import pickle
from tqdm import tqdm


def back_proj(pre_infer_path: Path, inferred_path: Path, point_num: int, classes_num: int):
    point_idx_img_paths = list(pre_infer_path.glob('*.pkl'))
    point_idx_img_paths.sort()
    points_logits = np.zeros((point_num, classes_num + 1), dtype= np.float32)
    one_hot_mat = np.eye(classes_num + 1)

    # iterate evey img
    for img_idx in tqdm(range(len(point_idx_img_paths))):
        # read point_idx_img
        point_idx_img_path = point_idx_img_paths[img_idx]
        with open(point_idx_img_path, "rb") as f:
            point_idx_img = pickle.load(f)
            point_idx_img = point_idx_img.flatten()
        
        # read logits_img
        name = point_idx_img_path.name.replace('point-idx', 'logits')
        file_path = inferred_path / name
        with open(file_path, 'rb') as f:
            pred_img = pickle.load(f) + 1 # ! the pred output of dinov2 is based on 150 classes, but we added a none class at 0, that's why all the pred should add 1.
            pred_img = one_hot_mat[pred_img.flatten()]
        
        mask = (point_idx_img != -1)
        points_logits[point_idx_img[mask]] += pred_img[mask]
        # if img_idx == 0: break

    points_pred = points_logits.argmax(axis= -1)
    return points_pred.astype(np.uint8)


def weight_func(dist_img):
    """
    Get the weight of each pixel according to the distance of the point cloud.

    Args:
        dist_img (np.ndarray): (batch_size x height x width) array of pixels' distance.

    Returns:
        np.ndarray: (batch_size x height x width) array of pixels' weight.
    """
    weights_img = 1 / (dist_img + 1e-6)
    return weights_img


def one_frame_weighted_mapping(points_logits, logits_img, pts_img_indices, pts_indices, dist_img= None, weight_func= weight_func):
    """
    Get logits for point cloud by the logits of each pixel according to correspending point-idx and weight.

    Args:
        points_logits (np.ndarray): (N x classes_num) array of points' logits.
        logits_img (np.ndarray): (batch_size x height x width x classes_num) array of pixels' logits.
        point_idx_img (np.ndarray): (batch_size x height x width) array of pixels' point_idx.
        weights_img (np.ndarray): (batch_size x height x width) array of pixels' weight.

    """
    
    flat_logits_img = logits_img.reshape(-1, logits_img.shape[-1])
    if dist_img is not None:
        weights_img = weight_func(dist_img)
        flat_logits_img = weights_img.reshape(-1, 1) * flat_logits_img
        
    points_logits[pts_indices] += flat_logits_img[pts_img_indices]
    return 


def non_labeled_points_labeling(points, points_pred):
    pass

def filter(points, points_pred):
    pass

def geometric_constraint_refine(points, points_pred):
    pass


class ImgModel():
    def __init__(self, onnx_fp):
        self.sess = ort.InferenceSession(
            onnx_fp,
            providers=[
                "CUDAExecutionProvider",
            ]
        )        
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        
    def __call__(self, imgs: NDArray[np.uint8]) -> NDArray[np.float32]:
        """inference process for inputing model and imgs and then outputing logits_imgs

        Args:
            imgs (_type_): img in RGB

        Returns:
            _type_: logits_imgs' shape: (batch_size, height, width, classes_num)
        """


        imgs = imgs.astype(np.float32)[:, :, ::-1]  # (height x width x 3)
        outputs: NDArray = self.sess.run([self.output_name], {self.input_name: imgs})[0]    # (batch_size x height x width x classes_num)
        logits_imgs = outputs.transpose(0, 2, 3, 1)

        return logits_imgs


class PcImgs():
    names: List[str]
    imgs: List[NDArray[np.uint8]]
    info: List[Dict[str, NDArray[np.int64]]]

    def __init__(self,names:List[str], imgs: List[NDArray[np.uint8]], info: List[Dict[str, NDArray[np.int64]]]):
        self.names = names
        self.imgs = imgs
        self.info = info

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return PcImgs(self.names[idx], self.imgs[idx], self.info[idx])
        else:
            return self.names[idx], self.imgs[idx], self.info[idx]
        
    def numpy_imgs(self):
        """convert imgs to numpy array"""
        return np.stack(self.imgs, axis= 0)


def img_inference(self, pc_imgs: PcImgs, num_points: int, is_save_img: bool = False):
    """img semantic segmentation inference and back-projection to point cloud.

    Args:
        imgs (_type_): _description_
        infos (_type_): _description_
    """
    batch_size = 3 
    img_num = len(pc_imgs)
    batch_num = img_num // batch_size
    batch_num = batch_num + 1 if img_num % batch_size != 0 else batch_num

    batches = [pc_imgs[i * batch_size: (i + 1) * batch_size] for i in range(batch_num)]
    points_logits = np.zeros((num_points, self.img_label_num), dtype= np.float32)
    for batch in tqdm(batches):
        imgs = batch.numpy_imgs()
        logits_imgs = self.img_model(imgs)

        
        for idx in range(len(batch)):
            img = logits_imgs[idx]
            name = batch.names[idx]
            info = batch.info[idx]

            if is_save_img:
                pred_mask = img.argmax(axis= -1)
                pred_mask = vectorized_mapping_func(pred_mask, self.img_label_map)
                pred_img = color_mapping(pred_mask, PCSS_COLORMAP)
                pred_img = Image.fromarray(pred_img, mode= 'RGB')
                pred_img.save(f'./imgs/{name}.png')

            dist_img = info['dist_img']
            pts_img_indices = info['pts_img_indices']
            pts_indices = info['pts_indices']

            one_frame_weighted_mapping(points_logits, img, pts_img_indices, pts_indices, dist_img)

    points_pred = points_logits.argmax(axis= -1)   # each is shape (N,)
    max_vals = points_logits.max(axis= -1)
    confidence = max_vals / points_logits.sum(axis= -1)
    # points_pred = points_logits.argsort(axis= -1)[:, :5]
    return points_pred.astype(np.uint8), confidence.astype(np.float32)   