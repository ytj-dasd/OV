import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
import json
from matplotlib.cm import get_cmap


    
def end_draw(points, info, height, stride, output_path):
    print('enter')
    points = np.array(points)
    ratio = info['ratio']
    left_top = info['left-top']
    points = points / ratio @ np.array([1, 0, 0, -1]).reshape(2, 2) + np.array([left_top[0], left_top[1]])
    points = points.tolist()

    station_config = create_stations_by_trajk(points, height= height, stride= stride)
    
    with open(output_path, mode='w') as f:
        json.dump(station_config, f)


# def get_ortho_img(points: NDArray, colors: NDArray, width: int= 1920):
#     min_bound = points.min(axis=0)
#     max_bound = points.max(axis=0)

#     # * get the position and pos of orth camera
#     left_top = np.array([min_bound[0], max_bound[1], max_bound[2]])
#     right_bot = np.array([max_bound[0], min_bound[1], min_bound[2]])
#     z_height = max_bound[2] - min_bound[2]
#     rotation_mat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).T
#     trans_mat = np.eye(4, 4)
#     trans_mat[:3, :3] = rotation_mat
#     trans_mat[:3, 3] = left_top
#     trans_mat = np.linalg.inv(trans_mat)

#     # * trans points to crs of orth camera
#     num_points = points.shape[0]
#     points_hom = np.hstack([points, np.ones((num_points, 1))])
#     trans_mat = trans_mat[:3,:]
#     points_cam = (trans_mat @ points_hom.T).T
#     depths = points_cam[:, 2].reshape(-1, 1) / z_height
#     points_cam = points_cam[:, :2]


#     ratio = width / (max_bound[0] - min_bound[0])   # * (width_img / real )
#     height = int((max_bound[1] - min_bound[1]) * ratio)

#     resize_mat = np.array([ratio, 0, 0, ratio]).reshape(2, 2)
#     points_img = (resize_mat @ points_cam.T).T


#     ortho_img = np.zeros((height, width, 3), dtype=np.uint8)
#     height_img = np.zeros((height, width, 3), dtype= np.uint8)
#     depth_img = np.full((height, width), -1, dtype=np.float32)

#     for idx, pt in tqdm(enumerate(points_img)):
#         x, y = int(pt[0]), int(pt[1])
#         if 0 <= x < width and 0 <= y < height:
#             pixel_depth = depth_img[y, x]
#             point_depth = depths[idx].item()

#             if pixel_depth == -1:
#                 depth_img[y, x] = point_depth
#             elif point_depth < pixel_depth:
#                 depth_img[y, x] = point_depth
#             else:
#                 continue

#             color =  colors[idx].astype(np.uint8).tolist()
#             # cv2.circle(ortho_img, (x, y), radius=2, color=color, thickness=-1)
#             ortho_img[y, x] = color
#             color = (np.array([255, 255, 255]) * point_depth).astype(np.uint8).tolist()
#             height_img[y, x] = color
#             # cv2.circle(height_img, (x, y), radius=2, color=color, thickness=-1)
    
#     info = {
#         'left-top': left_top.tolist(),
#         'right-bot': right_bot.tolist(),
#         'ratio': ratio.item()
#     }
#     return ortho_img, height_img, info


def get_ortho_img(points: np.ndarray,
                  colors: np.ndarray,
                  width: int = 1920,
                  view: str = 'side'):
    """
    生成点云的正射投影图（支持 top/front/side）。
    - view='top'   : 投到 XY 平面，沿 +Z 方向看（俯视）
    - view='front' : 投到 XZ 平面，沿 +Y 方向看（正视）
    - view='side'  : 投到 YZ 平面，沿 +X 方向看（侧视，满足你的需求）

    返回:
      ortho_img  : 颜色正射图 (H, W, 3) uint8
      height_img : 深度灰度图 (H, W, 3) uint8（越靠近相机越暗）
      info       : 一些投影信息（角点、缩放比例、视角）
    """
    assert points.ndim == 2 and points.shape[1] == 3
    assert colors.shape[0] == points.shape[0]

    # 选择投影/观测坐标轴（u,v为图像平面轴，w为深度轴）
    # u轴对应图像x向右，v轴对应图像y向上（之后再翻转到图像y向下）
    if view == 'top':
        u_axis = np.array([1, 0, 0], dtype=float)  # X
        v_axis = np.array([0, 1, 0], dtype=float)  # Y
        w_axis = np.array([0, 0, 1], dtype=float)  # Z（朝向相机）
    elif view == 'front':
        u_axis = np.array([1, 0, 0], dtype=float)  # X
        v_axis = np.array([0, 0, 1], dtype=float)  # Z
        w_axis = np.array([0, 1, 0], dtype=float)  # Y（朝向相机）
    elif view == 'side':
        u_axis = np.array([0, 1, 0], dtype=float)  # Y
        v_axis = np.array([0, 0, 1], dtype=float)  # Z
        w_axis = np.array([1, 0, 0], dtype=float)  # X（朝向相机）
    else:
        raise ValueError("view must be one of {'top','front','side'}")

    # 投影到 (u, v, w) 坐标
    p_u = points @ u_axis
    p_v = points @ v_axis
    p_w = points @ w_axis

    u_min, u_max = float(p_u.min()), float(p_u.max())
    v_min, v_max = float(p_v.min()), float(p_v.max())
    w_min, w_max = float(p_w.min()), float(p_w.max())

    # 像素尺度（以u方向匹配输入width）
    eps = 1e-9
    ratio = width / max(u_max - u_min, eps)
    height = int(max(1.0, (v_max - v_min) * ratio))

    # 将 (u,v) 映射到像素坐标系（左上为(0,0)，x向右，y向下）
    x_img = (p_u - u_min) * ratio
    y_img = (v_max - p_v) * ratio  # 翻转v以符合图像坐标向下为正

    # 深度归一化：越小越靠近相机（便于Z-Buffer选择最近点）
    # 约定“相机”在 w = w_max 处，沿 -w 方向看过去
    z_range = max(w_max - w_min, eps)
    depth_norm = (w_max - p_w) / z_range  # 0(最近) -> 1(最远)

    # 栅格化
    ortho_img = np.zeros((height, width, 3), dtype=np.uint8)
    height_img = np.zeros((height, width, 3), dtype=np.uint8)
    depth_img = np.full((height, width), -1.0, dtype=np.float32)

    # 逐点写入，保留距离相机更近者
    xi = np.clip(x_img.astype(np.int64), 0, width - 1)
    yi = np.clip(y_img.astype(np.int64), 0, height - 1)

    for idx in tqdm(range(points.shape[0])):
        x, y = int(xi[idx]), int(yi[idx])
        pd = depth_norm[idx]
        cur = depth_img[y, x]
        if cur < 0 or pd < cur:  # 选更近
            depth_img[y, x] = pd
            c = colors[idx].astype(np.uint8)
            ortho_img[y, x] = c
            gray = (np.array([255, 255, 255]) * pd).astype(np.uint8)
            height_img[y, x] = gray

    # 计算对应的“左上/右下”世界坐标（取位于投影平面上 w=w_max / w=w_min 的角点）
    left_top_world  = u_min * u_axis + v_max * v_axis + w_max * w_axis
    right_bot_world = u_max * u_axis + v_min * v_axis + w_min * w_axis

    info = {
        'view': view,
        'left-top': left_top_world.tolist(),
        'right-bot': right_bot_world.tolist(),
        'ratio': float(ratio),
        'u_range': [u_min, u_max],
        'v_range': [v_min, v_max],
        'w_range': [w_min, w_max],
    }
    return ortho_img, height_img, info


def grey2jet(gray_img):
    jet_map = get_cmap('jet')
    gray_img = gray_img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    gray_img = np.clip(gray_img, 0, 1)  # Ensure values are within [0, 1]
    img = jet_map(gray_img)  # Apply the colormap
    img = (img[:, :, :3] * 255).astype(np.uint8)  # Convert to uint8 format

    return img


def get_extrinsic_mat(x_axis, y_axis, z_axis, position):
    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    extrinsic_mat = np.eye(4)
    extrinsic_mat[:3, :3] = R
    extrinsic_mat[:3, 3] = position
    return extrinsic_mat


def create_camera_config(position, dir_vec, mode):
    camera_config = {}
    y_axis = np.array([0, 0, -1.])

    if 'left' in mode:
        x_axis = dir_vec
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        camera_config['left'] = get_extrinsic_mat(x_axis, y_axis, z_axis, position)

    if 'front' in mode:
        z_axis = dir_vec
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        camera_config['front'] = get_extrinsic_mat(x_axis, y_axis, z_axis, position)

    if 'right' in mode:
        x_axis = -dir_vec
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        camera_config['right'] = get_extrinsic_mat(x_axis, y_axis, z_axis, position)

    if 'back' in mode:
        z_axis = -dir_vec
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        camera_config['back'] = get_extrinsic_mat(x_axis, y_axis, z_axis, position)

    return camera_config


def create_high_camera_config(camera_config, height: float, overlook_angle: float = 30):
    high_camera_config = {}
    overlook_angle = -overlook_angle * np.pi / 180
    rotation_mat = np.array([
        [1, 0, 0],
        [0, np.cos(overlook_angle), -np.sin(overlook_angle)],
        [0, np.sin(overlook_angle), np.cos(overlook_angle)],
    ])

    for cam, extrinsic_mat in camera_config.items():
        # new_extrinsic_mat = rotation_mat.T @ extrinsic_mat
        new_extrinsic_mat = extrinsic_mat.copy()
        new_extrinsic_mat[:3, :3] = extrinsic_mat[:3, :3] @ rotation_mat
        new_extrinsic_mat[2, 3] = height
        high_camera_config[cam] = new_extrinsic_mat

    return high_camera_config


CAMS = ['front', 'left', 'right', 'back', 'top', 'bot']
def create_stations_by_trajk(trajk_points, height= 3, stride= 2, mode='front left right back', is_high= False, high_height= 5, overlook_angle= 30):

    mode = mode.split(' ')
    for cam in mode:
        assert cam in CAMS, f'The mode contains the cam not in {CAMS}!'


    trajk_points = np.array(trajk_points)
    trajk_points = np.concatenate([trajk_points, np.full((trajk_points.shape[0], 1), height)], axis= -1)

    dir_vecs = trajk_points[1:] - trajk_points[:-1]
    lengths = np.sqrt(np.pow(dir_vecs[:, 0], 2) + np.pow(dir_vecs[:, 1], 2))
    dir_vecs = dir_vecs / lengths[:, np.newaxis]

    idx = 0
    dir_vec = dir_vecs[idx]
    rest_length = lengths[idx]

    last_point = trajk_points[0]

    stations = []
    camera_config = create_camera_config(last_point, dir_vec, mode)
    stations.append({cam: mat.tolist() for cam, mat in camera_config.items()})
    if is_high:
        high_stations = []
        high_camera_config = create_high_camera_config(camera_config, high_height, overlook_angle= overlook_angle)
        high_stations.append({cam: mat.tolist() for cam, mat in high_camera_config.items()})


    while True:
        rest_length -= stride
        if rest_length > 0:
            last_point += dir_vec * stride
            camera_config = create_camera_config(last_point, dir_vec, mode)
            stations.append({cam: mat.tolist() for cam, mat in camera_config.items()})
            if is_high:
                high_camera_config = create_high_camera_config(camera_config, high_height, overlook_angle= overlook_angle)
                high_stations.append({cam: mat.tolist() for cam, mat in high_camera_config.items()})
        else:
            idx += 1
            if idx == len(lengths):
                break

            dir_vec = dir_vecs[idx]
            last_point = trajk_points[idx]
            last_point += dir_vec * (-rest_length)
            camera_config = create_camera_config(last_point, dir_vec, mode)
            stations.append({cam: mat.tolist() for cam, mat in camera_config.items()})
            if is_high:
                high_camera_config = create_high_camera_config(camera_config, high_height, overlook_angle= overlook_angle)
                high_stations.append({cam: mat.tolist() for cam, mat in high_camera_config.items()})

            rest_length += lengths[idx]
        
    stations_data = {
        'trajectory_length': lengths.sum().item(),
        'stride': stride,
        'station_num': len(stations),
        'stations': stations
    }
    if is_high:
        stations_data['high_stations'] = high_stations

    return stations_data