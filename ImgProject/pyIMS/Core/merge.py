import numpy as np
import pickle
from tqdm import tqdm


def read_img_res(fp):
    with open(fp, 'rb') as f:
        img_label = pickle.load(f)

    return img_label.astype(np.uint8)


def read_mapping_info(fp):
    
    import pandas as pd
    df = pd.read_excel(fp, sheet_name='Sheet1')
    label_map = df.set_index('label_id').to_dict()
    pc_label_map = df.set_index('pc_label_id')['label_id'].dropna().to_dict()
    img_label_map = df.set_index('img_label_id')['label_id'].dropna().to_dict()

    new_img_label_map = {}
    for key in img_label_map:
        if '+' in str(key):
            new_keys = key.split('+')
            for new_key in new_keys:
                new_img_label_map[int(new_key)] = img_label_map[key]
    
    img_label_map.update(new_img_label_map)

    pc_label_map = {k: int(v) for k, v in pc_label_map.items()}
    img_label_map = {k: int(v) for k, v in img_label_map.items() if not isinstance(k, str)}
    img_label_map[0] = 0

    return label_map, pc_label_map, img_label_map


class Merger():
    def __init__(self):
        pass

    def __call__(self, pc_res, img_res):

        mask = img_res == 0
        img_res[mask] = pc_res[mask]

        # 基于点云植被无错分，将点云预测为植被但图像预测为其他的点云改为植被
        mask = pc_res == 5
        img_res[mask] = 5

        # 基于点云汽车无错分，将点云预测为汽车但图像预测为其他的点云改为汽车
        mask = pc_res == 13
        img_res[mask] = 13

        # 基于图片汽车的大量错分，将图片预测为汽车但点云预测为其他的点改为对应的点云预测值
        mask = np.logical_and(img_res == 13, pc_res != 13)
        img_res[mask] = pc_res[mask]
        
        # 基于图片道路标志无错分，将图片预测为道路标志但点云预测为其他的点云改为道路标志
        mask = img_res == 6
        pc_res[mask] = 6

        mask = img_res == 24
        pc_res[mask] = 24

        
        return pc_res, img_res


def main():
    pc_res = read_pc_res('./data/map1-0/pred-pc/pc_res.ply')
    img_res = read_img_res('./data/map1-0/pred/res.pkl')
    label_map, pc_label_map, img_label_map = read_mapping_info('./label.xls')
    integrated_res = []

    print('pc_res len:', len(pc_res))
    for idx in tqdm(range(len(pc_res))):
        pc_label_id = pc_res[idx]
        img_label_id = img_res[idx]

        pc_output = pc_label_map.get(pc_label_id, None)
        img_output = img_label_map.get(img_label_id, None)

        if pc_output is None and img_output is None:
            integrated_res.append(0)
            continue
        elif pc_output is None:  # if pc output is None, just take the img output
            integrated_res.append(img_output)
            continue
        elif img_output is None:  # if img output is None, just take the pc output
            integrated_res.append(pc_output)
            continue
        

        if pc_output == img_output:   # if pc output and img output are the same, just take the pc output
            integrated_res.append(pc_output)
        elif img_output == -1:    # if img output is -1, just take the pc output
            integrated_res.append(pc_output)
        elif pc_output == 16:   # if pc output is 46(which is 电力杆), just take the pc output
            integrated_res.append(pc_output)
        else:
            integrated_res.append(0)

    integrated_res = np.array(integrated_res)
    with open('./data/map1-0/pred/integrated_res_v2.pkl', 'wb') as f:
        pickle.dump(integrated_res, f)
        


if __name__ == "__main__":
    main()