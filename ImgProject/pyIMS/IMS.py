from typing import Dict, List
import requests
import numpy as np
from numpy.typing import NDArray, ArrayLike
from tqdm import tqdm
from PIL import Image
from scipy.spatial import cKDTree

from .Core.pc2img import pc2img
from .Core.pc2img import pc2img_soft
from .Core.pc2img import render_img_gpu
from .utils.label import vectorized_mapping_func, img_label_map, pc_label_map


def pc_inference(pc_path: str, output_path: str):
    payload = {
        "pc_path": str(pc_path),
        "output_path": str(output_path)
    }
    
    try:
        url = "http://localhost:10000/pc/inference"
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = np.load(output_path, allow_pickle=True)
            pred = output['pred']
            pred = vectorized_mapping_func(pred, pc_label_map)
            return pred

        elif response.status_code == 400:
            raise ValueError("Invalid request. Please check the input data.")
        elif response.status_code == 404:
            raise FileNotFoundError("File not found. Please check the file path.")
        else:
            raise Exception(f"Unexpected error: {response.status_code} - {response.text}")
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)
    except Exception as err:
        raise SystemExit(err)




def adjust_camera_height_by_terrain(
    stations: List[Dict[str, ArrayLike]],
    points: NDArray[np.float32],
    radius: float = 5.0,
    percentile: int = 5,
    max_jump: float = 0.5
):

    xy = points[:, :2].astype(np.float32, copy=False)
    tree = cKDTree(xy)

    adjusted_stations: List[Dict[str, ArrayLike]] = []

    for station_idx, station in enumerate(stations):
        adjusted_station: Dict[str, ArrayLike] = {}
        for cam, cam_extrinsic in station.items():
            M = np.asarray(cam_extrinsic, dtype=np.float32)
            old_pos = M[:3, 3].copy()
            xy_cam = old_pos[:2]

            neighbor_idx = tree.query_ball_point(xy_cam, r=radius)
            n_neighbors = len(neighbor_idx)

            if n_neighbors > 0:
                z_coords = points[neighbor_idx, 2]
                z_sorted = np.sort(z_coords)[::-1]
                index = int(len(z_sorted) * (100 - percentile) / 100)
                index = max(0, min(index, len(z_sorted) - 1))
                target_z = float(z_sorted[index] + 2.0)

                if station_idx > 0 and abs(target_z - np.array(adjusted_stations[station_idx - 1][cam])[2, 3]) > max_jump:
                    target_z = np.array(adjusted_stations[station_idx - 1][cam])[2, 3]
                    # if station_idx == 186:
                    #     target_z -= 0.5

                new_pos = [float(old_pos[0]), float(old_pos[1]), target_z]
                M_new = M.copy()
                M_new[2, 3] = target_z
                adjusted_station[cam] = M_new.tolist()
            else:
                M_new = M.copy()
                M_new[2, 3] = np.array(adjusted_stations[station_idx - 1][cam])[2, 3]
                adjusted_station[cam] = M_new.tolist()
                
            if n_neighbors > 0:
                print(f"[Station {station_idx} | Cam {cam}] "
                        f"old: {old_pos.tolist()} -> new: {new_pos} ")
            else:
                print(f"[Station {station_idx} | Cam {cam}] ")

        adjusted_stations.append(adjusted_station)

    return adjusted_stations


def pc_proj_preprocess( 
    points: NDArray[np.float32],
    colors: NDArray[np.float32],
    intensities: NDArray[np.float32],
    stations: List[Dict[str, ArrayLike]], 
    output_path: str = None,
    img_shape: tuple = (720, 1280), 
    buffer_size: int = .05, 
    pathch_size: int = 3,
    adjust_height: bool = True,
    terrain_radius: float = 5.0,
    terrain_percentile: int = 5,
    max_jump: float = 1.0
):
    output_path = str(output_path)
    
    if adjust_height:
        stations = adjust_camera_height_by_terrain(stations, points, terrain_radius, terrain_percentile, max_jump)
    
    for station_idx in tqdm(range(len(stations))):
        station = stations[station_idx]
        for cam, cam_extrinsic in station.items():
            cam_prefix = f'station_{station_idx}_cam_{cam}'
            rgb_img_path = f'{output_path}/{cam_prefix}.png'
            intensity_img_path = f'{output_path}/{cam_prefix}_intensity.png'
            depth_img_path = f'{output_path}/{cam_prefix}_depth.png'
            info_path = f'{output_path}/{cam_prefix}.npz'

            cam_extrinsic = np.array(cam_extrinsic)
            # rgb_img, intensity_img, (dist_img, depth_img), (pts_img_indices, pts_indices) = pc2img(
            #     points, colors, intensities, cam_extrinsic,
            #     img_shape=img_shape,
            #     buffer_size=buffer_size,
            # )
            rgb_img, intensity_img, (dist_img, depth_img), (pts_img_indices, pts_indices) = pc2img_soft(
                points,
                colors,
                intensities,
                cam_extrinsic,
                img_shape=img_shape,
                buffer_size=buffer_size,
            )

            # Render RGB image
            # rendered_rgb_img, _ = render_img_gpu([rgb_img, depth_img], dist_img, pathch_size)
            rendered_rgb_img = rgb_img
            
            # Render intensity image
            # if intensity_img is not None:
            #     rendered_intensity_img, _ = render_img_gpu([intensity_img, depth_img], dist_img, pathch_size)

            info = {'dist_img': dist_img, 'pts_img_indices': pts_img_indices, 'pts_indices': pts_indices}

            # Save RGB image
            img = Image.fromarray(rendered_rgb_img, mode= 'RGB')
            img.save(rgb_img_path)
            
            # Save intensity image
            # if intensity_img is not None:
            #     img_intensity = Image.fromarray(rendered_intensity_img, mode= 'RGB')
            #     img_intensity.save(intensity_img_path)

            np.savez_compressed(info_path, **info)
            

def img_inference(imgs_path: str, num_points: int, output_path: str, is_save_img: bool = False):
    payload = {
        'imgs_path': str(imgs_path),
        'num_points': num_points,
        'output_path': str(output_path),
        'is_save_img': is_save_img
    }
    try:
        url = "http://localhost:10000/inference"
        # url = "http://localhost:10002/inference"
        response = requests.post(url, json=payload, timeout= 3600)
        if response.status_code == 200:
            output = np.load(output_path, allow_pickle=True)
            pred = output['pred']
            confidence = output['confidence']
            pred = vectorized_mapping_func(pred, img_label_map)
            return pred, confidence

        elif response.status_code == 400:
            raise ValueError("Invalid request. Please check the input data.")
        elif response.status_code == 404:
            raise FileNotFoundError("File not found. Please check the file path.")
        else:
            raise Exception(f"Unexpected error: {response.status_code} - {response.text}")
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)
    except Exception as err:
        raise SystemExit(err)
