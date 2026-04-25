from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from pathlib import Path
from typing import Any

import laspy
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))
IMGPROJECT_DIR = PIPELINE_DIR.parent
if str(IMGPROJECT_DIR) not in sys.path:
    sys.path.append(str(IMGPROJECT_DIR))

import utils as task5_utils
from pyIMS.Core.pc2img import pc2img_soft
from task6_geometry_utils import (
    compute_manhole_geometry_from_pixels,
    geometry_for_pole_group,
    geometry_for_scene_instance,
)
from task6_glm_utils import (
    DEFAULT_MODEL_PATH,
    MANHOLE_PROMPT,
    TREE_PROMPT,
    build_pole_group_prompt,
    load_glm_model_and_processor,
    run_dual_image_glm,
)


SUPPORTED_SCENE_INSTANCE_CLASSES = frozenset({7, 8, 9, 10, 12, 13, 14, 15})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task6 attribute extraction (BEV global + Front by scene).")
    parser.add_argument(
        "--run-branch",
        type=lambda s: str(s).strip().lower(),
        choices=("both", "a", "b", "bev", "front"),
        default="both",
        help="Run branch selector: A/BEV, B/Front, or both.",
    )
    parser.add_argument(
        "--run-stage",
        type=lambda s: str(s).strip().lower(),
        choices=("both", "geometry", "vlm"),
        default="both",
        help="Run stage selector inside each branch: geometry, vlm, or both.",
    )
    parser.add_argument("--data-root", required=True, help="Benchmark root.")
    parser.add_argument("--task5-output-dir", type=str, default="fusion", help="Task5 output directory under each scene.")
    parser.add_argument("--bev-dir", type=str, default="benchmark/bev", help="BEV global directory path.")
    parser.add_argument("--target-fill-ratio", type=float, default=0.8, help="Adaptive crop target fill ratio.")
    parser.add_argument("--bev-global-size", type=int, default=500, help="Global BEV context crop size in pixels.")
    parser.add_argument("--bev-resolution", type=float, default=0.02, help="BEV resolution (meters per pixel).")
    parser.add_argument("--front-render-height", type=int, default=720, help="Front-view re-projection render height.")
    parser.add_argument("--front-render-width", type=int, default=1280, help="Front-view re-projection render width.")
    parser.add_argument("--front-fov-deg", type=float, default=90.0, help="Fixed front-view render FOV in degrees.")
    parser.add_argument("--front-buffer-size", type=float, default=0.05, help="Buffer size passed to soft-splat renderer.")
    parser.add_argument("--front-distance-min", type=float, default=1.0, help="Minimum camera distance for front-view rendering.")
    parser.add_argument("--front-distance-max", type=float, default=120.0, help="Maximum camera distance for front-view rendering.")
    parser.add_argument("--front-distance-iters", type=int, default=6, help="Iterations used to solve camera distance for target fill ratio.")
    parser.add_argument("--max-image-side", type=int, default=1024, help="If output crop max side exceeds this value, downsample proportionally.")
    parser.add_argument("--mid-centroid-low-ratio", type=float, default=0.30, help="Lower ratio (within q05-q95 span) used to compute mid-band centroid for center_xy classes.")
    parser.add_argument("--mid-centroid-high-ratio", type=float, default=0.70, help="Upper ratio (within q05-q95 span) used to compute mid-band centroid for center_xy classes.")
    parser.add_argument("--fence-mid-band-low-ratio", type=float, default=0.30, help="Lower ratio for fence mid-band centerline extraction.")
    parser.add_argument("--fence-mid-band-high-ratio", type=float, default=0.70, help="Upper ratio for fence mid-band centerline extraction.")
    parser.add_argument("--fence-control-point-count", type=int, default=5, help="Number of fence control points sampled along centerline (includes endpoints).")
    parser.add_argument("--fence-centerline-grid-size", type=float, default=0.10, help="Grid size (meters) used before fence centerline graph extraction.")
    parser.add_argument("--fence-centerline-knn", type=int, default=8, help="KNN used for fence centerline graph construction.")
    parser.add_argument("--ground-neighborhood-radius", type=float, default=2.0, help="XY neighborhood radius (meters) used for local ground-z estimation.")
    parser.add_argument("--ground-neighborhood-quantile", type=float, default=0.10, help="Quantile of neighborhood ground z used as ground_z (default q10).")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="GLM model path.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument(
        "--disable-glm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable GLM semantic inference and keep semantic attributes empty/default.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Task6 outputs.",
    )
    return parser.parse_args()


def _normalize_run_branch(run_branch: str) -> str:
    value = str(run_branch).strip().lower()
    if value in {"a", "bev"}:
        return "bev"
    if value in {"b", "front"}:
        return "front"
    return "both"


def _json_dumps(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _load_npz_object_array(npz: Any, key: str) -> list[np.ndarray]:
    if key not in npz.files:
        return []
    arr = np.asarray(npz[key], dtype=object).reshape(-1)
    out: list[np.ndarray] = []
    for item in arr:
        out.append(np.asarray(item, dtype=np.int64).reshape(-1))
    return out


def _load_scene_front_objects(npz_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = np.load(npz_path, allow_pickle=True)

    scene_instances: list[dict[str, Any]] = []
    pole_groups: list[dict[str, Any]] = []

    has_compact_scene_instance = ("scene_instance_id" in data.files) and ("scene_instance_point_indices" in data.files)
    if has_compact_scene_instance:
        scene_ids = np.asarray(data["scene_instance_id"], dtype=np.int32).reshape(-1)
        class_ids = np.asarray(data["scene_instance_class_id"], dtype=np.int32).reshape(-1) if "scene_instance_class_id" in data.files else np.zeros_like(scene_ids)
        class_names = np.asarray(data["scene_instance_class_name"], dtype=object).reshape(-1) if "scene_instance_class_name" in data.files else np.asarray([""] * scene_ids.shape[0], dtype=object)
        point_arrays = _load_npz_object_array(data, "scene_instance_point_indices")
        for idx, scene_id in enumerate(scene_ids):
            points = point_arrays[idx] if idx < len(point_arrays) else np.zeros((0,), dtype=np.int64)
            scene_instances.append(
                {
                    "scene_instance_id": int(scene_id),
                    "class_id": int(class_ids[idx]) if idx < class_ids.shape[0] else 0,
                    "class_name": str(class_names[idx]) if idx < class_names.shape[0] else "",
                    "point_indices": points,
                }
            )
    else:
        # Backward compatibility for old final schema.
        scene_ids = np.asarray(data["scene_instance_id"], dtype=np.int32).reshape(-1) if "scene_instance_id" in data.files else np.arange(np.asarray(data["class_id"]).shape[0], dtype=np.int32)
        class_ids = np.asarray(data["class_id"], dtype=np.int32).reshape(-1) if "class_id" in data.files else np.zeros((scene_ids.shape[0],), dtype=np.int32)
        class_names = np.asarray(data["class_name"], dtype=object).reshape(-1) if "class_name" in data.files else np.asarray([""] * scene_ids.shape[0], dtype=object)
        point_arrays = _load_npz_object_array(data, "point_indices")
        for idx, scene_id in enumerate(scene_ids):
            points = point_arrays[idx] if idx < len(point_arrays) else np.zeros((0,), dtype=np.int64)
            scene_instances.append(
                {
                    "scene_instance_id": int(scene_id),
                    "class_id": int(class_ids[idx]) if idx < class_ids.shape[0] else 0,
                    "class_name": str(class_names[idx]) if idx < class_names.shape[0] else "",
                    "point_indices": points,
                }
            )

    if "pole_group_id" in data.files:
        pole_ids = np.asarray(data["pole_group_id"], dtype=np.int32).reshape(-1)
        candidate_ids_arr = _load_npz_object_array(data, "pole_group_candidate_class_ids")
        candidate_names_arr = np.asarray(data["pole_group_candidate_class_names"], dtype=object).reshape(-1) if "pole_group_candidate_class_names" in data.files else np.asarray([], dtype=object)
        point_arrays = _load_npz_object_array(data, "pole_group_point_indices")
        pole_diameter_arr = np.asarray(data["pole_group_diameter_m"], dtype=np.float32).reshape(-1) if "pole_group_diameter_m" in data.files else np.asarray([], dtype=np.float32)
        pole_center_x_arr = np.asarray(data["pole_group_center_x"], dtype=np.float32).reshape(-1) if "pole_group_center_x" in data.files else np.asarray([], dtype=np.float32)
        pole_center_y_arr = np.asarray(data["pole_group_center_y"], dtype=np.float32).reshape(-1) if "pole_group_center_y" in data.files else np.asarray([], dtype=np.float32)
        pole_metric_source_arr = np.asarray(data["pole_group_metric_source"], dtype=object).reshape(-1) if "pole_group_metric_source" in data.files else np.asarray([], dtype=object)
        for idx, pole_id in enumerate(pole_ids):
            candidate_ids = candidate_ids_arr[idx] if idx < len(candidate_ids_arr) else np.zeros((0,), dtype=np.int64)
            if idx < candidate_names_arr.shape[0]:
                raw_names = np.asarray(candidate_names_arr[idx], dtype=object).reshape(-1)
                candidate_names = [str(x) for x in raw_names if str(x)]
            else:
                candidate_names = [task5_utils.CLASS_ID_TO_NAME.get(int(x), "") for x in candidate_ids]
            points = point_arrays[idx] if idx < len(point_arrays) else np.zeros((0,), dtype=np.int64)
            diameter_m: float | None = None
            if idx < pole_diameter_arr.shape[0]:
                d = float(pole_diameter_arr[idx])
                if np.isfinite(d):
                    diameter_m = d
            center_xy: list[float] | None = None
            if idx < pole_center_x_arr.shape[0] and idx < pole_center_y_arr.shape[0]:
                cx = float(pole_center_x_arr[idx])
                cy = float(pole_center_y_arr[idx])
                if np.isfinite(cx) and np.isfinite(cy):
                    center_xy = [cx, cy]
            metric_source = str(pole_metric_source_arr[idx]) if idx < pole_metric_source_arr.shape[0] else ""
            pole_groups.append(
                {
                    "pole_group_id": int(pole_id),
                    "candidate_class_ids": [int(x) for x in np.asarray(candidate_ids, dtype=np.int32).reshape(-1)],
                    "candidate_class_names": candidate_names,
                    "point_indices": points,
                    "diameter_m": diameter_m,
                    "center_xy": center_xy,
                    "metric_source": metric_source,
                }
            )

    return scene_instances, pole_groups


def _load_tree_metrics(tree_metrics_path: Path) -> dict[int, dict[str, Any]]:
    if not tree_metrics_path.exists():
        return {}
    data = np.load(tree_metrics_path, allow_pickle=True)
    scene_ids = np.asarray(data["scene_instance_id"], dtype=np.int32).reshape(-1) if "scene_instance_id" in data.files else np.zeros((0,), dtype=np.int32)
    dbh_m = np.asarray(data["dbh_m"], dtype=np.float32).reshape(-1) if "dbh_m" in data.files else np.zeros_like(scene_ids, dtype=np.float32)
    center_x = np.asarray(data["trunk_center_x"], dtype=np.float32).reshape(-1) if "trunk_center_x" in data.files else np.zeros_like(scene_ids, dtype=np.float32)
    center_y = np.asarray(data["trunk_center_y"], dtype=np.float32).reshape(-1) if "trunk_center_y" in data.files else np.zeros_like(scene_ids, dtype=np.float32)
    out: dict[int, dict[str, Any]] = {}
    for idx, scene_id in enumerate(scene_ids):
        out[int(scene_id)] = {
            "dbh_m": float(dbh_m[idx]) if idx < dbh_m.shape[0] else None,
            "trunk_center_xy": [
                float(center_x[idx]) if idx < center_x.shape[0] else None,
                float(center_y[idx]) if idx < center_y.shape[0] else None,
            ],
        }
    return out


def _load_task5_csf_ground_points(csf_ground_las_path: Path) -> np.ndarray | None:
    if not csf_ground_las_path.exists():
        return None
    try:
        las_data = laspy.read(csf_ground_las_path)
    except Exception:
        return None
    try:
        points_xyz = np.vstack([las_data.x, las_data.y, las_data.z]).T.astype(np.float32, copy=False)
    except Exception:
        return None
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3 or points_xyz.shape[0] == 0:
        return None
    return points_xyz


def _load_scene_points(scene_dir: Path) -> tuple[np.ndarray, np.ndarray, Path]:
    las_path = task5_utils.find_las_path(scene_dir)
    if las_path is None:
        raise FileNotFoundError(f"Missing scene LAS under: {scene_dir}")
    las_data = laspy.read(las_path)
    points_xyz = np.vstack([las_data.x, las_data.y, las_data.z]).T.astype(np.float32, copy=False)
    num_points = int(points_xyz.shape[0])

    if all(hasattr(las_data, name) for name in ("red", "green", "blue")):
        red = np.asarray(las_data.red, dtype=np.float32).reshape(-1)
        green = np.asarray(las_data.green, dtype=np.float32).reshape(-1)
        blue = np.asarray(las_data.blue, dtype=np.float32).reshape(-1)
        if red.shape[0] == num_points and green.shape[0] == num_points and blue.shape[0] == num_points:
            points_rgb = np.stack([red, green, blue], axis=1)
            max_rgb = float(np.nanmax(points_rgb)) if points_rgb.size > 0 else 0.0
            if max_rgb > 255.0:
                points_rgb = points_rgb / 65535.0
            else:
                points_rgb = points_rgb / 255.0
            points_rgb = np.clip(points_rgb, 0.0, 1.0).astype(np.float32, copy=False)
            return points_xyz, points_rgb, las_path

    points_rgb = np.full((num_points, 3), 0.5, dtype=np.float32)
    return points_xyz, points_rgb, las_path


def _load_stations(projected_dir: Path) -> list[dict[str, Any]]:
    stations_path = projected_dir / "effective_stations.json"
    if not stations_path.exists():
        return []
    try:
        payload = json.loads(stations_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    raw_stations = payload.get("stations")
    if not isinstance(raw_stations, list):
        return []

    parsed: list[dict[str, Any]] = []
    for station_idx, station in enumerate(raw_stations):
        if not isinstance(station, dict):
            continue
        cam_items: list[tuple[str, np.ndarray, np.ndarray]] = []
        translations_xy: list[np.ndarray] = []
        translations_xyz: list[np.ndarray] = []
        for cam_name, mat_raw in station.items():
            try:
                mat = np.asarray(mat_raw, dtype=np.float32)
            except Exception:
                continue
            if mat.shape != (4, 4):
                continue
            t_xy = mat[:2, 3].astype(np.float32, copy=False)
            t_xyz = mat[:3, 3].astype(np.float32, copy=False)
            forward = mat[:2, 2].astype(np.float32, copy=False)
            norm = float(np.linalg.norm(forward))
            if norm <= 1e-6:
                continue
            forward = (forward / norm).astype(np.float32, copy=False)
            cam_items.append((str(cam_name), mat, forward))
            translations_xy.append(t_xy)
            translations_xyz.append(t_xyz)
        if not cam_items:
            continue
        station_xy = np.median(np.stack(translations_xy, axis=0), axis=0).astype(np.float32, copy=False)
        station_xyz = np.median(np.stack(translations_xyz, axis=0), axis=0).astype(np.float32, copy=False)
        parsed.append(
            {
                "station_idx": int(station_idx),
                "station_xy": station_xy,
                "station_xyz": station_xyz,
                "cameras": [
                    {
                        "name": cam_name,
                        "forward_xy": forward,
                    }
                    for cam_name, _, forward in cam_items
                ],
            }
        )
    return parsed


def _normalize_dir_xy(vec_xy: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec_xy, dtype=np.float32).reshape(2)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-6:
        return np.asarray([1.0, 0.0], dtype=np.float32)
    return (vec / norm).astype(np.float32, copy=False)


def _downsample_if_needed(image: Image.Image, *, max_side: int) -> Image.Image:
    img = image.convert("RGB")
    w, h = img.size
    long_side = max(int(w), int(h))
    limit = max(1, int(max_side))
    if long_side <= limit:
        return img
    scale = float(limit) / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.BILINEAR)


def _build_view_extrinsic(camera_center_xyz: np.ndarray, forward_xy: np.ndarray) -> np.ndarray:
    center = np.asarray(camera_center_xyz, dtype=np.float32).reshape(3)
    fxy = _normalize_dir_xy(np.asarray(forward_xy, dtype=np.float32).reshape(2))
    z_axis = np.asarray([fxy[0], fxy[1], 0.0], dtype=np.float32)
    z_norm = float(np.linalg.norm(z_axis))
    if z_norm <= 1e-6:
        z_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        z_axis = (z_axis / z_norm).astype(np.float32, copy=False)

    y_axis = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    x_axis = np.cross(y_axis, z_axis).astype(np.float32, copy=False)
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm <= 1e-6:
        x_axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        x_axis = (x_axis / x_norm).astype(np.float32, copy=False)
    z_axis = np.cross(x_axis, y_axis).astype(np.float32, copy=False)
    z_axis = (z_axis / max(float(np.linalg.norm(z_axis)), 1e-6)).astype(np.float32, copy=False)

    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float32, copy=False)
    extrinsic[:3, 3] = center
    return extrinsic


def _project_points_stats(
    points_xyz: np.ndarray,
    *,
    extrinsic: np.ndarray,
    img_shape: tuple[int, int],
    fov_deg: float,
) -> dict[str, Any]:
    h, w = int(img_shape[0]), int(img_shape[1])
    if points_xyz.size == 0:
        return {
            "pixels_xy": np.zeros((0, 2), dtype=np.int32),
            "front_count": 0,
            "valid_count": 0,
            "fill_ratio": 0.0,
            "visible_ratio": 0.0,
        }

    f = float(w) / (2.0 * math.tan(math.radians(float(fov_deg) * 0.5)))
    cx = float(w) * 0.5
    cy = float(h) * 0.5

    pts = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])
    try:
        trans_mat = np.linalg.inv(np.asarray(extrinsic, dtype=np.float32))[:3, :]
    except np.linalg.LinAlgError:
        return {
            "pixels_xy": np.zeros((0, 2), dtype=np.int32),
            "front_count": 0,
            "valid_count": 0,
            "fill_ratio": 0.0,
            "visible_ratio": 0.0,
        }

    pts_cam = (trans_mat @ pts_h.T).T
    front_mask = pts_cam[:, 2] > 1e-6
    front_pts = pts_cam[front_mask]
    front_count = int(front_pts.shape[0])
    if front_count == 0:
        return {
            "pixels_xy": np.zeros((0, 2), dtype=np.int32),
            "front_count": 0,
            "valid_count": 0,
            "fill_ratio": 0.0,
            "visible_ratio": 0.0,
        }

    x = f * (front_pts[:, 0] / front_pts[:, 2]) + cx
    y = f * (front_pts[:, 1] / front_pts[:, 2]) + cy
    valid = (x >= 0.0) & (x < float(w)) & (y >= 0.0) & (y < float(h))
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        return {
            "pixels_xy": np.zeros((0, 2), dtype=np.int32),
            "front_count": front_count,
            "valid_count": 0,
            "fill_ratio": 0.0,
            "visible_ratio": 0.0,
        }

    xv = x[valid]
    yv = y[valid]
    x_min = float(np.min(xv))
    x_max = float(np.max(xv))
    y_min = float(np.min(yv))
    y_max = float(np.max(yv))
    bbox_w = max(1.0, x_max - x_min + 1.0)
    bbox_h = max(1.0, y_max - y_min + 1.0)
    fill_ratio = float(max(bbox_w / float(w), bbox_h / float(h)))

    pixels_xy = np.stack(
        [
            np.clip(np.round(xv), 0, w - 1).astype(np.int32, copy=False),
            np.clip(np.round(yv), 0, h - 1).astype(np.int32, copy=False),
        ],
        axis=1,
    )
    pixels_xy = np.unique(pixels_xy, axis=0).astype(np.int32, copy=False)

    visible_ratio = float(valid_count / max(1, front_count))
    return {
        "pixels_xy": pixels_xy,
        "front_count": front_count,
        "valid_count": valid_count,
        "fill_ratio": fill_ratio,
        "visible_ratio": visible_ratio,
    }


def _estimate_view_distance(
    object_points_xyz: np.ndarray,
    *,
    object_center_xyz: np.ndarray,
    station_z: float,
    forward_xy: np.ndarray,
    img_shape: tuple[int, int],
    fov_deg: float,
    target_fill_ratio: float,
    distance_min: float,
    distance_max: float,
    max_iters: int,
) -> tuple[float, dict[str, Any]]:
    obj_center = np.asarray(object_center_xyz, dtype=np.float32).reshape(3)
    view_dir = _normalize_dir_xy(np.asarray(forward_xy, dtype=np.float32).reshape(2))
    target = min(0.98, max(0.2, float(target_fill_ratio)))
    d = max(float(distance_min), min(float(distance_max), 8.0))

    final_stats: dict[str, Any] = {
        "fill_ratio": 0.0,
        "visible_ratio": 0.0,
        "front_count": 0,
        "valid_count": 0,
    }
    for _ in range(max(1, int(max_iters))):
        cam_center = np.asarray(
            [
                obj_center[0] - view_dir[0] * d,
                obj_center[1] - view_dir[1] * d,
                float(station_z),
            ],
            dtype=np.float32,
        )
        extrinsic = _build_view_extrinsic(cam_center, view_dir)
        stats = _project_points_stats(
            object_points_xyz,
            extrinsic=extrinsic,
            img_shape=img_shape,
            fov_deg=fov_deg,
        )
        final_stats = {
            "fill_ratio": float(stats["fill_ratio"]),
            "visible_ratio": float(stats["visible_ratio"]),
            "front_count": int(stats["front_count"]),
            "valid_count": int(stats["valid_count"]),
        }

        fill = float(stats["fill_ratio"])
        visible_ratio = float(stats["visible_ratio"])
        if fill <= 0.0 or int(stats["valid_count"]) <= 8:
            d = min(float(distance_max), d * 1.5)
            continue
        if visible_ratio < 0.90:
            d = min(float(distance_max), d * 1.2)
            continue
        if abs(fill - target) <= 0.03:
            break

        scale = fill / max(target, 1e-6)
        d = d * scale
        d = max(float(distance_min), min(float(distance_max), d))

    return float(d), final_stats


def _flat_indices_to_xy(flat_indices: np.ndarray, *, img_width: int, img_height: int) -> np.ndarray:
    arr = np.asarray(flat_indices, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    valid = (arr >= 0) & (arr < int(img_width * img_height))
    arr = arr[valid]
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    y = (arr // int(img_width)).astype(np.int32, copy=False)
    x = (arr % int(img_width)).astype(np.int32, copy=False)
    xy = np.stack([x, y], axis=1)
    return np.unique(xy, axis=0).astype(np.int32, copy=False)


def _select_nearest_station(stations: list[dict[str, Any]], object_xy_center: np.ndarray) -> dict[str, Any] | None:
    if not stations:
        return None
    center = np.asarray(object_xy_center, dtype=np.float32).reshape(2)
    return min(
        stations,
        key=lambda item: float(np.linalg.norm(center - np.asarray(item["station_xy"], dtype=np.float32).reshape(2))),
    )


def _render_instance_view(
    *,
    object_points_xyz: np.ndarray,
    object_points_rgb: np.ndarray,
    object_center_xyz: np.ndarray,
    station_center_xyz: np.ndarray,
    view_dir_xy: np.ndarray,
    fixed_distance_m: float | None = None,
    args: argparse.Namespace,
) -> tuple[Image.Image, dict[str, Any]] | None:
    img_shape = (int(args.front_render_height), int(args.front_render_width))
    station_xyz = np.asarray(station_center_xyz, dtype=np.float32).reshape(3)
    obj_center = np.asarray(object_center_xyz, dtype=np.float32).reshape(3)
    view_dir = _normalize_dir_xy(np.asarray(view_dir_xy, dtype=np.float32).reshape(2))

    if fixed_distance_m is not None and np.isfinite(float(fixed_distance_m)) and float(fixed_distance_m) > 0:
        distance = max(
            float(args.front_distance_min),
            min(float(args.front_distance_max), float(fixed_distance_m)),
        )
        distance_stats = {
            "mode": "fixed_from_front",
            "distance_m": float(distance),
        }
    else:
        distance, distance_stats = _estimate_view_distance(
            object_points_xyz,
            object_center_xyz=obj_center,
            station_z=float(station_xyz[2]),
            forward_xy=view_dir,
            img_shape=img_shape,
            fov_deg=float(args.front_fov_deg),
            target_fill_ratio=float(args.target_fill_ratio),
            distance_min=float(args.front_distance_min),
            distance_max=float(args.front_distance_max),
            max_iters=int(args.front_distance_iters),
        )

    camera_center = np.asarray(
        [
            obj_center[0] - view_dir[0] * float(distance),
            obj_center[1] - view_dir[1] * float(distance),
            float(station_xyz[2]),
        ],
        dtype=np.float32,
    )
    extrinsic = _build_view_extrinsic(camera_center, view_dir)

    rgb_img, _, _, render_info = pc2img_soft(
        object_points_xyz,
        object_points_rgb,
        None,
        extrinsic,
        img_shape=img_shape,
        fov=float(args.front_fov_deg),
        buffer_size=float(args.front_buffer_size),
    )
    hit_img_indices = np.asarray(render_info[0], dtype=np.int64).reshape(-1) if isinstance(render_info, tuple) and len(render_info) >= 1 else np.zeros((0,), dtype=np.int64)
    pixels_xy = _flat_indices_to_xy(
        hit_img_indices,
        img_width=int(img_shape[1]),
        img_height=int(img_shape[0]),
    )
    if pixels_xy.shape[0] == 0:
        proj_stats = _project_points_stats(
            object_points_xyz,
            extrinsic=extrinsic,
            img_shape=img_shape,
            fov_deg=float(args.front_fov_deg),
        )
        pixels_xy = np.asarray(proj_stats["pixels_xy"], dtype=np.int32).reshape(-1, 2)
        if pixels_xy.shape[0] == 0:
            return None

    crop_img, crop_meta = _adaptive_crop_from_pixels(
        Image.fromarray(np.asarray(rgb_img, dtype=np.uint8), mode="RGB"),
        pixels_xy,
        fill_ratio=float(args.target_fill_ratio),
    )
    crop_img = _downsample_if_needed(crop_img, max_side=int(args.max_image_side))

    meta = {
        "camera_center_xyz": [float(camera_center[0]), float(camera_center[1]), float(camera_center[2])],
        "view_dir_xy": [float(view_dir[0]), float(view_dir[1])],
        "distance_m": float(distance),
        "distance_solver": distance_stats,
        "render_shape_hw": [int(img_shape[0]), int(img_shape[1])],
        "crop_meta": crop_meta,
        "crop_size_wh": [int(crop_img.size[0]), int(crop_img.size[1])],
        "hit_pixels": int(pixels_xy.shape[0]),
    }
    return crop_img, meta


def _render_instance_two_views(
    *,
    object_points_xyz: np.ndarray,
    object_points_rgb: np.ndarray,
    station_center_xyz: np.ndarray,
    args: argparse.Namespace,
) -> tuple[tuple[Image.Image, dict[str, Any]], tuple[Image.Image, dict[str, Any]] | None] | None:
    if object_points_xyz.shape[0] == 0:
        return None
    obj_center = np.asarray(object_points_xyz, dtype=np.float32).mean(axis=0)
    station_xyz = np.asarray(station_center_xyz, dtype=np.float32).reshape(3)
    dir0 = _normalize_dir_xy(obj_center[:2] - station_xyz[:2])
    dir1 = np.asarray([-dir0[1], dir0[0]], dtype=np.float32)

    front_view = _render_instance_view(
        object_points_xyz=object_points_xyz,
        object_points_rgb=object_points_rgb,
        object_center_xyz=obj_center,
        station_center_xyz=station_xyz,
        view_dir_xy=dir0,
        args=args,
    )
    if front_view is None:
        return None

    front_distance = float(front_view[1].get("distance_m", np.nan))
    if (not np.isfinite(front_distance)) or front_distance <= 0:
        front_distance = None  # type: ignore[assignment]

    side_view = _render_instance_view(
        object_points_xyz=object_points_xyz,
        object_points_rgb=object_points_rgb,
        object_center_xyz=obj_center,
        station_center_xyz=station_xyz,
        view_dir_xy=dir1,
        fixed_distance_m=front_distance,
        args=args,
    )
    if side_view is not None:
        side_view[1]["side_strategy"] = "plus_90deg_fixed_distance"
        return front_view, side_view

    # fallback: rotate to the opposite side and keep the same distance
    dir1_opposite = np.asarray([dir0[1], -dir0[0]], dtype=np.float32)
    side_view = _render_instance_view(
        object_points_xyz=object_points_xyz,
        object_points_rgb=object_points_rgb,
        object_center_xyz=obj_center,
        station_center_xyz=station_xyz,
        view_dir_xy=dir1_opposite,
        fixed_distance_m=front_distance,
        args=args,
    )
    if side_view is not None:
        side_view[1]["side_strategy"] = "minus_90deg_fixed_distance_fallback"
    return front_view, side_view


def _select_station_front_side_stems(
    stations: list[dict[str, Any]],
    object_xy_center: np.ndarray,
) -> tuple[str | None, str | None]:
    if not stations:
        return None, None

    center = np.asarray(object_xy_center, dtype=np.float32).reshape(2)
    nearest = min(
        stations,
        key=lambda item: float(np.linalg.norm(center - np.asarray(item["station_xy"], dtype=np.float32).reshape(2))),
    )
    station_idx = int(nearest["station_idx"])
    station_xy = np.asarray(nearest["station_xy"], dtype=np.float32).reshape(2)
    target_dir = _normalize_dir_xy(center - station_xy)
    side_dir = np.asarray([-target_dir[1], target_dir[0]], dtype=np.float32)

    cameras = list(nearest.get("cameras", []))
    if not cameras:
        return None, None

    front_cam = max(cameras, key=lambda cam: float(np.dot(target_dir, np.asarray(cam["forward_xy"], dtype=np.float32))))
    side_candidates = [cam for cam in cameras if str(cam["name"]) != str(front_cam["name"])]
    if side_candidates:
        side_cam = max(side_candidates, key=lambda cam: float(np.dot(side_dir, np.asarray(cam["forward_xy"], dtype=np.float32))))
    else:
        side_cam = front_cam

    front_stem = f"station_{station_idx}_cam_{front_cam['name']}"
    side_stem = f"station_{station_idx}_cam_{side_cam['name']}"
    return front_stem, side_stem


def _project_object_pixels(
    projected_dir: Path,
    image_stem: str,
    point_indices: np.ndarray,
) -> tuple[np.ndarray, Path | None]:
    mapping_path = projected_dir / f"{image_stem}.npz"
    image_path = projected_dir / f"{image_stem}.png"
    if (not mapping_path.exists()) or (not image_path.exists()):
        return np.zeros((0, 2), dtype=np.int32), None

    data = np.load(mapping_path)
    if "pts_img_indices" not in data.files or "pts_indices" not in data.files or "dist_img" not in data.files:
        return np.zeros((0, 2), dtype=np.int32), image_path
    pts_img_indices = np.asarray(data["pts_img_indices"], dtype=np.int64).reshape(-1)
    pts_indices = np.asarray(data["pts_indices"], dtype=np.int64).reshape(-1)
    h, w = np.asarray(data["dist_img"]).shape[:2]
    n = min(pts_img_indices.shape[0], pts_indices.shape[0])
    pts_img_indices = pts_img_indices[:n]
    pts_indices = pts_indices[:n]

    object_points = np.asarray(point_indices, dtype=np.int64).reshape(-1)
    if object_points.size == 0:
        return np.zeros((0, 2), dtype=np.int32), image_path
    mask = np.isin(pts_indices, object_points, assume_unique=False)
    if not np.any(mask):
        return np.zeros((0, 2), dtype=np.int32), image_path
    img_idx = pts_img_indices[mask]
    valid = (img_idx >= 0) & (img_idx < int(h * w))
    img_idx = img_idx[valid]
    if img_idx.size == 0:
        return np.zeros((0, 2), dtype=np.int32), image_path
    y = (img_idx // int(w)).astype(np.int32, copy=False)
    x = (img_idx % int(w)).astype(np.int32, copy=False)
    return np.stack([x, y], axis=1), image_path


def _adaptive_crop_from_pixels(image: Image.Image, pixel_xy: np.ndarray, *, fill_ratio: float) -> tuple[Image.Image, dict[str, Any]]:
    img = image.convert("RGB")
    w, h = img.size
    coords = np.asarray(pixel_xy, dtype=np.int32).reshape(-1, 2)
    ratio = min(1.0, max(0.1, float(fill_ratio)))

    if coords.shape[0] == 0:
        crop_w = min(512, w)
        crop_h = min(512, h)
        x0 = (w - crop_w) // 2
        y0 = (h - crop_h) // 2
    else:
        x_min = int(coords[:, 0].min())
        x_max = int(coords[:, 0].max())
        y_min = int(coords[:, 1].min())
        y_max = int(coords[:, 1].max())
        bbox_w = max(1, x_max - x_min + 1)
        bbox_h = max(1, y_max - y_min + 1)
        crop_w = max(32, int(math.ceil(bbox_w / ratio)))
        crop_h = max(32, int(math.ceil(bbox_h / ratio)))
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        x0 = int(math.floor(cx - crop_w / 2.0))
        y0 = int(math.floor(cy - crop_h / 2.0))

    canvas = Image.new("RGB", (crop_w, crop_h), (0, 0, 0))
    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(w, x0 + crop_w)
    src_y1 = min(h, y0 + crop_h)
    if src_x1 > src_x0 and src_y1 > src_y0:
        patch = img.crop((src_x0, src_y0, src_x1, src_y1))
        dst_x0 = src_x0 - x0
        dst_y0 = src_y0 - y0
        canvas.paste(patch, (dst_x0, dst_y0))

    meta = {
        "crop_x0": int(x0),
        "crop_y0": int(y0),
        "crop_w": int(crop_w),
        "crop_h": int(crop_h),
    }
    return canvas, meta


def _fixed_context_crop(image: Image.Image, center_xy: tuple[int, int], size: int) -> Image.Image:
    img = image.convert("RGB")
    cx, cy = int(center_xy[0]), int(center_xy[1])
    crop_w = int(size)
    crop_h = int(size)
    x0 = cx - crop_w // 2
    y0 = cy - crop_h // 2

    w, h = img.size
    canvas = Image.new("RGB", (crop_w, crop_h), (0, 0, 0))
    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(w, x0 + crop_w)
    src_y1 = min(h, y0 + crop_h)
    if src_x1 > src_x0 and src_y1 > src_y0:
        patch = img.crop((src_x0, src_y0, src_x1, src_y1))
        dst_x0 = src_x0 - x0
        dst_y0 = src_y0 - y0
        canvas.paste(patch, (dst_x0, dst_y0))
    return canvas


def _annotate_bev_global_image(
    *,
    global_image: Image.Image,
    semantic: dict[str, Any],
    geometry: dict[str, Any],
    bev_resolution: float,
    output_path: Path,
) -> Path:
    img = global_image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)

    functional_type = semantic.get("functional_type")
    shape = semantic.get("shape")
    circle_center_xy = geometry.get("circle_center_xy")
    circle_radius_m = geometry.get("circle_radius_m")

    lines = [
        f"functional_type: {functional_type}",
        f"shape: {shape}",
        f"circle_radius_m: {circle_radius_m}",
        f"circle_center_xy: {circle_center_xy}",
    ]
    text = "\n".join(lines)
    draw.rectangle((10, 10, 560, 130), fill=(0, 0, 0))
    draw.multiline_text((18, 18), text, fill=(255, 255, 255), spacing=4)

    try:
        center = np.asarray(circle_center_xy, dtype=np.float32).reshape(-1)
        radius_m = float(circle_radius_m)
    except Exception:
        center = np.asarray([], dtype=np.float32)
        radius_m = float("nan")

    if center.size >= 2 and np.isfinite(radius_m) and radius_m > 0 and float(bev_resolution) > 1e-8:
        cx = float(center[0]) / float(bev_resolution)
        cy = float(center[1]) / float(bev_resolution)
        radius_px = float(radius_m) / float(bev_resolution)
        if np.isfinite(cx) and np.isfinite(cy) and np.isfinite(radius_px):
            x0 = cx - radius_px
            y0 = cy - radius_px
            x1 = cx + radius_px
            y1 = cy + radius_px
            draw.ellipse((x0, y0, x1, y1), outline=(255, 60, 60), width=3)
            draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(60, 220, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return output_path


def _to_pretty_json_text(payload: Any, *, max_chars: int = 2400) -> str:
    try:
        if isinstance(payload, (dict, list)):
            txt = json.dumps(payload, ensure_ascii=False, indent=2)
        else:
            txt = str(payload)
    except Exception:
        txt = str(payload)
    txt = txt.strip()
    if len(txt) > int(max_chars):
        txt = txt[: int(max_chars)] + "\n...<truncated>"
    return txt


def _annotate_front_render_image(
    *,
    image: Image.Image,
    object_label: str,
    semantic: dict[str, Any],
    output_path: Path,
) -> Path:
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    semantic_txt = _to_pretty_json_text(semantic, max_chars=1600)
    raw_lines = [
        f"{object_label}",
        "semantic:",
        semantic_txt,
    ]

    wrapped_lines: list[str] = []
    wrap_width = max(40, min(96, int((w - 40) / 8)))
    for block in raw_lines:
        block_lines = str(block).splitlines() or [""]
        for ln in block_lines:
            parts = textwrap.wrap(ln, width=wrap_width) or [""]
            wrapped_lines.extend(parts)

    line_h = 14
    max_lines = max(10, int((h - 30) / line_h))
    if len(wrapped_lines) > max_lines:
        wrapped_lines = wrapped_lines[: max_lines - 1] + ["...<truncated>"]
    text = "\n".join(wrapped_lines)

    panel_w = min(w - 20, max(280, int(w * 0.72)))
    panel_h = min(h - 20, 16 + line_h * max(1, len(wrapped_lines)))
    draw.rectangle((10, 10, 10 + panel_w, 10 + panel_h), fill=(0, 0, 0))
    draw.multiline_text((18, 18), text, fill=(255, 255, 255), spacing=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return output_path


def _to_candidate_id_array(records: list[dict[str, Any]]) -> np.ndarray:
    arr = np.empty((len(records),), dtype=object)
    for i, rec in enumerate(records):
        arr[i] = np.asarray(rec.get("candidate_class_ids", []), dtype=np.int32)
    return arr


def _save_records_npz(output_path: Path, records: list[dict[str, Any]]) -> None:
    record_id = np.asarray([str(rec.get("record_id", "")) for rec in records], dtype="<U128")
    branch = np.asarray([str(rec.get("branch", "")) for rec in records], dtype="<U16")
    scene_name = np.asarray([str(rec.get("scene_name", "")) for rec in records], dtype="<U64")
    object_type = np.asarray([str(rec.get("object_type", "")) for rec in records], dtype="<U32")
    object_id = np.asarray([int(rec.get("object_id", -1)) for rec in records], dtype=np.int32)
    scene_id = np.asarray([int(rec.get("scene_id", -1)) for rec in records], dtype=np.int32)
    global_gid = np.asarray([int(rec.get("global_gid", -1)) for rec in records], dtype=np.int32)
    class_id = np.asarray([int(rec.get("class_id", -1)) for rec in records], dtype=np.int32)
    confidence = np.asarray([float(rec.get("confidence", 0.0)) for rec in records], dtype=np.float32)
    semantic_json = np.asarray([str(rec.get("semantic_attributes_json", "{}")) for rec in records], dtype=object)
    geometry_json = np.asarray([str(rec.get("geometry_attributes_json", "{}")) for rec in records], dtype=object)
    evidence_json = np.asarray([str(rec.get("evidence_json", "{}")) for rec in records], dtype=object)
    candidate_class_ids = _to_candidate_id_array(records)
    np.savez_compressed(
        output_path,
        record_id=record_id,
        branch=branch,
        scene_name=scene_name,
        object_type=object_type,
        object_id=object_id,
        scene_id=scene_id,
        global_gid=global_gid,
        class_id=class_id,
        candidate_class_ids=candidate_class_ids,
        semantic_attributes_json=semantic_json,
        geometry_attributes_json=geometry_json,
        evidence_json=evidence_json,
        confidence=confidence,
    )


def _save_records_json(output_path: Path, records: list[dict[str, Any]]) -> None:
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def _extract_pixel_xy_from_mask(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask)
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    if m.ndim != 2:
        return np.zeros((0, 2), dtype=np.int32)
    y, x = np.nonzero(m.astype(bool, copy=False))
    if x.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    return np.stack([x.astype(np.int32), y.astype(np.int32)], axis=1)


def _load_bev_instances(bev_instances_path: Path) -> list[dict[str, Any]]:
    if not bev_instances_path.exists():
        return []
    data = np.load(bev_instances_path, allow_pickle=True)
    out: list[dict[str, Any]] = []

    if "masks" in data.files:
        masks = np.asarray(data["masks"])
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim == 3:
            for i in range(masks.shape[0]):
                pixel_xy = _extract_pixel_xy_from_mask(masks[i])
                if pixel_xy.shape[0] == 0:
                    continue
                out.append(
                    {
                        "object_id": int(i),
                        "class_id": 9,
                        "pixel_xy": pixel_xy,
                    }
                )
        return out

    # Fallback: boxes only.
    if "boxes" in data.files:
        boxes = np.asarray(data["boxes"], dtype=np.float32).reshape(-1, 4)
        for i, box in enumerate(boxes):
            x0, y0, x1, y1 = [int(round(float(v))) for v in box.tolist()]
            xs = np.arange(min(x0, x1), max(x0, x1) + 1, dtype=np.int32)
            ys = np.arange(min(y0, y1), max(y0, y1) + 1, dtype=np.int32)
            if xs.size == 0 or ys.size == 0:
                continue
            grid_x, grid_y = np.meshgrid(xs, ys)
            pixel_xy = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)
            out.append(
                {
                    "object_id": int(i),
                    "class_id": 9,
                    "pixel_xy": pixel_xy,
                }
            )
    return out


def _run_semantic_glm(
    *,
    model: Any | None,
    processor: Any | None,
    image_1: Path,
    image_2: Path,
    prompt: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if model is None or processor is None:
        return {}
    try:
        return run_dual_image_glm(
            model=model,
            processor=processor,
            image_1_path=image_1,
            image_2_path=image_2,
            prompt=prompt,
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            repetition_penalty=float(args.repetition_penalty),
            top_p=float(args.top_p),
            top_k=int(args.top_k),
        )
    except Exception as exc:
        return {"error": str(exc)}


def run_bev_global(
    *,
    data_root: Path,
    bev_dir: Path,
    model: Any | None,
    processor: Any | None,
    run_vlm: bool,
    run_geometry: bool,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    bev_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    global_rgb_path = bev_dir / "global_rgb.png"
    bev_instances_path = bev_dir / "global_instances.npz"
    out_npz = bev_dir / "bev_attributes_global.npz"
    out_json = bev_dir / "bev_attributes_global.json"
    if out_npz.exists() and out_json.exists() and (not args.overwrite):
        try:
            return json.loads(out_json.read_text(encoding="utf-8"))
        except Exception:
            pass

    if not global_rgb_path.exists() or not bev_instances_path.exists():
        _save_records_npz(out_npz, records)
        _save_records_json(out_json, records)
        return records

    global_img = Image.open(global_rgb_path).convert("RGB")
    instances = _load_bev_instances(bev_instances_path)
    crop_dir = bev_dir / "task6_bev_crops"
    annotated_dir = bev_dir / "task6_bev_annotated"
    crop_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    for obj in tqdm(instances, desc="Task6 BEV", unit="obj"):
        object_id = int(obj["object_id"])
        class_id = int(obj.get("class_id", 9))
        pixel_xy = np.asarray(obj.get("pixel_xy", np.zeros((0, 2), dtype=np.int32)), dtype=np.int32).reshape(-1, 2)
        if pixel_xy.shape[0] == 0:
            continue

        local_crop, local_meta = _adaptive_crop_from_pixels(
            global_img,
            pixel_xy,
            fill_ratio=float(args.target_fill_ratio),
        )
        center = pixel_xy.mean(axis=0)
        context_crop = _fixed_context_crop(
            global_img,
            center_xy=(int(round(float(center[0]))), int(round(float(center[1])))),
            size=int(args.bev_global_size),
        )

        local_path = crop_dir / f"manhole_{object_id:05d}_local.png"
        context_path = crop_dir / f"manhole_{object_id:05d}_global.png"
        local_crop.save(local_path)
        context_crop.save(context_path)

        semantic = {}
        if bool(run_vlm):
            semantic = _run_semantic_glm(
                model=model,
                processor=processor,
                image_1=local_path,
                image_2=context_path,
                prompt=MANHOLE_PROMPT,
                args=args,
            )
        geometry: dict[str, Any] = {}
        if bool(run_geometry):
            geometry = compute_manhole_geometry_from_pixels(
                pixel_xy,
                resolution_m_per_px=float(args.bev_resolution),
            )
        annotated_global_path = _annotate_bev_global_image(
            global_image=global_img,
            semantic=semantic if isinstance(semantic, dict) else {},
            geometry=geometry if isinstance(geometry, dict) else {},
            bev_resolution=float(args.bev_resolution),
            output_path=annotated_dir / f"manhole_{object_id:05d}_global_annotated.png",
        )
        confidence = float(semantic.get("confidence", 0.0)) if isinstance(semantic, dict) else 0.0
        evidence = {
            "source_global_rgb": str(global_rgb_path),
            "local_image": str(local_path),
            "global_image": str(context_path),
            "annotated_global_image": str(annotated_global_path),
            "pixel_count": int(pixel_xy.shape[0]),
            "crop_meta": local_meta,
        }
        records.append(
            {
                "record_id": f"bev:global:manhole:{object_id}",
                "branch": "bev",
                "scene_name": "global",
                "object_type": "manhole",
                "object_id": int(object_id),
                "class_id": int(class_id),
                "candidate_class_ids": [int(class_id)],
                "semantic_attributes_json": _json_dumps(semantic if isinstance(semantic, dict) else {}),
                "geometry_attributes_json": _json_dumps(geometry if isinstance(geometry, dict) else {}),
                "evidence_json": _json_dumps(evidence),
                "confidence": confidence,
            }
        )

    _save_records_npz(out_npz, records)
    _save_records_json(out_json, records)
    return records


def _scan_all_mapping_stems(projected_dir: Path) -> list[str]:
    return sorted([p.stem for p in projected_dir.glob("station_*_cam_*.npz")])


def _choose_view_stems(
    projected_dir: Path,
    *,
    preferred_front: str | None,
    preferred_side: str | None,
    point_indices: np.ndarray,
) -> tuple[str | None, str | None, dict[str, np.ndarray]]:
    stem_cache: dict[str, np.ndarray] = {}
    score: dict[str, int] = {}

    def _load_pixels(stem: str) -> np.ndarray:
        if stem in stem_cache:
            return stem_cache[stem]
        pixels, _ = _project_object_pixels(projected_dir, stem, point_indices)
        stem_cache[stem] = pixels
        score[stem] = int(pixels.shape[0])
        return pixels

    preferred = [s for s in [preferred_front, preferred_side] if isinstance(s, str) and s]
    for stem in preferred:
        _load_pixels(stem)

    valid_pref = [stem for stem in preferred if score.get(stem, 0) > 0]
    if len(valid_pref) >= 2:
        return valid_pref[0], valid_pref[1], stem_cache

    all_stems = _scan_all_mapping_stems(projected_dir)
    for stem in all_stems:
        if stem in score:
            continue
        _load_pixels(stem)
    ranked = sorted(all_stems, key=lambda s: score.get(s, 0), reverse=True)
    ranked = [stem for stem in ranked if score.get(stem, 0) > 0]
    if not ranked:
        return None, None, stem_cache
    if len(ranked) == 1:
        return ranked[0], ranked[0], stem_cache
    return ranked[0], ranked[1], stem_cache


def _build_front_record(
    *,
    scene_name: str,
    scene_id: int,
    object_type: str,
    object_id: int,
    global_gid: int,
    class_id: int,
    candidate_class_ids: list[int],
    semantic: dict[str, Any],
    geometry: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    confidence = float(semantic.get("confidence", 0.0)) if isinstance(semantic, dict) else 0.0
    return {
        "record_id": f"front:{scene_name}:{object_type}:{object_id}",
        "branch": "front",
        "scene_name": scene_name,
        "scene_id": int(scene_id),
        "object_type": object_type,
        "object_id": int(object_id),
        "global_gid": int(global_gid),
        "class_id": int(class_id),
        "candidate_class_ids": [int(x) for x in candidate_class_ids],
        "semantic_attributes_json": _json_dumps(semantic if isinstance(semantic, dict) else {}),
        "geometry_attributes_json": _json_dumps(geometry if isinstance(geometry, dict) else {}),
        "evidence_json": _json_dumps(evidence),
        "confidence": confidence,
    }


def run_front_by_scene(
    *,
    data_root: Path,
    task5_output_dir: str,
    model: Any | None,
    processor: Any | None,
    run_vlm: bool,
    run_geometry: bool,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    scene_dirs = task5_utils.discover_scene_dirs(data_root)
    scene_id_map = task5_utils.build_scene_id_map(data_root)
    all_records: list[dict[str, Any]] = []

    for scene_dir in tqdm(scene_dirs, desc="Task6 Front Scenes", unit="scene"):
        scene_name = scene_dir.name
        scene_id = scene_id_map.get(scene_name, None)
        if scene_id is None:
            continue
        fusion_dir = scene_dir / task5_output_dir
        final_npz_path = fusion_dir / f"{scene_name}_instance_seg_final.npz"
        tree_metrics_path = fusion_dir / f"{scene_name}_tree_metrics.npz"
        if not final_npz_path.exists():
            continue

        scene_output_dir = scene_dir / "attributes"
        scene_output_dir.mkdir(parents=True, exist_ok=True)
        out_npz = scene_output_dir / f"{scene_name}_task6_front_attributes.npz"
        out_json = scene_output_dir / f"{scene_name}_task6_front_attributes.json"
        if out_npz.exists() and out_json.exists() and (not args.overwrite):
            try:
                records = json.loads(out_json.read_text(encoding="utf-8"))
                if isinstance(records, list):
                    has_gid = all(isinstance(rec, dict) and ("global_gid" in rec) for rec in records)
                    if has_gid:
                        all_records.extend(records)
                        continue
            except Exception:
                pass

        try:
            points_xyz, points_rgb, las_path = _load_scene_points(scene_dir)
        except Exception:
            continue
        task5_csf_ground_las_path = fusion_dir / f"{scene_name}_scene_csf_ground.las"
        if not task5_csf_ground_las_path.exists():
            task5_csf_ground_las_path = fusion_dir / f"{scene_name}_fence_csf_ground.las"
        task5_ground_points_xyz = _load_task5_csf_ground_points(task5_csf_ground_las_path)
        tree_metrics = _load_tree_metrics(tree_metrics_path)
        scene_instances, pole_groups = _load_scene_front_objects(final_npz_path)
        projected_dir = scene_dir / "projected_images"
        stations = _load_stations(projected_dir)
        render_dir = scene_output_dir / "task6_front_renders"
        render_dir.mkdir(parents=True, exist_ok=True)

        scene_records: list[dict[str, Any]] = []

        for pole_group in tqdm(pole_groups, desc=f"{scene_name} pole_groups", unit="obj", leave=False):
            pole_id = int(pole_group["pole_group_id"])
            candidate_class_ids = [int(x) for x in pole_group.get("candidate_class_ids", [])]
            candidate_class_names = [str(x) for x in pole_group.get("candidate_class_names", []) if str(x)]
            point_indices = np.asarray(pole_group.get("point_indices", np.zeros((0,), dtype=np.int64)), dtype=np.int64).reshape(-1)
            if point_indices.size == 0:
                continue
            valid = (point_indices >= 0) & (point_indices < points_xyz.shape[0])
            point_indices = np.unique(point_indices[valid]).astype(np.int64, copy=False)
            if point_indices.size == 0:
                continue
            object_points_xyz = points_xyz[point_indices.astype(np.int64, copy=False)]
            object_points_rgb = points_rgb[point_indices.astype(np.int64, copy=False)]
            object_center = object_points_xyz.mean(axis=0)

            nearest_station = _select_nearest_station(stations, object_center[:2])
            if nearest_station is None:
                station_center_xyz = np.asarray(
                    [float(object_center[0]) - 8.0, float(object_center[1]), float(object_center[2])],
                    dtype=np.float32,
                )
                station_idx: int | None = None
            else:
                station_center_xyz = np.asarray(nearest_station.get("station_xyz", np.asarray([object_center[0], object_center[1], object_center[2]], dtype=np.float32)), dtype=np.float32).reshape(3)
                station_idx = int(nearest_station.get("station_idx", -1))

            geometry: dict[str, Any] = {}
            if bool(run_geometry):
                geometry = geometry_for_pole_group(
                    points_xyz,
                    point_indices,
                    candidate_class_ids=candidate_class_ids,
                    ground_points_xyz=task5_ground_points_xyz,
                    neighborhood_radius=float(getattr(args, "ground_neighborhood_radius", 2.0)),
                    neighborhood_quantile=float(getattr(args, "ground_neighborhood_quantile", 0.10)),
                    precomputed_diameter_m=pole_group.get("diameter_m"),
                    precomputed_center_xy=pole_group.get("center_xy"),
                )

            semantic: dict[str, Any] = {}
            front_crop_path: Path | None = None
            side_crop_path: Path | None = None
            front_annotated_path: Path | None = None
            side_annotated_path: Path | None = None
            front_meta: dict[str, Any] | None = None
            side_meta: dict[str, Any] | None = None
            render_status = "front_render_failed"
            side_from_front = False

            rendered = _render_instance_two_views(
                object_points_xyz=object_points_xyz,
                object_points_rgb=object_points_rgb,
                station_center_xyz=station_center_xyz,
                args=args,
            )
            if rendered is not None:
                (front_crop, front_meta_raw), side_view = rendered
                front_meta = dict(front_meta_raw)
                front_crop_path = render_dir / f"pole_group_{pole_id:05d}_front.png"
                front_crop.save(front_crop_path)

                side_vlm_path = front_crop_path
                if side_view is not None:
                    side_crop, side_meta_raw = side_view
                    side_meta = dict(side_meta_raw)
                    side_crop_path = render_dir / f"pole_group_{pole_id:05d}_side90.png"
                    side_crop.save(side_crop_path)
                    side_vlm_path = side_crop_path
                    render_status = "ok_two_views"
                else:
                    side_from_front = True
                    render_status = "ok_front_only_side_reused"

                if bool(run_vlm):
                    semantic = _run_semantic_glm(
                        model=model,
                        processor=processor,
                        image_1=front_crop_path,
                        image_2=side_vlm_path,
                        prompt=build_pole_group_prompt(candidate_class_names),
                        args=args,
                    )
                    front_annotated_path = render_dir / f"pole_group_{pole_id:05d}_front_annotated.png"
                    _annotate_front_render_image(
                        image=front_crop,
                        object_label=f"pole_group:{pole_id}",
                        semantic=semantic if isinstance(semantic, dict) else {},
                        output_path=front_annotated_path,
                    )
                    if side_view is not None and side_crop_path is not None:
                        side_annotated_path = render_dir / f"pole_group_{pole_id:05d}_side90_annotated.png"
                        _annotate_front_render_image(
                            image=side_crop,
                            object_label=f"pole_group:{pole_id}",
                            semantic=semantic if isinstance(semantic, dict) else {},
                            output_path=side_annotated_path,
                        )

            evidence = {
                "task5_final_npz": str(final_npz_path),
                "task5_scene_csf_ground_las": str(task5_csf_ground_las_path) if task5_csf_ground_las_path.exists() else None,
                "task5_fence_csf_ground_las": str(task5_csf_ground_las_path) if task5_csf_ground_las_path.exists() else None,
                "source_las": str(las_path),
                "nearest_station_idx": station_idx,
                "nearest_station_center_xyz": [
                    float(station_center_xyz[0]),
                    float(station_center_xyz[1]),
                    float(station_center_xyz[2]),
                ],
                "ground_neighborhood_radius": float(getattr(args, "ground_neighborhood_radius", 2.0)),
                "ground_neighborhood_quantile": float(getattr(args, "ground_neighborhood_quantile", 0.10)),
                "front_image": str(front_crop_path) if front_crop_path is not None else None,
                "side_image": str(side_crop_path) if side_crop_path is not None else None,
                "front_annotated_image": str(front_annotated_path) if front_annotated_path is not None else None,
                "side_annotated_image": str(side_annotated_path) if side_annotated_path is not None else None,
                "point_count": int(point_indices.size),
                "task5_pole_metric_source": str(pole_group.get("metric_source", "")),
                "render_method": "task2_pc2img_soft_splat",
                "front_render_meta": front_meta,
                "side_render_meta": side_meta,
                "render_status": render_status,
                "side_from_front": bool(side_from_front),
            }
            scene_records.append(
                _build_front_record(
                    scene_name=scene_name,
                    scene_id=int(scene_id),
                    object_type="pole_group",
                    object_id=pole_id,
                    global_gid=task5_utils.encode_global_gid(
                        scene_id=int(scene_id),
                        object_type="pole_group",
                        object_id=int(pole_id),
                    ),
                    class_id=0,
                    candidate_class_ids=candidate_class_ids,
                    semantic=semantic,
                    geometry=geometry,
                    evidence=evidence,
                )
            )

        for scene_inst in tqdm(scene_instances, desc=f"{scene_name} scene_instances", unit="obj", leave=False):
            scene_instance_id = int(scene_inst["scene_instance_id"])
            class_id = int(scene_inst["class_id"])
            if class_id not in SUPPORTED_SCENE_INSTANCE_CLASSES:
                continue
            point_indices = np.asarray(scene_inst.get("point_indices", np.zeros((0,), dtype=np.int64)), dtype=np.int64).reshape(-1)
            if point_indices.size == 0:
                continue
            valid = (point_indices >= 0) & (point_indices < points_xyz.shape[0])
            point_indices = np.unique(point_indices[valid]).astype(np.int64, copy=False)
            if point_indices.size == 0:
                continue
            object_points_xyz = points_xyz[point_indices.astype(np.int64, copy=False)]
            object_points_rgb = points_rgb[point_indices.astype(np.int64, copy=False)]
            object_center = object_points_xyz.mean(axis=0)

            nearest_station = _select_nearest_station(stations, object_center[:2])
            if nearest_station is None:
                station_center_xyz = np.asarray(
                    [float(object_center[0]) - 8.0, float(object_center[1]), float(object_center[2])],
                    dtype=np.float32,
                )
                station_idx: int | None = None
            else:
                station_center_xyz = np.asarray(nearest_station.get("station_xyz", np.asarray([object_center[0], object_center[1], object_center[2]], dtype=np.float32)), dtype=np.float32).reshape(3)
                station_idx = int(nearest_station.get("station_idx", -1))

            tree_metric = tree_metrics.get(scene_instance_id) if class_id == 7 else None
            geometry: dict[str, Any] = {}
            if bool(run_geometry):
                geometry = geometry_for_scene_instance(
                    points_xyz,
                    point_indices,
                    class_id=class_id,
                    tree_metric=tree_metric,
                    ground_points_xyz=task5_ground_points_xyz,
                    neighborhood_radius=float(getattr(args, "ground_neighborhood_radius", 2.0)),
                    neighborhood_quantile=float(getattr(args, "ground_neighborhood_quantile", 0.10)),
                    mid_centroid_low_ratio=float(getattr(args, "mid_centroid_low_ratio", 0.30)),
                    mid_centroid_high_ratio=float(getattr(args, "mid_centroid_high_ratio", 0.70)),
                    fence_mid_band_low_ratio=float(getattr(args, "fence_mid_band_low_ratio", 0.30)),
                    fence_mid_band_high_ratio=float(getattr(args, "fence_mid_band_high_ratio", 0.70)),
                    fence_control_point_count=int(getattr(args, "fence_control_point_count", 5)),
                    fence_grid_size=float(getattr(args, "fence_centerline_grid_size", 0.10)),
                    fence_knn=int(getattr(args, "fence_centerline_knn", 8)),
                )

            semantic: dict[str, Any] = {}
            front_crop_path: Path | None = None
            side_crop_path: Path | None = None
            front_annotated_path: Path | None = None
            side_annotated_path: Path | None = None
            front_meta: dict[str, Any] | None = None
            side_meta: dict[str, Any] | None = None
            render_status = "skipped_non_tree"
            side_from_front = False

            if class_id == 7:
                rendered = _render_instance_two_views(
                    object_points_xyz=object_points_xyz,
                    object_points_rgb=object_points_rgb,
                    station_center_xyz=station_center_xyz,
                    args=args,
                )
                if rendered is not None:
                    (front_crop, front_meta_raw), side_view = rendered
                    front_meta = dict(front_meta_raw)
                    front_crop_path = render_dir / f"scene_instance_{scene_instance_id:05d}_front.png"
                    front_crop.save(front_crop_path)

                    side_vlm_path = front_crop_path
                    if side_view is not None:
                        side_crop, side_meta_raw = side_view
                        side_meta = dict(side_meta_raw)
                        side_crop_path = render_dir / f"scene_instance_{scene_instance_id:05d}_side90.png"
                        side_crop.save(side_crop_path)
                        side_vlm_path = side_crop_path
                        render_status = "ok_two_views"
                    else:
                        side_from_front = True
                        render_status = "ok_front_only_side_reused"

                    if bool(run_vlm):
                        semantic = _run_semantic_glm(
                            model=model,
                            processor=processor,
                            image_1=front_crop_path,
                            image_2=side_vlm_path,
                            prompt=TREE_PROMPT,
                            args=args,
                        )
                        front_annotated_path = render_dir / f"scene_instance_{scene_instance_id:05d}_front_annotated.png"
                        _annotate_front_render_image(
                            image=front_crop,
                            object_label=f"scene_instance:{scene_instance_id}:class_{class_id}",
                            semantic=semantic if isinstance(semantic, dict) else {},
                            output_path=front_annotated_path,
                        )
                        if side_view is not None and side_crop_path is not None:
                            side_annotated_path = render_dir / f"scene_instance_{scene_instance_id:05d}_side90_annotated.png"
                            _annotate_front_render_image(
                                image=side_crop,
                                object_label=f"scene_instance:{scene_instance_id}:class_{class_id}",
                                semantic=semantic if isinstance(semantic, dict) else {},
                                output_path=side_annotated_path,
                            )
                else:
                    render_status = "front_render_failed"

            evidence = {
                "task5_final_npz": str(final_npz_path),
                "task5_scene_csf_ground_las": str(task5_csf_ground_las_path) if task5_csf_ground_las_path.exists() else None,
                "task5_fence_csf_ground_las": str(task5_csf_ground_las_path) if task5_csf_ground_las_path.exists() else None,
                "source_las": str(las_path),
                "nearest_station_idx": station_idx,
                "nearest_station_center_xyz": [
                    float(station_center_xyz[0]),
                    float(station_center_xyz[1]),
                    float(station_center_xyz[2]),
                ],
                "ground_neighborhood_radius": float(getattr(args, "ground_neighborhood_radius", 2.0)),
                "ground_neighborhood_quantile": float(getattr(args, "ground_neighborhood_quantile", 0.10)),
                "front_image": str(front_crop_path) if front_crop_path is not None else None,
                "side_image": str(side_crop_path) if side_crop_path is not None else None,
                "point_count": int(point_indices.size),
                "render_method": "task2_pc2img_soft_splat",
                "front_render_meta": front_meta,
                "side_render_meta": side_meta,
                "render_status": render_status,
                "side_from_front": bool(side_from_front),
            }
            if front_annotated_path is not None and side_annotated_path is not None:
                evidence["front_annotated_image"] = str(front_annotated_path)
                evidence["side_annotated_image"] = str(side_annotated_path)
            scene_records.append(
                _build_front_record(
                    scene_name=scene_name,
                    scene_id=int(scene_id),
                    object_type="scene_instance",
                    object_id=scene_instance_id,
                    global_gid=task5_utils.encode_global_gid(
                        scene_id=int(scene_id),
                        object_type="scene_instance",
                        object_id=int(scene_instance_id),
                    ),
                    class_id=class_id,
                    candidate_class_ids=[class_id],
                    semantic=semantic,
                    geometry=geometry,
                    evidence=evidence,
                )
            )

        _save_records_npz(out_npz, scene_records)
        _save_records_json(out_json, scene_records)
        all_records.extend(scene_records)

    return all_records


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.exists() or not data_root.is_dir():
        raise NotADirectoryError(f"Invalid data root: {data_root}")

    bev_dir = Path(args.bev_dir).expanduser()
    if not bev_dir.is_absolute():
        bev_dir = (Path.cwd() / bev_dir).resolve()

    run_branch = _normalize_run_branch(str(getattr(args, "run_branch", "both")))
    run_bev = run_branch in {"both", "bev"}
    run_front = run_branch in {"both", "front"}
    run_stage = str(getattr(args, "run_stage", "both")).strip().lower()
    run_geometry = run_stage in {"both", "geometry"}
    run_vlm_requested = run_stage in {"both", "vlm"}
    run_vlm = bool(run_vlm_requested) and (not bool(args.disable_glm))

    model = None
    processor = None
    if run_vlm:
        model, processor = load_glm_model_and_processor(str(args.model_path))
    elif run_vlm_requested and bool(args.disable_glm):
        print("[task6] warning: run-stage requires vlm but --disable-glm is enabled, semantic inference will be skipped.")

    if not run_bev and not run_front:
        print("[task6] nothing to run: run-branch disables both BEV and Front.")
        return
    if not run_geometry and not run_vlm:
        print("[task6] nothing to run: run-stage disables both geometry and vlm.")
        return

    bev_records: list[dict[str, Any]] = []
    if run_bev:
        bev_records = run_bev_global(
            data_root=data_root,
            bev_dir=bev_dir,
            model=model,
            processor=processor,
            run_vlm=run_vlm,
            run_geometry=run_geometry,
            args=args,
        )
        print(f"[task6] bev records: {len(bev_records)}")
    else:
        print("[task6] bev skipped by --run-branch")

    front_records: list[dict[str, Any]] = []
    if run_front:
        front_records = run_front_by_scene(
            data_root=data_root,
            task5_output_dir=str(args.task5_output_dir),
            model=model,
            processor=processor,
            run_vlm=run_vlm,
            run_geometry=run_geometry,
            args=args,
        )
        print(f"[task6] front records: {len(front_records)}")

        global_dir = data_root / "attributes_global"
        global_dir.mkdir(parents=True, exist_ok=True)
        merged_npz = global_dir / "task6_front_attributes_merged.npz"
        merged_json = global_dir / "task6_front_attributes_merged.json"
        _save_records_npz(merged_npz, front_records)
        _save_records_json(merged_json, front_records)
        print(f"[task6] merged front outputs -> {merged_npz} | {merged_json}")
    else:
        print("[task6] front skipped by --run-branch")


if __name__ == "__main__":
    main()
