from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import laspy
import numpy as np
from PIL import Image
from tqdm import tqdm

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import utils as task5_utils
from task6_geometry_utils import (
    compute_global_ground_mask,
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


SUPPORTED_SCENE_INSTANCE_CLASSES = frozenset({7, 8, 10})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task6 attribute extraction (BEV global + Front by scene).")
    parser.add_argument("--data-root", required=True, help="Benchmark root.")
    parser.add_argument("--task5-output-dir", type=str, default="fusion", help="Task5 output directory under each scene.")
    parser.add_argument("--bev-dir", type=str, default="benchmark/bev", help="BEV global directory path.")
    parser.add_argument("--target-fill-ratio", type=float, default=0.8, help="Adaptive crop target fill ratio.")
    parser.add_argument("--bev-global-size", type=int, default=500, help="Global BEV context crop size in pixels.")
    parser.add_argument("--bev-resolution", type=float, default=0.02, help="BEV resolution (meters per pixel).")
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

    if "scene_instance_id" in data.files:
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
        for idx, pole_id in enumerate(pole_ids):
            candidate_ids = candidate_ids_arr[idx] if idx < len(candidate_ids_arr) else np.zeros((0,), dtype=np.int64)
            if idx < candidate_names_arr.shape[0]:
                raw_names = np.asarray(candidate_names_arr[idx], dtype=object).reshape(-1)
                candidate_names = [str(x) for x in raw_names if str(x)]
            else:
                candidate_names = [task5_utils.CLASS_ID_TO_NAME.get(int(x), "") for x in candidate_ids]
            points = point_arrays[idx] if idx < len(point_arrays) else np.zeros((0,), dtype=np.int64)
            pole_groups.append(
                {
                    "pole_group_id": int(pole_id),
                    "candidate_class_ids": [int(x) for x in np.asarray(candidate_ids, dtype=np.int32).reshape(-1)],
                    "candidate_class_names": candidate_names,
                    "point_indices": points,
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


def _load_scene_points(scene_dir: Path) -> tuple[np.ndarray, Path]:
    las_path = task5_utils.find_las_path(scene_dir)
    if las_path is None:
        raise FileNotFoundError(f"Missing scene LAS under: {scene_dir}")
    las_data = laspy.read(las_path)
    points_xyz = np.vstack([las_data.x, las_data.y, las_data.z]).T.astype(np.float32, copy=False)
    return points_xyz, las_path


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
        translations: list[np.ndarray] = []
        for cam_name, mat_raw in station.items():
            try:
                mat = np.asarray(mat_raw, dtype=np.float32)
            except Exception:
                continue
            if mat.shape != (4, 4):
                continue
            t = mat[:2, 3].astype(np.float32, copy=False)
            forward = mat[:2, 2].astype(np.float32, copy=False)
            norm = float(np.linalg.norm(forward))
            if norm <= 1e-6:
                continue
            forward = (forward / norm).astype(np.float32, copy=False)
            cam_items.append((str(cam_name), mat, forward))
            translations.append(t)
        if not cam_items:
            continue
        station_xy = np.median(np.stack(translations, axis=0), axis=0).astype(np.float32, copy=False)
        parsed.append(
            {
                "station_idx": int(station_idx),
                "station_xy": station_xy,
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
    crop_dir.mkdir(parents=True, exist_ok=True)

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

        semantic = _run_semantic_glm(
            model=model,
            processor=processor,
            image_1=local_path,
            image_2=context_path,
            prompt=MANHOLE_PROMPT,
            args=args,
        )
        geometry = compute_manhole_geometry_from_pixels(
            pixel_xy,
            resolution_m_per_px=float(args.bev_resolution),
        )
        confidence = float(semantic.get("confidence", 0.0)) if isinstance(semantic, dict) else 0.0
        evidence = {
            "source_global_rgb": str(global_rgb_path),
            "local_image": str(local_path),
            "global_image": str(context_path),
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
    object_type: str,
    object_id: int,
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
        "object_type": object_type,
        "object_id": int(object_id),
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
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    scene_dirs = task5_utils.discover_scene_dirs(data_root)
    all_records: list[dict[str, Any]] = []

    for scene_dir in tqdm(scene_dirs, desc="Task6 Front Scenes", unit="scene"):
        scene_name = scene_dir.name
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
                all_records.extend(records if isinstance(records, list) else [])
                continue
            except Exception:
                pass

        try:
            points_xyz, las_path = _load_scene_points(scene_dir)
        except Exception:
            continue
        global_ground_mask = compute_global_ground_mask(points_xyz)
        tree_metrics = _load_tree_metrics(tree_metrics_path)
        scene_instances, pole_groups = _load_scene_front_objects(final_npz_path)
        projected_dir = scene_dir / "projected_images"
        stations = _load_stations(projected_dir)
        crop_dir = scene_output_dir / "task6_front_crops"
        crop_dir.mkdir(parents=True, exist_ok=True)

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
            object_xy_center = points_xyz[point_indices, :2].mean(axis=0)
            pref_front, pref_side = _select_station_front_side_stems(stations, object_xy_center)
            front_stem, side_stem, pixel_cache = _choose_view_stems(
                projected_dir,
                preferred_front=pref_front,
                preferred_side=pref_side,
                point_indices=point_indices,
            )
            if not front_stem or not side_stem:
                continue
            front_pixels = pixel_cache.get(front_stem, np.zeros((0, 2), dtype=np.int32))
            side_pixels = pixel_cache.get(side_stem, np.zeros((0, 2), dtype=np.int32))
            front_pixels, front_image_path = _project_object_pixels(projected_dir, front_stem, point_indices) if front_pixels.size == 0 else (front_pixels, projected_dir / f"{front_stem}.png")
            side_pixels, side_image_path = _project_object_pixels(projected_dir, side_stem, point_indices) if side_pixels.size == 0 else (side_pixels, projected_dir / f"{side_stem}.png")
            if front_image_path is None or side_image_path is None:
                continue

            front_crop, front_meta = _adaptive_crop_from_pixels(
                Image.open(front_image_path).convert("RGB"),
                front_pixels,
                fill_ratio=float(args.target_fill_ratio),
            )
            side_crop, side_meta = _adaptive_crop_from_pixels(
                Image.open(side_image_path).convert("RGB"),
                side_pixels,
                fill_ratio=float(args.target_fill_ratio),
            )
            front_crop_path = crop_dir / f"pole_group_{pole_id:05d}_front.png"
            side_crop_path = crop_dir / f"pole_group_{pole_id:05d}_side90.png"
            front_crop.save(front_crop_path)
            side_crop.save(side_crop_path)

            semantic = _run_semantic_glm(
                model=model,
                processor=processor,
                image_1=front_crop_path,
                image_2=side_crop_path,
                prompt=build_pole_group_prompt(candidate_class_names),
                args=args,
            )
            geometry = geometry_for_pole_group(
                points_xyz,
                point_indices,
                candidate_class_ids=candidate_class_ids,
                global_ground_mask=global_ground_mask,
            )
            evidence = {
                "task5_final_npz": str(final_npz_path),
                "source_las": str(las_path),
                "front_stem": front_stem,
                "side_stem": side_stem,
                "front_image": str(front_crop_path),
                "side_image": str(side_crop_path),
                "point_count": int(point_indices.size),
                "front_crop_meta": front_meta,
                "side_crop_meta": side_meta,
            }
            scene_records.append(
                _build_front_record(
                    scene_name=scene_name,
                    object_type="pole_group",
                    object_id=pole_id,
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

            object_xy_center = points_xyz[point_indices, :2].mean(axis=0)
            pref_front, pref_side = _select_station_front_side_stems(stations, object_xy_center)
            front_stem, side_stem, pixel_cache = _choose_view_stems(
                projected_dir,
                preferred_front=pref_front,
                preferred_side=pref_side,
                point_indices=point_indices,
            )
            if not front_stem or not side_stem:
                continue
            front_pixels = pixel_cache.get(front_stem, np.zeros((0, 2), dtype=np.int32))
            side_pixels = pixel_cache.get(side_stem, np.zeros((0, 2), dtype=np.int32))
            front_pixels, front_image_path = _project_object_pixels(projected_dir, front_stem, point_indices) if front_pixels.size == 0 else (front_pixels, projected_dir / f"{front_stem}.png")
            side_pixels, side_image_path = _project_object_pixels(projected_dir, side_stem, point_indices) if side_pixels.size == 0 else (side_pixels, projected_dir / f"{side_stem}.png")
            if front_image_path is None or side_image_path is None:
                continue

            front_crop, front_meta = _adaptive_crop_from_pixels(
                Image.open(front_image_path).convert("RGB"),
                front_pixels,
                fill_ratio=float(args.target_fill_ratio),
            )
            side_crop, side_meta = _adaptive_crop_from_pixels(
                Image.open(side_image_path).convert("RGB"),
                side_pixels,
                fill_ratio=float(args.target_fill_ratio),
            )
            front_crop_path = crop_dir / f"scene_instance_{scene_instance_id:05d}_front.png"
            side_crop_path = crop_dir / f"scene_instance_{scene_instance_id:05d}_side90.png"
            front_crop.save(front_crop_path)
            side_crop.save(side_crop_path)

            if class_id == 7:
                semantic = _run_semantic_glm(
                    model=model,
                    processor=processor,
                    image_1=front_crop_path,
                    image_2=side_crop_path,
                    prompt=TREE_PROMPT,
                    args=args,
                )
            else:
                semantic = {}

            tree_metric = tree_metrics.get(scene_instance_id) if class_id == 7 else None
            geometry = geometry_for_scene_instance(
                points_xyz,
                point_indices,
                class_id=class_id,
                tree_metric=tree_metric,
                global_ground_mask=global_ground_mask,
            )
            evidence = {
                "task5_final_npz": str(final_npz_path),
                "source_las": str(las_path),
                "front_stem": front_stem,
                "side_stem": side_stem,
                "front_image": str(front_crop_path),
                "side_image": str(side_crop_path),
                "point_count": int(point_indices.size),
                "front_crop_meta": front_meta,
                "side_crop_meta": side_meta,
            }
            scene_records.append(
                _build_front_record(
                    scene_name=scene_name,
                    object_type="scene_instance",
                    object_id=scene_instance_id,
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

    model = None
    processor = None
    if not bool(args.disable_glm):
        model, processor = load_glm_model_and_processor(str(args.model_path))

    bev_records = run_bev_global(
        data_root=data_root,
        bev_dir=bev_dir,
        model=model,
        processor=processor,
        args=args,
    )
    print(f"[task6] bev records: {len(bev_records)}")

    front_records = run_front_by_scene(
        data_root=data_root,
        task5_output_dir=str(args.task5_output_dir),
        model=model,
        processor=processor,
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


if __name__ == "__main__":
    main()
