from __future__ import annotations

import argparse

import json

import math

from dataclasses import dataclass

from pathlib import Path

from typing import Any

import laspy

import numpy as np

from scipy.spatial import cKDTree

from sklearn.cluster import DBSCAN

from tqdm import tqdm

try:
    import CSF
except Exception:
    CSF = None

CLASS_ID_TO_NAME: dict[int, str] = {
    1: "电线杆",
    2: "路灯杆",
    3: "路牌",
    4: "交通标志",
    5: "红绿灯",
    6: "监控",
    7: "行道树",
    8: "果壳箱",
    9: "消防栓",
    10: "电箱",
    11: "雕塑",
    12: "座椅",
    13: "交通锥",
    14: "柱墩",
    15: "围栏",
}

CLASS_NAME_TO_ID = {name: cid for cid, name in CLASS_ID_TO_NAME.items()}

NUM_CLASSES = len(CLASS_ID_TO_NAME)

TREE_CLASS_ID = 7

FENCE_CLASS_ID = 15

FULL_POLE_CLASS_IDS = frozenset({1, 2})

SIGN_LIKE_CLASS_IDS = frozenset({3, 4, 5, 6})

GROUND_FILTER_CLASS_IDS = frozenset({1, 2, 8, 9, 11, 12, 13, 14, 15})

@dataclass
class CandidateInstance:
    image_stem: str
    class_id: int
    sam_score: float
    class_confidence: float
    view_weight: float
    weighted_score: float
    point_indices: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    xy_center: np.ndarray | None = None

@dataclass
class SceneInstance:
    class_scores: np.ndarray
    class_id: int
    class_confidence: float
    point_indices: np.ndarray

class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

def discover_scene_dirs(data_root: Path) -> list[Path]:
    scene_dirs: list[Path] = []
    for child in sorted(data_root.iterdir()):
        if child.is_dir() and (child / "projected_images").is_dir():
            scene_dirs.append(child)
    return scene_dirs

def find_las_path(scene_dir: Path) -> Path | None:
    source_dir = scene_dir / "source"
    candidates = sorted(source_dir.glob("*.las")) if source_dir.is_dir() else []
    if not candidates:
        candidates = sorted(scene_dir.glob("*.las"))
    return candidates[0] if candidates else None

def _npz_get(npz: Any, key: str) -> Any | None:
    return npz[key] if key in npz.files else None

def _normalize_masks(masks_raw: Any, image_shape: tuple[int, int]) -> np.ndarray:
    if masks_raw is None:
        h, w = image_shape
        return np.zeros((0, h, w), dtype=bool)
    arr = np.asarray(masks_raw)
    if arr.size == 0:
        h, w = image_shape
        return np.zeros((0, h, w), dtype=bool)
    if arr.ndim == 4:
        if arr.shape[1] != 1:
            raise ValueError(f"Unsupported mask shape: {arr.shape}")
        arr = arr[:, 0]
    elif arr.ndim != 3:
        raise ValueError(f"Unsupported mask shape: {arr.shape}")
    return arr.astype(bool, copy=False)

def _ensure_float_array(values: Any | None, n: int, default: float) -> np.ndarray:
    if values is None:
        return np.full((n,), default, dtype=np.float32)
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.shape[0] >= n:
        return arr[:n]
    out = np.full((n,), default, dtype=np.float32)
    out[: arr.shape[0]] = arr
    return out

def _ensure_class_ids(npz: Any, n: int) -> np.ndarray:
    class_ids_raw = _npz_get(npz, "class_ids")
    if class_ids_raw is not None:
        arr = np.asarray(class_ids_raw, dtype=np.int32).reshape(-1)
        if arr.shape[0] >= n:
            return arr[:n]
        out = np.zeros((n,), dtype=np.int32)
        out[: arr.shape[0]] = arr
        return out

    names_raw = _npz_get(npz, "class_names")
    if names_raw is None:
        return np.zeros((n,), dtype=np.int32)

    names = np.asarray(names_raw).reshape(-1)
    out = np.zeros((n,), dtype=np.int32)
    for i, name in enumerate(names[:n]):
        out[i] = CLASS_NAME_TO_ID.get(str(name), 0)
    return out

def _load_effective_stations(projected_dir: Path) -> list[dict[str, Any]] | None:
    metadata_path = projected_dir / "effective_stations.json"
    if not metadata_path.exists():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    stations = payload.get("stations")
    if not isinstance(stations, list):
        return None
    return stations

def _parse_image_stem(image_stem: str) -> tuple[int, str] | None:
    prefix = "station_"
    cam_token = "_cam_"
    if not image_stem.startswith(prefix) or cam_token not in image_stem:
        return None
    station_raw, cam_name = image_stem[len(prefix):].split(cam_token, 1)
    if not cam_name:
        return None
    try:
        return int(station_raw), cam_name
    except ValueError:
        return None

def _effective_extrinsic_for_image_stem(stations: list[dict[str, Any]] | None, image_stem: str) -> np.ndarray | None:
    if not stations:
        return None
    parsed = _parse_image_stem(image_stem)
    if parsed is None:
        return None
    station_idx, cam_name = parsed
    if station_idx < 0 or station_idx >= len(stations):
        return None
    station = stations[station_idx]
    if not isinstance(station, dict) or cam_name not in station:
        return None
    try:
        extrinsic = np.asarray(station[cam_name], dtype=np.float32)
    except Exception:
        return None
    if extrinsic.shape != (4, 4):
        return None
    return extrinsic

def _mapped_point_depths(points_xyz: np.ndarray, pts_indices: np.ndarray, extrinsic: np.ndarray | None) -> np.ndarray | None:
    if extrinsic is None or pts_indices.size == 0:
        return None
    try:
        trans_mat = np.linalg.inv(np.asarray(extrinsic, dtype=np.float32))[:3, :]
    except np.linalg.LinAlgError:
        return None
    pts = points_xyz[pts_indices.astype(np.int64, copy=False)]
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])
    pts_cam = (trans_mat @ pts_h.T).T
    return pts_cam[:, 2].astype(np.float32, copy=False)

def _view_weight_from_mask(mask: np.ndarray, fov_deg: float) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return 0.0, 0.0

    _, w = mask.shape
    cx = 0.5 * (w - 1)
    obj_cx = float(xs.mean())

    # Only use horizontal yaw offset from image center.
    f = w / (2.0 * math.tan(math.radians(fov_deg * 0.5)))
    theta_x = math.degrees(math.atan2(abs(obj_cx - cx), f))
    alpha_offset = theta_x

    alpha_max = max(0.0, 90.0 - alpha_offset)
    weight = max(0.0, 1.0 - abs(alpha_max - 90.0) / 90.0)
    return weight, alpha_max

def _project_mask_to_points(
    mask: np.ndarray,
    pts_img_indices: np.ndarray,
    pts_indices: np.ndarray,
    num_points: int,
    *,
    point_depths: np.ndarray | None = None,
    depth_threshold: float = 0.0,
) -> np.ndarray:
    if pts_img_indices.size == 0:
        return np.zeros((0,), dtype=np.int32)

    flat_mask = mask.reshape(-1)
    valid_img = (pts_img_indices >= 0) & (pts_img_indices < flat_mask.size)
    if not np.any(valid_img):
        return np.zeros((0,), dtype=np.int32)

    img_indices = pts_img_indices[valid_img]
    point_indices = pts_indices[valid_img]
    depth_values = None
    if point_depths is not None:
        depth_values = np.asarray(point_depths, dtype=np.float32).reshape(-1)
        depth_values = depth_values[valid_img]
        valid_depth = np.isfinite(depth_values) & (depth_values > 0)
        if not np.any(valid_depth):
            return np.zeros((0,), dtype=np.int32)
        img_indices = img_indices[valid_depth]
        point_indices = point_indices[valid_depth]
        depth_values = depth_values[valid_depth]

    point_keep = flat_mask[img_indices]
    if not np.any(point_keep):
        return np.zeros((0,), dtype=np.int32)

    kept_img_indices = img_indices[point_keep]
    projected = point_indices[point_keep]
    if depth_values is not None:
        kept_depth_values = depth_values[point_keep]
        if depth_threshold > 0:
            _, inverse = np.unique(kept_img_indices, return_inverse=True)
            front_depth = np.full((int(inverse.max()) + 1,), np.inf, dtype=np.float32)
            np.minimum.at(front_depth, inverse, kept_depth_values)
            depth_keep = kept_depth_values <= (front_depth[inverse] + float(depth_threshold))
            projected = projected[depth_keep]

    valid_pts = (projected >= 0) & (projected < num_points)
    projected = projected[valid_pts]
    if projected.size == 0:
        return np.zeros((0,), dtype=np.int32)
    return np.unique(projected.astype(np.int32, copy=False))

def _bboxes_overlap(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray) -> bool:
    return not (
        (a_max[0] < b_min[0])
        or (b_max[0] < a_min[0])
        or (a_max[1] < b_min[1])
        or (b_max[1] < a_min[1])
        or (a_max[2] < b_min[2])
        or (b_max[2] < a_min[2])
    )

def _point_iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    inter = np.intersect1d(a, b, assume_unique=True).size
    if inter == 0:
        return 0.0
    union = a.size + b.size - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)

def _classes_pass_merge_gate(class_id_a: int, class_id_b: int) -> bool:
    if class_id_a not in CLASS_ID_TO_NAME or class_id_b not in CLASS_ID_TO_NAME:
        return False
    if class_id_a == class_id_b:
        return True
    return class_id_a in FULL_POLE_CLASS_IDS and class_id_b in FULL_POLE_CLASS_IDS

def _candidate_xy_center_distance(a: CandidateInstance, b: CandidateInstance) -> float:
    center_a = np.asarray(a.xy_center, dtype=np.float32) if a.xy_center is not None else 0.5 * (a.bbox_min[:2] + a.bbox_max[:2])
    center_b = np.asarray(b.xy_center, dtype=np.float32) if b.xy_center is not None else 0.5 * (b.bbox_min[:2] + b.bbox_max[:2])
    return float(np.linalg.norm(center_a - center_b))

def _supports_xy_supplementary_merge(class_id_a: int, class_id_b: int) -> bool:
    if class_id_a == 7 or class_id_b == 7:
        return False
    if class_id_a == class_id_b and class_id_a in CLASS_ID_TO_NAME:
        return True
    return class_id_a in FULL_POLE_CLASS_IDS and class_id_b in FULL_POLE_CLASS_IDS

def _should_merge_candidates(
    a: CandidateInstance,
    b: CandidateInstance,
    *,
    iou_threshold: float,
    merge_xy_distance: float,
) -> bool:
    if not _classes_pass_merge_gate(a.class_id, b.class_id):
        return False

    if _bboxes_overlap(a.bbox_min, a.bbox_max, b.bbox_min, b.bbox_max):
        if _point_iou(a.point_indices, b.point_indices) >= iou_threshold:
            return True

    if not _supports_xy_supplementary_merge(a.class_id, b.class_id):
        return False

    return merge_xy_distance > 0 and _candidate_xy_center_distance(a, b) <= merge_xy_distance

def _summarize_candidate_points(candidates: list[CandidateInstance]) -> tuple[int, int]:
    if not candidates:
        return 0, 0
    point_sum = int(sum(int(c.point_indices.size) for c in candidates))
    point_unique = int(np.unique(np.concatenate([c.point_indices for c in candidates], axis=0)).size)
    return point_sum, point_unique

def _format_scene_backprojection_log(
    scene_name: str,
    *,
    candidate_stats: dict[str, int],
    candidate_point_sum: int,
    candidate_point_unique: int,
) -> str:
    return (
        f"[{scene_name}] back-projection: sam_masks={int(candidate_stats['sam_instances_total'])} "
        f"kept_candidates={int(candidate_stats['sam_instances_kept'])} "
        f"kept_points_sum={candidate_point_sum} kept_points_unique={candidate_point_unique} "
        f"filtered_small={int(candidate_stats['sam_instances_filtered_small'])} "
        f"filtered_invalid_class={int(candidate_stats['sam_instances_filtered_invalid_class'])} "
        f"missing_sam_files={int(candidate_stats['missing_sam_files'])}"
    )

def _format_iou_merge_log(scene_name: str, candidate_count: int, merged_count: int) -> str:
    return f"[{scene_name}] IoU merge: candidates={candidate_count} -> merged_instances={merged_count}"

def collect_scene_candidates(
    scene_dir: Path,
    points_xyz: np.ndarray,
    *,
    fov_deg: float,
    min_mask_points: int,
    backproject_depth_threshold: float = 0.20,
) -> tuple[list[CandidateInstance], dict[str, int]]:
    projected_dir = scene_dir / "projected_images"
    sam_dir = scene_dir / "sam_mask"
    mapping_files = sorted(projected_dir.glob("station_*_cam_*.npz"))
    effective_stations = _load_effective_stations(projected_dir)

    stats = {
        "mapping_files": len(mapping_files),
        "missing_sam_files": 0,
        "sam_instances_total": 0,
        "sam_instances_kept": 0,
        "sam_instances_filtered_small": 0,
        "sam_instances_filtered_invalid_class": 0,
    }

    candidates: list[CandidateInstance] = []
    num_points = points_xyz.shape[0]

    for mapping_path in tqdm(mapping_files, desc=f"BackProject [{scene_dir.name}]", unit="img"):
        image_stem = mapping_path.stem
        sam_path = sam_dir / f"{image_stem}.npz"
        if not sam_path.exists():
            stats["missing_sam_files"] += 1
            continue

        try:
            mapping_npz = np.load(mapping_path)
            dist_img = np.asarray(mapping_npz["dist_img"])
            pts_img_indices = np.asarray(mapping_npz["pts_img_indices"]).reshape(-1).astype(np.int64, copy=False)
            pts_indices = np.asarray(mapping_npz["pts_indices"]).reshape(-1).astype(np.int64, copy=False)
        except Exception:
            continue

        if pts_img_indices.shape[0] != pts_indices.shape[0]:
            k = min(pts_img_indices.shape[0], pts_indices.shape[0])
            pts_img_indices = pts_img_indices[:k]
            pts_indices = pts_indices[:k]

        valid_pts = (pts_indices >= 0) & (pts_indices < num_points)
        pts_img_indices = pts_img_indices[valid_pts]
        pts_indices = pts_indices[valid_pts]

        point_depths = _mapped_point_depths(
            points_xyz,
            pts_indices,
            _effective_extrinsic_for_image_stem(effective_stations, image_stem),
        )

        image_shape = tuple(int(x) for x in dist_img.shape)
        try:
            sam_npz = np.load(sam_path, allow_pickle=True)
            masks = _normalize_masks(_npz_get(sam_npz, "masks"), image_shape)
        except Exception:
            continue

        n_masks = int(masks.shape[0])
        stats["sam_instances_total"] += n_masks
        if n_masks == 0:
            continue

        scores = _ensure_float_array(_npz_get(sam_npz, "scores"), n_masks, default=1.0)
        class_confs = _ensure_float_array(_npz_get(sam_npz, "class_confidences"), n_masks, default=1.0)
        class_ids = _ensure_class_ids(sam_npz, n_masks)

        for i in range(n_masks):
            class_id = int(class_ids[i])
            mask = masks[i]
            projected_points = _project_mask_to_points(
                mask,
                pts_img_indices,
                pts_indices,
                num_points=num_points,
                point_depths=point_depths,
                depth_threshold=float(backproject_depth_threshold),
            )
            projected_count = int(projected_points.size)

            if class_id not in CLASS_ID_TO_NAME:
                stats["sam_instances_filtered_invalid_class"] += 1
                continue

            if projected_count < min_mask_points:
                stats["sam_instances_filtered_small"] += 1
                continue

            view_weight, _ = _view_weight_from_mask(mask, fov_deg=fov_deg)
            sam_score = float(scores[i])
            class_conf = float(class_confs[i])
            weighted_score = float(view_weight * sam_score * class_conf)

            xyz = points_xyz[projected_points]
            bbox_min = xyz.min(axis=0)
            bbox_max = xyz.max(axis=0)
            xy_center = xyz[:, :2].mean(axis=0)

            candidates.append(
                CandidateInstance(
                    image_stem=image_stem,
                    class_id=class_id,
                    sam_score=sam_score,
                    class_confidence=class_conf,
                    view_weight=float(view_weight),
                    weighted_score=weighted_score,
                    point_indices=projected_points,
                    bbox_min=bbox_min.astype(np.float32, copy=False),
                    bbox_max=bbox_max.astype(np.float32, copy=False),
                    xy_center=xy_center.astype(np.float32, copy=False),
                )
            )
            stats["sam_instances_kept"] += 1

    return candidates, stats

def merge_candidates(
    candidates: list[CandidateInstance],
    *,
    iou_threshold: float,
    merge_xy_distance: float,
    min_merged_points: int,
) -> tuple[list[SceneInstance], np.ndarray]:
    if not candidates:
        return [], np.zeros((0,), dtype=np.int32)

    n = len(candidates)
    uf = UnionFind(n)
    for i in range(n):
        ci = candidates[i]
        for j in range(i + 1, n):
            cj = candidates[j]
            if _should_merge_candidates(ci, cj, iou_threshold=iou_threshold, merge_xy_distance=merge_xy_distance):
                uf.union(i, j)

    groups: dict[int, list[int]] = {}
    for idx in range(n):
        root = uf.find(idx)
        groups.setdefault(root, []).append(idx)

    candidate_to_instance = np.full((n,), -1, dtype=np.int32)
    merged_instances: list[SceneInstance] = []
    for root in sorted(groups.keys()):
        group_indices = groups[root]
        union_points = np.unique(np.concatenate([candidates[idx].point_indices for idx in group_indices], axis=0))
        if union_points.size < max(0, int(min_merged_points)):
            continue
        class_scores = np.zeros((NUM_CLASSES + 1,), dtype=np.float32)
        for idx in group_indices:
            cand = candidates[idx]
            if cand.class_id in CLASS_ID_TO_NAME:
                class_scores[cand.class_id] += float(cand.weighted_score)

        class_only = class_scores[1:]
        if class_only.size > 0 and float(class_only.sum()) > 0:
            best_offset = int(np.argmax(class_only))
            class_id = best_offset + 1
            class_conf = float(class_scores[class_id] / class_only.sum())
        else:
            class_id = 0
            class_conf = 0.0

        instance_id = len(merged_instances)
        for idx in group_indices:
            candidate_to_instance[idx] = instance_id

        merged_instances.append(
            SceneInstance(
                class_scores=class_scores,
                class_id=class_id,
                class_confidence=class_conf,
                point_indices=union_points.astype(np.int32, copy=False),
            )
        )

    return merged_instances, candidate_to_instance

def _prune_tree_points_before_assignment(
    candidates: list[CandidateInstance],
    candidate_to_instance: np.ndarray,
    instances: list[SceneInstance],
) -> int:
    if not candidates or not instances or candidate_to_instance.size == 0:
        return 0

    full_pole_instance_ids = {
        inst_id for inst_id, inst in enumerate(instances) if inst.class_id in FULL_POLE_CLASS_IDS
    }
    sign_like_instance_ids = {
        inst_id for inst_id, inst in enumerate(instances) if inst.class_id in SIGN_LIKE_CLASS_IDS
    }
    tree_instance_ids = {inst_id for inst_id, inst in enumerate(instances) if inst.class_id == TREE_CLASS_ID}
    fence_instance_ids = {inst_id for inst_id, inst in enumerate(instances) if inst.class_id == FENCE_CLASS_ID}

    sign_like_chunks = [
        inst.point_indices.astype(np.int32, copy=False)
        for inst_id, inst in enumerate(instances)
        if inst_id in sign_like_instance_ids and inst.point_indices.size > 0
    ]
    sign_like_points = (
        np.unique(np.concatenate(sign_like_chunks, axis=0)).astype(np.int32, copy=False)
        if sign_like_chunks
        else np.zeros((0,), dtype=np.int32)
    )

    non_tree_chunks = [
        inst.point_indices.astype(np.int32, copy=False)
        for inst in instances
        if inst.class_id not in {TREE_CLASS_ID, FENCE_CLASS_ID} and inst.point_indices.size > 0
    ]
    non_tree_points = (
        np.unique(np.concatenate(non_tree_chunks, axis=0)).astype(np.int32, copy=False)
        if non_tree_chunks
        else np.zeros((0,), dtype=np.int32)
    )

    tree_chunks = [
        inst.point_indices.astype(np.int32, copy=False)
        for inst_id, inst in enumerate(instances)
        if inst_id in tree_instance_ids and inst.point_indices.size > 0
    ]
    fence_chunks = [
        inst.point_indices.astype(np.int32, copy=False)
        for inst_id, inst in enumerate(instances)
        if inst_id in fence_instance_ids and inst.point_indices.size > 0
    ]
    tree_points = (
        np.unique(np.concatenate(tree_chunks, axis=0)).astype(np.int32, copy=False)
        if tree_chunks
        else np.zeros((0,), dtype=np.int32)
    )
    fence_points = (
        np.unique(np.concatenate(fence_chunks, axis=0)).astype(np.int32, copy=False)
        if fence_chunks
        else np.zeros((0,), dtype=np.int32)
    )
    tree_fence_only_points = np.zeros((0,), dtype=np.int32)
    if tree_points.size > 0 and fence_points.size > 0:
        tree_fence_only_points = np.intersect1d(tree_points, fence_points, assume_unique=False)
        if non_tree_points.size > 0 and tree_fence_only_points.size > 0:
            tree_fence_only_points = tree_fence_only_points[
                ~np.isin(tree_fence_only_points, non_tree_points, assume_unique=False)
            ]

    removed_total = 0
    if sign_like_points.size > 0:
        for inst_id in full_pole_instance_ids:
            inst = instances[inst_id]
            keep_mask = ~np.isin(inst.point_indices, sign_like_points, assume_unique=False)
            removed_total += int(inst.point_indices.size - np.count_nonzero(keep_mask))
            inst.point_indices = inst.point_indices[keep_mask].astype(np.int32, copy=False)

    if non_tree_points.size > 0:
        for inst_id in tree_instance_ids:
            inst = instances[inst_id]
            keep_mask = ~np.isin(inst.point_indices, non_tree_points, assume_unique=False)
            removed_total += int(inst.point_indices.size - np.count_nonzero(keep_mask))
            inst.point_indices = inst.point_indices[keep_mask].astype(np.int32, copy=False)

    if tree_fence_only_points.size > 0:
        for inst_id in fence_instance_ids:
            inst = instances[inst_id]
            keep_mask = ~np.isin(inst.point_indices, tree_fence_only_points, assume_unique=False)
            removed_total += int(inst.point_indices.size - np.count_nonzero(keep_mask))
            inst.point_indices = inst.point_indices[keep_mask].astype(np.int32, copy=False)

    for cand_idx, cand in enumerate(candidates):
        if cand_idx >= candidate_to_instance.shape[0]:
            break
        if cand.point_indices.size == 0:
            continue
        inst_id = int(candidate_to_instance[cand_idx])
        if inst_id < 0:
            continue
        if inst_id in full_pole_instance_ids and sign_like_points.size > 0:
            keep_mask = ~np.isin(cand.point_indices, sign_like_points, assume_unique=False)
            cand.point_indices = cand.point_indices[keep_mask].astype(np.int32, copy=False)
        if inst_id in tree_instance_ids and non_tree_points.size > 0:
            keep_mask = ~np.isin(cand.point_indices, non_tree_points, assume_unique=False)
            cand.point_indices = cand.point_indices[keep_mask].astype(np.int32, copy=False)
        elif inst_id in fence_instance_ids and tree_fence_only_points.size > 0:
            keep_mask = ~np.isin(cand.point_indices, tree_fence_only_points, assume_unique=False)
            cand.point_indices = cand.point_indices[keep_mask].astype(np.int32, copy=False)

    return removed_total

def _ground_bbox_keep_mask(
    points_xyz: np.ndarray,
    *,
    ground_quantile: float,
    support_height: float,
    bbox_expand: float,
    support_top_ratio: float = 0.70,
) -> np.ndarray:
    n = int(points_xyz.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=bool)

    q = min(1.0, max(0.0, float(ground_quantile)))
    z = points_xyz[:, 2]
    z_low = float(np.quantile(z, q))
    z_high = float(np.quantile(z, 0.95))
    span = max(0.0, z_high - z_low)
    if not np.isfinite(span) or span <= 1e-6:
        return np.ones((n,), dtype=bool)

    low_ratio = min(1.0, max(0.0, float(support_height)))
    top_ratio = min(1.0, max(0.0, float(support_top_ratio)))
    lower_height = z_low + low_ratio * span
    upper_height = z_low + top_ratio * span
    if upper_height <= lower_height:
        support_mask = z > lower_height
    else:
        support_mask = (z >= lower_height) & (z <= upper_height)
        if not np.any(support_mask):
            support_mask = z > lower_height
    if not np.any(support_mask):
        return np.ones((n,), dtype=bool)

    support_xy = points_xyz[support_mask, :2]
    bbox_min = support_xy.min(axis=0) - float(bbox_expand)
    bbox_max = support_xy.max(axis=0) + float(bbox_expand)
    xy = points_xyz[:, :2]
    inside_bbox = np.all((xy >= bbox_min) & (xy <= bbox_max), axis=1)
    low_ground = points_xyz[:, 2] <= lower_height
    return inside_bbox | (~low_ground)


def _csf_ground_mask(
    points_xyz: np.ndarray,
    *,
    cloth_resolution: float = 1.0,
    rigidness: int = 1,
    time_step: float = 0.65,
    class_threshold: float = 1.2,
    iterations: int = 800,
    slope_smooth: bool = True,
) -> np.ndarray | None:
    """
    Return per-point ground mask with CSF.
    Returns None when CSF is unavailable or fails.
    """
    if CSF is None:
        return None
    n = int(points_xyz.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=bool)

    try:
        mins = points_xyz.min(axis=0)
        maxs = points_xyz.max(axis=0)
        center = (mins + maxs) / 2.0
        centered = points_xyz - center

        csf = CSF.CSF()
        csf.setPointCloud(centered.astype(np.float64, copy=False))
        csf.params.cloth_resolution = float(cloth_resolution)
        csf.params.rigidness = int(rigidness)
        csf.params.time_step = float(time_step)
        csf.params.class_threshold = float(class_threshold)
        csf.params.interations = int(iterations)
        csf.params.bSloopSmooth = bool(slope_smooth)

        ground = CSF.VecInt()
        non_ground = CSF.VecInt()
        csf.do_filtering(ground, non_ground)
    except Exception:
        return None

    mask = np.zeros((n,), dtype=bool)
    for idx in ground:
        idx_int = int(idx)
        if 0 <= idx_int < n:
            mask[idx_int] = True
    return mask

def _cluster_point_indices(
    points_xyz: np.ndarray,
    point_indices: np.ndarray,
    *,
    eps: float,
    use_xy_only: bool = False,
    min_samples: int = 1,
) -> list[np.ndarray]:
    point_indices = np.asarray(point_indices, dtype=np.int32).reshape(-1)
    if point_indices.size == 0:
        return []

    pts = points_xyz[point_indices]
    if use_xy_only:
        pts = pts[:, :2]

    min_samples = max(1, int(min_samples))
    if eps <= 0:
        if min_samples <= 1:
            return [np.asarray([idx], dtype=np.int32) for idx in point_indices]
        return []

    labels = DBSCAN(eps=float(eps), min_samples=min_samples).fit_predict(pts)
    unique_labels = [int(label) for label in np.unique(labels) if int(label) >= 0]
    clusters = [
        np.sort(point_indices[labels == label].astype(np.int32, copy=False))
        for label in unique_labels
    ]
    clusters.sort(key=lambda arr: (-int(arr.size), int(arr[0]) if arr.size > 0 else -1))
    return clusters

def _robust_height_range(points_xyz: np.ndarray, *, low_q: float = 0.05, high_q: float = 0.95) -> float:
    if points_xyz.size == 0:
        return 0.0
    z = np.asarray(points_xyz[:, 2], dtype=np.float32)
    return float(np.quantile(z, high_q) - np.quantile(z, low_q))

def _tree_ground_z_from_effective_stations(projected_dir: Path) -> float | None:
    stations = _load_effective_stations(projected_dir)
    if not stations:
        return None

    first_station = stations[0]
    if not isinstance(first_station, dict):
        return None

    for value in first_station.values():
        try:
            extrinsic = np.asarray(value, dtype=np.float32)
        except Exception:
            continue
        if extrinsic.shape == (4, 4) and np.isfinite(extrinsic[2, 3]):
            return float(extrinsic[2, 3] - 2.0)
    return None


def _tree_station_ground_refs_from_effective_stations(projected_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Build per-station XY + ground-z references from effective_stations.json.
    Each station ref is aggregated (median) from all valid camera extrinsics in that station.
    Returns:
      station_xy: (N, 2) float32
      station_ground_z: (N,) float32 where ground_z = station_z - 2.0
    """
    stations = _load_effective_stations(projected_dir)
    if not stations:
        return None

    xy_list: list[np.ndarray] = []
    gz_list: list[float] = []
    for station in stations:
        if not isinstance(station, dict):
            continue
        t_list: list[np.ndarray] = []
        for value in station.values():
            try:
                extrinsic = np.asarray(value, dtype=np.float32)
            except Exception:
                continue
            if extrinsic.shape != (4, 4):
                continue
            t = extrinsic[:3, 3].astype(np.float32, copy=False)
            if np.all(np.isfinite(t)):
                t_list.append(t)
        if not t_list:
            continue
        t_stack = np.stack(t_list, axis=0)
        t_median = np.median(t_stack, axis=0).astype(np.float32, copy=False)
        xy_list.append(t_median[:2].astype(np.float32, copy=False))
        gz_list.append(float(t_median[2] - 2.0))

    if not xy_list:
        return None

    station_xy = np.stack(xy_list, axis=0).astype(np.float32, copy=False)
    station_ground_z = np.asarray(gz_list, dtype=np.float32).reshape(-1)
    return station_xy, station_ground_z

def _robust_xy_radius(points_xyz: np.ndarray) -> float:
    if points_xyz.size == 0:
        return math.inf
    xy = np.asarray(points_xyz[:, :2], dtype=np.float32)
    center = xy.mean(axis=0)
    radii = np.linalg.norm(xy - center[None, :], axis=1)
    return float(np.quantile(radii, 0.90)) if radii.size > 0 else math.inf


def _fit_circle_taubin_svd_xy(points_xyz: np.ndarray) -> tuple[np.ndarray, float] | None:
    xy = np.asarray(points_xyz[:, :2], dtype=np.float64)
    if xy.shape[0] < 3:
        return None

    centroid = xy.mean(axis=0)
    centered = xy - centroid[None, :]
    x = centered[:, 0]
    y = centered[:, 1]
    z = x * x + y * y
    z_mean = float(z.mean())
    if (not np.isfinite(z_mean)) or z_mean <= 1e-12:
        return None

    z0 = (z - z_mean) / (2.0 * math.sqrt(z_mean))
    mat = np.column_stack((z0, x, y))
    try:
        _, _, vh = np.linalg.svd(mat, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if vh.shape[0] == 0:
        return None

    a = np.asarray(vh[-1], dtype=np.float64)
    a0 = float(a[0]) / (2.0 * math.sqrt(z_mean))
    if (not np.isfinite(a0)) or abs(a0) < 1e-12:
        return None

    a1 = float(a[1])
    a2 = float(a[2])
    a3 = -z_mean * a0

    cx_local = -a1 / (2.0 * a0)
    cy_local = -a2 / (2.0 * a0)
    center = np.asarray([cx_local, cy_local], dtype=np.float64) + centroid

    radicand = (a1 * a1 + a2 * a2 - 4.0 * a0 * a3) / (4.0 * a0 * a0)
    if (not np.isfinite(radicand)) or radicand <= 0:
        return None

    radius = float(math.sqrt(radicand))
    if not np.isfinite(radius):
        return None

    return center.astype(np.float32, copy=False), radius


def _trimmed_mean(values: np.ndarray, trim_ratio: float) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.inf
    ratio = min(0.49, max(0.0, float(trim_ratio)))
    if ratio <= 0.0:
        return float(arr.mean())
    k = int(math.floor(arr.size * ratio))
    if 2 * k >= arr.size:
        return float(arr.mean())
    arr_sorted = np.sort(arr)
    return float(arr_sorted[k: arr.size - k].mean())


def _taubin_radius_and_trimmed_residual(points_xyz: np.ndarray, *, trim_ratio: float = 0.05) -> tuple[float, float]:
    fit = _fit_circle_taubin_svd_xy(points_xyz)
    if fit is None:
        return math.inf, math.inf

    center, radius = fit
    xy = np.asarray(points_xyz[:, :2], dtype=np.float32)
    radial_dist = np.linalg.norm(xy - center[None, :], axis=1)
    residual = np.abs(radial_dist - float(radius))
    trimmed_residual = _trimmed_mean(residual, trim_ratio=trim_ratio)
    if not np.isfinite(trimmed_residual):
        return math.inf, math.inf
    return float(radius), float(trimmed_residual)


def _principal_verticality(points_xyz: np.ndarray) -> float:
    if points_xyz.shape[0] < 2:
        return 0.0
    centered = np.asarray(points_xyz, dtype=np.float32) - np.asarray(points_xyz, dtype=np.float32).mean(axis=0, keepdims=True)
    if not np.any(np.abs(centered) > 0):
        return 0.0
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.0
    if vh.size == 0:
        return 0.0
    return float(abs(vh[0, 2]))



def _principal_direction(points_xyz: np.ndarray) -> np.ndarray:
    if points_xyz.shape[0] < 2:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    centered = np.asarray(points_xyz, dtype=np.float32) - np.asarray(points_xyz, dtype=np.float32).mean(axis=0, keepdims=True)
    if not np.any(np.abs(centered) > 0):
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    if vh.size == 0:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    direction = np.asarray(vh[0], dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    return (direction / norm).astype(np.float32, copy=False)
def _copy_scene_instance(inst: SceneInstance) -> SceneInstance:
    return SceneInstance(
        class_scores=np.array(inst.class_scores, copy=True),
        class_id=int(inst.class_id),
        class_confidence=float(inst.class_confidence),
        point_indices=np.asarray(inst.point_indices, dtype=np.int32).copy(),
    )

def _build_point_instance_id_from_instances(num_points: int, instances: list[SceneInstance]) -> np.ndarray:
    point_instance_id = np.full((num_points,), -1, dtype=np.int32)
    for inst_id, inst in enumerate(instances):
        if inst.point_indices.size == 0:
            continue
        point_instance_id[inst.point_indices.astype(np.int64, copy=False)] = int(inst_id)
    return point_instance_id

def _refine_instances_with_ground_and_fence(
    points_xyz: np.ndarray,
    instances: list[SceneInstance],
    point_instance_id: np.ndarray,
    *,
    ground_quantile: float,
    support_height: float,
    bbox_expand: float,
    support_top_ratio: float = 0.70,
    fence_recluster_eps: float,
    fence_min_cluster_points: int,
    fence_min_height: float,
    fence_dbscan_min_samples: int = 1,
    fence_csf_cloth_resolution: float = 1.0,
    fence_csf_rigidness: int = 1,
    fence_csf_time_step: float = 0.65,
    fence_csf_class_threshold: float = 1.2,
    fence_csf_iterations: int = 800,
    fence_csf_slope_smooth: bool = True,
    fence_csf_low_band_ratio: float = 0.10,
    return_fence_global_ground_mask: bool = False,
) -> tuple[list[SceneInstance], np.ndarray, dict[str, int]] | tuple[list[SceneInstance], np.ndarray, dict[str, int], np.ndarray | None]:
    refined_instances: list[SceneInstance] = []
    refined_point_instance_id = np.full(point_instance_id.shape, -1, dtype=np.int32)

    ground_removed_total = 0
    fence_point_chunks: list[np.ndarray] = []
    fence_score_chunks: list[np.ndarray] = []
    fence_confidences: list[float] = []

    for inst in instances:
        pts = inst.point_indices.astype(np.int32, copy=False)
        if pts.size == 0:
            continue

        if inst.class_id == 15:
            # Fence instances skip bbox-based ground removal and rely on CSF only.
            fence_point_chunks.append(pts)
            fence_score_chunks.append(inst.class_scores.astype(np.float32, copy=False))
            fence_confidences.append(float(inst.class_confidence))
            continue

        refined_pts = pts
        keep_mask = _ground_bbox_keep_mask(
            points_xyz[pts],
            ground_quantile=ground_quantile,
            support_height=support_height,
            bbox_expand=bbox_expand,
            support_top_ratio=support_top_ratio,
        )
        ground_removed_total += int(pts.size - np.count_nonzero(keep_mask))
        refined_pts = pts[keep_mask].astype(np.int32, copy=False)

        if refined_pts.size == 0:
            continue

        refined_instances.append(
            SceneInstance(
                class_scores=inst.class_scores,
                class_id=inst.class_id,
                class_confidence=inst.class_confidence,
                point_indices=refined_pts,
            )
        )

    fence_instances_before = len(fence_point_chunks)
    fence_instances_after = 0
    fence_clusters_filtered_small = 0
    fence_clusters_filtered_low_height = 0
    fence_csf_instances_applied = 0
    fence_csf_instances_unavailable = 0
    fence_csf_all_ground_instances = 0
    fence_csf_removed_points_total = 0
    fence_effective_ground_mask = np.zeros((points_xyz.shape[0],), dtype=bool)
    fence_global_ground_mask = _csf_ground_mask(
        points_xyz,
        cloth_resolution=float(fence_csf_cloth_resolution),
        rigidness=int(fence_csf_rigidness),
        time_step=float(fence_csf_time_step),
        class_threshold=float(fence_csf_class_threshold),
        iterations=int(fence_csf_iterations),
        slope_smooth=bool(fence_csf_slope_smooth),
    )
    if fence_global_ground_mask is None or fence_global_ground_mask.shape[0] != points_xyz.shape[0]:
        fence_global_ground_mask = None
        fence_csf_instances_unavailable = 1
    else:
        fence_csf_instances_applied = 1

    if fence_point_chunks:
        fence_point_chunks_after_csf: list[np.ndarray] = []
        for chunk in fence_point_chunks:
            refined_chunk = chunk.astype(np.int32, copy=False)
            if fence_global_ground_mask is not None and refined_chunk.size > 0:
                local_z = points_xyz[refined_chunk, 2]
                q = min(1.0, max(0.0, float(ground_quantile)))
                low_ratio = min(1.0, max(0.0, float(fence_csf_low_band_ratio)))
                z_low = float(np.quantile(local_z, q))
                z_high = float(np.quantile(local_z, 0.95))
                span = max(0.0, z_high - z_low)
                z_cut = z_low + low_ratio * span
                local_low_mask = local_z <= z_cut

                global_ground_local = fence_global_ground_mask[refined_chunk.astype(np.int64, copy=False)]
                remove_mask = global_ground_local & local_low_mask
                if np.any(remove_mask):
                    fence_effective_ground_mask[refined_chunk[remove_mask].astype(np.int64, copy=False)] = True

                keep_mask = ~remove_mask
                kept = int(np.count_nonzero(keep_mask))
                if kept > 0:
                    fence_csf_removed_points_total += int(refined_chunk.size - kept)
                    refined_chunk = refined_chunk[keep_mask].astype(np.int32, copy=False)
                else:
                    # Guardrail: keep original chunk when CSF marks all points as ground.
                    fence_csf_all_ground_instances += 1
            if refined_chunk.size > 0:
                fence_point_chunks_after_csf.append(refined_chunk.astype(np.int32, copy=False))

        if not fence_point_chunks_after_csf:
            fence_point_chunks_after_csf = fence_point_chunks

        fence_points = np.unique(np.concatenate(fence_point_chunks_after_csf, axis=0)).astype(np.int32, copy=False)
        fence_clusters = _cluster_point_indices(
            points_xyz,
            fence_points,
            eps=float(fence_recluster_eps),
            use_xy_only=True,
            min_samples=max(1, int(fence_dbscan_min_samples)),
        )
        min_fence_points = max(1, int(fence_min_cluster_points))
        size_kept_clusters = [cluster for cluster in fence_clusters if int(cluster.size) >= min_fence_points]
        fence_clusters_filtered_small = int(len(fence_clusters) - len(size_kept_clusters))
        min_fence_height = max(0.0, float(fence_min_height))
        kept_fence_clusters = [
            cluster for cluster in size_kept_clusters
            if _robust_height_range(points_xyz[cluster]) >= min_fence_height
        ]
        fence_clusters_filtered_low_height = int(len(size_kept_clusters) - len(kept_fence_clusters))
        fence_instances_after = len(kept_fence_clusters)

        fence_scores = np.zeros((NUM_CLASSES + 1,), dtype=np.float32)
        for scores in fence_score_chunks:
            fence_scores += scores
        if not np.any(fence_scores):
            fence_scores[15] = 1.0
        non_bg_sum = float(fence_scores[1:].sum())
        fence_conf = float(fence_scores[15] / non_bg_sum) if non_bg_sum > 0 else 1.0
        if fence_confidences:
            fence_conf = max(fence_conf, float(np.mean(fence_confidences)))

        for cluster in kept_fence_clusters:
            refined_instances.append(
                SceneInstance(
                    class_scores=fence_scores.copy(),
                    class_id=15,
                    class_confidence=fence_conf,
                    point_indices=cluster.astype(np.int32, copy=False),
                )
            )

    for inst_id, inst in enumerate(refined_instances):
        refined_point_instance_id[inst.point_indices] = int(inst_id)

    stats = {
        "ground_removed_points_total": int(ground_removed_total),
        "fence_instances_before": int(fence_instances_before),
        "fence_instances_after": int(fence_instances_after),
        "fence_clusters_filtered_small": int(fence_clusters_filtered_small),
        "fence_clusters_filtered_low_height": int(fence_clusters_filtered_low_height),
        "fence_csf_instances_applied": int(fence_csf_instances_applied),
        "fence_csf_instances_unavailable": int(fence_csf_instances_unavailable),
        "fence_csf_all_ground_instances": int(fence_csf_all_ground_instances),
        "fence_csf_removed_points_total": int(fence_csf_removed_points_total),
        "fence_global_csf_applied": int(fence_csf_instances_applied),
        "fence_global_csf_unavailable": int(fence_csf_instances_unavailable),
        "fence_global_ground_points_total": int(np.count_nonzero(fence_global_ground_mask)) if fence_global_ground_mask is not None else 0,
        "fence_effective_ground_points_total": int(np.count_nonzero(fence_effective_ground_mask)),
        "fence_csf_low_band_ratio": float(fence_csf_low_band_ratio),
    }
    if return_fence_global_ground_mask:
        return refined_instances, refined_point_instance_id, stats, fence_effective_ground_mask
    return refined_instances, refined_point_instance_id, stats

def assign_points(
    candidates: list[CandidateInstance],
    candidate_to_instance: np.ndarray,
    *,
    num_points: int,
    num_instances: int,
) -> tuple[np.ndarray, np.ndarray]:
    point_instance_id = np.full((num_points,), -1, dtype=np.int32)
    point_confidence = np.zeros((num_points,), dtype=np.float32)
    if num_instances <= 0:
        return point_instance_id, point_confidence

    point_chunks: list[np.ndarray] = []
    inst_chunks: list[np.ndarray] = []
    score_chunks: list[np.ndarray] = []
    for cand_idx, cand in enumerate(candidates):
        inst_id = int(candidate_to_instance[cand_idx])
        if inst_id < 0 or cand.point_indices.size == 0:
            continue
        score = float(cand.weighted_score)
        if score <= 0:
            continue
        n = cand.point_indices.size
        point_chunks.append(cand.point_indices.astype(np.int64, copy=False))
        inst_chunks.append(np.full((n,), inst_id, dtype=np.int64))
        score_chunks.append(np.full((n,), score, dtype=np.float32))

    if not point_chunks:
        return point_instance_id, point_confidence

    point_arr = np.concatenate(point_chunks, axis=0)
    inst_arr = np.concatenate(inst_chunks, axis=0)
    score_arr = np.concatenate(score_chunks, axis=0)

    pair_key = point_arr * int(num_instances) + inst_arr
    unique_key, inverse = np.unique(pair_key, return_inverse=True)
    pair_scores = np.bincount(inverse, weights=score_arr).astype(np.float32, copy=False)
    point_ids = (unique_key // int(num_instances)).astype(np.int64, copy=False)
    inst_ids = (unique_key % int(num_instances)).astype(np.int32, copy=False)

    order = np.argsort(point_ids, kind="mergesort")
    point_ids = point_ids[order]
    inst_ids = inst_ids[order]
    pair_scores = pair_scores[order]

    i = 0
    while i < point_ids.size:
        j = i + 1
        pid = int(point_ids[i])
        while j < point_ids.size and point_ids[j] == pid:
            j += 1
        seg_scores = pair_scores[i:j]
        seg_insts = inst_ids[i:j]

        best_pos = int(np.argmax(seg_scores))
        best_score = float(seg_scores[best_pos])
        total_score = float(seg_scores.sum())
        point_instance_id[pid] = int(seg_insts[best_pos])
        point_confidence[pid] = float(best_score / total_score) if total_score > 0 else 0.0
        i = j

    return point_instance_id, point_confidence

def largest_cluster_mask(
    points_xyz: np.ndarray,
    eps: float,
    min_points: int,
    *,
    dbscan_min_samples: int = 1,
) -> np.ndarray:
    n = int(points_xyz.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=bool)

    point_indices = np.arange(n, dtype=np.int32)
    clusters = _cluster_point_indices(
        points_xyz,
        point_indices,
        eps=float(eps),
        use_xy_only=False,
        min_samples=dbscan_min_samples,
    )
    keep_mask = np.zeros((n,), dtype=bool)
    if not clusters:
        return keep_mask
    best_any = clusters[0]
    if best_any.size >= max(1, int(min_points)):
        keep_mask[best_any] = True
    return keep_mask

def denoise_assignments(
    points_xyz: np.ndarray,
    point_instance_id: np.ndarray,
    point_confidence: np.ndarray,
    *,
    num_instances: int,
    eps: float,
    min_points: int,
    dbscan_min_samples: int,
    skip_instance_ids: np.ndarray | list[int] | set[int] | None = None,
) -> dict[str, Any]:
    denoise_logs: list[dict[str, int]] = []
    removed_total = 0
    skip_ids: set[int] = set()
    if skip_instance_ids is not None:
        skip_arr = np.asarray(list(skip_instance_ids) if isinstance(skip_instance_ids, set) else skip_instance_ids, dtype=np.int32).reshape(-1)
        skip_ids = {int(x) for x in skip_arr}
    for inst_id in range(num_instances):
        point_idx = np.where(point_instance_id == inst_id)[0]
        if point_idx.size == 0:
            continue
        if inst_id in skip_ids:
            denoise_logs.append({"instance_id": int(inst_id), "before": int(point_idx.size), "removed": 0})
            continue
        keep_mask = largest_cluster_mask(
            points_xyz[point_idx],
            eps=eps,
            min_points=min_points,
            dbscan_min_samples=dbscan_min_samples,
        )
        if keep_mask.all():
            denoise_logs.append({"instance_id": int(inst_id), "before": int(point_idx.size), "removed": 0})
            continue

        remove_idx = point_idx[~keep_mask]
        point_instance_id[remove_idx] = -1
        point_confidence[remove_idx] = 0.0
        removed = int(remove_idx.size)
        removed_total += removed
        denoise_logs.append({"instance_id": int(inst_id), "before": int(point_idx.size), "removed": removed})

    return {"removed_points_total": removed_total, "instance_logs": denoise_logs}

def rebuild_instances(
    instances: list[SceneInstance],
    point_instance_id: np.ndarray,
) -> list[SceneInstance]:
    if not instances:
        return []

    old_to_new = np.full((len(instances),), -1, dtype=np.int32)
    new_instances: list[SceneInstance] = []
    for old_inst_id, inst in enumerate(instances):
        pts = np.where(point_instance_id == old_inst_id)[0].astype(np.int32, copy=False)
        if pts.size == 0:
            continue
        new_inst_id = len(new_instances)
        old_to_new[old_inst_id] = new_inst_id
        new_instances.append(
            SceneInstance(
                class_scores=inst.class_scores,
                class_id=inst.class_id,
                class_confidence=inst.class_confidence,
                point_indices=pts,
            )
        )

    valid = point_instance_id >= 0
    point_instance_id[valid] = old_to_new[point_instance_id[valid]]
    return new_instances

def _instance_colors(num_instances: int, seed: int) -> np.ndarray:
    if num_instances <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    colors = rng.integers(low=0, high=256, size=(num_instances, 3), dtype=np.uint8)
    zero_rows = np.where(colors.sum(axis=1) == 0)[0]
    if zero_rows.size > 0:
        colors[zero_rows] = np.array([255, 255, 255], dtype=np.uint8)
    return colors

def _point_class_ids_from_instances(instances: list[SceneInstance], point_instance_id: np.ndarray) -> np.ndarray:
    point_class_id = np.zeros(point_instance_id.shape, dtype=np.uint8)
    if point_instance_id.size == 0 or not instances:
        return point_class_id

    inst_class_ids = np.asarray([inst.class_id for inst in instances], dtype=np.uint8)
    valid = (point_instance_id >= 0) & (point_instance_id < inst_class_ids.shape[0])
    if np.any(valid):
        point_class_id[valid] = inst_class_ids[point_instance_id[valid]]
    return point_class_id

def write_scene_las(
    output_path: Path,
    las_in: laspy.LasData,
    points_xyz: np.ndarray,
    point_instance_id: np.ndarray,
    *,
    instances: list[SceneInstance],
    random_seed: int,
) -> int:
    selected = np.where(point_instance_id >= 0)[0]
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array(las_in.header.scales, copy=True)
    header.offsets = np.array(las_in.header.offsets, copy=True)
    las_out = laspy.LasData(header)

    if selected.size > 0:
        xyz = points_xyz[selected]
        las_out.x = xyz[:, 0]
        las_out.y = xyz[:, 1]
        las_out.z = xyz[:, 2]

        inst_ids = point_instance_id[selected]
        num_instances = int(inst_ids.max()) + 1
        color_table = _instance_colors(num_instances, seed=random_seed)
        rgb8 = color_table[inst_ids]
        rgb16 = rgb8.astype(np.uint16) * 256
        las_out.red = rgb16[:, 0]
        las_out.green = rgb16[:, 1]
        las_out.blue = rgb16[:, 2]

        class_ids = _point_class_ids_from_instances(instances, point_instance_id)[selected].astype(np.uint8, copy=False)
        las_out.classification = class_ids

        extra_bytes_params = getattr(laspy, "ExtraBytesParams", None)
        add_extra_dim = getattr(las_out, "add_extra_dim", None)
        if extra_bytes_params is not None and callable(add_extra_dim):
            las_out.add_extra_dim(extra_bytes_params(name="cls_id", type=np.uint8))
            las_out.cls_id = class_ids

    las_out.write(output_path)
    return int(selected.size)


def write_point_subset_las(
    output_path: Path,
    las_in: laspy.LasData,
    points_xyz: np.ndarray,
    point_indices: np.ndarray,
    *,
    classification: int = 2,
    rgb8: tuple[int, int, int] | None = None,
) -> int:
    selected = np.unique(np.asarray(point_indices, dtype=np.int32).reshape(-1))
    if selected.size > 0:
        valid = (selected >= 0) & (selected < int(points_xyz.shape[0]))
        selected = selected[valid]

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array(las_in.header.scales, copy=True)
    header.offsets = np.array(las_in.header.offsets, copy=True)
    las_out = laspy.LasData(header)

    if selected.size > 0:
        xyz = points_xyz[selected]
        las_out.x = xyz[:, 0]
        las_out.y = xyz[:, 1]
        las_out.z = xyz[:, 2]

        if rgb8 is not None:
            rgb_arr = np.asarray(rgb8, dtype=np.uint8).reshape(3)
            rgb16 = (rgb_arr.astype(np.uint16) * 256).reshape(1, 3)
            repeated = np.repeat(rgb16, selected.size, axis=0)
            las_out.red = repeated[:, 0]
            las_out.green = repeated[:, 1]
            las_out.blue = repeated[:, 2]

        cls = np.full((selected.size,), int(classification), dtype=np.uint8)
        las_out.classification = cls
        extra_bytes_params = getattr(laspy, "ExtraBytesParams", None)
        add_extra_dim = getattr(las_out, "add_extra_dim", None)
        if extra_bytes_params is not None and callable(add_extra_dim):
            las_out.add_extra_dim(extra_bytes_params(name="cls_id", type=np.uint8))
            las_out.cls_id = cls

    las_out.write(output_path)
    return int(selected.size)

def save_scene_npz(
    output_path: Path,
    scene_name: str,
    instances: list[SceneInstance],
    point_instance_id: np.ndarray,
    point_confidence: np.ndarray,
) -> None:
    instance_count = len(instances)
    inst_ids = np.arange(instance_count, dtype=np.int32)
    inst_class_ids = np.asarray([inst.class_id for inst in instances], dtype=np.int32)
    inst_class_names = np.asarray([CLASS_ID_TO_NAME.get(int(cid), "") for cid in inst_class_ids], dtype="<U16")
    inst_conf = np.asarray([inst.class_confidence for inst in instances], dtype=np.float32)
    inst_points = np.empty((instance_count,), dtype=object)
    for i, inst in enumerate(instances):
        inst_points[i] = inst.point_indices.astype(np.int32, copy=False)

    if instance_count > 0:
        class_scores = np.stack([inst.class_scores[1:] for inst in instances], axis=0).astype(np.float32, copy=False)
    else:
        class_scores = np.zeros((0, NUM_CLASSES), dtype=np.float32)

    np.savez_compressed(
        output_path,
        scene_name=np.array(scene_name),
        scene_instance_id=inst_ids,
        class_id=inst_class_ids,
        class_name=inst_class_names,
        confidence=inst_conf,
        class_scores=class_scores,
        point_indices=inst_points,
        point_instance_id=point_instance_id.astype(np.int32, copy=False),
        point_confidence=point_confidence.astype(np.float32, copy=False),
    )



def _build_tree_trunk_instances_from_stage(stage_clusters: list[dict[str, Any]]) -> list[SceneInstance]:
    out: list[SceneInstance] = []
    for cluster in stage_clusters:
        point_indices = np.unique(np.asarray(cluster.get('point_indices', np.zeros((0,), dtype=np.int32)), dtype=np.int32))
        if point_indices.size == 0:
            continue
        class_scores = np.zeros((NUM_CLASSES + 1,), dtype=np.float32)
        class_scores[TREE_CLASS_ID] = 1.0
        out.append(
            SceneInstance(
                class_scores=class_scores,
                class_id=TREE_CLASS_ID,
                class_confidence=1.0,
                point_indices=point_indices,
            )
        )
    return out


def write_tree_trunk_stage_las(
    output_path: Path,
    las_in: laspy.LasData,
    points_xyz: np.ndarray,
    *,
    stage_clusters: list[dict[str, Any]],
    random_seed: int,
) -> int:
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array(las_in.header.scales, copy=True)
    header.offsets = np.array(las_in.header.offsets, copy=True)
    las_out = laspy.LasData(header)

    extra_bytes_params = getattr(laspy, "ExtraBytesParams", None)
    add_extra_dim = getattr(las_out, "add_extra_dim", None)
    if extra_bytes_params is not None and callable(add_extra_dim):
        las_out.add_extra_dim(extra_bytes_params(name="cls_id", type=np.uint8))
        las_out.add_extra_dim(extra_bytes_params(name="trunk_height", type=np.float32))
        las_out.add_extra_dim(extra_bytes_params(name="trunk_radius", type=np.float32))
        las_out.add_extra_dim(extra_bytes_params(name="trunk_dir_x", type=np.float32))
        las_out.add_extra_dim(extra_bytes_params(name="trunk_dir_y", type=np.float32))
        las_out.add_extra_dim(extra_bytes_params(name="trunk_dir_z", type=np.float32))
        las_out.add_extra_dim(extra_bytes_params(name="trunk_dot_z", type=np.float32))

    if not stage_clusters:
        las_out.write(output_path)
        return 0

    num_points = int(points_xyz.shape[0])
    point_instance_id = np.full((num_points,), -1, dtype=np.int32)
    point_height = np.zeros((num_points,), dtype=np.float32)
    point_radius = np.zeros((num_points,), dtype=np.float32)
    point_dir_x = np.zeros((num_points,), dtype=np.float32)
    point_dir_y = np.zeros((num_points,), dtype=np.float32)
    point_dir_z = np.zeros((num_points,), dtype=np.float32)
    point_dot_z = np.zeros((num_points,), dtype=np.float32)

    for inst_id, cluster in enumerate(stage_clusters):
        pts = np.unique(np.asarray(cluster.get('point_indices', np.zeros((0,), dtype=np.int32)), dtype=np.int32))
        if pts.size == 0:
            continue
        point_instance_id[pts.astype(np.int64, copy=False)] = int(inst_id)
        h = float(cluster.get('robust_height', 0.0))
        r = float(cluster.get('robust_radius', 0.0))
        d = np.asarray(cluster.get('principal_direction', np.asarray([0.0, 0.0, 1.0], dtype=np.float32)), dtype=np.float32)
        if d.shape[0] != 3:
            d = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        dot_z = float(cluster.get('dot_z', d[2]))
        point_height[pts] = h
        point_radius[pts] = r
        point_dir_x[pts] = float(d[0])
        point_dir_y[pts] = float(d[1])
        point_dir_z[pts] = float(d[2])
        point_dot_z[pts] = dot_z

    selected = np.where(point_instance_id >= 0)[0]
    if selected.size > 0:
        xyz = points_xyz[selected]
        las_out.x = xyz[:, 0]
        las_out.y = xyz[:, 1]
        las_out.z = xyz[:, 2]

        inst_ids = point_instance_id[selected]
        num_instances = int(inst_ids.max()) + 1
        color_table = _instance_colors(num_instances, seed=random_seed)
        rgb8 = color_table[inst_ids]
        rgb16 = rgb8.astype(np.uint16) * 256
        las_out.red = rgb16[:, 0]
        las_out.green = rgb16[:, 1]
        las_out.blue = rgb16[:, 2]

        class_ids = np.full((selected.size,), TREE_CLASS_ID, dtype=np.uint8)
        las_out.classification = class_ids

        if extra_bytes_params is not None and callable(add_extra_dim):
            las_out.cls_id = class_ids
            las_out.trunk_height = point_height[selected]
            las_out.trunk_radius = point_radius[selected]
            las_out.trunk_dir_x = point_dir_x[selected]
            las_out.trunk_dir_y = point_dir_y[selected]
            las_out.trunk_dir_z = point_dir_z[selected]
            las_out.trunk_dot_z = point_dot_z[selected]

    las_out.write(output_path)
    return int(selected.size)
def _refine_tree_instances(
    points_xyz: np.ndarray,
    instances: list[SceneInstance],
    *,
    tree_ground_z: float | None,
    tree_station_xy: np.ndarray | None = None,
    tree_station_ground_z: np.ndarray | None = None,
    trunk_band_min: float,
    trunk_band_max: float,
    trunk_dbscan_eps: float,
    trunk_dbscan_min_samples: int,
    trunk_min_points: int,
    trunk_min_height: float,
    trunk_max_radius: float,
    trunk_min_verticality: float,
    crown_attach_distance: float,
    trunk_max_residual: float = 0.08,
    trunk_height_band_min: float | None = None,
    trunk_height_band_max: float | None = None,
    tree_final_denoise_eps: float = 0.50,
    return_trunk_stage_clusters: bool = False,
) -> tuple[list[SceneInstance], np.ndarray, dict[str, int]] | tuple[list[SceneInstance], np.ndarray, dict[str, int], dict[str, list[dict[str, Any]]]]:
    copied_instances = [_copy_scene_instance(inst) for inst in instances]
    empty_stats = {
        'tree_instances_before': 0,
        'tree_instances_after': 0,
        'trunk_candidate_groups': 0,
        'trunk_anchors': 0,
        'pending_crowns_before_attach': 0,
        'pending_crowns_attached': 0,
        'pending_crowns_kept': 0,
        'pending_crowns_dropped': 0,
        'tree_final_denoise_removed_points': 0,
        'tree_final_denoise_touched_instances': 0,
    }
    empty_stage = {'height': [], 'radius': [], 'verticality': []}
    if not copied_instances:
        empty_instances: list[SceneInstance] = []
        empty_point_instance_id = np.zeros((points_xyz.shape[0],), dtype=np.int32)
        if return_trunk_stage_clusters:
            return empty_instances, empty_point_instance_id, empty_stats, empty_stage
        return empty_instances, empty_point_instance_id, empty_stats

    non_tree_instances = [_copy_scene_instance(inst) for inst in copied_instances if inst.class_id != TREE_CLASS_ID]
    station_xy_arr: np.ndarray | None = None
    station_ground_arr: np.ndarray | None = None
    if tree_station_xy is not None and tree_station_ground_z is not None:
        try:
            xy_arr = np.asarray(tree_station_xy, dtype=np.float32).reshape(-1, 2)
            gz_arr = np.asarray(tree_station_ground_z, dtype=np.float32).reshape(-1)
            if xy_arr.shape[0] > 0 and xy_arr.shape[0] == gz_arr.shape[0]:
                finite_xy = np.all(np.isfinite(xy_arr), axis=1)
                finite_gz = np.isfinite(gz_arr)
                valid = finite_xy & finite_gz
                if np.any(valid):
                    station_xy_arr = xy_arr[valid].astype(np.float32, copy=False)
                    station_ground_arr = gz_arr[valid].astype(np.float32, copy=False)
        except Exception:
            station_xy_arr = None
            station_ground_arr = None
    has_station_ground_refs = (
        station_xy_arr is not None
        and station_ground_arr is not None
        and station_xy_arr.shape[0] > 0
        and station_ground_arr.shape[0] == station_xy_arr.shape[0]
    )
    has_any_tree_ground_ref = bool(tree_ground_z is not None) or has_station_ground_refs

    radius_band_low = min(float(trunk_band_min), float(trunk_band_max))
    radius_band_high = max(float(trunk_band_min), float(trunk_band_max))
    height_band_min = radius_band_low if trunk_height_band_min is None else float(trunk_height_band_min)
    height_band_max = radius_band_high if trunk_height_band_max is None else float(trunk_height_band_max)
    height_band_low = min(height_band_min, height_band_max)
    height_band_high = max(height_band_min, height_band_max)
    tree_infos: list[dict[str, Any]] = []
    for inst in copied_instances:
        if inst.class_id != TREE_CLASS_ID:
            continue
        point_indices = np.unique(np.asarray(inst.point_indices, dtype=np.int32))
        radius_band_points = np.zeros((0,), dtype=np.int32)
        height_band_points = np.zeros((0,), dtype=np.int32)
        local_tree_ground_z = tree_ground_z
        if has_station_ground_refs and point_indices.size > 0:
            tree_xy_center = points_xyz[point_indices, :2].mean(axis=0)
            dists = np.linalg.norm(station_xy_arr - tree_xy_center[None, :], axis=1)
            nearest_idx = int(np.argmin(dists))
            nearest_ground_z = float(station_ground_arr[nearest_idx])
            if np.isfinite(nearest_ground_z):
                local_tree_ground_z = nearest_ground_z
        if local_tree_ground_z is not None and point_indices.size > 0:
            z = points_xyz[point_indices, 2]
            radius_mask = (z >= float(local_tree_ground_z) + radius_band_low) & (z <= float(local_tree_ground_z) + radius_band_high)
            height_mask = (z >= float(local_tree_ground_z) + height_band_low) & (z <= float(local_tree_ground_z) + height_band_high)
            radius_band_points = point_indices[radius_mask].astype(np.int32, copy=False)
            height_band_points = point_indices[height_mask].astype(np.int32, copy=False)
        tree_infos.append(
            {
                'instance': _copy_scene_instance(inst),
                'point_indices': point_indices,
                'radius_band_points': radius_band_points,
                'height_band_points': height_band_points,
            }
        )

    tree_instances_before = len(tree_infos)
    if (not has_any_tree_ground_ref) or tree_instances_before == 0:
        final_instances = non_tree_instances + [_copy_scene_instance(info['instance']) for info in tree_infos]
        stats = dict(empty_stats)
        stats['tree_instances_before'] = int(tree_instances_before)
        stats['tree_instances_after'] = int(tree_instances_before)
        final_point_instance_id = _build_point_instance_id_from_instances(points_xyz.shape[0], final_instances)
        if return_trunk_stage_clusters:
            return final_instances, final_point_instance_id, stats, empty_stage
        return final_instances, final_point_instance_id, stats

    band_chunks = [info['radius_band_points'] for info in tree_infos if info['radius_band_points'].size > 0]
    if not band_chunks:
        final_instances = non_tree_instances
        stats = dict(empty_stats)
        stats['tree_instances_before'] = int(tree_instances_before)
        stats['tree_instances_after'] = 0
        stats['pending_crowns_before_attach'] = int(tree_instances_before)
        stats['pending_crowns_dropped'] = int(tree_instances_before)
        final_point_instance_id = _build_point_instance_id_from_instances(points_xyz.shape[0], final_instances)
        if return_trunk_stage_clusters:
            return final_instances, final_point_instance_id, stats, empty_stage
        return final_instances, final_point_instance_id, stats

    all_band_points = np.unique(np.concatenate(band_chunks, axis=0)).astype(np.int32, copy=False)
    # Map each radius-band point to its source tree instance so height补点 can be limited
    # to the tree instances that actually contribute to the merged trunk candidate cluster.
    radius_point_to_tree_info: dict[int, int] = {}
    for info_idx, info in enumerate(tree_infos):
        radius_points = np.asarray(info['radius_band_points'], dtype=np.int32).reshape(-1)
        for point_idx in radius_points.astype(np.int64, copy=False):
            radius_point_to_tree_info[int(point_idx)] = int(info_idx)
    height_link_radius = 0.05

    candidate_clusters = _cluster_point_indices(
        points_xyz,
        all_band_points,
        eps=float(trunk_dbscan_eps),
        use_xy_only=True,
        min_samples=int(trunk_dbscan_min_samples),
    )
    trunk_candidate_groups = len(candidate_clusters)

    # kept for CLI compatibility; verticality threshold is not applied right now
    _ = trunk_min_verticality

    min_trunk_points = max(1, int(trunk_min_points))
    metrics_all: list[dict[str, Any]] = []
    for cluster in candidate_clusters:
        if cluster.size < min_trunk_points:
            continue
        cluster_xyz = points_xyz[cluster]
        height_metric_points = cluster
        contributing_tree_info_ids = sorted(
            {
                radius_point_to_tree_info[int(point_idx)]
                for point_idx in cluster.astype(np.int64, copy=False)
                if int(point_idx) in radius_point_to_tree_info
            }
        )
        if contributing_tree_info_ids:
            local_height_chunks = [
                np.asarray(tree_infos[info_idx]['height_band_points'], dtype=np.int32).reshape(-1)
                for info_idx in contributing_tree_info_ids
                if np.asarray(tree_infos[info_idx]['height_band_points']).size > 0
            ]
            if local_height_chunks:
                local_height_points = np.unique(np.concatenate(local_height_chunks, axis=0)).astype(np.int32, copy=False)
                if local_height_points.size > 0:
                    local_height_tree = cKDTree(points_xyz[local_height_points, :2])
                    neighbor_lists = local_height_tree.query_ball_point(cluster_xyz[:, :2], r=height_link_radius)
                    matched_lists = [np.asarray(neighbors, dtype=np.int32) for neighbors in neighbor_lists if len(neighbors) > 0]
                    if matched_lists:
                        matched_local_idx = np.unique(np.concatenate(matched_lists, axis=0)).astype(np.int64, copy=False)
                        matched_points = local_height_points[matched_local_idx]
                        if matched_points.size > 0:
                            height_metric_points = np.unique(matched_points.astype(np.int32, copy=False))
        height_points_xyz = points_xyz[height_metric_points]
        z_vals = np.asarray(height_points_xyz[:, 2], dtype=np.float32)
        robust_height = float(z_vals.max() - z_vals.min()) if z_vals.size > 0 else 0.0
        robust_radius, radius_residual = _taubin_radius_and_trimmed_residual(cluster_xyz, trim_ratio=0.05)
        principal_direction = _principal_direction(cluster_xyz)
        dot_z = float(principal_direction[2])
        verticality = float(abs(dot_z))
        metrics_all.append(
            {
                'point_indices': np.unique(cluster.astype(np.int32, copy=False)),
                'xy_center': cluster_xyz[:, :2].mean(axis=0).astype(np.float32, copy=False),
                'robust_height': float(robust_height),
                'height_metric_points': int(height_metric_points.size),
                'robust_radius': float(robust_radius),
                'radius_residual': float(radius_residual),
                'principal_direction': principal_direction.astype(np.float32, copy=False),
                'dot_z': float(dot_z),
                'verticality': float(verticality),
            }
        )

    height_clusters = [m for m in metrics_all if m['robust_height'] > float(trunk_min_height)]
    radius_clusters = [
        m
        for m in height_clusters
        if m['robust_radius'] < float(trunk_max_radius)
        and m['radius_residual'] <= float(trunk_max_residual)
    ]
    # Verticality gating is intentionally disabled for tree trunks.
    # vertical_clusters = [m for m in radius_clusters if m['verticality'] >= float(trunk_min_verticality)]
    vertical_clusters = radius_clusters

    stage_clusters = {
        'height': height_clusters,
        'radius': radius_clusters,
        'verticality': vertical_clusters,
    }

    anchors = vertical_clusters
    point_to_anchor: dict[int, int] = {}
    for anchor_id, anchor in enumerate(anchors):
        for point_idx in anchor['point_indices'].astype(np.int64, copy=False):
            point_to_anchor[int(point_idx)] = anchor_id

    anchor_point_chunks: list[list[np.ndarray]] = [[] for _ in anchors]
    anchor_score_sums = [np.zeros((NUM_CLASSES + 1,), dtype=np.float32) for _ in anchors]
    anchor_conf_sums = [0.0 for _ in anchors]
    anchor_weight_sums = [0.0 for _ in anchors]
    pending_infos: list[dict[str, Any]] = []

    for info in tree_infos:
        point_indices = info['point_indices']
        band_points = info['radius_band_points']
        if not anchors or band_points.size == 0:
            pending_infos.append(info)
            continue

        associated_anchor_ids = sorted({point_to_anchor[int(point_idx)] for point_idx in band_points if int(point_idx) in point_to_anchor})
        if len(associated_anchor_ids) == 0:
            pending_infos.append(info)
            continue

        if len(associated_anchor_ids) == 1:
            anchor_id = associated_anchor_ids[0]
            anchor_point_chunks[anchor_id].append(point_indices.astype(np.int32, copy=False))
            anchor_score_sums[anchor_id] += info['instance'].class_scores.astype(np.float32, copy=False)
            anchor_conf_sums[anchor_id] += float(info['instance'].class_confidence)
            anchor_weight_sums[anchor_id] += 1.0
            continue

        point_xy = points_xyz[point_indices, :2]
        anchor_centers = np.stack([anchors[anchor_id]['xy_center'] for anchor_id in associated_anchor_ids], axis=0)
        dist_mat = np.linalg.norm(point_xy[:, None, :] - anchor_centers[None, :, :], axis=2)
        nearest_local = np.argmin(dist_mat, axis=1)
        total_points = max(1, int(point_indices.size))
        for local_anchor_idx, anchor_id in enumerate(associated_anchor_ids):
            local_mask = nearest_local == local_anchor_idx
            if not np.any(local_mask):
                continue
            local_points = point_indices[local_mask].astype(np.int32, copy=False)
            weight = float(local_points.size) / float(total_points)
            anchor_point_chunks[anchor_id].append(local_points)
            anchor_score_sums[anchor_id] += info['instance'].class_scores.astype(np.float32, copy=False) * weight
            anchor_conf_sums[anchor_id] += float(info['instance'].class_confidence) * weight
            anchor_weight_sums[anchor_id] += weight

    pending_before_attach = len(pending_infos)
    pending_attached = 0
    pending_dropped = 0

    if anchors:
        anchor_centers_all = np.stack([anchor['xy_center'] for anchor in anchors], axis=0)
    else:
        anchor_centers_all = np.zeros((0, 2), dtype=np.float32)

    for info in pending_infos:
        point_indices = info['point_indices']
        if point_indices.size == 0 or anchor_centers_all.shape[0] == 0:
            pending_dropped += 1
            continue
        point_xy_center = points_xyz[point_indices, :2].mean(axis=0)
        dists = np.linalg.norm(anchor_centers_all - point_xy_center[None, :], axis=1)
        nearest_anchor_id = int(np.argmin(dists))
        if float(dists[nearest_anchor_id]) < float(crown_attach_distance):
            anchor_point_chunks[nearest_anchor_id].append(point_indices.astype(np.int32, copy=False))
            anchor_score_sums[nearest_anchor_id] += info['instance'].class_scores.astype(np.float32, copy=False)
            anchor_conf_sums[nearest_anchor_id] += float(info['instance'].class_confidence)
            anchor_weight_sums[nearest_anchor_id] += 1.0
            pending_attached += 1
        else:
            pending_dropped += 1

    tree_instances_out: list[SceneInstance] = []
    for anchor_id, point_chunks in enumerate(anchor_point_chunks):
        if not point_chunks:
            continue
        merged_points = np.unique(np.concatenate(point_chunks, axis=0)).astype(np.int32, copy=False)
        if merged_points.size == 0:
            continue
        weight_sum = anchor_weight_sums[anchor_id]
        class_confidence = float(anchor_conf_sums[anchor_id] / weight_sum) if weight_sum > 0 else 1.0
        class_scores = anchor_score_sums[anchor_id]
        if not np.any(class_scores):
            class_scores = np.zeros((NUM_CLASSES + 1,), dtype=np.float32)
            class_scores[TREE_CLASS_ID] = 1.0
        tree_instances_out.append(
            SceneInstance(
                class_scores=class_scores.astype(np.float32, copy=False),
                class_id=TREE_CLASS_ID,
                class_confidence=class_confidence,
                point_indices=merged_points,
            )
        )

    tree_instances_out_final: list[SceneInstance] = []
    tree_final_denoise_removed_points = 0
    tree_final_denoise_touched_instances = 0
    tree_final_eps = float(tree_final_denoise_eps)
    for inst in tree_instances_out:
        point_indices = np.unique(np.asarray(inst.point_indices, dtype=np.int32))
        if point_indices.size == 0:
            continue
        if tree_final_eps > 0:
            clusters = _cluster_point_indices(
                points_xyz,
                point_indices,
                eps=tree_final_eps,
                use_xy_only=False,
                min_samples=1,
            )
            keep_points = clusters[0] if clusters else point_indices
        else:
            keep_points = point_indices
        keep_points = np.unique(np.asarray(keep_points, dtype=np.int32))
        if keep_points.size == 0:
            continue
        removed = int(point_indices.size - keep_points.size)
        if removed > 0:
            tree_final_denoise_touched_instances += 1
            tree_final_denoise_removed_points += removed
        tree_instances_out_final.append(
            SceneInstance(
                class_scores=inst.class_scores,
                class_id=inst.class_id,
                class_confidence=inst.class_confidence,
                point_indices=keep_points,
            )
        )

    final_instances = non_tree_instances + tree_instances_out_final
    final_point_instance_id = _build_point_instance_id_from_instances(points_xyz.shape[0], final_instances)

    stats = {
        'tree_instances_before': int(tree_instances_before),
        'tree_instances_after': int(len(tree_instances_out_final)),
        'trunk_candidate_groups': int(trunk_candidate_groups),
        'trunk_anchors': int(len(anchors)),
        'pending_crowns_before_attach': int(pending_before_attach),
        'pending_crowns_attached': int(pending_attached),
        'pending_crowns_kept': 0,
        'pending_crowns_dropped': int(pending_dropped),
        'tree_final_denoise_removed_points': int(tree_final_denoise_removed_points),
        'tree_final_denoise_touched_instances': int(tree_final_denoise_touched_instances),
    }

    if return_trunk_stage_clusters:
        return final_instances, final_point_instance_id, stats, stage_clusters
    return final_instances, final_point_instance_id, stats
