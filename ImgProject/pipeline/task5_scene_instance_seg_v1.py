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
from tqdm import tqdm


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
POLE_LIKE_CLASS_IDS = frozenset({1, 2, 3, 4, 5, 6})
GROUND_FILTER_CLASS_IDS = frozenset({1, 2, 3, 4, 8, 9, 11, 12, 13, 14, 15})


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task567 merged: 2D->3D back-projection, IoU merge, point conflict assignment, denoise."
    )
    parser.add_argument("--data-root", required=True, help="Benchmark root containing scene folders.")
    parser.add_argument("--iou-threshold", type=float, default=0.30, help="Point-IoU threshold for merging candidates.")
    parser.add_argument(
        "--merge-xy-distance",
        type=float,
        default=0.50,
        help="Supplementary merge distance threshold on candidate XY centers for classes 1-6 and same-class instances in 8-15 after category gating.",
    )
    parser.add_argument("--fov-deg", type=float, default=90.0, help="Projection horizontal FOV in degrees.")
    parser.add_argument(
        "--min-mask-points",
        type=int,
        default=20,
        help="Drop a 2D instance if back-projected points are fewer than this value.",
    )
    parser.add_argument(
        "--backproject-depth-threshold",
        type=float,
        default=0.20,
        help="Keep back-projected points whose camera depth is within this many meters of the per-pixel front depth.",
    )
    parser.add_argument(
        "--min-merged-points",
        type=int,
        default=0,
        help="Drop a merged instance before point assignment if its merged point count is below this value.",
    )
    parser.add_argument("--denoise-eps", type=float, default=0.60, help="Spatial clustering radius (meters).")
    parser.add_argument(
        "--denoise-min-points",
        type=int,
        default=30,
        help="Minimum points required for the retained largest spatial cluster; otherwise drop the instance.",
    )
    parser.add_argument(
        "--ground-z-quantile",
        type=float,
        default=0.10,
        help="Quantile used to estimate per-instance ground height for non-tree classes.",
    )
    parser.add_argument(
        "--ground-support-height",
        type=float,
        default=0.50,
        help="Points above ground height plus this margin define the horizontal support bbox.",
    )
    parser.add_argument(
        "--ground-bbox-expand",
        type=float,
        default=0.02,
        help="Expand the support bbox by this many meters before removing outside points.",
    )
    parser.add_argument(
        "--ground-support-top-ratio",
        type=float,
        default=0.70,
        help="Use only the mid-lower support band up to this fraction of the robust object height when building the ground-removal support bbox.",
    )
    parser.add_argument(
        "--fence-recluster-eps",
        type=float,
        default=0.10,
        help="Radius used to re-cluster fence points after ground removal.",
    )
    parser.add_argument(
        "--fence-min-cluster-points",
        type=int,
        default=500,
        help="Keep only fence clusters with at least this many points after fence re-clustering.",
    )
    parser.add_argument(
        "--fence-min-height",
        type=float,
        default=0.50,
        help="Keep only fence clusters whose robust height range is at least this many meters.",
    )
    parser.add_argument(
        "--tree-trunk-band-min",
        type=float,
        default=0.80,
        help="Lower height above ground used for tree trunk candidate points.",
    )
    parser.add_argument(
        "--tree-trunk-band-max",
        type=float,
        default=1.40,
        help="Upper height above ground used for tree trunk candidate points.",
    )
    parser.add_argument(
        "--tree-trunk-merge-xy-distance",
        type=float,
        default=0.35,
        help="Merge tree trunk candidates from different instances when their band-point XY centroids are within this distance.",
    )
    parser.add_argument(
        "--tree-trunk-min-points",
        type=int,
        default=10,
        help="Minimum number of band points required for a tree trunk candidate.",
    )
    parser.add_argument(
        "--tree-trunk-min-height",
        type=float,
        default=0.30,
        help="Minimum robust height range required for an accepted tree trunk candidate.",
    )
    parser.add_argument(
        "--tree-trunk-max-radius",
        type=float,
        default=0.30,
        help="Maximum robust XY radius allowed for an accepted tree trunk candidate.",
    )
    parser.add_argument(
        "--tree-trunk-min-verticality",
        type=float,
        default=0.55,
        help="Minimum verticality score required for an accepted tree trunk candidate.",
    )
    parser.add_argument(
        "--tree-crown-attach-distance",
        type=float,
        default=4.0,
        help="Attach a trunkless tree crown to the nearest trunk when within this XY distance.",
    )
    parser.add_argument("--output-dir-name", type=str, default="fusion", help="Per-scene output folder name.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for instance colors in LAS.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing {scene}_instance_seg outputs.",
    )
    return parser.parse_args()


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
    return class_id_a in POLE_LIKE_CLASS_IDS and class_id_b in POLE_LIKE_CLASS_IDS


def _candidate_xy_center_distance(a: CandidateInstance, b: CandidateInstance) -> float:
    center_a = np.asarray(a.xy_center, dtype=np.float32) if a.xy_center is not None else 0.5 * (a.bbox_min[:2] + a.bbox_max[:2])
    center_b = np.asarray(b.xy_center, dtype=np.float32) if b.xy_center is not None else 0.5 * (b.bbox_min[:2] + b.bbox_max[:2])
    return float(np.linalg.norm(center_a - center_b))


def _supports_xy_supplementary_merge(class_id_a: int, class_id_b: int) -> bool:
    if class_id_a == 7 or class_id_b == 7:
        return False
    if class_id_a in POLE_LIKE_CLASS_IDS and class_id_b in POLE_LIKE_CLASS_IDS:
        return True
    return class_id_a == class_id_b and class_id_a in CLASS_ID_TO_NAME


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

    tree_instance_ids = {inst_id for inst_id, inst in enumerate(instances) if inst.class_id == TREE_CLASS_ID}
    if not tree_instance_ids:
        return 0
    fence_instance_ids = {inst_id for inst_id, inst in enumerate(instances) if inst.class_id == FENCE_CLASS_ID}

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
    z_ground = float(np.quantile(z, q))
    lower_height = z_ground + float(support_height)
    z_top = float(np.quantile(z, 0.95))
    span = max(0.0, z_top - z_ground)
    top_ratio = min(1.0, max(0.0, float(support_top_ratio)))
    upper_height = z_ground + top_ratio * span
    if upper_height <= lower_height:
        support_mask = z > lower_height
    else:
        support_mask = (z > lower_height) & (z <= upper_height)
        if not np.any(support_mask):
            support_mask = z > lower_height
    if not np.any(support_mask):
        return np.ones((n,), dtype=bool)

    support_xy = points_xyz[support_mask, :2]
    bbox_min = support_xy.min(axis=0) - float(bbox_expand)
    bbox_max = support_xy.max(axis=0) + float(bbox_expand)
    xy = points_xyz[:, :2]
    inside_bbox = np.all((xy >= bbox_min) & (xy <= bbox_max), axis=1)
    low_ground = points_xyz[:, 2] <= (z_ground + float(support_height))
    return inside_bbox | (~low_ground)



def _cluster_point_indices(
    points_xyz: np.ndarray,
    point_indices: np.ndarray,
    *,
    eps: float,
    use_xy_only: bool = False,
) -> list[np.ndarray]:
    point_indices = np.asarray(point_indices, dtype=np.int32).reshape(-1)
    if point_indices.size == 0:
        return []
    if eps <= 0:
        return [np.asarray([idx], dtype=np.int32) for idx in point_indices]

    pts = points_xyz[point_indices]
    if use_xy_only:
        pts = pts[:, :2]
    tree = cKDTree(pts)
    visited = np.zeros((point_indices.shape[0],), dtype=bool)
    clusters: list[np.ndarray] = []

    for start in range(point_indices.shape[0]):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        comp: list[int] = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in tree.query_ball_point(pts[cur], r=eps):
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        clusters.append(np.sort(point_indices[np.asarray(comp, dtype=np.int32)]))

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



def _robust_xy_radius(points_xyz: np.ndarray) -> float:
    if points_xyz.size == 0:
        return math.inf
    xy = np.asarray(points_xyz[:, :2], dtype=np.float32)
    center = xy.mean(axis=0)
    radii = np.linalg.norm(xy - center[None, :], axis=1)
    return float(np.quantile(radii, 0.90)) if radii.size > 0 else math.inf



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



def _refine_tree_instances(
    points_xyz: np.ndarray,
    instances: list[SceneInstance],
    *,
    tree_ground_z: float | None,
    trunk_band_min: float,
    trunk_band_max: float,
    trunk_merge_xy_distance: float,
    trunk_min_points: int,
    trunk_min_height: float,
    trunk_max_radius: float,
    trunk_min_verticality: float,
    crown_attach_distance: float,
) -> tuple[list[SceneInstance], np.ndarray, dict[str, int]]:
    copied_instances = [_copy_scene_instance(inst) for inst in instances]
    if not copied_instances:
        return [], np.zeros((points_xyz.shape[0],), dtype=np.int32), {
            'tree_instances_before': 0,
            'tree_instances_after': 0,
            'trunk_candidate_groups': 0,
            'trunk_anchor_count': 0,
            'pending_crowns_before_attach': 0,
            'pending_crowns_attached': 0,
            'pending_crowns_kept': 0,
        }

    tree_infos: list[dict[str, Any]] = []
    non_tree_instances = [_copy_scene_instance(inst) for inst in copied_instances if inst.class_id != TREE_CLASS_ID]
    for inst in copied_instances:
        if inst.class_id != TREE_CLASS_ID:
            continue
        point_indices = np.unique(np.asarray(inst.point_indices, dtype=np.int32))
        band_points = np.zeros((0,), dtype=np.int32)
        band_xy_center = None
        if tree_ground_z is not None and point_indices.size > 0:
            z = points_xyz[point_indices, 2]
            band_mask = (z >= float(tree_ground_z) + float(trunk_band_min)) & (z <= float(tree_ground_z) + float(trunk_band_max))
            band_points = point_indices[band_mask].astype(np.int32, copy=False)
            if band_points.size > 0:
                band_xy_center = points_xyz[band_points, :2].mean(axis=0).astype(np.float32, copy=False)
        tree_infos.append(
            {
                'instance': _copy_scene_instance(inst),
                'point_indices': point_indices,
                'band_points': band_points,
                'band_xy_center': band_xy_center,
            }
        )

    tree_instances_before = len(tree_infos)
    if tree_ground_z is None or tree_instances_before == 0:
        final_instances = non_tree_instances + [_copy_scene_instance(info['instance']) for info in tree_infos]
        return final_instances, _build_point_instance_id_from_instances(points_xyz.shape[0], final_instances), {
            'tree_instances_before': int(tree_instances_before),
            'tree_instances_after': int(tree_instances_before),
            'trunk_candidate_groups': 0,
            'trunk_anchor_count': 0,
            'pending_crowns_before_attach': 0,
            'pending_crowns_attached': 0,
            'pending_crowns_kept': 0,
        }

    min_trunk_points = max(1, int(trunk_min_points))
    candidate_infos: list[dict[str, Any]] = []
    for tree_idx, info in enumerate(tree_infos):
        band_points = info['band_points']
        band_xy_center = info['band_xy_center']
        if band_points.size == 0 or band_xy_center is None:
            continue
        candidate_infos.append(
            {
                'tree_idx': tree_idx,
                'band_points': band_points,
                'xy_center': np.asarray(band_xy_center, dtype=np.float32),
            }
        )

    anchors: list[dict[str, Any]] = []
    trunk_candidate_groups = 0
    if candidate_infos:
        uf = UnionFind(len(candidate_infos))
        merge_dist = max(0.0, float(trunk_merge_xy_distance))
        if merge_dist > 0:
            for i in range(len(candidate_infos)):
                for j in range(i + 1, len(candidate_infos)):
                    if float(np.linalg.norm(candidate_infos[i]['xy_center'] - candidate_infos[j]['xy_center'])) <= merge_dist:
                        uf.union(i, j)

        groups: dict[int, list[int]] = {}
        for idx in range(len(candidate_infos)):
            groups.setdefault(uf.find(idx), []).append(idx)
        trunk_candidate_groups = len(groups)

        for group_indices in groups.values():
            group_band_points = np.unique(
                np.concatenate([candidate_infos[idx]['band_points'] for idx in group_indices], axis=0)
            ).astype(np.int32, copy=False)
            if group_band_points.size < min_trunk_points:
                continue
            group_xyz = points_xyz[group_band_points]
            robust_height = _robust_height_range(group_xyz)
            robust_radius = _robust_xy_radius(group_xyz)
            verticality = _principal_verticality(group_xyz)
            if robust_height <= float(trunk_min_height):
                continue
            if robust_radius >= float(trunk_max_radius):
                continue
            if verticality < float(trunk_min_verticality):
                continue
            anchors.append(
                {
                    'xy_center': group_xyz[:, :2].mean(axis=0).astype(np.float32, copy=False),
                }
            )

    anchor_point_chunks: list[list[np.ndarray]] = [[] for _ in anchors]
    anchor_score_sums = [np.zeros((NUM_CLASSES + 1,), dtype=np.float32) for _ in anchors]
    anchor_conf_sums = [0.0 for _ in anchors]
    anchor_weight_sums = [0.0 for _ in anchors]
    pending_infos: list[dict[str, Any]] = []

    for info in tree_infos:
        point_indices = info['point_indices']
        band_points = info['band_points']
        if not anchors or band_points.size == 0:
            pending_infos.append(info)
            continue

        band_xy = points_xyz[band_points, :2]
        associated_anchor_ids: list[int] = []
        for anchor_id, anchor in enumerate(anchors):
            dist = np.linalg.norm(band_xy - anchor['xy_center'][None, :], axis=1)
            if dist.size > 0 and float(dist.min()) <= float(trunk_max_radius):
                associated_anchor_ids.append(anchor_id)

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
    pending_kept_instances: list[SceneInstance] = []
    if anchors:
        anchor_centers_all = np.stack([anchor['xy_center'] for anchor in anchors], axis=0)
    else:
        anchor_centers_all = np.zeros((0, 2), dtype=np.float32)

    for info in pending_infos:
        point_indices = info['point_indices']
        if point_indices.size == 0 or anchor_centers_all.shape[0] == 0:
            pending_kept_instances.append(_copy_scene_instance(info['instance']))
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
            pending_kept_instances.append(_copy_scene_instance(info['instance']))

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

    final_instances = non_tree_instances + tree_instances_out + pending_kept_instances
    final_point_instance_id = _build_point_instance_id_from_instances(points_xyz.shape[0], final_instances)
    stats = {
        'tree_instances_before': int(tree_instances_before),
        'tree_instances_after': int(len(tree_instances_out) + len(pending_kept_instances)),
        'trunk_candidate_groups': int(trunk_candidate_groups),
        'trunk_anchor_count': int(len(anchors)),
        'pending_crowns_before_attach': int(pending_before_attach),
        'pending_crowns_attached': int(pending_attached),
        'pending_crowns_kept': int(len(pending_kept_instances)),
    }
    return final_instances, final_point_instance_id, stats



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
) -> tuple[list[SceneInstance], np.ndarray, dict[str, int]]:
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

        refined_pts = pts
        if inst.class_id in GROUND_FILTER_CLASS_IDS:
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

        if inst.class_id == 15:
            fence_point_chunks.append(refined_pts)
            fence_score_chunks.append(inst.class_scores.astype(np.float32, copy=False))
            fence_confidences.append(float(inst.class_confidence))
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
    if fence_point_chunks:
        fence_points = np.unique(np.concatenate(fence_point_chunks, axis=0)).astype(np.int32, copy=False)
        fence_clusters = _cluster_point_indices(
            points_xyz,
            fence_points,
            eps=float(fence_recluster_eps),
            use_xy_only=True,
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
    }
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


def largest_cluster_mask(points_xyz: np.ndarray, eps: float, min_points: int) -> np.ndarray:
    n = int(points_xyz.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=bool)

    tree = cKDTree(points_xyz)
    visited = np.zeros((n,), dtype=bool)
    best_any = np.zeros((0,), dtype=np.int32)

    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        comp: list[int] = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            neighbors = tree.query_ball_point(points_xyz[cur], r=eps)
            for nb in neighbors:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        comp_arr = np.asarray(comp, dtype=np.int32)
        if comp_arr.size > best_any.size:
            best_any = comp_arr

    keep_mask = np.zeros((n,), dtype=bool)
    if best_any.size >= max(1, min_points):
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
) -> dict[str, Any]:
    denoise_logs: list[dict[str, int]] = []
    removed_total = 0
    for inst_id in range(num_instances):
        point_idx = np.where(point_instance_id == inst_id)[0]
        if point_idx.size == 0:
            continue
        keep_mask = largest_cluster_mask(points_xyz[point_idx], eps=eps, min_points=min_points)
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


def process_scene(scene_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    scene_name = scene_dir.name
    output_dir = scene_dir / args.output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_npz_path = output_dir / f"{scene_name}_instance_seg.npz"
    scene_las_path = output_dir / f"{scene_name}_instance_seg.las"
    scene_refined_las_path = output_dir / f"{scene_name}_instance_seg_refined.las"
    scene_final_npz_path = output_dir / f"{scene_name}_instance_seg_final.npz"
    scene_final_las_path = output_dir / f"{scene_name}_instance_seg_final.las"
    scene_meta_path = output_dir / f"{scene_name}_instance_seg_meta.json"

    if (
        (not args.overwrite)
        and scene_npz_path.exists()
        and scene_las_path.exists()
        and scene_refined_las_path.exists()
        and scene_final_npz_path.exists()
        and scene_final_las_path.exists()
    ):
        return {
            "scene": scene_name,
            "status": "skipped_exists",
            "scene_npz": str(scene_npz_path),
            "scene_las": str(scene_las_path),
            "scene_refined_las": str(scene_refined_las_path),
            "scene_final_npz": str(scene_final_npz_path),
            "scene_final_las": str(scene_final_las_path),
        }

    las_path = find_las_path(scene_dir)
    if las_path is None:
        return {"scene": scene_name, "status": "missing_las"}

    las_data = laspy.read(las_path)
    points_xyz = np.vstack([las_data.x, las_data.y, las_data.z]).T.astype(np.float32, copy=False)
    num_points = int(points_xyz.shape[0])

    candidates, candidate_stats = collect_scene_candidates(
        scene_dir=scene_dir,
        points_xyz=points_xyz,
        fov_deg=float(args.fov_deg),
        min_mask_points=int(args.min_mask_points),
        backproject_depth_threshold=float(args.backproject_depth_threshold),
    )
    candidate_point_sum, candidate_point_unique = _summarize_candidate_points(candidates)
    print(
        _format_scene_backprojection_log(
            scene_name,
            candidate_stats=candidate_stats,
            candidate_point_sum=candidate_point_sum,
            candidate_point_unique=candidate_point_unique,
        )
    )

    instances, candidate_to_instance = merge_candidates(
        candidates,
        iou_threshold=float(args.iou_threshold),
        merge_xy_distance=float(args.merge_xy_distance),
        min_merged_points=int(args.min_merged_points),
    )
    print(_format_iou_merge_log(scene_name, candidate_count=len(candidates), merged_count=len(instances)))
    tree_pruned_points = _prune_tree_points_before_assignment(candidates, candidate_to_instance, instances)
    print(f"[{scene_name}] tree overlap pruning: removed_points={tree_pruned_points}")
    point_instance_id, point_confidence = assign_points(
        candidates,
        candidate_to_instance,
        num_points=num_points,
        num_instances=len(instances),
    )

    assigned_points_before_denoise = int(np.count_nonzero(point_instance_id >= 0))
    print(f"[{scene_name}] point assignment: assigned_points={assigned_points_before_denoise}")

    denoise_log = denoise_assignments(
        points_xyz=points_xyz,
        point_instance_id=point_instance_id,
        point_confidence=point_confidence,
        num_instances=len(instances),
        eps=float(args.denoise_eps),
        min_points=int(args.denoise_min_points),
    )

    instances = rebuild_instances(instances, point_instance_id)
    assigned_points_after_denoise = int(np.count_nonzero(point_instance_id >= 0))
    print(
        f"[{scene_name}] denoise: removed_points={int(denoise_log['removed_points_total'])} "
        f"final_instances={len(instances)} final_points={assigned_points_after_denoise}"
    )
    selected_points = write_scene_las(
        output_path=scene_las_path,
        las_in=las_data,
        points_xyz=points_xyz,
        point_instance_id=point_instance_id,
        instances=instances,
        random_seed=int(args.random_seed),
    )
    save_scene_npz(
        output_path=scene_npz_path,
        scene_name=scene_name,
        instances=instances,
        point_instance_id=point_instance_id,
        point_confidence=point_confidence,
    )

    refined_instances, refined_point_instance_id, refine_stats = _refine_instances_with_ground_and_fence(
        points_xyz,
        instances,
        point_instance_id,
        ground_quantile=float(args.ground_z_quantile),
        support_height=float(args.ground_support_height),
        bbox_expand=float(args.ground_bbox_expand),
        support_top_ratio=float(getattr(args, 'ground_support_top_ratio', 0.70)),
        fence_recluster_eps=float(args.fence_recluster_eps),
        fence_min_cluster_points=int(args.fence_min_cluster_points),
        fence_min_height=float(args.fence_min_height),
    )
    selected_points_refined = write_scene_las(
        output_path=scene_refined_las_path,
        las_in=las_data,
        points_xyz=points_xyz,
        point_instance_id=refined_point_instance_id,
        instances=refined_instances,
        random_seed=int(args.random_seed),
    )
    print(
        f"[{scene_name}] refined postprocess: ground_removed_points={int(refine_stats['ground_removed_points_total'])} "
        f"fence_instances={int(refine_stats['fence_instances_before'])}->{int(refine_stats['fence_instances_after'])} "
        f"fence_clusters_filtered_small={int(refine_stats['fence_clusters_filtered_small'])} "
        f"fence_clusters_filtered_low_height={int(refine_stats['fence_clusters_filtered_low_height'])} "
        f"final_instances={len(refined_instances)} final_points={selected_points_refined}"
    )

    tree_ground_z = _tree_ground_z_from_effective_stations(scene_dir / 'projected_images')
    final_instances, final_point_instance_id, tree_stats = _refine_tree_instances(
        points_xyz,
        refined_instances,
        tree_ground_z=tree_ground_z,
        trunk_band_min=float(getattr(args, 'tree_trunk_band_min', 0.80)),
        trunk_band_max=float(getattr(args, 'tree_trunk_band_max', 1.40)),
        trunk_merge_xy_distance=float(getattr(args, 'tree_trunk_merge_xy_distance', 0.35)),
        trunk_min_points=int(getattr(args, 'tree_trunk_min_points', 10)),
        trunk_min_height=float(getattr(args, 'tree_trunk_min_height', 0.30)),
        trunk_max_radius=float(getattr(args, 'tree_trunk_max_radius', 0.30)),
        trunk_min_verticality=float(getattr(args, 'tree_trunk_min_verticality', 0.55)),
        crown_attach_distance=float(getattr(args, 'tree_crown_attach_distance', 4.0)),
    )
    selected_points_final = write_scene_las(
        output_path=scene_final_las_path,
        las_in=las_data,
        points_xyz=points_xyz,
        point_instance_id=final_point_instance_id,
        instances=final_instances,
        random_seed=int(args.random_seed),
    )
    final_point_confidence = np.where(final_point_instance_id >= 0, 1.0, 0.0).astype(np.float32, copy=False)
    save_scene_npz(
        output_path=scene_final_npz_path,
        scene_name=scene_name,
        instances=final_instances,
        point_instance_id=final_point_instance_id,
        point_confidence=final_point_confidence,
    )
    print(
        f"[{scene_name}] tree postprocess: ground_z={tree_ground_z if tree_ground_z is not None else 'none'} "
        f"trunk_candidate_groups={int(tree_stats['trunk_candidate_groups'])} "
        f"trunk_anchors={int(tree_stats['trunk_anchor_count'])} "
        f"pending_crowns={int(tree_stats['pending_crowns_before_attach'])} "
        f"attached={int(tree_stats['pending_crowns_attached'])} kept={int(tree_stats['pending_crowns_kept'])} "
        f"final_instances={len(final_instances)} final_points={selected_points_final}"
    )

    meta = {
        "scene": scene_name,
        "las_path": str(las_path),
        "num_points": num_points,
        "candidate_stats": candidate_stats,
        "merged_instance_count_before_denoise": int(candidate_to_instance.max() + 1) if candidate_to_instance.size > 0 else 0,
        "final_instance_count": len(instances),
        "selected_instance_points": selected_points,
        "selected_instance_points_refined": selected_points_refined,
        "selected_instance_points_final": selected_points_final,
        "denoise": denoise_log,
        "refined_postprocess": refine_stats,
        "refined_instance_count": len(refined_instances),
        "tree_postprocess": tree_stats,
        "final_tree_instance_count": len(final_instances),
        "scene_npz": str(scene_npz_path),
        "scene_las": str(scene_las_path),
        "scene_refined_las": str(scene_refined_las_path),
        "scene_final_npz": str(scene_final_npz_path),
        "scene_final_las": str(scene_final_las_path),
    }
    scene_meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"scene": scene_name, "status": "ok", **meta}


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.exists() or not data_root.is_dir():
        raise NotADirectoryError(f"Invalid data root: {data_root}")

    scene_dirs = discover_scene_dirs(data_root)
    if not scene_dirs:
        raise FileNotFoundError(f"No scene directories with projected_images under: {data_root}")

    for scene_dir in scene_dirs:
        result = process_scene(scene_dir, args)
        print(f"[{result.get('status', 'unknown')}] {scene_dir.name}")
    print("done.")


if __name__ == "__main__":
    main()
