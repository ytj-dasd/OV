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
}
CLASS_NAME_TO_ID = {name: cid for cid, name in CLASS_ID_TO_NAME.items()}
NUM_CLASSES = len(CLASS_ID_TO_NAME)


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
    parser.add_argument("--fov-deg", type=float, default=90.0, help="Projection horizontal FOV in degrees.")
    parser.add_argument(
        "--min-mask-points",
        type=int,
        default=20,
        help="Drop a 2D instance if back-projected points are fewer than this value.",
    )
    parser.add_argument("--denoise-eps", type=float, default=0.60, help="Spatial clustering radius (meters).")
    parser.add_argument(
        "--denoise-min-points",
        type=int,
        default=30,
        help="Minimum points for a valid spatial cluster when denoising each instance.",
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
) -> np.ndarray:
    if pts_img_indices.size == 0:
        return np.zeros((0,), dtype=np.int32)

    flat_mask = mask.reshape(-1)
    valid_img = (pts_img_indices >= 0) & (pts_img_indices < flat_mask.size)
    if not np.any(valid_img):
        return np.zeros((0,), dtype=np.int32)

    img_indices = pts_img_indices[valid_img]
    point_indices = pts_indices[valid_img]
    point_keep = flat_mask[img_indices]
    if not np.any(point_keep):
        return np.zeros((0,), dtype=np.int32)

    projected = point_indices[point_keep]
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


def collect_scene_candidates(
    scene_dir: Path,
    points_xyz: np.ndarray,
    *,
    fov_deg: float,
    min_mask_points: int,
) -> tuple[list[CandidateInstance], dict[str, int]]:
    projected_dir = scene_dir / "projected_images"
    sam_dir = scene_dir / "sam_mask"
    mapping_files = sorted(projected_dir.glob("station_*_cam_*.npz"))

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
            if class_id not in CLASS_ID_TO_NAME:
                stats["sam_instances_filtered_invalid_class"] += 1
                continue

            mask = masks[i]
            projected_points = _project_mask_to_points(mask, pts_img_indices, pts_indices, num_points=num_points)
            if projected_points.size < min_mask_points:
                stats["sam_instances_filtered_small"] += 1
                continue

            view_weight, _ = _view_weight_from_mask(mask, fov_deg=fov_deg)
            sam_score = float(scores[i])
            class_conf = float(class_confs[i])
            weighted_score = float(view_weight * sam_score * class_conf)

            xyz = points_xyz[projected_points]
            bbox_min = xyz.min(axis=0)
            bbox_max = xyz.max(axis=0)

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
                )
            )
            stats["sam_instances_kept"] += 1

    return candidates, stats


def merge_candidates(
    candidates: list[CandidateInstance],
    *,
    iou_threshold: float,
) -> tuple[list[SceneInstance], np.ndarray]:
    if not candidates:
        return [], np.zeros((0,), dtype=np.int32)

    n = len(candidates)
    uf = UnionFind(n)
    for i in range(n):
        ci = candidates[i]
        for j in range(i + 1, n):
            cj = candidates[j]
            if not _bboxes_overlap(ci.bbox_min, ci.bbox_max, cj.bbox_min, cj.bbox_max):
                continue
            iou = _point_iou(ci.point_indices, cj.point_indices)
            if iou >= iou_threshold:
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
    if n < max(1, min_points):
        return np.ones((n,), dtype=bool)

    tree = cKDTree(points_xyz)
    visited = np.zeros((n,), dtype=bool)
    best_any = np.zeros((0,), dtype=np.int32)
    best_valid = np.zeros((0,), dtype=np.int32)

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
        if comp_arr.size >= min_points and comp_arr.size > best_valid.size:
            best_valid = comp_arr

    keep_idx = best_valid if best_valid.size > 0 else best_any
    keep_mask = np.zeros((n,), dtype=bool)
    keep_mask[keep_idx] = True
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


def write_scene_las(
    output_path: Path,
    las_in: laspy.LasData,
    points_xyz: np.ndarray,
    point_instance_id: np.ndarray,
    *,
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
    scene_meta_path = output_dir / f"{scene_name}_instance_seg_meta.json"

    if (not args.overwrite) and scene_npz_path.exists() and scene_las_path.exists():
        return {"scene": scene_name, "status": "skipped_exists", "scene_npz": str(scene_npz_path), "scene_las": str(scene_las_path)}

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
    )

    instances, candidate_to_instance = merge_candidates(candidates, iou_threshold=float(args.iou_threshold))
    point_instance_id, point_confidence = assign_points(
        candidates,
        candidate_to_instance,
        num_points=num_points,
        num_instances=len(instances),
    )

    denoise_log = denoise_assignments(
        points_xyz=points_xyz,
        point_instance_id=point_instance_id,
        point_confidence=point_confidence,
        num_instances=len(instances),
        eps=float(args.denoise_eps),
        min_points=int(args.denoise_min_points),
    )

    instances = rebuild_instances(instances, point_instance_id)
    selected_points = write_scene_las(
        output_path=scene_las_path,
        las_in=las_data,
        points_xyz=points_xyz,
        point_instance_id=point_instance_id,
        random_seed=int(args.random_seed),
    )
    save_scene_npz(
        output_path=scene_npz_path,
        scene_name=scene_name,
        instances=instances,
        point_instance_id=point_instance_id,
        point_confidence=point_confidence,
    )

    meta = {
        "scene": scene_name,
        "las_path": str(las_path),
        "num_points": num_points,
        "candidate_stats": candidate_stats,
        "merged_instance_count_before_denoise": int(candidate_to_instance.max() + 1) if candidate_to_instance.size > 0 else 0,
        "final_instance_count": len(instances),
        "selected_instance_points": selected_points,
        "denoise": denoise_log,
        "scene_npz": str(scene_npz_path),
        "scene_las": str(scene_las_path),
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
