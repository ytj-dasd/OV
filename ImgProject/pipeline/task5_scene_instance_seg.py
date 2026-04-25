from __future__ import annotations

import argparse
import gc
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import laspy
import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import utils as task5_utils

for _name in dir(task5_utils):
    if not _name.startswith("__"):
        globals()[_name] = getattr(task5_utils, _name)

# Keep the main module patchable in tests while delegating implementation to utils.
task5_utils.laspy = laspy


def write_scene_las(
    *,
    output_path: Path,
    las_in: Any,
    points_xyz: np.ndarray,
    point_instance_id: np.ndarray,
    instances: list[SceneInstance],
    random_seed: int,
) -> int:
    task5_utils.laspy = laspy
    return task5_utils.write_scene_las(
        output_path=output_path,
        las_in=las_in,
        points_xyz=points_xyz,
        point_instance_id=point_instance_id,
        instances=instances,
        random_seed=random_seed,
    )


def write_pole_groups_las(
    output_path: Path,
    las_in: Any,
    points_xyz: np.ndarray,
    *,
    pole_groups: list[dict[str, Any]],
    random_seed: int,
) -> int:
    task5_utils.laspy = laspy
    return task5_utils.write_pole_groups_las(
        output_path=output_path,
        las_in=las_in,
        points_xyz=points_xyz,
        pole_groups=pole_groups,
        random_seed=random_seed,
    )


def write_compact_final_scene_las(
    *,
    output_path: Path,
    las_in: Any,
    points_xyz: np.ndarray,
    point_scene_instance_id: np.ndarray,
    point_pole_group_id: np.ndarray,
    scene_instance_class_id: np.ndarray,
    random_seed: int,
) -> int:
    task5_utils.laspy = laspy
    return task5_utils.write_compact_final_scene_las(
        output_path=output_path,
        las_in=las_in,
        points_xyz=points_xyz,
        point_scene_instance_id=point_scene_instance_id,
        point_pole_group_id=point_pole_group_id,
        scene_instance_class_id=scene_instance_class_id,
        random_seed=random_seed,
    )


def _build_scene_instance_id_map(instances: list[SceneInstance], *, class_min: int = 7) -> np.ndarray:
    old_to_scene = np.full((len(instances),), -1, dtype=np.int32)
    next_id = 0
    for old_id, inst in enumerate(instances):
        if int(inst.class_id) < int(class_min):
            continue
        old_to_scene[old_id] = int(next_id)
        next_id += 1
    return old_to_scene


def _build_tree_metrics_from_stage_clusters(
    points_xyz: np.ndarray,
    instances: list[SceneInstance],
    trunk_stage_clusters: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    radius_clusters = list(trunk_stage_clusters.get("radius", []))
    if not radius_clusters:
        return []

    anchors: list[dict[str, Any]] = []
    for anchor in radius_clusters:
        point_indices = np.unique(np.asarray(anchor.get("point_indices", np.zeros((0,), dtype=np.int32)), dtype=np.int32))
        if point_indices.size == 0:
            continue
        xy_center = np.asarray(anchor.get("xy_center", np.asarray([np.nan, np.nan], dtype=np.float32)), dtype=np.float32).reshape(-1)
        robust_radius = float(anchor.get("robust_radius", np.nan))
        anchors.append(
            {
                "point_indices": point_indices.astype(np.int32, copy=False),
                "xy_center": xy_center[:2] if xy_center.size >= 2 else np.asarray([np.nan, np.nan], dtype=np.float32),
                "robust_radius": robust_radius,
            }
        )

    if not anchors:
        return []

    tree_instance_ids = [inst_id for inst_id, inst in enumerate(instances) if int(inst.class_id) == int(TREE_CLASS_ID)]
    used_anchor_ids: set[int] = set()
    metrics: list[dict[str, Any]] = []
    for inst_id in tree_instance_ids:
        inst_points = np.unique(np.asarray(instances[inst_id].point_indices, dtype=np.int32))
        if inst_points.size == 0:
            continue

        best_anchor_id = -1
        best_overlap = -1
        for anchor_id, anchor in enumerate(anchors):
            if anchor_id in used_anchor_ids:
                continue
            overlap = int(np.intersect1d(inst_points, anchor["point_indices"], assume_unique=False).size)
            if overlap > best_overlap:
                best_overlap = overlap
                best_anchor_id = int(anchor_id)

        if best_anchor_id < 0 or best_overlap <= 0:
            inst_xy_center = points_xyz[inst_points.astype(np.int64, copy=False), :2].mean(axis=0)
            best_dist = float("inf")
            for anchor_id, anchor in enumerate(anchors):
                if anchor_id in used_anchor_ids:
                    continue
                anchor_center = np.asarray(anchor["xy_center"], dtype=np.float32).reshape(2)
                if not np.all(np.isfinite(anchor_center)):
                    continue
                dist = float(np.linalg.norm(inst_xy_center - anchor_center))
                if dist < best_dist:
                    best_dist = dist
                    best_anchor_id = int(anchor_id)

        if best_anchor_id < 0:
            continue
        used_anchor_ids.add(best_anchor_id)
        anchor = anchors[best_anchor_id]
        radius = float(anchor["robust_radius"])
        dbh_m = float(2.0 * radius) if np.isfinite(radius) and radius > 0 else float("nan")
        center = np.asarray(anchor["xy_center"], dtype=np.float32).reshape(2)
        metrics.append(
            {
                "scene_instance_id": int(inst_id),
                "dbh_m": dbh_m,
                "trunk_center_xy": center.astype(np.float32, copy=False),
                "metric_source": "task5_tree_trunk_radius_stage",
            }
        )

    return metrics


def _build_old_to_new_instance_id_map(
    point_instance_id: np.ndarray,
    *,
    old_instance_count: int,
) -> np.ndarray:
    old_to_new = np.full((int(old_instance_count),), -1, dtype=np.int32)
    if old_instance_count <= 0:
        return old_to_new
    present_old_ids = np.unique(np.asarray(point_instance_id, dtype=np.int32).reshape(-1))
    present_old_ids = present_old_ids[present_old_ids >= 0]
    if present_old_ids.size == 0:
        return old_to_new
    old_to_new[present_old_ids.astype(np.int64, copy=False)] = np.arange(
        present_old_ids.size, dtype=np.int32
    )
    return old_to_new


def _resolve_local_ground_z(
    *,
    xy_center: np.ndarray,
    default_ground_z: float | None,
    station_xy: np.ndarray | None,
    station_ground_z: np.ndarray | None,
) -> float | None:
    local_ground_z = default_ground_z
    if (
        station_xy is not None
        and station_ground_z is not None
        and station_xy.shape[0] > 0
        and station_ground_z.shape[0] == station_xy.shape[0]
    ):
        dists = np.linalg.norm(station_xy - xy_center[None, :], axis=1)
        nearest_idx = int(np.argmin(dists))
        nearest_ground_z = float(station_ground_z[nearest_idx])
        if np.isfinite(nearest_ground_z):
            local_ground_z = nearest_ground_z
    if local_ground_z is None:
        return None
    if not np.isfinite(float(local_ground_z)):
        return None
    return float(local_ground_z)


def _compute_pole_group_metrics(
    *,
    points_xyz: np.ndarray,
    pole_groups: list[dict[str, Any]],
    default_ground_z: float | None,
    station_xy: np.ndarray | None,
    station_ground_z: np.ndarray | None,
    band_min: float,
    band_max: float,
) -> int:
    low = min(float(band_min), float(band_max))
    high = max(float(band_min), float(band_max))
    num_points = int(points_xyz.shape[0])
    valid_metrics = 0
    for group in pole_groups:
        group["diameter_m"] = float("nan")
        group["center_xy"] = np.asarray([np.nan, np.nan], dtype=np.float32)
        group["metric_source"] = "task5_pole_group_metric_unavailable"

        points = np.unique(
            np.asarray(group.get("point_indices", np.zeros((0,), dtype=np.int32)), dtype=np.int32).reshape(-1)
        )
        if points.size < 3:
            continue
        valid = (points >= 0) & (points < num_points)
        points = points[valid].astype(np.int32, copy=False)
        if points.size < 3:
            continue

        xy_center = points_xyz[points.astype(np.int64, copy=False), :2].mean(axis=0)
        local_ground_z = _resolve_local_ground_z(
            xy_center=xy_center,
            default_ground_z=default_ground_z,
            station_xy=station_xy,
            station_ground_z=station_ground_z,
        )
        if local_ground_z is None:
            continue

        z = points_xyz[points.astype(np.int64, copy=False), 2]
        band_mask = (z >= float(local_ground_z) + low) & (z <= float(local_ground_z) + high)
        band_points = points[band_mask].astype(np.int32, copy=False)
        if band_points.size < 3:
            continue

        fit = _fit_circle_taubin_svd_xy(points_xyz[band_points.astype(np.int64, copy=False)])
        if fit is None:
            continue
        center_xy, radius = fit
        diameter_m = float(2.0 * float(radius))
        if not np.isfinite(diameter_m) or diameter_m <= 0:
            continue

        group["diameter_m"] = diameter_m
        group["center_xy"] = np.asarray(center_xy, dtype=np.float32).reshape(2)
        group["metric_source"] = "task5_pole_group_band_taubin_svd"
        valid_metrics += 1
    return int(valid_metrics)


def _filter_pole_groups_by_height_range(
    *,
    points_xyz: np.ndarray,
    pole_groups: list[dict[str, Any]],
    point_pole_group_id: np.ndarray,
    min_height_diff_m: float,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, int]]:
    num_points = int(points_xyz.shape[0])
    source_point_group = np.asarray(point_pole_group_id, dtype=np.int32).reshape(-1)
    if source_point_group.shape[0] != num_points:
        fixed = np.full((num_points,), -1, dtype=np.int32)
        copy_n = min(num_points, source_point_group.shape[0])
        fixed[:copy_n] = source_point_group[:copy_n]
        source_point_group = fixed

    min_height = max(0.0, float(min_height_diff_m))
    stats = {
        "input_groups": int(len(pole_groups)),
        "kept_groups": 0,
        "dropped_groups": 0,
        "dropped_points": 0,
    }
    if min_height <= 0:
        return list(pole_groups), source_point_group.copy(), stats

    kept_groups: list[dict[str, Any]] = []
    point_group_out = np.full((num_points,), -1, dtype=np.int32)
    for group in pole_groups:
        points = np.unique(
            np.asarray(group.get("point_indices", np.zeros((0,), dtype=np.int32)), dtype=np.int32).reshape(-1)
        )
        if points.size == 0:
            stats["dropped_groups"] += 1
            continue
        valid = (points >= 0) & (points < num_points)
        points = points[valid].astype(np.int32, copy=False)
        if points.size == 0:
            stats["dropped_groups"] += 1
            continue

        z = points_xyz[points.astype(np.int64, copy=False), 2]
        height_diff = float(np.max(z) - np.min(z))
        if not np.isfinite(height_diff) or height_diff < min_height:
            stats["dropped_groups"] += 1
            stats["dropped_points"] += int(points.size)
            continue

        group_copy = dict(group)
        new_pole_id = int(len(kept_groups))
        group_copy["pole_id"] = new_pole_id
        group_copy["height_diff_m"] = float(height_diff)
        group_copy["point_indices"] = points.astype(np.int32, copy=False)
        kept_groups.append(group_copy)
        point_group_out[points.astype(np.int64, copy=False)] = new_pole_id

    stats["kept_groups"] = int(len(kept_groups))
    return kept_groups, point_group_out, stats


def _filter_pole_groups_by_max_diameter(
    *,
    pole_groups: list[dict[str, Any]],
    point_pole_group_id: np.ndarray,
    num_points: int,
    max_diameter_m: float,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    source_point_group = np.asarray(point_pole_group_id, dtype=np.int32).reshape(-1)
    if source_point_group.shape[0] != int(num_points):
        fixed = np.full((int(num_points),), -1, dtype=np.int32)
        copy_n = min(int(num_points), source_point_group.shape[0])
        fixed[:copy_n] = source_point_group[:copy_n]
        source_point_group = fixed

    threshold = float(max_diameter_m)
    stats: dict[str, Any] = {
        "input_groups": int(len(pole_groups)),
        "kept_groups": 0,
        "dropped_groups": 0,
        "dropped_points": 0,
        "max_diameter_m": threshold,
    }
    if (not np.isfinite(threshold)) or threshold <= 0.0:
        stats["kept_groups"] = int(len(pole_groups))
        return list(pole_groups), source_point_group.copy(), stats

    kept_groups: list[dict[str, Any]] = []
    point_group_out = np.full((int(num_points),), -1, dtype=np.int32)
    for group in pole_groups:
        points = np.unique(
            np.asarray(group.get("point_indices", np.zeros((0,), dtype=np.int32)), dtype=np.int32).reshape(-1)
        )
        valid = (points >= 0) & (points < int(num_points))
        points = points[valid].astype(np.int32, copy=False)
        if points.size == 0:
            stats["dropped_groups"] += 1
            continue

        diameter_m = float(group.get("diameter_m", np.nan))
        if np.isfinite(diameter_m) and diameter_m > threshold:
            stats["dropped_groups"] += 1
            stats["dropped_points"] += int(points.size)
            continue

        group_copy = dict(group)
        new_pole_id = int(len(kept_groups))
        group_copy["pole_id"] = new_pole_id
        group_copy["point_indices"] = points.astype(np.int32, copy=False)
        kept_groups.append(group_copy)
        point_group_out[points.astype(np.int64, copy=False)] = new_pole_id

    stats["kept_groups"] = int(len(kept_groups))
    return kept_groups, point_group_out, stats


def _drop_pseudo_tree_instances_by_pole_match(
    *,
    points_xyz: np.ndarray,
    instances: list[SceneInstance],
    point_instance_id: np.ndarray,
    tree_metrics_old_ids: list[dict[str, Any]],
    pole_groups: list[dict[str, Any]],
    center_distance_max: float,
    diameter_diff_max: float,
) -> dict[str, int]:
    stats = {
        "candidate_pairs": 0,
        "diameter_gate_rejected_pairs": 0,
        "matched_pairs": 0,
        "dropped_tree_instances": 0,
        "dropped_tree_points": 0,
    }
    max_dist = float(center_distance_max)
    if max_dist <= 0:
        return stats
    max_diameter_diff = float(diameter_diff_max)
    use_diameter_gate = bool(np.isfinite(max_diameter_diff) and max_diameter_diff >= 0.0)

    tree_centers: dict[int, np.ndarray] = {}
    tree_diameter_m: dict[int, float] = {}
    for item in tree_metrics_old_ids:
        tree_old_id = int(item.get("scene_instance_id", -1))
        if tree_old_id < 0 or tree_old_id >= len(instances):
            continue
        if int(instances[tree_old_id].class_id) != int(TREE_CLASS_ID):
            continue
        center = np.asarray(item.get("trunk_center_xy", np.asarray([np.nan, np.nan], dtype=np.float32)), dtype=np.float32).reshape(-1)
        if center.size < 2:
            continue
        center_xy = center[:2].astype(np.float32, copy=False)
        if not np.all(np.isfinite(center_xy)):
            continue
        tree_centers[tree_old_id] = center_xy
        dbh_m = float(item.get("dbh_m", np.nan))
        if np.isfinite(dbh_m) and dbh_m > 0:
            tree_diameter_m[tree_old_id] = dbh_m
    if not tree_centers:
        return stats

    pole_centers: dict[int, np.ndarray] = {}
    pole_diameter_m: dict[int, float] = {}
    for pole_id, group in enumerate(pole_groups):
        center = np.asarray(group.get("center_xy", np.asarray([np.nan, np.nan], dtype=np.float32)), dtype=np.float32).reshape(-1)
        if center.size < 2:
            continue
        center_xy = center[:2].astype(np.float32, copy=False)
        if not np.all(np.isfinite(center_xy)):
            continue
        pole_centers[int(pole_id)] = center_xy
        diameter_m = float(group.get("diameter_m", np.nan))
        if np.isfinite(diameter_m) and diameter_m > 0:
            pole_diameter_m[int(pole_id)] = diameter_m
    if not pole_centers:
        return stats

    candidate_pairs: list[tuple[float, int, int]] = []
    diameter_gate_rejected = 0
    for tree_old_id, tree_center in tree_centers.items():
        for pole_id, pole_center in pole_centers.items():
            dist = float(np.linalg.norm(tree_center - pole_center))
            if dist > max_dist:
                continue
            if use_diameter_gate:
                tree_d = float(tree_diameter_m.get(int(tree_old_id), np.nan))
                pole_d = float(pole_diameter_m.get(int(pole_id), np.nan))
                if (not np.isfinite(tree_d)) or (not np.isfinite(pole_d)):
                    diameter_gate_rejected += 1
                    continue
                if abs(tree_d - pole_d) > max_diameter_diff:
                    diameter_gate_rejected += 1
                    continue
            candidate_pairs.append((dist, int(tree_old_id), int(pole_id)))
    stats["diameter_gate_rejected_pairs"] = int(diameter_gate_rejected)
    if not candidate_pairs:
        return stats

    candidate_pairs.sort(key=lambda item: (float(item[0]), int(item[1]), int(item[2])))
    stats["candidate_pairs"] = int(len(candidate_pairs))

    assigned_tree_ids: set[int] = set()
    assigned_pole_ids: set[int] = set()
    matched_pairs: list[tuple[int, int]] = []
    for _, tree_old_id, pole_id in candidate_pairs:
        if tree_old_id in assigned_tree_ids or pole_id in assigned_pole_ids:
            continue
        assigned_tree_ids.add(tree_old_id)
        assigned_pole_ids.add(pole_id)
        matched_pairs.append((int(tree_old_id), int(pole_id)))
    if not matched_pairs:
        return stats

    num_points = int(points_xyz.shape[0])
    dropped_tree_instances = 0
    dropped_tree_points = 0
    for tree_old_id, _pole_id in matched_pairs:
        tree_points = np.unique(
            np.asarray(instances[tree_old_id].point_indices, dtype=np.int32).reshape(-1)
        )
        if tree_points.size == 0:
            continue
        valid = (tree_points >= 0) & (tree_points < num_points)
        tree_points = tree_points[valid].astype(np.int32, copy=False)
        if tree_points.size == 0:
            continue

        tree_points_i64 = tree_points.astype(np.int64, copy=False)
        point_instance_id[tree_points_i64] = -1
        instances[tree_old_id].point_indices = np.zeros((0,), dtype=np.int32)

        dropped_tree_instances += 1
        dropped_tree_points += int(tree_points.size)

    stats["matched_pairs"] = int(len(matched_pairs))
    stats["dropped_tree_instances"] = int(dropped_tree_instances)
    stats["dropped_tree_points"] = int(dropped_tree_points)
    return stats


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
        "--denoise-dbscan-min-samples",
        type=int,
        default=5,
        help="DBSCAN min_samples used by the generic denoise clustering.",
    )
    parser.add_argument(
        "--ground-z-quantile",
        type=float,
        default=0.05,
        help="Lower quantile used as z_low in relative-height support band construction for all classes.",
    )
    parser.add_argument(
        "--ground-support-height",
        type=float,
        default=0.20,
        help="Relative low-band ratio for support bbox (z_low + ratio * (q95 - z_low)).",
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
        default=0.80,
        help="Relative high-band ratio for support bbox (z_low + ratio * (q95 - z_low)).",
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
        "--fence-dbscan-min-samples",
        type=int,
        default=5,
        help="DBSCAN min_samples used when re-clustering fence points.",
    )
    parser.add_argument(
        "--fence-csf-cloth-resolution",
        type=float,
        default=1.0,
        help="CSF cloth resolution for scene-level ground extraction used by fence filtering.",
    )
    parser.add_argument(
        "--fence-csf-rigidness",
        type=int,
        default=1,
        help="CSF rigidness for scene-level ground extraction (1-3).",
    )
    parser.add_argument(
        "--fence-csf-time-step",
        type=float,
        default=0.65,
        help="CSF time step for scene-level ground extraction.",
    )
    parser.add_argument(
        "--fence-csf-class-threshold",
        type=float,
        default=1.2,
        help="CSF class threshold for scene-level ground extraction.",
    )
    parser.add_argument(
        "--fence-csf-iterations",
        type=int,
        default=800,
        help="CSF iteration count for scene-level ground extraction.",
    )
    parser.add_argument(
        "--fence-csf-slope-smooth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CSF slope smoothing for scene-level ground extraction.",
    )
    parser.add_argument(
        "--fence-csf-low-band-ratio",
        type=float,
        default=0.10,
        help="Only remove fence points within this low-height ratio band when also marked as global CSF ground.",
    )
    parser.add_argument(
        "--tree-trunk-band-min",
        type=float,
        default=0.80,
        help="Lower height above ground used for trunk radius-band candidate points.",
    )
    parser.add_argument(
        "--tree-trunk-band-max",
        type=float,
        default=1.40,
        help="Upper height above ground used for trunk radius-band candidate points.",
    )
    parser.add_argument(
        "--tree-trunk-height-band-min",
        type=float,
        default=0.80,
        help="Lower height above ground used when computing robust trunk height (can differ from radius band).",
    )
    parser.add_argument(
        "--tree-trunk-height-band-max",
        type=float,
        default=1.80,
        help="Upper height above ground used when computing robust trunk height (can differ from radius band).",
    )
    parser.add_argument(
        "--tree-trunk-dbscan-eps",
        type=float,
        default=0.20,
        help="DBSCAN eps used to cluster tree trunk band points in XY.",
    )
    parser.add_argument(
        "--tree-trunk-dbscan-min-samples",
        type=int,
        default=3,
        help="DBSCAN min_samples used to cluster tree trunk band points in XY.",
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
        help="Maximum Taubin-SVD fitted XY radius allowed for an accepted tree trunk candidate.",
    )
    parser.add_argument(
        "--tree-trunk-max-residual",
        type=float,
        default=0.08,
        help="Maximum 5 percent-trimmed mean absolute radial residual (meters) allowed after trunk circle fitting.",
    )
    parser.add_argument(
        "--tree-trunk-min-verticality",
        type=float,
        default=0.55,
        help="[currently unused] Minimum verticality score for accepted tree trunk candidates (kept for compatibility).",
    )
    parser.add_argument(
        "--tree-crown-attach-distance",
        type=float,
        default=4.0,
        help="Attach a trunkless tree crown to the nearest trunk when within this XY distance.",
    )
    parser.add_argument(
        "--tree-final-denoise-eps",
        type=float,
        default=0.50,
        help="Final per-tree DBSCAN eps (meters) after trunk/crown processing; keep largest cluster only.",
    )
    parser.add_argument(
        "--pole-cluster-eps",
        type=float,
        default=0.30,
        help="DBSCAN eps in 3D for merged pole-group clustering on final classes 1-6 points.",
    )
    parser.add_argument(
        "--pole-cluster-min-samples",
        type=int,
        default=10,
        help="DBSCAN min_samples for merged pole-group clustering on final classes 1-6 points.",
    )
    parser.add_argument(
        "--pole-min-cluster-points",
        type=int,
        default=100,
        help="Drop pole-group clusters whose point count is below this value.",
    )
    parser.add_argument(
        "--pole-min-height-diff",
        type=float,
        default=0.50,
        help="Drop pole-group clusters whose z-range (z_max-z_min) is below this value (meters).",
    )
    parser.add_argument(
        "--pole-metric-band-min",
        type=float,
        default=0.80,
        help="Lower height above local ground used for pole-group diameter/center fitting.",
    )
    parser.add_argument(
        "--pole-metric-band-max",
        type=float,
        default=1.40,
        help="Upper height above local ground used for pole-group diameter/center fitting.",
    )
    parser.add_argument(
        "--tree-pole-center-merge-distance",
        type=float,
        default=0.35,
        help="Drop a pseudo-tree instance when its trunk center is uniquely matched to a pole_group center within this threshold (meters).",
    )
    parser.add_argument(
        "--tree-pole-diameter-diff-max",
        type=float,
        default=0.30,
        help="Additional pseudo-tree gate: require |tree_dbh - pole_diameter| <= this threshold (meters).",
    )
    parser.add_argument(
        "--pole-max-diameter-m",
        type=float,
        default=5.0,
        help="Drop pole-group clusters whose fitted diameter exceeds this value (meters). <=0 disables this filter.",
    )
    parser.add_argument(
        "--save-tree-trunk-anchors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save tree trunk anchor LAS files for height/radius stages (one trunk cluster per instance color).",
    )
    parser.add_argument(
        "--save-pole-groups-las",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save merged pole-group LAS carrying pole_group_id and has_cls_1..has_cls_6 flags.",
    )
    parser.add_argument(
        "--save-tree-pre-denoise-las",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save a tree-only LAS before denoise for debugging tree over-merge/over-prune.",
    )
    parser.add_argument("--output-dir-name", type=str, default="fusion", help="Per-scene output folder name.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for instance colors in LAS.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing {scene}_instance_seg outputs.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of scenes to process in parallel. Set to 1 for serial mode to reduce memory usage.",
    )
    return parser.parse_args()


def process_scene(scene_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    scene_name = scene_dir.name
    output_dir = scene_dir / args.output_dir_name
    if (not args.overwrite) and output_dir.is_dir() and any(output_dir.iterdir()):
        return {
            "scene": scene_name,
            "status": "skipped_output_dir_exists",
            "output_dir": str(output_dir),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_npz_path = output_dir / f"{scene_name}_instance_seg.npz"
    scene_las_path = output_dir / f"{scene_name}_instance_seg.las"
    scene_refined_las_path = output_dir / f"{scene_name}_instance_seg_refined.las"
    scene_csf_ground_las_path = output_dir / f"{scene_name}_scene_csf_ground.las"
    scene_tree_pre_denoise_las_path = output_dir / f"{scene_name}_instance_seg_tree_pre_denoise.las"
    scene_final_npz_path = output_dir / f"{scene_name}_instance_seg_final.npz"
    scene_final_pre_pole_las_path = output_dir / f"{scene_name}_instance_seg_final_pre_pole.las"
    scene_final_las_path = output_dir / f"{scene_name}_instance_seg_final.las"
    scene_pole_groups_merged_las_path = output_dir / f"{scene_name}_pole_groups_merged.las"
    scene_tree_metrics_npz_path = output_dir / f"{scene_name}_tree_metrics.npz"
    scene_tree_trunks_height_las_path = output_dir / f"{scene_name}_instance_seg_tree_trunks_height.las"
    scene_tree_trunks_radius_las_path = output_dir / f"{scene_name}_instance_seg_tree_trunks_radius.las"
    scene_tree_trunks_las_path = scene_tree_trunks_radius_las_path
    scene_meta_path = output_dir / f"{scene_name}_instance_seg_meta.json"
    save_tree_trunk_anchors = bool(getattr(args, "save_tree_trunk_anchors", True))
    save_tree_pre_denoise_las = bool(getattr(args, "save_tree_pre_denoise_las", True))
    save_pole_groups_las = bool(getattr(args, "save_pole_groups_las", False))

    if (
        (not args.overwrite)
        and scene_npz_path.exists()
        and scene_las_path.exists()
        and scene_refined_las_path.exists()
        and scene_csf_ground_las_path.exists()
        and (not save_tree_pre_denoise_las or scene_tree_pre_denoise_las_path.exists())
        and scene_final_npz_path.exists()
        and scene_final_pre_pole_las_path.exists()
        and scene_final_las_path.exists()
        and scene_tree_metrics_npz_path.exists()
        and (not save_pole_groups_las or scene_pole_groups_merged_las_path.exists())
        and (not save_tree_trunk_anchors or (
            scene_tree_trunks_height_las_path.exists()
            and scene_tree_trunks_radius_las_path.exists()
        ))
    ):
        return {
            "scene": scene_name,
            "status": "skipped_exists",
            "scene_npz": str(scene_npz_path),
            "scene_las": str(scene_las_path),
            "scene_refined_las": str(scene_refined_las_path),
            "scene_csf_ground_las": str(scene_csf_ground_las_path),
            "scene_fence_csf_ground_las": str(scene_csf_ground_las_path),
            "scene_tree_pre_denoise_las": str(scene_tree_pre_denoise_las_path) if save_tree_pre_denoise_las else None,
            "scene_final_npz": str(scene_final_npz_path),
            "scene_final_pre_pole_las": str(scene_final_pre_pole_las_path),
            "scene_final_las": str(scene_final_las_path),
            "scene_tree_metrics_npz": str(scene_tree_metrics_npz_path),
            "scene_pole_groups_merged_las": str(scene_pole_groups_merged_las_path) if save_pole_groups_las else None,
            "scene_tree_trunks_las": str(scene_tree_trunks_las_path) if save_tree_trunk_anchors else None,
            "scene_tree_trunks_height_las": str(scene_tree_trunks_height_las_path) if save_tree_trunk_anchors else None,
            "scene_tree_trunks_radius_las": str(scene_tree_trunks_radius_las_path) if save_tree_trunk_anchors else None,
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
    preassign_pruned_points = _prune_tree_points_before_assignment(candidates, candidate_to_instance, instances)
    print(f"[{scene_name}] pre-assignment overlap pruning: removed_points={preassign_pruned_points}")
    point_instance_id, point_confidence = assign_points(
        candidates,
        candidate_to_instance,
        num_points=num_points,
        num_instances=len(instances),
    )

    assigned_points_before_denoise = int(np.count_nonzero(point_instance_id >= 0))
    print(f"[{scene_name}] point assignment: assigned_points={assigned_points_before_denoise}")
    tree_instance_ids = np.asarray(
        [inst_id for inst_id, inst in enumerate(instances) if inst.class_id == TREE_CLASS_ID],
        dtype=np.int32,
    )
    print(f"[{scene_name}] denoise config: skip_tree_instances={int(tree_instance_ids.size)}")

    selected_points_tree_pre_denoise = 0
    if save_tree_pre_denoise_las:
        tree_pre_denoise_point_instance_id = np.full(point_instance_id.shape, -1, dtype=np.int32)
        if tree_instance_ids.size > 0:
            tree_point_mask = np.isin(point_instance_id, tree_instance_ids, assume_unique=False)
            tree_pre_denoise_point_instance_id[tree_point_mask] = point_instance_id[tree_point_mask]
        selected_points_tree_pre_denoise = write_scene_las(
            output_path=scene_tree_pre_denoise_las_path,
            las_in=las_data,
            points_xyz=points_xyz,
            point_instance_id=tree_pre_denoise_point_instance_id,
            instances=instances,
            random_seed=int(args.random_seed),
        )
        print(
            f"[{scene_name}] tree pre-denoise export: "
            f"points={selected_points_tree_pre_denoise} trees={int(tree_instance_ids.size)}"
        )

    denoise_log = denoise_assignments(
        points_xyz=points_xyz,
        point_instance_id=point_instance_id,
        point_confidence=point_confidence,
        num_instances=len(instances),
        eps=float(args.denoise_eps),
        min_points=int(args.denoise_min_points),
        dbscan_min_samples=int(getattr(args, "denoise_dbscan_min_samples", 5)),
        skip_instance_ids=tree_instance_ids,
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

    refine_result = _refine_instances_with_ground_and_fence(
        points_xyz,
        instances,
        point_instance_id,
        ground_quantile=float(args.ground_z_quantile),
        support_height=float(args.ground_support_height),
        bbox_expand=float(args.ground_bbox_expand),
        support_top_ratio=float(getattr(args, "ground_support_top_ratio", 0.70)),
        fence_recluster_eps=float(args.fence_recluster_eps),
        fence_min_cluster_points=int(args.fence_min_cluster_points),
        fence_min_height=float(args.fence_min_height),
        fence_dbscan_min_samples=int(getattr(args, "fence_dbscan_min_samples", 5)),
        fence_csf_cloth_resolution=float(getattr(args, "fence_csf_cloth_resolution", 1.0)),
        fence_csf_rigidness=int(getattr(args, "fence_csf_rigidness", 1)),
        fence_csf_time_step=float(getattr(args, "fence_csf_time_step", 0.65)),
        fence_csf_class_threshold=float(getattr(args, "fence_csf_class_threshold", 1.2)),
        fence_csf_iterations=int(getattr(args, "fence_csf_iterations", 800)),
        fence_csf_slope_smooth=bool(getattr(args, "fence_csf_slope_smooth", True)),
        fence_csf_low_band_ratio=float(getattr(args, "fence_csf_low_band_ratio", 0.10)),
        return_fence_global_ground_mask=True,
    )
    if isinstance(refine_result, tuple) and len(refine_result) == 4:
        refined_instances, refined_point_instance_id, refine_stats, fence_global_ground_mask = refine_result
    else:
        refined_instances, refined_point_instance_id, refine_stats = refine_result  # type: ignore[misc]
        fence_global_ground_mask = None
    selected_points_refined = write_scene_las(
        output_path=scene_refined_las_path,
        las_in=las_data,
        points_xyz=points_xyz,
        point_instance_id=refined_point_instance_id,
        instances=refined_instances,
        random_seed=int(args.random_seed),
    )
    scene_ground_point_indices = (
        np.where(fence_global_ground_mask)[0].astype(np.int32, copy=False)
        if isinstance(fence_global_ground_mask, np.ndarray) and fence_global_ground_mask.shape[0] == points_xyz.shape[0]
        else np.zeros((0,), dtype=np.int32)
    )
    selected_points_scene_csf_ground = write_point_subset_las(
        output_path=scene_csf_ground_las_path,
        las_in=las_data,
        points_xyz=points_xyz,
        point_indices=scene_ground_point_indices,
        classification=2,
        rgb8=(90, 200, 90),
    )
    print(
        f"[{scene_name}] refined postprocess: ground_removed_points={int(refine_stats['ground_removed_points_total'])} "
        f"fence_instances={int(refine_stats['fence_instances_before'])}->{int(refine_stats['fence_instances_after'])} "
        f"fence_clusters_filtered_small={int(refine_stats['fence_clusters_filtered_small'])} "
        f"fence_clusters_filtered_low_height={int(refine_stats['fence_clusters_filtered_low_height'])} "
        f"fence_global_csf_applied={int(refine_stats.get('fence_global_csf_applied', refine_stats.get('fence_csf_instances_applied', 0)))} "
        f"fence_csf_removed={int(refine_stats.get('fence_csf_removed_points_total', 0))} "
        f"fence_effective_ground={int(refine_stats.get('fence_effective_ground_points_total', 0))} "
        f"scene_csf_ground_points={selected_points_scene_csf_ground} "
        f"final_instances={len(refined_instances)} final_points={selected_points_refined}"
    )

    tree_ground_refs = _tree_station_ground_refs_from_effective_stations(scene_dir / "projected_images")
    tree_station_xy = None
    tree_station_ground_z = None
    if tree_ground_refs is not None:
        tree_station_xy, tree_station_ground_z = tree_ground_refs
    tree_ground_z = _tree_ground_z_from_effective_stations(scene_dir / "projected_images")
    final_instances, final_point_instance_id, tree_stats, trunk_stage_clusters = _refine_tree_instances(
        points_xyz,
        refined_instances,
        tree_ground_z=tree_ground_z,
        tree_station_xy=tree_station_xy,
        tree_station_ground_z=tree_station_ground_z,
        trunk_band_min=float(getattr(args, "tree_trunk_band_min", 0.80)),
        trunk_band_max=float(getattr(args, "tree_trunk_band_max", 1.40)),
        trunk_height_band_min=float(getattr(args, "tree_trunk_height_band_min", getattr(args, "tree_trunk_band_min", 0.80))),
        trunk_height_band_max=float(getattr(args, "tree_trunk_height_band_max", getattr(args, "tree_trunk_band_max", 1.40))),
        trunk_dbscan_eps=float(getattr(args, "tree_trunk_dbscan_eps", 0.20)),
        trunk_dbscan_min_samples=int(getattr(args, "tree_trunk_dbscan_min_samples", 3)),
        trunk_min_points=int(getattr(args, "tree_trunk_min_points", 10)),
        trunk_min_height=float(getattr(args, "tree_trunk_min_height", 0.30)),
        trunk_max_radius=float(getattr(args, "tree_trunk_max_radius", 0.30)),
        trunk_max_residual=float(getattr(args, "tree_trunk_max_residual", 0.08)),
        trunk_min_verticality=float(getattr(args, "tree_trunk_min_verticality", 0.55)),
        crown_attach_distance=float(getattr(args, "tree_crown_attach_distance", 4.0)),
        tree_final_denoise_eps=float(getattr(args, "tree_final_denoise_eps", 0.50)),
        return_trunk_stage_clusters=True,
    )
    print(
        f"[{scene_name}] step done: tree_refine "
        f"final_tree_instances={len(final_instances)} "
        f"trunk_candidate_groups={int(tree_stats.get('trunk_candidate_groups', 0))} "
        f"trunk_anchors={int(tree_stats.get('trunk_anchors', 0))}"
    )

    tree_metrics_old_ids = _build_tree_metrics_from_stage_clusters(
        points_xyz=points_xyz,
        instances=final_instances,
        trunk_stage_clusters=trunk_stage_clusters,
    )
    print(
        f"[{scene_name}] step done: tree_metric_candidates "
        f"count={len(tree_metrics_old_ids)}"
    )

    selected_points_tree_trunks_height = 0
    selected_points_tree_trunks_radius = 0
    if save_tree_trunk_anchors:
        selected_points_tree_trunks_height = write_tree_trunk_stage_las(
            output_path=scene_tree_trunks_height_las_path,
            las_in=las_data,
            points_xyz=points_xyz,
            stage_clusters=trunk_stage_clusters.get('height', []),
            random_seed=int(args.random_seed),
        )
        selected_points_tree_trunks_radius = write_tree_trunk_stage_las(
            output_path=scene_tree_trunks_radius_las_path,
            las_in=las_data,
            points_xyz=points_xyz,
            stage_clusters=trunk_stage_clusters.get('radius', []),
            random_seed=int(args.random_seed),
        )
        print(
            f"[{scene_name}] step done: write_tree_trunk_stage_las "
            f"height_points={selected_points_tree_trunks_height} "
            f"radius_points={selected_points_tree_trunks_radius}"
        )
    else:
        print(
            f"[{scene_name}] step done: skip_write_tree_trunk_stage_las "
            f"reason=save_tree_trunk_anchors_false"
        )

    # Release heavy intermediate clusters before pole-group clustering.
    trunk_stage_clusters.clear()
    del trunk_stage_clusters
    gc.collect()
    print(f"[{scene_name}] step done: release_tree_intermediates")

    selected_points_final_pre_pole = write_scene_las(
        output_path=scene_final_pre_pole_las_path,
        las_in=las_data,
        points_xyz=points_xyz,
        point_instance_id=final_point_instance_id,
        instances=final_instances,
        random_seed=int(args.random_seed),
    )
    print(
        f"[{scene_name}] step done: write_final_pre_pole_las "
        f"points={selected_points_final_pre_pole} path={scene_final_pre_pole_las_path.name}"
    )

    pole_groups, point_pole_group_id = build_pole_groups(
        points_xyz=points_xyz,
        instances=final_instances,
        eps=float(getattr(args, "pole_cluster_eps", 0.30)),
        min_samples=int(getattr(args, "pole_cluster_min_samples", 10)),
        min_cluster_points=int(getattr(args, "pole_min_cluster_points", 100)),
    )
    raw_pole_group_count = int(len(pole_groups))
    pole_groups, point_pole_group_id, pole_height_filter_stats = _filter_pole_groups_by_height_range(
        points_xyz=points_xyz,
        pole_groups=pole_groups,
        point_pole_group_id=point_pole_group_id,
        min_height_diff_m=float(getattr(args, "pole_min_height_diff", 0.50)),
    )
    pole_metrics_valid_before_merge = _compute_pole_group_metrics(
        points_xyz=points_xyz,
        pole_groups=pole_groups,
        default_ground_z=tree_ground_z,
        station_xy=tree_station_xy,
        station_ground_z=tree_station_ground_z,
        band_min=float(getattr(args, "pole_metric_band_min", 0.80)),
        band_max=float(getattr(args, "pole_metric_band_max", 1.40)),
    )
    pole_groups, point_pole_group_id, pole_diameter_filter_stats = _filter_pole_groups_by_max_diameter(
        pole_groups=pole_groups,
        point_pole_group_id=point_pole_group_id,
        num_points=num_points,
        max_diameter_m=float(getattr(args, "pole_max_diameter_m", 5.0)),
    )
    pole_metrics_valid_after_diameter = int(
        sum(
            1
            for group in pole_groups
            if np.isfinite(float(group.get("diameter_m", np.nan))) and float(group.get("diameter_m", np.nan)) > 0.0
        )
    )
    print(
        f"[{scene_name}] step done: build_pole_groups "
        f"raw_groups={raw_pole_group_count} "
        f"groups={len(pole_groups)} "
        f"dropped_by_height={int(pole_height_filter_stats.get('dropped_groups', 0))} "
        f"dropped_by_diameter={int(pole_diameter_filter_stats.get('dropped_groups', 0))} "
        f"metrics_ready={pole_metrics_valid_after_diameter}"
    )

    pseudo_tree_drop_stats = _drop_pseudo_tree_instances_by_pole_match(
        points_xyz=points_xyz,
        instances=final_instances,
        point_instance_id=final_point_instance_id,
        tree_metrics_old_ids=tree_metrics_old_ids,
        pole_groups=pole_groups,
        center_distance_max=float(getattr(args, "tree_pole_center_merge_distance", 0.35)),
        diameter_diff_max=float(getattr(args, "tree_pole_diameter_diff_max", 0.30)),
    )
    old_to_new_after_pseudo = _build_old_to_new_instance_id_map(
        final_point_instance_id,
        old_instance_count=len(final_instances),
    )
    final_instances = rebuild_instances(final_instances, final_point_instance_id)
    pole_metrics_valid_after_drop = int(pole_metrics_valid_after_diameter)
    print(
        f"[{scene_name}] step done: pseudo_tree_drop "
        f"candidate_pairs={int(pseudo_tree_drop_stats.get('candidate_pairs', 0))} "
        f"diameter_gate_rejected_pairs={int(pseudo_tree_drop_stats.get('diameter_gate_rejected_pairs', 0))} "
        f"matched_pairs={int(pseudo_tree_drop_stats.get('matched_pairs', 0))} "
        f"dropped_tree_instances={int(pseudo_tree_drop_stats.get('dropped_tree_instances', 0))} "
        f"dropped_tree_points={int(pseudo_tree_drop_stats.get('dropped_tree_points', 0))} "
        f"pole_metrics_ready={pole_metrics_valid_after_drop}"
    )

    old_to_scene_instance = _build_scene_instance_id_map(final_instances, class_min=7)
    tree_metrics: list[dict[str, Any]] = []
    for item in tree_metrics_old_ids:
        old_id = int(item.get("scene_instance_id", -1))
        if old_id < 0 or old_id >= old_to_new_after_pseudo.shape[0]:
            continue
        new_old_id = int(old_to_new_after_pseudo[old_id])
        if new_old_id < 0 or new_old_id >= len(final_instances):
            continue
        if int(final_instances[new_old_id].class_id) != int(TREE_CLASS_ID):
            continue
        scene_instance_id = int(old_to_scene_instance[new_old_id])
        if scene_instance_id < 0:
            continue
        item_copy = dict(item)
        item_copy["scene_instance_id"] = int(scene_instance_id)
        tree_metrics.append(item_copy)

    selected_points_pole_groups = 0
    if save_pole_groups_las:
        selected_points_pole_groups = write_pole_groups_las(
            output_path=scene_pole_groups_merged_las_path,
            las_in=las_data,
            points_xyz=points_xyz,
            pole_groups=pole_groups,
            random_seed=int(args.random_seed),
        )
        print(
            f"[{scene_name}] step done: write_pole_groups_las "
            f"points={selected_points_pole_groups} path={scene_pole_groups_merged_las_path.name}"
        )
    else:
        print(
            f"[{scene_name}] step done: skip_write_pole_groups_las "
            f"reason=save_pole_groups_las_false"
        )

    point_pole_group_id_compact = np.asarray(point_pole_group_id, dtype=np.int32).reshape(-1)
    if point_pole_group_id_compact.shape[0] != num_points:
        fixed = np.full((num_points,), -1, dtype=np.int32)
        copy_n = min(num_points, point_pole_group_id_compact.shape[0])
        fixed[:copy_n] = point_pole_group_id_compact[:copy_n]
        point_pole_group_id_compact = fixed

    final_point_confidence = np.where(
        (final_point_instance_id >= 0) | (point_pole_group_id_compact >= 0),
        1.0,
        0.0,
    ).astype(np.float32, copy=False)
    save_scene_npz(
        output_path=scene_final_npz_path,
        scene_name=scene_name,
        instances=final_instances,
        point_instance_id=final_point_instance_id,
        point_confidence=final_point_confidence,
        pole_groups=pole_groups,
        point_pole_group_id=point_pole_group_id_compact,
        scene_instance_class_min=7,
    )
    print(
        f"[{scene_name}] step done: save_final_npz "
        f"path={scene_final_npz_path.name}"
    )
    scene_instance_class_id = np.asarray(
        [int(inst.class_id) for inst in final_instances if int(inst.class_id) >= 7],
        dtype=np.int32,
    )
    point_scene_instance_id = np.full((num_points,), -1, dtype=np.int32)
    for old_id, new_id in enumerate(old_to_scene_instance):
        if int(new_id) < 0:
            continue
        point_scene_instance_id[final_point_instance_id == int(old_id)] = int(new_id)
    selected_points_final = write_compact_final_scene_las(
        output_path=scene_final_las_path,
        las_in=las_data,
        points_xyz=points_xyz,
        point_scene_instance_id=point_scene_instance_id,
        point_pole_group_id=point_pole_group_id_compact,
        scene_instance_class_id=scene_instance_class_id,
        random_seed=int(args.random_seed),
    )
    print(
        f"[{scene_name}] step done: write_compact_final_las "
        f"points={selected_points_final} path={scene_final_las_path.name}"
    )
    save_tree_metrics_npz(
        output_path=scene_tree_metrics_npz_path,
        scene_name=scene_name,
        tree_metrics=tree_metrics,
    )
    print(
        f"[{scene_name}] step done: save_tree_metrics "
        f"count={len(tree_metrics)} path={scene_tree_metrics_npz_path.name}"
    )

    print(
        f"[{scene_name}] tree postprocess: ground_z={tree_ground_z if tree_ground_z is not None else 'none'} "
        f"trunk_candidate_groups={int(tree_stats['trunk_candidate_groups'])} "
        f"trunk_anchors={int(tree_stats['trunk_anchors'])} "
        f"pending_crowns={int(tree_stats['pending_crowns_before_attach'])} "
        f"attached={int(tree_stats['pending_crowns_attached'])} kept={int(tree_stats['pending_crowns_kept'])} dropped={int(tree_stats.get('pending_crowns_dropped', 0))} "
        f"tree_final_denoise_removed={int(tree_stats.get('tree_final_denoise_removed_points', 0))} "
        f"tree_final_denoise_touched={int(tree_stats.get('tree_final_denoise_touched_instances', 0))} "
        f"final_instances={len(final_instances)} final_points_pre_pole={selected_points_final_pre_pole} "
        f"final_points_compact={selected_points_final} "
        f"trunk_points(height/radius)={selected_points_tree_trunks_height}/{selected_points_tree_trunks_radius}"
    )
    print(
        f"[{scene_name}] pole groups: groups={len(pole_groups)} points={selected_points_pole_groups} "
        f"dropped_by_height={int(pole_height_filter_stats.get('dropped_groups', 0))} "
        f"dropped_by_diameter={int(pole_diameter_filter_stats.get('dropped_groups', 0))} "
        f"tree_metrics={len(tree_metrics)} pseudo_tree_dropped={int(pseudo_tree_drop_stats.get('dropped_tree_instances', 0))}"
    )

    meta = {
        "scene": scene_name,
        "las_path": str(las_path),
        "num_points": num_points,
        "candidate_stats": candidate_stats,
        "merged_instance_count_before_denoise": int(candidate_to_instance.max() + 1) if candidate_to_instance.size > 0 else 0,
        "final_instance_count": len(final_instances),
        "selected_instance_points": selected_points,
        "selected_instance_points_refined": selected_points_refined,
        "selected_scene_csf_ground_points": selected_points_scene_csf_ground,
        "selected_fence_csf_ground_points": selected_points_scene_csf_ground,
        "selected_tree_pre_denoise_points": selected_points_tree_pre_denoise,
        "selected_instance_points_final_pre_pole": selected_points_final_pre_pole,
        "selected_instance_points_final": selected_points_final,
        "selected_pole_group_points": selected_points_pole_groups,
        "pole_group_count": int(len(pole_groups)),
        "pole_height_filter": pole_height_filter_stats,
        "pole_diameter_filter": pole_diameter_filter_stats,
        "pole_metric_valid_groups_before_pseudo_merge": int(pole_metrics_valid_before_merge),
        "pole_metric_valid_groups_after_diameter_filter": int(pole_metrics_valid_after_diameter),
        "pole_metric_valid_groups_after_pseudo_drop": int(pole_metrics_valid_after_drop),
        "pole_metric_valid_groups_after_pseudo_merge": int(pole_metrics_valid_after_drop),
        "pseudo_tree_drop": pseudo_tree_drop_stats,
        "tree_metric_count": int(len(tree_metrics)),
        "selected_tree_trunk_points": selected_points_tree_trunks_radius,
        "selected_tree_trunk_points_height": selected_points_tree_trunks_height,
        "selected_tree_trunk_points_radius": selected_points_tree_trunks_radius,
        "denoise": denoise_log,
        "refined_postprocess": refine_stats,
        "refined_instance_count": len(refined_instances),
        "tree_postprocess": tree_stats,
        "final_tree_instance_count": len(final_instances),
        "scene_npz": str(scene_npz_path),
        "scene_las": str(scene_las_path),
        "scene_refined_las": str(scene_refined_las_path),
        "scene_csf_ground_las": str(scene_csf_ground_las_path),
        "scene_fence_csf_ground_las": str(scene_csf_ground_las_path),
        "scene_tree_pre_denoise_las": str(scene_tree_pre_denoise_las_path) if save_tree_pre_denoise_las else None,
        "scene_final_npz": str(scene_final_npz_path),
        "scene_final_pre_pole_las": str(scene_final_pre_pole_las_path),
        "scene_final_las": str(scene_final_las_path),
        "scene_tree_metrics_npz": str(scene_tree_metrics_npz_path),
        "scene_pole_groups_merged_las": str(scene_pole_groups_merged_las_path) if save_pole_groups_las else None,
        "scene_tree_trunks_las": str(scene_tree_trunks_las_path) if save_tree_trunk_anchors else None,
        "scene_tree_trunks_height_las": str(scene_tree_trunks_height_las_path) if save_tree_trunk_anchors else None,
        "scene_tree_trunks_radius_las": str(scene_tree_trunks_radius_las_path) if save_tree_trunk_anchors else None,
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

    num_workers = max(1, int(getattr(args, "num_workers", 2)))
    if num_workers == 1 or len(scene_dirs) == 1:
        for scene_dir in scene_dirs:
            result = process_scene(scene_dir, args)
            print(f"[{result.get('status', 'unknown')}] {scene_dir.name}")
        print("done.")
        return

    failed_scenes = 0
    max_workers = min(num_workers, len(scene_dirs))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_scene = {executor.submit(process_scene, scene_dir, args): scene_dir for scene_dir in scene_dirs}
        for future in as_completed(future_to_scene):
            scene_dir = future_to_scene[future]
            try:
                result = future.result()
                print(f"[{result.get('status', 'unknown')}] {scene_dir.name}")
            except Exception as exc:
                failed_scenes += 1
                print(f"[failed_exception] {scene_dir.name}: {exc}")

    if failed_scenes > 0:
        raise RuntimeError(f"Task5 parallel run failed on {failed_scenes} scene(s).")
    print("done.")


if __name__ == "__main__":
    main()
