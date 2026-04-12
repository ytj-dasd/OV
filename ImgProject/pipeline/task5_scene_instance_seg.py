from __future__ import annotations

import argparse
import json
import sys
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
        "--fence-csf-enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run per-fence-instance CSF ground filtering before fence re-clustering.",
    )
    parser.add_argument(
        "--fence-csf-cloth-resolution",
        type=float,
        default=1.0,
        help="CSF cloth resolution for fence pre-filtering.",
    )
    parser.add_argument(
        "--fence-csf-rigidness",
        type=int,
        default=1,
        help="CSF rigidness for fence pre-filtering (1-3).",
    )
    parser.add_argument(
        "--fence-csf-time-step",
        type=float,
        default=0.65,
        help="CSF time step for fence pre-filtering.",
    )
    parser.add_argument(
        "--fence-csf-class-threshold",
        type=float,
        default=1.2,
        help="CSF class threshold for fence pre-filtering.",
    )
    parser.add_argument(
        "--fence-csf-iterations",
        type=int,
        default=800,
        help="CSF iteration count for fence pre-filtering.",
    )
    parser.add_argument(
        "--fence-csf-slope-smooth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CSF slope smoothing for fence pre-filtering.",
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
        "--save-tree-trunk-anchors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save tree trunk anchor LAS files for height/radius stages (one trunk cluster per instance color).",
    )
    parser.add_argument("--output-dir-name", type=str, default="fusion", help="Per-scene output folder name.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for instance colors in LAS.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing {scene}_instance_seg outputs.",
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
    scene_final_npz_path = output_dir / f"{scene_name}_instance_seg_final.npz"
    scene_final_las_path = output_dir / f"{scene_name}_instance_seg_final.las"
    scene_tree_trunks_height_las_path = output_dir / f"{scene_name}_instance_seg_tree_trunks_height.las"
    scene_tree_trunks_radius_las_path = output_dir / f"{scene_name}_instance_seg_tree_trunks_radius.las"
    scene_tree_trunks_las_path = scene_tree_trunks_radius_las_path
    scene_meta_path = output_dir / f"{scene_name}_instance_seg_meta.json"
    save_tree_trunk_anchors = bool(getattr(args, "save_tree_trunk_anchors", True))

    if (
        (not args.overwrite)
        and scene_npz_path.exists()
        and scene_las_path.exists()
        and scene_refined_las_path.exists()
        and scene_final_npz_path.exists()
        and scene_final_las_path.exists()
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
            "scene_final_npz": str(scene_final_npz_path),
            "scene_final_las": str(scene_final_las_path),
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

    denoise_log = denoise_assignments(
        points_xyz=points_xyz,
        point_instance_id=point_instance_id,
        point_confidence=point_confidence,
        num_instances=len(instances),
        eps=float(args.denoise_eps),
        min_points=int(args.denoise_min_points),
        dbscan_min_samples=int(getattr(args, "denoise_dbscan_min_samples", 5)),
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
        support_top_ratio=float(getattr(args, "ground_support_top_ratio", 0.70)),
        fence_recluster_eps=float(args.fence_recluster_eps),
        fence_min_cluster_points=int(args.fence_min_cluster_points),
        fence_min_height=float(args.fence_min_height),
        fence_dbscan_min_samples=int(getattr(args, "fence_dbscan_min_samples", 5)),
        fence_csf_enable=bool(getattr(args, "fence_csf_enable", True)),
        fence_csf_cloth_resolution=float(getattr(args, "fence_csf_cloth_resolution", 1.0)),
        fence_csf_rigidness=int(getattr(args, "fence_csf_rigidness", 1)),
        fence_csf_time_step=float(getattr(args, "fence_csf_time_step", 0.65)),
        fence_csf_class_threshold=float(getattr(args, "fence_csf_class_threshold", 1.2)),
        fence_csf_iterations=int(getattr(args, "fence_csf_iterations", 800)),
        fence_csf_slope_smooth=bool(getattr(args, "fence_csf_slope_smooth", True)),
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
        f"fence_csf_applied={int(refine_stats.get('fence_csf_instances_applied', 0))} "
        f"fence_csf_removed={int(refine_stats.get('fence_csf_removed_points_total', 0))} "
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
        return_trunk_stage_clusters=True,
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
        f"[{scene_name}] tree postprocess: ground_z={tree_ground_z if tree_ground_z is not None else 'none'} "
        f"trunk_candidate_groups={int(tree_stats['trunk_candidate_groups'])} "
        f"trunk_anchors={int(tree_stats['trunk_anchors'])} "
        f"pending_crowns={int(tree_stats['pending_crowns_before_attach'])} "
        f"attached={int(tree_stats['pending_crowns_attached'])} kept={int(tree_stats['pending_crowns_kept'])} dropped={int(tree_stats.get('pending_crowns_dropped', 0))} "
        f"final_instances={len(final_instances)} final_points={selected_points_final} "
        f"trunk_points(height/radius)={selected_points_tree_trunks_height}/{selected_points_tree_trunks_radius}"
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
        "scene_final_npz": str(scene_final_npz_path),
        "scene_final_las": str(scene_final_las_path),
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

    for scene_dir in scene_dirs:
        result = process_scene(scene_dir, args)
        print(f"[{result.get('status', 'unknown')}] {scene_dir.name}")
    print("done.")


if __name__ == "__main__":
    main()
