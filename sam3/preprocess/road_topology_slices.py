import argparse
from pathlib import Path

import numpy as np

import scene_slice_utils as su


def _default_base_image_from_mask(mask_path: Path) -> Path | None:
    candidates = [
        mask_path.parent.parent / "bev_stitched_v1" / "bev_big_rgb_inpaint.png",
        mask_path.parent / "all_prompts_global_overlay.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Road topology slice planning main entry. "
            "This script only orchestrates the pipeline steps and calls scene_slice_utils."
        )
    )

    parser.add_argument("mask", type=Path, help="Global road mask (png/npz)")
    parser.add_argument("output", type=Path, help="Output directory")

    # I/O and core resolutions
    parser.add_argument("--mask-key", type=str, default="union_mask")
    parser.add_argument("--bev-resolution", type=float, default=0.02)
    parser.add_argument("--topology-resolution", type=float, default=0.2)
    parser.add_argument("--base-image", type=Path, default=None)

    # Step 2: mask cleanup
    parser.add_argument("--closing-radius-m", type=float, default=5.0)
    parser.add_argument("--min-component-area-m2", type=float, default=30.0)
    parser.add_argument("--max-hole-area-m2", type=float, default=200.0)

    # Step 3-5: skeleton + graph
    parser.add_argument("--spur-prune-length-m", type=float, default=30.0)
    parser.add_argument("--junction-cluster-eps-m", type=float, default=3.0)
    parser.add_argument("--junction-min-samples", type=int, default=2)
    parser.add_argument("--min-edge-length-m", type=float, default=0.0)
    parser.add_argument("--junction-min-valid-branches", type=int, default=3)
    parser.add_argument("--junction-min-branch-length-m", type=float, default=5.0)
    parser.add_argument("--junction-merge-distance-m", type=float, default=10.0)

    # Step 6: junction radius estimation
    parser.add_argument("--branch-sample-start-m", type=float, default=2.0)
    parser.add_argument("--branch-sample-end-m", type=float, default=6.0)
    parser.add_argument("--junction-radius-scale", type=float, default=1.5)
    parser.add_argument("--junction-radius-margin-m", type=float, default=0.8)
    parser.add_argument("--junction-radius-min-m", type=float, default=3.0)
    parser.add_argument("--junction-radius-max-m", type=float, default=20.0)

    # Step 7-8: segmentation
    parser.add_argument("--segment-max-length-m", type=float, default=100.0)
    parser.add_argument("--segment-overlap-m", type=float, default=10.0)
    parser.add_argument("--segment-width-margin-m", type=float, default=10.0)
    parser.add_argument("--min-segment-half-width-m", type=float, default=1.2)
    parser.add_argument("--endpoint-extension-m", type=float, default=20.0)

    # Visualization
    parser.add_argument("--draw-node-labels", action="store_true", default=True)
    parser.add_argument("--no-draw-node-labels", action="store_false", dest="draw_node_labels")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.bev_resolution <= 0:
        raise ValueError("--bev-resolution must be > 0")
    if args.topology_resolution <= 0:
        raise ValueError("--topology-resolution must be > 0")

    mask_path = args.mask.expanduser().absolute()
    out_dir = args.output.expanduser().absolute()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Input loading
    orig_mask = su.load_binary_mask(mask_path, mask_key=args.mask_key)
    orig_h, orig_w = orig_mask.shape[:2]

    # Step 1: downsample for topology planning
    topo_mask, topo_resolution, scale_x_topo_to_orig, scale_y_topo_to_orig = su.downsample_mask_for_topology(
        orig_mask,
        source_resolution=args.bev_resolution,
        topology_resolution=args.topology_resolution,
    )

    # Step 2: cleanup mask
    closing_radius_px = int(round(su._meters_to_pixels(args.closing_radius_m, topo_resolution)))
    min_component_area_px = su._area_m2_to_px(args.min_component_area_m2, topo_resolution)
    max_hole_area_px = su._area_m2_to_px(args.max_hole_area_m2, topo_resolution)
    cleaned = su.clean_road_mask(
        topo_mask,
        closing_radius_px=closing_radius_px,
        min_component_area_px=min_component_area_px,
        max_hole_area_px=max_hole_area_px,
    )

    # Step 3: skeletonize and prune spurs
    prune_length_px = su._meters_to_pixels(args.spur_prune_length_m, topo_resolution)
    skeleton = su.skeletonize_and_prune(cleaned, prune_length_px=prune_length_px)

    # Step 4: build graph from skeleton
    graph = su.build_road_graph(
        skeleton=skeleton,
        junction_cluster_eps_px=su._meters_to_pixels(args.junction_cluster_eps_m, topo_resolution),
        junction_min_samples=args.junction_min_samples,
        resolution=topo_resolution,
        min_edge_length_m=args.min_edge_length_m,
        junction_min_valid_branches=args.junction_min_valid_branches,
        junction_min_branch_length_m=args.junction_min_branch_length_m,
    )

    # Step 5: merge close junctions, then normalize node kinds by long-branch counts
    graph, merged_junction_nodes = su.merge_close_junctions(
        graph=graph,
        merge_distance_m=args.junction_merge_distance_m,
    )
    graph = su.normalize_graph_junctions_by_branch_lengths(
        graph=graph,
        min_edge_length_m=args.min_edge_length_m,
        junction_min_valid_branches=args.junction_min_valid_branches,
        junction_min_branch_length_m=args.junction_min_branch_length_m,
    )

    # Step 6: distance transform and junction radius estimation
    distance_map_m = su.compute_distance_map_m(cleaned, resolution=topo_resolution)
    junction_radii_m = su.estimate_junction_radii_m(
        graph=graph,
        distance_map_m=distance_map_m,
        sample_start_m=args.branch_sample_start_m,
        sample_end_m=args.branch_sample_end_m,
        radius_scale=args.junction_radius_scale,
        radius_margin_m=args.junction_radius_margin_m,
        radius_min_m=args.junction_radius_min_m,
        radius_max_m=args.junction_radius_max_m,
    )

    # Step 7: build slice polygons and centerline partitions
    slices = su.build_slice_polygons(
        graph=graph,
        distance_map_m=distance_map_m,
        resolution=topo_resolution,
        junction_radii_m=junction_radii_m,
        segment_max_length_m=args.segment_max_length_m,
        segment_overlap_m=args.segment_overlap_m,
        segment_width_margin_m=args.segment_width_margin_m,
        min_segment_half_width_m=args.min_segment_half_width_m,
        endpoint_extension_m=args.endpoint_extension_m,
    )
    partitions = su.build_segment_partitions(
        graph=graph,
        resolution=topo_resolution,
        segment_max_length_m=args.segment_max_length_m,
        segment_overlap_m=args.segment_overlap_m,
        endpoint_extension_m=args.endpoint_extension_m,
    )

    # Step 8: visualizations and export
    cleaned_orig = su._resize_bool_mask(cleaned, out_h=orig_h, out_w=orig_w)
    su.draw_clean_mask_black(cleaned_orig, out_dir / "step3_clean_mask_black.png")

    su.draw_skeleton_black(
        skeleton_mask=skeleton,
        nodes=graph.nodes,
        output_path=out_dir / "step4_skeleton_black.png",
        scale_x=scale_x_topo_to_orig,
        scale_y=scale_y_topo_to_orig,
        out_h=orig_h,
        out_w=orig_w,
        line_thickness_px=5,
        endpoint_radius_px=8,
    )

    su.draw_graph_black(
        graph=graph,
        output_path=out_dir / "step5_graph_black.png",
        scale_x=scale_x_topo_to_orig,
        scale_y=scale_y_topo_to_orig,
        out_h=orig_h,
        out_w=orig_w,
        draw_node_labels=args.draw_node_labels,
        edge_thickness_px=10,
    )

    base_image_path = args.base_image.expanduser().absolute() if args.base_image else _default_base_image_from_mask(mask_path)
    if base_image_path is not None and not base_image_path.exists():
        base_image_path = None

    base_canvas = su._build_base_canvas(cleaned_mask=cleaned_orig, base_image_path=base_image_path)
    su.draw_partition_centerlines_overlay(
        base_bgr=base_canvas,
        partitions=partitions,
        nodes=graph.nodes,
        output_path=out_dir / "step8_partitions_centerline_overlay.png",
        draw_node_labels=args.draw_node_labels,
        scale_x=scale_x_topo_to_orig,
        scale_y=scale_y_topo_to_orig,
        line_thickness_px=30,
        show_partition_labels=True,
    )

    su._save_graph_tables(graph, out_dir)
    su._save_slice_json(slices, resolution=topo_resolution, output_path=out_dir / "slice_polygons.json")

    np.savez_compressed(
        out_dir / "pipeline_summary.npz",
        source_mask=orig_mask.astype(np.uint8),
        topo_mask=topo_mask.astype(np.uint8),
        clean_mask=cleaned.astype(np.uint8),
        skeleton=skeleton.astype(np.uint8),
        distance_map_m=distance_map_m.astype(np.float32),
        source_resolution_m=np.array(args.bev_resolution, dtype=np.float32),
        topology_resolution_m=np.array(topo_resolution, dtype=np.float32),
        topo_to_source_scale_x=np.array(scale_x_topo_to_orig, dtype=np.float32),
        topo_to_source_scale_y=np.array(scale_y_topo_to_orig, dtype=np.float32),
        num_nodes=np.array(len(graph.nodes), dtype=np.int32),
        num_edges=np.array(len(graph.edges), dtype=np.int32),
        num_slices=np.array(len(slices), dtype=np.int32),
        num_partitions=np.array(len(partitions), dtype=np.int32),
        merged_junction_nodes=np.array(merged_junction_nodes, dtype=np.int32),
    )

    junction_count = sum(1 for n in graph.nodes if n.kind == "junction")
    endpoint_count = sum(1 for n in graph.nodes if n.kind == "endpoint")
    road_slice_count = sum(1 for s in slices if s.kind == "road_segment")
    junction_slice_count = sum(1 for s in slices if s.kind == "junction")

    print(f"Saved outputs to: {out_dir}")
    print(
        "Resolution summary: "
        f"source={args.bev_resolution:.4f}m/px, topology={topo_resolution:.4f}m/px, "
        f"topo_to_source_scale=({scale_x_topo_to_orig:.3f}, {scale_y_topo_to_orig:.3f})"
    )
    print(
        "Graph summary: "
        f"nodes={len(graph.nodes)} (junction={junction_count}, endpoint={endpoint_count}), "
        f"edges={len(graph.edges)}"
    )
    print(f"Junction merge summary: merged_junction_nodes={merged_junction_nodes}")
    print(
        "Slices summary: "
        f"total={len(slices)} (road_segment={road_slice_count}, junction={junction_slice_count})"
    )
    print(f"Partition centerlines: total={len(partitions)}")


if __name__ == "__main__":
    main()
