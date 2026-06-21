"""Belt v2: tighten green-belt candidate envelopes with height edges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from landmark.apps import belt
from landmark.apps.sidewalk_v2 import (
    DEFAULT_ANGLE_SMOOTH_THRESHOLD_DEG,
    DEFAULT_CONTROL_SAMPLE_M,
    DEFAULT_ENVELOPE_BUFFER_M,
    DEFAULT_ENVELOPE_CORE_M,
    DEFAULT_INTERPOLATE_GAP_M,
    DEFAULT_SEARCH_DIRECTION_MODE,
    DEFAULT_TIGHTEN_SEARCH_M,
    _build_nearest_core_lookup,
    _closed,
    _control_search_unit,
    _default_map_dir,
    _interpolate_missed_points,
    _nearest_core_target,
    _render_envelope_compare_on_height,
    _render_label_on_height,
    _render_raw_controls_on_height,
    _resolve_height_assets,
    _resample_closed_ring,
    _smooth_closed_ring_for_direction,
    _smooth_displacements,
    _smooth_sharp_ring_points,
    _tighten_point_bidirectional_along_unit,
    _write_envelope_boundary_shp,
    _write_envelope_polygon_shp,
)
from landmark.tools.to_shp.geometry import pixel_to_xy


Image.MAX_IMAGE_PIXELS = None


def _load_json(path: Path | str) -> dict[str, Any]:
    path = Path(path).expanduser()
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _label_components(mask: np.ndarray) -> np.ndarray:
    num, labels, _stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    label_map = np.full(mask.shape, -1, dtype=np.int32)
    for comp_id in range(1, num):
        label_map[labels == comp_id] = int(comp_id - 1)
    return label_map


def _extract_belt_envelope_records(
    label_map: np.ndarray,
    *,
    height_values: np.ndarray,
    meta: dict[str, Any],
    envelope_buffer_m: float,
    envelope_core_m: float,
    tighten_search_m: float,
    control_sample_m: float,
    interpolate_gap_m: float,
    angle_smooth_threshold_deg: float,
    search_direction_mode: str,
) -> list[dict[str, Any]]:
    if search_direction_mode not in {"core", "normal"}:
        raise ValueError("search_direction_mode must be 'core' or 'normal'")

    mpp = float(meta["meters_per_pixel"])
    buffer_px = max(0, int(round(envelope_buffer_m / mpp)))
    core_px = max(1, int(round(envelope_core_m / mpp)))
    search_px = max(1, int(round(tighten_search_m / mpp)))
    sample_px = max(1.0, control_sample_m / mpp)
    max_interpolate_run_points = max(0, int(round(interpolate_gap_m / max(1e-6, control_sample_m))))

    buffer_kernel = None
    if buffer_px > 0:
        buffer_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (buffer_px * 2 + 1, buffer_px * 2 + 1))
    core_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (core_px * 2 + 1, core_px * 2 + 1))

    records: list[dict[str, Any]] = []
    ids = [int(x) for x in np.unique(label_map) if int(x) >= 0]
    for label_id in ids:
        rows, cols = np.nonzero(label_map == label_id)
        if rows.size < 3:
            continue

        x0, x1 = int(cols.min()), int(cols.max()) + 1
        y0, y1 = int(rows.min()), int(rows.max()) + 1
        pad = max(buffer_px, search_px) + 4
        r0 = max(0, y0 - pad)
        r1 = min(label_map.shape[0], y1 + pad)
        c0 = max(0, x0 - pad)
        c1 = min(label_map.shape[1], x1 + pad)
        crop = np.asarray(label_map[r0:r1, c0:c1] == label_id, dtype=np.uint8)
        if int(np.count_nonzero(crop)) < 3:
            continue

        buffered = cv2.dilate(crop, buffer_kernel, iterations=1) > 0 if buffer_kernel is not None else crop > 0
        contours, _ = cv2.findContours(buffered.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        raw_control_points = contour.reshape(-1, 2).astype(np.float64)
        approx = _resample_closed_ring(raw_control_points, sample_px)
        if len(approx) < 3:
            continue

        smoothed_for_direction = _smooth_closed_ring_for_direction(approx) if search_direction_mode == "normal" else None
        crop_rows, crop_cols = np.nonzero(crop)
        center_cr = np.array([float(crop_cols.mean()), float(crop_rows.mean())], dtype=np.float64)
        core = cv2.erode(crop, core_kernel, iterations=1) > 0
        if not np.any(core):
            core = crop > 0
        core_lookup = _build_nearest_core_lookup(core)
        height_crop = np.asarray(height_values[r0:r1, c0:c1], dtype=np.float32)

        tightened: list[np.ndarray] = []
        hit_flags: list[bool] = []
        hit_sides: list[str] = []
        direction_segments: list[list[list[float]]] = []
        outward_direction_segments: list[list[list[float]]] = []
        inward_hits = 0
        outward_hits = 0
        core_targeted = 0
        for idx, point in enumerate(approx):
            target = _nearest_core_target(core_lookup, point)
            if target is None:
                target = center_cr
            else:
                core_targeted += 1
            unit = _control_search_unit(
                approx,
                smoothed_for_direction,
                idx,
                target,
                search_direction_mode=search_direction_mode,
            )
            if unit is None:
                tightened.append(point.astype(np.float64))
                hit_flags.append(False)
                hit_sides.append("none")
                continue

            direction_end = point + unit * float(search_px)
            outward_direction_end = point - unit * float(search_px)
            direction_segments.append(
                [[float(point[0] + c0), float(point[1] + r0)], [float(direction_end[0] + c0), float(direction_end[1] + r0)]]
            )
            outward_direction_segments.append(
                [
                    [float(point[0] + c0), float(point[1] + r0)],
                    [float(outward_direction_end[0] + c0), float(outward_direction_end[1] + r0)],
                ]
            )
            p, hit, _step, hit_side = _tighten_point_bidirectional_along_unit(
                point,
                unit,
                height_crop,
                max_steps_px=search_px,
            )
            tightened.append(p)
            hit_flags.append(hit)
            hit_sides.append(hit_side)
            if hit_side == "inward":
                inward_hits += 1
            elif hit_side == "outward":
                outward_hits += 1

        tightened_arr = np.asarray(tightened, dtype=np.float64)
        tightened_arr, interpolated = _interpolate_missed_points(
            tightened_arr,
            hit_flags,
            max_run_points=max_interpolate_run_points,
        )
        tightened_arr = _smooth_displacements(approx, tightened_arr, mpp=mpp)
        tightened_arr, angle_smoothed = _smooth_sharp_ring_points(
            tightened_arr,
            threshold_deg=angle_smooth_threshold_deg,
        )

        offset = np.array([float(c0), float(r0)], dtype=np.float64)
        raw_ring = _closed(approx + offset)
        ring = _closed(tightened_arr + offset)
        final_move = np.linalg.norm(tightened_arr - approx, axis=1)
        moved_mask = final_move > 0.5
        moved = int(np.count_nonzero(moved_mask))
        total_move = float(final_move[moved_mask].sum()) if moved else 0.0
        max_move = float(final_move[moved_mask].max()) if moved else 0.0
        move_segments = [
            [[float(point[0] + c0), float(point[1] + r0)], [float(final_point[0] + c0), float(final_point[1] + r0)]]
            for point, final_point, is_moved in zip(approx, tightened_arr, moved_mask, strict=False)
            if is_moved
        ]

        records.append(
            {
                "label_id": int(label_id),
                "area_px": int(np.count_nonzero(crop)),
                "area_m2": float(np.count_nonzero(crop) * mpp * mpp),
                "raw_contour_points": int(len(raw_control_points)),
                "control_points": int(len(ring) - 1),
                "moved_points": int(moved),
                "diff_hit_points": int(np.count_nonzero(hit_flags)),
                "inward_hit_points": int(inward_hits),
                "outward_hit_points": int(outward_hits),
                "interpolated_points": int(interpolated),
                "angle_smoothed_points": int(angle_smoothed),
                "core_targeted_points": int(core_targeted),
                "search_direction_mode": search_direction_mode,
                "move_ratio": float(moved / max(1, len(ring) - 1)),
                "mean_move_px": float(total_move / max(1, moved)) if moved else 0.0,
                "max_move_px": float(max_move),
                "xy": pixel_to_xy(ring, meta),
                "raw_pixel_ring_cr": raw_ring,
                "raw_control_points_cr": raw_ring[:-1],
                "raw_control_hit_sides": hit_sides,
                "direction_segments_cr": direction_segments,
                "outward_direction_segments_cr": outward_direction_segments,
                "pixel_ring_cr": ring,
                "move_segments_cr": move_segments,
            }
        )
    return records


def run_belt_v2(
    pre_part_dir: Path | str,
    output_dir: Path | str | None = None,
    *,
    green_label_map_path: Path | str | None = None,
    road_label_map_path: Path | str | None = None,
    envelope_buffer_m: float = DEFAULT_ENVELOPE_BUFFER_M,
    envelope_core_m: float = DEFAULT_ENVELOPE_CORE_M,
    tighten_search_m: float = DEFAULT_TIGHTEN_SEARCH_M,
    control_sample_m: float = DEFAULT_CONTROL_SAMPLE_M,
    interpolate_gap_m: float = DEFAULT_INTERPOLATE_GAP_M,
    angle_smooth_threshold_deg: float = DEFAULT_ANGLE_SMOOTH_THRESHOLD_DEG,
    search_direction_mode: str = DEFAULT_SEARCH_DIRECTION_MODE,
) -> dict[str, Path]:
    pre_part_dir = Path(pre_part_dir).expanduser()
    map_dir = _default_map_dir(pre_part_dir)
    output_dir = Path(output_dir).expanduser() if output_dir is not None else map_dir / "belt_v2"
    debug_dir = output_dir / "debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    if green_label_map_path is None:
        green_label_map_path = map_dir / "objs" / "green_veg" / "result" / "label_map.npy"
    if road_label_map_path is None:
        default_road = map_dir / "objs" / "road_score05" / "result" / "label_map.npy"
        road_label_map_path = default_road if default_road.is_file() else None
    green_label_map_path = Path(green_label_map_path).expanduser()
    road_label_map_path = Path(road_label_map_path).expanduser() if road_label_map_path else None
    if not green_label_map_path.is_file():
        raise FileNotFoundError(green_label_map_path)
    if road_label_map_path is not None and not road_label_map_path.is_file():
        raise FileNotFoundError(road_label_map_path)

    assets = _resolve_height_assets(pre_part_dir)
    height_meta = _load_json(assets["height_meta"])
    height_values = np.load(assets["height_values"], mmap_mode="r")
    green_label_map = np.load(green_label_map_path, mmap_mode="r")
    if tuple(green_label_map.shape) != tuple(height_values.shape):
        raise ValueError(f"green label_map shape {green_label_map.shape} does not match height raster {height_values.shape}")

    mpp = float(height_meta["meters_per_pixel"])
    diff = belt._height_diff_mask(height_values)
    belt._write_mask(diff, debug_dir / "diff.png")

    road_buffer: np.ndarray | None = None
    if road_label_map_path is not None:
        road_label_map = np.load(road_label_map_path, mmap_mode="r")
        if tuple(road_label_map.shape) != tuple(height_values.shape):
            raise ValueError(f"road label_map shape {road_label_map.shape} does not match height raster {height_values.shape}")
        road_mask = np.asarray(road_label_map >= 0)
        road_buffer = cv2.dilate(road_mask.astype(np.uint8), belt._kernel(belt.ROAD_BUFFER_M, mpp)) > 0
        belt._write_mask(road_buffer, debug_dir / "road_buffer.png")

    green_union = belt._build_green_union(green_label_map, mpp=mpp)
    belt._write_mask(green_union, debug_dir / "green_union.png")

    final_candidates, candidate_records, shape_candidates = belt._select_candidates(
        green_union,
        diff,
        road_buffer=road_buffer,
        mpp=mpp,
    )
    belt._write_mask(shape_candidates, debug_dir / "method_green_shape.png")
    belt._write_mask(final_candidates, debug_dir / "candidate_strip_mask.png")
    belt._write_rgb(diff, final_candidates, debug_dir / "candidate_with_diff.png", overlay_color=(0, 255, 0))
    if road_buffer is not None:
        belt._write_rgb(road_buffer, final_candidates, debug_dir / "candidate_with_road_buffer.png", overlay_color=(0, 255, 0))

    final_mask = belt._finalize_mask(final_candidates, mpp=mpp)
    belt._write_mask(final_mask, debug_dir / "final_belt_mask.png")
    belt_label_map = _label_components(final_mask)
    label_vis_out = _render_label_on_height(belt_label_map, assets["height_png"], debug_dir / "belt_label_map_on_height.png")

    envelope_records = _extract_belt_envelope_records(
        belt_label_map,
        height_values=height_values,
        meta=height_meta,
        envelope_buffer_m=envelope_buffer_m,
        envelope_core_m=envelope_core_m,
        tighten_search_m=tighten_search_m,
        control_sample_m=control_sample_m,
        interpolate_gap_m=interpolate_gap_m,
        angle_smooth_threshold_deg=angle_smooth_threshold_deg,
        search_direction_mode=search_direction_mode,
    )
    polygon_out = _write_envelope_polygon_shp(envelope_records, output_dir / "belt")
    boundary_out = _write_envelope_boundary_shp(envelope_records, output_dir / "belt_boundary")
    belt_vis_out = _render_envelope_compare_on_height(envelope_records, assets["height_png"], output_dir / "belt_on_height.png")
    search_dirs_out = _render_raw_controls_on_height(envelope_records, assets["height_png"], debug_dir / "search_dirs_on_height.png")

    summary = {
        "pipeline": "green_candidate_tightened_envelope",
        "pre_part_dir": str(pre_part_dir),
        "green_label_map": str(green_label_map_path),
        "road_label_map": str(road_label_map_path) if road_label_map_path is not None else None,
        "height_png": str(assets["height_png"]),
        "height_values": str(assets["height_values"]),
        "height_meta": str(assets["height_meta"]),
        "output_dir": str(output_dir),
        "debug_dir": str(debug_dir),
        "meters_per_pixel": mpp,
        "thresholds": {
            "height_diff_min_m": belt.HEIGHT_DIFF_MIN_M,
            "height_diff_max_m": belt.HEIGHT_DIFF_MAX_M,
            "green_close_radius_m": belt.GREEN_CLOSE_RADIUS_M,
            "strip_connect_radius_m": belt.STRIP_CONNECT_RADIUS_M,
            "final_close_radius_m": belt.FINAL_CLOSE_RADIUS_M,
            "diff_ring_radius_m": belt.DIFF_RING_RADIUS_M,
            "min_area_m2": belt.MIN_AREA_M2,
            "min_width_m": belt.MIN_WIDTH_M,
            "max_width_m": belt.MAX_WIDTH_M,
            "min_length_m": belt.MIN_LENGTH_M,
            "min_aspect": belt.MIN_ASPECT,
            "min_rect_fill": belt.MIN_RECT_FILL,
            "max_rect_fill": belt.MAX_RECT_FILL,
            "require_diff_support": belt.REQUIRE_DIFF_SUPPORT,
            "min_diff_support_ratio": belt.MIN_DIFF_SUPPORT_RATIO,
            "road_buffer_m": belt.ROAD_BUFFER_M,
            "min_road_buffer_overlap": belt.MIN_ROAD_BUFFER_OVERLAP,
            "envelope_buffer_m": float(envelope_buffer_m),
            "envelope_core_m": float(envelope_core_m),
            "tighten_search_m": float(tighten_search_m),
            "control_sample_m": float(control_sample_m),
            "interpolate_gap_m": float(interpolate_gap_m),
            "angle_smooth_threshold_deg": float(angle_smooth_threshold_deg),
            "search_direction_mode": search_direction_mode,
        },
        "diff_pixels": int(np.count_nonzero(diff)),
        "green_union_pixels": int(np.count_nonzero(green_union)),
        "shape_candidate_pixels": int(np.count_nonzero(shape_candidates)),
        "final_candidate_pixels": int(np.count_nonzero(final_candidates)),
        "final_mask_pixels": int(np.count_nonzero(final_mask)),
        "candidate_count": len(candidate_records),
        "shape_candidate_count": int(sum(1 for r in candidate_records if r["keep_shape"])),
        "final_candidate_count": int(sum(1 for r in candidate_records if r["keep_final"])),
        "belt_component_count": int(np.count_nonzero(np.unique(belt_label_map) >= 0)),
        "envelope_record_count": len(envelope_records),
        "polygon_output": str(polygon_out),
        "boundary_output": str(boundary_out),
        "belt_visualization_output": str(belt_vis_out),
        "search_dirs_visualization_output": str(search_dirs_out),
        "label_visualization_output": str(label_vis_out),
        "candidate_records": candidate_records,
        "records": [
            {
                key: value
                for key, value in rec.items()
                if key
                not in {
                    "xy",
                    "raw_pixel_ring_cr",
                    "raw_control_points_cr",
                    "direction_segments_cr",
                    "outward_direction_segments_cr",
                    "pixel_ring_cr",
                    "move_segments_cr",
                }
            }
            for rec in envelope_records
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "belt": polygon_out,
        "belt_boundary": boundary_out,
        "belt_visualization": belt_vis_out,
        "debug_search_dirs_visualization": search_dirs_out,
        "summary": summary_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract tightened green-belt envelope polygons and boundaries.")
    parser.add_argument("pre_part_dir", help="pre-part output directory.")
    parser.add_argument("-o", "--output-dir", default=None, help="Output directory. Defaults to <map>/belt_v2.")
    parser.add_argument("--green-label-map", default=None, help="Existing green vegetation label_map.npy.")
    parser.add_argument("--road-label-map", default=None, help="Optional road label_map.npy. Defaults to <map>/objs/road_score05/result/label_map.npy when it exists.")
    parser.add_argument("--envelope-buffer-m", type=float, default=DEFAULT_ENVELOPE_BUFFER_M, help="Envelope buffer distance before tightening.")
    parser.add_argument("--envelope-core-m", type=float, default=DEFAULT_ENVELOPE_CORE_M, help="Internal core erosion distance used as tightening target.")
    parser.add_argument("--tighten-search-m", type=float, default=DEFAULT_TIGHTEN_SEARCH_M, help="Maximum control-point tightening search distance in metres.")
    parser.add_argument("--control-sample-m", type=float, default=DEFAULT_CONTROL_SAMPLE_M, help="Maximum envelope control-point spacing in metres.")
    parser.add_argument("--interpolate-gap-m", type=float, default=DEFAULT_INTERPOLATE_GAP_M, help="Maximum no-hit control-point run length to interpolate in metres.")
    parser.add_argument("--angle-smooth-threshold-deg", type=float, default=DEFAULT_ANGLE_SMOOTH_THRESHOLD_DEG, help="Sequentially smooth control points with angle below this threshold.")
    parser.add_argument(
        "--search-direction-mode",
        choices=("core", "normal"),
        default=DEFAULT_SEARCH_DIRECTION_MODE,
        help="Control-point tightening direction: nearest-core vector or smoothed-envelope normal.",
    )
    args = parser.parse_args()
    outputs = run_belt_v2(
        args.pre_part_dir,
        args.output_dir,
        green_label_map_path=args.green_label_map,
        road_label_map_path=args.road_label_map,
        envelope_buffer_m=args.envelope_buffer_m,
        envelope_core_m=args.envelope_core_m,
        tighten_search_m=args.tighten_search_m,
        control_sample_m=args.control_sample_m,
        interpolate_gap_m=args.interpolate_gap_m,
        angle_smooth_threshold_deg=args.angle_smooth_threshold_deg,
        search_direction_mode=args.search_direction_mode,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
