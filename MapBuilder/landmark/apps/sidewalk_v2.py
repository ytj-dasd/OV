"""Sidewalk v2: refine sidewalk label maps by per-instance morphology."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import shapefile
from PIL import Image

from landmark.tools.to_shp.geometry import pixel_to_xy


Image.MAX_IMAGE_PIXELS = None

DEFAULT_BUFFER_M = 0.00
DEFAULT_CLOSE_M = 0.00
DEFAULT_MAX_HOLE_AREA_M2 = 20.0
DEFAULT_ENVELOPE_BUFFER_M = 0.00
DEFAULT_ENVELOPE_CORE_M = 0.10
DEFAULT_TIGHTEN_SEARCH_M = 1.00
DEFAULT_CONTROL_SAMPLE_M = 0.20
DEFAULT_INTERPOLATE_GAP_M = 2.00
DEFAULT_ANGLE_SMOOTH_THRESHOLD_DEG = 80.0
DEFAULT_SEARCH_DIRECTION_MODE = "normal"
DEFAULT_ROAD_PROBE_M = 1.20
DEFAULT_ROAD_DILATE_M = 0.20
DEFAULT_ROAD_GAP_CLOSE_M = 0.80
DEFAULT_ROAD_MIN_RUN_M = 2.00
DEFAULT_ROAD_SIDE_BRIDGE_M = 3.00
DEFAULT_LOCAL_DENT_WIDTH_M = 1.60
DEFAULT_LOCAL_DENT_DEPTH_M = 0.16
DEFAULT_ROAD_SIDE_SMOOTH_ITERS = 2
DEFAULT_FINAL_SMOOTH_ITERS = 10
DEFAULT_DISP_SPIKE_M = 0.40
DEFAULT_DISP_SMOOTH_RADIUS = 2
DEFAULT_DEBUG_CROP_BOX = (7638, 4662, 9921, 5137)
DEFAULT_GREEN_VEG_HULL_RATIO = 0.90
DEFAULT_GREEN_VEG_BUFFER_M = 0.30
DEFAULT_GREEN_VEG_MIN_BUFFER_OVERLAP_RATIO = 0.05
HEIGHT_DIFF_MIN_M = 0.03
HEIGHT_DIFF_MAX_M = 0.20
DIFF_DENOISE_MIN_AREA_PX = 200
DIFF_HIT_RADIUS_PX = 1
SIGNED_HIT_PROFILE_OFFSET_PX = 2.0
SIGNED_HIT_HEIGHT_RADIUS_PX = 1


def _load_json(path: Path | str) -> dict[str, Any]:
    path = Path(path).expanduser()
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_image_atomic(image: Image.Image, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f"{out_path.stem}.tmp{out_path.suffix}")
    image.save(tmp_path)
    try:
        tmp_path.replace(out_path)
        return out_path
    except PermissionError:
        fallback = out_path.with_name(f"{out_path.stem}_{int(time.time())}{out_path.suffix}")
        tmp_path.replace(fallback)
        return fallback


def _save_image_with_crop(image: Image.Image, out_path: Path) -> Path:
    saved = _save_image_atomic(image, out_path)
    left, top, right, bottom = DEFAULT_DEBUG_CROP_BOX
    crop = image.crop((left, top, right, bottom))
    _save_image_atomic(crop, out_path.with_name(f"{out_path.stem}_crop{out_path.suffix}"))
    return saved


def _default_map_dir(pre_part_dir: Path) -> Path:
    pre_part_dir = Path(pre_part_dir).expanduser()
    return pre_part_dir.parent if pre_part_dir.name == "pre-part" else pre_part_dir


def _resolve_height_assets(pre_part_dir: Path) -> dict[str, Path]:
    pre_part_dir = Path(pre_part_dir).expanduser()
    pc_csf_dir = pre_part_dir / "bev_pc_csf"
    asset_dir = pc_csf_dir if pc_csf_dir.is_dir() else pre_part_dir
    filled_height_png = asset_dir / "bev_pc_csf_height_filled.png"
    filled_height_values = asset_dir / "bev_pc_csf_height_values_filled.npy"
    assets = {
        "height_png": filled_height_png if filled_height_png.is_file() else asset_dir / "bev_pc_csf_height.png",
        "height_values": filled_height_values if filled_height_values.is_file() else asset_dir / "bev_pc_csf_height_values.npy",
        "height_meta": asset_dir / "bev_pc_csf_height_meta.json",
    }
    missing = [name for name, path in assets.items() if not path.is_file()]
    if missing:
        details = ", ".join(f"{name}={assets[name]}" for name in missing)
        raise FileNotFoundError(f"Missing sidewalk_v2 inputs: {details}")
    return assets


def _fill_small_internal_holes(
    mask_u8: np.ndarray,
    *,
    max_hole_area_px: int,
    border_margin_px: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    inv = (mask_u8 == 0).astype(np.uint8)
    flood = inv.copy()
    flood_mask = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, seedPoint=(0, 0), newVal=2)
    holes = flood == 1
    num, labels, stats, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), connectivity=8)

    keep = np.zeros(mask_u8.shape, dtype=bool)
    accepted = 0
    rejected_area = 0
    rejected_border = 0
    for hole_id in range(1, num):
        area = int(stats[hole_id, cv2.CC_STAT_AREA])
        x = int(stats[hole_id, cv2.CC_STAT_LEFT])
        y = int(stats[hole_id, cv2.CC_STAT_TOP])
        w = int(stats[hole_id, cv2.CC_STAT_WIDTH])
        h = int(stats[hole_id, cv2.CC_STAT_HEIGHT])
        near_border = (
            x <= border_margin_px
            or y <= border_margin_px
            or x + w >= mask_u8.shape[1] - border_margin_px
            or y + h >= mask_u8.shape[0] - border_margin_px
        )
        if near_border:
            rejected_border += 1
            continue
        if area > max_hole_area_px:
            rejected_area += 1
            continue
        keep[labels == hole_id] = True
        accepted += 1

    filled = (mask_u8 > 0) | keep
    stats_payload = {
        "accepted_holes": accepted,
        "rejected_holes_area": rejected_area,
        "rejected_holes_border": rejected_border,
        "hole_added_pixels": int(np.count_nonzero(keep)),
    }
    return filled, keep, stats_payload


def _area_filter(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros(mask.shape, dtype=bool)
    for comp_id in range(1, num):
        if int(stats[comp_id, cv2.CC_STAT_AREA]) >= int(min_area_px):
            out[labels == comp_id] = True
    return out


def _height_diff_mask(height_values: np.ndarray) -> np.ndarray:
    h = np.asarray(height_values, dtype=np.float32)
    finite = np.isfinite(h)
    target = np.zeros(h.shape, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            src_r0 = max(0, -dr)
            src_r1 = h.shape[0] - max(0, dr)
            src_c0 = max(0, -dc)
            src_c1 = h.shape[1] - max(0, dc)
            dst_r0 = max(0, dr)
            dst_r1 = h.shape[0] - max(0, -dr)
            dst_c0 = max(0, dc)
            dst_c1 = h.shape[1] - max(0, -dc)
            src = h[src_r0:src_r1, src_c0:src_c1]
            dst = h[dst_r0:dst_r1, dst_c0:dst_c1]
            valid = finite[src_r0:src_r1, src_c0:src_c1] & finite[dst_r0:dst_r1, dst_c0:dst_c1]
            delta = np.abs(src - dst)
            hit = valid & (delta >= HEIGHT_DIFF_MIN_M) & (delta <= HEIGHT_DIFF_MAX_M)
            target[src_r0:src_r1, src_c0:src_c1] |= hit
    return target


def _diff_hit(diff: np.ndarray, col: int, row: int) -> bool:
    if row < 0 or row >= diff.shape[0] or col < 0 or col >= diff.shape[1]:
        return False
    r0 = max(0, row - DIFF_HIT_RADIUS_PX)
    r1 = min(diff.shape[0], row + DIFF_HIT_RADIUS_PX + 1)
    c0 = max(0, col - DIFF_HIT_RADIUS_PX)
    c1 = min(diff.shape[1], col + DIFF_HIT_RADIUS_PX + 1)
    return bool(np.any(diff[r0:r1, c0:c1]))


def _sample_height(height_values: np.ndarray, point_cr: np.ndarray) -> float | None:
    col = int(round(float(point_cr[0])))
    row = int(round(float(point_cr[1])))
    if row < 0 or row >= height_values.shape[0] or col < 0 or col >= height_values.shape[1]:
        return None
    value = float(height_values[row, col])
    return value if np.isfinite(value) else None


def _local_height_median(height_values: np.ndarray, point_cr: np.ndarray, *, radius_px: int) -> float | None:
    col = int(round(float(point_cr[0])))
    row = int(round(float(point_cr[1])))
    if row < 0 or row >= height_values.shape[0] or col < 0 or col >= height_values.shape[1]:
        return None
    r0 = max(0, row - radius_px)
    r1 = min(height_values.shape[0], row + radius_px + 1)
    c0 = max(0, col - radius_px)
    c1 = min(height_values.shape[1], col + radius_px + 1)
    vals = np.asarray(height_values[r0:r1, c0:c1], dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    return float(np.median(vals))


def _signed_height_hit(
    height_values: np.ndarray,
    curr_cr: np.ndarray,
    unit: np.ndarray,
    *,
    mode: str,
) -> bool:
    unit = np.asarray(unit, dtype=np.float64)
    norm = float(np.linalg.norm(unit))
    if norm <= 1e-6:
        return False
    unit = unit / norm
    curr = np.asarray(curr_cr, dtype=np.float64)
    before_h = _local_height_median(
        height_values,
        curr - unit * SIGNED_HIT_PROFILE_OFFSET_PX,
        radius_px=SIGNED_HIT_HEIGHT_RADIUS_PX,
    )
    after_h = _local_height_median(
        height_values,
        curr + unit * SIGNED_HIT_PROFILE_OFFSET_PX,
        radius_px=SIGNED_HIT_HEIGHT_RADIUS_PX,
    )
    if before_h is None or after_h is None:
        return False
    delta = after_h - before_h
    if mode == "rise":
        return HEIGHT_DIFF_MIN_M <= delta <= HEIGHT_DIFF_MAX_M
    if mode == "fall":
        return -HEIGHT_DIFF_MAX_M <= delta <= -HEIGHT_DIFF_MIN_M
    raise ValueError(f"Unsupported signed height hit mode: {mode}")


def _search_signed_height_hit(
    point_cr: np.ndarray,
    unit: np.ndarray,
    height_values: np.ndarray,
    *,
    max_steps_px: int,
    mode: str,
) -> tuple[np.ndarray, bool, int]:
    origin = np.asarray(point_cr, dtype=np.float64)
    for step in range(1, max_steps_px + 1):
        curr = origin + unit * float(step)
        if _signed_height_hit(height_values, curr, unit, mode=mode):
            return curr, True, step
    return origin, False, 0


def _tighten_point_bidirectional_along_unit(
    point_cr: np.ndarray,
    unit: np.ndarray,
    height_values: np.ndarray,
    *,
    max_steps_px: int,
) -> tuple[np.ndarray, bool, int, str]:
    unit = np.asarray(unit, dtype=np.float64)
    norm = float(np.linalg.norm(unit))
    if norm <= 1e-6:
        return point_cr.astype(np.float64), False, 0, "none"
    unit = unit / norm
    inward_p, inward_hit, inward_step = _search_signed_height_hit(
        point_cr,
        unit,
        height_values,
        max_steps_px=max_steps_px,
        mode="rise",
    )
    outward_p, outward_hit, outward_step = _search_signed_height_hit(
        point_cr,
        -unit,
        height_values,
        max_steps_px=max_steps_px,
        mode="fall",
    )
    if inward_hit and outward_hit:
        if inward_step <= outward_step:
            return inward_p, True, inward_step, "inward"
        return outward_p, True, outward_step, "outward"
    if inward_hit:
        return inward_p, True, inward_step, "inward"
    if outward_hit:
        return outward_p, True, outward_step, "outward"
    return point_cr.astype(np.float64), False, 0, "none"


def _tighten_point_bidirectional(
    point_cr: np.ndarray,
    center_cr: np.ndarray,
    height_values: np.ndarray,
    *,
    max_steps_px: int,
) -> tuple[np.ndarray, bool, int, str]:
    direction = np.asarray(center_cr, dtype=np.float64) - np.asarray(point_cr, dtype=np.float64)
    return _tighten_point_bidirectional_along_unit(
        point_cr,
        direction,
        height_values,
        max_steps_px=max_steps_px,
    )


def _interpolate_missed_points(points_cr: np.ndarray, hit_flags: list[bool], *, max_run_points: int) -> tuple[np.ndarray, int]:
    points = np.asarray(points_cr, dtype=np.float64).copy()
    n = len(points)
    if n < 3:
        return points, 0
    hit_indices = [idx for idx, hit in enumerate(hit_flags) if hit]
    if len(hit_indices) < 2:
        return points, 0

    interpolated = 0
    for seq_idx, start_idx in enumerate(hit_indices):
        end_idx = hit_indices[(seq_idx + 1) % len(hit_indices)]
        if end_idx > start_idx:
            between = list(range(start_idx + 1, end_idx))
        else:
            between = list(range(start_idx + 1, n)) + list(range(0, end_idx))
        if not between:
            continue
        if len(between) > max_run_points:
            continue

        start = points[start_idx]
        end = points[end_idx]
        for offset, point_idx in enumerate(between, start=1):
            ratio = float(offset) / float(len(between) + 1)
            points[point_idx] = start + (end - start) * ratio
            interpolated += 1
    return points, interpolated


def _smooth_sharp_ring_points(points_cr: np.ndarray, *, threshold_deg: float) -> tuple[np.ndarray, int]:
    points = np.asarray(points_cr, dtype=np.float64).copy()
    n = len(points)
    if n < 3:
        return points, 0
    changed = 0
    threshold_cos = float(np.cos(np.deg2rad(threshold_deg)))
    for idx in range(n):
        prev_pt = points[(idx - 1) % n]
        curr_pt = points[idx]
        next_pt = points[(idx + 1) % n]
        v1 = prev_pt - curr_pt
        v2 = next_pt - curr_pt
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 <= 1e-6 or n2 <= 1e-6:
            continue
        cos_angle = float(np.dot(v1, v2) / (n1 * n2))
        cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
        # Smaller angles have larger cosines.
        if cos_angle > threshold_cos:
            points[idx] = (prev_pt + next_pt) * 0.5
            changed += 1
    return points, changed


def _resample_closed_ring(points_cr: np.ndarray, max_spacing_px: float) -> np.ndarray:
    points = np.asarray(points_cr, dtype=np.float64).reshape(-1, 2)
    if len(points) < 2:
        return points
    if np.allclose(points[0], points[-1]):
        points = points[:-1]
    if len(points) < 2:
        return points

    sampled: list[np.ndarray] = []
    for idx, start in enumerate(points):
        end = points[(idx + 1) % len(points)]
        segment = end - start
        length = float(np.linalg.norm(segment))
        sampled.append(start.copy())
        if length <= max(1.0, max_spacing_px):
            continue
        interior_count = max(0, int(np.floor(length / max(1.0, max_spacing_px))))
        for step in range(1, interior_count + 1):
            ratio = float(step) * max_spacing_px / length
            if ratio >= 1.0:
                continue
            sampled.append(start + segment * ratio)
    return np.asarray(sampled, dtype=np.float64)


def _linearize_closed_bool(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    n = len(mask)
    if n == 0:
        return mask.copy(), np.asarray([], dtype=np.int64)
    false_indices = np.flatnonzero(~mask)
    if false_indices.size == 0:
        return mask.copy(), np.arange(n, dtype=np.int64)
    start = int(false_indices[0])
    order = (np.arange(n, dtype=np.int64) + start + 1) % n
    return mask[order].copy(), order


def _fill_short_false_runs_closed(mask: np.ndarray, *, max_gap_points: int) -> np.ndarray:
    linear, order = _linearize_closed_bool(mask)
    out = linear.copy()
    idx = 0
    while idx < len(out):
        if out[idx]:
            idx += 1
            continue
        start = idx
        while idx < len(out) and not out[idx]:
            idx += 1
        end = idx
        if start > 0 and end < len(out) and end - start <= max_gap_points:
            out[start:end] = True
    restored = np.zeros_like(out, dtype=bool)
    restored[order] = out
    return restored


def _remove_short_true_runs_closed(mask: np.ndarray, *, min_run_points: int) -> np.ndarray:
    linear, order = _linearize_closed_bool(mask)
    out = linear.copy()
    idx = 0
    while idx < len(out):
        if not out[idx]:
            idx += 1
            continue
        start = idx
        while idx < len(out) and out[idx]:
            idx += 1
        end = idx
        if end - start < min_run_points:
            out[start:end] = False
    restored = np.zeros_like(out, dtype=bool)
    restored[order] = out
    return restored


def _segments_from_closed_mask(mask: np.ndarray) -> list[np.ndarray]:
    linear, order = _linearize_closed_bool(mask)
    segments: list[np.ndarray] = []
    idx = 0
    while idx < len(linear):
        if not linear[idx]:
            idx += 1
            continue
        start = idx
        while idx < len(linear) and linear[idx]:
            idx += 1
        end = idx
        segments.append(order[start:end])
    return segments


def _bridge_adjacent_segments_closed(mask: np.ndarray, *, max_gap_points: int) -> tuple[np.ndarray, int, int]:
    linear, order = _linearize_closed_bool(mask)
    out = linear.copy()
    bridged_runs = 0
    bridged_points = 0
    idx = 0
    while idx < len(out):
        if out[idx]:
            idx += 1
            continue
        start = idx
        while idx < len(out) and not out[idx]:
            idx += 1
        end = idx
        if start > 0 and end < len(out) and end - start <= max_gap_points:
            out[start:end] = True
            bridged_runs += 1
            bridged_points += end - start
    restored = np.zeros_like(out, dtype=bool)
    restored[order] = out
    return restored, bridged_runs, bridged_points


def _force_single_segment_closed(mask: np.ndarray) -> tuple[np.ndarray, int, int, int]:
    linear, order = _linearize_closed_bool(mask)
    if len(linear) == 0 or not np.any(linear):
        return np.asarray(mask, dtype=bool).copy(), 0, 0, 0

    true_runs = 0
    idx = 0
    while idx < len(linear):
        if not linear[idx]:
            idx += 1
            continue
        true_runs += 1
        while idx < len(linear) and linear[idx]:
            idx += 1
    if true_runs <= 1:
        return np.asarray(mask, dtype=bool).copy(), 0, 0, 0

    false_runs: list[np.ndarray] = []
    idx = 0
    while idx < len(linear):
        if linear[idx]:
            idx += 1
            continue
        start = idx
        while idx < len(linear) and not linear[idx]:
            idx += 1
        false_runs.append(np.arange(start, idx, dtype=np.int64))

    if len(false_runs) > 1 and false_runs[0][0] == 0 and false_runs[-1][-1] == len(linear) - 1:
        false_runs = [np.concatenate([false_runs[-1], false_runs[0]])] + false_runs[1:-1]

    keep_idx = int(max(range(len(false_runs)), key=lambda run_idx: len(false_runs[run_idx])))
    out = linear.copy()
    forced_runs = 0
    forced_points = 0
    for run_idx, run in enumerate(false_runs):
        if run_idx == keep_idx:
            continue
        out[run] = True
        forced_runs += 1
        forced_points += int(len(run))

    restored = np.zeros_like(out, dtype=bool)
    restored[order] = out
    return restored, forced_runs, forced_points, int(len(false_runs[keep_idx]))


def _point_hits_mask_along_ray(mask: np.ndarray, point_cr: np.ndarray, unit: np.ndarray, *, max_steps_px: int) -> bool:
    point = np.asarray(point_cr, dtype=np.float64)
    unit = np.asarray(unit, dtype=np.float64)
    norm = float(np.linalg.norm(unit))
    if norm <= 1e-6:
        return False
    unit = unit / norm
    for step in range(1, max_steps_px + 1):
        curr = point + unit * float(step)
        col = int(round(float(curr[0])))
        row = int(round(float(curr[1])))
        if row < 0 or row >= mask.shape[0] or col < 0 or col >= mask.shape[1]:
            continue
        if mask[row, col]:
            return True
    return False


def _smooth_open_polyline(points: np.ndarray, *, iterations: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).copy()
    if len(pts) < 5:
        return pts
    for _ in range(max(0, iterations)):
        nxt = pts.copy()
        nxt[1:-1] = 0.25 * pts[:-2] + 0.50 * pts[1:-1] + 0.25 * pts[2:]
        pts = nxt
    return pts


def _suppress_local_dents(points: np.ndarray, *, mpp: float, sample_m: float, width_m: float, depth_m: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).copy()
    if len(pts) < 7:
        return pts
    half_window = max(3, int(round(width_m / max(sample_m, 1e-6) / 2.0)))
    max_depth_px = max(2.0, depth_m / mpp)
    out = pts.copy()
    for idx in range(half_window, len(pts) - half_window):
        left = np.median(pts[idx - half_window : idx], axis=0)
        right = np.median(pts[idx + 1 : idx + half_window + 1], axis=0)
        baseline = right - left
        norm = float(np.linalg.norm(baseline))
        if norm <= 1e-6:
            continue
        ratio = float(np.dot(pts[idx] - left, baseline) / (norm * norm))
        ratio = float(np.clip(ratio, 0.0, 1.0))
        projection = left + baseline * ratio
        if float(np.linalg.norm(pts[idx] - projection)) > max_depth_px:
            out[idx] = projection
    return out


def _smooth_displacements(raw_points: np.ndarray, points: np.ndarray, *, mpp: float) -> np.ndarray:
    raw = np.asarray(raw_points, dtype=np.float64)
    out = np.asarray(points, dtype=np.float64)
    if len(out) < 5:
        return out

    max_spike_px = max(3.0, DEFAULT_DISP_SPIKE_M / mpp)
    disp = out - raw
    clipped = disp.copy()
    for idx in range(len(disp)):
        neighbors = [disp[(idx + offset) % len(disp)] for offset in range(-4, 5) if offset != 0]
        med = np.median(np.asarray(neighbors, dtype=np.float64), axis=0)
        if float(np.linalg.norm(disp[idx] - med)) > max_spike_px:
            clipped[idx] = med

    smoothed = np.zeros_like(clipped)
    total_weight = 0.0
    for offset in range(-DEFAULT_DISP_SMOOTH_RADIUS, DEFAULT_DISP_SMOOTH_RADIUS + 1):
        weight = float(DEFAULT_DISP_SMOOTH_RADIUS + 1 - abs(offset))
        smoothed += np.roll(clipped, shift=offset, axis=0) * weight
        total_weight += weight
    return raw + smoothed / total_weight


def _smooth_closed_ring_coordinates(points: np.ndarray, *, iterations: int) -> np.ndarray:
    out = np.asarray(points, dtype=np.float64).copy()
    if len(out) < 5:
        return out
    for _ in range(max(0, iterations)):
        out = 0.25 * np.roll(out, shift=1, axis=0) + 0.50 * out + 0.25 * np.roll(out, shift=-1, axis=0)
    return out


def _closed(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return points
    return np.vstack([points, points[0]])


def _smooth_closed_ring_for_direction(points_cr: np.ndarray, *, window_points: int = 9) -> np.ndarray:
    points = np.asarray(points_cr, dtype=np.float64).reshape(-1, 2)
    if len(points) < 3:
        return points.copy()
    window_points = max(3, int(window_points))
    if window_points % 2 == 0:
        window_points += 1
    radius = min(window_points // 2, max(1, (len(points) - 1) // 2))
    smoothed = np.zeros_like(points, dtype=np.float64)
    for offset in range(-radius, radius + 1):
        smoothed += np.roll(points, shift=offset, axis=0)
    return smoothed / float(radius * 2 + 1)


def _unit_from_target(point_cr: np.ndarray, target_cr: np.ndarray) -> np.ndarray | None:
    direction = np.asarray(target_cr, dtype=np.float64) - np.asarray(point_cr, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return None
    return direction / norm


def _unit_from_smoothed_normal(
    points_cr: np.ndarray,
    smoothed_points_cr: np.ndarray,
    idx: int,
    target_cr: np.ndarray,
) -> np.ndarray | None:
    if len(points_cr) < 3:
        return _unit_from_target(points_cr[idx], target_cr)

    prev_pt = smoothed_points_cr[(idx - 1) % len(smoothed_points_cr)]
    next_pt = smoothed_points_cr[(idx + 1) % len(smoothed_points_cr)]
    tangent = next_pt - prev_pt
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1e-6:
        return _unit_from_target(points_cr[idx], target_cr)

    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64) / tangent_norm
    target_unit = _unit_from_target(points_cr[idx], target_cr)
    if target_unit is not None and float(np.dot(normal, target_unit)) < 0.0:
        normal = -normal
    return normal


def _control_search_unit(
    points_cr: np.ndarray,
    smoothed_points_cr: np.ndarray | None,
    idx: int,
    target_cr: np.ndarray,
    *,
    search_direction_mode: str,
) -> np.ndarray | None:
    if search_direction_mode == "core":
        return _unit_from_target(points_cr[idx], target_cr)
    if search_direction_mode == "normal":
        if smoothed_points_cr is None:
            smoothed_points_cr = _smooth_closed_ring_for_direction(points_cr)
        return _unit_from_smoothed_normal(points_cr, smoothed_points_cr, idx, target_cr)
    raise ValueError(f"Unsupported search direction mode: {search_direction_mode}")


def _build_nearest_core_lookup(core_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if not np.any(core_mask):
        return None
    inv = (~core_mask.astype(bool)).astype(np.uint8)
    _dist, labels = cv2.distanceTransformWithLabels(inv, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    core_rows, core_cols = np.nonzero(core_mask)
    label_ids = labels[core_rows, core_cols]
    order = np.argsort(label_ids)
    sorted_ids = label_ids[order]
    sorted_rows = core_rows[order]
    sorted_cols = core_cols[order]
    return labels, sorted_ids, sorted_rows, sorted_cols


def _nearest_core_target(
    lookup: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
    point_cr: np.ndarray,
) -> np.ndarray | None:
    if lookup is None:
        return None
    labels, sorted_ids, sorted_rows, sorted_cols = lookup
    row = int(round(float(point_cr[1])))
    col = int(round(float(point_cr[0])))
    if row < 0 or row >= labels.shape[0] or col < 0 or col >= labels.shape[1]:
        return None
    target_id = int(labels[row, col])
    if target_id <= 0:
        return None
    idx = int(np.searchsorted(sorted_ids, target_id))
    if idx >= len(sorted_ids) or int(sorted_ids[idx]) != target_id:
        return None
    return np.array([float(sorted_cols[idx]), float(sorted_rows[idx])], dtype=np.float64)


def _extract_tightened_envelope_records(
    label_map: np.ndarray,
    *,
    height_values: np.ndarray,
    diff: np.ndarray,
    meta: dict[str, Any],
    envelope_buffer_m: float,
    envelope_core_m: float,
    tighten_search_m: float,
    control_sample_m: float,
    interpolate_gap_m: float,
    angle_smooth_threshold_deg: float,
    search_direction_mode: str = DEFAULT_SEARCH_DIRECTION_MODE,
) -> list[dict[str, Any]]:
    if search_direction_mode not in {"core", "normal"}:
        raise ValueError("search_direction_mode must be 'core' or 'normal'")

    mpp = float(meta["meters_per_pixel"])
    buffer_px = max(1, int(round(envelope_buffer_m / mpp)))
    core_px = max(1, int(round(envelope_core_m / mpp)))
    search_px = max(1, int(round(tighten_search_m / mpp)))
    sample_px = max(1.0, control_sample_m / mpp)
    max_interpolate_run_points = max(0, int(round(interpolate_gap_m / max(1e-6, control_sample_m))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (buffer_px * 2 + 1, buffer_px * 2 + 1))
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
        buffered = cv2.dilate(crop, kernel, iterations=1) > 0
        contours, _ = cv2.findContours(buffered.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        raw_control_points = contour.reshape(-1, 2).astype(np.float64)
        approx = _resample_closed_ring(raw_control_points, sample_px)
        if len(approx) < 3:
            continue
        smoothed_for_direction = (
            _smooth_closed_ring_for_direction(approx) if search_direction_mode == "normal" else None
        )
        crop_rows, crop_cols = np.nonzero(crop)
        center_cr = np.array([float(crop_cols.mean()), float(crop_rows.mean())], dtype=np.float64)
        core = cv2.erode(crop, core_kernel, iterations=1) > 0
        if not np.any(core):
            core = crop > 0
        core_lookup = _build_nearest_core_lookup(core)
        height_crop = height_values[r0:r1, c0:c1]
        raw_ring = approx.copy()
        raw_ring[:, 0] += c0
        raw_ring[:, 1] += r0
        if not np.array_equal(raw_ring[0], raw_ring[-1]):
            raw_ring = np.vstack([raw_ring, raw_ring[0]])

        tightened: list[np.ndarray] = []
        hit_flags: list[bool] = []
        hit_sides: list[str] = []
        direction_segments: list[list[list[float]]] = []
        outward_direction_segments: list[list[list[float]]] = []
        core_targeted = 0
        inward_hits = 0
        outward_hits = 0
        for idx, point in enumerate(approx):
            target = _nearest_core_target(core_lookup, point)
            if target is None:
                target = center_cr
            else:
                core_targeted += 1
            direction_unit = _control_search_unit(
                approx,
                smoothed_for_direction,
                idx,
                target,
                search_direction_mode=search_direction_mode,
            )
            if direction_unit is not None:
                direction_end = point + direction_unit * float(search_px)
                outward_direction_end = point - direction_unit * float(search_px)
                direction_segments.append(
                    [
                        [float(point[0] + c0), float(point[1] + r0)],
                        [float(direction_end[0] + c0), float(direction_end[1] + r0)],
                    ]
                )
                outward_direction_segments.append(
                    [
                        [float(point[0] + c0), float(point[1] + r0)],
                        [float(outward_direction_end[0] + c0), float(outward_direction_end[1] + r0)],
                    ]
                )
                p, hit, step, hit_side = _tighten_point_bidirectional_along_unit(
                    point,
                    direction_unit,
                    height_crop,
                    max_steps_px=search_px,
                )
            else:
                p, hit, step, hit_side = point.astype(np.float64), False, 0, "none"
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
        tightened_arr, angle_smoothed = _smooth_sharp_ring_points(
            tightened_arr,
            threshold_deg=angle_smooth_threshold_deg,
        )
        final_move = np.linalg.norm(tightened_arr - approx, axis=1)
        moved_mask = final_move > 0.5
        moved = int(np.count_nonzero(moved_mask))
        diff_hit_points = int(np.count_nonzero(hit_flags))
        total_move = float(final_move[moved_mask].sum()) if moved else 0.0
        max_move = float(final_move[moved_mask].max()) if moved else 0.0
        move_segments: list[list[list[float]]] = []
        for point, final_point, is_moved in zip(approx, tightened_arr, moved_mask, strict=False):
            if is_moved:
                move_segments.append(
                    [
                        [float(point[0] + c0), float(point[1] + r0)],
                        [float(final_point[0] + c0), float(final_point[1] + r0)],
                    ]
                )
        ring = tightened_arr
        ring[:, 0] += c0
        ring[:, 1] += r0
        if not np.array_equal(ring[0], ring[-1]):
            ring = np.vstack([ring, ring[0]])
        xy = pixel_to_xy(ring, meta)
        records.append(
            {
                "label_id": int(label_id),
                "area_px": int(np.count_nonzero(crop)),
                "area_m2": float(np.count_nonzero(crop) * mpp * mpp),
                "raw_contour_points": int(len(raw_control_points)),
                "control_points": int(len(ring) - 1),
                "moved_points": int(moved),
                "diff_hit_points": diff_hit_points,
                "inward_hit_points": int(inward_hits),
                "outward_hit_points": int(outward_hits),
                "interpolated_points": int(interpolated),
                "angle_smoothed_points": int(angle_smoothed),
                "core_targeted_points": int(core_targeted),
                "search_direction_mode": search_direction_mode,
                "move_ratio": float(moved / max(1, len(ring) - 1)),
                "mean_move_px": float(total_move / max(1, moved)) if moved else 0.0,
                "max_move_px": float(max_move),
                "xy": xy,
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


def _extract_road_side_boundary_records(
    label_map: np.ndarray,
    road_label_map: np.ndarray,
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
    road_probe_m: float,
    road_dilate_m: float,
    road_gap_close_m: float,
    road_min_run_m: float,
    road_side_bridge_m: float,
    local_dent_width_m: float,
    local_dent_depth_m: float,
    road_side_smooth_iters: int,
    final_smooth_iters: int,
) -> list[dict[str, Any]]:
    if search_direction_mode not in {"core", "normal"}:
        raise ValueError("search_direction_mode must be 'core' or 'normal'")
    if tuple(label_map.shape) != tuple(road_label_map.shape):
        raise ValueError(f"sidewalk label_map shape {label_map.shape} does not match road label_map {road_label_map.shape}")

    mpp = float(meta["meters_per_pixel"])
    buffer_px = max(0, int(round(envelope_buffer_m / mpp)))
    core_px = max(1, int(round(envelope_core_m / mpp)))
    search_px = max(1, int(round(tighten_search_m / mpp)))
    sample_px = max(1.0, control_sample_m / mpp)
    max_interpolate_run_points = max(0, int(round(interpolate_gap_m / max(1e-6, control_sample_m))))
    road_probe_px = max(1, int(round(road_probe_m / mpp)))
    road_dilate_px = max(0, int(round(road_dilate_m / mpp)))
    road_gap_close_points = max(1, int(round(road_gap_close_m / max(1e-6, control_sample_m))))
    road_min_run_points = max(2, int(round(road_min_run_m / max(1e-6, control_sample_m))))
    road_side_bridge_points = max(0, int(round(road_side_bridge_m / max(1e-6, control_sample_m))))

    buffer_kernel = None
    if buffer_px > 0:
        buffer_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (buffer_px * 2 + 1, buffer_px * 2 + 1))
    core_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (core_px * 2 + 1, core_px * 2 + 1))
    road_dilate_kernel = None
    if road_dilate_px > 0:
        road_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (road_dilate_px * 2 + 1, road_dilate_px * 2 + 1))

    records: list[dict[str, Any]] = []
    ids = [int(x) for x in np.unique(label_map) if int(x) >= 0]
    for label_id in ids:
        rows, cols = np.nonzero(label_map == label_id)
        if rows.size < 3:
            continue
        x0, x1 = int(cols.min()), int(cols.max()) + 1
        y0, y1 = int(rows.min()), int(rows.max()) + 1
        pad = max(buffer_px, search_px, road_probe_px, road_dilate_px) + 4
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

        crop_rows, crop_cols = np.nonzero(crop)
        center_cr = np.array([float(crop_cols.mean()), float(crop_rows.mean())], dtype=np.float64)
        core = cv2.erode(crop, core_kernel, iterations=1) > 0
        if not np.any(core):
            core = crop > 0
        core_lookup = _build_nearest_core_lookup(core)
        smoothed_for_direction = _smooth_closed_ring_for_direction(approx) if search_direction_mode == "normal" else None
        height_crop = np.asarray(height_values[r0:r1, c0:c1], dtype=np.float32)
        road_crop = np.asarray(road_label_map[r0:r1, c0:c1] >= 0, dtype=np.uint8)
        if road_dilate_kernel is not None:
            road_crop = cv2.dilate(road_crop, road_dilate_kernel, iterations=1)
        road_crop_bool = road_crop > 0

        stage2_points: list[np.ndarray] = []
        hit_flags: list[bool] = []
        hit_sides: list[str] = []
        road_side_mask: list[bool] = []
        direction_segments: list[list[list[float]]] = []
        outward_direction_segments: list[list[list[float]]] = []
        compare_segments: list[list[list[float]]] = []
        for idx, point in enumerate(approx):
            target = _nearest_core_target(core_lookup, point)
            if target is None:
                target = center_cr
            unit = _control_search_unit(
                approx,
                smoothed_for_direction,
                idx,
                target,
                search_direction_mode=search_direction_mode,
            )
            if unit is None:
                stage2_points.append(point.astype(np.float64))
                hit_flags.append(False)
                hit_sides.append("none")
                road_side_mask.append(False)
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
            candidate, hit, _step, side = _tighten_point_bidirectional_along_unit(
                point,
                unit,
                height_crop,
                max_steps_px=search_px,
            )
            if hit:
                compare_end = candidate
            elif side == "outward":
                compare_end = outward_direction_end
            else:
                compare_end = direction_end
            compare_segments.append(
                [
                    [float(point[0] + c0), float(point[1] + r0)],
                    [float(compare_end[0] + c0), float(compare_end[1] + r0)],
                ]
            )
            stage2_points.append(candidate)
            hit_flags.append(hit)
            hit_sides.append(side)
            road_side_mask.append(_point_hits_mask_along_ray(road_crop_bool, point, -unit, max_steps_px=road_probe_px))

        stage2 = np.asarray(stage2_points, dtype=np.float64)
        stage3, interpolated = _interpolate_missed_points(
            stage2,
            hit_flags,
            max_run_points=max_interpolate_run_points,
        )
        stage3 = _smooth_displacements(approx, stage3, mpp=mpp)
        stage3, angle_smoothed = _smooth_sharp_ring_points(stage3, threshold_deg=angle_smooth_threshold_deg)
        stage4 = _smooth_closed_ring_coordinates(stage3, iterations=2)
        stage4, final_angle_smoothed = _smooth_sharp_ring_points(stage4, threshold_deg=angle_smooth_threshold_deg)

        road_side_arr = np.asarray(road_side_mask, dtype=bool)
        road_side_arr = _fill_short_false_runs_closed(road_side_arr, max_gap_points=road_gap_close_points)
        road_side_arr = _remove_short_true_runs_closed(road_side_arr, min_run_points=road_min_run_points)
        pre_bridge_segments = _segments_from_closed_mask(road_side_arr)
        pre_bridge_segment_count = len(pre_bridge_segments)
        bridged_runs = 0
        bridged_points = 0
        if road_side_bridge_points > road_gap_close_points:
            road_side_arr, bridged_runs, bridged_points = _bridge_adjacent_segments_closed(
                road_side_arr,
                max_gap_points=road_side_bridge_points,
            )
        post_bridge_segment_count = len(_segments_from_closed_mask(road_side_arr))
        forced_bridge_runs = 0
        forced_bridge_points = 0
        preserved_gap_points = 0
        road_side_arr, forced_bridge_runs, forced_bridge_points, preserved_gap_points = _force_single_segment_closed(road_side_arr)
        road_side_segments = _segments_from_closed_mask(road_side_arr)

        raw_lines: list[np.ndarray] = []
        stage5_lines: list[np.ndarray] = []
        final_lines: list[np.ndarray] = []
        for segment_indices in road_side_segments:
            if len(segment_indices) < 2:
                continue
            raw_segment = approx[segment_indices]
            refined_segment = stage4[segment_indices]
            denoised_segment = _suppress_local_dents(
                refined_segment,
                mpp=mpp,
                sample_m=control_sample_m,
                width_m=local_dent_width_m,
                depth_m=local_dent_depth_m,
            )
            denoised_segment = _smooth_open_polyline(denoised_segment, iterations=road_side_smooth_iters)
            final_segment = _smooth_open_polyline(denoised_segment, iterations=final_smooth_iters)
            raw_lines.append(raw_segment)
            stage5_lines.append(denoised_segment)
            final_lines.append(final_segment)

        if not final_lines:
            continue

        offset = np.array([float(c0), float(r0)], dtype=np.float64)
        global_final_lines = [line + offset for line in final_lines]
        records.append(
            {
                "label_id": int(label_id),
                "area_px": int(np.count_nonzero(crop)),
                "area_m2": float(np.count_nonzero(crop) * mpp * mpp),
                "control_points": int(len(approx)),
                "stage2_hits": int(np.count_nonzero(hit_flags)),
                "inward_hits": int(sum(1 for side in hit_sides if side == "inward")),
                "outward_hits": int(sum(1 for side in hit_sides if side == "outward")),
                "interpolated_points": int(interpolated),
                "angle_smoothed_points": int(angle_smoothed),
                "final_angle_smoothed_points": int(final_angle_smoothed),
                "road_side_points": int(np.count_nonzero(road_side_arr)),
                "road_side_segments_before_bridge": int(pre_bridge_segment_count),
                "road_side_segments_after_bridge": int(post_bridge_segment_count),
                "road_side_bridge_runs": int(bridged_runs),
                "road_side_bridge_points": int(bridged_points),
                "road_side_force_single_bridge_runs": int(forced_bridge_runs),
                "road_side_force_single_bridge_points": int(forced_bridge_points),
                "road_side_preserved_gap_points": int(preserved_gap_points),
                "road_side_segments": int(len(global_final_lines)),
                "raw_ring_global": _closed(approx + offset),
                "stage2_ring_global": _closed(stage2 + offset),
                "stage3_ring_global": _closed(stage3 + offset),
                "stage4_ring_global": _closed(stage4 + offset),
                "road_side_raw_lines_global": [line + offset for line in raw_lines],
                "road_side_stage5_lines_global": [line + offset for line in stage5_lines],
                "road_side_final_lines_global": global_final_lines,
                "direction_segments_global": direction_segments,
                "outward_direction_segments_global": outward_direction_segments,
                "compare_segments_global": compare_segments,
                "xy_lines": [pixel_to_xy(line, meta) for line in global_final_lines],
            }
        )
    return records


def _write_sidewalk_boundary_shp(records: list[dict[str, Any]], out_base: Path) -> Path:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    writer = shapefile.Writer(str(out_base))
    writer.shapeType = shapefile.POLYLINE
    writer.field("id", "N", decimal=0)
    writer.field("label_id", "N", decimal=0)
    writer.field("seg_id", "N", decimal=0)
    writer.field("pts", "N", decimal=0)
    writer.field("area_m2", "F", decimal=3)

    feature_id = 0
    for rec in records:
        label_id = int(rec["label_id"])
        for seg_id, xy in enumerate(rec.get("xy_lines", [])):
            line = [[float(x), float(y)] for x, y in np.asarray(xy, dtype=np.float64)]
            if len(line) < 2:
                continue
            writer.line([line])
            writer.record(
                id=feature_id,
                label_id=label_id,
                seg_id=int(seg_id),
                pts=int(len(line)),
                area_m2=float(rec["area_m2"]),
            )
            feature_id += 1
    writer.close()
    return out_base.with_suffix(".shp")


def _write_envelope_polygon_shp(records: list[dict[str, Any]], out_base: Path) -> Path:
    writer = shapefile.Writer(str(out_base))
    writer.shapeType = shapefile.POLYGON
    writer.field("id", "N", decimal=0)
    writer.field("label_id", "N", decimal=0)
    writer.field("area_m2", "F", decimal=3)
    writer.field("ctrl_pts", "N", decimal=0)
    writer.field("moved", "N", decimal=0)
    writer.field("move_ratio", "F", decimal=3)
    for idx, rec in enumerate(records):
        ring = [[float(x), float(y)] for x, y in np.asarray(rec["xy"], dtype=np.float64)]
        writer.poly([ring])
        writer.record(
            id=idx,
            label_id=int(rec["label_id"]),
            area_m2=float(rec["area_m2"]),
            ctrl_pts=int(rec["control_points"]),
            moved=int(rec["moved_points"]),
            move_ratio=float(rec["move_ratio"]),
        )
    writer.close()
    return out_base.with_suffix(".shp")


def _write_envelope_boundary_shp(records: list[dict[str, Any]], out_base: Path) -> Path:
    writer = shapefile.Writer(str(out_base))
    writer.shapeType = shapefile.POLYLINE
    writer.field("id", "N", decimal=0)
    writer.field("label_id", "N", decimal=0)
    writer.field("ctrl_pts", "N", decimal=0)
    writer.field("moved", "N", decimal=0)
    writer.field("move_ratio", "F", decimal=3)
    for idx, rec in enumerate(records):
        line = [[float(x), float(y)] for x, y in np.asarray(rec["xy"], dtype=np.float64)]
        writer.line([line])
        writer.record(
            id=idx,
            label_id=int(rec["label_id"]),
            ctrl_pts=int(rec["control_points"]),
            moved=int(rec["moved_points"]),
            move_ratio=float(rec["move_ratio"]),
        )
    writer.close()
    return out_base.with_suffix(".shp")


def _render_boundary_on_height(records: list[dict[str, Any]], height_png: Path, out_path: Path) -> Path:
    canvas = np.asarray(Image.open(height_png).convert("RGB"), dtype=np.uint8).copy()
    for rec in records:
        pts = np.asarray(rec["pixel_ring_cr"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
    return _save_image_atomic(Image.fromarray(canvas), out_path)


def _render_envelope_compare_on_height(records: list[dict[str, Any]], height_png: Path, out_path: Path) -> Path:
    canvas = np.asarray(Image.open(height_png).convert("RGB"), dtype=np.uint8).copy()
    for rec in records:
        if "raw_pixel_ring_cr" in rec:
            raw_pts = np.asarray(rec["raw_pixel_ring_cr"], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [raw_pts], isClosed=True, color=(255, 180, 0), thickness=2)
        pts = np.asarray(rec["pixel_ring_cr"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
        for start, end in rec.get("move_segments_cr", []):
            p0 = tuple(int(round(v)) for v in start)
            p1 = tuple(int(round(v)) for v in end)
            cv2.line(canvas, p0, p1, color=(0, 255, 0), thickness=1)
            cv2.circle(canvas, p1, radius=2, color=(0, 255, 0), thickness=-1)
    return _save_image_atomic(Image.fromarray(canvas), out_path)


def _render_raw_controls_on_height(records: list[dict[str, Any]], height_png: Path, out_path: Path) -> Path:
    canvas = np.asarray(Image.open(height_png).convert("RGB"), dtype=np.uint8).copy()
    for rec in records:
        raw_pts = np.asarray(rec["raw_pixel_ring_cr"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [raw_pts], isClosed=True, color=(255, 180, 0), thickness=2)
        controls = np.asarray(rec.get("raw_control_points_cr", []), dtype=np.int32)
        hit_sides = rec.get("raw_control_hit_sides", [])
        for idx, (col, row) in enumerate(controls):
            side = hit_sides[idx] if idx < len(hit_sides) else "none"
            if side == "inward":
                color = (0, 255, 0)
            elif side == "outward":
                color = (255, 80, 0)
            else:
                color = (0, 0, 255)
            cv2.circle(canvas, (int(col), int(row)), radius=1, color=color, thickness=-1)
        for start, end in rec.get("direction_segments_cr", []):
            p0 = tuple(int(round(v)) for v in start)
            p1 = tuple(int(round(v)) for v in end)
            cv2.arrowedLine(canvas, p0, p1, color=(0, 255, 0), thickness=1, tipLength=0.25)
        for start, end in rec.get("outward_direction_segments_cr", []):
            p0 = tuple(int(round(v)) for v in start)
            p1 = tuple(int(round(v)) for v in end)
            cv2.arrowedLine(canvas, p0, p1, color=(255, 80, 0), thickness=1, tipLength=0.25)
    return _save_image_atomic(Image.fromarray(canvas), out_path)


def _render_stage2_compare_on_height(records: list[dict[str, Any]], height_png: Path, out_path: Path) -> Path:
    canvas = np.asarray(Image.open(height_png).convert("RGB"), dtype=np.uint8).copy()
    for rec in records:
        raw_ring = np.asarray(rec["raw_ring_global"], dtype=np.int32).reshape(-1, 1, 2)
        stage2_ring = np.asarray(rec["stage2_ring_global"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [raw_ring], isClosed=True, color=(255, 0, 0), thickness=2)
        cv2.polylines(canvas, [stage2_ring], isClosed=True, color=(0, 255, 0), thickness=2)
        for start, end in rec.get("compare_segments_global", []):
            p0 = tuple(int(round(v)) for v in start)
            p1 = tuple(int(round(v)) for v in end)
            cv2.line(canvas, p0, p1, color=(0, 0, 255), thickness=1)
        controls = np.asarray(rec.get("raw_ring_global", []), dtype=np.float64)
        if len(controls) > 1:
            controls = controls[:-1]
        for col, row in controls:
            cv2.circle(canvas, (int(round(float(col))), int(round(float(row)))), radius=1, color=(255, 0, 255), thickness=-1)
    return _save_image_atomic(Image.fromarray(canvas), out_path)


def _render_sidewalk_boundary_on_height(records: list[dict[str, Any]], height_png: Path, out_path: Path) -> Path:
    canvas = np.asarray(Image.open(height_png).convert("RGB"), dtype=np.uint8).copy()
    for rec in records:
        raw_ring = np.asarray(rec["raw_ring_global"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [raw_ring], isClosed=True, color=(255, 180, 0), thickness=2)
        for line in rec.get("road_side_stage5_lines_global", []):
            pts = np.asarray(line, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], isClosed=False, color=(255, 170, 0), thickness=2)
        for line in rec.get("road_side_final_lines_global", []):
            pts = np.asarray(line, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], isClosed=False, color=(0, 255, 255), thickness=2)
    return _save_image_with_crop(Image.fromarray(canvas), out_path)


def _render_sidewalk_debug_dirs_on_height(records: list[dict[str, Any]], height_png: Path, out_path: Path) -> Path:
    canvas = np.asarray(Image.open(height_png).convert("RGB"), dtype=np.uint8).copy()
    for rec in records:
        raw_ring = np.asarray(rec["raw_ring_global"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [raw_ring], isClosed=True, color=(255, 180, 0), thickness=2)
        for start, end in rec.get("direction_segments_global", []):
            p0 = tuple(int(round(v)) for v in start)
            p1 = tuple(int(round(v)) for v in end)
            cv2.arrowedLine(canvas, p0, p1, color=(0, 255, 0), thickness=1, tipLength=0.25)
        for start, end in rec.get("outward_direction_segments_global", []):
            p0 = tuple(int(round(v)) for v in start)
            p1 = tuple(int(round(v)) for v in end)
            cv2.arrowedLine(canvas, p0, p1, color=(255, 80, 0), thickness=1, tipLength=0.25)
    return _save_image_with_crop(Image.fromarray(canvas), out_path)


def _render_label_on_height(label_map: np.ndarray, height_png: Path, out_path: Path) -> Path:
    height = np.asarray(Image.open(height_png).convert("RGB"), dtype=np.float32)
    if tuple(height.shape[:2]) != tuple(label_map.shape):
        raise ValueError(f"height image shape {height.shape[:2]} does not match label_map {label_map.shape}")

    vis = height.copy()
    ids = [int(x) for x in np.unique(label_map) if int(x) >= 0]
    rng = np.random.default_rng(20260427)
    for label_id in ids:
        color = rng.integers(60, 256, size=3).astype(np.float32)
        mask = label_map == label_id
        vis[mask] = vis[mask] * 0.35 + color * 0.65
    return _save_image_with_crop(Image.fromarray(np.clip(vis, 0, 255).astype(np.uint8)), out_path)


def _render_refined_sidewalk_on_height(
    original_label_map: np.ndarray,
    green_refined_label_map: np.ndarray,
    refined_label_map: np.ndarray,
    height_png: Path,
    out_path: Path,
) -> Path:
    height = np.asarray(Image.open(height_png).convert("RGB"), dtype=np.float32)
    if tuple(height.shape[:2]) != tuple(original_label_map.shape):
        raise ValueError(f"height image shape {height.shape[:2]} does not match label_map {original_label_map.shape}")
    if tuple(original_label_map.shape) != tuple(green_refined_label_map.shape):
        raise ValueError(f"original label_map shape {original_label_map.shape} does not match green-refined label_map {green_refined_label_map.shape}")
    if tuple(original_label_map.shape) != tuple(refined_label_map.shape):
        raise ValueError(f"original label_map shape {original_label_map.shape} does not match refined label_map {refined_label_map.shape}")

    original_mask = np.asarray(original_label_map >= 0)
    green_refined_mask = np.asarray(green_refined_label_map >= 0)
    refined_mask = np.asarray(refined_label_map >= 0)
    green_added_mask = green_refined_mask & ~original_mask
    hole_added_mask = refined_mask & ~green_refined_mask

    vis = height.copy()
    vis[original_mask] = vis[original_mask] * 0.35 + np.array([60.0, 180.0, 255.0], dtype=np.float32) * 0.65
    vis[green_added_mask] = vis[green_added_mask] * 0.20 + np.array([0.0, 255.0, 0.0], dtype=np.float32) * 0.80
    vis[hole_added_mask] = vis[hole_added_mask] * 0.20 + np.array([255.0, 255.0, 0.0], dtype=np.float32) * 0.80
    return _save_image_with_crop(Image.fromarray(np.clip(vis, 0, 255).astype(np.uint8)), out_path)


def refine_sidewalk_with_green_veg(
    label_map: np.ndarray,
    green_veg_label_map: np.ndarray,
    *,
    meters_per_pixel: float,
    hull_inside_ratio: float = DEFAULT_GREEN_VEG_HULL_RATIO,
    green_buffer_m: float = DEFAULT_GREEN_VEG_BUFFER_M,
    min_buffer_overlap_ratio: float = DEFAULT_GREEN_VEG_MIN_BUFFER_OVERLAP_RATIO,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if tuple(label_map.shape) != tuple(green_veg_label_map.shape):
        raise ValueError(f"sidewalk label_map shape {label_map.shape} does not match green_veg label_map {green_veg_label_map.shape}")

    refined = np.asarray(label_map, dtype=np.int32).copy()
    green_ids, green_counts = np.unique(green_veg_label_map[green_veg_label_map >= 0], return_counts=True)
    green_areas = {int(label_id): int(count) for label_id, count in zip(green_ids, green_counts)}
    if not green_areas:
        return refined, []

    buffer_px = max(1, int(round(float(green_buffer_m) / float(meters_per_pixel))))
    buffer_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (buffer_px * 2 + 1, buffer_px * 2 + 1))
    candidates: list[dict[str, Any]] = []

    sidewalk_ids = [int(x) for x in np.unique(label_map) if int(x) >= 0]
    for sidewalk_id in sidewalk_ids:
        rows, cols = np.nonzero(label_map == sidewalk_id)
        if rows.size < 3:
            continue

        x0, x1 = int(cols.min()), int(cols.max()) + 1
        y0, y1 = int(rows.min()), int(rows.max()) + 1
        r0 = max(0, y0 - buffer_px - 2)
        r1 = min(label_map.shape[0], y1 + buffer_px + 2)
        c0 = max(0, x0 - buffer_px - 2)
        c1 = min(label_map.shape[1], x1 + buffer_px + 2)

        sidewalk_crop = np.asarray(label_map[r0:r1, c0:c1] == sidewalk_id, dtype=np.uint8)
        contours, _ = cv2.findContours(sidewalk_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull_mask = np.zeros(sidewalk_crop.shape, dtype=np.uint8)
        hull_points = []
        for contour in contours:
            if len(contour) >= 3:
                hull_points.append(contour)
        if not hull_points:
            continue
        hull = cv2.convexHull(np.vstack(hull_points))
        cv2.fillConvexPoly(hull_mask, hull, 1)

        green_crop = np.asarray(green_veg_label_map[r0:r1, c0:c1])
        candidate_green_ids = [int(x) for x in np.unique(green_crop[hull_mask > 0]) if int(x) >= 0]
        for green_id in candidate_green_ids:
            green_area = green_areas.get(green_id, 0)
            if green_area <= 0:
                continue
            green_in_crop = green_crop == green_id
            inside_hull_px = int(np.count_nonzero(green_in_crop & (hull_mask > 0)))
            hull_ratio = float(inside_hull_px / green_area)
            if hull_ratio < float(hull_inside_ratio):
                continue

            gr, gc = np.nonzero(green_in_crop)
            if gr.size == 0:
                continue
            gy0, gy1 = int(gr.min()), int(gr.max()) + 1
            gx0, gx1 = int(gc.min()), int(gc.max()) + 1
            br0 = max(0, gy0 - buffer_px)
            br1 = min(green_in_crop.shape[0], gy1 + buffer_px)
            bc0 = max(0, gx0 - buffer_px)
            bc1 = min(green_in_crop.shape[1], gx1 + buffer_px)
            green_local = np.asarray(green_in_crop[br0:br1, bc0:bc1], dtype=np.uint8)
            buffered_green = cv2.dilate(green_local, buffer_kernel, iterations=1) > 0
            sidewalk_local = sidewalk_crop[br0:br1, bc0:bc1] > 0
            overlap_px = int(np.count_nonzero(buffered_green & sidewalk_local))
            buffer_overlap_ratio = float(overlap_px / max(1, int(np.count_nonzero(buffered_green))))
            if buffer_overlap_ratio < float(min_buffer_overlap_ratio):
                continue

            candidates.append(
                {
                    "sidewalk_label_id": int(sidewalk_id),
                    "green_veg_label_id": int(green_id),
                    "green_area_px": int(green_area),
                    "inside_hull_px": int(inside_hull_px),
                    "hull_inside_ratio": float(hull_ratio),
                    "buffer_overlap_px": int(overlap_px),
                    "buffer_overlap_ratio": float(buffer_overlap_ratio),
                    "bbox_px": [int(c0), int(r0), int(c1 - c0), int(r1 - r0)],
                }
            )

    best_by_green: dict[int, dict[str, Any]] = {}
    for rec in candidates:
        green_id = int(rec["green_veg_label_id"])
        prev = best_by_green.get(green_id)
        if prev is None or (
            int(rec["buffer_overlap_px"]),
            float(rec["hull_inside_ratio"]),
        ) > (
            int(prev["buffer_overlap_px"]),
            float(prev["hull_inside_ratio"]),
        ):
            best_by_green[green_id] = rec

    records: list[dict[str, Any]] = []
    for rec in best_by_green.values():
        green_id = int(rec["green_veg_label_id"])
        sidewalk_id = int(rec["sidewalk_label_id"])
        write = (green_veg_label_map == green_id) & (refined < 0)
        added_px = int(np.count_nonzero(write))
        if added_px <= 0:
            continue
        refined[write] = sidewalk_id
        out_rec = dict(rec)
        out_rec["added_px"] = added_px
        out_rec["added_m2"] = float(added_px * meters_per_pixel * meters_per_pixel)
        records.append(out_rec)

    records.sort(key=lambda item: (int(item["sidewalk_label_id"]), int(item["green_veg_label_id"])))
    return refined, records


def fill_sidewalk_label_map_holes(
    label_map: np.ndarray,
    *,
    meters_per_pixel: float,
    max_hole_area_m2: float = DEFAULT_MAX_HOLE_AREA_M2,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    max_hole_area_px = max(1, int(round(float(max_hole_area_m2) / (float(meters_per_pixel) * float(meters_per_pixel)))))
    output = np.asarray(label_map, dtype=np.int32).copy()
    records: list[dict[str, Any]] = []

    ids = [int(x) for x in np.unique(label_map) if int(x) >= 0]
    for label_id in ids:
        rows, cols = np.nonzero(output == label_id)
        if rows.size < 3:
            continue

        x0, x1 = int(cols.min()), int(cols.max()) + 1
        y0, y1 = int(rows.min()), int(rows.max()) + 1
        pad = 2
        r0 = max(0, y0 - pad)
        r1 = min(output.shape[0], y1 + pad)
        c0 = max(0, x0 - pad)
        c1 = min(output.shape[1], x1 + pad)
        crop = np.asarray(output[r0:r1, c0:c1] == label_id, dtype=np.uint8)

        filled, kept_holes, hole_stats = _fill_small_internal_holes(
            crop,
            max_hole_area_px=max_hole_area_px,
            border_margin_px=1,
        )
        region = output[r0:r1, c0:c1]
        write = filled & (crop == 0) & (region < 0)
        added_px = int(np.count_nonzero(write))
        if added_px <= 0:
            continue

        region[write] = label_id
        records.append(
            {
                "label_id": int(label_id),
                "hole_added_pixels": added_px,
                "hole_added_m2": float(added_px * meters_per_pixel * meters_per_pixel),
                "candidate_hole_pixels": int(np.count_nonzero(kept_holes)),
                "max_hole_area_m2": float(max_hole_area_m2),
                "bbox_px": [int(c0), int(r0), int(c1 - c0), int(r1 - r0)],
                **hole_stats,
            }
        )

    return output, records


def refine_sidewalk_label_map(
    label_map: np.ndarray,
    *,
    meters_per_pixel: float,
    buffer_m: float = DEFAULT_BUFFER_M,
    close_m: float = DEFAULT_CLOSE_M,
    max_hole_area_m2: float = DEFAULT_MAX_HOLE_AREA_M2,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    buffer_px = max(1, int(round(buffer_m / meters_per_pixel)))
    close_px = max(1, int(round(close_m / meters_per_pixel)))
    max_hole_area_px = max(1, int(round(max_hole_area_m2 / (meters_per_pixel * meters_per_pixel))))
    border_margin_px = buffer_px + 2
    buffer_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (buffer_px * 2 + 1, buffer_px * 2 + 1))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px * 2 + 1, close_px * 2 + 1))

    output = np.full(label_map.shape, -1, dtype=np.int32)
    records: list[dict[str, Any]] = []
    ids = [int(x) for x in np.unique(label_map) if int(x) >= 0]
    for label_id in ids:
        rows, cols = np.nonzero(label_map == label_id)
        if rows.size < 3:
            continue
        x0, x1 = int(cols.min()), int(cols.max()) + 1
        y0, y1 = int(rows.min()), int(rows.max()) + 1
        pad = buffer_px + close_px + 4
        r0 = max(0, y0 - pad)
        r1 = min(label_map.shape[0], y1 + pad)
        c0 = max(0, x0 - pad)
        c1 = min(label_map.shape[1], x1 + pad)
        crop = np.asarray(label_map[r0:r1, c0:c1] == label_id, dtype=np.uint8)

        buffered = cv2.dilate(crop, buffer_kernel, iterations=1) > 0
        filled, kept_holes, hole_stats = _fill_small_internal_holes(
            buffered.astype(np.uint8),
            max_hole_area_px=max_hole_area_px,
            border_margin_px=border_margin_px,
        )
        closed = cv2.morphologyEx(filled.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel, iterations=1) > 0

        region = output[r0:r1, c0:c1]
        write = closed & (region < 0)
        region[write] = label_id

        records.append(
            {
                "label_id": int(label_id),
                "original_pixels": int(np.count_nonzero(crop)),
                "buffered_pixels": int(np.count_nonzero(buffered)),
                "filled_pixels": int(np.count_nonzero(filled)),
                "closed_pixels": int(np.count_nonzero(closed)),
                "buffer_added_pixels": int(np.count_nonzero(buffered & (crop == 0))),
                "hole_added_pixels": int(np.count_nonzero(kept_holes)),
                "close_added_pixels": int(np.count_nonzero(closed & ~filled)),
                **hole_stats,
                "bbox_px": [int(c0), int(r0), int(c1 - c0), int(r1 - r0)],
            }
        )
    return output, records


def run_sidewalk_v2(
    pre_part_dir: Path | str,
    output_dir: Path | str | None = None,
    *,
    label_map_path: Path | str | None = None,
    road_label_map_path: Path | str | None = None,
    green_veg_label_map_path: Path | str | None = None,
    buffer_m: float = DEFAULT_BUFFER_M,
    close_m: float = DEFAULT_CLOSE_M,
    max_hole_area_m2: float = DEFAULT_MAX_HOLE_AREA_M2,
    envelope_buffer_m: float = DEFAULT_ENVELOPE_BUFFER_M,
    envelope_core_m: float = DEFAULT_ENVELOPE_CORE_M,
    tighten_search_m: float = DEFAULT_TIGHTEN_SEARCH_M,
    control_sample_m: float = DEFAULT_CONTROL_SAMPLE_M,
    interpolate_gap_m: float = DEFAULT_INTERPOLATE_GAP_M,
    angle_smooth_threshold_deg: float = DEFAULT_ANGLE_SMOOTH_THRESHOLD_DEG,
    search_direction_mode: str = DEFAULT_SEARCH_DIRECTION_MODE,
    road_probe_m: float = DEFAULT_ROAD_PROBE_M,
    road_dilate_m: float = DEFAULT_ROAD_DILATE_M,
    road_gap_close_m: float = DEFAULT_ROAD_GAP_CLOSE_M,
    road_min_run_m: float = DEFAULT_ROAD_MIN_RUN_M,
    road_side_bridge_m: float = DEFAULT_ROAD_SIDE_BRIDGE_M,
    local_dent_width_m: float = DEFAULT_LOCAL_DENT_WIDTH_M,
    local_dent_depth_m: float = DEFAULT_LOCAL_DENT_DEPTH_M,
    road_side_smooth_iters: int = DEFAULT_ROAD_SIDE_SMOOTH_ITERS,
    final_smooth_iters: int = DEFAULT_FINAL_SMOOTH_ITERS,
    green_veg_hull_ratio: float = DEFAULT_GREEN_VEG_HULL_RATIO,
    green_veg_buffer_m: float = DEFAULT_GREEN_VEG_BUFFER_M,
    green_veg_min_buffer_overlap_ratio: float = DEFAULT_GREEN_VEG_MIN_BUFFER_OVERLAP_RATIO,
) -> dict[str, Path]:
    pre_part_dir = Path(pre_part_dir).expanduser()
    map_dir = _default_map_dir(pre_part_dir)
    output_dir = Path(output_dir).expanduser() if output_dir is not None else map_dir / "sidewalk_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    if label_map_path is None:
        label_map_path = map_dir / "objs" / "sidewalk" / "result" / "label_map.npy"
    if road_label_map_path is None:
        road_label_map_path = map_dir / "objs" / "road" / "result" / "label_map.npy"
    if green_veg_label_map_path is None:
        green_veg_label_map_path = map_dir / "objs" / "green_veg" / "result" / "label_map.npy"
    label_map_path = Path(label_map_path).expanduser()
    road_label_map_path = Path(road_label_map_path).expanduser()
    green_veg_label_map_path = Path(green_veg_label_map_path).expanduser()
    if not label_map_path.is_file():
        raise FileNotFoundError(label_map_path)
    if not road_label_map_path.is_file():
        raise FileNotFoundError(road_label_map_path)
    if green_veg_label_map_path is not None and not green_veg_label_map_path.is_file():
        green_veg_label_map_path = None

    assets = _resolve_height_assets(pre_part_dir)
    height_meta = _load_json(assets["height_meta"])
    mpp = float(height_meta["meters_per_pixel"])
    label_map = np.load(label_map_path, mmap_mode="r")
    road_label_map = np.load(road_label_map_path, mmap_mode="r")
    working_label_map = np.asarray(label_map, dtype=np.int32)
    green_refined_label_map = working_label_map
    refine_records: list[dict[str, Any]] = []
    hole_fill_records: list[dict[str, Any]] = []
    refine_out: Path | None = None
    if green_veg_label_map_path is not None:
        green_veg_label_map = np.load(green_veg_label_map_path, mmap_mode="r")
        green_refined_label_map, refine_records = refine_sidewalk_with_green_veg(
            label_map,
            green_veg_label_map,
            meters_per_pixel=mpp,
            hull_inside_ratio=green_veg_hull_ratio,
            green_buffer_m=green_veg_buffer_m,
            min_buffer_overlap_ratio=green_veg_min_buffer_overlap_ratio,
        )
        working_label_map, hole_fill_records = fill_sidewalk_label_map_holes(
            green_refined_label_map,
            meters_per_pixel=mpp,
            max_hole_area_m2=max_hole_area_m2,
        )
        refine_out = _render_refined_sidewalk_on_height(
            label_map,
            green_refined_label_map,
            working_label_map,
            assets["height_png"],
            debug_dir / "refine_sidewalk.png",
        )

    vis_out = output_dir / "label_map_on_height.png"
    boundary_out = output_dir / "sidewalk_boundary.shp"
    boundary_vis_out = output_dir / "sidewalk_boundary_on_height.png"
    debug_dirs_vis_out = debug_dir / "search_dirs_on_height.png"
    compare_vis_out = debug_dir / "compare.png"
    summary_out = output_dir / "summary.json"
    vis_out = _render_label_on_height(working_label_map, assets["height_png"], vis_out)

    height_values = np.load(assets["height_values"], mmap_mode="r")
    boundary_records = _extract_road_side_boundary_records(
        working_label_map,
        road_label_map,
        height_values=height_values,
        meta=height_meta,
        envelope_buffer_m=envelope_buffer_m,
        envelope_core_m=envelope_core_m,
        tighten_search_m=tighten_search_m,
        control_sample_m=control_sample_m,
        interpolate_gap_m=interpolate_gap_m,
        angle_smooth_threshold_deg=angle_smooth_threshold_deg,
        search_direction_mode=search_direction_mode,
        road_probe_m=road_probe_m,
        road_dilate_m=road_dilate_m,
        road_gap_close_m=road_gap_close_m,
        road_min_run_m=road_min_run_m,
        road_side_bridge_m=road_side_bridge_m,
        local_dent_width_m=local_dent_width_m,
        local_dent_depth_m=local_dent_depth_m,
        road_side_smooth_iters=road_side_smooth_iters,
        final_smooth_iters=final_smooth_iters,
    )
    boundary_out = _write_sidewalk_boundary_shp(boundary_records, output_dir / "sidewalk_boundary")
    boundary_vis_out = _render_sidewalk_boundary_on_height(boundary_records, assets["height_png"], boundary_vis_out)
    debug_dirs_vis_out = _render_sidewalk_debug_dirs_on_height(boundary_records, assets["height_png"], debug_dirs_vis_out)
    compare_vis_out = _render_stage2_compare_on_height(boundary_records, assets["height_png"], compare_vis_out)

    summary = {
        "pipeline": "road_side_simple_first_hit",
        "pre_part_dir": str(pre_part_dir),
        "label_map": str(label_map_path),
        "road_label_map": str(road_label_map_path),
        "green_veg_label_map": str(green_veg_label_map_path) if green_veg_label_map_path is not None else None,
        "height_png": str(assets["height_png"]),
        "height_values": str(assets["height_values"]),
        "height_meta": str(assets["height_meta"]),
        "output_dir": str(output_dir),
        "meters_per_pixel": mpp,
        "buffer_m": float(buffer_m),
        "buffer_px": int(round(buffer_m / mpp)),
        "close_m": float(close_m),
        "close_px": int(round(close_m / mpp)),
        "max_hole_area_m2": float(max_hole_area_m2),
        "envelope_buffer_m": float(envelope_buffer_m),
        "envelope_core_m": float(envelope_core_m),
        "tighten_search_m": float(tighten_search_m),
        "control_sample_m": float(control_sample_m),
        "interpolate_gap_m": float(interpolate_gap_m),
        "angle_smooth_threshold_deg": float(angle_smooth_threshold_deg),
        "search_direction_mode": search_direction_mode,
        "road_probe_m": float(road_probe_m),
        "road_dilate_m": float(road_dilate_m),
        "road_gap_close_m": float(road_gap_close_m),
        "road_min_run_m": float(road_min_run_m),
        "road_side_bridge_m": float(road_side_bridge_m),
        "local_dent_width_m": float(local_dent_width_m),
        "local_dent_depth_m": float(local_dent_depth_m),
        "road_side_smooth_iters": int(road_side_smooth_iters),
        "final_smooth_iters": int(final_smooth_iters),
        "green_veg_hull_ratio": float(green_veg_hull_ratio),
        "green_veg_buffer_m": float(green_veg_buffer_m),
        "green_veg_min_buffer_overlap_ratio": float(green_veg_min_buffer_overlap_ratio),
        "height_diff_min_m": HEIGHT_DIFF_MIN_M,
        "height_diff_max_m": HEIGHT_DIFF_MAX_M,
        "original_pixels": int(np.count_nonzero(label_map >= 0)),
        "refined_pixels": int(np.count_nonzero(working_label_map >= 0)),
        "green_veg_added_pixels": int(np.count_nonzero((green_refined_label_map >= 0) & (label_map < 0))),
        "hole_fill_added_pixels": int(np.count_nonzero((working_label_map >= 0) & (green_refined_label_map < 0))),
        "visualization_output": str(vis_out),
        "refine_sidewalk_visualization_output": str(refine_out) if refine_out is not None else None,
        "sidewalk_boundary_output": str(boundary_out),
        "sidewalk_boundary_visualization_output": str(boundary_vis_out),
        "debug_search_dirs_visualization_output": str(debug_dirs_vis_out),
        "debug_compare_visualization_output": str(compare_vis_out),
        "green_veg_refine_records": refine_records,
        "hole_fill_records": hole_fill_records,
        "records": [
            {
                key: value
                for key, value in rec.items()
                if key
                not in {
                    "raw_ring_global",
                    "stage2_ring_global",
                    "stage3_ring_global",
                    "stage4_ring_global",
                    "road_side_raw_lines_global",
                    "road_side_stage5_lines_global",
                    "road_side_final_lines_global",
                    "direction_segments_global",
                    "outward_direction_segments_global",
                    "compare_segments_global",
                    "xy_lines",
                }
            }
            for rec in boundary_records
        ],
    }
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs = {
        "visualization": vis_out,
        "sidewalk_boundary": boundary_out,
        "sidewalk_boundary_visualization": boundary_vis_out,
        "debug_search_dirs_visualization": debug_dirs_vis_out,
        "debug_compare_visualization": compare_vis_out,
        "summary": summary_out,
    }
    if refine_out is not None:
        outputs["refine_sidewalk_visualization"] = refine_out
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract road-side sidewalk boundary polylines from instance label maps.")
    parser.add_argument("pre_part_dir", help="pre-part output directory.")
    parser.add_argument("-o", "--output-dir", default=None, help="Output directory. Defaults to <map>/sidewalk_v2.")
    parser.add_argument("--label-map", default=None, help="Existing sidewalk label_map.npy.")
    parser.add_argument("--road-label-map", default=None, help="Existing road label_map.npy used to select the road-side boundary.")
    parser.add_argument("--green-veg-label-map", default=None, help="Existing green_veg label_map.npy used to refine sidewalk masks.")
    parser.add_argument("--buffer-m", type=float, default=DEFAULT_BUFFER_M, help="Deprecated compatibility option; not used by the current road-side flow.")
    parser.add_argument("--close-m", type=float, default=DEFAULT_CLOSE_M, help="Deprecated compatibility option; not used by the current road-side flow.")
    parser.add_argument("--max-hole-area-m2", type=float, default=DEFAULT_MAX_HOLE_AREA_M2, help="Maximum internal sidewalk hole area filled after green_veg refinement.")
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
    parser.add_argument("--road-probe-m", type=float, default=DEFAULT_ROAD_PROBE_M, help="Probe distance used to classify control points as road-side.")
    parser.add_argument("--road-dilate-m", type=float, default=DEFAULT_ROAD_DILATE_M, help="Road mask dilation before road-side classification.")
    parser.add_argument("--road-gap-close-m", type=float, default=DEFAULT_ROAD_GAP_CLOSE_M, help="Maximum short non-road-side gap to close along the boundary.")
    parser.add_argument("--road-min-run-m", type=float, default=DEFAULT_ROAD_MIN_RUN_M, help="Minimum road-side run length kept as an output segment.")
    parser.add_argument("--road-side-bridge-m", type=float, default=DEFAULT_ROAD_SIDE_BRIDGE_M, help="Maximum same-instance gap to bridge between adjacent road-side segments.")
    parser.add_argument("--local-dent-width-m", type=float, default=DEFAULT_LOCAL_DENT_WIDTH_M, help="Local window width for dent suppression.")
    parser.add_argument("--local-dent-depth-m", type=float, default=DEFAULT_LOCAL_DENT_DEPTH_M, help="Minimum local dent depth to project back to the local baseline.")
    parser.add_argument("--road-side-smooth-iters", type=int, default=DEFAULT_ROAD_SIDE_SMOOTH_ITERS, help="Initial open-polyline smoothing iterations after dent suppression.")
    parser.add_argument("--final-smooth-iters", type=int, default=DEFAULT_FINAL_SMOOTH_ITERS, help="Final mild open-polyline smoothing iterations.")
    parser.add_argument("--green-veg-hull-ratio", type=float, default=DEFAULT_GREEN_VEG_HULL_RATIO, help="Minimum green_veg instance ratio inside a sidewalk convex hull before it can refine that sidewalk.")
    parser.add_argument("--green-veg-buffer-m", type=float, default=DEFAULT_GREEN_VEG_BUFFER_M, help="green_veg buffer distance used before testing overlap with the sidewalk mask.")
    parser.add_argument("--green-veg-min-buffer-overlap-ratio", type=float, default=DEFAULT_GREEN_VEG_MIN_BUFFER_OVERLAP_RATIO, help="Minimum buffered green_veg overlap ratio with a sidewalk mask.")
    args = parser.parse_args()
    outputs = run_sidewalk_v2(
        args.pre_part_dir,
        args.output_dir,
        label_map_path=args.label_map,
        road_label_map_path=args.road_label_map,
        green_veg_label_map_path=args.green_veg_label_map,
        buffer_m=args.buffer_m,
        close_m=args.close_m,
        max_hole_area_m2=args.max_hole_area_m2,
        envelope_buffer_m=args.envelope_buffer_m,
        envelope_core_m=args.envelope_core_m,
        tighten_search_m=args.tighten_search_m,
        control_sample_m=args.control_sample_m,
        interpolate_gap_m=args.interpolate_gap_m,
        angle_smooth_threshold_deg=args.angle_smooth_threshold_deg,
        search_direction_mode=args.search_direction_mode,
        road_probe_m=args.road_probe_m,
        road_dilate_m=args.road_dilate_m,
        road_gap_close_m=args.road_gap_close_m,
        road_min_run_m=args.road_min_run_m,
        road_side_bridge_m=args.road_side_bridge_m,
        local_dent_width_m=args.local_dent_width_m,
        local_dent_depth_m=args.local_dent_depth_m,
        road_side_smooth_iters=args.road_side_smooth_iters,
        final_smooth_iters=args.final_smooth_iters,
        green_veg_hull_ratio=args.green_veg_hull_ratio,
        green_veg_buffer_m=args.green_veg_buffer_m,
        green_veg_min_buffer_overlap_ratio=args.green_veg_min_buffer_overlap_ratio,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
