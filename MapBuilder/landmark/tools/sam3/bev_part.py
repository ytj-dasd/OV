from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


Image.MAX_IMAGE_PIXELS = None
FIXED_TILE_SIZE = 1008
DEFAULT_TILE_OVERLAP_RATIO = 0.10


def _load_geo_meta(path: Path | str) -> dict[str, Any]:
    path = Path(path).expanduser()
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_bev_image(path: Path | str) -> np.ndarray:
    path = Path(path).expanduser()
    image = Image.open(path)
    return np.asarray(image)


def _valid_mask_from_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image != 0
    if image.ndim == 3:
        return np.any(image != 0, axis=2)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _pixel_to_world_xy(col: np.ndarray, row: np.ndarray, geo_meta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    min_x, min_y = [float(v) for v in geo_meta["min_xy"]]
    mpp = float(geo_meta["meters_per_pixel"])
    height = int(geo_meta["height"])
    x = min_x + (col.astype(np.float64) + 0.5) * mpp
    y = min_y + (height - row.astype(np.float64) - 0.5) * mpp
    return x, y


def _estimate_mask_principal_angle(mask: np.ndarray, geo_meta: dict[str, Any]) -> float:
    rows, cols = np.where(mask)
    if rows.size < 2:
        return 0.0

    xs, ys = _pixel_to_world_xy(cols, rows, geo_meta)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    centered = pts - np.mean(pts, axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    if eigvals[0] <= 1e-12:
        return 0.0
    if eigvals.shape[0] > 1 and float(eigvals[1] / eigvals[0]) >= 0.98:
        return 0.0

    main_vec = eigvecs[:, 0]
    angle = math.atan2(float(main_vec[1]), float(main_vec[0]))
    if angle < 0:
        angle += math.pi
    if angle >= math.pi / 2:
        angle -= math.pi / 2
    return float(math.degrees(angle))


def _principal_axis_to_image_rotation(principal_angle_deg: float) -> float:
    """Convert a road-axis angle into the image correction rotation.

    ``principal_angle_deg`` matches ``tile_part.py`` semantics: angle of the
    road/grid axis relative to world X in ``[0, 90)``. For image rectification we
    want the road direction to become vertical in the rotated image, so we apply
    only the residual correction to 90 degrees instead of rotating by the axis
    angle itself.
    """
    return float(90.0 - float(principal_angle_deg))


def _rotate_image_with_bounds(image: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos_a = abs(matrix[0, 0])
    sin_a = abs(matrix[0, 1])
    bound_w = int(math.ceil(height * sin_a + width * cos_a))
    bound_h = int(math.ceil(height * cos_a + width * sin_a))
    matrix[0, 2] += bound_w / 2.0 - center[0]
    matrix[1, 2] += bound_h / 2.0 - center[1]
    border_value = 0 if image.ndim == 2 else (0,) * int(image.shape[2])
    rotated = cv2.warpAffine(
        image,
        matrix,
        (bound_w, bound_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return rotated, matrix


def _invert_affine(matrix: np.ndarray) -> np.ndarray:
    return cv2.invertAffineTransform(np.asarray(matrix, dtype=np.float64))


def _pixel_to_world_corner_xy(col: np.ndarray, row: np.ndarray, geo_meta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    min_x, _min_y = [float(v) for v in geo_meta["min_xy"]]
    _max_x, max_y = [float(v) for v in geo_meta["max_xy"]]
    mpp = float(geo_meta["meters_per_pixel"])
    x = min_x + col.astype(np.float64) * mpp
    y = max_y - row.astype(np.float64) * mpp
    return x, y


def _tile_dst_corners() -> np.ndarray:
    return np.asarray(
        [
            [0.0, FIXED_TILE_SIZE - 1.0],
            [FIXED_TILE_SIZE - 1.0, FIXED_TILE_SIZE - 1.0],
            [FIXED_TILE_SIZE - 1.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )


def _rotated_tile_corners(left: int, top: int, tile_size_px: int) -> np.ndarray:
    right = left + tile_size_px - 1
    bottom = top + tile_size_px - 1
    return np.asarray(
        [
            [float(left), float(bottom)],
            [float(right), float(bottom)],
            [float(right), float(top)],
            [float(left), float(top)],
        ],
        dtype=np.float32,
    )


def _tile_fill_ratio(valid_mask: np.ndarray, left: int, top: int, tile_size_px: int) -> float:
    crop = valid_mask[top : top + tile_size_px, left : left + tile_size_px]
    if crop.size == 0:
        return 0.0
    return float(np.count_nonzero(crop)) / float(crop.size)


def _warp_rotated_tile(image: np.ndarray, rotated_corners_xy: np.ndarray) -> np.ndarray:
    src = np.asarray(rotated_corners_xy, dtype=np.float32)
    dst = _tile_dst_corners()
    matrix = cv2.getPerspectiveTransform(src, dst)
    border_value = 0 if image.ndim == 2 else (0,) * int(image.shape[2])
    return cv2.warpPerspective(
        np.asarray(image),
        matrix,
        (FIXED_TILE_SIZE, FIXED_TILE_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def _render_rotated_grid_debug(
    rotated_image: np.ndarray,
    *,
    row_starts: list[int],
    col_starts: list[int],
    tile_size_px: int,
) -> np.ndarray:
    if rotated_image.ndim == 2:
        canvas = cv2.cvtColor(rotated_image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        canvas = np.asarray(rotated_image)[..., :3].copy()

    h, w = canvas.shape[:2]
    color = (160, 90, 0)
    thickness = 2
    for top in row_starts:
        for left in col_starts:
            right = min(w - 1, left + tile_size_px - 1)
            bottom = min(h - 1, top + tile_size_px - 1)
            cv2.rectangle(
                canvas,
                (int(left), int(top)),
                (int(right), int(bottom)),
                color,
                thickness=thickness,
            )
    return canvas


def _grid_starts(total_size_px: int, tile_size_px: int, stride_px: int) -> list[int]:
    if total_size_px <= tile_size_px:
        return [0]
    starts = list(range(0, total_size_px - tile_size_px + 1, stride_px))
    last_start = total_size_px - tile_size_px
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _longest_true_run(row_mask: np.ndarray) -> tuple[int, int, int]:
    cols = np.flatnonzero(row_mask)
    if cols.size == 0:
        return -1, -1, 0
    best_start = int(cols[0])
    best_end = int(cols[0])
    best_len = 1
    cur_start = int(cols[0])
    cur_prev = int(cols[0])
    for col in cols[1:]:
        col = int(col)
        if col == cur_prev + 1:
            cur_prev = col
            cur_len = cur_prev - cur_start + 1
            if cur_len > best_len:
                best_start = cur_start
                best_end = cur_prev
                best_len = cur_len
            continue
        cur_start = col
        cur_prev = col
    return best_start, best_end, best_len


def _estimate_stable_width_group(
    rotated_mask: np.ndarray,
    *,
    mpp: float,
) -> dict[str, float]:
    slice_h_px = max(1, int(round(1.0 / mpp)))
    widths: list[float] = []
    centers: list[float] = []
    top_rows: list[int] = []
    bottom_rows: list[int] = []
    for top in range(0, int(rotated_mask.shape[0]), slice_h_px):
        bottom = min(int(rotated_mask.shape[0]), top + slice_h_px)
        start, end, width = _longest_true_run(np.any(rotated_mask[top:bottom], axis=0))
        if width <= 0:
            continue
        widths.append(float(width))
        centers.append((float(start) + float(end)) / 2.0)
        top_rows.append(top)
        bottom_rows.append(bottom)

    if not widths:
        return {
            "road_width_px": float(rotated_mask.shape[1]),
            "road_center_x_px": float(rotated_mask.shape[1] - 1) / 2.0,
            "group_row_start": 0.0,
            "group_row_end": 0.0,
            "slice_height_px": float(slice_h_px),
        }

    widths_arr = np.asarray(widths, dtype=np.float64)
    centers_arr = np.asarray(centers, dtype=np.float64)
    tops_arr = np.asarray(top_rows, dtype=np.int32)
    bottoms_arr = np.asarray(bottom_rows, dtype=np.int32)
    target_width = float(np.median(widths_arr))
    tol = max(5.0, target_width * 0.10)
    candidate = np.abs(widths_arr - target_width) <= tol
    if not np.any(candidate):
        candidate = np.ones_like(widths_arr, dtype=bool)

    best_start = None
    best_end = None
    cur_start = None
    cur_prev = None
    for idx, top in enumerate(tops_arr):
        if not candidate[idx]:
            cur_start = None
            cur_prev = None
            continue
        if cur_start is None:
            cur_start = int(idx)
            cur_prev = int(idx)
        elif int(idx) == cur_prev + 1:
            cur_prev = int(idx)
        else:
            if best_start is None or (cur_prev - cur_start) > (best_end - best_start):
                best_start, best_end = cur_start, cur_prev
            cur_start = int(idx)
            cur_prev = int(idx)
    if cur_start is not None and (best_start is None or (cur_prev - cur_start) > (best_end - best_start)):
        best_start, best_end = cur_start, cur_prev

    if best_start is None or best_end is None:
        group_mask = candidate
        group_row_start = int(tops_arr[0])
        group_row_end = int(bottoms_arr[-1])
    else:
        idxs = np.arange(len(candidate))
        group_mask = candidate & (idxs >= best_start) & (idxs <= best_end)
        group_row_start = int(tops_arr[best_start])
        group_row_end = int(bottoms_arr[best_end])

    if not np.any(group_mask):
        group_mask = candidate

    road_width_px = float(np.median(widths_arr[group_mask]))
    road_center_x_px = float(np.median(centers_arr[group_mask]))
    return {
        "road_width_px": road_width_px,
        "road_center_x_px": road_center_x_px,
        "group_row_start": float(group_row_start),
        "group_row_end": float(group_row_end),
        "slice_height_px": float(slice_h_px),
    }


def _coverage_width_for_n(tile_size_px: int, stride_px: int, n_tiles: int) -> int:
    return int(tile_size_px + max(0, n_tiles - 1) * stride_px)


def _estimate_column_offset(
    rotated_mask: np.ndarray,
    tile_size_px: int,
    stride_px: int,
    *,
    mpp: float,
) -> dict[str, float]:
    section = _estimate_stable_width_group(rotated_mask, mpp=mpp)
    road_width_px = float(section["road_width_px"])
    road_center_x_px = float(section["road_center_x_px"])
    n_side_tiles = 0
    n_tiles = 2 * n_side_tiles + 1
    while _coverage_width_for_n(tile_size_px, stride_px, n_tiles) < road_width_px:
        n_side_tiles += 1
        n_tiles = 2 * n_side_tiles + 1
    coverage_width_px = float(_coverage_width_for_n(tile_size_px, stride_px, n_tiles))
    desired_left = road_center_x_px - (coverage_width_px - 1.0) / 2.0
    if stride_px <= 0:
        offset_px = desired_left
    else:
        offset_px = float(desired_left % stride_px)
    return {
        **section,
        "orthogonal_grid_side_tiles": float(n_side_tiles),
        "orthogonal_grid_tiles": float(n_tiles),
        "orthogonal_grid_coverage_px": coverage_width_px,
        "orthogonal_offset_px": offset_px,
    }


def _grid_starts_with_offset(
    total_size_px: int,
    tile_size_px: int,
    stride_px: int,
    offset_px: float,
) -> list[int]:
    if total_size_px <= tile_size_px:
        return [0]
    last_start = total_size_px - tile_size_px
    base = int(round(offset_px))
    starts_set: set[int] = set()
    k = 0
    while True:
        start = base + k * stride_px
        if start > last_start:
            break
        if start >= 0:
            starts_set.add(start)
        k += 1
    k = 1
    while True:
        start = base - k * stride_px
        if start < 0:
            break
        starts_set.add(start)
        k += 1
    starts_set.add(last_start)
    return sorted(starts_set)


def build_bev_parts_payload(
    bev_path: Path | str,
    geo_meta_path: Path | str,
    *,
    spt_road_path: Path | str | None = None,
    tile_size_m: float,
    fill_ratio_threshold: float = 0.10,
    tile_overlap_ratio: float = DEFAULT_TILE_OVERLAP_RATIO,
) -> tuple[dict[str, Any], np.ndarray]:
    if tile_size_m <= 0:
        raise ValueError("tile_size_m must be > 0")
    if not (0.0 <= fill_ratio_threshold <= 1.0):
        raise ValueError("fill_ratio_threshold must be in [0,1]")
    if not (0.0 <= tile_overlap_ratio < 1.0):
        raise ValueError("tile_overlap_ratio must be in [0,1)")

    bev_path = Path(bev_path).expanduser()
    geo_meta_path = Path(geo_meta_path).expanduser()
    geo_meta = _load_geo_meta(geo_meta_path)
    image = _load_bev_image(bev_path)
    valid_mask = _valid_mask_from_image(image)
    spt_road_image = _load_bev_image(spt_road_path) if spt_road_path is not None else None
    direction_mask = _valid_mask_from_image(spt_road_image) if spt_road_image is not None else valid_mask

    mpp = float(geo_meta["meters_per_pixel"])
    tile_size_px = max(1, int(round(tile_size_m / mpp)))
    stride_px = max(1, int(round(tile_size_px * (1.0 - tile_overlap_ratio))))
    principal_angle_deg = _estimate_mask_principal_angle(direction_mask, geo_meta)
    rotation_deg = _principal_axis_to_image_rotation(principal_angle_deg)
    rotated_image, orig_to_rot = _rotate_image_with_bounds(image, rotation_deg)
    rotated_mask, _ = _rotate_image_with_bounds(valid_mask.astype(np.uint8) * 255, rotation_deg)
    rotated_valid_mask = rotated_mask > 0
    rotated_direction_mask = rotated_valid_mask
    if spt_road_image is not None:
        rotated_direction, _ = _rotate_image_with_bounds(direction_mask.astype(np.uint8) * 255, rotation_deg)
        rotated_direction_mask = rotated_direction > 0
    rot_to_orig = _invert_affine(orig_to_rot)

    rot_h, rot_w = rotated_valid_mask.shape
    offset_info = _estimate_column_offset(rotated_valid_mask, tile_size_px, stride_px, mpp=mpp)
    col_starts = _grid_starts_with_offset(
        rot_w,
        tile_size_px,
        stride_px,
        offset_info["orthogonal_offset_px"],
    )
    row_starts = _grid_starts(rot_h, tile_size_px, stride_px)

    parts: list[dict[str, Any]] = []
    part_id = 1
    for row, top in enumerate(row_starts):
        for col, left in enumerate(col_starts):
            fill_ratio = _tile_fill_ratio(rotated_valid_mask, left, top, tile_size_px)
            if fill_ratio < fill_ratio_threshold:
                continue

            rotated_corners = _rotated_tile_corners(left, top, tile_size_px)
            ones = np.ones((4, 1), dtype=np.float64)
            rotated_homo = np.hstack([rotated_corners.astype(np.float64), ones])
            orig_corners = (rotated_homo @ rot_to_orig.T).astype(np.float64)
            world_x, world_y = _pixel_to_world_corner_xy(
                orig_corners[:, 0],
                orig_corners[:, 1],
                geo_meta,
            )
            corners_xy = np.column_stack([world_x, world_y]).tolist()

            parts.append(
                {
                    "part_id": part_id,
                    "tile_name": f"part_{part_id:03d}",
                    "grid_row": row,
                    "grid_col": col,
                    "fill_ratio": float(fill_ratio),
                    "rotated_pixel_corners_xy": rotated_corners.tolist(),
                    "original_pixel_corners_xy": orig_corners.tolist(),
                    "corners_xy": corners_xy,
                    "pixel_corners_xy": _tile_dst_corners().tolist(),
                    "bev_path": None,
                }
            )
            part_id += 1

    payload = {
        "source_bev": str(bev_path),
        "source_geo_meta": str(geo_meta_path),
        "source_spt_road": str(Path(spt_road_path).expanduser()) if spt_road_path is not None else None,
        "original_shape": [int(image.shape[0]), int(image.shape[1])],
        "rotated_shape": [int(rotated_image.shape[0]), int(rotated_image.shape[1])],
        "principal_axis_deg": float(principal_angle_deg),
        "rotation_deg": float(rotation_deg),
        "original_to_rotated_affine": orig_to_rot.tolist(),
        "rotated_to_original_affine": rot_to_orig.tolist(),
        "meters_per_pixel": mpp,
        "tile_size_m": float(tile_size_m),
        "tile_size_px_rotated": int(tile_size_px),
        "tile_overlap_ratio": float(tile_overlap_ratio),
        "tile_stride_px_rotated": int(stride_px),
        "row_starts_px": [int(v) for v in row_starts],
        "col_starts_px": [int(v) for v in col_starts],
        "orthogonal_offset_px": float(offset_info["orthogonal_offset_px"]),
        "road_width_px": float(offset_info["road_width_px"]),
        "road_center_x_px": float(offset_info["road_center_x_px"]),
        "width_slice_height_px": int(offset_info["slice_height_px"]),
        "width_slice_height_m": float(offset_info["slice_height_px"] * mpp),
        "orthogonal_grid_side_tiles": int(offset_info["orthogonal_grid_side_tiles"]),
        "orthogonal_grid_tiles": int(offset_info["orthogonal_grid_tiles"]),
        "orthogonal_grid_coverage_px": float(offset_info["orthogonal_grid_coverage_px"]),
        "road_width_group_row_range": [
            int(offset_info["group_row_start"]),
            int(offset_info["group_row_end"]),
        ],
        "fill_ratio_threshold": float(fill_ratio_threshold),
        "num_parts": len(parts),
        "parts": parts,
    }
    return payload, rotated_image


def write_bev_parts(
    bev_path: Path | str,
    geo_meta_path: Path | str,
    output_dir: Path | str,
    *,
    spt_road_path: Path | str | None = None,
    tile_size_m: float,
    fill_ratio_threshold: float = 0.10,
    tile_overlap_ratio: float = DEFAULT_TILE_OVERLAP_RATIO,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser()
    parts_dir = output_dir / "parts"
    bev_out_dir = parts_dir / "bev"
    bev_out_dir.mkdir(parents=True, exist_ok=True)

    payload, rotated_image = build_bev_parts_payload(
        bev_path,
        geo_meta_path,
        spt_road_path=spt_road_path,
        tile_size_m=tile_size_m,
        fill_ratio_threshold=fill_ratio_threshold,
        tile_overlap_ratio=tile_overlap_ratio,
    )
    Image.fromarray(np.asarray(rotated_image)).save(parts_dir / "rotated_bev.png")
    grid_debug = _render_rotated_grid_debug(
        rotated_image,
        row_starts=[int(v) for v in payload["row_starts_px"]],
        col_starts=[int(v) for v in payload["col_starts_px"]],
        tile_size_px=int(payload["tile_size_px_rotated"]),
    )
    Image.fromarray(grid_debug).save(parts_dir / "rotated_bev_grid.png")

    for part in payload["parts"]:
        part_name = part["tile_name"]
        tile = _warp_rotated_tile(rotated_image, np.asarray(part["rotated_pixel_corners_xy"], dtype=np.float32))
        tile_path = bev_out_dir / f"{part_name}.png"
        Image.fromarray(np.asarray(tile)).save(tile_path)
        part["bev_path"] = str(tile_path)

    with (parts_dir / "parts.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a BEV image into rotated SAM3 tiles.")
    parser.add_argument("bev_path", help="Input BEV image path.")
    parser.add_argument("geo_meta_path", help="Input geo_meta.json path.")
    parser.add_argument("--spt-road", dest="spt_road_path", help="Optional spt-road.png used only for rotation estimation.")
    parser.add_argument("--tile-size", type=float, required=True, dest="tile_size_m", help="Tile size in meters.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument(
        "--fill-threshold",
        type=float,
        default=0.10,
        dest="fill_ratio_threshold",
        help="Minimum valid-pixel ratio for keeping a tile.",
    )
    parser.add_argument(
        "--tile-overlap",
        type=float,
        default=DEFAULT_TILE_OVERLAP_RATIO,
        dest="tile_overlap_ratio",
        help="Neighboring-tile overlap ratio in rotated-image pixels.",
    )
    args = parser.parse_args()
    write_bev_parts(
        args.bev_path,
        args.geo_meta_path,
        args.out,
        spt_road_path=args.spt_road_path,
        tile_size_m=args.tile_size_m,
        fill_ratio_threshold=args.fill_ratio_threshold,
        tile_overlap_ratio=args.tile_overlap_ratio,
    )


if __name__ == "__main__":
    main()
