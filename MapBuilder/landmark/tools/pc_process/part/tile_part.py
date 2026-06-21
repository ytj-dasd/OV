from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData
import rasterio


def build_default_geo_tile_payload(
    geotiff_path: Path | str,
    *,
    tile_size: list[int] | tuple[int, int] = (2048, 2048),
    overlap_values: list[float] | tuple[float, ...] = (200.0,),
    fill_ratio_threshold: float = 0.10,
) -> dict[str, Any]:
    geotiff_path = Path(geotiff_path).expanduser()
    if len(tile_size) != 2:
        raise ValueError("tile_size must have length 2")
    if not overlap_values:
        raise ValueError("overlap_values must not be empty")

    tile_w = int(tile_size[0])
    tile_h = int(tile_size[1])
    overlap = float(overlap_values[0])
    stride_x = max(1, int(round(tile_w * (1.0 - overlap))) if overlap <= 1.0 else tile_w - int(round(overlap)))
    stride_y = max(1, int(round(tile_h * (1.0 - overlap))) if overlap <= 1.0 else tile_h - int(round(overlap)))

    with rasterio.open(geotiff_path) as ds:
        arr = ds.read()
        width = int(ds.width)
        height = int(ds.height)
        transform = ds.transform

    if arr.ndim == 3:
        non_black = np.any(arr > 0, axis=0)
    else:
        non_black = arr > 0

    tiles: list[dict[str, Any]] = []
    for pixel_min_y in range(0, height, stride_y):
        for pixel_min_x in range(0, width, stride_x):
            pixel_max_x = min(pixel_min_x + tile_w, width)
            pixel_max_y = min(pixel_min_y + tile_h, height)
            if pixel_max_x <= pixel_min_x or pixel_max_y <= pixel_min_y:
                continue

            crop = non_black[pixel_min_y:pixel_max_y, pixel_min_x:pixel_max_x]
            fill_ratio = float(np.count_nonzero(crop)) / float(crop.size)
            if fill_ratio <= fill_ratio_threshold:
                continue

            left, top = transform * (pixel_min_x, pixel_min_y)
            right, bottom = transform * (pixel_max_x, pixel_max_y)
            tiles.append({
                "pixel_min_x": int(pixel_min_x),
                "pixel_min_y": int(pixel_min_y),
                "pixel_max_x": int(pixel_max_x),
                "pixel_max_y": int(pixel_max_y),
                "pixel_width": int(pixel_max_x - pixel_min_x),
                "pixel_height": int(pixel_max_y - pixel_min_y),
                "geo_left": float(left),
                "geo_top": float(top),
                "geo_right": float(right),
                "geo_bottom": float(bottom),
            })

    return {
        "source": str(geotiff_path),
        "tile_size": [tile_w, tile_h],
        "overlap": [overlap],
        "num_tiles": len(tiles),
        "tiles": tiles,
    }


def _part_vertices_from_bounds(left: float, right: float, bottom: float, top: float) -> list[list[float]]:
    left = round(float(left), 6)
    right = round(float(right), 6)
    bottom = round(float(bottom), 6)
    top = round(float(top), 6)
    return [
        [left, bottom],
        [right, bottom],
        [right, top],
        [left, top],
    ]


def _part_vertices_from_rotated_tile(
    origin_xy: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
    *,
    u0: float,
    v0: float,
    tile_size_m: float,
) -> list[list[float]]:
    corners = [
        origin_xy + axis_u * u0 + axis_v * v0,
        origin_xy + axis_u * (u0 + tile_size_m) + axis_v * v0,
        origin_xy + axis_u * (u0 + tile_size_m) + axis_v * (v0 + tile_size_m),
        origin_xy + axis_u * u0 + axis_v * (v0 + tile_size_m),
    ]
    return [[round(float(x), 6), round(float(y), 6)] for x, y in corners]


def _load_geo_meta(path: Path | str) -> dict[str, Any]:
    path = Path(path).expanduser()
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_binary_mask_image(path: Path | str) -> np.ndarray:
    path = Path(path).expanduser()
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = np.any(arr > 0, axis=2)
    elif arr.ndim == 2:
        arr = arr > 0
    else:
        raise ValueError(f"Unsupported mask image shape: {arr.shape}")
    return np.asarray(arr, dtype=bool)


def _world_xy_to_pixel(x: float, y: float, geo_meta: dict[str, Any]) -> tuple[int, int]:
    min_x, min_y = geo_meta["min_xy"]
    mpp = float(geo_meta["meters_per_pixel"])
    height = int(geo_meta["height"])
    px = int(round((float(x) - float(min_x)) / mpp))
    py = int(round((float(y) - float(min_y)) / mpp))
    row = int(height - 1 - py)
    return px, row


def _pixel_to_world_xy(col: np.ndarray, row: np.ndarray, geo_meta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    min_x, min_y = [float(v) for v in geo_meta["min_xy"]]
    mpp = float(geo_meta["meters_per_pixel"])
    height = int(geo_meta["height"])
    x = min_x + (col.astype(np.float64) + 0.5) * mpp
    y = min_y + (height - row.astype(np.float64) - 0.5) * mpp
    return x, y


def write_tile_parts_preview(
    parts_payload: dict[str, Any],
    mask_bev_path: Path | str,
    geo_meta_path: Path | str,
    output_path: Path | str,
) -> Path:
    mask_bev_path = Path(mask_bev_path).expanduser()
    output_path = Path(output_path).expanduser()
    geo_meta = _load_geo_meta(geo_meta_path)

    img = Image.open(mask_bev_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size

    for part in parts_payload.get("parts", []):
        vertices_xy = part.get("vertices_xy")
        if not isinstance(vertices_xy, list) or len(vertices_xy) != 4:
            continue
        polygon_px: list[tuple[int, int]] = []
        for x, y in vertices_xy:
            px, row = _world_xy_to_pixel(float(x), float(y), geo_meta)
            px = max(0, min(width - 1, px))
            row = max(0, min(height - 1, row))
            polygon_px.append((px, row))
        if len(polygon_px) < 4:
            continue
        polygon_closed = [*polygon_px, polygon_px[0]]
        draw.line(polygon_closed, fill=(255, 128, 0), width=2)
        label_x = int(round(sum(p[0] for p in polygon_px) / len(polygon_px)))
        label_y = int(round(sum(p[1] for p in polygon_px) / len(polygon_px)))
        draw.text((label_x, label_y), str(part.get("part_id", "")), fill=(0, 255, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return output_path


def _world_bounds_to_pixel_window(
    *,
    left: float,
    right: float,
    bottom: float,
    top: float,
    geo_meta: dict[str, Any],
) -> tuple[int, int, int, int]:
    min_x, min_y = [float(v) for v in geo_meta["min_xy"]]
    mpp = float(geo_meta["meters_per_pixel"])
    width = int(geo_meta["width"])
    height = int(geo_meta["height"])
    world_top_exclusive = float(min_y + height * mpp)

    pixel_min_x = int(math.floor((left - min_x) / mpp))
    pixel_max_x = int(math.ceil((right - min_x) / mpp))
    pixel_min_y = int(math.floor((world_top_exclusive - top) / mpp))
    pixel_max_y = int(math.ceil((world_top_exclusive - bottom) / mpp))

    pixel_min_x = max(0, min(width, pixel_min_x))
    pixel_max_x = max(0, min(width, pixel_max_x))
    pixel_min_y = max(0, min(height, pixel_min_y))
    pixel_max_y = max(0, min(height, pixel_max_y))
    return pixel_min_x, pixel_max_x, pixel_min_y, pixel_max_y


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
    return float(angle)


def _points_in_convex_polygon(points_xy: np.ndarray, polygon_xy: np.ndarray) -> np.ndarray:
    polygon = np.asarray(polygon_xy, dtype=np.float64)
    points = np.asarray(points_xy, dtype=np.float64)
    signs: list[np.ndarray] = []
    for i in range(len(polygon)):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % len(polygon)]
        edge = p1 - p0
        rel = points - p0
        cross = edge[0] * rel[:, 1] - edge[1] * rel[:, 0]
        signs.append(cross)
    stacked = np.stack(signs, axis=1)
    eps = 1e-9
    return np.all(stacked >= -eps, axis=1) | np.all(stacked <= eps, axis=1)


def _rotated_tile_fill_stats(
    *,
    vertices_xy: np.ndarray,
    mask: np.ndarray,
    geo_meta: dict[str, Any],
) -> tuple[float, int]:
    min_x = float(np.min(vertices_xy[:, 0]))
    max_x = float(np.max(vertices_xy[:, 0]))
    min_y = float(np.min(vertices_xy[:, 1]))
    max_y = float(np.max(vertices_xy[:, 1]))
    pixel_min_x, pixel_max_x, pixel_min_y, pixel_max_y = _world_bounds_to_pixel_window(
        left=min_x,
        right=max_x,
        bottom=min_y,
        top=max_y,
        geo_meta=geo_meta,
    )
    if pixel_max_x <= pixel_min_x or pixel_max_y <= pixel_min_y:
        return 0.0, 0
    rows = np.arange(pixel_min_y, pixel_max_y)
    cols = np.arange(pixel_min_x, pixel_max_x)
    grid_cols, grid_rows = np.meshgrid(cols, rows)
    xs, ys = _pixel_to_world_xy(grid_cols, grid_rows, geo_meta)
    pts = np.column_stack([xs.reshape(-1), ys.reshape(-1)])
    inside = _points_in_convex_polygon(pts, vertices_xy).reshape(grid_rows.shape)
    if not np.any(inside):
        return 0.0, 0
    crop = mask[pixel_min_y:pixel_max_y, pixel_min_x:pixel_max_x]
    inside_count = int(np.count_nonzero(inside))
    num_mask_pixels = int(np.count_nonzero(np.logical_and(crop, inside)))
    return float(num_mask_pixels) / float(inside_count), num_mask_pixels


def build_tile_parts_payload(
    ply_path: Path | str,
    *,
    tile_size_m: float,
    fill_ratio_threshold: float = 0.10,
    fill_cell_size_m: float = 0.50,
    origin_xy: tuple[float, float] | None = None,
    mask_bev_path: Path | str,
    geo_meta_path: Path | str,
) -> dict[str, Any]:
    if tile_size_m <= 0:
        raise ValueError("tile_size_m must be > 0")
    if fill_cell_size_m <= 0:
        raise ValueError("fill_cell_size_m must be > 0")
    if not (0.0 <= fill_ratio_threshold <= 1.0):
        raise ValueError("fill_ratio_threshold must be in [0,1]")

    mask_bev_path = Path(mask_bev_path).expanduser()
    geo_meta = _load_geo_meta(geo_meta_path)
    mask = _load_binary_mask_image(mask_bev_path)
    if mask.shape != (int(geo_meta["height"]), int(geo_meta["width"])):
        raise ValueError(
            "mask_bev shape does not match geo_meta size: "
            f"mask={mask.shape}, meta=({geo_meta['height']}, {geo_meta['width']})"
        )

    min_xy = np.asarray(geo_meta["min_xy"], dtype=np.float64)
    mpp = float(geo_meta["meters_per_pixel"])
    width = int(geo_meta["width"])
    height = int(geo_meta["height"])
    world_max_x = float(min_xy[0] + width * mpp)
    world_max_y = float(min_xy[1] + height * mpp)
    origin = np.asarray(origin_xy if origin_xy is not None else min_xy, dtype=np.float64)
    rotation_rad = _estimate_mask_principal_angle(mask, geo_meta)
    cos_a = math.cos(rotation_rad)
    sin_a = math.sin(rotation_rad)
    axis_u = np.asarray([cos_a, sin_a], dtype=np.float64)
    axis_v = np.asarray([-sin_a, cos_a], dtype=np.float64)

    world_corners = np.asarray(
        [
            [float(min_xy[0]), float(min_xy[1])],
            [float(world_max_x), float(min_xy[1])],
            [float(world_max_x), float(world_max_y)],
            [float(min_xy[0]), float(world_max_y)],
        ],
        dtype=np.float64,
    )
    rel_corners = world_corners - origin
    u_coords = rel_corners @ axis_u
    v_coords = rel_corners @ axis_v
    min_u = float(np.min(u_coords))
    max_u = float(np.max(u_coords))
    min_v = float(np.min(v_coords))
    max_v = float(np.max(v_coords))

    num_cols = max(1, int(math.ceil((max_u - min_u) / tile_size_m)))
    num_rows = max(1, int(math.ceil((max_v - min_v) / tile_size_m)))

    parts: list[dict[str, Any]] = []
    part_id = 1
    for row in range(num_rows):
        v0 = float(min_v + row * tile_size_m)
        for col in range(num_cols):
            u0 = float(min_u + col * tile_size_m)
            vertices_xy = np.asarray(
                _part_vertices_from_rotated_tile(
                    origin,
                    axis_u,
                    axis_v,
                    u0=u0,
                    v0=v0,
                    tile_size_m=tile_size_m,
                ),
                dtype=np.float64,
            )
            fill_ratio, num_mask_pixels = _rotated_tile_fill_stats(
                vertices_xy=vertices_xy,
                mask=mask,
                geo_meta=geo_meta,
            )
            if fill_ratio < fill_ratio_threshold:
                continue

            left = float(np.min(vertices_xy[:, 0]))
            right = float(np.max(vertices_xy[:, 0]))
            bottom = float(np.min(vertices_xy[:, 1]))
            top = float(np.max(vertices_xy[:, 1]))
            parts.append({
                "part_id": part_id,
                "grid_row": row,
                "grid_col": col,
                "tile_size_m": float(tile_size_m),
                "fill_ratio": fill_ratio,
                "fill_cell_size_m": float(fill_cell_size_m),
                "fill_ratio_mode": "mask_pixels",
                "num_mask_pixels": int(num_mask_pixels),
                "num_points": None,
                "geo_left": left,
                "geo_right": right,
                "geo_bottom": bottom,
                "geo_top": top,
                "vertices_xy": [[float(x), float(y)] for x, y in vertices_xy],
            })
            part_id += 1

    return {
        "source_ply": str(Path(ply_path).expanduser()),
        "tile_size_m": float(tile_size_m),
        "fill_ratio_threshold": float(fill_ratio_threshold),
        "fill_cell_size_m": float(fill_cell_size_m),
        "origin_xy": [float(origin[0]), float(origin[1])],
        "grid_rotation_deg": float(math.degrees(rotation_rad)),
        "num_parts": len(parts),
        "parts": parts,
    }


def write_tile_parts_json(
    ply_path: Path | str,
    output_json_path: Path | str,
    *,
    tile_size_m: float,
    fill_ratio_threshold: float = 0.10,
    fill_cell_size_m: float = 0.50,
    origin_xy: tuple[float, float] | None = None,
    mask_bev_path: Path | str,
    geo_meta_path: Path | str,
    preview_path: Path | str | None = None,
) -> dict[str, Any]:
    payload = build_tile_parts_payload(
        ply_path,
        tile_size_m=tile_size_m,
        fill_ratio_threshold=fill_ratio_threshold,
        fill_cell_size_m=fill_cell_size_m,
        origin_xy=origin_xy,
        mask_bev_path=mask_bev_path,
        geo_meta_path=geo_meta_path,
    )
    output_json_path = Path(output_json_path).expanduser()
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    preview_target = (
        Path(preview_path).expanduser()
        if preview_path is not None
        else output_json_path.with_name(f"{output_json_path.stem}_preview.png")
    )
    write_tile_parts_preview(payload, mask_bev_path, geo_meta_path, preview_target)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate grid-based parts.json from a PLY file.")
    parser.add_argument("ply_path", help="Input PLY path.")
    parser.add_argument("output_json_path", help="Output parts.json path.")
    parser.add_argument("--tile-size", type=float, required=True, help="Square tile size in meters.")
    parser.add_argument("--fill-threshold", type=float, default=0.10, help="Minimum fill ratio to keep a part.")
    parser.add_argument("--fill-cell-size", type=float, default=0.50, help="Fill-ratio occupancy cell size in meters.")
    parser.add_argument("--origin-x", type=float, default=None, help="Optional grid origin x.")
    parser.add_argument("--origin-y", type=float, default=None, help="Optional grid origin y.")
    parser.add_argument("--mask-bev", required=True, help="pre-part mask BEV PNG path used for pixel-ratio filtering.")
    parser.add_argument("--geo-meta", required=True, help="pre-part geo_meta.json path used for pixel-ratio filtering.")
    parser.add_argument("--preview", default=None, help="Optional parts preview PNG path.")
    args = parser.parse_args()

    origin_xy = None
    if args.origin_x is not None or args.origin_y is not None:
        if args.origin_x is None or args.origin_y is None:
            raise ValueError("origin-x and origin-y must be provided together")
        origin_xy = (float(args.origin_x), float(args.origin_y))

    payload = write_tile_parts_json(
        args.ply_path,
        args.output_json_path,
        tile_size_m=args.tile_size,
        fill_ratio_threshold=args.fill_threshold,
        fill_cell_size_m=args.fill_cell_size,
        origin_xy=origin_xy,
        mask_bev_path=args.mask_bev,
        geo_meta_path=args.geo_meta,
        preview_path=args.preview,
    )
    print(f"Wrote {payload['num_parts']} parts to {Path(args.output_json_path).expanduser()}")


if __name__ == "__main__":
    main()
