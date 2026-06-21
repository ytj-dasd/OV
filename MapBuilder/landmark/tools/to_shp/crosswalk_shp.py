"""Crosswalk mask and zebra-line vectorization helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import shapefile
from PIL import Image
from shapely.geometry import MultiPolygon, Polygon

from landmark.tools.to_shp.geometry import pixel_to_xy


def _coerce_bev_meta(meta_or_path: dict[str, Any] | Path | str) -> dict[str, Any]:
    if isinstance(meta_or_path, dict):
        meta = dict(meta_or_path)
    else:
        path = Path(meta_or_path).expanduser()
        with path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    if "min_xy" in meta and "max_xy" in meta:
        return meta
    if "canvas_size" in meta and "global_origin_xy" in meta:
        canvas_w, canvas_h = meta["canvas_size"]
        mpp = float(meta["meters_per_pixel"])
        g_min_x, g_max_y = meta["global_origin_xy"]
        min_y = g_max_y - (canvas_h - 1) * mpp
        return {
            "width": int(canvas_w),
            "height": int(canvas_h),
            "meters_per_pixel": mpp,
            "min_xy": [float(g_min_x), float(min_y)],
            "max_xy": [float(g_min_x + (canvas_w - 1) * mpp), float(g_max_y)],
        }
    raise KeyError("Expected bev_meta with min_xy/max_xy or summary with canvas_size/global_origin_xy")


def _normalize_masks(masks: np.ndarray) -> np.ndarray:
    arr = np.asarray(masks)
    if arr.ndim == 3:
        return arr.astype(bool, copy=False)
    if arr.ndim == 2:
        return arr.astype(np.int32, copy=False)
    raise ValueError(f"masks must have shape (K,H,W) or (H,W), got {arr.shape}")


def _iter_object_crops(masks: np.ndarray) -> list[tuple[int, np.ndarray, int, int, int, int]]:
    arr = np.asarray(masks)
    results: list[tuple[int, np.ndarray, int, int, int, int]] = []
    if arr.ndim == 3:
        for oid in range(arr.shape[0]):
            mask = arr[oid].astype(bool)
            rows, cols = np.where(mask)
            if rows.size == 0:
                continue
            r0, r1 = int(rows.min()), int(rows.max()) + 1
            c0, c1 = int(cols.min()), int(cols.max()) + 1
            crop = mask[r0:r1, c0:c1]
            results.append((oid, crop, r0, r1, c0, c1))
        return results

    if arr.ndim != 2:
        raise ValueError(f"masks must have shape (K,H,W) or (H,W), got {arr.shape}")

    ids = np.unique(arr)
    ids = ids[ids >= 0]
    for oid in ids:
        rows, cols = np.where(arr == oid)
        if rows.size == 0:
            continue
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        c0, c1 = int(cols.min()), int(cols.max()) + 1
        crop = arr[r0:r1, c0:c1] == oid
        results.append((int(oid), crop, r0, r1, c0, c1))
    return results


def _bbox_from_crop(
    crop: np.ndarray,
    r0: int,
    c0: int,
    bev_meta: dict[str, Any],
    obj_id: int,
) -> dict[str, Any] | None:
    ys, xs = np.where(crop)
    if ys.size < 10:
        return None
    pts_pix = np.column_stack([xs + c0, ys + r0]).astype(np.float32)
    rect = cv2.minAreaRect(pts_pix)
    box_pix = cv2.boxPoints(rect).astype(np.float32)
    corners_xy = pixel_to_xy(box_pix, bev_meta).astype(np.float32)

    center_xy = ((corners_xy[0] + corners_xy[2]) * 0.5).astype(np.float32)
    d01 = float(np.linalg.norm(corners_xy[1] - corners_xy[0]))
    d12 = float(np.linalg.norm(corners_xy[2] - corners_xy[1]))
    if d01 >= d12:
        v = corners_xy[1] - corners_xy[0]
        size_lw = (d01, d12)
    else:
        v = corners_xy[2] - corners_xy[1]
        size_lw = (d12, d01)
    yaw = float(np.arctan2(float(v[1]), float(v[0])))
    return {
        "id": int(obj_id),
        "center": [float(center_xy[0]), float(center_xy[1]), 0.0],
        "yaw": yaw,
        "size": [float(size_lw[0]), float(size_lw[1])],
        "corners_xy": corners_xy.tolist(),
        "pixel_corners": box_pix.tolist(),
    }


def extract_crosswalk_bboxes(
    masks: np.ndarray,
    bev_meta: dict[str, Any] | Path | str,
) -> list[dict[str, Any]]:
    bev_meta_d = _coerce_bev_meta(bev_meta)
    masks_bool = _normalize_masks(masks)
    bboxes: list[dict[str, Any]] = []
    for oid, crop, r0, r1, c0, c1 in _iter_object_crops(masks_bool):
        bbox = _bbox_from_crop(crop, r0, c0, bev_meta_d, oid)
        if bbox is not None:
            bboxes.append(bbox)
    return bboxes


def _corners_to_closed_ring(corners_xy: list[list[float]]) -> list[list[float]]:
    ring = [list(c[:2]) for c in corners_xy]
    ring.append(ring[0])
    return ring


def _bbox_contains_point(corners_xy: list[list[float]], px: float, py: float) -> bool:
    n = len(corners_xy)
    signs: list[float] = []
    for i in range(n):
        x1, y1 = corners_xy[i][:2]
        x2, y2 = corners_xy[(i + 1) % n][:2]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        signs.append(cross)
    return all(s >= 0 for s in signs) or all(s <= 0 for s in signs)


def _bbox_center_in_obb(inner_bbox: dict[str, Any], outer_bbox: dict[str, Any]) -> bool:
    cx, cy = inner_bbox["center"][0], inner_bbox["center"][1]
    return _bbox_contains_point(outer_bbox["corners_xy"], cx, cy)


def _bbox_to_polygon(bbox: dict[str, Any]) -> Polygon:
    corners = [(float(c[0]), float(c[1])) for c in bbox["corners_xy"]]
    if corners[0] != corners[-1]:
        corners.append(corners[0])
    poly = Polygon(corners)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def _canonicalize_direction(dx: float, dy: float) -> tuple[float, float]:
    norm = math.hypot(dx, dy)
    if norm == 0:
        return (1.0, 0.0)
    dx /= norm
    dy /= norm
    if dx < 0 or (abs(dx) < 1e-9 and dy < 0):
        dx = -dx
        dy = -dy
    return (dx, dy)


def _polygon_major_direction(poly: Polygon) -> tuple[float, float]:
    rect = poly.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)
    best_dx, best_dy, best_len = 1.0, 0.0, -1.0
    for i in range(4):
        dx = coords[i + 1][0] - coords[i][0]
        dy = coords[i + 1][1] - coords[i][1]
        edge_len = math.hypot(dx, dy)
        if edge_len > best_len:
            best_len = edge_len
            best_dx, best_dy = dx, dy
    return _canonicalize_direction(best_dx, best_dy)


def _polygon_major_length(poly: Polygon) -> float:
    rect = poly.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)
    return max(
        math.hypot(coords[i + 1][0] - coords[i][0], coords[i + 1][1] - coords[i][1])
        for i in range(4)
    )


def _average_directions(directions: list[tuple[float, float]]) -> tuple[float, float]:
    if not directions:
        return (1.0, 0.0)
    ref_x, ref_y = directions[0]
    sum_x, sum_y = 0.0, 0.0
    for dx, dy in directions:
        if dx * ref_x + dy * ref_y < 0:
            dx, dy = -dx, -dy
        sum_x += dx
        sum_y += dy
    return _canonicalize_direction(sum_x, sum_y)


def _modal_length(lengths: list[float], bin_width: float = 0.05) -> tuple[float, float, float]:
    if not lengths:
        return (0.0, 0.0, bin_width)
    min_l = min(lengths)
    counts: dict[int, list[float]] = {}
    for length in lengths:
        idx = int((float(length) - min_l) / bin_width)
        counts.setdefault(idx, []).append(float(length))
    best_idx = max(counts, key=lambda k: (len(counts[k]), k))
    values = counts[best_idx]
    avg = sum(values) / len(values)
    return (avg, min_l + (best_idx + 0.5) * bin_width, bin_width)


def _polygon_short_edge_midpoints(
    poly: Polygon, axis_dir: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    rect = poly.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)
    edges = []
    for i in range(4):
        mx = (coords[i][0] + coords[i + 1][0]) / 2.0
        my = (coords[i][1] + coords[i + 1][1]) / 2.0
        el = math.hypot(coords[i + 1][0] - coords[i][0], coords[i + 1][1] - coords[i][1])
        edges.append((el, (mx, my)))
    edges.sort(key=lambda e: e[0])
    pts = [edges[0][1], edges[1][1]]
    ax, ay = axis_dir
    if pts[0][0] * ax + pts[0][1] * ay > pts[1][0] * ax + pts[1][1] * ay:
        pts.reverse()
    return pts[0], pts[1]


def _fit_axis_as_fn_of_normal(
    points: list[tuple[float, float]],
    axis_dir: tuple[float, float],
    normal_dir: tuple[float, float],
) -> tuple[float, float]:
    if not points:
        return (0.0, 0.0)
    ax, ay = axis_dir
    nx, ny = normal_dir
    normals = [x * nx + y * ny for x, y in points]
    axes = [x * ax + y * ay for x, y in points]
    if len(points) == 1:
        return (0.0, axes[0])
    mean_n = sum(normals) / len(normals)
    mean_a = sum(axes) / len(axes)
    num = sum((n - mean_n) * (a - mean_a) for n, a in zip(normals, axes))
    den = sum((n - mean_n) ** 2 for n in normals)
    normal_range = max(normals) - min(normals)
    if den < 1e-4 * (normal_range ** 2 + 1e-6):
        slope = 0.0
    else:
        slope = max(-2.0, min(2.0, num / den))
    intercept = mean_a - slope * mean_n
    return (slope, intercept)


def build_crosswalk_region(
    stripe_polys: list[Polygon],
) -> tuple[Polygon, list[list[float]], float, float, float, int]:
    directions = [_polygon_major_direction(p) for p in stripe_polys]
    lengths = [_polygon_major_length(p) for p in stripe_polys]
    axis_x, axis_y = _average_directions(directions)
    normal_x, normal_y = -axis_y, axis_x

    point_normals: list[float] = []
    for p in stripe_polys:
        point_normals.extend(
            float(c[0]) * normal_x + float(c[1]) * normal_y
            for c in p.exterior.coords[:-1]
        )

    base_length, mode_center, mode_bw = _modal_length(lengths)
    min_normal = min(point_normals)
    max_normal = max(point_normals)

    half_bw = mode_bw / 2.0
    left_mids: list[tuple[float, float]] = []
    right_mids: list[tuple[float, float]] = []
    for p, ln in zip(stripe_polys, lengths):
        if abs(float(ln) - mode_center) > half_bw:
            continue
        lm, rm = _polygon_short_edge_midpoints(p, (axis_x, axis_y))
        left_mids.append(lm)
        right_mids.append(rm)

    if not left_mids or not right_mids:
        for p in stripe_polys:
            lm, rm = _polygon_short_edge_midpoints(p, (axis_x, axis_y))
            left_mids.append(lm)
            right_mids.append(rm)

    ls, li = _fit_axis_as_fn_of_normal(left_mids, (axis_x, axis_y), (normal_x, normal_y))
    rs, ri = _fit_axis_as_fn_of_normal(right_mids, (axis_x, axis_y), (normal_x, normal_y))

    corners = [
        (axis_x * (ls * min_normal + li) + normal_x * min_normal,
         axis_y * (ls * min_normal + li) + normal_y * min_normal),
        (axis_x * (ls * max_normal + li) + normal_x * max_normal,
         axis_y * (ls * max_normal + li) + normal_y * max_normal),
        (axis_x * (rs * max_normal + ri) + normal_x * max_normal,
         axis_y * (rs * max_normal + ri) + normal_y * max_normal),
        (axis_x * (rs * min_normal + ri) + normal_x * min_normal,
         axis_y * (rs * min_normal + ri) + normal_y * min_normal),
    ]
    ring = [[float(x), float(y)] for x, y in corners]
    ring.append(ring[0])
    poly = Polygon(ring)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
        ring = [[float(x), float(y)] for x, y in poly.exterior.coords]

    yaw = math.atan2(axis_y, axis_x)
    return poly, ring, yaw, base_length, max_normal - min_normal, len(stripe_polys)


def _render_masks_overview(masks: np.ndarray, output_path: Path) -> Path:
    arr = np.asarray(masks)
    if arr.ndim == 3:
        rgb = np.zeros((arr.shape[1], arr.shape[2], 3), dtype=np.uint8)
        rng = np.random.default_rng(42)
        for idx, mask in enumerate(arr):
            rgb[mask.astype(bool)] = rng.integers(60, 256, size=3, dtype=np.uint8)
            if idx >= 254:
                break
    elif arr.ndim == 2:
        rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
        ids = np.unique(arr)
        ids = ids[ids >= 0]
        rng = np.random.default_rng(42)
        for oid in ids[:255]:
            rgb[arr == oid] = rng.integers(60, 256, size=3, dtype=np.uint8)
    else:
        raise ValueError(f"masks must have shape (K,H,W) or (H,W), got {arr.shape}")
    Image.fromarray(rgb).save(output_path)
    return output_path


def _polygon_stats(poly: Polygon) -> tuple[float, float, float, tuple[float, float]]:
    rect = poly.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)
    lengths = [
        math.hypot(coords[i + 1][0] - coords[i][0], coords[i + 1][1] - coords[i][1])
        for i in range(4)
    ]
    if lengths[0] >= lengths[1]:
        dx = coords[1][0] - coords[0][0]
        dy = coords[1][1] - coords[0][1]
        length_m, width_m = lengths[0], lengths[1]
    else:
        dx = coords[2][0] - coords[1][0]
        dy = coords[2][1] - coords[1][1]
        length_m, width_m = lengths[1], lengths[0]
    center = rect.centroid
    return length_m, width_m, math.atan2(dy, dx), (center.x, center.y)


def _load_laneline_candidates(
    laneline_shp_path: Path | str,
    *,
    short_length_threshold_m: float,
    linearity_threshold: float,
) -> list[dict[str, Any]]:
    sf = shapefile.Reader(str(Path(laneline_shp_path).expanduser()))
    fields = [f[0] for f in sf.fields[1:]]
    candidates: list[dict[str, Any]] = []
    for idx, sr in enumerate(sf.iterShapeRecords()):
        row = dict(zip(fields, sr.record))
        pts = sr.shape.points
        if len(pts) < 4:
            continue
        poly = Polygon(pts)
        if poly.is_empty:
            continue
        length_m, width_m, yaw, center = _polygon_stats(poly)
        if width_m <= 1e-6:
            continue
        if length_m > short_length_threshold_m:
            continue
        if length_m / width_m < linearity_threshold:
            continue
        corners = list(poly.minimum_rotated_rectangle.exterior.coords)[:4]
        candidates.append({
            "id": int(row.get("id", idx)),
            "center": [float(center[0]), float(center[1]), 0.0],
            "yaw": float(yaw),
            "size": [float(length_m), float(width_m)],
            "corners_xy": [[float(x), float(y)] for x, y in corners],
        })
    return candidates


def crosswalk_masks_to_shp(
    masks: np.ndarray,
    geo_meta: dict[str, Any] | Path | str,
    laneline_shp_path: Path | str,
    output_dir: Path | str,
    *,
    short_length_threshold_m: float = 10.0,
    linearity_threshold: float = 3.0,
) -> Path:
    bev_meta = _coerce_bev_meta(geo_meta)
    masks_bool = _normalize_masks(masks)
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    crosswalk_bboxes = extract_crosswalk_bboxes(masks_bool, bev_meta)
    with (output_dir / "crosswalk_bboxes.json").open("w", encoding="utf-8") as f:
        json.dump({"bev_meta": bev_meta, "bboxes": crosswalk_bboxes}, f, ensure_ascii=False, indent=2)
    _render_masks_overview(masks_bool, output_dir / "crosswalk_masks.png")

    laneline_candidates = _load_laneline_candidates(
        laneline_shp_path,
        short_length_threshold_m=short_length_threshold_m,
        linearity_threshold=linearity_threshold,
    )

    cw_parts: list[tuple[dict[str, Any], int]] = []
    for b in laneline_candidates:
        for cw in crosswalk_bboxes:
            if _bbox_center_in_obb(b, cw):
                cw_parts.append((b, cw["id"]))
                break

    shp_path = output_dir / "crosswalk"
    w = shapefile.Writer(str(shp_path))
    w.shapeType = shapefile.POLYGON
    w.field("category", "C", size=20)
    w.field("id", "N", decimal=0)
    w.field("cw_id", "N", decimal=0)
    w.field("length", "F", decimal=3)
    w.field("width", "F", decimal=3)
    w.field("yaw", "F", decimal=6)
    w.field("stripe_num", "N", decimal=0)

    grouped: dict[int, list[tuple[dict[str, Any], int]]] = {}
    for b, cw_id in cw_parts:
        grouped.setdefault(cw_id, []).append((b, cw_id))

    region_count = 0
    for cw_id in sorted(grouped):
        stripe_bboxes = [b for b, _ in grouped[cw_id]]
        stripe_polys = [_bbox_to_polygon(b) for b in stripe_bboxes]
        valid_polys = [p for p in stripe_polys if not p.is_empty and isinstance(p, Polygon)]
        if len(valid_polys) >= 2:
            try:
                poly, ring, yaw, para_len, para_wid, sc = build_crosswalk_region(valid_polys)
                if not poly.is_empty:
                    w.poly([ring])
                    w.record(
                        category="crosswalk",
                        id=region_count,
                        cw_id=cw_id,
                        length=para_len,
                        width=para_wid,
                        yaw=yaw,
                        stripe_num=sc,
                    )
                    region_count += 1
            except Exception as e:
                print(f"[crosswalk] region build failed for cw_id={cw_id}: {e}", flush=True)

    stripe_count = 0
    for b, cw_id in cw_parts:
        ring = _corners_to_closed_ring(b["corners_xy"])
        w.poly([ring])
        w.record(
            category="zebra-line",
            id=b["id"],
            cw_id=cw_id,
            length=b["size"][0],
            width=b["size"][1],
            yaw=b["yaw"],
            stripe_num=0,
        )
        stripe_count += 1

    w.close()
    print(
        f"[crosswalk] wrote {region_count} crosswalk regions + {stripe_count} zebra-lines "
        f"→ {shp_path}.shp",
        flush=True,
    )
    return Path(f"{shp_path}.shp")
