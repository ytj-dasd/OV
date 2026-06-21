"""Laneline vectorization app."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import shapefile
from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon

from landmark.tools.to_shp.box_shp import box_masks_to_shp
from landmark.tools.to_shp.long_laneline_shp import (
    _chain_components,
    _connected_components_sorted,
    _skeletonize_mask,
    _smooth_polyline,
)
from landmark.tools.to_shp.polygon_centerline import polygon_shp_to_centerline_shp
from landmark.tools.to_shp.wide_curve_shp import wide_curve_masks_to_shp


def _coerce_bev_meta(meta_or_path: Path | str) -> dict:
    import json

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


def _polyline_metrics(mask: np.ndarray, mpp: float, *, smooth_m: float) -> dict[str, float | None]:
    try:
        skel = _skeletonize_mask(mask)
        if not np.any(skel):
            return {"polyline_length_px": None, "chord_length_px": None, "path_ratio": None, "turn_sum_deg": None, "max_turn_deg": None}
        components = _connected_components_sorted(skel)
        if not components:
            return {"polyline_length_px": None, "chord_length_px": None, "path_ratio": None, "turn_sum_deg": None, "max_turn_deg": None}
        polyline = _chain_components(components)
        polyline = np.asarray(polyline, dtype=np.float64)
        if len(polyline) < 2:
            return {"polyline_length_px": None, "chord_length_px": None, "path_ratio": None, "turn_sum_deg": None, "max_turn_deg": None}

        smooth_sigma_px = max(0.0, float(smooth_m) / float(mpp))
        polyline = _smooth_polyline(polyline, smooth_sigma_px=smooth_sigma_px)
        if len(polyline) < 2:
            return {"polyline_length_px": None, "chord_length_px": None, "path_ratio": None, "turn_sum_deg": None, "max_turn_deg": None}

        segs = np.diff(polyline, axis=0)
        seg_lens = np.linalg.norm(segs, axis=1)
        valid = seg_lens > 1e-6
        if not np.any(valid):
            return {"polyline_length_px": None, "chord_length_px": None, "path_ratio": None, "turn_sum_deg": None, "max_turn_deg": None}
        segs = segs[valid]
        seg_lens = seg_lens[valid]

        polyline_length_px = float(seg_lens.sum())
        chord_length_px = float(np.linalg.norm(polyline[-1] - polyline[0]))
        path_ratio = float(polyline_length_px / max(chord_length_px, 1e-6))
        if chord_length_px > 1e-6:
            chord_vec = polyline[-1] - polyline[0]
            rel = polyline - polyline[0]
            cross = np.abs(chord_vec[0] * rel[:, 1] - chord_vec[1] * rel[:, 0])
            deviations = cross / chord_length_px
            max_dev_px = float(np.max(deviations))
            mean_dev_px = float(np.mean(deviations))
            max_dev_ratio = float(max_dev_px / chord_length_px)
            mean_dev_ratio = float(mean_dev_px / chord_length_px)
        else:
            max_dev_px = None
            mean_dev_px = None
            max_dev_ratio = None
            mean_dev_ratio = None

        unit = segs / seg_lens[:, None]
        if len(unit) < 2:
            turn_sum_deg = 0.0
            max_turn_deg = 0.0
        else:
            dots = np.sum(unit[:-1] * unit[1:], axis=1)
            angles = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
            turn_sum_deg = float(np.sum(np.abs(angles)))
            max_turn_deg = float(np.max(np.abs(angles)))
        return {
            "polyline_length_px": polyline_length_px,
            "chord_length_px": chord_length_px,
            "path_ratio": path_ratio,
            "turn_sum_deg": turn_sum_deg,
            "max_turn_deg": max_turn_deg,
            "max_dev_px": max_dev_px,
            "mean_dev_px": mean_dev_px,
            "max_dev_ratio": max_dev_ratio,
            "mean_dev_ratio": mean_dev_ratio,
        }
    except Exception:
        return {
            "polyline_length_px": None,
            "chord_length_px": None,
            "path_ratio": None,
            "turn_sum_deg": None,
            "max_turn_deg": None,
            "max_dev_px": None,
            "mean_dev_px": None,
            "max_dev_ratio": None,
            "mean_dev_ratio": None,
        }


def _analyze_laneline_mask(
    mask: np.ndarray,
    bev_meta: dict,
    *,
    short_threshold_m: float,
    curve_path_ratio_threshold: float,
    curve_turn_sum_deg_threshold: float,
    curve_deviation_ratio_threshold: float,
    curve_centerline_smooth_m: float,
) -> dict[str, float | bool | str | None]:
    mpp = float(bev_meta["meters_per_pixel"])
    ys, xs = np.where(mask)
    if ys.size < 10:
        return {
            "length_m": None,
            "polyline_length_px": None,
            "chord_length_px": None,
            "path_ratio": None,
            "turn_sum_deg": None,
            "max_turn_deg": None,
            "max_dev_px": None,
            "mean_dev_px": None,
            "max_dev_ratio": None,
            "mean_dev_ratio": None,
            "is_curve": False,
            "route": "skip_small",
        }

    pts = np.column_stack([xs, ys]).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    (_, _), (w_px, h_px), _ = rect
    length_m = max(float(w_px), float(h_px)) * mpp
    metrics = _polyline_metrics(mask, mpp, smooth_m=curve_centerline_smooth_m)
    path_ratio = metrics["path_ratio"]
    turn_sum_deg = metrics["turn_sum_deg"]
    max_dev_ratio = metrics["max_dev_ratio"]
    path_ratio_ok = path_ratio is not None and float(path_ratio) >= float(curve_path_ratio_threshold)
    turn_sum_ok = turn_sum_deg is not None and float(turn_sum_deg) >= float(curve_turn_sum_deg_threshold)
    deviation_ok = max_dev_ratio is not None and float(max_dev_ratio) >= float(curve_deviation_ratio_threshold)
    is_curve = bool(
        deviation_ok and (path_ratio_ok or turn_sum_ok)
    )

    if length_m >= short_threshold_m:
        route = "wide_curve_by_length"
    elif is_curve:
        route = "wide_curve_by_curve"
    else:
        route = "box"

    return {
        "length_m": float(length_m),
        **metrics,
        "is_curve": is_curve,
        "route": route,
    }


def _split_masks_by_shape(
    masks: np.ndarray,
    bev_meta: dict,
    *,
    short_threshold_m: float,
    curve_path_ratio_threshold: float,
    curve_turn_sum_deg_threshold: float,
    curve_deviation_ratio_threshold: float,
    curve_centerline_smooth_m: float,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    arr = np.asarray(masks)
    box_idx: list[int] = []
    wide_curve_idx: list[int] = []
    routing_debug: list[dict] = []

    if arr.ndim == 3:
        for idx, mask in enumerate(arr):
            analysis = _analyze_laneline_mask(
                mask.astype(bool),
                bev_meta,
                short_threshold_m=short_threshold_m,
                curve_path_ratio_threshold=curve_path_ratio_threshold,
                curve_turn_sum_deg_threshold=curve_turn_sum_deg_threshold,
                curve_deviation_ratio_threshold=curve_deviation_ratio_threshold,
                curve_centerline_smooth_m=curve_centerline_smooth_m,
            )
            analysis["id"] = idx
            routing_debug.append(analysis)
            route = analysis["route"]
            if route == "box":
                box_idx.append(idx)
            elif route in {"wide_curve_by_length", "wide_curve_by_curve"}:
                wide_curve_idx.append(idx)

        box_masks = arr[box_idx].astype(bool, copy=False) if box_idx else np.zeros((0, *arr.shape[1:]), dtype=bool)
        wide_curve_masks = arr[wide_curve_idx].astype(bool, copy=False) if wide_curve_idx else np.zeros((0, *arr.shape[1:]), dtype=bool)
        return box_masks, wide_curve_masks, routing_debug

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D or 3D masks array, got {arr.shape}")

    from scipy import ndimage

    positive_labels = np.asarray(arr, dtype=np.int32) + 1
    object_slices = ndimage.find_objects(positive_labels)
    for oid, slc in enumerate(object_slices):
        if slc is None:
            continue
        r_slc, c_slc = slc
        r0, r1 = int(r_slc.start), int(r_slc.stop)
        c0, c1 = int(c_slc.start), int(c_slc.stop)
        crop = np.asarray(arr[r0:r1, c0:c1] == oid)
        if not np.any(crop):
            continue
        analysis = _analyze_laneline_mask(
            crop,
            bev_meta,
            short_threshold_m=short_threshold_m,
            curve_path_ratio_threshold=curve_path_ratio_threshold,
            curve_turn_sum_deg_threshold=curve_turn_sum_deg_threshold,
            curve_deviation_ratio_threshold=curve_deviation_ratio_threshold,
            curve_centerline_smooth_m=curve_centerline_smooth_m,
        )
        analysis["id"] = oid
        routing_debug.append(analysis)
        route = analysis["route"]
        if route == "box":
            box_idx.append(oid)
        elif route in {"wide_curve_by_length", "wide_curve_by_curve"}:
            wide_curve_idx.append(oid)

    def _filter_label_map(keep_ids: list[int]) -> np.ndarray:
        filtered = np.full(arr.shape, -1, dtype=np.int32)
        for oid in keep_ids:
            filtered[arr == oid] = oid
        return filtered

    return _filter_label_map(box_idx), _filter_label_map(wide_curve_idx), routing_debug


def _write_routing_debug(records: list[dict], output_path: Path) -> Path:
    output_path.write_text(json.dumps({"records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _polygon_record(shape_points: list[tuple[float, float]], source: str, default_id: int) -> dict:
    poly = Polygon(shape_points)
    if poly.is_empty:
        raise ValueError("empty polygon")
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
    return {
        "id": default_id,
        "source": source,
        "length_m": float(length_m),
        "width_m": float(width_m),
        "yaw": float(math.atan2(dy, dx)),
        "cx": float(center.x),
        "cy": float(center.y),
        "ring": [[float(x), float(y)] for x, y in shape_points],
    }


def _collect_polygons_from_shp(shp_path: Path, source: str) -> list[dict]:
    if not shp_path.is_file():
        return []
    sf = shapefile.Reader(str(shp_path))
    records: list[dict] = []
    for idx, shape in enumerate(sf.shapes()):
        if len(shape.points) < 4:
            continue
        try:
            records.append(_polygon_record(shape.points, source, idx))
        except Exception:
            continue
    return records


def _write_laneline_shp(records: list[dict], output_path: Path) -> Path:
    w = shapefile.Writer(str(output_path))
    w.shapeType = shapefile.POLYGON
    w.field("id", "N", decimal=0)
    w.field("source", "C", size=12)
    w.field("length_m", "F", decimal=3)
    w.field("width_m", "F", decimal=3)
    w.field("yaw", "F", decimal=6)
    w.field("cx", "F", decimal=3)
    w.field("cy", "F", decimal=3)
    for idx, rec in enumerate(records):
        ring = rec["ring"]
        if ring[0] != ring[-1]:
            ring = [*ring, ring[0]]
        w.poly([ring])
        w.record(
            id=idx,
            source=rec["source"],
            length_m=rec["length_m"],
            width_m=rec["width_m"],
            yaw=rec["yaw"],
            cx=rec["cx"],
            cy=rec["cy"],
        )
    w.close()
    return Path(f"{output_path}.shp")


def _iter_polygon_parts(geom) -> list[Polygon]:
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom] if float(geom.area) > 0 else []
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        polygons: list[Polygon] = []
        for part in geom.geoms:
            polygons.extend(_iter_polygon_parts(part))
        return polygons
    return []


def _shape_to_polygons(shape) -> list[Polygon]:
    points = shape.points
    if len(points) < 4:
        return []
    parts = list(shape.parts) + [len(points)]
    polygons: list[Polygon] = []
    for idx in range(len(parts) - 1):
        ring = points[int(parts[idx]) : int(parts[idx + 1])]
        if len(ring) < 4:
            continue
        try:
            poly = Polygon(ring)
        except Exception:
            continue
        if poly.is_empty:
            continue
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or float(poly.area) <= 0:
            continue
        polygons.extend(_iter_polygon_parts(poly))
    return polygons


def _read_laneline_records(shp_path: Path | str) -> list[dict]:
    path = Path(shp_path).expanduser()
    if not path.is_file():
        return []
    sf = shapefile.Reader(str(path))
    records: list[dict] = []
    for idx, shape_record in enumerate(sf.iterShapeRecords()):
        polygons = _shape_to_polygons(shape_record.shape)
        if not polygons:
            continue
        rec = shape_record.record.as_dict()
        for poly in polygons:
            ring = [[float(x), float(y)] for x, y in poly.exterior.coords]
            records.append(
                {
                    "id": int(rec.get("id", idx)),
                    "source": str(rec.get("source", "")),
                    "length_m": float(rec.get("length_m", 0.0)),
                    "width_m": float(rec.get("width_m", 0.0)),
                    "yaw": float(rec.get("yaw", 0.0)),
                    "cx": float(rec.get("cx", poly.centroid.x)),
                    "cy": float(rec.get("cy", poly.centroid.y)),
                    "ring": ring,
                }
            )
    return records


def _read_shp_polygons(shp_path: Path | str) -> list[Polygon]:
    path = Path(shp_path).expanduser()
    if not path.is_file():
        return []
    sf = shapefile.Reader(str(path))
    polygons: list[Polygon] = []
    for shape in sf.shapes():
        polygons.extend(_shape_to_polygons(shape))
    return polygons


def _record_is_inside_crosswalk(record: dict, crosswalk_polygons: list[Polygon], *, overlap_ratio_threshold: float) -> bool:
    ring = record.get("ring", [])
    if len(ring) < 3:
        return False
    try:
        poly = Polygon(ring)
    except Exception:
        return False
    if poly.is_empty or float(poly.area) <= 0:
        return False
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or float(poly.area) <= 0:
        return False
    centroid = Point(float(record.get("cx", poly.centroid.x)), float(record.get("cy", poly.centroid.y)))
    area = float(poly.area)
    for crosswalk_poly in crosswalk_polygons:
        if crosswalk_poly.covers(centroid):
            return True
        overlap_ratio = float(poly.intersection(crosswalk_poly).area) / max(area, 1e-9)
        if overlap_ratio >= float(overlap_ratio_threshold):
            return True
    return False


def filter_laneline_shp_by_crosswalk(
    laneline_shp_path: Path | str,
    crosswalk_shp_path: Path | str,
    *,
    output_path: Path | str | None = None,
    overlap_ratio_threshold: float = 0.50,
) -> Path:
    laneline_path = Path(laneline_shp_path).expanduser()
    crosswalk_path = Path(crosswalk_shp_path).expanduser()
    out_base = laneline_path.with_suffix("") if output_path is None else Path(output_path).expanduser().with_suffix("")

    records = _read_laneline_records(laneline_path)
    crosswalk_polygons = _read_shp_polygons(crosswalk_path)
    if not crosswalk_polygons:
        filtered = records
        removed_records: list[dict] = []
    else:
        filtered = []
        removed_records = []
        for rec in records:
            if _record_is_inside_crosswalk(rec, crosswalk_polygons, overlap_ratio_threshold=overlap_ratio_threshold):
                removed_records.append({"id": rec.get("id"), "source": rec.get("source")})
            else:
                filtered.append(rec)

    out_path = _write_laneline_shp(filtered, out_base)
    summary = {
        "laneline_shp": str(laneline_path),
        "crosswalk_shp": str(crosswalk_path),
        "output_shp": str(out_path),
        "overlap_ratio_threshold": float(overlap_ratio_threshold),
        "input_records": int(len(records)),
        "removed_records": int(len(removed_records)),
        "kept_records": int(len(filtered)),
        "removed": removed_records,
    }
    (out_base.parent / "crosswalk_filter_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    polygon_shp_to_centerline_shp(out_path, out_base.parent / "laneline_centerline")
    return out_path


def _write_laneline_preview(records: list[dict], bev_meta: dict, output_path: Path) -> Path:
    width = int(bev_meta["width"])
    height = int(bev_meta["height"])
    mpp = float(bev_meta["meters_per_pixel"])
    min_x, _min_y = [float(v) for v in bev_meta["min_xy"]]
    _max_x, max_y = [float(v) for v in bev_meta["max_xy"]]

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    colors = {
        "box": (80, 220, 80),
        "wide_curve": (80, 180, 255),
    }

    for rec in records:
        ring = rec.get("ring", [])
        if len(ring) < 3:
            continue
        pts: list[list[int]] = []
        for x, y in ring:
            px = int(round((float(x) - min_x) / mpp))
            py = int(round((max_y - float(y)) / mpp))
            pts.append([px, py])
        poly = np.asarray(pts, dtype=np.int32)
        poly[:, 0] = np.clip(poly[:, 0], 0, width - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, height - 1)
        color = colors.get(str(rec.get("source", "")), (220, 220, 220))
        cv2.fillPoly(canvas, [poly], color)
        cv2.polylines(canvas, [poly], isClosed=True, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    return output_path


def _load_label_map(label_map_path: Path | str) -> np.ndarray:
    label_map = np.load(Path(label_map_path).expanduser())
    if label_map.ndim != 2:
        raise ValueError(f"label_map must have shape (H,W), got {label_map.shape}: {label_map_path}")
    return np.asarray(label_map, dtype=np.int32)


def _label_areas(label_map: np.ndarray | None) -> dict[int, int]:
    if label_map is None:
        return {}
    ids, counts = np.unique(label_map[label_map >= 0], return_counts=True)
    return {int(label_id): int(count) for label_id, count in zip(ids, counts)}


def _max_instance_iou(
    source_instance_mask: np.ndarray,
    source_area: int,
    target_label_map: np.ndarray | None,
    target_areas: dict[int, int],
) -> tuple[float, int | None, int, int]:
    if target_label_map is None or source_area <= 0:
        return 0.0, None, 0, source_area
    overlap_labels = np.asarray(target_label_map[source_instance_mask], dtype=np.int32)
    overlap_labels = overlap_labels[overlap_labels >= 0]
    if overlap_labels.size == 0:
        return 0.0, None, 0, source_area
    ids, counts = np.unique(overlap_labels, return_counts=True)
    best_iou = 0.0
    best_id: int | None = None
    best_intersection = 0
    best_union = source_area
    for target_id, intersection in zip(ids, counts):
        target_id_int = int(target_id)
        intersection_int = int(intersection)
        union = int(source_area + target_areas.get(target_id_int, 0) - intersection_int)
        iou = float(intersection_int / max(union, 1))
        if iou > best_iou:
            best_iou = iou
            best_id = target_id_int
            best_intersection = intersection_int
            best_union = union
    return best_iou, best_id, best_intersection, best_union


def _cross_like_road_marking_stats(mask: np.ndarray) -> dict[str, float | int | bool]:
    from skimage.morphology import skeletonize

    ys, xs = np.where(mask)
    if ys.size == 0:
        return {
            "is_cross_like": False,
            "area_px": 0,
            "fill_ratio": 0.0,
            "aspect": 0.0,
            "branch_points": 0,
            "endpoints": 0,
        }

    area = int(ys.size)
    bbox_h = int(ys.max() - ys.min() + 1)
    bbox_w = int(xs.max() - xs.min() + 1)
    fill_ratio = float(area / max(bbox_h * bbox_w, 1))

    pts = np.column_stack([xs, ys]).astype(np.float32)
    (_center_x, _center_y), (rect_w, rect_h), _angle = cv2.minAreaRect(pts)
    length = max(float(rect_w), float(rect_h))
    width = max(min(float(rect_w), float(rect_h)), 1e-6)
    aspect = float(length / width)

    skel = skeletonize(mask)
    branch_points = 0
    endpoints = 0
    for y, x in zip(*np.where(skel)):
        y0 = max(0, int(y) - 1)
        y1 = min(skel.shape[0], int(y) + 2)
        x0 = max(0, int(x) - 1)
        x1 = min(skel.shape[1], int(x) + 2)
        neighbors = int(skel[y0:y1, x0:x1].sum()) - 1
        if neighbors == 1:
            endpoints += 1
        elif neighbors >= 3:
            branch_points += 1

    return {
        "is_cross_like": bool(aspect < 3.0 and fill_ratio < 0.25 and branch_points >= 8),
        "area_px": area,
        "fill_ratio": fill_ratio,
        "aspect": aspect,
        "branch_points": int(branch_points),
        "endpoints": int(endpoints),
    }


def _build_supplemented_laneline_label_map(
    laneline_label_map: np.ndarray,
    road_marking_label_map: np.ndarray | None,
    arrow_label_map: np.ndarray | None,
    *,
    min_supplement_area_px: int,
    arrow_remove_iou_threshold: float,
    laneline_remove_iou_threshold: float,
) -> tuple[np.ndarray, dict]:
    if road_marking_label_map is None:
        return laneline_label_map, {
            "enabled": False,
            "reason": "missing_road_marking_label_map",
            "original_laneline_instances": int(laneline_label_map.max()) + 1 if int(laneline_label_map.max()) >= 0 else 0,
            "supplement_instances": 0,
            "final_laneline_instances": int(laneline_label_map.max()) + 1 if int(laneline_label_map.max()) >= 0 else 0,
        }
    if road_marking_label_map.shape != laneline_label_map.shape:
        raise ValueError(
            "road_marking_label_map shape must match laneline label_map, "
            f"got {road_marking_label_map.shape} vs {laneline_label_map.shape}"
        )
    if arrow_label_map is not None and arrow_label_map.shape != laneline_label_map.shape:
        raise ValueError(
            "arrow_label_map shape must match laneline label_map, "
            f"got {arrow_label_map.shape} vs {laneline_label_map.shape}"
        )

    laneline_mask = laneline_label_map >= 0
    arrow_areas = _label_areas(arrow_label_map)
    laneline_areas = _label_areas(laneline_label_map)

    max_id = int(laneline_label_map.max())
    next_id = max_id + 1 if max_id >= 0 else 0
    output = np.array(laneline_label_map, dtype=np.int32, copy=True)
    supplement_instances = 0
    supplement_pixels = 0
    skipped_small_components = 0
    arrow_removed_instances = 0
    arrow_removed_pixels = 0
    arrow_removed_records: list[dict] = []
    laneline_removed_instances = 0
    laneline_removed_pixels = 0
    laneline_removed_records: list[dict] = []
    cross_like_removed_instances = 0
    cross_like_removed_pixels = 0
    cross_like_removed_records: list[dict] = []
    existing_laneline_overlap_pixels = 0
    candidate_pixels = 0

    road_marking_ids = [int(v) for v in np.unique(road_marking_label_map) if int(v) >= 0]
    for road_marking_id in road_marking_ids:
        road_marking_instance = road_marking_label_map == road_marking_id
        road_marking_area = int(np.count_nonzero(road_marking_instance))
        if road_marking_area <= 0:
            continue
        iou, matched_arrow_id, intersection, union = _max_instance_iou(
            road_marking_instance,
            road_marking_area,
            arrow_label_map,
            arrow_areas,
        )
        if iou >= float(arrow_remove_iou_threshold):
            arrow_removed_instances += 1
            arrow_removed_pixels += road_marking_area
            arrow_removed_records.append(
                {
                    "road_marking_id": int(road_marking_id),
                    "matched_arrow_id": None if matched_arrow_id is None else int(matched_arrow_id),
                    "iou": float(iou),
                    "area_px": int(road_marking_area),
                    "intersection_px": int(intersection),
                    "union_px": int(union),
                }
            )
            continue

        laneline_iou, matched_laneline_id, laneline_intersection, laneline_union = _max_instance_iou(
            road_marking_instance,
            road_marking_area,
            laneline_label_map,
            laneline_areas,
        )
        if laneline_iou >= float(laneline_remove_iou_threshold):
            laneline_removed_instances += 1
            laneline_removed_pixels += road_marking_area
            laneline_removed_records.append(
                {
                    "road_marking_id": int(road_marking_id),
                    "matched_laneline_id": None if matched_laneline_id is None else int(matched_laneline_id),
                    "iou": float(laneline_iou),
                    "area_px": int(road_marking_area),
                    "intersection_px": int(laneline_intersection),
                    "union_px": int(laneline_union),
                }
            )
            continue

        cross_stats = _cross_like_road_marking_stats(road_marking_instance)
        if bool(cross_stats["is_cross_like"]):
            cross_like_removed_instances += 1
            cross_like_removed_pixels += road_marking_area
            cross_like_removed_records.append(
                {
                    "road_marking_id": int(road_marking_id),
                    **cross_stats,
                }
            )
            continue

        candidate = road_marking_instance & ~laneline_mask
        existing_laneline_overlap_pixels += laneline_intersection
        area = int(np.count_nonzero(candidate))
        candidate_pixels += area
        if area < int(min_supplement_area_px):
            skipped_small_components += 1
            continue
        output[candidate] = next_id
        next_id += 1
        supplement_instances += 1
        supplement_pixels += area

    original_instances = max_id + 1 if max_id >= 0 else 0
    road_marking_instances = int(road_marking_label_map.max()) + 1 if int(road_marking_label_map.max()) >= 0 else 0
    arrow_instances = 0 if arrow_label_map is None or int(arrow_label_map.max()) < 0 else int(arrow_label_map.max()) + 1
    summary = {
        "enabled": True,
        "original_laneline_instances": int(original_instances),
        "road_marking_instances": int(road_marking_instances),
        "arrow_instances": int(arrow_instances),
        "road_marking_pixels": int(np.count_nonzero(road_marking_label_map >= 0)),
        "arrow_remove_iou_threshold": float(arrow_remove_iou_threshold),
        "laneline_remove_iou_threshold": float(laneline_remove_iou_threshold),
        "arrow_removed_instances": int(arrow_removed_instances),
        "arrow_removed_pixels": int(arrow_removed_pixels),
        "arrow_removed_records": arrow_removed_records,
        "laneline_removed_instances": int(laneline_removed_instances),
        "laneline_removed_pixels": int(laneline_removed_pixels),
        "laneline_removed_records": laneline_removed_records,
        "cross_like_removed_instances": int(cross_like_removed_instances),
        "cross_like_removed_pixels": int(cross_like_removed_pixels),
        "cross_like_removed_records": cross_like_removed_records,
        "existing_laneline_overlap_pixels": int(existing_laneline_overlap_pixels),
        "candidate_pixels": int(candidate_pixels),
        "supplement_instances": int(supplement_instances),
        "supplement_pixels": int(supplement_pixels),
        "skipped_small_components": int(skipped_small_components),
        "min_supplement_area_px": int(min_supplement_area_px),
        "final_laneline_instances": int(next_id),
    }
    return output, summary


def run_laneline(
    label_map_path: Path | str,
    geo_meta_path: Path | str,
    ply_path: Path | str,
    output_dir: Path | str,
    *,
    road_marking_label_map_path: Path | str | None = None,
    arrow_label_map_path: Path | str | None = None,
    min_supplement_area_px: int = 10,
    arrow_remove_iou_threshold: float = 0.30,
    laneline_remove_iou_threshold: float = 0.30,
    short_threshold_m: float = 10.0,
    curve_path_ratio_threshold: float = 1.08,
    curve_turn_sum_deg_threshold: float = 25.0,
    curve_deviation_ratio_threshold: float = 0.08,
    curve_centerline_smooth_m: float = 0.10,
    box_debug: bool = False,
) -> Path:
    laneline_label_map = _load_label_map(label_map_path)
    road_marking_label_map = _load_label_map(road_marking_label_map_path) if road_marking_label_map_path is not None else None
    arrow_label_map = _load_label_map(arrow_label_map_path) if arrow_label_map_path is not None else None
    bev_meta = _coerce_bev_meta(geo_meta_path)
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    masks, supplement_summary = _build_supplemented_laneline_label_map(
        laneline_label_map,
        road_marking_label_map,
        arrow_label_map,
        min_supplement_area_px=min_supplement_area_px,
        arrow_remove_iou_threshold=arrow_remove_iou_threshold,
        laneline_remove_iou_threshold=laneline_remove_iou_threshold,
    )
    np.save(output_dir / "supplemented_label_map.npy", masks)
    (output_dir / "supplement_summary.json").write_text(
        json.dumps(supplement_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    box_masks, wide_curve_masks, routing_debug = _split_masks_by_shape(
        masks,
        bev_meta,
        short_threshold_m=short_threshold_m,
        curve_path_ratio_threshold=curve_path_ratio_threshold,
        curve_turn_sum_deg_threshold=curve_turn_sum_deg_threshold,
        curve_deviation_ratio_threshold=curve_deviation_ratio_threshold,
        curve_centerline_smooth_m=curve_centerline_smooth_m,
    )
    _write_routing_debug(routing_debug, output_dir / "routing_debug.json")

    records: list[dict] = []
    if box_masks.shape[0] > 0:
        short_shp = box_masks_to_shp(
            box_masks,
            bev_meta,
            ply_path,
            output_dir / "short_box",
            shp_stem="short_box",
            debug=box_debug,
        )
        records.extend(_collect_polygons_from_shp(short_shp, "box"))

    if wide_curve_masks.shape[0] > 0:
        long_shp = wide_curve_masks_to_shp(wide_curve_masks, bev_meta, ply_path, output_dir / "long_curve")
        records.extend(_collect_polygons_from_shp(long_shp, "wide_curve"))

    final_path = _write_laneline_shp(records, output_dir / "laneline")
    polygon_shp_to_centerline_shp(final_path, output_dir / "laneline_centerline")
    _write_laneline_preview(records, bev_meta, output_dir / "laneline.png")
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Vectorize laneline label_map with box or wide-curve pipelines.")
    parser.add_argument("label_map_npy", help="Laneline instance label_map .npy with shape (H,W).")
    parser.add_argument("geo_meta_json", help="geo_meta.json or summary.json path.")
    parser.add_argument("ply_path", help="PLY path used for short-line refinement.")
    parser.add_argument("-o", "--output-dir", default="outputs/apps/laneline", help="Output directory.")
    parser.add_argument("--road-marking-label-map", help="Optional road marking label_map used to supplement laneline.")
    parser.add_argument("--arrow-label-map", help="Optional arrow label_map removed from road marking supplement.")
    parser.add_argument("--min-supplement-area-px", type=int, default=10)
    parser.add_argument("--arrow-remove-iou-threshold", type=float, default=0.30)
    parser.add_argument("--laneline-remove-iou-threshold", type=float, default=0.30)
    parser.add_argument("--short-threshold", type=float, default=10.0, help="Threshold in meters for box vs wide-curve.")
    parser.add_argument("--curve-path-ratio-threshold", type=float, default=1.08)
    parser.add_argument("--curve-turn-sum-threshold", type=float, default=25.0)
    parser.add_argument("--curve-deviation-ratio-threshold", type=float, default=0.08)
    parser.add_argument("--curve-smooth", type=float, default=0.10)
    parser.add_argument("--box-debug", action="store_true", help="Pass debug mode to box_shp and skip short-box refine.")
    args = parser.parse_args()
    run_laneline(
        args.label_map_npy,
        args.geo_meta_json,
        args.ply_path,
        args.output_dir,
        road_marking_label_map_path=args.road_marking_label_map,
        arrow_label_map_path=args.arrow_label_map,
        min_supplement_area_px=args.min_supplement_area_px,
        arrow_remove_iou_threshold=args.arrow_remove_iou_threshold,
        laneline_remove_iou_threshold=args.laneline_remove_iou_threshold,
        short_threshold_m=args.short_threshold,
        curve_path_ratio_threshold=args.curve_path_ratio_threshold,
        curve_turn_sum_deg_threshold=args.curve_turn_sum_threshold,
        curve_deviation_ratio_threshold=args.curve_deviation_ratio_threshold,
        curve_centerline_smooth_m=args.curve_smooth,
        box_debug=args.box_debug,
    )


if __name__ == "__main__":
    main()
