"""Extract green-belt outer polygons from vegetation masks and height edges."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import shapefile
from PIL import Image

from landmark.tools.to_shp.geometry import pixel_to_xy


Image.MAX_IMAGE_PIXELS = None

HEIGHT_DIFF_MIN_M = 0.03
HEIGHT_DIFF_MAX_M = 0.20
GREEN_CLOSE_RADIUS_M = 0.30
STRIP_CONNECT_RADIUS_M = 0.50
FINAL_CLOSE_RADIUS_M = 0.20
DIFF_RING_RADIUS_M = 0.50
MIN_AREA_M2 = 1.0
MIN_WIDTH_M = 0.50
MAX_WIDTH_M = 5.00
MIN_LENGTH_M = 2.0
MIN_ASPECT = 4.0
MIN_RECT_FILL = 0.15
MAX_RECT_FILL = 0.95
REQUIRE_DIFF_SUPPORT = True
MIN_DIFF_SUPPORT_RATIO = 0.03
ROAD_BUFFER_M = 0.30
MIN_ROAD_BUFFER_OVERLAP = 0.15
POLYGON_SIMPLIFY_M = 0.06


def _load_json(path: Path | str) -> dict[str, Any]:
    path = Path(path).expanduser()
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _default_map_dir(pre_part_dir: Path) -> Path:
    pre_part_dir = Path(pre_part_dir).expanduser()
    return pre_part_dir.parent if pre_part_dir.name == "pre-part" else pre_part_dir


def _resolve_pre_part_assets(pre_part_dir: Path) -> dict[str, Path]:
    pre_part_dir = Path(pre_part_dir).expanduser()
    pc_csf_dir = pre_part_dir / "bev_pc_csf"
    asset_dir = pc_csf_dir if pc_csf_dir.is_dir() else pre_part_dir
    assets = {
        "height_values": asset_dir / "bev_pc_csf_height_values.npy",
        "height_meta": asset_dir / "bev_pc_csf_height_meta.json",
    }
    missing = [name for name, path in assets.items() if not path.is_file()]
    if missing:
        details = ", ".join(f"{name}={assets[name]}" for name in missing)
        raise FileNotFoundError(f"Missing belt inputs: {details}")
    return assets


def _kernel(radius_m: float, mpp: float) -> np.ndarray:
    radius_px = max(1, int(round(float(radius_m) / float(mpp))))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius_px * 2 + 1, radius_px * 2 + 1))


def _write_mask(mask: np.ndarray, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(out_path)
    return out_path


def _write_rgb(base_mask: np.ndarray, overlay_mask: np.ndarray, out_path: Path, *, overlay_color: tuple[int, int, int]) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.zeros((*base_mask.shape, 3), dtype=np.uint8)
    rgb[base_mask] = (255, 255, 255)
    rgb[overlay_mask] = overlay_color
    Image.fromarray(rgb).save(out_path)
    return out_path


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


def _area_filter(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros(mask.shape, dtype=bool)
    for comp_id in range(1, num):
        if int(stats[comp_id, cv2.CC_STAT_AREA]) >= int(min_area_px):
            out[labels == comp_id] = True
    return out


def _fill_external_contours(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(mask.shape, dtype=np.uint8)
    if contours:
        cv2.drawContours(out, contours, -1, 1, thickness=-1)
    return out > 0


def _build_green_union(label_map: np.ndarray, *, mpp: float) -> np.ndarray:
    green = np.asarray(label_map >= 0)
    closed = cv2.morphologyEx(green.astype(np.uint8), cv2.MORPH_CLOSE, _kernel(GREEN_CLOSE_RADIUS_M, mpp)) > 0
    closed = _fill_external_contours(closed)
    return _area_filter(closed, max(1, int(round(MIN_AREA_M2 / (mpp * mpp)))))


def _component_stats(mask: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return num, labels, stats


def _shape_metrics(
    *,
    comp_id: int,
    labels: np.ndarray,
    stats: np.ndarray,
    mask: np.ndarray,
    diff: np.ndarray,
    road_buffer: np.ndarray | None,
    mpp: float,
) -> dict[str, Any] | None:
    area_px = int(stats[comp_id, cv2.CC_STAT_AREA])
    if area_px <= 0:
        return None
    x = int(stats[comp_id, cv2.CC_STAT_LEFT])
    y = int(stats[comp_id, cv2.CC_STAT_TOP])
    w = int(stats[comp_id, cv2.CC_STAT_WIDTH])
    h = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
    crop = labels[y : y + h, x : x + w] == comp_id
    rr, cc = np.nonzero(crop)
    if len(rr) < 5:
        return None

    pts = np.column_stack([cc.astype(np.float32), rr.astype(np.float32)])
    rect = cv2.minAreaRect(pts)
    (_cx, _cy), (rw, rh), angle = rect
    major_px = float(max(rw, rh))
    minor_px = float(min(rw, rh))
    if major_px <= 0.0 or minor_px <= 0.0:
        return None

    ring_radius_px = max(1, int(round(DIFF_RING_RADIUS_M / mpp)))
    pad = ring_radius_px + 2
    r0 = max(0, y - pad)
    r1 = min(mask.shape[0], y + h + pad)
    c0 = max(0, x - pad)
    c1 = min(mask.shape[1], x + w + pad)
    full_crop = labels[r0:r1, c0:c1] == comp_id
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_radius_px * 2 + 1, ring_radius_px * 2 + 1))
    dilated = cv2.dilate(full_crop.astype(np.uint8), kernel) > 0
    eroded = cv2.erode(full_crop.astype(np.uint8), kernel) > 0
    ring = dilated & ~eroded
    diff_crop = diff[r0:r1, c0:c1]
    diff_hits = int(np.count_nonzero(ring & diff_crop))
    diff_support_ratio = float(diff_hits) / float(max(1, np.count_nonzero(ring)))
    road_buffer_overlap = 1.0
    if road_buffer is not None:
        road_buffer_crop = road_buffer[y : y + h, x : x + w]
        road_buffer_overlap = float(np.count_nonzero(crop & road_buffer_crop)) / float(max(1, area_px))
    rect_fill = float(area_px) / float(max(1.0, rw * rh))

    return {
        "component_id": int(comp_id),
        "area_px": area_px,
        "area_m2": float(area_px * mpp * mpp),
        "bbox_px": [x, y, w, h],
        "length_m": float(major_px * mpp),
        "width_m": float(minor_px * mpp),
        "aspect": float(major_px / max(minor_px, 1e-6)),
        "rect_fill": rect_fill,
        "angle_deg": float(angle),
        "diff_hits": diff_hits,
        "diff_support_ratio": diff_support_ratio,
        "road_buffer_overlap": road_buffer_overlap,
    }


def _candidate_reason(metrics: dict[str, Any], *, require_road: bool) -> str:
    if metrics["area_m2"] < MIN_AREA_M2:
        return "area"
    if metrics["width_m"] < MIN_WIDTH_M:
        return "width"
    if metrics["width_m"] > MAX_WIDTH_M:
        return "width"
    if metrics["length_m"] < MIN_LENGTH_M:
        return "length"
    if metrics["aspect"] < MIN_ASPECT:
        return "aspect"
    if metrics["rect_fill"] < MIN_RECT_FILL or metrics["rect_fill"] > MAX_RECT_FILL:
        return "rect_fill"
    if REQUIRE_DIFF_SUPPORT and metrics["diff_support_ratio"] < MIN_DIFF_SUPPORT_RATIO:
        return "diff"
    if require_road and metrics["road_buffer_overlap"] < MIN_ROAD_BUFFER_OVERLAP:
        return "road"
    return "keep"


def _select_candidates(
    mask: np.ndarray,
    diff: np.ndarray,
    *,
    road_buffer: np.ndarray | None,
    mpp: float,
) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray]:
    connected = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, _kernel(STRIP_CONNECT_RADIUS_M, mpp)) > 0
    connected = _fill_external_contours(connected)
    connected = _area_filter(connected, max(1, int(round(MIN_AREA_M2 / (mpp * mpp)))))
    num, labels, stats = _component_stats(connected)
    out = np.zeros(mask.shape, dtype=bool)
    shape_only = np.zeros(mask.shape, dtype=bool)
    records: list[dict[str, Any]] = []
    for comp_id in range(1, num):
        metrics = _shape_metrics(
            comp_id=comp_id,
            labels=labels,
            stats=stats,
            mask=connected,
            diff=diff,
            road_buffer=road_buffer,
            mpp=mpp,
        )
        if metrics is None:
            continue
        shape_keep = (
            metrics["area_m2"] >= MIN_AREA_M2
            and metrics["width_m"] >= MIN_WIDTH_M
            and metrics["width_m"] <= MAX_WIDTH_M
            and metrics["length_m"] >= MIN_LENGTH_M
            and metrics["aspect"] >= MIN_ASPECT
            and MIN_RECT_FILL <= metrics["rect_fill"] <= MAX_RECT_FILL
        )
        reason = _candidate_reason(metrics, require_road=False)
        metrics["keep_shape"] = bool(shape_keep)
        metrics["keep_final"] = reason == "keep"
        metrics["reject_reason"] = reason
        records.append(metrics)
        if shape_keep:
            shape_only[labels == comp_id] = True
        if reason == "keep":
            out[labels == comp_id] = True
    return out, records, shape_only


def _finalize_mask(mask: np.ndarray, *, mpp: float) -> np.ndarray:
    if not np.any(mask):
        return mask.astype(bool)
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, _kernel(FINAL_CLOSE_RADIUS_M, mpp)) > 0
    closed = _fill_external_contours(closed)
    return _area_filter(closed, max(1, int(round(MIN_AREA_M2 / (mpp * mpp)))))


def _contour_metrics(
    mask: np.ndarray,
    comp_id: int,
    labels: np.ndarray,
    stats: np.ndarray,
    *,
    diff: np.ndarray,
    road_buffer: np.ndarray | None,
    mpp: float,
) -> dict[str, Any] | None:
    return _shape_metrics(
        comp_id=comp_id,
        labels=labels,
        stats=stats,
        mask=mask,
        diff=diff,
        road_buffer=road_buffer,
        mpp=mpp,
    )


def _extract_polygons(
    mask: np.ndarray,
    meta: dict[str, Any],
    *,
    diff: np.ndarray,
    road_buffer: np.ndarray | None,
) -> list[dict[str, Any]]:
    mpp = float(meta["meters_per_pixel"])
    num, labels, stats = _component_stats(mask)
    records: list[dict[str, Any]] = []
    simplify_px = max(1.0, POLYGON_SIMPLIFY_M / mpp)
    for comp_id in range(1, num):
        x = int(stats[comp_id, cv2.CC_STAT_LEFT])
        y = int(stats[comp_id, cv2.CC_STAT_TOP])
        w = int(stats[comp_id, cv2.CC_STAT_WIDTH])
        h = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
        crop = labels[y : y + h, x : x + w] == comp_id
        contours, _ = cv2.findContours(crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 3:
            continue
        approx = cv2.approxPolyDP(contour, epsilon=float(simplify_px), closed=True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        approx = approx.astype(np.float64)
        approx[:, 0] += x
        approx[:, 1] += y
        if not np.array_equal(approx[0], approx[-1]):
            approx = np.vstack([approx, approx[0]])
        xy = pixel_to_xy(approx, meta)
        metrics = _contour_metrics(mask, comp_id, labels, stats, diff=diff, road_buffer=road_buffer, mpp=mpp)
        if metrics is None:
            continue
        records.append(
            {
                "id": len(records),
                "component_id": int(comp_id),
                "xy": xy,
                **metrics,
            }
        )
    return records


def _write_polygon_shp(records: list[dict[str, Any]], out_base: Path) -> Path:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    writer = shapefile.Writer(str(out_base))
    writer.shapeType = shapefile.POLYGON
    writer.field("id", "N", decimal=0)
    writer.field("comp_id", "N", decimal=0)
    writer.field("area_m2", "F", decimal=3)
    writer.field("length_m", "F", decimal=3)
    writer.field("width_m", "F", decimal=3)
    writer.field("aspect", "F", decimal=3)
    writer.field("fill", "F", decimal=3)
    writer.field("diff", "F", decimal=5)
    writer.field("roadbuf", "F", decimal=5)
    for rec in records:
        ring = [[float(x), float(y)] for x, y in np.asarray(rec["xy"], dtype=np.float64)]
        writer.poly([ring])
        writer.record(
            id=int(rec["id"]),
            comp_id=int(rec["component_id"]),
            area_m2=float(rec["area_m2"]),
            length_m=float(rec["length_m"]),
            width_m=float(rec["width_m"]),
            aspect=float(rec["aspect"]),
            fill=float(rec["rect_fill"]),
            diff=float(rec.get("diff_support_ratio", 0.0)),
            roadbuf=float(rec.get("road_buffer_overlap", 0.0)),
        )
    writer.close()
    return out_base.with_suffix(".shp")


def _rotated_eval(rotated_label_map_path: Path | None) -> dict[str, Any] | None:
    if rotated_label_map_path is None or not rotated_label_map_path.is_file():
        return None
    arr = np.load(rotated_label_map_path, mmap_mode="r")
    ids = np.unique(arr)
    return {
        "path": str(rotated_label_map_path),
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
        "object_count": int(np.count_nonzero(ids >= 0)),
    }


def run_belt(
    pre_part_dir: Path | str,
    output_dir: Path | str | None = None,
    *,
    green_label_map_path: Path | str | None = None,
    green_rotated_label_map_path: Path | str | None = None,
    parts_json_path: Path | str | None = None,
    road_label_map_path: Path | str | None = None,
) -> dict[str, Path]:
    pre_part_dir = Path(pre_part_dir).expanduser()
    map_dir = _default_map_dir(pre_part_dir)
    output_dir = Path(output_dir).expanduser() if output_dir is not None else map_dir / "belt"
    debug_dir = output_dir / "debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    if green_label_map_path is None:
        green_label_map_path = map_dir / "objs" / "green_veg" / "result" / "label_map.npy"
    green_label_map_path = Path(green_label_map_path).expanduser()
    if not green_label_map_path.is_file():
        raise FileNotFoundError(green_label_map_path)

    green_rotated = Path(green_rotated_label_map_path).expanduser() if green_rotated_label_map_path else None
    parts_json = Path(parts_json_path).expanduser() if parts_json_path else None
    if road_label_map_path is None:
        default_road = map_dir / "objs" / "road_score05" / "result" / "label_map.npy"
        road_label_map_path = default_road if default_road.is_file() else None
    road_label_map = Path(road_label_map_path).expanduser() if road_label_map_path else None

    assets = _resolve_pre_part_assets(pre_part_dir)
    height_values = np.load(assets["height_values"], mmap_mode="r")
    height_meta = _load_json(assets["height_meta"])
    green_label_map = np.load(green_label_map_path, mmap_mode="r")
    if tuple(green_label_map.shape) != tuple(height_values.shape):
        raise ValueError(f"green label_map shape {green_label_map.shape} does not match height raster {height_values.shape}")

    mpp = float(height_meta["meters_per_pixel"])
    diff = _height_diff_mask(height_values)
    _write_mask(diff, debug_dir / "diff.png")

    road_buffer: np.ndarray | None = None
    if road_label_map is not None:
        if not road_label_map.is_file():
            raise FileNotFoundError(road_label_map)
        road_label = np.load(road_label_map, mmap_mode="r")
        if tuple(road_label.shape) != tuple(height_values.shape):
            raise ValueError(f"road label_map shape {road_label.shape} does not match height raster {height_values.shape}")
        road_mask = np.asarray(road_label >= 0)
        road_buffer = cv2.dilate(road_mask.astype(np.uint8), _kernel(ROAD_BUFFER_M, mpp)) > 0
        _write_mask(road_buffer, debug_dir / "road_buffer.png")

    green_union = _build_green_union(green_label_map, mpp=mpp)
    _write_mask(green_union, debug_dir / "green_union.png")

    final_candidates, candidate_records, shape_candidates = _select_candidates(green_union, diff, road_buffer=road_buffer, mpp=mpp)
    _write_mask(shape_candidates, debug_dir / "method_green_shape.png")
    _write_mask(final_candidates, debug_dir / "candidate_strip_mask.png")
    _write_rgb(diff, final_candidates, debug_dir / "candidate_with_diff.png", overlay_color=(0, 255, 0))
    if road_buffer is not None:
        _write_rgb(road_buffer, final_candidates, debug_dir / "candidate_with_road_buffer.png", overlay_color=(0, 255, 0))

    final_mask = _finalize_mask(final_candidates, mpp=mpp)
    _write_mask(final_mask, debug_dir / "final_belt_mask.png")
    final_records = _extract_polygons(final_mask, height_meta, diff=diff, road_buffer=road_buffer)
    shp_path = _write_polygon_shp(final_records, output_dir / "belt")

    parts_summary: dict[str, Any] | None = None
    if parts_json is not None and parts_json.is_file():
        parts = _load_json(parts_json)
        parts_summary = {
            "path": str(parts_json),
            "rotation_deg": parts.get("rotation_deg"),
            "rotated_shape": parts.get("rotated_shape"),
            "num_parts": parts.get("num_parts"),
        }

    summary = {
        "pre_part_dir": str(pre_part_dir),
        "green_label_map": str(green_label_map_path),
        "green_rotated_label_map": str(green_rotated) if green_rotated else None,
        "parts_json": str(parts_json) if parts_json else None,
        "road_label_map": str(road_label_map) if road_label_map else None,
        "height_values": str(assets["height_values"]),
        "height_meta": str(assets["height_meta"]),
        "output_dir": str(output_dir),
        "debug_dir": str(debug_dir),
        "mpp": mpp,
        "thresholds": {
            "height_diff_min_m": HEIGHT_DIFF_MIN_M,
            "height_diff_max_m": HEIGHT_DIFF_MAX_M,
            "green_close_radius_m": GREEN_CLOSE_RADIUS_M,
            "strip_connect_radius_m": STRIP_CONNECT_RADIUS_M,
            "final_close_radius_m": FINAL_CLOSE_RADIUS_M,
            "diff_ring_radius_m": DIFF_RING_RADIUS_M,
            "min_area_m2": MIN_AREA_M2,
            "min_width_m": MIN_WIDTH_M,
            "max_width_m": MAX_WIDTH_M,
            "min_length_m": MIN_LENGTH_M,
            "min_aspect": MIN_ASPECT,
            "min_rect_fill": MIN_RECT_FILL,
            "max_rect_fill": MAX_RECT_FILL,
            "require_diff_support": REQUIRE_DIFF_SUPPORT,
            "min_diff_support_ratio": MIN_DIFF_SUPPORT_RATIO,
            "road_buffer_m": ROAD_BUFFER_M,
            "min_road_buffer_overlap": MIN_ROAD_BUFFER_OVERLAP,
        },
        "diff_pixels": int(np.count_nonzero(diff)),
        "green_union_pixels": int(np.count_nonzero(green_union)),
        "shape_candidate_pixels": int(np.count_nonzero(shape_candidates)),
        "final_candidate_pixels": int(np.count_nonzero(final_candidates)),
        "final_mask_pixels": int(np.count_nonzero(final_mask)),
        "candidate_count": len(candidate_records),
        "shape_candidate_count": int(sum(1 for r in candidate_records if r["keep_shape"])),
        "final_candidate_count": int(sum(1 for r in candidate_records if r["keep_final"])),
        "polygon_count": len(final_records),
        "shp": str(shp_path),
        "candidate_records": candidate_records,
        "rotated_eval": _rotated_eval(green_rotated),
        "parts_summary": parts_summary,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "shp": shp_path,
        "summary": summary_path,
        "debug_dir": debug_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract green-belt outer polygons from vegetation masks and height-diff BEV.")
    parser.add_argument("pre_part_dir", help="pre-part output directory.")
    parser.add_argument("-o", "--output-dir", default=None, help="Output directory. Defaults to <map>/belt.")
    parser.add_argument("--green-label-map", default=None, help="Existing green vegetation label_map.npy.")
    parser.add_argument("--green-rotated-label-map", default=None, help="Optional rotated_label_map.npy for summary/debug.")
    parser.add_argument("--parts-json", default=None, help="Optional instance_seg_v2 parts.json for rotation summary.")
    parser.add_argument("--road-label-map", default=None, help="Optional road label_map.npy. Defaults to <map>/objs/road_score05/result/label_map.npy when it exists.")
    args = parser.parse_args()
    outputs = run_belt(
        args.pre_part_dir,
        args.output_dir,
        green_label_map_path=args.green_label_map,
        green_rotated_label_map_path=args.green_rotated_label_map,
        parts_json_path=args.parts_json,
        road_label_map_path=args.road_label_map,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
