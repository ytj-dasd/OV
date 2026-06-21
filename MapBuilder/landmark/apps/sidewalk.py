"""Extract sidewalk inner-boundary candidates from height-diff BEV.

Current production flow:
1. build height-diff pixels from ``bev_pc_csf_height_values.npy``;
2. keep diff pixels inside a 50 cm buffer of the sidewalk ``label_map``;
3. close with a 5x5 ellipse and remove connected components below 200 px;
4. keep locally thin linear pixels;
5. extend each thin cluster along its fixed component direction;
6. link small raster breaks, merge strongly linear clusters, skeletonize;
7. export centerlines and endpoint-linked centerlines as shapefiles.
"""

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
from landmark.tools.to_shp.long_laneline_shp import (
    _chain_components,
    _connected_components_sorted,
    _skeletonize_mask,
    _smooth_polyline,
)


Image.MAX_IMAGE_PIXELS = None

DEFAULT_LABEL_MAP_PATH = Path(
    r"C:\Users\shang\workspace\MapBuilder\data\cjb\map\obj\sidewalk\result\label_map.npy"
)

HEIGHT_DIFF_MIN_M = 0.03
HEIGHT_DIFF_MAX_M = 0.20
LABEL_BUFFER_M = 0.50
AREA_FILTER_MIN_PX = 200
LOCAL_THIN_RADIUS_PX = 10
LOCAL_THIN_ANGLE_STEP_DEG = 15
LOCAL_THIN_RATIO = 3.0
LOCAL_THIN_MIN_LONG = 9
LOCAL_THIN_MAX_THIN = 7
LOCAL_THIN_POST_MIN_AREA_PX = 80
ENDPOINT_GROW_GAP_LIMIT_PX = 6
RASTER_LINK_MAX_GAP_PX = 30
RASTER_LINK_MAX_ANGLE_DEG = 25.0
RASTER_LINK_ENDPOINT_RADIUS_PX = 8
RASTER_LINK_MIN_COMPONENT_PX = 20
LINEAR_MERGE_MIN_AREA_PX = 80
LINEAR_MERGE_MIN_MAJOR_LENGTH_PX = 30.0
LINEAR_MERGE_MIN_LINEARITY = 8.0
LINEAR_MERGE_MAX_DIR_ANGLE_DEG = 10.0
LINEAR_MERGE_MAX_CENTER_ANGLE_DEG = 8.0
LINEAR_MERGE_MAX_PERP_OFFSET_PX = 6.0
LINEAR_MERGE_MAX_GAP_PX = 250.0
CENTERLINE_MIN_COMPONENT_PX = 2
CENTERLINE_MIN_LENGTH_M = 0.20
CENTERLINE_SMOOTH_M = 0.10
ENDPOINT_LINK_MAX_GAP_M = 5.0


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
        raise FileNotFoundError(f"Missing sidewalk inputs: {details}")
    return assets


def _write_mask(mask: np.ndarray, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(out_path)
    return out_path


def _write_rgb(mask: np.ndarray, added: np.ndarray, out_path: Path, *, added_color: tuple[int, int, int]) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask] = (255, 255, 255)
    rgb[added] = added_color
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
            diff = np.abs(src - dst)
            hit = valid & (diff >= HEIGHT_DIFF_MIN_M) & (diff <= HEIGHT_DIFF_MAX_M)
            target[src_r0:src_r1, src_c0:src_c1] |= hit
    return target


def _filter_by_label_buffer(diff: np.ndarray, label_map: np.ndarray, *, mpp: float) -> np.ndarray:
    radius_px = max(1, int(round(LABEL_BUFFER_M / mpp)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius_px * 2 + 1, radius_px * 2 + 1))
    buffered = cv2.dilate((label_map >= 0).astype(np.uint8), kernel, iterations=1) > 0
    return diff & buffered


def _close_and_area_filter(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.erode(cv2.dilate(mask.astype(np.uint8), kernel, iterations=1), kernel, iterations=1) > 0
    return _area_filter(closed, AREA_FILTER_MIN_PX)


def _area_filter(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros(mask.shape, dtype=bool)
    for label_id in range(1, num):
        if int(stats[label_id, cv2.CC_STAT_AREA]) >= int(min_area_px):
            out[labels == label_id] = True
    return out


def _offsets_for_angle(deg: float, radius: int) -> list[tuple[int, int]]:
    rad = math.radians(deg)
    dr_unit = math.sin(rad)
    dc_unit = math.cos(rad)
    offsets: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for t in range(-radius, radius + 1):
        dr = int(round(t * dr_unit))
        dc = int(round(t * dc_unit))
        if dr * dr + dc * dc > radius * radius + 1:
            continue
        if (dr, dc) not in seen:
            offsets.append((dr, dc))
            seen.add((dr, dc))
    return offsets


def _shifted_sum(mask: np.ndarray, offsets: list[tuple[int, int]]) -> np.ndarray:
    acc = np.zeros(mask.shape, dtype=np.uint8)
    for dr, dc in offsets:
        src_r0 = max(0, -dr)
        src_r1 = mask.shape[0] - max(0, dr)
        src_c0 = max(0, -dc)
        src_c1 = mask.shape[1] - max(0, dc)
        dst_r0 = max(0, dr)
        dst_r1 = mask.shape[0] - max(0, -dr)
        dst_c0 = max(0, dc)
        dst_c1 = mask.shape[1] - max(0, -dc)
        acc[dst_r0:dst_r1, dst_c0:dst_c1] += mask[src_r0:src_r1, src_c0:src_c1]
    return acc


def _local_thin_filter(mask: np.ndarray) -> np.ndarray:
    keep = np.zeros(mask.shape, dtype=bool)
    for angle in range(0, 180, LOCAL_THIN_ANGLE_STEP_DEG):
        long_count = _shifted_sum(mask, _offsets_for_angle(angle, LOCAL_THIN_RADIUS_PX))
        thin_count = _shifted_sum(mask, _offsets_for_angle(angle + 90, LOCAL_THIN_RADIUS_PX))
        keep |= (
            mask
            & (long_count >= LOCAL_THIN_MIN_LONG)
            & (thin_count <= LOCAL_THIN_MAX_THIN)
            & (long_count >= thin_count.astype(np.float32) * LOCAL_THIN_RATIO)
        )
    return _area_filter(keep, LOCAL_THIN_POST_MIN_AREA_PX)


def _component_records(mask: np.ndarray, min_area_px: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    ids = [i for i in range(1, num) if int(stats[i, cv2.CC_STAT_AREA]) >= int(min_area_px)]
    return labels, stats, ids


def _component_crop(labels: np.ndarray, stats: np.ndarray, comp_id: int) -> tuple[int, int, int, int, np.ndarray]:
    x = int(stats[comp_id, cv2.CC_STAT_LEFT])
    y = int(stats[comp_id, cv2.CC_STAT_TOP])
    w = int(stats[comp_id, cv2.CC_STAT_WIDTH])
    h = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
    return x, y, w, h, labels[y : y + h, x : x + w] == comp_id


def _component_direction(labels: np.ndarray, stats: np.ndarray, comp_id: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    x, y, _w, _h, crop = _component_crop(labels, stats, comp_id)
    rr, cc = np.nonzero(crop)
    if len(rr) < 3:
        return None, None
    pts_xy = np.column_stack([cc.astype(np.float64), rr.astype(np.float64)])
    center_xy = pts_xy.mean(axis=0)
    try:
        _u, _s, vt = np.linalg.svd(pts_xy - center_xy, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, None
    vec_xy = vt[0]
    vec_rc = np.array([vec_xy[1], vec_xy[0]], dtype=np.float64)
    norm = float(np.linalg.norm(vec_rc))
    if norm <= 1e-9:
        return None, None
    return vec_rc / norm, np.array([center_xy[1] + y, center_xy[0] + x], dtype=np.float64)


def _component_endpoints(labels: np.ndarray, stats: np.ndarray, comp_id: int) -> list[tuple[int, int]]:
    x, y, _w, _h, crop = _component_crop(labels, stats, comp_id)
    if not np.any(crop):
        return []
    skel = _skeletonize_mask(crop)
    counts = cv2.filter2D(
        skel.astype(np.uint8),
        ddepth=-1,
        kernel=np.ones((3, 3), dtype=np.uint8),
        borderType=cv2.BORDER_CONSTANT,
    )
    ep = np.column_stack(np.nonzero(skel & (counts == 2)))
    if len(ep) >= 2:
        pts = ep.astype(np.float64)
        best = (0.0, ep[0], ep[-1])
        for i in range(len(ep)):
            if i + 1 >= len(ep):
                continue
            d = np.sum((pts[i + 1 :] - pts[i]) ** 2, axis=1)
            if len(d) and float(d.max()) > best[0]:
                j = i + 1 + int(np.argmax(d))
                best = (float(d.max()), ep[i], ep[j])
        return [(int(best[1][0] + y), int(best[1][1] + x)), (int(best[2][0] + y), int(best[2][1] + x))]
    rr, cc = np.nonzero(crop)
    if len(rr) < 2:
        return []
    pts = np.column_stack([cc.astype(np.float64), rr.astype(np.float64)])
    center = pts.mean(axis=0)
    try:
        _u, _s, vt = np.linalg.svd(pts - center, full_matrices=False)
    except np.linalg.LinAlgError:
        return []
    proj = (pts - center) @ vt[0]
    lo = int(np.argmin(proj))
    hi = int(np.argmax(proj))
    return [(int(rr[lo] + y), int(cc[lo] + x)), (int(rr[hi] + y), int(cc[hi] + x))]


def _endpoint_grow_component_line(seed: np.ndarray, diff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels, stats, component_ids = _component_records(seed, LOCAL_THIN_POST_MIN_AREA_PX)
    h, w = seed.shape
    owner = labels.copy()
    combined = seed.copy()
    added = np.zeros(seed.shape, dtype=bool)
    for comp_id in component_ids:
        base_dir, comp_center = _component_direction(owner, stats, comp_id)
        if base_dir is None or comp_center is None:
            continue
        for ep_rc in _component_endpoints(owner, stats, comp_id):
            start = np.array(ep_rc, dtype=np.float64)
            direction = base_dir.copy()
            if float(np.dot(direction, start - comp_center)) < 0:
                direction = -direction
            gap = 0
            seen: set[tuple[int, int]] = set()
            t = 1.0
            while True:
                p = start + direction * t
                rr = int(round(float(p[0])))
                cc = int(round(float(p[1])))
                t += 1.0
                if rr < 0 or rr >= h or cc < 0 or cc >= w:
                    break
                key = (rr, cc)
                if key in seen:
                    continue
                seen.add(key)
                if owner[rr, cc] > 0 and owner[rr, cc] != comp_id:
                    break
                if diff[rr, cc] and owner[rr, cc] == 0:
                    owner[rr, cc] = comp_id
                    combined[rr, cc] = True
                    added[rr, cc] = True
                    gap = 0
                else:
                    gap += 1
                    if gap >= ENDPOINT_GROW_GAP_LIMIT_PX:
                        break
    return combined, added


def _angle_diff_deg(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-9 or nb <= 1e-9:
        return 90.0
    c = abs(float(np.dot(a / na, b / nb)))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _endpoint_direction(labels: np.ndarray, comp_id: int, endpoint_rc: np.ndarray, radius_px: int) -> np.ndarray | None:
    h, w = labels.shape
    r = int(endpoint_rc[0])
    c = int(endpoint_rc[1])
    r0 = max(0, r - radius_px)
    r1 = min(h, r + radius_px + 1)
    c0 = max(0, c - radius_px)
    c1 = min(w, c + radius_px + 1)
    local = labels[r0:r1, c0:c1] == comp_id
    rr, cc = np.nonzero(local)
    if len(rr) < 3:
        return None
    pts = np.column_stack([rr + r0, cc + c0]).astype(np.float64)
    d2 = np.sum((pts - endpoint_rc[None, :]) ** 2, axis=1)
    pts = pts[np.argsort(d2)[:40]]
    pts_xy = np.column_stack([pts[:, 1], pts[:, 0]])
    center = pts_xy.mean(axis=0)
    try:
        _u, _s, vt = np.linalg.svd(pts_xy - center, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    vec_xy = vt[0]
    ep_xy = np.array([endpoint_rc[1], endpoint_rc[0]], dtype=np.float64)
    if float(np.dot(vec_xy, ep_xy - center)) < 0:
        vec_xy = -vec_xy
    vec_rc = np.array([vec_xy[1], vec_xy[0]], dtype=np.float64)
    norm = float(np.linalg.norm(vec_rc))
    return None if norm <= 1e-9 else vec_rc / norm


def _raster_link_breaks(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels, stats, component_ids = _component_records(mask, RASTER_LINK_MIN_COMPONENT_PX)
    endpoints: list[dict[str, Any]] = []
    for comp_id in component_ids:
        for ep in _component_endpoints(labels, stats, comp_id):
            ep_rc = np.array(ep, dtype=np.float64)
            direction = _endpoint_direction(labels, comp_id, ep_rc, RASTER_LINK_ENDPOINT_RADIUS_PX)
            if direction is not None:
                endpoints.append({"comp_id": comp_id, "rc": ep_rc, "dir": direction})
    candidates: list[tuple[float, int, int]] = []
    for i, a in enumerate(endpoints):
        for j in range(i + 1, len(endpoints)):
            b = endpoints[j]
            if a["comp_id"] == b["comp_id"]:
                continue
            delta = b["rc"] - a["rc"]
            dist = float(np.linalg.norm(delta))
            if dist <= 1e-6 or dist > RASTER_LINK_MAX_GAP_PX:
                continue
            link_dir = delta / dist
            if _angle_diff_deg(a["dir"], link_dir) > RASTER_LINK_MAX_ANGLE_DEG:
                continue
            if _angle_diff_deg(b["dir"], -link_dir) > RASTER_LINK_MAX_ANGLE_DEG:
                continue
            candidates.append((dist, i, j))
    candidates.sort(key=lambda x: x[0])
    used: set[int] = set()
    link_mask = np.zeros(mask.shape, dtype=np.uint8)
    for _dist, i, j in candidates:
        if i in used or j in used:
            continue
        a = endpoints[i]
        b = endpoints[j]
        temp = np.zeros(mask.shape, dtype=np.uint8)
        cv2.line(
            temp,
            (int(round(a["rc"][1])), int(round(a["rc"][0]))),
            (int(round(b["rc"][1])), int(round(b["rc"][0]))),
            color=1,
            thickness=1,
            lineType=cv2.LINE_8,
        )
        new_pixels = (temp > 0) & ~mask
        if np.any(new_pixels):
            link_mask[new_pixels] = 255
            used.add(i)
            used.add(j)
    return mask | (link_mask > 0), link_mask > 0


def _linear_component_merge(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    components: list[dict[str, Any]] = []
    for comp_id in range(1, num):
        area = int(stats[comp_id, cv2.CC_STAT_AREA])
        if area < LINEAR_MERGE_MIN_AREA_PX:
            continue
        x, y, _w, _h, crop = _component_crop(labels, stats, comp_id)
        rr, cc = np.nonzero(crop)
        if len(rr) < 3:
            continue
        pts_rc = np.column_stack([rr + y, cc + x]).astype(np.float64)
        pts_xy = np.column_stack([pts_rc[:, 1], pts_rc[:, 0]])
        center_xy = pts_xy.mean(axis=0)
        try:
            _u, s, vt = np.linalg.svd(pts_xy - center_xy, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        linearity = float("inf") if len(s) < 2 or float(s[1]) <= 1e-9 else float(s[0] / s[1])
        vec_xy = vt[0]
        direction = np.array([vec_xy[1], vec_xy[0]], dtype=np.float64)
        direction /= max(float(np.linalg.norm(direction)), 1e-9)
        center = np.array([center_xy[1], center_xy[0]], dtype=np.float64)
        proj = (pts_rc - center[None, :]) @ direction
        half_len = max(abs(float(np.min(proj))), abs(float(np.max(proj))))
        major_len = float(np.max(proj) - np.min(proj))
        if linearity < LINEAR_MERGE_MIN_LINEARITY or major_len < LINEAR_MERGE_MIN_MAJOR_LENGTH_PX:
            continue
        components.append(
            {
                "id": comp_id,
                "center": center,
                "direction": direction,
                "half_len": half_len,
            }
        )
    candidates: list[tuple[float, float, int, int, float]] = []
    for i, a in enumerate(components):
        da = a["direction"]
        for j in range(i + 1, len(components)):
            b = components[j]
            if _angle_diff_deg(da, b["direction"]) > LINEAR_MERGE_MAX_DIR_ANGLE_DEG:
                continue
            delta = b["center"] - a["center"]
            dist = float(np.linalg.norm(delta))
            if dist <= 1e-6 or _angle_diff_deg(da, delta) > LINEAR_MERGE_MAX_CENTER_ANGLE_DEG:
                continue
            proj_delta = float(np.dot(delta, da))
            perp = float(np.linalg.norm(delta - proj_delta * da))
            gap = abs(proj_delta) - (a["half_len"] + b["half_len"])
            if perp <= LINEAR_MERGE_MAX_PERP_OFFSET_PX and 0.0 < gap <= LINEAR_MERGE_MAX_GAP_PX:
                candidates.append((gap, dist, i, j, proj_delta))
    candidates.sort(key=lambda x: (x[0], x[1]))
    used_sides: set[tuple[int, int]] = set()
    link_mask = np.zeros(mask.shape, dtype=np.uint8)
    for _gap, _dist, i, j, proj_delta in candidates:
        a = components[i]
        b = components[j]
        direction = a["direction"]
        sign = 1 if proj_delta > 0 else -1
        if (a["id"], sign) in used_sides or (b["id"], -sign) in used_sides:
            continue
        p1 = a["center"] + direction * a["half_len"] * sign
        p2 = b["center"] - direction * b["half_len"] * sign
        temp = np.zeros(mask.shape, dtype=np.uint8)
        cv2.line(
            temp,
            (int(round(float(p1[1]))), int(round(float(p1[0])))),
            (int(round(float(p2[1]))), int(round(float(p2[0])))),
            color=1,
            thickness=1,
            lineType=cv2.LINE_8,
        )
        line_bool = temp > 0
        line_labels = labels[line_bool]
        other_cross = (line_labels > 0) & (line_labels != a["id"]) & (line_labels != b["id"])
        if int(np.count_nonzero(other_cross)) > 5:
            continue
        new_pixels = line_bool & ~mask
        if np.any(new_pixels):
            link_mask[new_pixels] = 255
            used_sides.add((a["id"], sign))
            used_sides.add((b["id"], -sign))
    return mask | (link_mask > 0), link_mask > 0


def _extract_centerlines(mask: np.ndarray, meta: dict[str, Any]) -> list[dict[str, Any]]:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    mpp = float(meta["meters_per_pixel"])
    records: list[dict[str, Any]] = []
    for comp_id in range(1, num):
        area = int(stats[comp_id, cv2.CC_STAT_AREA])
        if area < CENTERLINE_MIN_COMPONENT_PX:
            continue
        x, y, w, h, crop = _component_crop(labels, stats, comp_id)
        skel = _skeletonize_mask(crop)
        comps = _connected_components_sorted(skel)
        if not comps:
            continue
        pix_rc = np.asarray(_chain_components(comps), dtype=np.float64)
        if len(pix_rc) < 2:
            continue
        pix_rc = _smooth_polyline(pix_rc, smooth_sigma_px=max(0.0, CENTERLINE_SMOOTH_M / mpp))
        if len(pix_rc) < 2:
            continue
        pix_rc[:, 0] += y
        pix_rc[:, 1] += x
        xy = pixel_to_xy(pix_rc[:, [1, 0]], meta)
        length_m = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
        if length_m < CENTERLINE_MIN_LENGTH_M:
            continue
        records.append(
            {
                "id": len(records),
                "component_id": int(comp_id),
                "area_px": area,
                "length_m": length_m,
                "n_pts": int(len(xy)),
                "xy": xy,
            }
        )
    return records


def _write_centerline_shp(records: list[dict[str, Any]], out_base: Path) -> Path:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    writer = shapefile.Writer(str(out_base))
    writer.shapeType = shapefile.POLYLINE
    writer.field("id", "N", decimal=0)
    writer.field("comp_id", "N", decimal=0)
    writer.field("area_px", "N", decimal=0)
    writer.field("length_m", "F", decimal=3)
    writer.field("n_pts", "N", decimal=0)
    for rec in records:
        writer.line([[[float(x), float(y)] for x, y in rec["xy"]]])
        writer.record(
            id=int(rec["id"]),
            comp_id=int(rec["component_id"]),
            area_px=int(rec["area_px"]),
            length_m=float(rec["length_m"]),
            n_pts=int(rec["n_pts"]),
        )
    writer.close()
    return out_base.with_suffix(".shp")


def _write_endpoint_linked_shp(records: list[dict[str, Any]], out_base: Path) -> tuple[Path, int]:
    endpoints: list[dict[str, Any]] = []
    for idx, rec in enumerate(records):
        xy = np.asarray(rec["xy"], dtype=np.float64)
        if len(xy) < 2:
            continue
        endpoints.append({"line_idx": idx, "which": 0, "xy": xy[0]})
        endpoints.append({"line_idx": idx, "which": 1, "xy": xy[-1]})
    candidates: list[tuple[float, int, int]] = []
    for i, a in enumerate(endpoints):
        for j in range(i + 1, len(endpoints)):
            b = endpoints[j]
            if a["line_idx"] == b["line_idx"]:
                continue
            dist = float(np.linalg.norm(a["xy"] - b["xy"]))
            if 1e-9 < dist <= ENDPOINT_LINK_MAX_GAP_M:
                candidates.append((dist, i, j))
    candidates.sort(key=lambda x: x[0])
    used: set[int] = set()
    links: list[dict[str, Any]] = []
    for dist, i, j in candidates:
        if i in used or j in used:
            continue
        links.append({"a": endpoints[i], "b": endpoints[j], "dist_m": dist})
        used.add(i)
        used.add(j)

    writer = shapefile.Writer(str(out_base))
    writer.shapeType = shapefile.POLYLINE
    writer.field("id", "N", decimal=0)
    writer.field("src", "C", size=16)
    writer.field("line_a", "N", decimal=0)
    writer.field("line_b", "N", decimal=0)
    writer.field("length_m", "F", decimal=3)
    writer.field("n_pts", "N", decimal=0)
    out_id = 0
    for idx, rec in enumerate(records):
        xy = np.asarray(rec["xy"], dtype=np.float64)
        writer.line([[[float(x), float(y)] for x, y in xy]])
        writer.record(id=out_id, src="orig", line_a=idx, line_b=-1, length_m=float(rec["length_m"]), n_pts=len(xy))
        out_id += 1
    for link in links:
        a = link["a"]
        b = link["b"]
        xy = np.vstack([a["xy"], b["xy"]])
        writer.line([[[float(x), float(y)] for x, y in xy]])
        writer.record(
            id=out_id,
            src="endpoint_link",
            line_a=int(a["line_idx"]),
            line_b=int(b["line_idx"]),
            length_m=float(link["dist_m"]),
            n_pts=2,
        )
        out_id += 1
    writer.close()
    return out_base.with_suffix(".shp"), len(links)


def run_sidewalk(
    pre_part_dir: Path | str,
    output_dir: Path | str | None = None,
    *,
    label_map_path: Path | str | None = DEFAULT_LABEL_MAP_PATH,
) -> dict[str, Path]:
    pre_part_dir = Path(pre_part_dir).expanduser()
    map_dir = _default_map_dir(pre_part_dir)
    output_dir = Path(output_dir).expanduser() if output_dir is not None else map_dir / "sidewalk"
    debug_dir = output_dir / "debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    label_map_path = Path(label_map_path).expanduser() if label_map_path is not None else map_dir / "obj" / "sidewalk" / "result" / "label_map.npy"
    if not label_map_path.is_file():
        raise FileNotFoundError(label_map_path)

    assets = _resolve_pre_part_assets(pre_part_dir)
    height_values = np.load(assets["height_values"], mmap_mode="r")
    height_meta = _load_json(assets["height_meta"])
    label_map = np.load(label_map_path, mmap_mode="r")
    if tuple(label_map.shape) != tuple(height_values.shape):
        raise ValueError(f"label_map shape {label_map.shape} does not match height raster {height_values.shape}")

    diff = _height_diff_mask(height_values)
    _write_mask(diff, debug_dir / "diff.png")
    filtered = _filter_by_label_buffer(diff, label_map, mpp=float(height_meta["meters_per_pixel"]))
    _write_mask(filtered, debug_dir / "diff_filtered_by50cm.png")
    closed_area = _close_and_area_filter(filtered)
    _write_mask(closed_area, debug_dir / "diff_filtered_by50cm_close_5x5_area200.png")
    localthin = _local_thin_filter(closed_area)
    _write_mask(localthin, debug_dir / "diff_filtered_by50cm_close_5x5_area200_localthin_r10_ratio3.png")
    grown, grown_added = _endpoint_grow_component_line(localthin, diff)
    _write_mask(grown, debug_dir / "diff_localthin_endpoint_grow_componentline_binary.png")
    _write_rgb(localthin, grown_added, debug_dir / "diff_localthin_endpoint_grow_componentline_green.png", added_color=(0, 255, 0))
    linked, link_added = _raster_link_breaks(grown)
    _write_mask(linked, debug_dir / "diff_localthin_endpoint_grow_componentline_linkbreak_binary.png")
    _write_rgb(grown, link_added, debug_dir / "diff_localthin_endpoint_grow_componentline_linkbreak_red.png", added_color=(255, 0, 0))
    merged, merge_added = _linear_component_merge(linked)
    _write_mask(merged, debug_dir / "diff_localthin_component_centerline_merge_binary.png")
    _write_rgb(linked, merge_added, debug_dir / "diff_localthin_component_centerline_merge_yellow.png", added_color=(255, 255, 0))

    records = _extract_centerlines(merged, height_meta)
    centerline_shp = _write_centerline_shp(records, output_dir / "diff_localthin_component_centerline_merge_centerlines")
    final_shp, endpoint_links = _write_endpoint_linked_shp(
        records,
        output_dir / "diff_localthin_component_centerline_merge_centerlines_endpoint_linked",
    )
    summary_path = output_dir / "summary.json"
    summary = {
        "pre_part_dir": str(pre_part_dir),
        "label_map": str(label_map_path),
        "height_values": str(assets["height_values"]),
        "height_meta": str(assets["height_meta"]),
        "output_dir": str(output_dir),
        "debug_dir": str(debug_dir),
        "diff_pixels": int(np.count_nonzero(diff)),
        "filtered_pixels": int(np.count_nonzero(filtered)),
        "closed_area_pixels": int(np.count_nonzero(closed_area)),
        "localthin_pixels": int(np.count_nonzero(localthin)),
        "grown_added_pixels": int(np.count_nonzero(grown_added)),
        "raster_link_added_pixels": int(np.count_nonzero(link_added)),
        "linear_merge_added_pixels": int(np.count_nonzero(merge_added)),
        "centerlines_exported": len(records),
        "endpoint_links_added": endpoint_links,
        "centerline_shp": str(centerline_shp),
        "final_shp": str(final_shp),
        "endpoint_link_max_gap_m": ENDPOINT_LINK_MAX_GAP_M,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "final_shp": final_shp,
        "centerline_shp": centerline_shp,
        "summary": summary_path,
        "debug_dir": debug_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract sidewalk boundary centerlines from height-diff BEV.")
    parser.add_argument("pre_part_dir", help="pre-part output directory.")
    parser.add_argument("-o", "--output-dir", default=None, help="Output directory. Defaults to <map>/sidewalk.")
    parser.add_argument("--label-map", default=str(DEFAULT_LABEL_MAP_PATH), help="Existing sidewalk label_map.npy.")
    args = parser.parse_args()
    outputs = run_sidewalk(args.pre_part_dir, args.output_dir, label_map_path=args.label_map)
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
