"""Rectangular label-map vectorization tool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import shapefile
from PIL import Image
from plyfile import PlyData

from landmark.tools.to_shp.bbox_ops import get_init_bbox, vis_bbox


def _xy_to_pixel(xy: np.ndarray, bev_meta: dict[str, Any]) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32)
    min_xy = np.asarray(bev_meta["min_xy"], dtype=np.float32)
    mpp = float(bev_meta["meters_per_pixel"])
    w = int(bev_meta["width"])
    h = int(bev_meta["height"])
    col = (xy[:, 0] - min_xy[0]) / mpp
    row = (h - 1) - (xy[:, 1] - min_xy[1]) / mpp
    pix = np.stack([col, row], axis=-1)
    pix[:, 0] = np.clip(pix[:, 0], 0, w - 1)
    pix[:, 1] = np.clip(pix[:, 1], 0, h - 1)
    return pix


def _pca_axes(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = np.asarray(xy, dtype=np.float64)
    mean = xy.mean(axis=0)
    centered = xy - mean
    cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    a1 = vecs[:, order[0]]
    a2 = vecs[:, order[1]]
    a1 = a1 / (np.linalg.norm(a1) + 1e-12)
    a2 = a2 / (np.linalg.norm(a2) + 1e-12)
    return mean.astype(np.float32), a1.astype(np.float32), a2.astype(np.float32)


def _bbox_frame_from_corners(
    corners_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    c = np.asarray(corners_xy, dtype=np.float32)
    if c.ndim != 2 or c.shape[0] < 4 or c.shape[1] != 2:
        raise ValueError("corners_xy must be (4,2) or (N,2)")
    c4 = c[:4]
    origin = c4.mean(axis=0)
    edges = [c4[(i + 1) % 4] - c4[i] for i in range(4)]
    lens = [float(np.linalg.norm(e)) for e in edges]
    i_long = int(np.argmax(lens))
    axis1 = edges[i_long]
    axis1 = axis1 / (np.linalg.norm(axis1) + 1e-12)
    axis2 = np.array([-axis1[1], axis1[0]], dtype=np.float32)

    uv = (c4 - origin) @ np.stack([axis1, axis2], axis=1)
    umin, umax = float(np.min(uv[:, 0])), float(np.max(uv[:, 0]))
    vmin, vmax = float(np.min(uv[:, 1])), float(np.max(uv[:, 1]))
    return origin, axis1, axis2, (umin, umax, vmin, vmax)


def _otsu_threshold_1d(x: np.ndarray, bins: int = 256) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    xmin = float(x.min())
    xmax = float(x.max())
    if xmax <= xmin:
        return xmin
    hist, edges = np.histogram(x, bins=bins, range=(xmin, xmax))
    hist = hist.astype(np.float64)
    p = hist / (hist.sum() + 1e-12)
    omega = np.cumsum(p)
    mu = np.cumsum(p * (edges[:-1] + edges[1:]) * 0.5)
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    k = int(np.nanargmax(sigma_b2))
    return float((edges[k] + edges[k + 1]) * 0.5)


def _lane_mask_by_otsu(intensity: np.ndarray) -> np.ndarray:
    intensity = np.asarray(intensity, dtype=np.float64)
    thr = _otsu_threshold_1d(intensity)
    cls = intensity >= thr
    if cls.sum() == 0 or cls.sum() == cls.size:
        return cls
    m1 = float(np.mean(intensity[cls]))
    m0 = float(np.mean(intensity[~cls]))
    lane = cls if m1 >= m0 else ~cls
    if float(lane.mean()) > 0.85:
        lane = ~lane
    return lane


def _best_cut_1d(coord: np.ndarray, lane: np.ndarray, *, inside_is_leq: bool) -> float:
    coord = np.asarray(coord, dtype=np.float64)
    lane = np.asarray(lane, dtype=bool)
    finite = np.isfinite(coord)
    coord = coord[finite]
    lane = lane[finite]
    if coord.size < 20 or lane.size != coord.size:
        return float(np.median(coord)) if coord.size else 0.0

    order = np.argsort(coord)
    c = coord[order]
    y = lane[order]

    n = c.size
    step = max(1, n // 512)
    cand_idx = np.unique(np.concatenate([np.arange(0, n, step), np.array([n - 1])]))
    cands = c[cand_idx]

    lane_prefix = np.cumsum(y.astype(np.int64))
    total_lane = int(lane_prefix[-1])
    total = int(n)
    total_non = total - total_lane
    if total_lane <= 0 or total_non <= 0:
        return float(np.median(c))

    best_t = float(cands[0])
    best_score = -1.0
    for t in cands:
        if inside_is_leq:
            r = int(np.searchsorted(c, t, side="right"))
            lane_in = int(lane_prefix[r - 1]) if r > 0 else 0
            lane_out = total_lane - lane_in
            non_out = (total - r) - lane_out
        else:
            r = int(np.searchsorted(c, t, side="left"))
            lane_below = int(lane_prefix[r - 1]) if r > 0 else 0
            non_below = r - lane_below
            lane_in = total_lane - lane_below
            non_out = non_below

        tpr = lane_in / total_lane
        tnr = non_out / total_non
        score = tpr + tnr
        if score > best_score:
            best_score = score
            best_t = float(t)

    return best_t


def refine_bbox(
    points: np.ndarray,
    intensity: np.ndarray,
    bbox: dict[str, Any],
    bev_meta: dict[str, Any],
    *,
    max_segment_length_m: float = 15.0,
) -> dict[str, Any]:
    points = np.asarray(points)
    intensity = np.asarray(intensity)
    poly0 = np.asarray(bbox.get("corners_xy"), dtype=np.float32)
    if poly0.ndim != 2 or poly0.shape[1] != 2 or poly0.shape[0] < 3:
        return bbox

    try:
        origin, axis1, axis2, (umin0, umax0, vmin0, vmax0) = _bbox_frame_from_corners(poly0)
    except Exception:
        return bbox

    dxy = points[:, :2] - origin[None, :]
    u_all = dxy @ axis1
    v_all = dxy @ axis2

    expand_m = 0.10
    umin_e = umin0 - expand_m
    umax_e = umax0 + expand_m
    vmin_e = vmin0 - expand_m
    vmax_e = vmax0 + expand_m
    n1 = (u_all >= umin_e) & (u_all <= umax_e) & (v_all >= vmin_e) & (v_all <= vmax_e)
    if int(n1.sum()) < 80:
        return bbox

    length0 = float(umax0 - umin0)
    segments = max(1, int(np.ceil(length0 / float(max_segment_length_m))))
    u_edges = np.linspace(umin0, umax0, segments + 1, dtype=np.float32)

    v_low_seg = np.full((segments,), np.nan, dtype=np.float32)
    v_high_seg = np.full((segments,), np.nan, dtype=np.float32)

    v_center0 = 0.5 * (vmin0 + vmax0)
    max_long_shift = 0.20
    max_short_shift = 0.20

    for s in range(segments):
        u0 = float(u_edges[s])
        u1 = float(u_edges[s + 1])
        seg = n1 & (u_all >= (u0 - expand_m)) & (u_all <= (u1 + expand_m))
        if int(seg.sum()) < 50:
            continue
        vv = v_all[seg]
        ii = intensity[seg]

        v_center_seg = float(np.median(vv)) if vv.size else float(v_center0)
        upper = vv >= v_center_seg
        lower = ~upper

        if int(upper.sum()) >= 20:
            lane_u = _lane_mask_by_otsu(ii[upper])
            t_up = _best_cut_1d(vv[upper], lane_u, inside_is_leq=True)
            v_high_seg[s] = float(np.clip(t_up, vmax0 - max_long_shift, vmax0 + max_long_shift))

        if int(lower.sum()) >= 20:
            lane_l = _lane_mask_by_otsu(ii[lower])
            t_lo = _best_cut_1d(vv[lower], lane_l, inside_is_leq=False)
            v_low_seg[s] = float(np.clip(t_lo, vmin0 - max_long_shift, vmin0 + max_long_shift))

    if not np.any(np.isfinite(v_low_seg)):
        v_low_seg[:] = float(vmin0)
    else:
        last = float(vmin0)
        for i in range(segments):
            if np.isfinite(v_low_seg[i]):
                last = float(v_low_seg[i])
            else:
                v_low_seg[i] = last

    if not np.any(np.isfinite(v_high_seg)):
        v_high_seg[:] = float(vmax0)
    else:
        last = float(vmax0)
        for i in range(segments):
            if np.isfinite(v_high_seg[i]):
                last = float(v_high_seg[i])
            else:
                v_high_seg[i] = last

    v_low_k = np.empty((segments + 1,), dtype=np.float32)
    v_high_k = np.empty((segments + 1,), dtype=np.float32)
    v_low_k[0] = v_low_seg[0]
    v_high_k[0] = v_high_seg[0]
    v_low_k[-1] = v_low_seg[-1]
    v_high_k[-1] = v_high_seg[-1]
    for i in range(1, segments):
        v_low_k[i] = 0.5 * (v_low_seg[i - 1] + v_low_seg[i])
        v_high_k[i] = 0.5 * (v_high_seg[i - 1] + v_high_seg[i])

    swap = v_low_k > v_high_k
    if np.any(swap):
        tmp = v_low_k[swap].copy()
        v_low_k[swap] = v_high_k[swap]
        v_high_k[swap] = tmp

    umin_r = float(umin0)
    umax_r = float(umax0)
    short_band = 0.02
    v_low_left = float(v_low_k[0])
    v_high_left = float(v_high_k[0])
    v_low_right = float(v_low_k[-1])
    v_high_right = float(v_high_k[-1])

    if segments == 1:
        n2 = n1 & (v_all >= float(np.min(v_low_k))) & (v_all <= float(np.max(v_high_k)))
        left_band = n2 & (u_all >= (umin0 - short_band)) & (u_all <= (umin0 + short_band))
        right_band = n2 & (u_all >= (umax0 - short_band)) & (u_all <= (umax0 + short_band))

        if int(left_band.sum()) >= 30:
            lane = _lane_mask_by_otsu(intensity[left_band])
            t = float(_best_cut_1d(u_all[left_band], lane, inside_is_leq=False))
            umin_r = float(np.clip(t, umin0 - max_short_shift, umin0 + max_short_shift))
        if int(right_band.sum()) >= 30:
            lane = _lane_mask_by_otsu(intensity[right_band])
            t = float(_best_cut_1d(u_all[right_band], lane, inside_is_leq=True))
            umax_r = float(np.clip(t, umax0 - max_short_shift, umax0 + max_short_shift))

        u_edges = np.array([umin_r, umax_r], dtype=np.float32)
    else:
        n2_left = n1 & (v_all >= (v_low_left - expand_m)) & (v_all <= (v_high_left + expand_m))
        n2_right = n1 & (v_all >= (v_low_right - expand_m)) & (v_all <= (v_high_right + expand_m))
        left_band = n2_left & (u_all >= (umin0 - short_band)) & (u_all <= (umin0 + short_band))
        if int(left_band.sum()) >= 30:
            lane = _lane_mask_by_otsu(intensity[left_band])
            t = float(_best_cut_1d(u_all[left_band], lane, inside_is_leq=False))
            umin_r = float(np.clip(t, umin0 - max_short_shift, umin0 + max_short_shift))
        right_band = n2_right & (u_all >= (umax0 - short_band)) & (u_all <= (umax0 + short_band))
        if int(right_band.sum()) >= 30:
            lane = _lane_mask_by_otsu(intensity[right_band])
            t = float(_best_cut_1d(u_all[right_band], lane, inside_is_leq=True))
            umax_r = float(np.clip(t, umax0 - max_short_shift, umax0 + max_short_shift))

        u_edges[0] = float(umin_r)
        u_edges[-1] = float(umax_r)

    lower = np.stack([u_edges, v_low_k], axis=-1)
    upper = np.stack([u_edges[::-1], v_high_k[::-1]], axis=-1)
    poly_uv = np.concatenate([lower, upper], axis=0).astype(np.float32)
    poly_xy = origin[None, :] + poly_uv[:, 0:1] * axis1[None, :] + poly_uv[:, 1:2] * axis2[None, :]
    poly_xy = poly_xy.astype(np.float32)

    umin_g = float(np.min(u_edges))
    umax_g = float(np.max(u_edges))
    vmin_g = float(np.min(v_low_k))
    vmax_g = float(np.max(v_high_k))
    inside_uv = (u_all >= umin_g) & (u_all <= umax_g) & (v_all >= vmin_g) & (v_all <= vmax_g)
    if int(inside_uv.sum()) > 0:
        z_mean = float(points[inside_uv, 2].mean())
    else:
        z_mean = float(points[n1, 2].mean())

    center_xy = poly_xy.mean(axis=0)
    origin2, a1, a2 = _pca_axes(poly_xy)
    yaw = float(np.arctan2(a1[1], a1[0]))
    uv2 = (poly_xy - origin2) @ np.stack([a1, a2], axis=1)
    size_lw = [
        float(np.max(uv2[:, 0]) - np.min(uv2[:, 0])),
        float(np.max(uv2[:, 1]) - np.min(uv2[:, 1])),
    ]

    refined = dict(bbox)
    refined["center"] = [float(center_xy[0]), float(center_xy[1]), z_mean]
    refined["yaw"] = yaw
    refined["size"] = size_lw
    refined["corners_xy"] = poly_xy.tolist()
    refined["pixel_corners"] = _xy_to_pixel(poly_xy, bev_meta).tolist()
    return refined


def refine_bboxs(
    points: np.ndarray,
    intensity: np.ndarray,
    bbox_list: list[dict[str, Any]],
    bev_meta: dict[str, Any],
    *,
    max_segment_length_m: float = 15.0,
) -> list[dict[str, Any]]:
    return [
        refine_bbox(
            points,
            intensity,
            bbox,
            bev_meta,
            max_segment_length_m=max_segment_length_m,
        )
        for bbox in bbox_list
    ]


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


def _iter_vertex_chunks(vertex_data: np.ndarray, chunk_size: int = 2_000_000):
    total = int(len(vertex_data))
    for start in range(0, total, int(chunk_size)):
        stop = min(total, start + int(chunk_size))
        yield vertex_data[start:stop]


def _bbox_local_frames(
    bbox_list: list[dict[str, Any]],
    *,
    buffer_m: float,
) -> list[dict[str, Any] | None]:
    frames: list[dict[str, Any] | None] = []
    for bbox in bbox_list:
        try:
            corners = np.asarray(bbox["corners_xy"], dtype=np.float32)
            origin, axis1, axis2, (umin, umax, vmin, vmax) = _bbox_frame_from_corners(corners)
        except Exception:
            frames.append(None)
            continue
        buffer_m = float(buffer_m)
        frames.append(
            {
                "origin": origin,
                "axis1": axis1,
                "axis2": axis2,
                "umin": float(umin - buffer_m),
                "umax": float(umax + buffer_m),
                "vmin": float(vmin - buffer_m),
                "vmax": float(vmax + buffer_m),
                "xmin": float(np.min(corners[:, 0]) - buffer_m),
                "xmax": float(np.max(corners[:, 0]) + buffer_m),
                "ymin": float(np.min(corners[:, 1]) - buffer_m),
                "ymax": float(np.max(corners[:, 1]) + buffer_m),
            }
        )
    return frames


def _load_local_points_and_intensity_by_bbox(
    ply_path: Path | str,
    bbox_list: list[dict[str, Any]],
    *,
    local_buffer_m: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if not bbox_list:
        return []

    ply_path = Path(ply_path).expanduser()
    ply = PlyData.read(str(ply_path), mmap=True)
    vertex = ply["vertex"]
    vertex_data = vertex.data
    vertex_count = int(vertex.count)
    names = vertex_data.dtype.names or ()
    required = {"x", "y", "z"}
    if not required.issubset(names):
        raise KeyError(f"PLY vertex fields must include {sorted(required)}")

    intensity_name = None
    for key in ("scalar_Intensity", "intensity"):
        if key in names:
            intensity_name = key
            break

    frames = _bbox_local_frames(bbox_list, buffer_m=local_buffer_m)
    point_chunks: list[list[np.ndarray]] = [[] for _ in bbox_list]
    intensity_chunks: list[list[np.ndarray]] = [[] for _ in bbox_list]

    print(
        f"[box_shp] collecting local point clouds: bboxes={len(bbox_list)} "
        f"points={vertex_count} buffer={float(local_buffer_m):.2f}m",
        flush=True,
    )
    processed = 0
    for chunk_idx, chunk in enumerate(_iter_vertex_chunks(vertex_data), start=1):
        processed += int(len(chunk))
        points = np.stack(
            [
                np.asarray(chunk["x"], dtype=np.float32),
                np.asarray(chunk["y"], dtype=np.float32),
                np.asarray(chunk["z"], dtype=np.float32),
            ],
            axis=-1,
        )
        if intensity_name is None:
            intensity = np.zeros(points.shape[0], dtype=np.float32)
        else:
            intensity = np.asarray(chunk[intensity_name], dtype=np.float32)
        x = points[:, 0]
        y = points[:, 1]
        for idx, frame in enumerate(frames):
            if frame is None:
                continue
            coarse = (
                (x >= frame["xmin"])
                & (x <= frame["xmax"])
                & (y >= frame["ymin"])
                & (y <= frame["ymax"])
            )
            if not np.any(coarse):
                continue
            candidate_points = points[coarse]
            dxy = candidate_points[:, :2] - frame["origin"][None, :]
            u = dxy @ frame["axis1"]
            v = dxy @ frame["axis2"]
            keep = (
                (u >= frame["umin"])
                & (u <= frame["umax"])
                & (v >= frame["vmin"])
                & (v <= frame["vmax"])
            )
            if not np.any(keep):
                continue
            candidate_intensity = intensity[coarse]
            point_chunks[idx].append(candidate_points[keep].astype(np.float32, copy=False))
            intensity_chunks[idx].append(candidate_intensity[keep].astype(np.float32, copy=False))
        if chunk_idx == 1 or processed >= vertex_count or chunk_idx % 10 == 0:
            print(
                f"[box_shp] local point collection: {processed}/{vertex_count}",
                flush=True,
            )

    local_sets: list[tuple[np.ndarray, np.ndarray]] = []
    for pts_parts, int_parts in zip(point_chunks, intensity_chunks):
        if pts_parts:
            local_points = np.concatenate(pts_parts, axis=0)
            local_intensity = np.concatenate(int_parts, axis=0)
        else:
            local_points = np.zeros((0, 3), dtype=np.float32)
            local_intensity = np.zeros((0,), dtype=np.float32)
        local_sets.append((local_points, local_intensity))
    return local_sets


def _attach_bbox_z_from_points(bbox: dict[str, Any], local_points: np.ndarray) -> dict[str, Any]:
    updated = dict(bbox)
    center = list(updated.get("center", [0.0, 0.0, 0.0]))
    while len(center) < 3:
        center.append(0.0)
    if local_points.size > 0:
        z = np.asarray(local_points[:, 2], dtype=np.float32)
        finite = np.isfinite(z)
        if np.any(finite):
            center[2] = float(np.mean(z[finite]))
    updated["center"] = [float(center[0]), float(center[1]), float(center[2])]
    return updated


def _refine_bboxs_with_local_points(
    bbox_list: list[dict[str, Any]],
    local_sets: list[tuple[np.ndarray, np.ndarray]],
    bev_meta: dict[str, Any],
    *,
    max_segment_length_m: float,
    debug: bool,
) -> list[dict[str, Any]]:
    refined: list[dict[str, Any]] = []
    total = len(bbox_list)
    for idx, (bbox, (local_points, local_intensity)) in enumerate(zip(bbox_list, local_sets), start=1):
        bbox_with_z = _attach_bbox_z_from_points(bbox, local_points)
        if debug:
            refined.append(bbox_with_z)
            continue
        if local_points.shape[0] < 80:
            print(
                f"[box_shp] refine {idx}/{total}: skip id={bbox.get('id')} local_points={local_points.shape[0]}",
                flush=True,
            )
            refined.append(bbox_with_z)
            continue
        print(
            f"[box_shp] refine {idx}/{total}: id={bbox.get('id')} local_points={local_points.shape[0]}",
            flush=True,
        )
        refined.append(
            refine_bbox(
                local_points,
                local_intensity,
                bbox_with_z,
                bev_meta,
                max_segment_length_m=max_segment_length_m,
            )
        )
    return refined


def _write_bbox_payload(path: Path, bev_meta: dict[str, Any], bboxes: list[dict[str, Any]]) -> None:
    payload = {"bev_meta": bev_meta, "bboxes": bboxes}
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_bbox_shp(bboxes: list[dict[str, Any]], output_dir: Path, *, shp_stem: str) -> Path:
    shp_path = output_dir / shp_stem
    w = shapefile.Writer(str(shp_path))
    w.shapeType = shapefile.POLYGON
    w.field("id", "N", decimal=0)
    w.field("length", "F", decimal=3)
    w.field("width", "F", decimal=3)
    w.field("yaw", "F", decimal=6)
    w.field("cx", "F", decimal=3)
    w.field("cy", "F", decimal=3)

    for bbox in bboxes:
        ring = [list(c[:2]) for c in bbox["corners_xy"]]
        ring.append(ring[0])
        w.poly([ring])
        w.record(
            id=int(bbox["id"]),
            length=float(bbox["size"][0]),
            width=float(bbox["size"][1]),
            yaw=float(bbox["yaw"]),
            cx=float(bbox["center"][0]),
            cy=float(bbox["center"][1]),
        )

    w.close()
    return Path(f"{shp_path}.shp")


def box_masks_to_shp(
    label_map: np.ndarray,
    geo_meta: dict[str, Any] | Path | str,
    ply_path: Path | str,
    output_dir: Path | str,
    *,
    max_segment_length_m: float = 15.0,
    local_buffer_m: float = 0.5,
    shp_stem: str = "box",
    debug: bool = False,
) -> Path:
    bev_meta = _coerce_bev_meta(geo_meta)
    masks_bool = _normalize_masks(label_map)
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    init_bboxes = get_init_bbox(masks_bool, bev_meta, output_dir)
    local_sets = _load_local_points_and_intensity_by_bbox(
        ply_path,
        init_bboxes,
        local_buffer_m=local_buffer_m,
    )
    refined_bboxes = _refine_bboxs_with_local_points(
        init_bboxes,
        local_sets,
        bev_meta,
        max_segment_length_m=max_segment_length_m,
        debug=debug,
    )

    _write_bbox_payload(output_dir / "refined_bboxes.json", bev_meta, refined_bboxes)

    base_img = _render_masks_overview(masks_bool, output_dir / "box_masks.png")
    init_vis, _ = vis_bbox(base_img, init_bboxes)
    Image.fromarray(init_vis).save(output_dir / "box_init_bboxes_vis.png")
    refined_vis, _ = vis_bbox(base_img, refined_bboxes)
    Image.fromarray(refined_vis).save(output_dir / "box_refined_bboxes_vis.png")

    shp_path = _write_bbox_shp(refined_bboxes, output_dir, shp_stem=shp_stem)
    print(f"[shp] wrote {len(refined_bboxes)} box features → {shp_path}", flush=True)
    return shp_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Vectorize rectangular instance label_map → box.shp.")
    parser.add_argument("label_map_npy", help="Path to label_map .npy file with shape (H,W).")
    parser.add_argument("geo_meta_json", help="Path to geo_meta.json or summary.json.")
    parser.add_argument("ply_path", help="PLY path used for bbox refinement.")
    parser.add_argument("-o", "--output-dir", default="outputs/apps/shp/box", help="Output directory.")
    parser.add_argument("--max-segment-length", type=float, default=15.0, help="Refinement segment length in meters.")
    parser.add_argument("--local-buffer", type=float, default=0.5, help="BBox buffer in meters used to select local point-cloud points.")
    parser.add_argument("--debug", action="store_true", help="Skip bbox refine and write init boxes directly.")
    args = parser.parse_args()

    masks = np.load(args.label_map_npy)
    box_masks_to_shp(
        masks,
        args.geo_meta_json,
        args.ply_path,
        args.output_dir,
        max_segment_length_m=args.max_segment_length,
        local_buffer_m=args.local_buffer,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
