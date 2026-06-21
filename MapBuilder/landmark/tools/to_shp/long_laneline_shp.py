"""long_laneline — vectorize long lane-line label maps into SHP polygons.

Algorithm
---------
For each object mask:

1. **Skeleton extraction** — ``skimage.morphology.skeletonize``.
2. **Connected-component ordering** — if one mask has multiple skeleton
   fragments, sort them along the principal direction and connect head→tail
   with straight segments.
3. **Control-point sampling** — walk along the ordered skeleton polyline at
   fixed intervals (default 10 cm), always including the first and the last point.
4. **Normal cross-section** — at each control point, cast a short segment
   perpendicular to the local tangent direction and find mask boundaries.
5. **Polygon construction** — left boundary polyline + reversed right
   boundary polyline → closed polygon.
6. **SHP export** — write all polygons as POLYGON features.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    from skimage.morphology import skeletonize
    return skeletonize(mask > 0)


def _connected_components_sorted(
    skel: np.ndarray,
) -> list[np.ndarray]:
    from scipy import ndimage

    labels, n = ndimage.label(skel)
    if n == 0:
        return []

    components: list[np.ndarray] = []
    for i in range(1, n + 1):
        ys, xs = np.where(labels == i)
        pts = np.stack([ys, xs], axis=1)
        if len(pts) < 2:
            continue
        pts = _order_points_along_curve(pts)
        components.append(pts)

    if len(components) <= 1:
        return components

    all_pts = np.vstack(components)
    mean = all_pts.mean(axis=0)
    centered = all_pts - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, -1]

    def _proj(comp: np.ndarray) -> float:
        c = comp.mean(axis=0) - mean
        return float(c @ principal)

    components.sort(key=_proj)

    for i in range(len(components) - 1):
        tail = components[i][-1]
        head_next = components[i + 1][0]
        tail_next = components[i + 1][-1]
        if np.linalg.norm(tail - tail_next) < np.linalg.norm(tail - head_next):
            components[i + 1] = components[i + 1][::-1]

    return components


def _order_points_along_curve(pts: np.ndarray) -> np.ndarray:
    if len(pts) <= 2:
        return pts

    centroid = pts.mean(axis=0)
    dists = np.linalg.norm(pts - centroid, axis=1)
    start = int(np.argmax(dists))

    from scipy.spatial import cKDTree

    tree = cKDTree(pts)
    visited = np.zeros(len(pts), dtype=bool)
    order = [start]
    visited[start] = True

    for _ in range(len(pts) - 1):
        cur = order[-1]
        k = min(len(pts), 20)
        dd, ii = tree.query(pts[cur], k=k)
        found = False
        for idx in ii:
            if not visited[idx]:
                order.append(idx)
                visited[idx] = True
                found = True
                break
        if not found:
            break

    return pts[order]


def _chain_components(components: list[np.ndarray]) -> np.ndarray:
    if not components:
        return np.empty((0, 2), dtype=np.float64)
    if len(components) == 1:
        return components[0].astype(np.float64)

    parts: list[np.ndarray] = [components[0].astype(np.float64)]
    for i in range(1, len(components)):
        tail = parts[-1][-1]
        head = components[i][0].astype(np.float64)
        dist = np.linalg.norm(head - tail)
        n_bridge = max(int(dist), 2)
        bridge = np.linspace(tail, head, n_bridge + 1)[1:-1]
        if len(bridge) > 0:
            parts.append(bridge)
        parts.append(components[i].astype(np.float64))

    return np.vstack(parts)


def _should_use_projection_polyline(mask: np.ndarray) -> bool:
    if mask.size <= 5_000_000:
        return False
    fill_ratio = float(np.count_nonzero(mask)) / float(mask.size)
    return fill_ratio < 0.02


def _polyline_from_sparse_projection(
    mask: np.ndarray,
    *,
    bin_size_px: float = 2.0,
) -> np.ndarray:
    pts = np.column_stack(np.where(mask)).astype(np.float64)
    if len(pts) < 2:
        return pts

    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, -1]
    if principal[0] < 0 or (abs(principal[0]) < 1e-9 and principal[1] < 0):
        principal = -principal

    u = centered @ principal
    bins = np.floor((u - float(u.min())) / bin_size_px).astype(np.int32)
    _, inv = np.unique(bins, return_inverse=True)
    counts = np.bincount(inv).astype(np.float64)
    sum_r = np.bincount(inv, weights=pts[:, 0])
    sum_c = np.bincount(inv, weights=pts[:, 1])
    polyline = np.column_stack([sum_r / counts, sum_c / counts])
    return polyline


def _cumulative_arclength(pts: np.ndarray) -> np.ndarray:
    diffs = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_lens)])


def _sample_control_points(
    pts: np.ndarray,
    interval_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    arc = _cumulative_arclength(pts)
    total = arc[-1]
    if total < 1e-6:
        return pts[:1], np.array([[1.0, 0.0]])

    n_samples = max(int(total / interval_px), 1) + 1
    sample_dists = np.linspace(0, total, n_samples)

    positions = np.zeros((n_samples, 2), dtype=np.float64)
    tangents = np.zeros((n_samples, 2), dtype=np.float64)

    seg_idx = 0
    for i, s in enumerate(sample_dists):
        while seg_idx < len(arc) - 2 and arc[seg_idx + 1] < s:
            seg_idx += 1
        seg_len = arc[seg_idx + 1] - arc[seg_idx]
        if seg_len < 1e-9:
            t = 0.0
        else:
            t = (s - arc[seg_idx]) / seg_len
        t = min(max(t, 0.0), 1.0)
        positions[i] = pts[seg_idx] * (1 - t) + pts[min(seg_idx + 1, len(pts) - 1)] * t
        tang = pts[min(seg_idx + 1, len(pts) - 1)] - pts[seg_idx]
        norm = np.linalg.norm(tang)
        if norm > 1e-9:
            tangents[i] = tang / norm
        else:
            tangents[i] = tangents[max(i - 1, 0)]

    return positions, tangents


def _find_boundary_on_normal(
    mask: np.ndarray,
    center: np.ndarray,
    normal: np.ndarray,
    half_width_px: float,
    *,
    step_px: float = 0.25,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    normal = np.asarray(normal, dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-9:
        return None, None
    normal = normal / norm

    h, w = mask.shape
    sample_ts = np.arange(-half_width_px, half_width_px + step_px, step_px, dtype=np.float64)

    inside_samples: list[tuple[float, np.ndarray]] = []
    for t in sample_ts:
        pt = center + normal * t
        r, c = int(round(pt[0])), int(round(pt[1]))
        if r < 0 or r >= h or c < 0 or c >= w:
            continue
        if mask[r, c]:
            inside_samples.append((float(t), np.array([pt[0], pt[1]], dtype=np.float64)))

    if not inside_samples:
        return None, None

    left = min(inside_samples, key=lambda item: item[0])[1]
    right = max(inside_samples, key=lambda item: item[0])[1]
    return left, right


def _adaptive_half_width_px(
    mask: np.ndarray,
    base_half_width_px: float,
) -> float:
    if mask.size == 0 or not np.any(mask):
        return base_half_width_px

    rows, cols = np.where(mask)
    if rows.size < 2:
        return base_half_width_px

    pts = np.column_stack([rows, cols]).astype(np.float64)
    center = pts.mean(axis=0)
    centered = pts - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, -1]
    normal = np.array([-principal[1], principal[0]], dtype=np.float64)

    transverse = centered @ normal
    lo = float(np.percentile(transverse, 1.0))
    hi = float(np.percentile(transverse, 99.0))
    robust_half_width_px = 0.5 * (hi - lo)
    return max(float(base_half_width_px), robust_half_width_px + 2.0)


def _smooth_polyline(
    pts: np.ndarray,
    *,
    smooth_sigma_px: float,
) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 5 or smooth_sigma_px <= 1e-6:
        return pts

    diffs = np.diff(pts, axis=0)
    step_lengths = np.linalg.norm(diffs, axis=1)
    step_lengths = step_lengths[step_lengths > 1e-6]
    avg_step_px = float(np.median(step_lengths)) if len(step_lengths) else 1.0
    sigma_samples = float(smooth_sigma_px) / avg_step_px
    if sigma_samples < 0.75:
        return pts

    from scipy.ndimage import gaussian_filter1d

    smoothed = gaussian_filter1d(pts, sigma=sigma_samples, axis=0, mode="nearest")
    smoothed[0] = pts[0]
    smoothed[-1] = pts[-1]

    preserve = min(len(pts) // 4, max(2, int(round(2.5 * sigma_samples))))
    for i in range(1, preserve):
        alpha = i / preserve
        smoothed[i] = pts[i] * (1.0 - alpha) + smoothed[i] * alpha
        smoothed[-1 - i] = pts[-1 - i] * (1.0 - alpha) + smoothed[-1 - i] * alpha

    return smoothed


def _angle_diff_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-9 or nb <= 1e-9:
        return 180.0
    c = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _stabilize_normals(
    normals: np.ndarray,
    *,
    jump_angle_deg: float = 35.0,
    neighbor_radius: int = 2,
) -> np.ndarray:
    normals = np.asarray(normals, dtype=np.float64).copy()
    if len(normals) == 0:
        return normals

    for i in range(len(normals)):
        n = float(np.linalg.norm(normals[i]))
        if n > 1e-9:
            normals[i] /= n

    for i in range(1, len(normals)):
        if float(np.dot(normals[i], normals[i - 1])) < 0.0:
            normals[i] = -normals[i]

    stabilized = normals.copy()
    for _ in range(2):
        updated = stabilized.copy()
        for i in range(len(stabilized)):
            lo = max(0, i - neighbor_radius)
            hi = min(len(stabilized), i + neighbor_radius + 1)
            neighbours = [stabilized[j] for j in range(lo, hi) if j != i]
            if len(neighbours) < 2:
                continue

            ref = np.mean(neighbours, axis=0)
            ref_norm = float(np.linalg.norm(ref))
            if ref_norm <= 1e-9:
                continue
            ref /= ref_norm

            aligned = []
            for nb in neighbours:
                aligned.append(nb if float(np.dot(nb, ref)) >= 0.0 else -nb)
            aligned = np.asarray(aligned, dtype=np.float64)
            ref = aligned.mean(axis=0)
            ref_norm = float(np.linalg.norm(ref))
            if ref_norm <= 1e-9:
                continue
            ref /= ref_norm

            max_nb_angle = max(_angle_diff_deg(nb, ref) for nb in aligned)
            cur_angle = _angle_diff_deg(stabilized[i], ref)
            if max_nb_angle <= jump_angle_deg and cur_angle > jump_angle_deg:
                updated[i] = ref
        stabilized = updated

    return stabilized


def _trace_long_laneline(
    mask: np.ndarray,
    bev_meta: dict[str, Any],
    *,
    sample_interval_m: float = 0.10,
    cross_half_width_m: float = 0.30,
    centerline_smooth_m: float = 0.10,
    row_offset: int = 0,
    col_offset: int = 0,
) -> tuple[list[list[float]] | None, dict[str, Any]]:
    from landmark.tools.to_shp.geometry import pixel_to_xy

    mpp = float(bev_meta["meters_per_pixel"])
    interval_px = sample_interval_m / mpp
    half_width_px = max(cross_half_width_m / mpp, 2.0)

    method = "skeleton"
    skel: np.ndarray | None = None
    if _should_use_projection_polyline(mask):
        method = "projection"
        polyline = _polyline_from_sparse_projection(mask)
    else:
        skel = _skeletonize_mask(mask)
        if not np.any(skel):
            return None, {
                "method": method,
                "mask": mask,
                "skeleton": skel,
                "raw_polyline": np.empty((0, 2), dtype=np.float64),
                "polyline": np.empty((0, 2), dtype=np.float64),
                "positions": np.empty((0, 2), dtype=np.float64),
                "left_pts": np.empty((0, 2), dtype=np.float64),
                "right_pts": np.empty((0, 2), dtype=np.float64),
                "slice_segments": [],
                "row_offset": row_offset,
                "col_offset": col_offset,
                "half_width_px": half_width_px,
            }

        components = _connected_components_sorted(skel)
        if not components:
            return None, {
                "method": method,
                "mask": mask,
                "skeleton": skel,
                "raw_polyline": np.empty((0, 2), dtype=np.float64),
                "polyline": np.empty((0, 2), dtype=np.float64),
                "positions": np.empty((0, 2), dtype=np.float64),
                "left_pts": np.empty((0, 2), dtype=np.float64),
                "right_pts": np.empty((0, 2), dtype=np.float64),
                "slice_segments": [],
                "row_offset": row_offset,
                "col_offset": col_offset,
                "half_width_px": half_width_px,
            }

        polyline = _chain_components(components)

    if len(polyline) < 2:
        return None, {
            "method": method,
            "mask": mask,
            "skeleton": skel,
            "raw_polyline": np.asarray(polyline, dtype=np.float64),
            "polyline": polyline,
            "positions": np.empty((0, 2), dtype=np.float64),
            "left_pts": np.empty((0, 2), dtype=np.float64),
            "right_pts": np.empty((0, 2), dtype=np.float64),
            "slice_segments": [],
            "row_offset": row_offset,
            "col_offset": col_offset,
            "half_width_px": half_width_px,
        }

    raw_polyline = np.asarray(polyline, dtype=np.float64)
    smooth_sigma_px = max(0.0, float(centerline_smooth_m) / mpp)
    polyline = _smooth_polyline(raw_polyline, smooth_sigma_px=smooth_sigma_px)

    half_width_px = _adaptive_half_width_px(mask, half_width_px)
    positions, tangents = _sample_control_points(polyline, interval_px)
    if len(positions) < 2:
        return None, {
            "method": method,
            "mask": mask,
            "skeleton": skel,
            "raw_polyline": raw_polyline,
            "polyline": polyline,
            "positions": positions,
            "left_pts": np.empty((0, 2), dtype=np.float64),
            "right_pts": np.empty((0, 2), dtype=np.float64),
            "slice_segments": [],
            "row_offset": row_offset,
            "col_offset": col_offset,
            "half_width_px": half_width_px,
        }

    normals = np.column_stack([tangents[:, 1], -tangents[:, 0]])
    normals = _stabilize_normals(normals)

    left_pts: list[np.ndarray] = []
    right_pts: list[np.ndarray] = []
    slice_segments: list[tuple[np.ndarray, np.ndarray]] = []
    used_positions: list[np.ndarray] = []
    used_normals: list[np.ndarray] = []
    for pos, normal in zip(positions, normals):
        n_norm = float(np.linalg.norm(normal))
        if n_norm <= 1e-9:
            continue
        normal = normal / n_norm
        seg_a = pos - normal * half_width_px
        seg_b = pos + normal * half_width_px
        slice_segments.append((seg_a, seg_b))
        left, right = _find_boundary_on_normal(mask, pos, normal, half_width_px)
        if left is not None:
            left_pts.append(left)
        if right is not None:
            right_pts.append(right)
        used_positions.append(pos)
        used_normals.append(normal)

    left_arr = np.asarray(left_pts, dtype=np.float64)
    right_arr = np.asarray(right_pts, dtype=np.float64)
    ring: list[list[float]] | None = None
    if len(left_arr) >= 2 and len(right_arr) >= 2:
        ring_rc = np.vstack([left_arr, right_arr[::-1]])
        ring_cr = np.stack(
            [ring_rc[:, 1] + col_offset, ring_rc[:, 0] + row_offset],
            axis=1,
        )
        ring_xy = pixel_to_xy(ring_cr, bev_meta)
        ring = [[float(x), float(y)] for x, y in ring_xy]
        ring.append(ring[0])

    debug = {
        "method": method,
        "mask": mask,
        "skeleton": skel,
        "raw_polyline": raw_polyline,
        "polyline": polyline,
        "positions": np.asarray(used_positions, dtype=np.float64),
        "normals": np.asarray(used_normals, dtype=np.float64),
        "left_pts": left_arr,
        "right_pts": right_arr,
        "slice_segments": slice_segments,
        "row_offset": row_offset,
        "col_offset": col_offset,
        "half_width_px": half_width_px,
    }
    return ring, debug


def _draw_polyline(
    image: np.ndarray,
    pts_rc: np.ndarray,
    *,
    color: tuple[int, int, int],
    thickness: int = 1,
    offset_rc: tuple[int, int] = (0, 0),
) -> None:
    import cv2

    if len(pts_rc) < 2:
        return
    pts = np.asarray(pts_rc, dtype=np.float64).copy()
    pts[:, 0] += offset_rc[0]
    pts[:, 1] += offset_rc[1]
    pts_xy = np.round(np.stack([pts[:, 1], pts[:, 0]], axis=1)).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [pts_xy], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def _draw_points(
    image: np.ndarray,
    pts_rc: np.ndarray,
    *,
    color: tuple[int, int, int],
    radius: int = 2,
    offset_rc: tuple[int, int] = (0, 0),
) -> None:
    import cv2

    for pt in np.asarray(pts_rc, dtype=np.float64):
        r = int(round(pt[0] + offset_rc[0]))
        c = int(round(pt[1] + offset_rc[1]))
        cv2.circle(image, (c, r), radius, color, thickness=-1, lineType=cv2.LINE_AA)


def _write_object_debug_image(
    obj_id: int,
    debug: dict[str, Any],
    output_path: Path,
) -> None:
    import cv2

    mask = np.asarray(debug["mask"], dtype=bool)
    pad = 16
    h, w = mask.shape
    canvas = np.full((h + 2 * pad, w + 2 * pad, 3), 28, dtype=np.uint8)
    canvas[pad:pad + h, pad:pad + w][mask] = np.array([215, 215, 215], dtype=np.uint8)

    skeleton = debug.get("skeleton")
    if skeleton is not None and np.any(skeleton):
        canvas[pad:pad + h, pad:pad + w][np.asarray(skeleton, dtype=bool)] = np.array([0, 220, 255], dtype=np.uint8)

    offset = (pad, pad)
    _draw_polyline(canvas, np.asarray(debug["raw_polyline"]), color=(180, 80, 255), thickness=1, offset_rc=offset)
    _draw_polyline(canvas, np.asarray(debug["polyline"]), color=(0, 255, 255), thickness=1, offset_rc=offset)
    _draw_points(canvas, np.asarray(debug["positions"]), color=(255, 180, 0), radius=2, offset_rc=offset)
    _draw_points(canvas, np.asarray(debug["left_pts"]), color=(0, 255, 0), radius=2, offset_rc=offset)
    _draw_points(canvas, np.asarray(debug["right_pts"]), color=(255, 64, 64), radius=2, offset_rc=offset)

    for seg_a, seg_b in debug["slice_segments"]:
        a = (int(round(seg_a[1] + pad)), int(round(seg_a[0] + pad)))
        b = (int(round(seg_b[1] + pad)), int(round(seg_b[0] + pad)))
        cv2.line(canvas, a, b, (255, 128, 0), thickness=1, lineType=cv2.LINE_AA)

    title = f"id={obj_id} method={debug['method']} half_width_px={debug['half_width_px']:.1f}"
    cv2.putText(canvas, title, (12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def _write_global_debug_overview(
    entries: list[tuple[int, dict[str, Any]]],
    bev_meta: dict[str, Any],
    output_path: Path,
    *,
    overview_mpp: float = 0.02,
) -> None:
    import cv2

    h = int(bev_meta["height"])
    w = int(bev_meta["width"])
    src_mpp = float(bev_meta["meters_per_pixel"])
    scale = src_mpp / float(overview_mpp)
    scale = min(1.0, scale)
    out_h = max(1, int(round(h * scale)))
    out_w = max(1, int(round(w * scale)))
    canvas = np.full((out_h, out_w, 3), 24, dtype=np.uint8)

    def _to_xy(pt_rc: np.ndarray, row_offset: int, col_offset: int) -> tuple[int, int]:
        r = (pt_rc[0] + row_offset) * scale
        c = (pt_rc[1] + col_offset) * scale
        return (int(round(c)), int(round(r)))

    for obj_id, debug in entries:
        row_offset = int(debug["row_offset"])
        col_offset = int(debug["col_offset"])

        raw_polyline = np.asarray(debug["raw_polyline"], dtype=np.float64)
        if len(raw_polyline) >= 2:
            pts = []
            for pt in raw_polyline:
                pts.append(_to_xy(pt, row_offset, col_offset))
            pts_arr = np.asarray(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts_arr], isClosed=False, color=(180, 80, 255), thickness=1, lineType=cv2.LINE_AA)

        polyline = np.asarray(debug["polyline"], dtype=np.float64)
        if len(polyline) >= 2:
            pts = []
            for pt in polyline:
                pts.append(_to_xy(pt, row_offset, col_offset))
            pts_arr = np.asarray(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts_arr], isClosed=False, color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        for seg_a, seg_b in debug["slice_segments"][:: max(1, len(debug["slice_segments"]) // 80 or 1)]:
            cv2.line(
                canvas,
                _to_xy(seg_a, row_offset, col_offset),
                _to_xy(seg_b, row_offset, col_offset),
                (255, 128, 0),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        for pts_rc, color in (
            (np.asarray(debug["positions"], dtype=np.float64), (255, 180, 0)),
            (np.asarray(debug["left_pts"], dtype=np.float64), (0, 255, 0)),
            (np.asarray(debug["right_pts"], dtype=np.float64), (255, 64, 64)),
        ):
            step = max(1, len(pts_rc) // 120 or 1)
            for pt in pts_rc[::step]:
                cv2.circle(canvas, _to_xy(pt, row_offset, col_offset), 1, color, thickness=-1, lineType=cv2.LINE_AA)

        if len(polyline) > 0:
            anchor = _to_xy(polyline[len(polyline) // 2], row_offset, col_offset)
            cv2.putText(canvas, str(obj_id), anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def vectorize_long_laneline(
    mask: np.ndarray,
    bev_meta: dict[str, Any],
    *,
    sample_interval_m: float = 0.10,
    cross_half_width_m: float = 0.30,
    centerline_smooth_m: float = 0.10,
    row_offset: int = 0,
    col_offset: int = 0,
) -> list[list[float]] | None:
    ring, _debug = _trace_long_laneline(
        mask,
        bev_meta,
        sample_interval_m=sample_interval_m,
        cross_half_width_m=cross_half_width_m,
        centerline_smooth_m=centerline_smooth_m,
        row_offset=row_offset,
        col_offset=col_offset,
    )
    return ring


def _iter_masks_for_vectorization(
    masks: np.ndarray,
) -> list[tuple[int, np.ndarray, int, int]]:
    if masks.ndim == 3:
        return [(i, masks[i].astype(bool), 0, 0) for i in range(masks.shape[0])]

    if masks.ndim != 2:
        raise ValueError(f"Expected 2D or 3D array, got shape {masks.shape}")

    label_map = masks
    from scipy import ndimage

    positive_labels = np.asarray(label_map, dtype=np.int32) + 1
    object_slices = ndimage.find_objects(positive_labels)
    results: list[tuple[int, np.ndarray, int, int]] = []
    for oid, slc in enumerate(object_slices):
        if slc is None:
            continue
        r_slc, c_slc = slc
        r0, r1 = int(r_slc.start), int(r_slc.stop)
        c0, c1 = int(c_slc.start), int(c_slc.stop)
        crop = np.asarray(label_map[r0:r1, c0:c1] == oid)
        if not np.any(crop):
            continue
        results.append((int(oid), crop, r0, c0))
    return results


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
        return {
            "width": canvas_w,
            "height": canvas_h,
            "meters_per_pixel": mpp,
            "min_xy": [g_min_x, g_max_y - canvas_h * mpp],
            "max_xy": [g_min_x + canvas_w * mpp, g_max_y],
        }
    raise KeyError(
        "Expected bev_meta with min_xy/max_xy or summary with canvas_size/global_origin_xy"
    )


def long_laneline_to_shp(
    masks: np.ndarray,
    bev_meta: dict[str, Any],
    output_dir: Path,
    *,
    sample_interval_m: float = 0.10,
    cross_half_width_m: float = 0.30,
    min_length_m: float = 10.0,
    centerline_smooth_m: float = 0.10,
    debug_dir: Path | None = None,
    shp_stem: str = "long_laneline",
) -> Path:
    import shapefile
    from shapely.geometry import Polygon

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mpp = float(bev_meta["meters_per_pixel"])
    debug_entries: list[tuple[int, dict[str, Any]]] = []
    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        objects_dir = debug_dir / "objects"
        if objects_dir.is_dir():
            for p in objects_dir.glob("obj_*.png"):
                p.unlink()
        overview_path = debug_dir / "overview.png"
        if overview_path.exists():
            overview_path.unlink()

    shp_path = output_dir / shp_stem
    w = shapefile.Writer(str(shp_path))
    w.shapeType = shapefile.POLYGON
    w.field("id", "N", decimal=0)
    w.field("length_m", "F", decimal=3)
    w.field("area_m2", "F", decimal=3)
    w.field("n_ctrl", "N", decimal=0)

    count = 0
    skipped = 0
    for obj_id, mask, row_offset, col_offset in _iter_masks_for_vectorization(masks):
        if not np.any(mask):
            continue

        skel = _skeletonize_mask(mask)
        skel_length_m = float(np.sum(skel)) * mpp
        if skel_length_m < min_length_m:
            skipped += 1
            continue

        ring, debug = _trace_long_laneline(
            mask, bev_meta,
            sample_interval_m=sample_interval_m,
            cross_half_width_m=cross_half_width_m,
            centerline_smooth_m=centerline_smooth_m,
            row_offset=row_offset,
            col_offset=col_offset,
        )
        if ring is None or len(ring) < 4:
            skipped += 1
            continue

        poly = Polygon(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            skipped += 1
            continue

        w.poly([ring])
        w.record(
            id=obj_id,
            length_m=skel_length_m,
            area_m2=float(poly.area),
            n_ctrl=len(ring) - 1,
        )
        if debug_dir is not None:
            debug_entries.append((obj_id, debug))
            _write_object_debug_image(
                obj_id,
                debug,
                Path(debug_dir) / "objects" / f"obj_{obj_id:03d}.png",
            )
        count += 1

    w.close()
    if debug_dir is not None:
        _write_global_debug_overview(
            debug_entries,
            bev_meta,
            Path(debug_dir) / "overview.png",
        )
    print(
        f"[shp] wrote {count} long-laneline polygons → {shp_path}.shp"
        f"  (skipped {skipped})",
        flush=True,
    )
    return Path(f"{shp_path}.shp")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Vectorize long lane-line instance label_map → long_laneline.shp.",
    )
    parser.add_argument(
        "label_map_npy",
        help="Path to label_map .npy file with shape (H,W).",
    )
    parser.add_argument(
        "bev_meta_json",
        help="Path to bev_meta / summary JSON with min_xy, max_xy, meters_per_pixel, width, height.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="outputs/apps/shp",
        help="Output directory (default: outputs/apps/shp).",
    )
    parser.add_argument(
        "--interval", type=float, default=0.10,
        help="Control point interval in metres (default: 0.10).",
    )
    parser.add_argument(
        "--half-width", type=float, default=0.30,
        help="Normal cross-section half-width in metres (default: 0.30).",
    )
    parser.add_argument(
        "--min-length", type=float, default=10.0,
        help="Minimum skeleton length in metres (default: 10.0).",
    )
    parser.add_argument(
        "--smooth", type=float, default=0.10,
        help="Centerline smoothing sigma in metres before control-point sampling (default: 0.10).",
    )
    parser.add_argument(
        "--debug-dir",
        default=None,
        help="Optional directory to save debug visualizations.",
    )
    args = parser.parse_args()

    masks = np.load(args.label_map_npy, mmap_mode="r")
    if masks.ndim == 3:
        masks = masks.astype(bool)
    elif masks.ndim != 2:
        raise ValueError(f"Expected 2D or 3D array, got shape {masks.shape}")

    bev_meta = _coerce_bev_meta(args.bev_meta_json)

    long_laneline_to_shp(
        masks, bev_meta,
        Path(args.output_dir),
        sample_interval_m=args.interval,
        cross_half_width_m=args.half_width,
        min_length_m=args.min_length,
        centerline_smooth_m=args.smooth,
        debug_dir=Path(args.debug_dir) if args.debug_dir else None,
    )


if __name__ == "__main__":
    main()
