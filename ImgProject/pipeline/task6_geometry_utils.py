from __future__ import annotations

import math
from typing import Any

import numpy as np
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    from scipy.spatial import cKDTree  # type: ignore
    from scipy.sparse import csr_matrix  # type: ignore
    from scipy.sparse.csgraph import connected_components, dijkstra  # type: ignore
except Exception:  # pragma: no cover
    cKDTree = None
    csr_matrix = None
    connected_components = None
    dijkstra = None

try:
    import CSF  # type: ignore
except Exception:  # pragma: no cover
    CSF = None


def _round_float(value: float | int | np.floating[Any] | np.integer[Any] | None, ndigits: int = 4) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except Exception:
        return None
    if not np.isfinite(f):
        return None
    return round(f, ndigits)


def _as_valid_point_indices(point_indices: np.ndarray | list[int], num_points: int) -> np.ndarray:
    idx = np.asarray(point_indices, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        return np.zeros((0,), dtype=np.int64)
    valid = (idx >= 0) & (idx < int(num_points))
    return np.unique(idx[valid]).astype(np.int64, copy=False)


def compute_global_ground_mask(
    points_xyz: np.ndarray,
    *,
    cloth_resolution: float = 1.0,
    rigidness: int = 1,
    time_step: float = 0.65,
    class_threshold: float = 1.2,
    iterations: int = 800,
    slope_smooth: bool = True,
) -> np.ndarray | None:
    if CSF is None:
        return None
    num_points = int(points_xyz.shape[0])
    if num_points == 0:
        return np.zeros((0,), dtype=bool)
    try:
        mins = points_xyz.min(axis=0)
        maxs = points_xyz.max(axis=0)
        center = (mins + maxs) / 2.0
        centered = points_xyz - center

        csf = CSF.CSF()
        csf.setPointCloud(centered.astype(np.float64, copy=False))
        csf.params.cloth_resolution = float(cloth_resolution)
        csf.params.rigidness = int(rigidness)
        csf.params.time_step = float(time_step)
        csf.params.class_threshold = float(class_threshold)
        csf.params.interations = int(iterations)
        csf.params.bSloopSmooth = bool(slope_smooth)

        ground = CSF.VecInt()
        non_ground = CSF.VecInt()
        csf.do_filtering(ground, non_ground)
    except Exception:
        return None

    mask = np.zeros((num_points,), dtype=bool)
    for idx in ground:
        idx_int = int(idx)
        if 0 <= idx_int < num_points:
            mask[idx_int] = True
    return mask


def estimate_ground_z(
    points_xyz: np.ndarray,
    point_indices: np.ndarray | list[int],
    *,
    ground_points_xyz: np.ndarray | None = None,
    global_ground_mask: np.ndarray | None = None,
    neighborhood_radius: float = 2.0,
    neighborhood_quantile: float = 0.10,
) -> tuple[float, str]:
    idx = _as_valid_point_indices(point_indices, num_points=points_xyz.shape[0])
    if idx.size == 0:
        return 0.0, "empty_fallback"

    object_xyz = points_xyz[idx]
    object_xy_center = object_xyz[:, :2].mean(axis=0)
    q = min(1.0, max(0.0, float(neighborhood_quantile)))

    if isinstance(ground_points_xyz, np.ndarray):
        gxyz = np.asarray(ground_points_xyz, dtype=np.float32).reshape(-1, 3)
        if gxyz.shape[0] > 0:
            gxy = gxyz[:, :2]
            dist2 = np.sum((gxy - object_xy_center[None, :]) ** 2, axis=1)
            near = dist2 <= float(neighborhood_radius) ** 2
            if int(np.count_nonzero(near)) >= 5:
                z = gxyz[near, 2]
                return float(np.quantile(z, q)), f"task5_csf_ground_neighborhood_q{int(round(q * 100))}"

    if isinstance(global_ground_mask, np.ndarray) and global_ground_mask.shape[0] == points_xyz.shape[0]:
        xy = points_xyz[:, :2]
        dist2 = np.sum((xy - object_xy_center[None, :]) ** 2, axis=1)
        near = dist2 <= float(neighborhood_radius) ** 2
        near_ground = near & global_ground_mask
        if int(np.count_nonzero(near_ground)) >= 5:
            z = points_xyz[near_ground, 2]
            return float(np.quantile(z, q)), f"csf_neighborhood_q{int(round(q * 100))}"

    object_z = object_xyz[:, 2]
    return float(np.quantile(object_z, 0.02)), "q02_fallback"


def compute_height_m(points_xyz: np.ndarray, point_indices: np.ndarray | list[int], *, ground_z: float) -> float:
    idx = _as_valid_point_indices(point_indices, num_points=points_xyz.shape[0])
    if idx.size == 0:
        return 0.0
    rel_z = points_xyz[idx, 2] - float(ground_z)
    h = float(np.quantile(rel_z, 0.98))
    return max(0.0, h)


def _fit_circle_taubin_svd_xy(points_xy: np.ndarray) -> tuple[np.ndarray, float] | None:
    xy = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if xy.shape[0] < 3:
        return None

    centroid = xy.mean(axis=0)
    centered = xy - centroid[None, :]
    x = centered[:, 0]
    y = centered[:, 1]
    z = x * x + y * y
    z_mean = float(z.mean())
    if (not np.isfinite(z_mean)) or z_mean <= 1e-12:
        return None

    z0 = (z - z_mean) / (2.0 * math.sqrt(z_mean))
    mat = np.column_stack((z0, x, y))
    try:
        _, _, vh = np.linalg.svd(mat, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if vh.shape[0] == 0:
        return None

    a = np.asarray(vh[-1], dtype=np.float64)
    a0 = float(a[0]) / (2.0 * math.sqrt(z_mean))
    if (not np.isfinite(a0)) or abs(a0) < 1e-12:
        return None

    a1 = float(a[1])
    a2 = float(a[2])
    a3 = -z_mean * a0

    cx_local = -a1 / (2.0 * a0)
    cy_local = -a2 / (2.0 * a0)
    center = np.asarray([cx_local, cy_local], dtype=np.float64) + centroid

    radicand = (a1 * a1 + a2 * a2 - 4.0 * a0 * a3) / (4.0 * a0 * a0)
    if (not np.isfinite(radicand)) or radicand <= 0:
        return None
    radius = float(math.sqrt(radicand))
    if not np.isfinite(radius):
        return None
    return center.astype(np.float32, copy=False), radius


def _fit_min_enclosing_circle_xy(points_xy: np.ndarray) -> tuple[np.ndarray, float] | None:
    xy = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if xy.shape[0] < 1:
        return None

    if cv2 is not None:
        try:
            center_xy, radius = cv2.minEnclosingCircle(xy)
            center = np.asarray(center_xy, dtype=np.float32)
            radius_f = float(radius)
        except Exception:
            return None
        if center.size < 2 or (not np.isfinite(radius_f)):
            return None
        return center, radius_f

    center = xy.mean(axis=0, dtype=np.float64).astype(np.float32, copy=False)
    dist = np.sqrt(np.sum((xy.astype(np.float64) - center.astype(np.float64)) ** 2, axis=1))
    if dist.size == 0:
        return None
    radius = float(np.max(dist))
    if not np.isfinite(radius):
        return None
    return center, radius


def estimate_diameter_and_center_xy(
    points_xyz: np.ndarray,
    point_indices: np.ndarray | list[int],
    *,
    ground_z: float,
    band_min: float = 0.8,
    band_max: float = 1.4,
) -> tuple[float | None, list[float] | None]:
    idx = _as_valid_point_indices(point_indices, num_points=points_xyz.shape[0])
    if idx.size < 3:
        return None, None
    rel_z = points_xyz[idx, 2] - float(ground_z)
    low = min(float(band_min), float(band_max))
    high = max(float(band_min), float(band_max))
    band_mask = (rel_z >= low) & (rel_z <= high)
    band_points = points_xyz[idx[band_mask], :2]
    if band_points.shape[0] < 3:
        return None, None
    fit = _fit_circle_taubin_svd_xy(band_points)
    if fit is None:
        return None, None
    center, radius = fit
    diameter = float(2.0 * radius)
    return diameter, [_round_float(center[0]), _round_float(center[1])]


def estimate_centroid_xy(points_xyz: np.ndarray, point_indices: np.ndarray | list[int]) -> list[float] | None:
    idx = _as_valid_point_indices(point_indices, num_points=points_xyz.shape[0])
    if idx.size == 0:
        return None
    center = points_xyz[idx, :2].mean(axis=0)
    return [_round_float(center[0]), _round_float(center[1])]


def estimate_mid_band_centroid_xy(
    points_xyz: np.ndarray,
    point_indices: np.ndarray | list[int],
    *,
    low_ratio: float = 0.30,
    high_ratio: float = 0.70,
) -> tuple[list[float] | None, str]:
    idx = _as_valid_point_indices(point_indices, num_points=points_xyz.shape[0])
    if idx.size == 0:
        return None, "empty"

    z = points_xyz[idx, 2]
    q05 = float(np.quantile(z, 0.05))
    q95 = float(np.quantile(z, 0.95))
    span = max(1e-6, q95 - q05)
    low = q05 + min(float(low_ratio), float(high_ratio)) * span
    high = q05 + max(float(low_ratio), float(high_ratio)) * span
    mid_mask = (z >= low) & (z <= high)
    mid_idx = idx[mid_mask]

    min_mid_points = max(8, min(80, int(idx.size * 0.1)))
    if mid_idx.size < min_mid_points:
        center = points_xyz[idx, :2].mean(axis=0)
        return [_round_float(center[0]), _round_float(center[1])], "full_centroid_fallback"

    center = points_xyz[mid_idx, :2].mean(axis=0)
    return [_round_float(center[0]), _round_float(center[1])], "mid_band_centroid"


def _downsample_xy_by_grid(
    xy: np.ndarray,
    *,
    grid_size: float = 0.10,
    max_points: int = 4000,
) -> np.ndarray:
    coords = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if coords.shape[0] <= 1:
        return coords.astype(np.float32, copy=False)
    if grid_size <= 1e-6:
        out = np.unique(coords, axis=0)
    else:
        mins = coords.min(axis=0, keepdims=True)
        grid = np.floor((coords - mins) / float(grid_size)).astype(np.int64)
        uniq, inv = np.unique(grid, axis=0, return_inverse=True)
        sums = np.zeros((uniq.shape[0], 2), dtype=np.float64)
        cnt = np.zeros((uniq.shape[0],), dtype=np.int64)
        np.add.at(sums, inv, coords)
        np.add.at(cnt, inv, 1)
        out = sums / np.maximum(cnt[:, None], 1)
    if out.shape[0] > int(max_points):
        keep = np.linspace(0, out.shape[0] - 1, num=int(max_points), dtype=np.int64)
        out = out[keep]
    return out.astype(np.float32, copy=False)


def _sample_polyline_by_count(polyline_xy: np.ndarray, count: int) -> np.ndarray:
    pts = np.asarray(polyline_xy, dtype=np.float64).reshape(-1, 2)
    n = int(max(2, count))
    if pts.shape[0] == 0:
        return np.zeros((n, 2), dtype=np.float32)
    if pts.shape[0] == 1:
        return np.repeat(pts.astype(np.float32, copy=False), n, axis=0)

    seg = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])
    if total <= 1e-9:
        return np.repeat(pts[:1].astype(np.float32, copy=False), n, axis=0)

    targets = np.linspace(0.0, total, num=n, dtype=np.float64)
    out = np.zeros((n, 2), dtype=np.float64)
    j = 0
    for i, t in enumerate(targets):
        while j < seg_len.shape[0] - 1 and cum[j + 1] < t:
            j += 1
        start = pts[j]
        end = pts[j + 1]
        den = max(cum[j + 1] - cum[j], 1e-12)
        alpha = float((t - cum[j]) / den)
        alpha = min(1.0, max(0.0, alpha))
        out[i] = start * (1.0 - alpha) + end * alpha
    return out.astype(np.float32, copy=False)


def _build_pca_centerline(xy: np.ndarray, point_count: int) -> np.ndarray:
    coords = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if coords.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if coords.shape[0] == 1:
        return np.repeat(coords.astype(np.float32, copy=False), int(max(2, point_count)), axis=0)

    centered = coords - coords.mean(axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        axis = np.asarray([1.0, 0.0], dtype=np.float64)
    else:
        axis = vh[0] if vh.shape[0] >= 1 else np.asarray([1.0, 0.0], dtype=np.float64)
    proj = centered @ axis
    t_min = float(np.min(proj))
    t_max = float(np.max(proj))
    t = np.linspace(t_min, t_max, num=int(max(2, point_count)), dtype=np.float64)
    line = coords.mean(axis=0, keepdims=True) + t[:, None] * axis[None, :]
    return line.astype(np.float32, copy=False)


def _longest_path_xy_from_knn_graph(
    xy: np.ndarray,
    *,
    knn: int = 8,
) -> np.ndarray | None:
    coords = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    n = int(coords.shape[0])
    if n < 2:
        return None
    if (
        cKDTree is None
        or csr_matrix is None
        or connected_components is None
        or dijkstra is None
    ):
        return None

    k = int(max(2, min(knn + 1, n)))
    tree = cKDTree(coords)
    dists, nbrs = tree.query(coords, k=k)
    if k == 1:
        return None
    dists = np.asarray(dists, dtype=np.float64).reshape(n, k)
    nbrs = np.asarray(nbrs, dtype=np.int64).reshape(n, k)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for i in range(n):
        for j in range(1, k):
            nb = int(nbrs[i, j])
            dist = float(dists[i, j])
            if nb < 0 or nb >= n or nb == i:
                continue
            if not np.isfinite(dist) or dist <= 0.0:
                continue
            rows.extend([i, nb])
            cols.extend([nb, i])
            vals.extend([dist, dist])
    if not rows:
        return None

    graph = csr_matrix((np.asarray(vals), (np.asarray(rows), np.asarray(cols))), shape=(n, n))
    comp_n, comp_labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    if int(comp_n) <= 0:
        return None
    comp_sizes = np.bincount(comp_labels.astype(np.int64), minlength=int(comp_n))
    keep_comp = int(np.argmax(comp_sizes))
    keep_nodes = np.where(comp_labels == keep_comp)[0]
    if keep_nodes.size < 2:
        return None

    sub = graph[keep_nodes][:, keep_nodes]
    sub_xy = coords[keep_nodes]
    seed = int(np.argmin(np.sum(sub_xy, axis=1)))
    dist_seed = dijkstra(csgraph=sub, directed=False, indices=seed)
    if not np.any(np.isfinite(dist_seed)):
        return None
    a = int(np.nanargmax(np.where(np.isfinite(dist_seed), dist_seed, -1.0)))
    dist_a, pred = dijkstra(csgraph=sub, directed=False, indices=a, return_predecessors=True)
    if not np.any(np.isfinite(dist_a)):
        return None
    b = int(np.nanargmax(np.where(np.isfinite(dist_a), dist_a, -1.0)))

    path_sub: list[int] = []
    cur = b
    guard = 0
    while cur != -9999 and cur >= 0 and guard <= int(sub.shape[0] + 5):
        path_sub.append(int(cur))
        if cur == a:
            break
        cur = int(pred[cur])
        guard += 1
    if len(path_sub) < 2:
        return None
    path_sub = path_sub[::-1]
    return sub_xy[np.asarray(path_sub, dtype=np.int64)].astype(np.float32, copy=False)


def estimate_fence_control_points_xy(
    points_xyz: np.ndarray,
    point_indices: np.ndarray | list[int],
    *,
    mid_band_low_ratio: float = 0.30,
    mid_band_high_ratio: float = 0.70,
    control_point_count: int = 5,
    grid_size: float = 0.10,
    knn: int = 8,
) -> tuple[list[list[float]] | None, str]:
    idx = _as_valid_point_indices(point_indices, num_points=points_xyz.shape[0])
    if idx.size == 0:
        return None, "empty"

    z = points_xyz[idx, 2]
    q05 = float(np.quantile(z, 0.05))
    q95 = float(np.quantile(z, 0.95))
    span = max(1e-6, q95 - q05)
    low = q05 + min(float(mid_band_low_ratio), float(mid_band_high_ratio)) * span
    high = q05 + max(float(mid_band_low_ratio), float(mid_band_high_ratio)) * span
    mid_mask = (z >= low) & (z <= high)
    mid_idx = idx[mid_mask]
    if mid_idx.size < max(20, int(idx.size * 0.1)):
        mid_idx = idx
        method_prefix = "full_band"
    else:
        method_prefix = "mid_band"

    xy = points_xyz[mid_idx, :2]
    if xy.shape[0] < 2:
        center = points_xyz[idx, :2].mean(axis=0)
        cp = [[_round_float(center[0]), _round_float(center[1])] for _ in range(int(max(2, control_point_count)))]
        return cp, f"{method_prefix}_centroid_repeat"

    xy_ds = _downsample_xy_by_grid(xy, grid_size=float(grid_size))
    line = _longest_path_xy_from_knn_graph(xy_ds, knn=int(knn))
    if line is None or line.shape[0] < 2:
        line = _build_pca_centerline(xy_ds, point_count=max(2, int(control_point_count)))
        method = f"{method_prefix}_pca_fallback"
    else:
        method = f"{method_prefix}_knn_longest_path"

    sampled = _sample_polyline_by_count(line, int(max(2, control_point_count)))
    out: list[list[float]] = []
    for p in sampled:
        out.append([_round_float(p[0]), _round_float(p[1])])
    return out, method


def estimate_xy_box_size(points_xyz: np.ndarray, point_indices: np.ndarray | list[int]) -> tuple[float | None, float | None]:
    idx = _as_valid_point_indices(point_indices, num_points=points_xyz.shape[0])
    if idx.size < 3:
        return None, None
    xy = points_xyz[idx, :2].astype(np.float64, copy=False)
    centered = xy - xy.mean(axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, None
    if vh.shape[0] < 2:
        return None, None
    axis1 = vh[0]
    axis2 = vh[1]
    proj1 = centered @ axis1
    proj2 = centered @ axis2
    length = float(proj1.max() - proj1.min())
    width = float(proj2.max() - proj2.min())
    if width > length:
        length, width = width, length
    return length, width


def compute_manhole_geometry_from_pixels(
    pixel_xy: np.ndarray,
    *,
    resolution_m_per_px: float = 0.02,
    global_origin_min_x: float | None = None,
    global_origin_max_y: float | None = None,
) -> dict[str, Any]:
    coords = np.asarray(pixel_xy, dtype=np.float32).reshape(-1, 2)
    empty = {"circle_center_global_xy": None, "circle_radius_m": None, "circle_center_px": None}
    if coords.shape[0] < 3:
        return empty

    fit = _fit_min_enclosing_circle_xy(coords)
    if fit is None:
        return empty

    resolution = float(resolution_m_per_px)
    if (not np.isfinite(resolution)) or resolution <= 1e-12:
        return empty

    center_px, radius_px = fit
    radius_m = float(radius_px) * resolution
    center_global_xy: list[float] | None = None
    if global_origin_min_x is not None and global_origin_max_y is not None:
        gx = float(global_origin_min_x) + float(center_px[0]) * resolution
        gy = float(global_origin_max_y) - float(center_px[1]) * resolution
        if np.isfinite(gx) and np.isfinite(gy):
            center_global_xy = [_round_float(gx), _round_float(gy)]

    return {
        "circle_center_global_xy": center_global_xy,
        "circle_radius_m": _round_float(radius_m),
        "circle_center_px": [_round_float(center_px[0]), _round_float(center_px[1])],
    }


def geometry_for_pole_group(
    points_xyz: np.ndarray,
    point_indices: np.ndarray | list[int],
    *,
    candidate_class_ids: list[int] | np.ndarray,
    ground_points_xyz: np.ndarray | None = None,
    global_ground_mask: np.ndarray | None = None,
    neighborhood_radius: float = 2.0,
    neighborhood_quantile: float = 0.10,
    precomputed_diameter_m: float | None = None,
    precomputed_center_xy: np.ndarray | list[float] | tuple[float, float] | None = None,
) -> dict[str, Any]:
    class_ids = {int(x) for x in np.asarray(candidate_class_ids, dtype=np.int32).reshape(-1)}
    ground_z, ground_method = estimate_ground_z(
        points_xyz,
        point_indices,
        ground_points_xyz=ground_points_xyz,
        global_ground_mask=global_ground_mask,
        neighborhood_radius=float(neighborhood_radius),
        neighborhood_quantile=float(neighborhood_quantile),
    )
    height_m = compute_height_m(points_xyz, point_indices, ground_z=ground_z)
    out: dict[str, Any] = {
        "height_m": _round_float(height_m),
        "ground_z_method": ground_method,
    }
    diameter_out = _round_float(precomputed_diameter_m)
    center_out: list[float] | None = None
    if precomputed_center_xy is not None:
        center_arr = np.asarray(precomputed_center_xy, dtype=np.float32).reshape(-1)
        if center_arr.size >= 2 and np.isfinite(float(center_arr[0])) and np.isfinite(float(center_arr[1])):
            center_out = [_round_float(center_arr[0]), _round_float(center_arr[1])]

    # Keep diameter empty when Task5 has no reliable precomputed metric.
    # Do not fallback to re-estimation in Task6.
    if center_out is None:
        center_out = estimate_centroid_xy(points_xyz, point_indices)

    out["diameter_m"] = diameter_out
    out["center_xy"] = center_out
    if class_ids:
        out["candidate_class_ids"] = sorted(class_ids)
    return out


def geometry_for_scene_instance(
    points_xyz: np.ndarray,
    point_indices: np.ndarray | list[int],
    *,
    class_id: int,
    tree_metric: dict[str, Any] | None = None,
    ground_points_xyz: np.ndarray | None = None,
    global_ground_mask: np.ndarray | None = None,
    neighborhood_radius: float = 2.0,
    neighborhood_quantile: float = 0.10,
    mid_centroid_low_ratio: float = 0.30,
    mid_centroid_high_ratio: float = 0.70,
    fence_mid_band_low_ratio: float = 0.30,
    fence_mid_band_high_ratio: float = 0.70,
    fence_control_point_count: int = 5,
    fence_grid_size: float = 0.10,
    fence_knn: int = 8,
) -> dict[str, Any]:
    cls_id = int(class_id)
    ground_z, ground_method = estimate_ground_z(
        points_xyz,
        point_indices,
        ground_points_xyz=ground_points_xyz,
        global_ground_mask=global_ground_mask,
        neighborhood_radius=float(neighborhood_radius),
        neighborhood_quantile=float(neighborhood_quantile),
    )
    height_m = compute_height_m(points_xyz, point_indices, ground_z=ground_z)

    if cls_id in {8, 9, 10, 12, 13, 14}:
        center_xy, center_method = estimate_mid_band_centroid_xy(
            points_xyz,
            point_indices,
            low_ratio=float(mid_centroid_low_ratio),
            high_ratio=float(mid_centroid_high_ratio),
        )
        return {
            "center_xy": center_xy,
            "height_m": _round_float(height_m),
            "ground_z_method": ground_method,
            "center_xy_method": center_method,
        }
    if cls_id == 7:
        dbh_m = None
        trunk_center_xy = None
        if tree_metric is not None:
            dbh_m = tree_metric.get("dbh_m")
            trunk_center_xy = tree_metric.get("trunk_center_xy")
        return {
            "dbh_m": _round_float(dbh_m),
            "trunk_center_xy": trunk_center_xy,
            "height_m": _round_float(height_m),
            "ground_z_method": ground_method,
        }
    if cls_id == 15:
        control_points_xy, centerline_method = estimate_fence_control_points_xy(
            points_xyz,
            point_indices,
            mid_band_low_ratio=float(fence_mid_band_low_ratio),
            mid_band_high_ratio=float(fence_mid_band_high_ratio),
            control_point_count=int(max(2, fence_control_point_count)),
            grid_size=float(fence_grid_size),
            knn=int(max(2, fence_knn)),
        )
        return {
            "control_points_xy": control_points_xy,
            "height_m": _round_float(height_m),
            "ground_z_method": ground_method,
            "centerline_method": centerline_method,
        }
    return {
        "center_xy": estimate_centroid_xy(points_xyz, point_indices),
        "height_m": _round_float(height_m),
        "ground_z_method": ground_method,
    }
