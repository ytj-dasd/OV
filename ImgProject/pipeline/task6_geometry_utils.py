from __future__ import annotations

import math
from typing import Any

import numpy as np

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
    global_ground_mask: np.ndarray | None = None,
    neighborhood_radius: float = 2.0,
) -> tuple[float, str]:
    idx = _as_valid_point_indices(point_indices, num_points=points_xyz.shape[0])
    if idx.size == 0:
        return 0.0, "empty_fallback"

    object_xyz = points_xyz[idx]
    object_xy_center = object_xyz[:, :2].mean(axis=0)

    if isinstance(global_ground_mask, np.ndarray) and global_ground_mask.shape[0] == points_xyz.shape[0]:
        xy = points_xyz[:, :2]
        dist2 = np.sum((xy - object_xy_center[None, :]) ** 2, axis=1)
        near = dist2 <= float(neighborhood_radius) ** 2
        near_ground = near & global_ground_mask
        if int(np.count_nonzero(near_ground)) >= 5:
            z = points_xyz[near_ground, 2]
            return float(np.median(z)), "csf_neighborhood"

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
) -> dict[str, Any]:
    coords = np.asarray(pixel_xy, dtype=np.float32).reshape(-1, 2)
    if coords.shape[0] < 3:
        return {"circle_center_xy": None, "circle_radius_m": None}
    coords_m = coords * float(resolution_m_per_px)
    fit = _fit_circle_taubin_svd_xy(coords_m)
    if fit is None:
        return {"circle_center_xy": None, "circle_radius_m": None}
    center, radius = fit
    return {
        "circle_center_xy": [_round_float(center[0]), _round_float(center[1])],
        "circle_radius_m": _round_float(radius),
    }


def geometry_for_pole_group(
    points_xyz: np.ndarray,
    point_indices: np.ndarray | list[int],
    *,
    candidate_class_ids: list[int] | np.ndarray,
    global_ground_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    class_ids = {int(x) for x in np.asarray(candidate_class_ids, dtype=np.int32).reshape(-1)}
    ground_z, ground_method = estimate_ground_z(
        points_xyz,
        point_indices,
        global_ground_mask=global_ground_mask,
    )
    height_m = compute_height_m(points_xyz, point_indices, ground_z=ground_z)
    out: dict[str, Any] = {
        "height_m": _round_float(height_m),
        "ground_z_method": ground_method,
    }
    if 1 in class_ids or 2 in class_ids:
        diameter_m, center_xy = estimate_diameter_and_center_xy(
            points_xyz,
            point_indices,
            ground_z=ground_z,
        )
        out["diameter_m"] = _round_float(diameter_m)
        out["center_xy"] = center_xy
    else:
        out["centroid_xy"] = estimate_centroid_xy(points_xyz, point_indices)
    return out


def geometry_for_scene_instance(
    points_xyz: np.ndarray,
    point_indices: np.ndarray | list[int],
    *,
    class_id: int,
    tree_metric: dict[str, Any] | None = None,
    global_ground_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    cls_id = int(class_id)
    ground_z, ground_method = estimate_ground_z(
        points_xyz,
        point_indices,
        global_ground_mask=global_ground_mask,
    )
    height_m = compute_height_m(points_xyz, point_indices, ground_z=ground_z)

    if cls_id in {3, 4, 5, 6}:
        return {
            "centroid_xy": estimate_centroid_xy(points_xyz, point_indices),
            "height_m": _round_float(height_m),
            "ground_z_method": ground_method,
        }
    if cls_id in {8, 10}:
        length_m, width_m = estimate_xy_box_size(points_xyz, point_indices)
        return {
            "length_m": _round_float(length_m),
            "width_m": _round_float(width_m),
            "height_m": _round_float(height_m),
            "ground_z_method": ground_method,
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
    if cls_id in {1, 2}:
        diameter_m, center_xy = estimate_diameter_and_center_xy(
            points_xyz,
            point_indices,
            ground_z=ground_z,
        )
        return {
            "diameter_m": _round_float(diameter_m),
            "center_xy": center_xy,
            "height_m": _round_float(height_m),
            "ground_z_method": ground_method,
        }
    return {
        "height_m": _round_float(height_m),
        "ground_z_method": ground_method,
    }
