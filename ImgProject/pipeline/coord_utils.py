from __future__ import annotations

from typing import Any

import numpy as np


def _normalize_origin_xy(origin_xy: Any) -> np.ndarray:
    arr = np.asarray(origin_xy, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        raise ValueError(f"origin_xy must contain at least 2 values, got shape={arr.shape}")
    return arr[:2].astype(np.float64, copy=False)


def compute_scene_origin_xy(points_xyz: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xyz)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError(f"points_xyz must have shape (N,>=2), got {pts.shape}")
    if pts.shape[0] == 0:
        return np.zeros((2,), dtype=np.float64)
    origin = np.asarray(pts[0, :2], dtype=np.float64).reshape(2)
    if not np.all(np.isfinite(origin)):
        finite_rows = np.where(np.all(np.isfinite(np.asarray(pts[:, :2], dtype=np.float64)), axis=1))[0]
        if finite_rows.size == 0:
            return np.zeros((2,), dtype=np.float64)
        origin = np.asarray(pts[int(finite_rows[0]), :2], dtype=np.float64).reshape(2)
    return origin


def to_local_xy(points_xyz: np.ndarray, origin_xy: Any) -> np.ndarray:
    pts = np.asarray(points_xyz)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError(f"points_xyz must have shape (N,>=2), got {pts.shape}")
    origin = _normalize_origin_xy(origin_xy)
    out = np.asarray(pts, dtype=np.float64).copy()
    out[:, 0] -= float(origin[0])
    out[:, 1] -= float(origin[1])
    return out.astype(pts.dtype, copy=False)


def restore_global_xy(values: np.ndarray, origin_xy: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 0:
        return np.asarray(arr)
    if arr.shape[-1] < 2:
        raise ValueError(f"values last dimension must be >=2, got {arr.shape}")
    origin = _normalize_origin_xy(origin_xy)
    out = np.asarray(arr, dtype=np.float64).copy()
    out[..., 0] += float(origin[0])
    out[..., 1] += float(origin[1])
    return out.astype(arr.dtype, copy=False)


def shift_extrinsic_xy(extrinsic: np.ndarray, origin_xy: Any) -> np.ndarray:
    mat = np.asarray(extrinsic)
    if mat.shape != (4, 4):
        raise ValueError(f"extrinsic must have shape (4,4), got {mat.shape}")
    origin = _normalize_origin_xy(origin_xy)
    out = np.asarray(mat, dtype=np.float64).copy()
    out[0, 3] -= float(origin[0])
    out[1, 3] -= float(origin[1])
    return out.astype(mat.dtype, copy=False)
