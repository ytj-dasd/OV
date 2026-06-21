"""Geometry helpers for BEV coordinate conversion and rectangle operations."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def xy_to_pixel(xy: np.ndarray, bev_meta: dict[str, Any]) -> np.ndarray:
    """Convert world XY coordinates to pixel (col, row)."""
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


def points_to_pixel(
    points: np.ndarray, bev_meta: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert 3-D points to pixel (row, col, valid_mask)."""
    points = np.asarray(points)
    xy = points[:, :2]
    min_xy = np.asarray(bev_meta["min_xy"], dtype=np.float32)
    mpp = float(bev_meta["meters_per_pixel"])
    w = int(bev_meta["width"])
    h = int(bev_meta["height"])
    col = np.floor((xy[:, 0] - min_xy[0]) / mpp).astype(np.int64)
    row = (h - 1) - np.floor((xy[:, 1] - min_xy[1]) / mpp).astype(np.int64)
    valid = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    return row, col, valid


def pixel_to_xy(pix: np.ndarray, bev_meta: dict[str, Any]) -> np.ndarray:
    """Convert pixel (col, row) back to world XY."""
    pix = np.asarray(pix, dtype=np.float32)
    min_xy = np.asarray(bev_meta["min_xy"], dtype=np.float32)
    mpp = float(bev_meta["meters_per_pixel"])
    h = int(bev_meta["height"])
    col = pix[:, 0]
    row = pix[:, 1]
    x = min_xy[0] + col * mpp
    y = min_xy[1] + (h - 1 - row) * mpp
    return np.stack([x, y], axis=-1)


def as_rect4_px(pixel_corners: Any) -> np.ndarray:
    """Return a 4x2 float32 convex rectangle in pixel coordinates."""
    pts = np.asarray(pixel_corners, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 4:
        raise ValueError("pixel_corners must be (N,2) with N>=4")
    if pts.shape[0] == 4:
        return pts
    rect = cv2.minAreaRect(pts)
    return cv2.boxPoints(rect).astype(np.float32)


def rect_iou_px(a4: np.ndarray, b4: np.ndarray) -> float:
    """IoU of two convex quads in pixel space."""
    a4 = np.asarray(a4, dtype=np.float32).reshape(-1, 2)
    b4 = np.asarray(b4, dtype=np.float32).reshape(-1, 2)
    area_a = float(abs(cv2.contourArea(a4)))
    area_b = float(abs(cv2.contourArea(b4)))
    if area_a <= 1e-6 or area_b <= 1e-6:
        return 0.0
    try:
        inter_area, _ = cv2.intersectConvexConvex(a4, b4)
        inter_area = float(inter_area)
    except Exception:
        return 0.0
    if inter_area <= 0.0:
        return 0.0
    union = area_a + area_b - inter_area
    if union <= 1e-6:
        return 0.0
    return float(inter_area / union)


def rect_overlap_ratio_px(a4: np.ndarray, b4: np.ndarray) -> float:
    """Intersection area over the smaller rectangle area."""
    a4 = np.asarray(a4, dtype=np.float32).reshape(-1, 2)
    b4 = np.asarray(b4, dtype=np.float32).reshape(-1, 2)
    area_a = float(abs(cv2.contourArea(a4)))
    area_b = float(abs(cv2.contourArea(b4)))
    if area_a <= 1e-6 or area_b <= 1e-6:
        return 0.0
    try:
        inter_area, _ = cv2.intersectConvexConvex(a4, b4)
        inter_area = float(inter_area)
    except Exception:
        return 0.0
    if inter_area <= 0.0:
        return 0.0
    denom = min(area_a, area_b)
    if denom <= 1e-6:
        return 0.0
    return float(inter_area / denom)


def rect_main_dir_px(pts4: np.ndarray) -> np.ndarray:
    """Return unit main direction (2,) from the longer edge in pixel space."""
    p = np.asarray(pts4, dtype=np.float32).reshape(4, 2)
    edges = [p[(i + 1) % 4] - p[i] for i in range(4)]
    lens = [float(np.linalg.norm(e)) for e in edges]
    e = edges[int(np.argmax(lens))]
    n = float(np.linalg.norm(e))
    if n <= 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    return (e / n).astype(np.float32)


def rect_uv_bounds_px(
    pts4: np.ndarray,
    axis1: np.ndarray,
    axis2: np.ndarray,
    origin: np.ndarray,
) -> tuple[float, float, float, float]:
    """Compute (umin, umax, vmin, vmax) in the (axis1, axis2) frame."""
    p = np.asarray(pts4, dtype=np.float32).reshape(-1, 2)
    a1 = np.asarray(axis1, dtype=np.float32).reshape(2)
    a2 = np.asarray(axis2, dtype=np.float32).reshape(2)
    o = np.asarray(origin, dtype=np.float32).reshape(2)
    d = p - o[None, :]
    u = d @ a1
    v = d @ a2
    return float(u.min()), float(u.max()), float(v.min()), float(v.max())


def line_angle_diff_deg(u: np.ndarray, v: np.ndarray) -> float:
    """Angle diff for undirected lines (0..90 deg)."""
    u = np.asarray(u, dtype=np.float64).reshape(2)
    v = np.asarray(v, dtype=np.float64).reshape(2)
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu <= 1e-12 or nv <= 1e-12:
        return 90.0
    u = u / nu
    v = v / nv
    c = float(np.clip(abs(np.dot(u, v)), 0.0, 1.0))
    return float(np.degrees(np.arccos(c)))
