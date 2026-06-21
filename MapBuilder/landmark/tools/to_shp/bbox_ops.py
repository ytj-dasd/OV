"""Bounding-box initialisation, merge, and visualisation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from landmark.tools.to_shp.geometry import (
    as_rect4_px,
    line_angle_diff_deg,
    pixel_to_xy,
    rect_main_dir_px,
    rect_overlap_ratio_px,
    rect_uv_bounds_px,
)


def merge_overlapping_collinear_bboxes(
    bbox_list: list[dict[str, Any]],
    bev_meta: dict[str, Any],
    *,
    iou_threshold: float = 0.2,
    angle_threshold_deg: float = 10.0,
    center_perp_threshold_px: float = 20.0,
) -> list[dict[str, Any]]:
    """Merge bboxes that overlap and belong to the same lane line."""
    if not bbox_list:
        return []

    rects: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    dirs: list[np.ndarray] = []
    for bbox in bbox_list:
        pts4 = as_rect4_px(bbox.get("pixel_corners"))
        rects.append(pts4)
        c = pts4.mean(axis=0)
        centers.append(c)
        d = rect_main_dir_px(pts4)
        if d[0] < 0:
            d = -d
        dirs.append(d)

    n = len(bbox_list)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    iou_threshold = float(iou_threshold)
    angle_threshold_deg = float(angle_threshold_deg)
    center_perp_threshold_px = float(center_perp_threshold_px)

    for i in range(n):
        for j in range(i + 1, n):
            if rect_overlap_ratio_px(rects[i], rects[j]) < iou_threshold:
                continue
            if line_angle_diff_deg(dirs[i], dirs[j]) > angle_threshold_deg:
                continue
            d = (centers[j] - centers[i]).astype(np.float32)
            dir_ = dirs[i]
            perp = np.array([-dir_[1], dir_[0]], dtype=np.float32)
            perp_dist = float(abs(np.dot(d, perp)))
            if perp_dist > center_perp_threshold_px:
                continue
            union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)

    merged: list[dict[str, Any]] = []
    for _, idxs in groups.items():
        if len(idxs) == 1:
            merged.append(bbox_list[idxs[0]])
            continue

        dir_stack = np.stack([dirs[k] for k in idxs], axis=0).astype(np.float32)
        dir_mean = dir_stack.mean(axis=0)
        nrm = float(np.linalg.norm(dir_mean))
        axis1 = (dir_mean / (nrm + 1e-12)).astype(np.float32)
        if axis1[0] < 0:
            axis1 = -axis1
        axis2 = np.array([-axis1[1], axis1[0]], dtype=np.float32)

        origin = (
            np.stack([centers[k] for k in idxs], axis=0).mean(axis=0).astype(np.float32)
        )
        umins: list[float] = []
        umaxs: list[float] = []
        vcenters: list[float] = []
        vexts: list[float] = []
        for k in idxs:
            umin, umax, vmin, vmax = rect_uv_bounds_px(rects[k], axis1, axis2, origin)
            umins.append(float(umin))
            umaxs.append(float(umax))
            vc = 0.5 * (vmin + vmax)
            ve = float(vmax - vmin)
            vcenters.append(float(vc))
            vexts.append(float(ve))

        umin_g = float(np.min(umins))
        umax_g = float(np.max(umaxs))
        v_center = float(np.median(np.asarray(vcenters, dtype=np.float32)))
        v_extent = float(np.max(np.asarray(vexts, dtype=np.float32)))
        vmin_g = v_center - 0.5 * v_extent
        vmax_g = v_center + 0.5 * v_extent

        cu = 0.5 * (umin_g + umax_g)
        cv = 0.5 * (vmin_g + vmax_g)
        center_pix = origin + cu * axis1 + cv * axis2
        du = 0.5 * (umax_g - umin_g)
        dv = 0.5 * (vmax_g - vmin_g)
        box_pix = np.stack(
            [
                center_pix + du * axis1 + dv * axis2,
                center_pix - du * axis1 + dv * axis2,
                center_pix - du * axis1 - dv * axis2,
                center_pix + du * axis1 - dv * axis2,
            ],
            axis=0,
        ).astype(np.float32)

        corners_xy = pixel_to_xy(box_pix, bev_meta).astype(np.float32)

        z_vals = []
        for k in idxs:
            c = bbox_list[k].get("center")
            if isinstance(c, (list, tuple)) and len(c) >= 3:
                z_vals.append(float(c[2]))
        z_mean = float(np.mean(z_vals)) if z_vals else 0.0

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

        merged_id = min(int(bbox_list[k].get("id", k)) for k in idxs)
        merged.append(
            {
                "id": merged_id,
                "center": [float(center_xy[0]), float(center_xy[1]), z_mean],
                "yaw": yaw,
                "size": [float(size_lw[0]), float(size_lw[1])],
                "corners_xy": corners_xy.tolist(),
                "pixel_corners": box_pix.tolist(),
            }
        )

    merged.sort(key=lambda b: int(b.get("id", 0)))
    return merged


def get_init_bbox(*args) -> list[dict[str, Any]]:
    """Compute oriented bboxes from masks and write init_bboxes.json."""
    if len(args) == 4:
        _points, masks, bev_meta, output_path = args
    elif len(args) == 3:
        masks, bev_meta, output_path = args
    else:
        raise TypeError("get_init_bbox expects (masks, bev_meta, output_path) or legacy (points, masks, bev_meta, output_path)")

    masks = np.asarray(masks)
    if masks.ndim not in {2, 3}:
        raise ValueError("masks must have shape (K, H, W) or (H, W) after filtering")

    out_dir = Path(output_path).expanduser().absolute()
    out_dir.mkdir(parents=True, exist_ok=True)

    def _mask_to_bbox(mask_closed: np.ndarray, obj_id: int, *, row_offset: int = 0, col_offset: int = 0) -> dict[str, Any] | None:
        ys, xs = np.where(mask_closed)
        if ys.size < 10:
            return None

        pts_pix = np.column_stack([xs + col_offset, ys + row_offset]).astype(np.float32)
        rect = cv2.minAreaRect(pts_pix)
        box_pix = cv2.boxPoints(rect).astype(np.float32)

        corners_xy = pixel_to_xy(box_pix, bev_meta).astype(np.float32)
        corners_pix = box_pix

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
            "pixel_corners": corners_pix.tolist(),
        }

    bbox_list: list[dict[str, Any]] = []
    mpp = float(bev_meta["meters_per_pixel"])
    close_m = 0.10
    k = int(round(close_m / mpp))
    k = max(1, min(k, 7))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    if masks.ndim == 3:
        for i in range(masks.shape[0]):
            mask = masks[i].astype(bool)
            if mask.shape[0] != int(bev_meta["height"]) or mask.shape[1] != int(bev_meta["width"]):
                raise ValueError(
                    "Mask resolution does not match BEV image/meta. "
                    f"mask={mask.shape}, meta={(bev_meta['height'], bev_meta['width'])}"
                )
            mask_u8 = mask.astype(np.uint8) * 255
            mask_closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel) > 0
            bbox = _mask_to_bbox(mask_closed, i)
            if bbox is not None:
                bbox_list.append(bbox)
    else:
        from scipy import ndimage

        label_map = np.asarray(masks, dtype=np.int32)
        positive_labels = label_map + 1
        object_slices = ndimage.find_objects(positive_labels)
        for obj_id, slc in enumerate(object_slices):
            if slc is None:
                continue
            r_slc, c_slc = slc
            r0, r1 = int(r_slc.start), int(r_slc.stop)
            c0, c1 = int(c_slc.start), int(c_slc.stop)
            crop = np.asarray(label_map[r0:r1, c0:c1] == obj_id)
            if not np.any(crop):
                continue
            mask_u8 = crop.astype(np.uint8) * 255
            mask_closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel) > 0
            bbox = _mask_to_bbox(mask_closed, obj_id, row_offset=r0, col_offset=c0)
            if bbox is not None:
                bbox_list.append(bbox)

    save_path = out_dir / "init_bboxes.json"
    payload = {"bev_meta": bev_meta, "bboxes": bbox_list}
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return bbox_list


def vis_bbox(
    img_path: Path | str,
    bbox_list: list[dict[str, Any]],
    is_fill: bool = False,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Draw oriented bboxes on an image and return (rgb_img, features)."""
    img_path = Path(img_path).expanduser().absolute()
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    rng = np.random.default_rng()
    features: list[dict[str, Any]] = []
    for idx, bbox in enumerate(bbox_list):
        corners = np.asarray(bbox["pixel_corners"], dtype=np.float32)
        pts = np.round(corners).astype(np.int32).reshape((-1, 1, 2))
        color = tuple(int(v) for v in rng.integers(32, 256, size=3))
        if is_fill:
            cv2.fillPoly(img, [pts], color=color)
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=1)
        else:
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
        features.append({"object_id": bbox.get("id", idx), "color": color})

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img, features
