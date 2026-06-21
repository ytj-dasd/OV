"""Cross-mode mask fusion for SAM3 instance outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _iter_object_crops(
    label_map: np.ndarray,
) -> list[tuple[int, np.ndarray, int, int, int, int]]:
    """Return cropped masks as (obj_id, crop, r0, r1, c0, c1)."""
    ids = np.unique(label_map)
    ids = ids[ids >= 0]
    results: list[tuple[int, np.ndarray, int, int, int, int]] = []
    for oid in ids:
        rows, cols = np.where(label_map == oid)
        if rows.size == 0:
            continue
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        c0, c1 = int(cols.min()), int(cols.max()) + 1
        crop = label_map[r0:r1, c0:c1] == oid
        results.append((int(oid), crop, r0, r1, c0, c1))
    return results


def fuse_cross_mode_masks(
    objs_dir: Path | str,
    prompt_dir: str,
    modes: list[str],
    *,
    overlap_threshold: float = 0.5,
) -> Path | None:
    """Fuse per-mode label maps into one label map under ``fused/``."""
    objs_dir = Path(objs_dir).expanduser()
    label_maps: list[np.ndarray] = []
    for mode in modes:
        path = objs_dir / prompt_dir / mode / "final_masks.npy"
        if path.is_file():
            label_maps.append(np.load(path))

    if not label_maps:
        return None

    fused_dir = objs_dir / prompt_dir / "fused"
    fused_dir.mkdir(parents=True, exist_ok=True)
    out_path = fused_dir / "final_masks.npy"

    if len(label_maps) == 1:
        np.save(out_path, label_maps[0])
        return out_path

    crops_a = _iter_object_crops(label_maps[0])
    crops_b = _iter_object_crops(label_maps[1])
    fused = np.full(label_maps[0].shape, -1, dtype=np.int32)
    next_id = 0
    matched_b: set[int] = set()

    for _, a_crop, a_r0, a_r1, a_c0, a_c1 in crops_a:
        a_area = int(np.sum(a_crop))
        best_b_idx: int | None = None
        best_overlap = 0.0

        for bi, (_, b_crop, b_r0, b_r1, b_c0, b_c1) in enumerate(crops_b):
            if bi in matched_b:
                continue

            ir0 = max(a_r0, b_r0)
            ir1 = min(a_r1, b_r1)
            ic0 = max(a_c0, b_c0)
            ic1 = min(a_c1, b_c1)
            if ir0 >= ir1 or ic0 >= ic1:
                continue

            a_sub = a_crop[ir0 - a_r0:ir1 - a_r0, ic0 - a_c0:ic1 - a_c0]
            b_sub = b_crop[ir0 - b_r0:ir1 - b_r0, ic0 - b_c0:ic1 - b_c0]
            inter = int(np.sum(a_sub & b_sub))
            if inter == 0:
                continue

            b_area = int(np.sum(b_crop))
            ratio = inter / min(a_area, b_area)
            if ratio > best_overlap:
                best_overlap = ratio
                best_b_idx = bi

        fused[a_r0:a_r1, a_c0:a_c1][a_crop] = next_id
        if best_b_idx is not None and best_overlap >= overlap_threshold:
            _, b_crop, b_r0, b_r1, b_c0, b_c1 = crops_b[best_b_idx]
            fused[b_r0:b_r1, b_c0:b_c1][b_crop] = next_id
            matched_b.add(best_b_idx)
        next_id += 1

    for bi, (_, b_crop, b_r0, b_r1, b_c0, b_c1) in enumerate(crops_b):
        if bi in matched_b:
            continue
        fused[b_r0:b_r1, b_c0:b_c1][b_crop] = next_id
        next_id += 1

    np.save(out_path, fused)
    return out_path
