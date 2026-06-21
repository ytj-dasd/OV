"""SAM3 visual-prompt instance segmentation using existing MapBuilder RGB parts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from landmark.tools.sam3.bev_part import FIXED_TILE_SIZE
from landmark.tools.sam3.instance_seg_v2 import (
    _append_warped_object_to_rotated,
    _rotate_label_map_back,
    _save_overlay,
)
from landmark.tools.sam3.sam3_about import (
    DEFAULT_CONDA_ENV,
    DEFAULT_SAM3_DIR,
    _resolve_sam3_python,
    get_connected_components_filtered_masks,
    get_overlap_reduced_masks,
    load_masks,
)


COMPOSITE_WIDTH = FIXED_TILE_SIZE * 2
COMPOSITE_HEIGHT = FIXED_TILE_SIZE


@dataclass(frozen=True)
class VisualSample:
    sample_id: str
    image_path: Path
    box_xyxy: np.ndarray
    grid_box_cxcywh: np.ndarray


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _box_to_grid_cxcywh(box_xyxy: np.ndarray) -> np.ndarray:
    box = np.asarray(box_xyxy, dtype=np.float32).reshape(4).copy()
    x0, y0, x1, y1 = [float(v) for v in box]
    return np.asarray(
        [
            (x0 + x1) * 0.5 / COMPOSITE_WIDTH,
            (y0 + y1) * 0.5 / COMPOSITE_HEIGHT,
            (x1 - x0) / COMPOSITE_WIDTH,
            (y1 - y0) / COMPOSITE_HEIGHT,
        ],
        dtype=np.float32,
    )


def load_visual_samples(manifest_path: Path | str) -> list[VisualSample]:
    """Load enabled 1008x1008 samples and map their boxes into the left composite half."""
    manifest_path = Path(manifest_path).expanduser()
    payload = _load_json(manifest_path)
    enabled = [item for item in payload.get("samples", []) if bool(item.get("enabled", True))]
    if not enabled:
        raise ValueError(f"Visual manhole manifest must contain at least one enabled sample: {manifest_path}")

    samples: list[VisualSample] = []
    seen: set[str] = set()
    for item in enabled:
        sample_id = str(item.get("id", "")).strip()
        if not sample_id or sample_id in seen:
            raise ValueError(f"Sample ids must be non-empty and unique: {sample_id!r}")
        seen.add(sample_id)
        image_path = manifest_path.parent / str(item["image"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image:
            if image.size != (FIXED_TILE_SIZE, FIXED_TILE_SIZE):
                raise ValueError(f"Sample image must be 1008x1008: {image_path} size={image.size}")
        box = np.asarray(item["box_xyxy"], dtype=np.float32).reshape(4)
        x0, y0, x1, y1 = [float(v) for v in box]
        if not (0 <= x0 < x1 <= FIXED_TILE_SIZE and 0 <= y0 < y1 <= FIXED_TILE_SIZE):
            raise ValueError(f"Invalid sample box for {sample_id}: {box.tolist()}")
        samples.append(
            VisualSample(
                sample_id=sample_id,
                image_path=image_path,
                box_xyxy=box,
                grid_box_cxcywh=_box_to_grid_cxcywh(box),
            )
        )
    return samples


def _read_rgb_tile(path: Path | str | None) -> np.ndarray:
    if path is None:
        return np.zeros((FIXED_TILE_SIZE, FIXED_TILE_SIZE, 3), dtype=np.uint8)
    image = np.asarray(Image.open(Path(path)).convert("RGB"))
    if image.shape != (FIXED_TILE_SIZE, FIXED_TILE_SIZE, 3):
        raise ValueError(f"Visual grid image must be 1008x1008 RGB: {path} shape={image.shape}")
    return image


def build_visual_grid(
    sample_path: Path | str,
    target_path: Path | str,
) -> np.ndarray:
    """Build a 1008x2016 composite with one sample left and one target right."""
    return np.concatenate([_read_rgb_tile(sample_path), _read_rgb_tile(target_path)], axis=1)


def split_grid_target_masks(
    masks: np.ndarray,
    scores: np.ndarray,
    *,
    target_name: str,
) -> dict[str, dict[str, np.ndarray]]:
    """Crop full-composite masks into the right target half."""
    masks = np.asarray(masks)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3 or masks.shape[1:] != (COMPOSITE_HEIGHT, COMPOSITE_WIDTH):
        raise ValueError(
            f"Expected masks shape (N,{COMPOSITE_HEIGHT},{COMPOSITE_WIDTH}), got {masks.shape}"
        )
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if scores.shape[0] != masks.shape[0]:
        raise ValueError("scores length must match masks")

    crops: list[np.ndarray] = []
    kept_scores: list[float] = []
    for mask, score in zip(masks, scores, strict=True):
        crop = np.asarray(mask[:, FIXED_TILE_SIZE:], dtype=bool)
        if np.any(crop):
            crops.append(crop)
            kept_scores.append(float(score))
    return {
        target_name: {
            "masks": np.stack(crops) if crops else np.zeros((0, FIXED_TILE_SIZE, FIXED_TILE_SIZE), dtype=bool),
            "scores": np.asarray(kept_scores, dtype=np.float32),
        }
    }


def cross_part_merge_by_bbox_iou(
    all_objs: list[tuple[np.ndarray, tuple[slice, slice], int]],
    image_hw: tuple[int, int],
    *,
    tile_grid_positions: list[tuple[int, int]],
    iou_threshold: float = 0.1,
) -> np.ndarray:
    """Merge instances from neighboring parts when their global bbox IoU reaches the threshold."""
    height, width = image_hw
    if not all_objs:
        return np.full((height, width), -1, dtype=np.int32)
    parent = list(range(len(all_objs)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    boxes: list[tuple[int, int, int, int]] = []
    for _mask, (rows, cols), _tile in all_objs:
        boxes.append((int(cols.start), int(rows.start), int(cols.stop), int(rows.stop)))
    for i, box_i in enumerate(boxes):
        tile_i = all_objs[i][2]
        row_i, col_i = tile_grid_positions[tile_i]
        for j in range(i + 1, len(boxes)):
            tile_j = all_objs[j][2]
            if tile_i != tile_j:
                row_j, col_j = tile_grid_positions[tile_j]
                if abs(row_i - row_j) > 1 or abs(col_i - col_j) > 1:
                    continue
            box_j = boxes[j]
            ix0, iy0 = max(box_i[0], box_j[0]), max(box_i[1], box_j[1])
            ix1, iy1 = min(box_i[2], box_j[2]), min(box_i[3], box_j[3])
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            area_i = max(0, box_i[2] - box_i[0]) * max(0, box_i[3] - box_i[1])
            area_j = max(0, box_j[2] - box_j[0]) * max(0, box_j[3] - box_j[1])
            union_area = area_i + area_j - inter
            iou = inter / float(union_area) if union_area > 0 else 0.0
            if iou >= float(iou_threshold):
                union(i, j)

    root_to_id: dict[int, int] = {}
    result = np.full((height, width), -1, dtype=np.int32)
    for index, (mask, (rows, cols), _tile) in enumerate(all_objs):
        root = find(index)
        obj_id = root_to_id.setdefault(root, len(root_to_id))
        view = result[rows, cols]
        view[np.asarray(mask, dtype=bool)] = obj_id
    return result


def _run_visual_grid_inference(
    grid_paths: list[Path],
    output_dirs: list[Path],
    prompt_boxes: list[list[float]],
    *,
    sam3_dir: Path,
    conda_env: str,
    score_threshold: float,
) -> None:
    adapter = Path(__file__).with_name("sam3_visual_inference_adapter.py")
    cmd = [
        *_resolve_sam3_python(conda_env),
        str(adapter),
        json.dumps([str(path.resolve()) for path in grid_paths], ensure_ascii=False),
        "--out",
        json.dumps([str(path.resolve()) for path in output_dirs], ensure_ascii=False),
        "--boxes",
        json.dumps(prompt_boxes),
        "--score-th",
        str(float(score_threshold)),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(sam3_dir)
    subprocess.run(cmd, cwd=str(sam3_dir), check=True, env=env)


def _validate_existing_parts(parts_json_path: Path) -> tuple[dict[str, Any], list[Path]]:
    if not parts_json_path.is_file():
        raise FileNotFoundError(f"Missing road RGB parts; run landmark-full through instance-seg-v2: {parts_json_path}")
    payload = _load_json(parts_json_path)
    parts_dir = parts_json_path.parent / "bev"
    tile_paths = [parts_dir / f"{part['tile_name']}.png" for part in payload["parts"]]
    for tile_path in tile_paths:
        if not tile_path.is_file():
            raise FileNotFoundError(f"Missing road RGB part: {tile_path}")
        with Image.open(tile_path) as image:
            if image.size != (FIXED_TILE_SIZE, FIXED_TILE_SIZE):
                raise ValueError(f"Road RGB part must be 1008x1008: {tile_path} size={image.size}")
    return payload, tile_paths


def _file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _cache_signature(
    *,
    parts_json_path: Path,
    manifest_path: Path,
    samples: list[VisualSample],
    rgb_filled_path: Path,
    score_threshold: float,
    iou_threshold: float,
) -> dict[str, Any]:
    payload = {
        "version": 3,
        "parts_json": _file_signature(parts_json_path),
        "manifest": _file_signature(manifest_path),
        "samples": [_file_signature(sample.image_path) for sample in samples],
        "rgb_filled": _file_signature(rgb_filled_path),
        "score_threshold": float(score_threshold),
        "iou_threshold": float(iou_threshold),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return {**payload, "sha256": hashlib.sha256(canonical).hexdigest()}


def run_visual_instance_seg(
    parts_json_path: Path | str,
    manifest_path: Path | str,
    rgb_filled_path: Path | str,
    output_dir: Path | str,
    *,
    sam3_dir: Path | str = DEFAULT_SAM3_DIR,
    conda_env: str = DEFAULT_CONDA_ENV,
    score_threshold: float = 0.5,
    iou_threshold: float = 0.1,
    force: bool = False,
) -> dict[str, Path]:
    """Run each visual sample against each target part and merge detections by IoU."""
    parts_json_path = Path(parts_json_path).expanduser()
    manifest_path = Path(manifest_path).expanduser()
    rgb_filled_path = Path(rgb_filled_path).expanduser()
    output_dir = Path(output_dir).expanduser()
    result_dir = output_dir / "result"
    label_map_path = result_dir / "label_map.npy"
    samples = load_visual_samples(manifest_path)
    signature = _cache_signature(
        parts_json_path=parts_json_path,
        manifest_path=manifest_path,
        samples=samples,
        rgb_filled_path=rgb_filled_path,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
    )
    signature_path = result_dir / "cache_signature.json"
    cached_signature = _load_json(signature_path) if signature_path.is_file() else None
    cache_valid = cached_signature == signature
    duplicate_overlay = result_dir / "rgb_filled_manhole_overlay.png"
    if duplicate_overlay.exists():
        duplicate_overlay.unlink()
    if label_map_path.is_file() and cache_valid and not force:
        return {
            "label_map": label_map_path,
            "objs": result_dir / "objs.png",
            "summary": result_dir / "summary.json",
        }

    parts_payload, tile_paths = _validate_existing_parts(parts_json_path)
    sam3_root = output_dir / "sam3"
    grid_root = sam3_root / "pairs"
    grid_root.mkdir(parents=True, exist_ok=True)

    part_by_name = {str(part["tile_name"]): part for part in parts_payload["parts"]}
    composite_specs: list[tuple[str, str, Path, Path]] = []
    for sample in samples:
        sample_grid_root = grid_root / sample.sample_id
        sample_grid_root.mkdir(parents=True, exist_ok=True)
        sample_specs: list[tuple[str, str, Path, Path]] = []
        for tile_path in tile_paths:
            grid_path = sample_grid_root / f"{tile_path.stem}.png"
            Image.fromarray(build_visual_grid(sample.image_path, tile_path)).save(grid_path)
            sample_specs.append(
                (sample.sample_id, tile_path.stem, grid_path, sample_grid_root / tile_path.stem)
            )
        composite_specs.extend(sample_specs)
        missing = [
            spec for spec in sample_specs
            if force or not cache_valid or not (spec[3] / "masks.npz").is_file()
        ]
        if missing:
            _run_visual_grid_inference(
                [spec[2] for spec in missing],
                [spec[3] for spec in missing],
                [sample.grid_box_cxcywh.tolist()],
                sam3_dir=Path(sam3_dir).expanduser(),
                conda_env=conda_env,
                score_threshold=score_threshold,
            )

    all_objs: list[tuple[np.ndarray, tuple[slice, slice], int]] = []
    tile_grid_positions = [(int(part["grid_row"]), int(part["grid_col"])) for part in parts_payload["parts"]]
    part_index = {str(part["tile_name"]): idx for idx, part in enumerate(parts_payload["parts"])}
    tile_summaries: list[dict[str, Any]] = []
    part_masks: dict[str, list[np.ndarray]] = {name: [] for name in part_by_name}
    part_scores: dict[str, list[float]] = {name: [] for name in part_by_name}
    sample_summaries: list[dict[str, Any]] = []
    for sample_id, target_name, _grid_path, composite_out_dir in tqdm(
        composite_specs,
        desc="Merge manhole samples",
        unit="composite",
    ):
        with np.load(composite_out_dir / "masks.npz") as data:
            split = split_grid_target_masks(data["masks"], data["scores"], target_name=target_name)
        for target_name, target_result in split.items():
            masks = get_connected_components_filtered_masks(target_result["masks"], angle_threshold=None)
            masks = get_overlap_reduced_masks(masks, overlap_regions=[], cover_ratio=0.9)
            part_masks[target_name].extend(list(masks))
            part_scores[target_name].extend(float(v) for v in target_result["scores"][: len(masks)])
            sample_summaries.append(
                {"sample": sample_id, "tile": target_name, "num_masks": int(len(masks))}
            )

    for target_name in tqdm(part_by_name, desc="Restore manhole parts", unit="part"):
        masks = np.asarray(part_masks[target_name], dtype=bool)
        scores = np.asarray(part_scores[target_name], dtype=np.float32)
        part = part_by_name[target_name]
        idx = part_index[target_name]
        accepted = 0
        part_label = np.full((FIXED_TILE_SIZE, FIXED_TILE_SIZE), -1, dtype=np.int32)
        for local_id, mask in enumerate(masks):
            part_label[mask] = local_id
            accepted += int(
                _append_warped_object_to_rotated(
                    all_objs,
                    mask,
                    part,
                    rotated_shape=tuple(int(v) for v in parts_payload["rotated_shape"]),
                    tile_index=idx,
                )
            )
        part_out = sam3_root / target_name
        part_out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            part_out / "masks.npz",
            masks=masks,
            scores=scores,
        )
        _save_overlay(np.asarray(Image.open(tile_paths[idx]).convert("RGB")), part_label, part_out / "mask_overlay.png")
        tile_summaries.append({"tile": target_name, "num_masks": int(len(masks)), "num_accepted": int(accepted)})

    rotated_label_map = cross_part_merge_by_bbox_iou(
        all_objs,
        tuple(int(v) for v in parts_payload["rotated_shape"]),
        tile_grid_positions=tile_grid_positions,
        iou_threshold=float(iou_threshold),
    )
    final_label_map = _rotate_label_map_back(
        rotated_label_map,
        np.asarray(parts_payload["rotated_to_original_affine"], dtype=np.float32),
        tuple(int(v) for v in parts_payload["original_shape"]),
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    np.save(label_map_path, final_label_map)
    rgb = np.asarray(Image.open(Path(rgb_filled_path)).convert("RGB"))
    _save_overlay(rgb, final_label_map, result_dir / "objs.png")
    summary = {
        "parts_json": str(parts_json_path),
        "manifest": str(Path(manifest_path)),
        "num_parts": len(tile_paths),
        "num_samples": len(samples),
        "num_composites": len(composite_specs),
        "iou_threshold": float(iou_threshold),
        "final_obj_num": int(final_label_map.max()) + 1 if int(final_label_map.max()) >= 0 else 0,
        "tile_summaries": tile_summaries,
        "sample_summaries": sample_summaries,
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    signature_path.write_text(json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "label_map": label_map_path,
        "objs": result_dir / "objs.png",
        "summary": result_dir / "summary.json",
    }
