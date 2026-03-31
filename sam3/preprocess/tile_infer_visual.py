from __future__ import annotations

import argparse
import colorsys
from pathlib import Path
import sys
from typing import Any, TYPE_CHECKING

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if TYPE_CHECKING:
    from sam3.model.sam3_image_processor import Sam3Processor

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MOSTLY_BLACK_TILE_SKIP_RATIO = 0.8
DEFAULT_CHECKPOINT = "/home/guitu/文档/vector/sam3/model/sam3.pt"


def _tile_starts(full: int, tile: int, stride: int) -> list[int]:
    if tile <= 0:
        raise ValueError("tile must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")
    if full <= tile:
        return [0]
    starts = list(range(0, full - tile + 1, stride))
    last = full - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def _normalize_hw(value: Any, name: str) -> tuple[int, int]:
    """Normalize a scalar or (h, w) pair into (h, w) ints."""
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"{name} must be an int or a pair (h, w)")
        h, w = int(value[0]), int(value[1])
        return h, w
    return int(value), int(value)


def _normalize_mask_array(masks: np.ndarray) -> np.ndarray:
    """Convert SAM3 mask output to bool array with shape (N, H, W)."""
    arr = np.asarray(masks)
    if arr.ndim == 4:
        if arr.shape[1] != 1:
            raise ValueError(f"Expected masks with shape (N,1,H,W), got {arr.shape}")
        arr = arr[:, 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected masks with shape (N,H,W), got {arr.shape}")
    return arr.astype(bool, copy=False)


def _bbox_iou_xyxy(box1: np.ndarray, box2: np.ndarray) -> float:
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0

    area1 = max(0.0, float(box1[2] - box1[0])) * max(0.0, float(box1[3] - box1[1]))
    area2 = max(0.0, float(box2[2] - box2[0])) * max(0.0, float(box2[3] - box2[1]))
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def _bbox_from_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return np.zeros((4,), dtype=np.float32)
    return np.array(
        [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1],
        dtype=np.float32,
    )


def merge_instances_by_iou(
    masks: np.ndarray,
    boxes: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    iou_threshold: float = 0.2,
) -> dict[str, np.ndarray]:
    """Merge instance masks when bbox IoU is above threshold."""
    if not (0.0 <= iou_threshold <= 1.0):
        raise ValueError("iou_threshold must be in [0, 1]")

    masks_3d = _normalize_mask_array(masks)
    n = masks_3d.shape[0]
    h, w = masks_3d.shape[1], masks_3d.shape[2]
    if n == 0:
        return {
            "masks": np.zeros((0, 1, h, w), dtype=bool),
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "scores": np.zeros((0,), dtype=np.float32),
        }

    if boxes is None:
        boxes_arr = np.stack([_bbox_from_mask(m) for m in masks_3d], axis=0)
    else:
        boxes_arr = np.asarray(boxes, dtype=np.float32)
        if boxes_arr.ndim == 1 and boxes_arr.size == 4:
            boxes_arr = boxes_arr.reshape(1, 4)
        if boxes_arr.shape != (n, 4):
            raise ValueError(f"boxes shape must be ({n}, 4), got {boxes_arr.shape}")

    if scores is None:
        scores_arr = np.ones((n,), dtype=np.float32)
    else:
        scores_arr = np.asarray(scores, dtype=np.float32).reshape(-1)
        if scores_arr.shape[0] != n:
            raise ValueError(f"scores length must be {n}, got {scores_arr.shape[0]}")

    order = np.argsort(-scores_arr)
    instances = [
        {
            "mask": masks_3d[i].copy(),
            "box": boxes_arr[i].copy(),
            "score": float(scores_arr[i]),
        }
        for i in order
    ]

    merged = True
    while merged:
        merged = False
        for i in range(len(instances)):
            for j in range(i + 1, len(instances)):
                iou = _bbox_iou_xyxy(instances[i]["box"], instances[j]["box"])
                if iou <= iou_threshold:
                    continue

                instances[i]["mask"] = instances[i]["mask"] | instances[j]["mask"]
                instances[i]["box"] = _bbox_from_mask(instances[i]["mask"])
                instances[i]["score"] = max(instances[i]["score"], instances[j]["score"])
                del instances[j]
                merged = True
                break
            if merged:
                break

    out_masks = np.stack([x["mask"] for x in instances], axis=0)[:, None, :, :]
    out_boxes = np.stack([x["box"] for x in instances], axis=0).astype(np.float32)
    out_scores = np.array([x["score"] for x in instances], dtype=np.float32)
    return {"masks": out_masks, "boxes": out_boxes, "scores": out_scores}


def _instance_color(index: int) -> np.ndarray:
    golden_ratio = 0.618033988749895
    hue = (index * golden_ratio) % 1.0
    sat = 0.75
    val = 0.95
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return np.array([int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8)


def draw_instances_on_image(
    image: np.ndarray,
    masks: np.ndarray,
    *,
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay each instance mask with a distinct color on the original image."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected image shape (H,W,3), got {image.shape}")
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")

    masks_3d = _normalize_mask_array(masks)
    vis = image.astype(np.float32, copy=True)
    for idx, mask in enumerate(masks_3d):
        if not np.any(mask):
            continue
        color = _instance_color(idx).astype(np.float32)
        vis[mask] = (1.0 - alpha) * vis[mask] + alpha * color
    return np.clip(vis, 0, 255).astype(np.uint8)


def _save_merged_npz(
    merged: dict[str, np.ndarray],
    out_npz_path: Path,
    *,
    img_path: Path,
    concept: str,
    image_shape: tuple[int, int],
    sample_image: Path,
    sample_box_xyxy: np.ndarray,
) -> None:
    masks = np.asarray(merged["masks"])
    boxes = np.asarray(merged["boxes"], dtype=np.float32)
    scores = np.asarray(merged["scores"], dtype=np.float32)
    np.savez_compressed(
        out_npz_path,
        masks=masks,
        boxes=boxes,
        scores=scores,
        concept=np.array(concept),
        source_image=np.array(str(img_path)),
        image_stem=np.array(img_path.stem),
        image_height=np.array(int(image_shape[0]), dtype=np.int32),
        image_width=np.array(int(image_shape[1]), dtype=np.int32),
        sample_image=np.array(str(sample_image)),
        sample_box_xyxy=sample_box_xyxy.astype(np.float32),
    )
    print(f"Merged masks/boxes/scores saved to: {out_npz_path}")


def _safe_name(text: str) -> str:
    out = text.strip().replace(" ", "_")
    out = "".join(ch if (ch.isalnum() or ch in {"_", "-", "."}) else "_" for ch in out)
    out = out.strip("._")
    return out or "visual_retrieval"


def _clip_box_xyxy(box_xyxy: np.ndarray, *, h: int, w: int) -> np.ndarray:
    box = np.asarray(box_xyxy, dtype=np.float32).reshape(4)
    x0, y0, x1, y1 = [float(v) for v in box]
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0

    x0 = min(max(x0, 0.0), float(w - 1))
    y0 = min(max(y0, 0.0), float(h - 1))
    x1 = min(max(x1, 0.0), float(w))
    y1 = min(max(y1, 0.0), float(h))

    if x1 <= x0:
        x1 = min(float(w), x0 + 1.0)
        x0 = max(0.0, x1 - 1.0)
    if y1 <= y0:
        y1 = min(float(h), y0 + 1.0)
        y0 = max(0.0, y1 - 1.0)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def _crop_patch_around_box(
    image: np.ndarray,
    box_xyxy: np.ndarray,
    *,
    patch_h: int,
    patch_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Crop a fixed-size patch around box center.
    If near borders, shift crop window into image; if image is smaller than patch, pad with black.
    Returns (patch, box_xyxy_in_patch_pixels).
    """
    h, w = int(image.shape[0]), int(image.shape[1])
    box = _clip_box_xyxy(box_xyxy, h=h, w=w)

    cx = 0.5 * (float(box[0]) + float(box[2]))
    cy = 0.5 * (float(box[1]) + float(box[3]))
    x_start = int(np.floor(cx - patch_w / 2.0))
    y_start = int(np.floor(cy - patch_h / 2.0))

    if w > patch_w:
        x_start = min(max(0, x_start), w - patch_w)
    else:
        x_start = 0
    if h > patch_h:
        y_start = min(max(0, y_start), h - patch_h)
    else:
        y_start = 0

    x_end = min(x_start + patch_w, w)
    y_end = min(y_start + patch_h, h)

    crop = image[y_start:y_end, x_start:x_end]
    patch = np.zeros((patch_h, patch_w, 3), dtype=np.uint8)
    crop_h, crop_w = crop.shape[0], crop.shape[1]
    patch[:crop_h, :crop_w] = crop

    box_patch = box.copy()
    box_patch[[0, 2]] -= float(x_start)
    box_patch[[1, 3]] -= float(y_start)
    box_patch[0] = np.clip(box_patch[0], 0.0, float(patch_w - 1))
    box_patch[2] = np.clip(box_patch[2], 0.0, float(patch_w))
    box_patch[1] = np.clip(box_patch[1], 0.0, float(patch_h - 1))
    box_patch[3] = np.clip(box_patch[3], 0.0, float(patch_h))
    if box_patch[2] <= box_patch[0]:
        box_patch[2] = min(float(patch_w), box_patch[0] + 1.0)
    if box_patch[3] <= box_patch[1]:
        box_patch[3] = min(float(patch_h), box_patch[1] + 1.0)

    return patch, box_patch


def _box_xyxy_to_cxcywh_norm(box_xyxy: np.ndarray, *, h: int, w: int) -> list[float]:
    x0, y0, x1, y1 = [float(v) for v in box_xyxy.reshape(4)]
    cx = 0.5 * (x0 + x1) / float(w)
    cy = 0.5 * (y0 + y1) / float(h)
    bw = (x1 - x0) / float(w)
    bh = (y1 - y0) / float(h)
    return [cx, cy, bw, bh]


def _prepare_sample_patch(
    sample_image_path: Path,
    sample_box_xyxy: np.ndarray,
    *,
    patch_h: int,
    patch_w: int,
) -> tuple[np.ndarray, list[float], np.ndarray]:
    sample_img = np.array(Image.open(sample_image_path).convert("RGB"))
    sample_patch, sample_box_in_patch = _crop_patch_around_box(
        sample_img,
        sample_box_xyxy,
        patch_h=patch_h,
        patch_w=patch_w,
    )

    concat_h = patch_h
    concat_w = patch_w * 2
    prompt_box_norm = _box_xyxy_to_cxcywh_norm(
        sample_box_in_patch,
        h=concat_h,
        w=concat_w,
    )
    return sample_patch, prompt_box_norm, sample_box_in_patch


def _collect_image_paths(input_path: Path, recursive: bool = False) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if recursive:
        image_paths = [
            p for p in sorted(input_path.rglob("*"))
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
        ]
    else:
        image_paths = [
            p for p in sorted(input_path.iterdir())
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
        ]
    if not image_paths:
        raise FileNotFoundError(
            f"No images found in directory: {input_path} (supported: {sorted(SUPPORTED_IMAGE_EXTS)})"
        )
    return image_paths


def _get_visual_retrieval_masks(
    img: np.ndarray,
    *,
    processor: Sam3Processor,
    sample_patch: np.ndarray,
    sample_prompt_box_norm: list[float],
    tile_size: tuple[int, int] = (2048, 1024),
    tile_overlap: tuple[int, int] = (200, 100),
    iou_threshold: float = 0.2,
) -> dict[str, np.ndarray]:
    h, w = int(img.shape[0]), int(img.shape[1])
    tile_h, tile_w = _normalize_hw(tile_size, "tile_size")
    overlap_h, overlap_w = _normalize_hw(tile_overlap, "tile_overlap")
    if tile_h <= 0 or tile_w <= 0:
        raise ValueError("tile_size must be > 0 (int) or (h,w) with both > 0")
    if overlap_h < 0 or overlap_w < 0:
        raise ValueError("tile_overlap must be >= 0 (int) or (h,w) with both >= 0")
    if overlap_h >= tile_h or overlap_w >= tile_w:
        raise ValueError("tile_overlap must be smaller than tile_size in both dimensions")
    if sample_patch.shape[:2] != (tile_h, tile_w):
        raise ValueError(
            f"sample_patch shape must be {(tile_h, tile_w, 3)}, got {sample_patch.shape}"
        )

    stride_h = max(1, tile_h - overlap_h)
    stride_w = max(1, tile_w - overlap_w)
    ys = _tile_starts(h, tile_h, stride_h)
    xs = _tile_starts(w, tile_w, stride_w)

    global_masks: list[np.ndarray] = []
    global_boxes: list[np.ndarray] = []
    global_scores: list[float] = []

    tile_regions = [
        (y0, x0, min(y0 + tile_h, h), min(x0 + tile_w, w))
        for y0 in ys
        for x0 in xs
    ]

    for y0, x0, y1, x1 in tqdm(tile_regions, desc="Tile retrieval inference", unit="tile"):
        tile_valid = img[y0:y1, x0:x1]
        if tile_valid.ndim >= 3:
            black_pixels = np.all(tile_valid == 0, axis=-1)
        else:
            black_pixels = tile_valid == 0
        black_ratio = float(np.count_nonzero(black_pixels)) / float(black_pixels.size)
        if black_ratio > MOSTLY_BLACK_TILE_SKIP_RATIO:
            continue

        tile_patch = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        valid_h, valid_w = int(tile_valid.shape[0]), int(tile_valid.shape[1])
        tile_patch[:valid_h, :valid_w] = tile_valid

        concat_img = np.concatenate([sample_patch, tile_patch], axis=1)
        state = processor.set_image(Image.fromarray(concat_img))
        out = processor.add_geometric_prompt(
            box=sample_prompt_box_norm,
            label=True,
            state=state,
        )

        masks_np = _normalize_mask_array(out["masks"].detach().cpu().numpy())
        scores_np = out["scores"].detach().cpu().numpy().reshape(-1)
        n_masks = int(masks_np.shape[0])
        if n_masks == 0:
            continue

        for i in range(n_masks):
            tile_side_mask = masks_np[i, :, tile_w : (tile_w + tile_w)]
            tile_side_valid = tile_side_mask[:valid_h, :valid_w]
            if not np.any(tile_side_valid):
                continue

            local_box = _bbox_from_mask(tile_side_valid)
            if local_box[2] <= local_box[0] or local_box[3] <= local_box[1]:
                continue

            global_box = local_box + np.array([x0, y0, x0, y0], dtype=np.float32)
            canvas = np.zeros((h, w), dtype=bool)
            canvas[y0:y1, x0:x1] = tile_side_valid

            global_masks.append(canvas)
            global_boxes.append(global_box)
            global_scores.append(float(scores_np[i]))

    if len(global_masks) == 0:
        return {
            "masks": np.zeros((0, 1, h, w), dtype=bool),
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "scores": np.zeros((0,), dtype=np.float32),
        }

    return merge_instances_by_iou(
        masks=np.stack(global_masks, axis=0),
        boxes=np.stack(global_boxes, axis=0),
        scores=np.asarray(global_scores, dtype=np.float32),
        iou_threshold=iou_threshold,
    )


def run_visual_retrieval_inference(
    img_path: Path | str,
    *,
    processor: Sam3Processor,
    sample_patch: np.ndarray,
    sample_prompt_box_norm: list[float],
    sample_image_path: Path,
    sample_box_xyxy: np.ndarray,
    tile_size: tuple[int, int] = (2048, 1024),
    tile_overlap: tuple[int, int] = (200, 100),
    iou_threshold: float = 0.2,
    alpha: float = 0.45,
    output_path: Path | str | None = None,
    concept_name: str = "visual_retrieval",
) -> dict[str, np.ndarray]:
    """Run tiled visual retrieval segmentation and save merged visualization."""
    img_path = Path(img_path).expanduser().absolute()
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    image = np.array(Image.open(img_path).convert("RGB"))
    merged = _get_visual_retrieval_masks(
        image,
        processor=processor,
        sample_patch=sample_patch,
        sample_prompt_box_norm=sample_prompt_box_norm,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        iou_threshold=iou_threshold,
    )

    vis = draw_instances_on_image(image, merged["masks"], alpha=alpha)
    safe_name = _safe_name(concept_name)
    if output_path is None:
        out_path = img_path.parent / f"{img_path.stem}_{safe_name}_merged.png"
    else:
        out_path = Path(output_path).expanduser().absolute()
    Image.fromarray(vis).save(out_path)
    out_npz_path = out_path.with_suffix(".npz")
    _save_merged_npz(
        merged,
        out_npz_path,
        img_path=img_path,
        concept=concept_name,
        image_shape=(int(image.shape[0]), int(image.shape[1])),
        sample_image=sample_image_path,
        sample_box_xyxy=sample_box_xyxy,
    )
    print(f"Merged visualization saved to: {out_path}")
    print(f"Merged instances: {merged['masks'].shape[0]}")
    return merged


def run_visual_retrieval_inference_batch(
    input_path: Path | str,
    *,
    processor: Sam3Processor,
    sample_patch: np.ndarray,
    sample_prompt_box_norm: list[float],
    sample_image_path: Path,
    sample_box_xyxy: np.ndarray,
    tile_size: tuple[int, int] = (2048, 1024),
    tile_overlap: tuple[int, int] = (200, 100),
    iou_threshold: float = 0.2,
    alpha: float = 0.45,
    output_path: Path | str | None = None,
    recursive: bool = False,
    concept_name: str = "visual_retrieval",
) -> dict[str, dict[str, np.ndarray]]:
    """
    Run visual retrieval inference on a single image or all images in a directory.
    - Single file: output_path behaves like run_visual_retrieval_inference(output file path).
    - Directory: output_path must be a directory path; each image writes one merged result.
    """
    input_path = Path(input_path).expanduser().absolute()
    image_paths = _collect_image_paths(input_path, recursive=recursive)
    is_single_image = input_path.is_file()

    output_dir: Path | None = None
    if not is_single_image and output_path is not None:
        output_dir = Path(output_path).expanduser().absolute()
        if output_dir.suffix:
            raise ValueError("When input is a directory, --output must be an output directory path.")
        output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, np.ndarray]] = {}
    iterator = image_paths
    if not is_single_image:
        iterator = tqdm(image_paths, desc="Image retrieval inference", unit="img")

    safe_name = _safe_name(concept_name)
    for img_path in iterator:
        per_image_output: Path | str | None = output_path
        if not is_single_image:
            if output_dir is None:
                per_image_output = None
            else:
                per_image_output = output_dir / f"{img_path.stem}_{safe_name}_merged.png"

        merged = run_visual_retrieval_inference(
            img_path=img_path,
            processor=processor,
            sample_patch=sample_patch,
            sample_prompt_box_norm=sample_prompt_box_norm,
            sample_image_path=sample_image_path,
            sample_box_xyxy=sample_box_xyxy,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            iou_threshold=iou_threshold,
            alpha=alpha,
            output_path=per_image_output,
            concept_name=concept_name,
        )
        results[str(img_path)] = merged
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tile-based visual retrieval segmentation with SAM3 (sample patch on the left, tile on the right)."
    )
    parser.add_argument("input_path", type=str, help="Input image path or image directory")
    parser.add_argument(
        "--sample-image",
        type=str,
        required=True,
        help="Sample image path containing the target object",
    )
    parser.add_argument(
        "--sample-box",
        nargs=4,
        type=float,
        required=True,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Sample bounding box in sample-image pixel XYXY",
    )
    parser.add_argument("--tile-height", type=int, default=2048, help="Tile (and sample patch) height")
    parser.add_argument("--tile-width", type=int, default=1024, help="Tile (and sample patch) width")
    parser.add_argument("--tile-overlap-height", type=int, default=200, help="Tile overlap height")
    parser.add_argument("--tile-overlap-width", type=int, default=100, help="Tile overlap width")
    parser.add_argument("--iou-threshold", type=float, default=0.2, help="Merge IoU threshold")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path. For directory input, must be output directory.",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively find images in directory input")
    parser.add_argument("--concept-name", type=str, default="visual_retrieval", help="Concept tag in filenames/npz metadata")
    parser.add_argument("--device", type=str, default="cuda", help="SAM3 device, e.g., cuda or cpu")
    parser.add_argument("--sam-resolution", type=int, default=1008, help="SAM3 processor square resize resolution")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Mask confidence threshold")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="SAM3 checkpoint path")
    parser.add_argument(
        "--save-sample-patch",
        action="store_true",
        help="Save the generated sample patch for inspection",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    input_path = Path(args.input_path).expanduser().absolute()
    sample_image_path = Path(args.sample_image).expanduser().absolute()
    if not sample_image_path.exists():
        raise FileNotFoundError(f"Sample image not found: {sample_image_path}")

    tile_size = (int(args.tile_height), int(args.tile_width))
    tile_overlap = (int(args.tile_overlap_height), int(args.tile_overlap_width))
    sample_box_xyxy = np.asarray(args.sample_box, dtype=np.float32).reshape(4)

    sample_patch, sample_prompt_box_norm, sample_box_in_patch = _prepare_sample_patch(
        sample_image_path,
        sample_box_xyxy,
        patch_h=tile_size[0],
        patch_w=tile_size[1],
    )

    if args.save_sample_patch:
        sample_patch_path = sample_image_path.parent / f"{sample_image_path.stem}_sample_patch.png"
        Image.fromarray(sample_patch).save(sample_patch_path)
        print(f"Saved sample patch: {sample_patch_path}")
        print(f"Sample box in patch (xyxy): {sample_box_in_patch.tolist()}")

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available, fallback to CPU.")
        device = "cpu"

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(
        checkpoint_path=args.checkpoint,
        device=device,
    )
    processor = Sam3Processor(
        model,
        resolution=int(args.sam_resolution),
        device=device,
        confidence_threshold=float(args.confidence_threshold),
    )

    run_visual_retrieval_inference_batch(
        input_path=input_path,
        processor=processor,
        sample_patch=sample_patch,
        sample_prompt_box_norm=sample_prompt_box_norm,
        sample_image_path=sample_image_path,
        sample_box_xyxy=sample_box_xyxy,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        iou_threshold=float(args.iou_threshold),
        alpha=float(args.alpha),
        output_path=args.output,
        recursive=bool(args.recursive),
        concept_name=args.concept_name,
    )
