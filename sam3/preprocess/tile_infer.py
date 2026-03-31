import argparse
import colorsys
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3.model_builder import build_sam3_image_model
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
    """
    Merge instance masks when bbox IoU is above threshold.
    Returns masks in shape (N, 1, H, W), plus merged boxes/scores.
    """
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
    # Deterministic HSV wheel by instance id; supports many instances with
    # stable and distinguishable colors.
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
    )
    print(f"Merged masks/boxes/scores saved to: {out_npz_path}")


def _to_numpy_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def get_masks(
    img_path: Path | str,
    text_prompt: str,
    *,
    sam3_processor: Sam3Processor,
) -> dict[str, Any]:
    """Run SAM3 text-prompt segmentation in-process using provided processor."""
    img_path = Path(img_path).expanduser().absolute()
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    image = Image.open(img_path).convert("RGB")
    inference_state = sam3_processor.set_image(image)
    output = sam3_processor.set_text_prompt(state=inference_state, prompt=text_prompt)

    if "masks" not in output:
        raise KeyError("'masks' not found in SAM3 output")

    out: dict[str, Any] = {"masks": _to_numpy_array(output["masks"])}
    if "scores" in output:
        out["scores"] = _to_numpy_array(output["scores"])
    if "boxes" in output:
        out["boxes"] = _to_numpy_array(output["boxes"])
    return out


def _get_concept_masks(
    img_path: Path,
    img: np.ndarray,
    *,
    concept: str = "arrow",
    tile_size: int | tuple[int, int] = (2048, 2048),
    tile_overlap: int | tuple[int, int] = 200,
    enable_nms: bool = True,
    iou_threshold: float = 0.2,
    overlap_cover_ratio: float = 0.9,
    sam3_processor: Sam3Processor,
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

    if h > tile_h or w > tile_w:
        stride_h = max(1, tile_h - overlap_h)
        stride_w = max(1, tile_w - overlap_w)
        ys = _tile_starts(h, tile_h, stride_h)
        xs = _tile_starts(w, tile_w, stride_w)
        tiles_dir = img_path.parent / f"{img_path.stem}_tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        global_masks: list[np.ndarray] = []
        global_boxes: list[np.ndarray] = []
        global_scores: list[float] = []
        tile_regions = [
            (y0, x0, min(y0 + tile_h, h), min(x0 + tile_w, w))
            for y0 in ys
            for x0 in xs
        ]

        for y0, x0, y1, x1 in tqdm(
            tile_regions,
            desc=f"Tile inference [{concept}]",
            unit="tile",
        ):
            tile_img = img[y0:y1, x0:x1]
            if tile_img.ndim >= 3:
                black_pixels = np.all(tile_img == 0, axis=-1)
            else:
                black_pixels = tile_img == 0
            black_ratio = float(np.count_nonzero(black_pixels)) / float(black_pixels.size)
            if black_ratio > MOSTLY_BLACK_TILE_SKIP_RATIO:
                continue

            tile_path = tiles_dir / f"{img_path.stem}_y{y0}_x{x0}.png"
            Image.fromarray(tile_img).save(tile_path)

            tile_out = get_masks(
                tile_path,
                text_prompt=concept,
                sam3_processor=sam3_processor,
            )
            tile_masks = _normalize_mask_array(tile_out["masks"])
            n_masks = tile_masks.shape[0]
            if n_masks == 0:
                continue

            tile_boxes = tile_out.get("boxes")
            if tile_boxes is None:
                boxes_xyxy = np.stack([_bbox_from_mask(m) for m in tile_masks], axis=0)
            else:
                boxes_xyxy = np.asarray(tile_boxes, dtype=np.float32)
                if boxes_xyxy.ndim == 1 and boxes_xyxy.size == 4:
                    boxes_xyxy = boxes_xyxy.reshape(1, 4)
                if boxes_xyxy.shape != (n_masks, 4):
                    raise ValueError(
                        f"Tile boxes shape mismatch: expected ({n_masks},4), got {boxes_xyxy.shape}"
                    )

            scores = tile_out.get("scores")
            if scores is None:
                scores_1d = np.ones((n_masks,), dtype=np.float32)
            else:
                scores_1d = np.asarray(scores, dtype=np.float32).reshape(-1)
                if scores_1d.shape[0] != n_masks:
                    raise ValueError(
                        f"Tile scores length mismatch: expected {n_masks}, got {scores_1d.shape[0]}"
                    )

            boxes_xyxy = boxes_xyxy.astype(np.float32, copy=True)
            boxes_xyxy[:, [0, 2]] += float(x0)
            boxes_xyxy[:, [1, 3]] += float(y0)

            canvas_masks = np.zeros((n_masks, h, w), dtype=bool)
            canvas_masks[:, y0:y1, x0:x1] = tile_masks[:, : y1 - y0, : x1 - x0]

            global_masks.extend([canvas_masks[i] for i in range(n_masks)])
            global_boxes.extend([boxes_xyxy[i] for i in range(n_masks)])
            global_scores.extend([float(s) for s in scores_1d])

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

    # Keep output format aligned with tiled branch.
    single = get_masks(
        img_path,
        text_prompt=concept,
        sam3_processor=sam3_processor,
    )
    single_masks = _normalize_mask_array(single["masks"])
    return merge_instances_by_iou(
        masks=single_masks,
        boxes=single.get("boxes"),
        scores=single.get("scores"),
        iou_threshold=iou_threshold,
    )

def run_concept_inference(
    img_path: Path | str,
    concept: str,
    *,
    tile_size: int | tuple[int, int] = (2048, 2048),
    tile_overlap: int | tuple[int, int] = 200,
    iou_threshold: float = 0.2,
    alpha: float = 0.45,
    output_path: Path | str | None = None,
    sam3_processor: Sam3Processor,
) -> dict[str, np.ndarray]:
    """Run tiled concept inference and save colorized merged instance visualization."""
    img_path = Path(img_path).expanduser().absolute()
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    image = np.array(Image.open(img_path).convert("RGB"))
    merged = _get_concept_masks(
        img_path=img_path,
        img=image,
        concept=concept,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        iou_threshold=iou_threshold,
        sam3_processor=sam3_processor,
    )

    vis = draw_instances_on_image(image, merged["masks"], alpha=alpha)
    if output_path is None:
        safe_concept = concept.replace(" ", "_")
        out_path = img_path.parent / f"{img_path.stem}_{safe_concept}_merged.png"
    else:
        out_path = Path(output_path).expanduser().absolute()
    Image.fromarray(vis).save(out_path)
    out_npz_path = out_path.with_suffix(".npz")
    _save_merged_npz(
        merged,
        out_npz_path,
        img_path=img_path,
        concept=concept,
        image_shape=(int(image.shape[0]), int(image.shape[1])),
    )
    print(f"Merged visualization saved to: {out_path}")
    print(f"Merged instances: {merged['masks'].shape[0]}")
    return merged


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


def run_concept_inference_batch(
    input_path: Path | str,
    concept: str,
    *,
    tile_size: int | tuple[int, int] = (2048, 2048),
    tile_overlap: int | tuple[int, int] = 200,
    iou_threshold: float = 0.2,
    alpha: float = 0.45,
    output_path: Path | str | None = None,
    recursive: bool = False,
    sam3_processor: Sam3Processor,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Run concept inference on a single image or all images in a directory.
    - Single file: output_path behaves like run_concept_inference(output file path).
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
        iterator = tqdm(image_paths, desc=f"Image inference [{concept}]", unit="img")

    safe_concept = concept.replace(" ", "_")
    for img_path in iterator:
        per_image_output: Path | str | None = output_path
        if not is_single_image:
            if output_dir is None:
                per_image_output = None
            else:
                per_image_output = output_dir / f"{img_path.stem}_{safe_concept}_merged.png"

        merged = run_concept_inference(
            img_path=img_path,
            concept=concept,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            iou_threshold=iou_threshold,
            alpha=alpha,
            output_path=per_image_output,
            sam3_processor=sam3_processor,
        )
        results[str(img_path)] = merged
    return results


def _safe_prompt_name(prompt: str) -> str:
    out = prompt.strip().replace(" ", "_")
    out = "".join(ch if (ch.isalnum() or ch in {"_", "-", "."}) else "_" for ch in out)
    out = out.strip("._")
    return out or "prompt"


def run_multi_prompt_inference(
    input_path: Path | str,
    prompts: list[str],
    *,
    tile_size: int | tuple[int, int] = (2048, 2048),
    tile_overlap: int | tuple[int, int] = 200,
    iou_threshold: float = 0.2,
    alpha: float = 0.45,
    output_root: Path | str | None = None,
    recursive: bool = False,
    sam3_processor: Sam3Processor,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Run inference for multiple prompts; each prompt writes to its own folder."""
    if not prompts:
        raise ValueError("prompts must not be empty")

    input_path = Path(input_path).expanduser().absolute()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if output_root is None:
        base_dir = input_path.parent if input_path.is_file() else input_path
        output_root_path = (base_dir / "prompt_results").absolute()
    else:
        output_root_path = Path(output_root).expanduser().absolute()
    output_root_path.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for prompt in tqdm(prompts, desc="Prompt inference", unit="prompt"):
        safe_prompt = _safe_prompt_name(prompt)
        prompt_dir = output_root_path / safe_prompt
        prompt_dir.mkdir(parents=True, exist_ok=True)

        if input_path.is_file():
            out_png = prompt_dir / f"{input_path.stem}_{safe_prompt}_merged.png"
            merged = run_concept_inference(
                img_path=input_path,
                concept=prompt,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                iou_threshold=iou_threshold,
                alpha=alpha,
                output_path=out_png,
                sam3_processor=sam3_processor,
            )
            all_results[prompt] = {str(input_path): merged}
        else:
            merged_map = run_concept_inference_batch(
                input_path=input_path,
                concept=prompt,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                iou_threshold=iou_threshold,
                alpha=alpha,
                output_path=prompt_dir,
                recursive=recursive,
                sam3_processor=sam3_processor,
            )
            all_results[prompt] = merged_map
    return all_results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiled concept inference with mask merging.")
    parser.add_argument("input_path", type=str, help="Input image path or image directory")
    parser.add_argument("prompts", nargs="+", type=str, help="One or more concept text prompts")
    parser.add_argument("--tile-size", type=int, default=2048, help="Tile size")
    parser.add_argument("--tile-overlap", type=int, default=200, help="Tile overlap")
    parser.add_argument("--iou-threshold", type=float, default=0.2, help="Merge IoU threshold")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output root directory. Results are saved into per-prompt subfolders.",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively find images in directory input")
    parser.add_argument("--device", type=str, default="cuda", help="SAM3 device, e.g., cuda or cpu")
    parser.add_argument("--sam-resolution", type=int, default=1008, help="SAM3 processor square resize resolution")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Mask confidence threshold")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="SAM3 checkpoint path")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available, fallback to CPU.")
        device = "cpu"

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

    run_multi_prompt_inference(
        input_path=args.input_path,
        prompts=args.prompts,
        tile_size=(args.tile_size, args.tile_size),
        tile_overlap=(args.tile_overlap, args.tile_overlap),
        iou_threshold=args.iou_threshold,
        alpha=args.alpha,
        output_root=args.output,
        recursive=args.recursive,
        sam3_processor=processor,
    )
