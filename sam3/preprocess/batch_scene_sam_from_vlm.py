from __future__ import annotations

import argparse
import json
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
from preprocess.tile_infer import draw_instances_on_image, get_masks


SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_CHECKPOINT = "/home/guitu/文档/vector/sam3/model/sam3.pt"

CLASS_TO_ID = {
    "电线杆": 1,
    "路灯杆": 2,
    "路牌": 3,
    "交通标志": 4,
    "红绿灯": 5,
    "监控": 6,
    "行道树": 7,
    "果壳箱": 8,
    "消防栓": 9,
    "电箱": 10,
    "雕塑": 11,
    "座椅": 12,
    "交通锥": 13,
    "柱墩": 14,
    "围栏": 15,
    "杆状物": 1,
    "标识标牌": 3,
    "箱体": 10,
    "树木": 7,
}

CLASS_TO_EN_PROMPT = {
    "电线杆": "utility pole",
    "路灯杆": "street light",
    "路牌": "signboard",
    "交通标志": "traffic sign",
    "红绿灯": "signal light",
    "监控": "surveillance camera",
    "行道树": "tree",
    "果壳箱": "trash bin",
    "消防栓": "fire hydrant",
    "电箱": "utility box",
    "雕塑": "sculpture",
    "座椅": "bench",
    "交通锥": "traffic cone",
    "柱墩": "bollard",
    "围栏": "fence",
    "杆状物": "pole",
    "标识标牌": "sign",
    "箱体": "box",
    "树木": "tree",
}

EN_TO_CLASS = {v.lower(): k for k, v in CLASS_TO_EN_PROMPT.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Task4: batch SAM inference by reading benchmark scenes and each image's "
            "vlm_desc/*.vlm.txt class list."
        )
    )
    parser.add_argument("--data-root", required=True, help="benchmark root")
    parser.add_argument("--alpha", type=float, default=0.45, help="overlay alpha")
    parser.add_argument("--device", type=str, default="cuda", help="SAM3 device")
    parser.add_argument("--sam-resolution", type=int, default=1008, help="SAM3 resolution")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="SAM3 confidence threshold (default=0 keeps all masks)",
    )
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="SAM3 checkpoint path")
    parser.add_argument(
        "--save-per-class-artifacts",
        action="store_true",
        help="Also save per-class npz and duplicate merged png under sam_mask/per_class",
    )
    return parser.parse_args()


def _safe_name_token(text: str) -> str:
    return text.replace("/", "_").replace(" ", "_")


def _normalize_masks_n1hw(masks: Any) -> np.ndarray:
    arr = np.asarray(masks)
    if arr.ndim == 3:
        arr = arr[:, None, :, :]
    elif arr.ndim == 4 and arr.shape[1] == 1:
        pass
    else:
        raise ValueError(f"Unexpected masks shape: {arr.shape}")
    return arr.astype(bool, copy=False)


def _bbox_from_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return np.zeros((4,), dtype=np.float32)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def _ensure_boxes_n4(masks_n1hw: np.ndarray, boxes: Any | None) -> np.ndarray:
    n = int(masks_n1hw.shape[0])
    if n == 0:
        return np.zeros((0, 4), dtype=np.float32)

    if boxes is None:
        return np.stack([_bbox_from_mask(masks_n1hw[i, 0]) for i in range(n)], axis=0)

    arr = np.asarray(boxes, dtype=np.float32)
    if arr.ndim == 1 and arr.size == 4:
        arr = arr.reshape(1, 4)
    if arr.shape != (n, 4):
        return np.stack([_bbox_from_mask(masks_n1hw[i, 0]) for i in range(n)], axis=0)
    return arr


def _ensure_scores_n(masks_n1hw: np.ndarray, scores: Any | None) -> np.ndarray:
    n = int(masks_n1hw.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    if scores is None:
        return np.ones((n,), dtype=np.float32)
    arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    if arr.shape[0] != n:
        return np.ones((n,), dtype=np.float32)
    return arr


def collect_scene_dirs(data_root: Path) -> list[Path]:
    scene_dirs: list[Path] = []
    for child in sorted(data_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "projected_images").is_dir():
            scene_dirs.append(child)
    return scene_dirs


def collect_images(projected_images_dir: Path) -> list[Path]:
    return [
        p
        for p in sorted(projected_images_dir.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
    ]


def _parse_float_or_none(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def parse_vlm_txt(vlm_txt_path: Path) -> list[tuple[str, str, float]]:
    if not vlm_txt_path.exists():
        return []
    raw = vlm_txt_path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    class_info: dict[str, tuple[str, float]] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = [p.strip() for p in line.split("\t") if p.strip()]
        if not parts:
            continue

        class_token = parts[0]
        prompt_en = ""
        score = 1.0

        if len(parts) >= 3:
            prompt_en = parts[1]
            parsed_score = _parse_float_or_none(parts[-1])
            score = parsed_score if parsed_score is not None else 1.0
        elif len(parts) == 2:
            parsed_score = _parse_float_or_none(parts[1])
            if parsed_score is not None:
                score = parsed_score
            else:
                prompt_en = parts[1]

        class_name = class_token if class_token in CLASS_TO_ID else EN_TO_CLASS.get(class_token.lower())
        if class_name is None:
            class_name = EN_TO_CLASS.get(prompt_en.lower()) if prompt_en else None
        if class_name is None:
            continue

        if not prompt_en:
            prompt_en = CLASS_TO_EN_PROMPT.get(class_name, class_name)

        prev = class_info.get(class_name)
        if prev is None or score > prev[1]:
            class_info[class_name] = (prompt_en, score)

    items = [(class_name, en_prompt, score) for class_name, (en_prompt, score) in class_info.items()]
    return sorted(items, key=lambda x: x[0])


def save_empty_npz(
    npz_path: Path,
    image_path: Path,
    image_shape: tuple[int, int],
) -> None:
    h, w = image_shape
    np.savez_compressed(
        npz_path,
        masks=np.zeros((0, 1, h, w), dtype=bool),
        boxes=np.zeros((0, 4), dtype=np.float32),
        scores=np.zeros((0,), dtype=np.float32),
        class_names=np.array([], dtype="<U1"),
        class_prompts_en=np.array([], dtype="<U1"),
        class_ids=np.zeros((0,), dtype=np.int32),
        class_confidences=np.zeros((0,), dtype=np.float32),
        source_image=np.array(str(image_path)),
        image_stem=np.array(image_path.stem),
        image_height=np.array(h, dtype=np.int32),
        image_width=np.array(w, dtype=np.int32),
    )


def save_aggregated_npz(
    npz_path: Path,
    image_path: Path,
    image_shape: tuple[int, int],
    masks: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_names: list[str],
    class_prompts_en: list[str],
    class_ids: list[int],
    class_confidences: list[float],
) -> None:
    h, w = image_shape
    np.savez_compressed(
        npz_path,
        masks=masks,
        boxes=boxes.astype(np.float32),
        scores=scores.astype(np.float32),
        class_names=np.asarray(class_names, dtype="<U64"),
        class_prompts_en=np.asarray(class_prompts_en, dtype="<U64"),
        class_ids=np.asarray(class_ids, dtype=np.int32),
        class_confidences=np.asarray(class_confidences, dtype=np.float32),
        source_image=np.array(str(image_path)),
        image_stem=np.array(image_path.stem),
        image_height=np.array(h, dtype=np.int32),
        image_width=np.array(w, dtype=np.int32),
    )


def run_scene_batch(
    scene_dir: Path,
    sam3_processor: Sam3Processor,
    *,
    alpha: float,
    save_per_class_artifacts: bool,
) -> dict[str, Any]:
    projected_images_dir = scene_dir / "projected_images"
    vlm_desc_dir = scene_dir / "vlm_desc"
    sam_mask_dir = scene_dir / "sam_mask"
    sam_mask_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(projected_images_dir)
    results: list[dict[str, Any]] = []

    for image_path in tqdm(image_paths, desc=f"SAM Task4 [{scene_dir.name}]", unit="img"):
        image_np = np.array(Image.open(image_path).convert("RGB"))
        h, w = int(image_np.shape[0]), int(image_np.shape[1])
        vlm_txt_path = vlm_desc_dir / f"{image_path.stem}.vlm.txt"
        scene_row: dict[str, Any] = {
            "image": image_path.name,
            "vlm_txt": str(vlm_txt_path),
        }

        classes = parse_vlm_txt(vlm_txt_path)
        unknown_vlm_classes: list[str] = []
        valid_classes: list[tuple[str, str, float]] = []
        for class_name, class_prompt_en, class_conf in classes:
            if class_name not in CLASS_TO_ID:
                unknown_vlm_classes.append(f"{class_name}({class_prompt_en})")
                continue
            valid_classes.append((class_name, class_prompt_en, class_conf))

        if valid_classes:
            class_pairs = [f"{class_name}({class_prompt_en})" for class_name, class_prompt_en, _ in valid_classes]
            print(f"{image_path.name}: {', '.join(class_pairs)}")
        else:
            print(f"{image_path.name}: <no valid classes>")

        if not classes:
            save_empty_npz(
                sam_mask_dir / f"{image_path.stem}.npz",
                image_path,
                (h, w),
            )
            scene_row["status"] = "empty_or_missing_vlm"
            scene_row["num_classes"] = 0
            scene_row["num_masks"] = 0
            scene_row["sam_npz"] = str(sam_mask_dir / f"{image_path.stem}.npz")
            scene_row["unknown_vlm_classes"] = []
            results.append(scene_row)
            continue

        all_masks: list[np.ndarray] = []
        all_boxes: list[np.ndarray] = []
        all_scores: list[float] = []
        all_class_names: list[str] = []
        all_class_prompts_en: list[str] = []
        all_class_ids: list[int] = []
        all_class_confidences: list[float] = []
        per_class_stats: list[dict[str, Any]] = []

        per_class_dir = sam_mask_dir / "per_class" / image_path.stem
        if save_per_class_artifacts:
            per_class_dir.mkdir(parents=True, exist_ok=True)

        for class_name, class_prompt_en, class_conf in valid_classes:
            raw_out = get_masks(
                image_path,
                text_prompt=class_prompt_en,
                sam3_processor=sam3_processor,
            )
            masks = _normalize_masks_n1hw(raw_out["masks"])
            boxes = _ensure_boxes_n4(masks, raw_out.get("boxes"))
            scores = _ensure_scores_n(masks, raw_out.get("scores"))
            n = int(scores.shape[0])

            per_class_stats.append(
                {
                    "class_name": class_name,
                    "class_prompt_en": class_prompt_en,
                    "class_confidence": float(class_conf),
                    "num_masks": n,
                }
            )
            if n == 0:
                continue

            safe_name = _safe_name_token(class_name)
            per_class_png_root = sam_mask_dir / f"{image_path.stem}_{safe_name}_merged.png"
            vis = draw_instances_on_image(image_np, masks, alpha=alpha)
            Image.fromarray(vis).save(per_class_png_root)
            per_class_stats[-1]["sam_vis"] = str(per_class_png_root)

            all_masks.append(masks)
            all_boxes.append(boxes)
            all_scores.extend([float(x) for x in scores.tolist()])
            all_class_names.extend([class_name] * n)
            all_class_prompts_en.extend([class_prompt_en] * n)
            all_class_ids.extend([int(CLASS_TO_ID.get(class_name, 0))] * n)
            all_class_confidences.extend([float(class_conf)] * n)

            if save_per_class_artifacts:
                per_class_png = per_class_dir / f"{safe_name}_merged.png"
                per_class_npz = per_class_dir / f"{safe_name}.npz"
                Image.fromarray(vis).save(per_class_png)
                np.savez_compressed(
                    per_class_npz,
                    masks=masks,
                    boxes=boxes,
                    scores=scores,
                    class_name=np.array(class_name),
                    class_prompt_en=np.array(class_prompt_en),
                    class_id=np.array(int(CLASS_TO_ID.get(class_name, 0)), dtype=np.int32),
                    class_confidence=np.array(float(class_conf), dtype=np.float32),
                    source_image=np.array(str(image_path)),
                    image_stem=np.array(image_path.stem),
                    image_height=np.array(h, dtype=np.int32),
                    image_width=np.array(w, dtype=np.int32),
                )

        if all_masks:
            agg_masks = np.concatenate(all_masks, axis=0)
            agg_boxes = np.concatenate(all_boxes, axis=0)
            agg_scores = np.asarray(all_scores, dtype=np.float32)
            agg_npz_path = sam_mask_dir / f"{image_path.stem}.npz"
            save_aggregated_npz(
                agg_npz_path,
                image_path,
                (h, w),
                agg_masks,
                agg_boxes,
                agg_scores,
                all_class_names,
                all_class_prompts_en,
                all_class_ids,
                all_class_confidences,
            )

            scene_row["status"] = "ok"
            scene_row["num_classes"] = len(valid_classes)
            scene_row["num_masks"] = int(agg_scores.shape[0])
            scene_row["sam_npz"] = str(agg_npz_path)
            scene_row["per_class"] = per_class_stats
            scene_row["unknown_vlm_classes"] = unknown_vlm_classes
        else:
            agg_npz_path = sam_mask_dir / f"{image_path.stem}.npz"
            save_empty_npz(
                agg_npz_path,
                image_path,
                (h, w),
            )

            scene_row["status"] = "ok_but_no_masks"
            scene_row["num_classes"] = len(valid_classes)
            scene_row["num_masks"] = 0
            scene_row["sam_npz"] = str(agg_npz_path)
            scene_row["per_class"] = per_class_stats
            scene_row["unknown_vlm_classes"] = unknown_vlm_classes

        results.append(scene_row)

    summary = {
        "scene": scene_dir.name,
        "projected_images_dir": str(projected_images_dir),
        "vlm_desc_dir": str(vlm_desc_dir),
        "sam_mask_dir": str(sam_mask_dir),
        "total_images": len(image_paths),
        "results": results,
    }
    summary_path = sam_mask_dir / "scene_sam_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.exists() or not data_root.is_dir():
        raise NotADirectoryError(f"data-root not found or invalid: {data_root}")

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available, fallback to CPU.")
        device = "cpu"

    model = build_sam3_image_model(
        checkpoint_path=str(Path(args.checkpoint).expanduser().absolute()),
        device=device,
    )
    processor = Sam3Processor(
        model,
        resolution=int(args.sam_resolution),
        device=device,
        confidence_threshold=float(args.confidence_threshold),
    )

    scene_dirs = collect_scene_dirs(data_root)
    if not scene_dirs:
        raise FileNotFoundError(f"No scene with projected_images found under: {data_root}")

    for scene_dir in scene_dirs:
        run_scene_batch(
            scene_dir=scene_dir,
            sam3_processor=processor,
            alpha=float(args.alpha),
            save_per_class_artifacts=bool(args.save_per_class_artifacts),
        )
        print(f"scene done: {scene_dir.name}")
    print("all done.")


if __name__ == "__main__":
    main()
