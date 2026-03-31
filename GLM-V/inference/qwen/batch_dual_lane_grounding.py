from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image

from lane_grounding import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PROMPT,
    draw_detections,
    parse_grounding_response,
)

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


SUPPORTED_IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

# VISION_ROLE_HINT = (
#     "Picture 1 is a point-cloud RGB projection image. "
#     "Picture 2 is a point-cloud reflectance intensity projection image. "
#     "All output bbox_2d coordinates must refer to Picture 1 only. "
# )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch lane-grounding inference with dual-image input. "
            "Each sample uses same-named images from two folders."
        )
    )
    parser.add_argument("--rgb-dir", required=True, help="Folder of point-cloud RGB projection images (Picture 1)")
    parser.add_argument(
        "--intensity-dir",
        required=True,
        help="Folder of point-cloud reflectance intensity projection images (Picture 2)",
    )
    parser.add_argument("--output-dir", required=True, help="Folder to save inference results")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Model path used by transformers.from_pretrained",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Grounding prompt body")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation max new tokens")
    parser.add_argument(
        "--black-ratio-threshold",
        type=float,
        default=0.8,
        help="Skip inference if black pixel ratio in RGB image > this threshold (default: 0.8)",
    )
    parser.add_argument(
        "--black-pixel-threshold",
        type=int,
        default=10,
        help="Pixel is treated as black if R,G,B <= this threshold (default: 10)",
    )
    parser.add_argument(
        "--iou-merge-threshold",
        type=float,
        default=0.5,
        help="Merge boxes when IoU > threshold (default: 0.5)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan image folders",
    )
    parser.add_argument(
        "--strict-pairing",
        action="store_true",
        help="Raise error when same-name pairing is incomplete between the two folders",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling generation (default: deterministic)",
    )
    return parser.parse_args()


def collect_relative_images(input_dir: Path, recursive: bool) -> dict[Path, Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    images: dict[Path, Path] = {}
    for path in iterator:
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        images[path.relative_to(input_dir)] = path
    return images


def calculate_black_ratio(image: Image.Image, black_threshold: int) -> float:
    rgb_image = image.convert("RGB")

    if np is not None:
        arr = np.asarray(rgb_image)
        if arr.size == 0:
            return 0.0
        black_mask = np.all(arr <= black_threshold, axis=-1)
        return float(black_mask.mean())

    pixels = list(rgb_image.getdata())
    if not pixels:
        return 0.0
    black_count = sum(1 for r, g, b in pixels if r <= black_threshold and g <= black_threshold and b <= black_threshold)
    return float(black_count / len(pixels))


def _normalize_bbox_1000(
    bbox: list[float] | tuple[float, float, float, float],
) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]

    x1 = min(max(x1, 0.0), 1000.0)
    y1 = min(max(y1, 0.0), 1000.0)
    x2 = min(max(x2, 0.0), 1000.0)
    y2 = min(max(y2, 0.0), 1000.0)

    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    return [x1, y1, x2, y2]


def _bbox_iou_1000(
    box_a: list[float] | tuple[float, float, float, float],
    box_b: list[float] | tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = _normalize_bbox_1000(box_a)
    bx1, by1, bx2, by2 = _normalize_bbox_1000(box_b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def merge_detections_by_iou(
    detections: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[list[dict[str, Any]], int]:
    normalized: list[dict[str, Any]] = []

    for i, det in enumerate(detections, start=1):
        bbox = det.get("bbox_2d")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            bbox_norm = _normalize_bbox_1000(bbox)
        except Exception:
            continue

        label = str(det.get("label", f"object_{i}"))
        normalized.append({"bbox_2d": bbox_norm, "label": label})

    n = len(normalized)
    if n <= 1:
        return normalized, 0

    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return
        if rank[root_a] < rank[root_b]:
            parent[root_a] = root_b
        elif rank[root_a] > rank[root_b]:
            parent[root_b] = root_a
        else:
            parent[root_b] = root_a
            rank[root_a] += 1

    for i in range(n):
        for j in range(i + 1, n):
            if _bbox_iou_1000(normalized[i]["bbox_2d"], normalized[j]["bbox_2d"]) > iou_threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for idx in range(n):
        root = find(idx)
        groups.setdefault(root, []).append(idx)

    merged: list[dict[str, Any]] = []
    for _, indices in sorted(groups.items(), key=lambda item: min(item[1])):
        boxes = [normalized[idx]["bbox_2d"] for idx in indices]
        labels = [normalized[idx]["label"] for idx in indices]

        x1 = min(box[0] for box in boxes)
        y1 = min(box[1] for box in boxes)
        x2 = max(box[2] for box in boxes)
        y2 = max(box[3] for box in boxes)

        label_count: dict[str, int] = {}
        first_pos: dict[str, int] = {}
        for pos, label in enumerate(labels):
            label_count[label] = label_count.get(label, 0) + 1
            if label not in first_pos:
                first_pos[label] = pos

        best_label = max(
            label_count,
            key=lambda name: (label_count[name], -first_pos[name]),
        )

        merged.append(
            {
                "bbox_2d": [x1, y1, x2, y2],
                "label": best_label,
                "merged_from": len(indices),
            }
        )

    num_merged = n - len(merged)
    return merged, num_merged


def run_grounding_dual(
    model: Any,
    processor: Any,
    rgb_image_path: str,
    intensity_image_path: str,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
) -> str:
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": rgb_image_path, "resized_height": 1024, "resized_width": 1024},
                {"type": "image", "image": intensity_image_path, "resized_height": 1024, "resized_width": 1024},
                {"type": "text", "text": f"{prompt}"},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        add_vision_id=True,
    )
    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
        do_resize=False,
        **video_kwargs,
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )
    generated_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]

    response = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return response[0] if response else ""


def _result_json_path(rel_key: Path, output_dir: Path) -> Path:
    return (output_dir / "results" / rel_key).with_suffix(".json")


def _visualization_path(rel_key: Path, output_dir: Path) -> Path:
    return output_dir / "visualizations" / rel_key.with_name(f"{rel_key.stem}_lane_grounding{rel_key.suffix}")


def save_result(result_path: Path, payload: dict[str, Any]) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    rgb_dir = Path(args.rgb_dir).expanduser().resolve()
    intensity_dir = Path(args.intensity_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not rgb_dir.exists() or not rgb_dir.is_dir():
        raise NotADirectoryError(f"RGB directory not found or invalid: {rgb_dir}")
    if not intensity_dir.exists() or not intensity_dir.is_dir():
        raise NotADirectoryError(f"Intensity directory not found or invalid: {intensity_dir}")

    if not (0.0 <= args.black_ratio_threshold <= 1.0):
        raise ValueError("--black-ratio-threshold must be in [0, 1]")

    if not (0 <= args.black_pixel_threshold <= 255):
        raise ValueError("--black-pixel-threshold must be in [0, 255]")

    if not (0.0 <= args.iou_merge_threshold <= 1.0):
        raise ValueError("--iou-merge-threshold must be in [0, 1]")

    rgb_images = collect_relative_images(rgb_dir, recursive=args.recursive)
    intensity_images = collect_relative_images(intensity_dir, recursive=args.recursive)

    rgb_keys = set(rgb_images.keys())
    intensity_keys = set(intensity_images.keys())
    common_keys = sorted(rgb_keys & intensity_keys)
    only_rgb = sorted(rgb_keys - intensity_keys)
    only_intensity = sorted(intensity_keys - rgb_keys)

    if args.strict_pairing and (only_rgb or only_intensity):
        raise ValueError(
            "Pairing mismatch under --strict-pairing. "
            f"missing_in_intensity={len(only_rgb)}, missing_in_rgb={len(only_intensity)}"
        )

    if not common_keys:
        print("No paired images found between --rgb-dir and --intensity-dir")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"Loading model from: {args.model_path}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    summary: list[dict[str, Any]] = []
    processed = 0
    skipped = 0
    failed = 0

    for idx, rel_key in enumerate(common_keys, start=1):
        rgb_path = rgb_images[rel_key]
        intensity_path = intensity_images[rel_key]

        print(f"[{idx}/{len(common_keys)}] {rel_key}")

        result_path = _result_json_path(rel_key, output_dir)
        vis_path = _visualization_path(rel_key, output_dir)

        payload: dict[str, Any] = {
            "relative_name": str(rel_key),
            "rgb_image_path": str(rgb_path),
            "intensity_image_path": str(intensity_path),
            "status": "unknown",
            "add_vision_id": True,
            "model_output_mode": "single_text_for_two_images",
        }

        try:
            rgb_image = Image.open(rgb_path).convert("RGB")
            black_ratio_rgb = calculate_black_ratio(rgb_image, args.black_pixel_threshold)

            payload["black_ratio_rgb"] = round(black_ratio_rgb, 6)

            if black_ratio_rgb > args.black_ratio_threshold:
                payload["status"] = "skipped"
                payload["reason"] = (
                    "black_ratio_rgb exceeded threshold: "
                    f"rgb={black_ratio_rgb:.6f}, threshold={args.black_ratio_threshold:.6f}"
                )
                skipped += 1
                save_result(result_path, payload)
                summary.append(
                    {
                        "relative_name": str(rel_key),
                        "status": payload["status"],
                        "black_ratio_rgb": payload["black_ratio_rgb"],
                        "result_json": str(result_path),
                    }
                )
                continue

            raw_response = run_grounding_dual(
                model=model,
                processor=processor,
                rgb_image_path=str(rgb_path),
                intensity_image_path=str(intensity_path),
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
            )
            print(raw_response)

            raw_detections = parse_grounding_response(raw_response)
            detections, num_merged_by_iou = merge_detections_by_iou(
                raw_detections,
                iou_threshold=args.iou_merge_threshold,
            )

            payload["status"] = "ok"
            payload["raw_response"] = raw_response
            payload["num_detections_raw"] = len(raw_detections)
            payload["num_detections"] = len(detections)
            payload["num_merged_by_iou"] = num_merged_by_iou
            payload["iou_merge_threshold"] = args.iou_merge_threshold

            if raw_detections and num_merged_by_iou > 0:
                print(
                    f"[IoU Merge] raw={len(raw_detections)}, merged={len(detections)}, "
                    f"threshold={args.iou_merge_threshold:.3f}"
                )

            if detections:
                image_for_draw = rgb_image.copy()
                rendered = draw_detections(image_for_draw, detections)
                vis_path.parent.mkdir(parents=True, exist_ok=True)
                image_for_draw.save(vis_path)

                print("\n[Detections]")
                for item in rendered:
                    print(
                        f"label={item['label']}, bbox_2d={item['bbox_2d']}, bbox_px={item['bbox_px']}"
                    )

                payload["detections"] = rendered
                payload["visualization_path"] = str(vis_path)
            else:
                payload["detections"] = []

            processed += 1

        except Exception as exc:  # noqa: BLE001
            payload["status"] = "error"
            payload["error"] = f"{type(exc).__name__}: {exc}"
            failed += 1

        save_result(result_path, payload)
        summary.append(
            {
                "relative_name": str(rel_key),
                "status": payload["status"],
                "black_ratio_rgb": payload.get("black_ratio_rgb"),
                "num_detections_raw": payload.get("num_detections_raw"),
                "num_detections": payload.get("num_detections"),
                "num_merged_by_iou": payload.get("num_merged_by_iou"),
                "result_json": str(result_path),
                "visualization_path": payload.get("visualization_path"),
            }
        )

    summary_payload = {
        "rgb_dir": str(rgb_dir),
        "intensity_dir": str(intensity_dir),
        "output_dir": str(output_dir),
        "total_rgb_images": len(rgb_images),
        "total_intensity_images": len(intensity_images),
        "paired_images": len(common_keys),
        "missing_in_intensity": [str(p) for p in only_rgb],
        "missing_in_rgb": [str(p) for p in only_intensity],
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "black_ratio_threshold": args.black_ratio_threshold,
        "black_pixel_threshold": args.black_pixel_threshold,
        "iou_merge_threshold": args.iou_merge_threshold,
        "results": summary,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Done.")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
