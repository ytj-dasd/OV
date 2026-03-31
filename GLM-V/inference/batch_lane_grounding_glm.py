from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

from qwen.lane_grounding import draw_detections, parse_grounding_response

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


DEFAULT_MODEL_PATH = "ZhipuAI/GLM-4.6V-Flash"
DEFAULT_PROMPT = (
    'Locate every instance that belongs to the following categories: "curb", "white solid line", "white dashed line", "yellow solid line", "yellow dashed line", "double yellow line", "crosswalk", "arrow". '
    "Report bbox coordinates in JSON format. Output JSON array only, in this format: [{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"category\"}]."
)

SUPPORTED_IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch lane grounding inference with GLM-4.6V-Flash, including "
            "black-pixel-ratio filtering and visualization export."
        )
    )
    parser.add_argument("--input-dir", required=True, help="Folder containing input images")
    parser.add_argument("--output-dir", required=True, help="Folder to save inference results")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Model path used by transformers.from_pretrained",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Grounding prompt")
    parser.add_argument("--max-new-tokens", type=int, default=8192, help="Max generation tokens")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--top-p", type=float, default=0.6, help="Top-p for sampling")
    parser.add_argument("--top-k", type=int, default=2, help="Top-k for sampling")
    parser.add_argument(
        "--black-ratio-threshold",
        type=float,
        default=0.8,
        help="Skip inference if black pixel ratio is greater than this value (default: 0.8)",
    )
    parser.add_argument(
        "--black-pixel-threshold",
        type=int,
        default=10,
        help="Pixel is treated as black if R,G,B <= this threshold (default: 10)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan input-dir for images",
    )
    return parser.parse_args()


def collect_images(input_dir: Path, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    images = [
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    ]
    return sorted(images)


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
    black_count = sum(
        1
        for r, g, b in pixels
        if r <= black_threshold and g <= black_threshold and b <= black_threshold
    )
    return float(black_count / len(pixels))


def _result_json_path(image_path: Path, input_dir: Path, output_dir: Path) -> Path:
    rel = image_path.relative_to(input_dir)
    return (output_dir / "results" / rel).with_suffix(".json")


def _visualization_path(image_path: Path, input_dir: Path, output_dir: Path) -> Path:
    rel = image_path.relative_to(input_dir)
    return output_dir / "visualizations" / rel.with_name(f"{rel.stem}_lane_grounding{rel.suffix}")


def save_result(result_path: Path, payload: dict[str, Any]) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def extract_answer_text(raw: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw.strip()


def load_glm_model_and_processor(model_path: str) -> tuple[Any, Any]:
    from transformers import AutoProcessor, Glm4vForConditionalGeneration, Glm4vMoeForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_path)
    if "GLM-4.5V" in model_path:
        model = Glm4vMoeForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
        )
    else:
        model = Glm4vForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
        )
    return model, processor


def run_grounding_glm(
    model: Any,
    processor: Any,
    image_path: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    repetition_penalty: float,
    top_p: float,
    top_k: int,
) -> tuple[str, str]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    inputs.pop("token_type_ids", None)

    do_sample = temperature > 0
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "do_sample": do_sample,
        "top_k": top_k,
        "top_p": top_p,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature

    output_ids = model.generate(**inputs, **generation_kwargs)
    generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    raw_response = processor.decode(generated_ids, skip_special_tokens=False)
    answer_text = extract_answer_text(raw_response)
    return raw_response, answer_text


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found or invalid: {input_dir}")

    if not (0.0 <= args.black_ratio_threshold <= 1.0):
        raise ValueError("--black-ratio-threshold must be in [0, 1]")

    if not (0 <= args.black_pixel_threshold <= 255):
        raise ValueError("--black-pixel-threshold must be in [0, 255]")

    image_paths = collect_images(input_dir, recursive=args.recursive)
    if not image_paths:
        print(f"No supported images found in: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from: {args.model_path}")
    model, processor = load_glm_model_and_processor(args.model_path)

    summary: list[dict[str, Any]] = []
    processed = 0
    skipped = 0
    failed = 0

    for idx, image_path in enumerate(image_paths, start=1):
        print(f"[{idx}/{len(image_paths)}] {image_path}")

        result_path = _result_json_path(image_path, input_dir, output_dir)
        vis_path = _visualization_path(image_path, input_dir, output_dir)

        payload: dict[str, Any] = {
            "image_path": str(image_path),
            "status": "unknown",
        }

        try:
            image = Image.open(image_path).convert("RGB")
            black_ratio = calculate_black_ratio(image, args.black_pixel_threshold)
            payload["black_ratio"] = round(black_ratio, 6)

            if black_ratio > args.black_ratio_threshold:
                payload["status"] = "skipped"
                payload["reason"] = (
                    f"black_ratio {black_ratio:.6f} > threshold {args.black_ratio_threshold:.6f}"
                )
                skipped += 1
                save_result(result_path, payload)
                summary.append(
                    {
                        "image_path": str(image_path),
                        "status": payload["status"],
                        "black_ratio": payload["black_ratio"],
                        "result_json": str(result_path),
                    }
                )
                continue

            raw_response, answer_text = run_grounding_glm(
                model=model,
                processor=processor,
                image_path=str(image_path),
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            print(answer_text)

            detections = parse_grounding_response(answer_text)

            payload["status"] = "ok"
            payload["raw_response"] = raw_response
            payload["answer_text"] = answer_text
            payload["num_detections"] = len(detections)

            if detections:
                image_for_draw = image.copy()
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
                "image_path": str(image_path),
                "status": payload["status"],
                "black_ratio": payload.get("black_ratio"),
                "num_detections": payload.get("num_detections"),
                "result_json": str(result_path),
                "visualization_path": payload.get("visualization_path"),
            }
        )

    summary_payload = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "total_images": len(image_paths),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "black_ratio_threshold": args.black_ratio_threshold,
        "black_pixel_threshold": args.black_pixel_threshold,
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
