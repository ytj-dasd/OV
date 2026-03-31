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
DEFAULT_DUAL_PROMPT = (
    'Picture 1 is an RGB projection image. Picture 2 is a reflectance intensity projection image. '
    'Use both images jointly for detection.'
    'Locate every instance that belongs to the following categories: "white solid line", "white dashed line", "yellow solid line", "yellow dashed line", "double yellow line", "crosswalk", "arrow". '
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
            "Batch dual-image lane grounding with GLM-4.6V-Flash. "
            "Each sample uses same-name RGB and intensity images."
        )
    )
    parser.add_argument("--rgb-dir", required=True, help="Folder of RGB images (Picture 1)")
    parser.add_argument(
        "--intensity-dir",
        required=True,
        help="Folder of intensity images (Picture 2)",
    )
    parser.add_argument("--output-dir", required=True, help="Folder to save inference results")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Model path used by transformers.from_pretrained",
    )
    parser.add_argument("--prompt", default=DEFAULT_DUAL_PROMPT, help="Grounding prompt")
    parser.add_argument("--max-new-tokens", type=int, default=8192, help="Max generation tokens")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--top-p", type=float, default=0.6, help="Top-p for sampling")
    parser.add_argument("--top-k", type=int, default=2, help="Top-k for sampling")
    parser.add_argument(
        "--black-ratio-threshold",
        type=float,
        default=0.8,
        help="Skip inference if black pixel ratio in RGB image > this value (default: 0.8)",
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
        help="Recursively scan image folders",
    )
    parser.add_argument(
        "--strict-pairing",
        action="store_true",
        help="Raise error when same-name pairing is incomplete between folders",
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
    black_count = sum(
        1
        for r, g, b in pixels
        if r <= black_threshold and g <= black_threshold and b <= black_threshold
    )
    return float(black_count / len(pixels))


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


def run_grounding_dual_glm(
    model: Any,
    processor: Any,
    rgb_image_path: str,
    intensity_image_path: str,
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
                {"type": "image", "url": rgb_image_path},
                {"type": "image", "url": intensity_image_path},
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

    print(f"Loading model from: {args.model_path}")
    model, processor = load_glm_model_and_processor(args.model_path)

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

            raw_response, answer_text = run_grounding_dual_glm(
                model=model,
                processor=processor,
                rgb_image_path=str(rgb_path),
                intensity_image_path=str(intensity_path),
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
                "num_detections": payload.get("num_detections"),
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
