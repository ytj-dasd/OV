from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


DEFAULT_MODEL_PATH = "Qwen/Qwen3-VL-8B-Instruct"
# DEFAULT_PROMPT = (
#     # "图1和图是同一场景的不同数据源,两者分辨率一致,图1是点云根据RGB颜色投影得到的,图2是点云根据反射强度得到的,根据两张图像请检测图像中的路沿、车道线和箭头。"
#     # "输出的检测框需要与图1对齐"
#     # "每一个独立目标都是一个实例(一个 bbox)，严禁将多个目标合并到同一个 bbox。"
#     # "label 必须使用英文类别，例如：\"curb\", \"white solid line\", \"white dashed line\", \"yellow solid line\", \"yellow dashed line\", \"double yellow line\", \"crosswalk\", \"arrow\"。"
#     # "仅输出 JSON 数组，格式为: [{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"category\"}]。"


#     # "请检测图像中的路沿、车道线和箭头。"
#     # "每一个独立目标都是一个实例(一个 bbox)，严禁将多个目标合并到同一个 bbox。"
#     # "label 必须使用英文类别，例如：\"curb\", \"white solid line\", \"white dashed line\", \"yellow solid line\", \"yellow dashed line\", \"double yellow line\", \"crosswalk\", \"arrow\"。"
#     # "仅输出 JSON 数组，格式为: [{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"category\"}]。"


#     # "Detect all visible curbs, lane lines, and arrows in the image."
#     # "Each independent target must be a separate instance (one bbox), and do not merge multiple targets into one bbox."
#     # "The label must be an English category, such as \"curb\", \"white solid line\", \"white dashed line\", \"yellow solid line\", \"yellow dashed line\", \"double yellow line\", \"crosswalk\", \"arrow\"."
#     # "If a candidate target is unreliable, ambiguous, or low-confidence, do not output it."
#     # "Output JSON array only, in this format: [{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"category\"}]."
# )

DEFAULT_PROMPT = (
    "Locate every instance that belongs to the following categories: \"curb\", \"white solid line\", \"white dashed line\", \"yellow solid line\", \"yellow dashed line\", \"double yellow line\", \"crosswalk\", \"arrow\". Report bbox coordinates in JSON format."
)


def strip_markdown_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, count=1, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned, count=1)
    return cleaned.strip()


def _collect_json_candidates(text: str) -> list[str]:
    cleaned = strip_markdown_code_fence(text)
    candidates: list[str] = [cleaned]

    list_match = re.search(r"\[[\s\S]*\]", cleaned)
    if list_match:
        candidates.append(list_match.group(0).strip())

    dict_match = re.search(r"\{[\s\S]*\}", cleaned)
    if dict_match:
        candidates.append(dict_match.group(0).strip())

    uniq: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in uniq:
            uniq.append(candidate)
    return uniq


def _normalize_parsed_data(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        raw_items = [data]
    elif isinstance(data, list):
        raw_items = data
    else:
        return []

    detections: list[dict[str, Any]] = []
    for index, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            continue

        bbox = item.get("bbox_2d")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue

        try:
            bbox_numeric = [float(value) for value in bbox]
        except (TypeError, ValueError):
            continue

        label = str(item.get("label", f"lane_line_{index}"))
        detections.append({"bbox_2d": bbox_numeric, "label": label})

    return detections


def _parse_with_known_parsers(candidate: str) -> Any | None:
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(candidate)
        except Exception:
            continue
    return None


def _extract_partial_detection_items(raw_text: str) -> list[dict[str, Any]]:
    cleaned = strip_markdown_code_fence(raw_text)
    object_candidates = re.findall(r"\{[^{}]*\}", cleaned, flags=re.DOTALL)
    if not object_candidates:
        return []

    partial_items: list[dict[str, Any]] = []
    for obj_text in object_candidates:
        if "bbox_2d" not in obj_text:
            continue

        parsed_item = _parse_with_known_parsers(obj_text)
        if isinstance(parsed_item, dict):
            partial_items.append(parsed_item)

    return _normalize_parsed_data(partial_items)


def parse_grounding_response(raw_text: str) -> list[dict[str, Any]]:
    for candidate in _collect_json_candidates(raw_text):
        parsed = _parse_with_known_parsers(candidate)
        if parsed is None:
            continue

        detections = _normalize_parsed_data(parsed)
        if detections:
            return detections

    partial_detections = _extract_partial_detection_items(raw_text)
    if partial_detections:
        return partial_detections
    return []


def bbox_1000_to_pixels(
    bbox_2d: list[float] | tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    if image_width <= 0 or image_height <= 0:
        return 0, 0, 0, 0

    x1 = int(round(float(bbox_2d[0]) / 1000.0 * image_width))
    y1 = int(round(float(bbox_2d[1]) / 1000.0 * image_height))
    x2 = int(round(float(bbox_2d[2]) / 1000.0 * image_width))
    y2 = int(round(float(bbox_2d[3]) / 1000.0 * image_height))

    x1 = min(max(x1, 0), image_width - 1)
    y1 = min(max(y1, 0), image_height - 1)
    x2 = min(max(x2, 0), image_width - 1)
    y2 = min(max(y2, 0), image_height - 1)

    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    return x1, y1, x2, y2


def _draw_label(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, x: int, y: int, text: str, color: str) -> None:
    text_bbox = draw.textbbox((x, y), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    pad = 2
    y_top = max(0, y - text_h - 2 * pad)
    draw.rectangle((x, y_top, x + text_w + 2 * pad, y_top + text_h + 2 * pad), fill=color)
    draw.text((x + pad, y_top + pad), text, fill="white", font=font)


def draw_detections(
    image: Image.Image,
    detections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    colors = ["red", "lime", "cyan", "yellow", "orange", "magenta", "blue"]

    width, height = image.size
    rendered: list[dict[str, Any]] = []

    for idx, detection in enumerate(detections):
        bbox_2d = detection["bbox_2d"]
        label = detection["label"]
        x1, y1, x2, y2 = bbox_1000_to_pixels(bbox_2d, width, height)

        color = colors[idx % len(colors)]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        _draw_label(draw, font, x1, y1, label, color)

        rendered.append(
            {
                "label": label,
                "bbox_2d": [round(float(v), 2) for v in bbox_2d],
                "bbox_px": [x1, y1, x2, y2],
            }
        )

    return rendered


def run_grounding(
    model: Any,
    processor: Any,
    image_path: str,
    prompt: str,
    max_new_tokens: int,
) -> str:
    from qwen_vl_utils import process_vision_info
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": image_path,
                    "resized_height": 1024,
                    "resized_width": 1024,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # processor.image_processor.size = {"longest_edge": 5120*32*32, "shortest_edge": 1280*32*32}
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
        # padding=True,
        return_tensors="pt",
        do_resize=False,
        **video_kwargs,
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]

    response = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return response[0] if response else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lane-line grounding with local Qwen3-VL model")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", default=None, help="Output image path with drawn detections")
    parser.add_argument("--result-json", default=None, help="Optional JSON save path for parsed detections")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Model path used by run.py")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Grounding prompt")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation max new tokens")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_path = Path(args.output).expanduser().resolve() if args.output else image_path.with_name(f"{image_path.stem}_lane_grounding{image_path.suffix}")

    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"Loading model from: {args.model_path}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    raw_response = run_grounding(
        model=model,
        processor=processor,
        image_path=str(image_path),
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )

    print("\n[VLM raw response]")
    print(raw_response)

    detections = parse_grounding_response(raw_response)
    if not detections:
        print("\nNo valid bbox_2d detections parsed from model response.")
        return

    image = Image.open(image_path).convert("RGB")
    rendered = draw_detections(image, detections)
    image.save(output_path)

    print("\n[Detections]")
    for item in rendered:
        print(
            f"label={item['label']}, bbox_2d={item['bbox_2d']}, bbox_px={item['bbox_px']}"
        )

    print(f"\nSaved visualization: {output_path}")

    if args.result_json:
        result_json_path = Path(args.result_json).expanduser().resolve()
        result_json_path.parent.mkdir(parents=True, exist_ok=True)
        result_json_path.write_text(
            json.dumps(rendered, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved detection JSON: {result_json_path}")


if __name__ == "__main__":
    main()
