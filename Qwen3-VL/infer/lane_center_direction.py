from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


DEFAULT_MODEL_PATH = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_PROMPT = (
    "请检测图像中所有可见车道线。每一条车道线是一个独立实例，不要把多条线合并。"
    "对每条车道线输出其中心点和方向向量。"
    "point_2d 表示中心点坐标；direction_2d 表示该车道线在图像坐标系中的单位方向向量 [dx, dy]，"
    "其中 x 向右为正，y 向下为正。"
    "仅输出 JSON 数组，格式为: "
    "[{\"point_2d\": [x, y], \"direction_2d\": [dx, dy], \"label\": \"lane_line\"}]。"
    "point_2d 使用 0-1000 相对坐标。"
)

_NUMBER_PATTERN = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


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


def _float_pair(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return [float(value[0]), float(value[1])]
    except (TypeError, ValueError):
        return None


def _normalize_lane_items(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        raw_items = [data]
    elif isinstance(data, list):
        raw_items = data
    else:
        return []

    lanes: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            continue

        point = _float_pair(item.get("point_2d") or item.get("center_2d"))
        direction = _float_pair(item.get("direction_2d") or item.get("direction_vector"))

        if direction is None and "direction_deg" in item:
            try:
                angle_deg = float(item["direction_deg"])
                angle_rad = math.radians(angle_deg)
                direction = [math.cos(angle_rad), math.sin(angle_rad)]
            except (TypeError, ValueError):
                direction = None

        if point is None or direction is None:
            continue

        label = str(item.get("label", f"lane_line_{idx}"))
        lanes.append(
            {
                "point_2d": point,
                "direction_2d": direction,
                "label": label,
            }
        )

    return lanes


def _parse_truncated_lane_items(raw_text: str) -> list[dict[str, Any]]:
    cleaned = strip_markdown_code_fence(raw_text)
    object_fragments = re.findall(r"\{[^{}]*\}", cleaned, flags=re.DOTALL)

    point_pattern = re.compile(
        rf"[\"'](?:point_2d|center_2d)[\"']\s*:\s*\[\s*({_NUMBER_PATTERN})\s*,\s*({_NUMBER_PATTERN})\s*\]",
        flags=re.IGNORECASE,
    )
    direction_pattern = re.compile(
        rf"[\"'](?:direction_2d|direction_vector)[\"']\s*:\s*\[\s*({_NUMBER_PATTERN})\s*,\s*({_NUMBER_PATTERN})\s*\]",
        flags=re.IGNORECASE,
    )
    label_pattern = re.compile(r"[\"']label[\"']\s*:\s*[\"']([^\"'\r\n]*)[\"']", flags=re.IGNORECASE)

    lanes: list[dict[str, Any]] = []
    for idx, frag in enumerate(object_fragments, start=1):
        point_match = point_pattern.search(frag)
        direction_match = direction_pattern.search(frag)
        if not point_match or not direction_match:
            continue

        point = [float(point_match.group(1)), float(point_match.group(2))]
        direction = [float(direction_match.group(1)), float(direction_match.group(2))]

        label_match = label_pattern.search(frag)
        label = label_match.group(1).strip() if label_match else f"lane_line_{idx}"

        lanes.append(
            {
                "point_2d": point,
                "direction_2d": direction,
                "label": label,
            }
        )

    return lanes


def parse_lane_response(raw_text: str) -> list[dict[str, Any]]:
    for candidate in _collect_json_candidates(raw_text):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(candidate)
            except Exception:
                continue

            lanes = _normalize_lane_items(parsed)
            if lanes:
                return lanes

    return _parse_truncated_lane_items(raw_text)


def point_1000_to_pixels(
    point_2d: list[float] | tuple[float, float],
    image_width: int,
    image_height: int,
) -> tuple[int, int]:
    if image_width <= 0 or image_height <= 0:
        return 0, 0

    x_val, y_val = float(point_2d[0]), float(point_2d[1])
    if max(abs(x_val), abs(y_val)) <= 1000.0:
        x = int(round(x_val / 1000.0 * image_width))
        y = int(round(y_val / 1000.0 * image_height))
    else:
        x = int(round(x_val))
        y = int(round(y_val))

    x = min(max(x, 0), image_width - 1)
    y = min(max(y, 0), image_height - 1)
    return x, y


def _normalize_direction(direction_2d: list[float] | tuple[float, float]) -> tuple[float, float]:
    dx, dy = float(direction_2d[0]), float(direction_2d[1])
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return 1.0, 0.0
    return dx / norm, dy / norm


def _draw_arrow_head(draw: ImageDraw.ImageDraw, end_x: int, end_y: int, angle: float, color: str) -> None:
    head_len = 12
    spread = math.pi / 7

    x1 = int(round(end_x - head_len * math.cos(angle - spread)))
    y1 = int(round(end_y - head_len * math.sin(angle - spread)))
    x2 = int(round(end_x - head_len * math.cos(angle + spread)))
    y2 = int(round(end_y - head_len * math.sin(angle + spread)))

    draw.line((end_x, end_y, x1, y1), fill=color, width=3)
    draw.line((end_x, end_y, x2, y2), fill=color, width=3)


def _draw_label(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, x: int, y: int, text: str, color: str) -> None:
    text_bbox = draw.textbbox((x, y), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    pad = 2
    y_top = max(0, y - text_h - 2 * pad)
    draw.rectangle((x, y_top, x + text_w + 2 * pad, y_top + text_h + 2 * pad), fill=color)
    draw.text((x + pad, y_top + pad), text, fill="white", font=font)


def draw_lane_points_and_directions(
    image: Image.Image,
    lanes: list[dict[str, Any]],
    arrow_length_px: int,
) -> list[dict[str, Any]]:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    colors = ["red", "lime", "cyan", "yellow", "orange", "magenta", "blue"]

    width, height = image.size
    rendered: list[dict[str, Any]] = []

    for idx, lane in enumerate(lanes):
        color = colors[idx % len(colors)]
        label = lane["label"]

        cx, cy = point_1000_to_pixels(lane["point_2d"], width, height)
        ux, uy = _normalize_direction(lane["direction_2d"])

        ex = int(round(cx + ux * arrow_length_px))
        ey = int(round(cy + uy * arrow_length_px))

        ex = min(max(ex, 0), width - 1)
        ey = min(max(ey, 0), height - 1)

        radius = 5
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color, outline=color)
        draw.line((cx, cy, ex, ey), fill=color, width=3)

        angle = math.atan2(ey - cy, ex - cx)
        _draw_arrow_head(draw, ex, ey, angle, color)
        _draw_label(draw, font, cx + 6, cy + 6, label, color)

        rendered.append(
            {
                "label": label,
                "point_2d": [round(float(v), 2) for v in lane["point_2d"]],
                "direction_2d": [round(float(v), 4) for v in lane["direction_2d"]],
                "point_px": [cx, cy],
                "arrow_end_px": [ex, ey],
            }
        )

    return rendered


def run_inference(
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
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
        **video_kwargs,
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0] if output_text else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lane center-point and direction inference with local Qwen3-VL")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", default=None, help="Output image path with points and arrows")
    parser.add_argument("--result-json", default=None, help="Optional JSON save path for parsed lanes")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Model path used by run.py")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt for center-point and direction extraction")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Generation max new tokens")
    parser.add_argument("--arrow-length", type=int, default=70, help="Arrow length in pixels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else image_path.with_name(f"{image_path.stem}_lane_center_direction{image_path.suffix}")
    )

    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"Loading model from: {args.model_path}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    raw_response = run_inference(
        model=model,
        processor=processor,
        image_path=str(image_path),
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )

    print("\n[VLM raw response]")
    print(raw_response)

    lanes = parse_lane_response(raw_response)
    if not lanes:
        print("\nNo valid point_2d + direction_2d lanes parsed from model response.")
        return

    image = Image.open(image_path).convert("RGB")
    rendered = draw_lane_points_and_directions(image, lanes, arrow_length_px=args.arrow_length)
    image.save(output_path)

    print("\n[Lanes]")
    for item in rendered:
        print(
            f"label={item['label']}, point_2d={item['point_2d']}, direction_2d={item['direction_2d']}, "
            f"point_px={item['point_px']}, arrow_end_px={item['arrow_end_px']}"
        )

    print(f"\nSaved visualization: {output_path}")

    if args.result_json:
        result_json_path = Path(args.result_json).expanduser().resolve()
        result_json_path.parent.mkdir(parents=True, exist_ok=True)
        result_json_path.write_text(
            json.dumps(rendered, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved lane JSON: {result_json_path}")


if __name__ == "__main__":
    main()
