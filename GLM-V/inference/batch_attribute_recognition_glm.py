"""Batch dual-image attribute recognition with GLM-4.6V-Flash.

Each sample pairs a local crop (Picture 1) and a global context image (Picture 2).
Given a target category, the model extracts fine-grained attributes.

Supported categories:
  - 道路标记 (road_marking)
  - 井盖 (manhole)
  - 路灯杆 (street_light)
  - 交通标志 (traffic_sign)
  - 树木 (tree)

Usage:
    python inference/batch_attribute_recognition_glm.py \
        --crop-dir path/to/crops \
        --global-dir path/to/global_images \
        --category road_marking \
        --output-dir output/attributes
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


DEFAULT_MODEL_PATH = "ZhipuAI/GLM-4.6V-Flash"

SUPPORTED_IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
}

# ---------------------------------------------------------------------------
# Category attribute definitions
# ---------------------------------------------------------------------------

CATEGORY_ATTRIBUTE_PROMPTS: dict[str, dict[str, Any]] = {
    "road_marking": {
        "name_zh": "道路标记",
        "name_en": "road marking",
        "attributes": ["type", "direction", "color"],
        "prompt": (
            "Picture 1 是目标道路标记的局部放大图，Picture 2 是包含该目标的全局道路场景图。"
            "请根据两张图片识别该道路标记的具体属性。\n"
            "需要判断的属性：\n"
            "1. type（标记类型）：停止线、白色实线、白色虚线、黄色实线、黄色虚线、双黄线、斑马线、转向箭头、其他\n"
            "2. direction（方向，仅转向箭头需要）：直行、左转、右转、掉头、直行左转、直行右转、左转掉头、无\n"
            "3. color（主要颜色）：白色、黄色、其他\n"
            "请注意图像是俯视视角，如果只根据箭头指向左侧就判断是左转那会出错，因为可能是右转箭头，在俯视图上这段路是从上往下的，所以看上去是指向左边的。\n"
            "请严格以 JSON 格式输出，不要输出其他内容：\n"
            '{"type": "标记类型", "direction": "方向", "color": "颜色", "confidence": 0.0到1.0的小数}'
        ),
    },
    "manhole": {
        "name_zh": "井盖",
        "name_en": "manhole",
        "attributes": ["shape", "material", "pattern"],
        "prompt": (
            "Picture 1 是目标井盖的局部放大图，Picture 2 是包含该目标的全局道路场景图。"
            "请根据两张图片识别该井盖的具体属性。\n"
            "需要判断的属性：\n"
            "1. shape（形状）：圆形、方形、其他\n"
            "2. material（材质）：金属、复合材料、其他\n"
            "3. pattern（表面纹路/标识）：有无特殊标识，如有请描述\n"
            "请严格以 JSON 格式输出，不要输出其他内容：\n"
            '{"shape": "形状", "material": "材质", "pattern": "纹路描述", "confidence": 0.0到1.0的小数}'
        ),
    },
    "street_light": {
        "name_zh": "路灯杆",
        "name_en": "street light",
        "attributes": ["arm_type", "light_count", "pole_shape"],
        "prompt": (
            "Picture 1 是目标路灯杆的局部放大图，Picture 2 是包含该目标的全局道路场景图。"
            "请根据两张图片识别该路灯杆的具体属性。\n"
            "需要判断的属性：\n"
            "1. arm_type（灯臂类型）：单臂、双臂、无臂（吸顶/悬挂式）、其他\n"
            "2. light_count（灯具数量）：1、2、3、其他\n"
            "请严格以 JSON 格式输出，不要输出其他内容：\n"
            '{"arm_type": "灯臂类型", "light_count": "数量", "confidence": 0.0到1.0的小数}'
        ),
    },
    "traffic_sign": {
        "name_zh": "交通标志",
        "name_en": "traffic sign",
        "attributes": ["sign_shape", "sign_color", "sign_content"],
        "prompt": (
            "Picture 1 是目标交通标志的局部放大图，Picture 2 是包含该目标的全局道路场景图。"
            "请根据两张图片识别该交通标志的具体属性。\n"
            "需要判断的属性：\n"
            "1. sign_shape（标志形状）：圆形、三角形、方形、菱形、八角形、倒三角形、其他\n"
            "2. sign_color（标志主色调）：红色、蓝色、黄色、绿色、白色、黑色、其他\n"
            "3. sign_content（标志内容/含义）：限速XX、禁止XX、指示XX方向、警告XX、辅助说明XX等，请尽可能描述\n"
            "请严格以 JSON 格式输出，不要输出其他内容：\n"
            '{"sign_shape": "标志形状", "sign_color": "主色调", "sign_content": "内容描述", "confidence": 0.0到1.0的小数}'
        ),
    },
    "tree": {
        "name_zh": "树木",
        "name_en": "tree",
        "attributes": ["tree_type", "tree_crown_shape", "tree_trunk_visible", "tree_pit"],
        "prompt": (
            "Picture 1 是目标树木的局部放大图，Picture 2 是包含该目标的全局道路场景图。"
            "请根据两张图片识别该树木的具体属性。\n"
            "需要判断的属性：\n"
            "1. tree_type（树木类型）：阔叶树、针叶树、棕榈类、灌木、其他\n"
            "2. tree_crown_shape（树冠形态）：球形、锥形、圆柱形、伞形、不规则、其他\n"
            "3. tree_trunk_visible（树干是否可见）：是、否\n"
            "4. tree_pit（树穴是否存在）：是（有树穴/树池）、否（无明显树穴）\n"
            "请严格以 JSON 格式输出，不要输出其他内容：\n"
            '{"tree_type": "树木类型", "tree_crown_shape": "树冠形态", "tree_trunk_visible": "是/否", "tree_pit": "是/否", "confidence": 0.0到1.0的小数}'
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch dual-image attribute recognition with GLM-4.6V.",
    )
    parser.add_argument("--crop-dir", required=True, help="Folder of local crop images (Picture 1)")
    parser.add_argument("--global-dir", required=True, help="Folder of global context images (Picture 2)")
    parser.add_argument("--output-dir", required=True, help="Folder to save attribute results")
    parser.add_argument("--category", required=True, choices=list(CATEGORY_ATTRIBUTE_PROMPTS.keys()),
                        help="Target category for attribute recognition")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                        help="Model path (default: ZhipuAI/GLM-4.6V-Flash)")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max generation tokens")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--top-p", type=float, default=0.8, help="Top-p for sampling")
    parser.add_argument("--top-k", type=int, default=2, help="Top-k for sampling")
    parser.add_argument("--black-ratio-threshold", type=float, default=0.999,
                        help="Skip if black pixel ratio > this (default: 0.8)")
    parser.add_argument("--black-pixel-threshold", type=int, default=10,
                        help="Pixel is black if R,G,B <= this (default: 10)")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan image folders")
    parser.add_argument("--strict-pairing", action="store_true",
                        help="Raise error when pairing is incomplete")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

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
        1 for r, g, b in pixels
        if r <= black_threshold and g <= black_threshold and b <= black_threshold
    )
    return float(black_count / len(pixels))


def extract_answer_text(raw: str) -> str:
    """Extract content within <answer>...</answer> tags, or return raw text."""
    match = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw.strip()


def parse_attribute_response(text: str) -> dict[str, Any]:
    """Parse model output into structured attribute dict.

    Tries JSON parse first; falls back to key=value extraction.
    """
    # Try to find JSON block
    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
            return result
        except json.JSONDecodeError:
            pass

    # Fallback: extract key-value pairs like "key": "value"
    attrs: dict[str, Any] = {}
    for kv_match in re.finditer(r'"(\w+)"\s*[:：]\s*"([^"]*)"', text):
        attrs[kv_match.group(1)] = kv_match.group(2)
    for num_match in re.finditer(r'"(\w+)"\s*[:：]\s*([\d.]+)', text):
        key = num_match.group(1)
        if key not in attrs:
            val = num_match.group(2)
            try:
                attrs[key] = float(val) if "." in val else int(val)
            except ValueError:
                attrs[key] = val

    return attrs if attrs else {"raw_text": text, "parse_error": True}


# ---------------------------------------------------------------------------
# Model loading & inference
# ---------------------------------------------------------------------------

def load_glm_model_and_processor(model_path: str) -> tuple[Any, Any]:
    from transformers import AutoProcessor, Glm4vForConditionalGeneration, Glm4vMoeForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_path)
    if "GLM-4.5V" in model_path:
        model = Glm4vMoeForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto",
        )
    else:
        model = Glm4vForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto",
        )
    return model, processor


def run_attribute_recognition(
    model: Any,
    processor: Any,
    crop_image_path: str,
    global_image_path: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    repetition_penalty: float,
    top_p: float,
    top_k: int,
) -> tuple[str, str]:
    """Run dual-image attribute recognition through GLM chat template."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": crop_image_path},
                {"type": "image", "url": global_image_path},
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
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_response = processor.decode(generated_ids, skip_special_tokens=False)
    answer_text = extract_answer_text(raw_response)
    return raw_response, answer_text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    crop_dir = Path(args.crop_dir).expanduser().resolve()
    global_dir = Path(args.global_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not crop_dir.is_dir():
        raise NotADirectoryError(f"Crop directory not found: {crop_dir}")
    if not global_dir.is_dir():
        raise NotADirectoryError(f"Global directory not found: {global_dir}")

    cat_def = CATEGORY_ATTRIBUTE_PROMPTS[args.category]
    prompt = cat_def["prompt"]

    crop_images = collect_relative_images(crop_dir, recursive=args.recursive)
    global_images = collect_relative_images(global_dir, recursive=args.recursive)

    crop_keys = set(crop_images.keys())
    global_keys = set(global_images.keys())
    common_keys = sorted(crop_keys & global_keys)
    only_crop = sorted(crop_keys - global_keys)
    only_global = sorted(global_keys - crop_keys)

    if args.strict_pairing and (only_crop or only_global):
        raise ValueError(
            f"Pairing mismatch under --strict-pairing. "
            f"missing_in_global={len(only_crop)}, missing_in_crop={len(only_global)}"
        )

    if not common_keys:
        print("No paired images found between --crop-dir and --global-dir")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Category: {cat_def['name_zh']} ({cat_def['name_en']})")
    print(f"Attributes: {cat_def['attributes']}")
    print(f"Loading model from: {args.model_path}")
    model, processor = load_glm_model_and_processor(args.model_path)

    summary: list[dict[str, Any]] = []
    processed = 0
    skipped = 0
    failed = 0

    for idx, rel_key in enumerate(common_keys, start=1):
        crop_path = crop_images[rel_key]
        global_path = global_images[rel_key]

        print(f"\n[{idx}/{len(common_keys)}] {rel_key}")

        result_path = output_dir / "results" / rel_key.with_suffix(".json")
        result_path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "relative_name": str(rel_key),
            "crop_image_path": str(crop_path),
            "global_image_path": str(global_path),
            "category": args.category,
            "category_zh": cat_def["name_zh"],
            "status": "unknown",
        }

        try:
            crop_image = Image.open(crop_path).convert("RGB")
            black_ratio = calculate_black_ratio(crop_image, args.black_pixel_threshold)
            payload["black_ratio"] = round(black_ratio, 6)

            if black_ratio > args.black_ratio_threshold:
                payload["status"] = "skipped"
                payload["reason"] = f"black_ratio={black_ratio:.6f} > threshold={args.black_ratio_threshold:.6f}"
                skipped += 1
                result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                summary.append({"relative_name": str(rel_key), "status": "skipped", "result_json": str(result_path)})
                print(f"  Skipped: black_ratio={black_ratio:.4f}")
                continue

            raw_response, answer_text = run_attribute_recognition(
                model=model,
                processor=processor,
                crop_image_path=str(crop_path),
                global_image_path=str(global_path),
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                top_p=args.top_p,
                top_k=args.top_k,
            )

            attributes = parse_attribute_response(answer_text)

            payload["status"] = "ok"
            payload["raw_response"] = raw_response
            payload["answer_text"] = answer_text
            payload["attributes"] = attributes

            print(f"  Attributes: {json.dumps(attributes, ensure_ascii=False)}")

            processed += 1

        except Exception as exc:  # noqa: BLE001
            payload["status"] = "error"
            payload["error"] = f"{type(exc).__name__}: {exc}"
            failed += 1
            print(f"  Error: {exc}")

        result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        summary.append({
            "relative_name": str(rel_key),
            "status": payload["status"],
            "attributes": payload.get("attributes"),
            "result_json": str(result_path),
        })

    # Write summary
    summary_payload = {
        "category": args.category,
        "category_zh": cat_def["name_zh"],
        "attributes_expected": cat_def["attributes"],
        "crop_dir": str(crop_dir),
        "global_dir": str(global_dir),
        "output_dir": str(output_dir),
        "total_crop_images": len(crop_images),
        "total_global_images": len(global_images),
        "paired_images": len(common_keys),
        "missing_in_global": [str(p) for p in only_crop],
        "missing_in_crop": [str(p) for p in only_global],
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "results": summary,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"Done. Processed: {processed}, Skipped: {skipped}, Failed: {failed}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
