#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze local images with Gemma-4-E4B-it."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "gemma-4-E4B-it"),
        help="Local model path or Hugging Face model id (e.g. google/gemma-4-E4B-it).",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing images to analyze.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of generated tokens per image.",
    )
    return parser.parse_args()


def infer_one_image(
    processor: AutoProcessor,
    model: AutoModelForMultimodalLM,
    image_path: Path,
    max_new_tokens: int,
) -> str:
    prompt = (
        "Describe this image in Chinese. "
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
        )

    input_len = inputs["input_ids"].shape[-1]
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    try:
        parsed = processor.parse_response(response)
        if isinstance(parsed, dict) and "response" in parsed:
            return str(parsed["response"]).strip()
        return str(parsed).strip()
    except Exception:
        return response.strip()


def main() -> None:
    args = build_args()
    if not args.image_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {args.image_dir}")

    image_paths = sorted(args.image_dir.glob("*.png"))

    model_source = args.model_dir
    print(f"[INFO] loading model from: {model_source}")
    processor = AutoProcessor.from_pretrained(model_source)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_source,
        torch_dtype="auto",
        device_map="auto",
    )

    for image_path in image_paths[:1]:
        print(f"\n=== {image_path.name} ===")
        answer = infer_one_image(
            processor=processor,
            model=model,
            image_path=image_path,
            max_new_tokens=args.max_new_tokens,
        )
        print(answer)


if __name__ == "__main__":
    main()
