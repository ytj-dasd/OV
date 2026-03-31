"""
A command-line interface for GLM-V models supporting images and videos.

Examples:
    # Text-only chat
    python trans_infer_cli.py
    # Single-turn text inference (no manual typing)
    python trans_infer_cli.py --prompt "Describe this image."
    # Disable thinking mode
    python trans_infer_cli.py --prompt "Describe this image." --no-thinking
    # Chat with single image
    python trans_infer_cli.py --image_paths /path/to/image.jpg
    # Chat with multiple images
    python trans_infer_cli.py --image_paths /path/to/img1.jpg /path/to/img2.png /path/to/img3.png
    # Chat with single video
    python trans_infer_cli.py --video_path /path/to/video.mp4
    # Custom generation parameters
    python trans_infer_cli.py --temperature 0.8 --top_k 5 --max_tokens 4096

Notes:
    - Media files are loaded once at startup and persist throughout the conversation
    - Type 'exit' to quit the chat
    - Chat with images and video is NOT allowed
    - If --prompt is provided, the first round runs without manual input
    - The model will remember the conversation history and can reference uploaded media in subsequent turns
"""

import argparse
import re

from transformers import (
    AutoProcessor,
    Glm4vForConditionalGeneration,
    Glm4vMoeForConditionalGeneration,
)


def build_content(image_paths, video_path, text):
    content = []
    if image_paths:
        for img_path in image_paths:
            content.append({"type": "image", "url": img_path})
    if video_path:
        content.append({"type": "video", "url": video_path})
    content.append({"type": "text", "text": text})
    return content


def infer_one_turn(model, processor, messages, generation_args, content):
    messages.append({"role": "user", "content": content})
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    inputs.pop("token_type_ids", None)
    output = model.generate(
        **inputs,
        max_new_tokens=generation_args.max_tokens,
        repetition_penalty=generation_args.repetition_penalty,
        do_sample=generation_args.temperature > 0,
        top_k=generation_args.top_k,
        top_p=generation_args.top_p,
        temperature=generation_args.temperature if generation_args.temperature > 0 else None,
    )
    raw = processor.decode(
        output[0][inputs["input_ids"].shape[1] : -1], skip_special_tokens=False
    )
    match = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
    answer = match.group(1).strip() if match else raw.strip()
    messages.append(
        {"role": "assistant", "content": [{"type": "text", "text": answer}]}
    )
    return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="ZhipuAI/GLM-4.6V-Flash")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Run first turn from a fixed prompt (useful for debug/launch.json).",
    )
    parser.add_argument(
        "--interactive_after_prompt",
        action="store_true",
        help="Continue interactive chat after running --prompt once.",
    )
    parser.add_argument("--image_paths", type=str, nargs="*", default=None)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--top_p", type=float, default=0.6)
    parser.add_argument("--top_k", type=int, default=2)

    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(args.model_path)
    if "GLM-4.5V" in args.model_path:
        model = Glm4vMoeForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype="auto", device_map="auto"
        )
    else:
        model = Glm4vForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype="auto", device_map="auto"
        )

    messages = []
    first_turn = True
    if args.image_paths is not None and args.video_path is not None:
        raise ValueError(
            "Chat with images and video is NOT allowed. Please use either --image_paths or --video_path, not both."
        )

    if args.prompt is not None:
        content = build_content(args.image_paths, args.video_path, args.prompt)
        raw = infer_one_turn(model, processor, messages, args, content)
        first_turn = False
        print(f"\nUser: {args.prompt}")
        print(f"Assistant: {raw}")
        if not args.interactive_after_prompt:
            return

    while True:
        question = input("\nUser: ").strip()
        if not question:
            continue
        if question.lower() == "exit":
            break
        if first_turn:
            content = build_content(args.image_paths, args.video_path, question)
            first_turn = False
        else:
            content = [{"type": "text", "text": question}]
        raw = infer_one_turn(model, processor, messages, args, content)
        print(f"Assistant: {raw}")


if __name__ == "__main__":
    main()
