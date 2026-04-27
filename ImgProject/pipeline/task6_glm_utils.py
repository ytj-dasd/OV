from __future__ import annotations

import json
import re
from importlib.util import find_spec
from pathlib import Path
from typing import Any


DEFAULT_MODEL_PATH = "ZhipuAI/GLM-4.6V-Flash"
DEFAULT_QWEN_MODEL_PATH = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_GEMMA_MODEL_PATH = "google/gemma-4-E4B-it"


MANHOLE_PROMPT = (
    "Picture 1 是井盖局部俯视图，Picture 2 是井盖所在区域全局俯视图。"
    "请判断：\n"
    "1) Functional Type: Rain/Sewage/Electric/Telecom/Gas/Water\n"
    "2) Shape: Round/Square\n"
    "只输出 JSON：\n"
    '{"functional_type":"...","shape":"...","confidence":0.0}'
)


TREE_PROMPT = (
    "Picture 1 是树木 front 视图，Picture 2 是 side90 视图。"
    "请判断 tree_type、tree_trunk_visible。"
    "只输出 JSON：\n"
    '{"tree_type":"...","tree_trunk_visible":"是/否","confidence":0.0}'
)


def build_pole_group_prompt(candidate_class_names: list[str]) -> str:
    names = [str(x) for x in candidate_class_names if str(x)]
    if not names:
        names = ["电线杆", "路灯杆", "路牌", "交通标志", "红绿灯", "监控"]
    name_str = ", ".join(names)
    full_name_str = "电线杆, 路灯杆, 路牌, 交通标志, 红绿灯, 监控"
    return (
        "Picture 1 是同一杆状物的 front 视图，Picture 2 是 side90 视图。"
        f"候选类别参考（非限制）：[{name_str}]。"
        f"最终 contains_classes 允许从以下 6 类中多选：[{full_name_str}]，不要受候选类别限制。"
        "请输出包含的类别（可多选），并按命中类别补充属性：\n"
        "- 路灯杆: arm_type, light_count；若包含路灯杆，"
        "arm_type 只能输出 single_arm、double_arm、no_arm 中的一个"
        "（single_arm=单臂，double_arm=双臂，no_arm=无臂）\n"
        "- 路牌/交通标志: sign_shape, sign_color, sign_content\n"
        "只输出 JSON：\n"
        "{"
        '"contains_classes": ["..."], '
        '"arm_type": null, '
        '"light_count": null, '
        '"sign_shape": null, '
        '"sign_color": null, '
        '"sign_content": null, '
        '"confidence": 0.0'
        "}"
    )


def extract_answer_text(raw: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw.strip()


def parse_json_response(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        return {}
    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return value
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", stripped)
    if match:
        block = match.group(0)
        try:
            value = json.loads(block)
            if isinstance(value, dict):
                return value
        except Exception:
            pass
    return {"raw_text": stripped, "parse_error": True}


def _require_accelerate(vlm_backend: str) -> None:
    if find_spec("accelerate") is None:
        raise RuntimeError(
            f"Task6 {vlm_backend} loading uses device_map='auto', which requires accelerate. "
            "Install it in the current environment with: pip install 'accelerate>=1.13.0'"
        )


def resolve_vlm_model_path(vlm_backend: str, model_path: str | None) -> str:
    explicit = "" if model_path is None else str(model_path).strip()
    if explicit:
        return explicit

    backend = str(vlm_backend).strip().lower()
    if backend == "qwen":
        return DEFAULT_QWEN_MODEL_PATH
    if backend == "glm":
        return DEFAULT_MODEL_PATH
    if backend == "gemma":
        return DEFAULT_GEMMA_MODEL_PATH
    raise ValueError(f"Unsupported VLM backend: {vlm_backend}")


def load_glm_model_and_processor(model_path: str) -> tuple[Any, Any]:
    _require_accelerate("GLM")

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


def load_qwen_model_and_processor(model_path: str) -> tuple[Any, Any]:
    _require_accelerate("Qwen")

    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def load_gemma_model_and_processor(model_path: str) -> tuple[Any, Any]:
    _require_accelerate("Gemma")

    from transformers import AutoModelForMultimodalLM, AutoProcessor

    model = AutoModelForMultimodalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def load_vlm_model_and_processor(vlm_backend: str, model_path: str | None) -> tuple[Any, Any, str]:
    backend = str(vlm_backend).strip().lower()
    resolved_model_path = resolve_vlm_model_path(backend, model_path)
    if backend == "qwen":
        model, processor = load_qwen_model_and_processor(resolved_model_path)
    elif backend == "glm":
        model, processor = load_glm_model_and_processor(resolved_model_path)
    elif backend == "gemma":
        model, processor = load_gemma_model_and_processor(resolved_model_path)
    else:
        raise ValueError(f"Unsupported VLM backend: {vlm_backend}")
    return model, processor, resolved_model_path


def run_dual_image_glm(
    *,
    model: Any,
    processor: Any,
    image_1_path: Path,
    image_2_path: Path,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    repetition_penalty: float = 1.1,
    top_p: float = 0.8,
    top_k: int = 2,
) -> dict[str, Any]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": str(image_1_path)},
                {"type": "image", "url": str(image_2_path)},
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

    do_sample = float(temperature) > 0
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "repetition_penalty": float(repetition_penalty),
        "do_sample": bool(do_sample),
        "top_k": int(top_k),
        "top_p": float(top_p),
    }
    if do_sample:
        generation_kwargs["temperature"] = float(temperature)

    output_ids = model.generate(**inputs, **generation_kwargs)
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_response = processor.decode(generated_ids, skip_special_tokens=False)
    answer = extract_answer_text(raw_response)
    parsed = parse_json_response(answer)
    if "raw_text" not in parsed:
        parsed["raw_text"] = answer
    return parsed


def run_dual_image_qwen(
    *,
    model: Any,
    processor: Any,
    image_1_path: Path,
    image_2_path: Path,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    repetition_penalty: float = 1.1,
    top_p: float = 0.8,
    top_k: int = 2,
) -> dict[str, Any]:
    if find_spec("qwen_vl_utils") is None:
        raise RuntimeError(
            "Task6 Qwen inference requires qwen-vl-utils. "
            "Install it in the current environment, for example: pip install qwen-vl-utils"
        )
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_1_path), "resized_height": 1024, "resized_width": 1024},
                {"type": "image", "image": str(image_2_path), "resized_height": 1024, "resized_width": 1024},
                {"type": "text", "text": prompt},
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

    do_sample = float(temperature) > 0
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "repetition_penalty": float(repetition_penalty),
        "do_sample": bool(do_sample),
        "top_k": int(top_k),
        "top_p": float(top_p),
    }
    if do_sample:
        generation_kwargs["temperature"] = float(temperature)

    generated_ids = model.generate(**inputs, **generation_kwargs)
    generated_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    answer = response[0].strip() if response else ""
    parsed = parse_json_response(answer)
    if "raw_text" not in parsed:
        parsed["raw_text"] = answer
    return parsed


def run_dual_image_gemma(
    *,
    model: Any,
    processor: Any,
    image_1_path: Path,
    image_2_path: Path,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    repetition_penalty: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 64,
) -> dict[str, Any]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": str(image_1_path)},
                {"type": "image", "url": str(image_2_path)},
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
        enable_thinking=False,
    ).to(model.device)
    inputs.pop("token_type_ids", None)

    do_sample = float(temperature) > 0
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "repetition_penalty": float(repetition_penalty),
        "do_sample": bool(do_sample),
        "top_k": int(top_k),
        "top_p": float(top_p),
    }
    if do_sample:
        generation_kwargs["temperature"] = float(temperature)

    output_ids = model.generate(**inputs, **generation_kwargs)
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_response = processor.decode(generated_ids, skip_special_tokens=False)

    answer = raw_response.strip()
    try:
        parsed_response = processor.parse_response(raw_response)
        if isinstance(parsed_response, dict) and "response" in parsed_response:
            answer = str(parsed_response["response"]).strip()
        else:
            answer = str(parsed_response).strip()
    except Exception:
        pass

    parsed = parse_json_response(answer)
    if "raw_text" not in parsed:
        parsed["raw_text"] = answer
    return parsed


def run_dual_image_vlm(
    *,
    vlm_backend: str,
    model: Any,
    processor: Any,
    image_1_path: Path,
    image_2_path: Path,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    repetition_penalty: float = 1.1,
    top_p: float = 0.8,
    top_k: int = 2,
) -> dict[str, Any]:
    backend = str(vlm_backend).strip().lower()
    kwargs = {
        "model": model,
        "processor": processor,
        "image_1_path": image_1_path,
        "image_2_path": image_2_path,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "top_p": top_p,
        "top_k": top_k,
    }
    if backend == "qwen":
        return run_dual_image_qwen(**kwargs)
    if backend == "glm":
        return run_dual_image_glm(**kwargs)
    if backend == "gemma":
        return run_dual_image_gemma(**kwargs)
    raise ValueError(f"Unsupported VLM backend: {vlm_backend}")
