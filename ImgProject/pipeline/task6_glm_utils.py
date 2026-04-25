from __future__ import annotations

import json
import re
from importlib.util import find_spec
from pathlib import Path
from typing import Any


DEFAULT_MODEL_PATH = "ZhipuAI/GLM-4.6V-Flash"


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
    "请判断 tree_type、tree_trunk_visible、tree_pit。"
    "只输出 JSON：\n"
    '{"tree_type":"...","tree_trunk_visible":"是/否","tree_pit":"是/否","confidence":0.0}'
)


def build_pole_group_prompt(candidate_class_names: list[str]) -> str:
    names = [str(x) for x in candidate_class_names if str(x)]
    if not names:
        names = ["电线杆", "路灯杆", "路牌", "交通标志", "红绿灯", "监控"]
    name_str = ", ".join(names)
    return (
        "Picture 1 是同一杆状物的 front 视图，Picture 2 是 side90 视图。"
        f"候选类别仅限：[{name_str}]。"
        "请输出包含的类别（可多选），并按命中类别补充属性：\n"
        "- 路灯杆: arm_type, light_count\n"
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


def load_glm_model_and_processor(model_path: str) -> tuple[Any, Any]:
    if find_spec("accelerate") is None:
        raise RuntimeError(
            "Task6 GLM loading uses device_map='auto', which requires accelerate. "
            "Install it in the current environment with: pip install 'accelerate>=1.13.0'"
        )

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
