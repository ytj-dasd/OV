from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

from lane_grounding import DEFAULT_MODEL_PATH, run_grounding

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

FINE_CLASS_VOCAB: list[dict[str, Any]] = [
    {"id": 1, "name_zh": "电线杆", "name_en": "utility pole", "aliases": ["电线杆", "电杆", "电力杆", "utility pole", "power pole"]},
    {"id": 2, "name_zh": "路灯杆", "name_en": "street light", "aliases": ["路灯杆", "灯杆", "street light pole", "streetlight pole", "lamp post"]},
    {"id": 3, "name_zh": "路牌", "name_en": "signboard", "aliases": ["路牌", "street signboard", "road signboard", "signboard"]},
    {"id": 4, "name_zh": "交通标志", "name_en": "traffic sign", "aliases": ["交通标志", "交通标识", "traffic sign"]},
    {"id": 5, "name_zh": "红绿灯", "name_en": "signal light", "aliases": ["红绿灯", "信号灯", "traffic light"]},
    {"id": 6, "name_zh": "监控", "name_en": "surveillance camera", "aliases": ["监控", "监控摄像头", "camera", "cctv", "surveillance camera"]},
    {"id": 7, "name_zh": "行道树", "name_en": "tree", "aliases": ["行道树", "道路树木", "street tree", "tree"]},
    {"id": 8, "name_zh": "果壳箱", "name_en": "litter bin", "aliases": ["果壳箱", "垃圾桶", "bin", "trash can", "litter bin"]},
    {"id": 9, "name_zh": "消防栓", "name_en": "fire hydrant", "aliases": ["消防栓", "消火栓", "fire hydrant"]},
    {"id": 10, "name_zh": "电箱", "name_en": "utility box", "aliases": ["电箱", "配电箱", "electric box", "utility box", "electrical cabinet", "box"]},
    {"id": 11, "name_zh": "雕塑", "name_en": "sculpture", "aliases": ["雕塑", "雕像", "statue", "sculpture"]},
    {"id": 12, "name_zh": "座椅", "name_en": "bench", "aliases": ["座椅", "长椅", "bench", "seat"]},
    {"id": 13, "name_zh": "交通锥", "name_en": "traffic cone", "aliases": ["交通锥", "路锥", "traffic cone", "cone"]},
    {"id": 14, "name_zh": "柱墩", "name_en": "bollard", "aliases": ["柱墩", "防撞柱", "bollard"]},
    {"id": 15, "name_zh": "围栏", "name_en": "fence", "aliases": ["围栏", "栏杆", "护栏", "fence", "guardrail"]},
]

COARSE_CLASS_VOCAB: list[dict[str, Any]] = [
    {"id": 1, "name_zh": "杆状物", "name_en": "pole", "aliases": ["杆状物", "杆体", "pole", "utility pole", "streetlight pole", "traffic light pole", "camera pole"]},
    {"id": 2, "name_zh": "标识标牌", "name_en": "sign", "aliases": ["标识标牌", "标牌", "标志", "sign", "traffic sign", "signboard"]},
    {"id": 3, "name_zh": "围栏", "name_en": "fence", "aliases": ["围栏", "护栏", "栏杆", "fence", "guardrail"]},
    {"id": 4, "name_zh": "柱墩", "name_en": "bollard", "aliases": ["柱墩", "防撞柱", "bollard"]},
    {"id": 5, "name_zh": "交通锥", "name_en": "traffic cone", "aliases": ["交通锥", "路锥", "traffic cone", "cone"]},
    {"id": 6, "name_zh": "箱体", "name_en": "box", "aliases": ["箱体", "电箱", "配电箱", "box", "utility box", "electrical cabinet"]},
    {"id": 7, "name_zh": "树木", "name_en": "tree", "aliases": ["树木", "行道树", "tree", "street tree"]},
    {"id": 8, "name_zh": "雕塑", "name_en": "sculpture", "aliases": ["雕塑", "雕像", "sculpture", "statue"]},
    {"id": 9, "name_zh": "座椅", "name_en": "bench", "aliases": ["座椅", "长椅", "bench", "seat"]},
]


def build_default_prompt(class_vocab: list[dict[str, Any]]) -> str:
    class_text = "、".join([f"{item['name_zh']}({item['name_en']})" for item in class_vocab])
    return (
        "你是城市道路场景类别筛选器。"
        "请从给定候选类别中判断图像里实际存在的目标类别。"
        f"候选类别仅限：{class_text}。"
        "输出必须是 JSON 数组，元素格式为："
        "{\"class_name\":\"中文类别名\",\"class_name_en\":\"英文文本\",\"confidence\":0到1之间的小数}。"
        "如果都不存在，输出 []。禁止输出候选类别之外的类别。"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task3 batch scene classification with Qwen3-VL."
    )
    parser.add_argument("--data-root", required=True, help="benchmark data root")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Qwen model path")
    parser.add_argument("--prompt", default=None, help="Classification prompt; default follows selected class mode")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max generation tokens")
    parser.add_argument(
        "--use-coarse-classes",
        action="store_true",
        help="Use coarse classes: 杆状物/pole, 标识标牌/sign, 围栏/fence, 柱墩/bollard, 交通锥/traffic cone, 箱体/box, 树木/tree, 雕塑/sculpture, 座椅/bench.",
    )
    parser.add_argument(
        "--black-ratio-threshold",
        type=float,
        default=0.8,
        help="Skip inference when black ratio is higher than this threshold",
    )
    parser.add_argument(
        "--black-pixel-threshold",
        type=int,
        default=10,
        help="Pixel is considered black when R,G,B <= this value",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only write class_vocab.yaml and discover scenes; no model inference",
    )
    return parser.parse_args()


def _normalize_token(text: str) -> str:
    token = text.strip().lower()
    token = re.sub(r"[\s_\-]+", "", token)
    return token


def build_alias_to_class(class_vocab: list[dict[str, Any]]) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for item in class_vocab:
        class_name = str(item["name_zh"])
        alias_map[_normalize_token(class_name)] = class_name
        alias_map[_normalize_token(str(item["name_en"]))] = class_name
        for alias in item.get("aliases", []):
            alias_map[_normalize_token(str(alias))] = class_name
    return alias_map


CLASS_VOCAB: list[dict[str, Any]] = []
ALIAS_TO_CLASS: dict[str, str] = {}
CLASS_ORDER: list[str] = []
ORDER_INDEX: dict[str, int] = {}
CLASS_NAME_TO_EN: dict[str, str] = {}
DEFAULT_PROMPT = ""


def activate_vocab(class_vocab: list[dict[str, Any]]) -> None:
    global CLASS_VOCAB, ALIAS_TO_CLASS, CLASS_ORDER, ORDER_INDEX, CLASS_NAME_TO_EN, DEFAULT_PROMPT
    CLASS_VOCAB = class_vocab
    ALIAS_TO_CLASS = build_alias_to_class(class_vocab)
    CLASS_ORDER = [item["name_zh"] for item in class_vocab]
    ORDER_INDEX = {name: idx for idx, name in enumerate(CLASS_ORDER)}
    CLASS_NAME_TO_EN = {str(item["name_zh"]): str(item["name_en"]) for item in class_vocab}
    DEFAULT_PROMPT = build_default_prompt(class_vocab)


activate_vocab(FINE_CLASS_VOCAB)


def collect_scene_dirs(data_root: Path) -> list[Path]:
    scenes: list[Path] = []
    for child in sorted(data_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "projected_images").is_dir():
            scenes.append(child)
    return scenes


def collect_images(projected_images_dir: Path) -> list[Path]:
    images = [
        p
        for p in sorted(projected_images_dir.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    ]
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


def _extract_json_candidates(text: str) -> list[str]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, count=1, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped, count=1)
        stripped = stripped.strip()

    candidates: list[str] = [stripped]
    match = re.search(r"\[[\s\S]*\]", stripped)
    if match:
        candidates.append(match.group(0).strip())
    uniq: list[str] = []
    for item in candidates:
        if item and item not in uniq:
            uniq.append(item)
    return uniq


def _parse_score(raw_score: Any, default: float = 1.0) -> float:
    try:
        value = float(raw_score)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, value))


def _canonical_class_name(raw_name: str) -> str | None:
    normalized = _normalize_token(raw_name)
    return ALIAS_TO_CLASS.get(normalized)


def _parse_with_known_parsers(candidate: str) -> Any | None:
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(candidate)
        except Exception:
            continue
    return None


def _normalize_class_items(data: Any) -> list[tuple[str, float]]:
    if isinstance(data, dict):
        raw_items = [data]
    elif isinstance(data, list):
        raw_items = data
    else:
        return []

    parsed_entries: list[tuple[str, float]] = []
    for item in raw_items:
        if isinstance(item, str):
            class_name = _canonical_class_name(item)
            if class_name is not None:
                parsed_entries.append((class_name, 1.0))
            continue
        if not isinstance(item, dict):
            continue
        raw_name = (
            item.get("class_name")
            or item.get("class")
            or item.get("label")
            or item.get("name")
            or item.get("class_name_en")
        )
        if not raw_name:
            continue
        class_name = _canonical_class_name(str(raw_name))
        if class_name is None:
            continue
        score = _parse_score(
            item.get("confidence", item.get("score", item.get("prob", 1.0))),
            default=1.0,
        )
        parsed_entries.append((class_name, score))
    return parsed_entries


def _extract_partial_class_items(raw_text: str) -> list[tuple[str, float]]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, count=1, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned, count=1)
        cleaned = cleaned.strip()

    object_candidates = re.findall(r"\{[^{}]*\}", cleaned, flags=re.DOTALL)
    if not object_candidates:
        return []

    recovered: list[tuple[str, float]] = []
    for obj_text in object_candidates:
        if not any(k in obj_text for k in ("class_name", "class", "label", "name")):
            continue
        parsed_item = _parse_with_known_parsers(obj_text)
        if parsed_item is None:
            continue
        recovered.extend(_normalize_class_items(parsed_item))
    return recovered


def parse_classification_response(raw_text: str) -> list[tuple[str, float]]:
    parsed_entries: list[tuple[str, float]] = []

    for candidate in _extract_json_candidates(raw_text):
        parsed = _parse_with_known_parsers(candidate)
        if parsed is None:
            continue

        parsed_entries.extend(_normalize_class_items(parsed))
        if parsed_entries:
            break

    if not parsed_entries:
        parsed_entries = _extract_partial_class_items(raw_text)

    dedup: dict[str, float] = {}
    for class_name, score in parsed_entries:
        dedup[class_name] = max(dedup.get(class_name, 0.0), score)

    return sorted(dedup.items(), key=lambda x: ORDER_INDEX.get(x[0], 10**9))


def write_vlm_txt(output_path: Path, classes: list[tuple[str, float]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not classes:
        output_path.write_text("", encoding="utf-8")
        return
    lines = [f"{class_name}\t{CLASS_NAME_TO_EN.get(class_name, class_name)}\t{score:.4f}" for class_name, score in classes]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _yaml_quote(text: str) -> str:
    escaped = text.replace("'", "''")
    return f"'{escaped}'"


def write_class_vocab_yaml(data_root: Path) -> Path:
    out_path = data_root / "class_vocab.yaml"
    lines: list[str] = ["classes:"]
    for item in CLASS_VOCAB:
        lines.append(f"  - id: {item['id']}")
        lines.append(f"    name_zh: {_yaml_quote(str(item['name_zh']))}")
        lines.append(f"    name_en: {_yaml_quote(str(item['name_en']))}")
        lines.append("    aliases:")
        for alias in item["aliases"]:
            lines.append(f"      - {_yaml_quote(str(alias))}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    args = parse_args()

    if args.use_coarse_classes:
        activate_vocab(COARSE_CLASS_VOCAB)
    else:
        activate_vocab(FINE_CLASS_VOCAB)
    if args.prompt is None:
        args.prompt = DEFAULT_PROMPT

    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.exists() or not data_root.is_dir():
        raise NotADirectoryError(f"data-root not found or invalid: {data_root}")

    if not (0.0 <= args.black_ratio_threshold <= 1.0):
        raise ValueError("--black-ratio-threshold must be in [0, 1]")
    if not (0 <= args.black_pixel_threshold <= 255):
        raise ValueError("--black-pixel-threshold must be in [0, 255]")

    class_vocab_path = write_class_vocab_yaml(data_root)
    print(f"class vocab written: {class_vocab_path}")

    scene_dirs = collect_scene_dirs(data_root)
    if not scene_dirs:
        print(f"No scene directories with projected_images found in: {data_root}")
        return

    print(f"discovered scenes: {len(scene_dirs)}")
    pending_scene_dirs: list[Path] = []
    skipped_scene_count = 0
    for scene in scene_dirs:
        vlm_desc_dir = scene / "vlm_desc"
        if vlm_desc_dir.is_dir():
            skipped_scene_count += 1
            print(f"[skip scene] {scene.name}: found existing {vlm_desc_dir}")
            continue
        pending_scene_dirs.append(scene)

    if not pending_scene_dirs:
        print("all scenes already processed; nothing to do.")
        return
    print(f"scenes to process: {len(pending_scene_dirs)}, skipped: {skipped_scene_count}")

    if args.prepare_only:
        for scene in pending_scene_dirs:
            (scene / "vlm_desc").mkdir(parents=True, exist_ok=True)
        print("prepare-only enabled; skip model inference.")
        return

    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"Loading model from: {args.model_path}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    for scene_dir in pending_scene_dirs:
        projected_images_dir = scene_dir / "projected_images"
        images = collect_images(projected_images_dir)
        vlm_desc_dir = scene_dir / "vlm_desc"
        vlm_desc_dir.mkdir(parents=True, exist_ok=True)
        scene_summary: list[dict[str, Any]] = []

        print(f"\n[scene] {scene_dir.name} images={len(images)}")
        for idx, image_path in enumerate(images, start=1):
            print(f"[{idx}/{len(images)}] {image_path.name}")
            output_txt_path = vlm_desc_dir / f"{image_path.stem}.vlm.txt"

            row: dict[str, Any] = {
                "image": image_path.name,
                "txt": str(output_txt_path),
                "status": "unknown",
            }

            try:
                image = Image.open(image_path).convert("RGB")
                black_ratio = calculate_black_ratio(image, args.black_pixel_threshold)
                row["black_ratio"] = round(black_ratio, 6)
                if black_ratio > args.black_ratio_threshold:
                    write_vlm_txt(output_txt_path, [])
                    row["status"] = "skipped_black"
                    row["classes"] = []
                    print("  -> skipped_black: []")
                    scene_summary.append(row)
                    continue

                raw_response = run_grounding(
                    model=model,
                    processor=processor,
                    image_path=str(image_path),
                    prompt=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                )
                classes = parse_classification_response(raw_response)
                write_vlm_txt(output_txt_path, classes)

                row["status"] = "ok"
                row["classes"] = [name for name, _ in classes]
                row["classes_en"] = [CLASS_NAME_TO_EN.get(name, name) for name, _ in classes]
                row["num_classes"] = len(classes)
                if classes:
                    cls_text = ", ".join(
                        [f"{name}/{CLASS_NAME_TO_EN.get(name, name)}:{score:.2f}" for name, score in classes]
                    )
                    print(f"  -> classes({len(classes)}): {cls_text}")
                else:
                    print("  -> classes(0): []")
            except Exception as exc:  # noqa: BLE001
                write_vlm_txt(output_txt_path, [])
                row["status"] = "error"
                row["error"] = f"{type(exc).__name__}: {exc}"
                print(f"  -> error: {row['error']}")

            scene_summary.append(row)

        summary_path = vlm_desc_dir / "scene_vlm_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "scene": scene_dir.name,
                    "class_mode": "coarse" if args.use_coarse_classes else "fine",
                    "class_count": len(CLASS_VOCAB),
                    "projected_images_dir": str(projected_images_dir),
                    "total_images": len(images),
                    "results": scene_summary,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"scene summary saved: {summary_path}")


if __name__ == "__main__":
    main()
