from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import utils as task5_utils


def _to_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _to_str(value: Any, default: str = "") -> str:
    try:
        return str(value)
    except Exception:
        return str(default)


def _parse_json_field(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    text = _to_str(value).strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        return {"raw_text": text, "parse_error": True}


def _load_records_from_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _load_records_from_npz(path: Path) -> list[dict[str, Any]]:
    data = np.load(path, allow_pickle=True)
    n = 0
    for key in ("global_gid", "record_id", "scene_name", "object_id"):
        if key in data.files:
            n = max(n, int(np.asarray(data[key]).reshape(-1).shape[0]))
    if n <= 0:
        return []

    def _arr(key: str, dtype: Any = None) -> np.ndarray:
        if key not in data.files:
            return np.asarray([], dtype=dtype if dtype is not None else object)
        arr = np.asarray(data[key])
        if dtype is not None:
            try:
                arr = arr.astype(dtype, copy=False)
            except Exception:
                pass
        return arr.reshape(-1)

    record_id = _arr("record_id", object)
    branch = _arr("branch", object)
    scene_name = _arr("scene_name", object)
    scene_id = _arr("scene_id", np.int32)
    object_type = _arr("object_type", object)
    object_id = _arr("object_id", np.int32)
    class_id = _arr("class_id", np.int32)
    global_gid = _arr("global_gid", np.int32)
    confidence = _arr("confidence", np.float32)
    candidate_class_ids = _arr("candidate_class_ids", object)
    semantic_json = _arr("semantic_attributes_json", object)
    geometry_json = _arr("geometry_attributes_json", object)
    evidence_json = _arr("evidence_json", object)

    records: list[dict[str, Any]] = []
    for i in range(n):
        candidates = []
        if i < candidate_class_ids.shape[0]:
            candidates = [
                int(x) for x in np.asarray(candidate_class_ids[i], dtype=np.int32).reshape(-1)
            ]
        rec = {
            "record_id": _to_str(record_id[i]) if i < record_id.shape[0] else "",
            "branch": _to_str(branch[i]) if i < branch.shape[0] else "",
            "scene_name": _to_str(scene_name[i]) if i < scene_name.shape[0] else "",
            "scene_id": _to_int(scene_id[i]) if i < scene_id.shape[0] else -1,
            "object_type": _to_str(object_type[i]) if i < object_type.shape[0] else "",
            "object_id": _to_int(object_id[i]) if i < object_id.shape[0] else -1,
            "class_id": _to_int(class_id[i]) if i < class_id.shape[0] else -1,
            "global_gid": _to_int(global_gid[i]) if i < global_gid.shape[0] else -1,
            "candidate_class_ids": candidates,
            "confidence": float(confidence[i]) if i < confidence.shape[0] else 0.0,
            "semantic_attributes_json": _to_str(semantic_json[i]) if i < semantic_json.shape[0] else "{}",
            "geometry_attributes_json": _to_str(geometry_json[i]) if i < geometry_json.shape[0] else "{}",
            "evidence_json": _to_str(evidence_json[i]) if i < evidence_json.shape[0] else "{}",
        }
        records.append(rec)
    return records


def load_front_records(front_attrs_path: Path) -> list[dict[str, Any]]:
    path = Path(front_attrs_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Front attributes file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        records = _load_records_from_json(path)
    elif suffix == ".npz":
        records = _load_records_from_npz(path)
    else:
        raise ValueError(f"Unsupported attrs file format: {path}")

    normalized: list[dict[str, Any]] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        normalized.append(
            {
                **rec,
                "scene_id": _to_int(rec.get("scene_id", -1), -1),
                "object_id": _to_int(rec.get("object_id", -1), -1),
                "class_id": _to_int(rec.get("class_id", -1), -1),
                "global_gid": _to_int(rec.get("global_gid", -1), -1),
                "candidate_class_ids": [
                    int(x) for x in np.asarray(rec.get("candidate_class_ids", []), dtype=np.int32).reshape(-1)
                ],
                "semantic_attributes": _parse_json_field(rec.get("semantic_attributes_json", "{}")),
                "geometry_attributes": _parse_json_field(rec.get("geometry_attributes_json", "{}")),
                "evidence": _parse_json_field(rec.get("evidence_json", "{}")),
            }
        )
    return normalized


def build_gid_index(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    index: dict[int, dict[str, Any]] = {}
    for rec in records:
        gid = _to_int(rec.get("global_gid", -1), -1)
        if gid < 0:
            continue
        index[int(gid)] = rec
    return index


def query_by_global_gid(
    *,
    gid_index: dict[int, dict[str, Any]],
    global_gid: int,
) -> dict[str, Any]:
    gid = int(global_gid)
    decoded = task5_utils.decode_global_gid(gid)
    record = gid_index.get(gid)
    return {
        "global_gid": gid,
        "decoded": decoded,
        "record": record,
        "found": record is not None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task6.1: query Task6 attributes by global_gid.")
    parser.add_argument(
        "--front-attrs",
        required=True,
        help="Path to Task6 merged front attrs (.json or .npz).",
    )
    parser.add_argument(
        "--global-gid",
        type=int,
        required=True,
        help="Input global_gid.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_front_records(Path(args.front_attrs))
    gid_index = build_gid_index(records)
    result = query_by_global_gid(gid_index=gid_index, global_gid=int(args.global_gid))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

