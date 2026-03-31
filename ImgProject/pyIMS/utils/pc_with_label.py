#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import numpy as np
from plyfile import PlyData, PlyElement

def load_labels(npz_path: str, n_verts: int) -> np.ndarray:
    data = np.load(npz_path)
    # 自动识别常见键名
    preferred_keys = ["labels", "pred", "m2f_label", "arr_0"]
    key = None
    for k in preferred_keys:
        if k in data:
            key = k
            break
    if key is None:
        # 回退：若只有一个数组，也可取第一个
        if len(data.files) == 1:
            key = data.files[0]
        else:
            raise KeyError(
                f"在 {npz_path} 中未找到 {preferred_keys} 任一键，且包含多个数组，请指定统一键名。"
            )

    labels = np.asarray(data[key]).reshape(-1)
    if labels.shape[0] != n_verts:
        raise ValueError(
            f"标签长度({labels.shape[0]})与顶点数({n_verts})不一致。"
        )
    # 写 PLY 建议使用有符号整型（i4）。如需无符号可改为 uint16/uint32。
    return labels.astype(np.int32)

def add_field_to_vertex(vertex_recarray: np.ndarray, field_name: str, field_values: np.ndarray) -> np.ndarray:
    if field_name in vertex_recarray.dtype.names:
        raise ValueError(f"vertex 中已存在字段 '{field_name}'。")

    # 构造新 dtype（在原字段后追加）
    new_descr = list(vertex_recarray.dtype.descr) + [(field_name, '<i4')]
    new_rec = np.empty(vertex_recarray.shape, dtype=new_descr)

    # 拷贝旧字段
    for name in vertex_recarray.dtype.names:
        new_rec[name] = vertex_recarray[name]
    # 写入新字段
    new_rec[field_name] = field_values
    return new_rec

def main():
    ap = argparse.ArgumentParser(description="将 npz 的按点标签插入到 PLY 的 vertex 新字段 m2f_label 中。")
    ap.add_argument("--in_ply", required=True, help="输入 PLY 路径")
    ap.add_argument("--in_npz", required=True, help="输入 NPZ 路径（包含逐点一维标签）")
    ap.add_argument("--out_ply", required=True, help="输出 PLY 路径")
    ap.add_argument("--field", default="m2f_label", help="新增字段名（默认 m2f_label）")
    args = ap.parse_args()

    # 读取 PLY，并保留格式与元信息
    ply = PlyData.read(args.in_ply)
    if "vertex" not in {e.name for e in ply.elements}:
        print("错误：PLY 中不包含 'vertex' 元素。", file=sys.stderr)
        sys.exit(1)

    vertex = ply["vertex"]
    n_verts = len(vertex.data)

    # 读取标签并校验长度
    labels = load_labels(args.in_npz, n_verts)

    # 生成带新增字段的 vertex 记录
    new_vertex_np = add_field_to_vertex(vertex.data, args.field, labels)
    new_vertex = PlyElement.describe(new_vertex_np, "vertex")

    # 其他元素保持不变且顺序尽量一致（vertex 通常放第一个；若原来不是第一个也无伤大雅）
    other_elems = [e for e in ply.elements if e.name != "vertex"]
    new_elements = [new_vertex] + other_elems

    # 写回，尽量保持原始 ASCII/二进制、字节序、注释等
    new_ply = PlyData(
        new_elements,
        text=ply.text,                 # True=ASCII, False=binary
        byte_order=ply.byte_order,     # '<' 小端, '>' 大端, 或 None(ASCII)
        comments=ply.comments,
        obj_info=ply.obj_info
    )
    new_ply.write(args.out_ply)
    print(f"完成：写入 {args.out_ply}，新增字段 '{args.field}'，共 {n_verts} 个标签。")

if __name__ == "__main__":
    main()
