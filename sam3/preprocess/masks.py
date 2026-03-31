#!/usr/bin/env python3
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# SAM3 模型相关
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results

def main():
    if len(sys.argv) < 3:
        print("Usage: python masks.py path/to/image 'text prompt'")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    prompt = sys.argv[2]

    if not img_path.exists():
        print(f"Error: {img_path} does not exist.")
        sys.exit(1)

    stem = img_path.stem

    # 1. 加载 SAM3 模型
    print("Loading SAM3 model...")
    checkpoint_path = "/home/guitu/文档/vector/sam3/model/sam3.pt"
    model = build_sam3_image_model(checkpoint_path= checkpoint_path)
    processor = Sam3Processor(model)

    # 2. 读取图片
    print(f"Loading image {img_path}...")
    image = Image.open(img_path).convert("RGB")

    # 3. 生成推理状态
    inference_state = processor.set_image(image)

    # 4. 使用文本提示生成 mask
    print(f"Running SAM3 with prompt: '{prompt}'...")
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)

    # 5. 保存可视化结果
    save_path = img_path.parent / f"{stem}_sam3_output.png"
    plot_results(image, output)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Output visualization saved to {save_path}")

    # 6. 保存 masks、boxes、scores 到 npz 文件
    tgt_keys = ["masks", "boxes", "scores"]
    tgt = {k: output[k].cpu().numpy() for k in tgt_keys}
    detail_save_path = img_path.parent / f"{stem}_sam3_output.npz"
    np.savez(detail_save_path, **tgt)
    print(f"Masks, boxes, scores saved to {detail_save_path}")

if __name__ == "__main__":
    main()
