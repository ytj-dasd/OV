import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results, COLORS

model = build_sam3_image_model()
processor = Sam3Processor(model)

image = Image.open("sam3/assets/images/mesh1.png").convert("RGB")
img_w, img_h = image.size

visual_prompts = [
    # {
    #     "box": [0.325, 0.601, 0.1, 0.63],  # [cx, cy, w, h] 归一化坐标
    #     "label": True,  
    #     "text": None,  
    # },
    # {
    #     "box": [0.325, 0.601, 0.1, 0.63],
    #     "label": True,
    #     "text": "child",
    # },
    {
        "box": [0.135, 0.038, 0.033, 0.071],  # [cx, cy, w, h] 归一化坐标
        "label": True,  
        "text": None,  
    },
]

all_results = {}

for idx, prompt_config in enumerate(visual_prompts):
    concept_name = prompt_config.get("text") or f"visual_prompt_{idx}"
    print(f"正在处理: {concept_name}")
    
    inference_state = processor.set_image(image)
    
    if prompt_config["text"]:
        processor.set_text_prompt(state=inference_state, prompt=prompt_config["text"])

    output = processor.add_geometric_prompt(
        box=prompt_config["box"],
        label=prompt_config["label"],
        state=inference_state
    )
    
    all_results[concept_name] = {
        "masks": output["masks"],
        "boxes": output["boxes"],
        "scores": output["scores"]
    }
    
    print(f"找到 {len(output['masks'])} 个实例")

for concept, result in all_results.items():
    if len(result["masks"]) > 0:
        plt.figure(figsize=(12, 8))
        plot_results(image, result)
        output_file = f"result_{concept}.png"
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"{concept} 的结果已保存到 {output_file}")
        plt.close()