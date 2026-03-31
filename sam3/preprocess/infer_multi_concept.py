import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results, COLORS

model = build_sam3_image_model()
processor = Sam3Processor(model)

image = Image.open("assets/images/mesh1.png").convert("RGB")

concepts = [
    "crosswalk",
    "straight arrow",
    "left-turn arrow",
    "right-turn arrow",
    "straight or left-turn arrow",
    "straight or right-turn arrow",
    "U-turn arrow",
    "white line",
    "yellow line",
    "manhole",
    # "one arrow",
    # "two arrows",
]

all_results = {}

for concept in concepts:
    print(f"正在分割: {concept}")
    
    inference_state = processor.set_image(image)
    
    output = processor.set_text_prompt(state=inference_state, prompt=concept)
    
    
    all_results[concept] = {
        "masks": output["masks"],
        "boxes": output["boxes"],
        "scores": output["scores"]
    }
    
    print(f"找到 {len(output['masks'])} 个 {concept} 实例")



# 可选：为每个概念单独保存结果
for concept, result in all_results.items():
    if len(result["masks"]) > 0:
        plt.figure(figsize=(12, 8))
        plot_results(image, result)
        output_file = f"result_{concept}.png"
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"{concept} 的结果已保存到 {output_file}")
        plt.close()
        
        
# 保存所有结果
# def save_multi_concept_results(image, results, output_path="multi_concept_results.png"):
#     fig, axes = plt.subplots(2, 3, figsize=(18, 12))
#     axes = axes.flatten()
    
#     # 显示原图
#     axes[0].imshow(image)
#     axes[0].set_title("raw_image", fontsize=14, fontweight='bold')
#     axes[0].axis('off')
    
#     for i, (concept, result) in enumerate(results.items()):
#         # if i >= 5:  
#         #     break
            
#         axes[i+1].imshow(image)
        
#         for j, mask in enumerate(result["masks"]):
#             color = COLORS[j % len(COLORS)]
#             mask_np = mask.squeeze(0).cpu().numpy()
#             colored_mask = np.zeros((*mask_np.shape, 4))
#             colored_mask[mask_np > 0.5] = [*[c/255 for c in color], 0.4]
#             axes[i+1].imshow(colored_mask)
        
#         axes[i+1].set_title(f'{concept} ({len(result["masks"])})', 
#                            fontsize=14, fontweight='bold')
#         axes[i+1].axis('off')
    
#     for i in range(len(results) + 1, len(axes)):
#         axes[i].axis('off')
    
#     plt.tight_layout()
#     plt.savefig(output_path, bbox_inches='tight', dpi=300)
#     print(f"结果已保存到 {output_path}")
#     plt.close()

# save_multi_concept_results(image, all_results, "multi_concept_results.png")