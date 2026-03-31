import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results, COLORS

model = build_sam3_image_model()
processor = Sam3Processor(model)

# 加载源图像和目标图像
source_image = Image.open("/home/guitu/文档/vector/rs_image/JpgFiles/1750126471.036936.jpg").convert("RGB")
target_image = Image.open("/home/guitu/文档/vector/rs_image/JpgFiles/1750126504.088536.jpg").convert("RGB")

source_w, source_h = source_image.size
target_w, target_h = target_image.size

# 拼接图像：源图像在左，目标图像在右
max_h = max(source_h, target_h)
concat_image = Image.new('RGB', (source_w + target_w, max_h), (255, 255, 255))
concat_image.paste(source_image, (0, 0))
concat_image.paste(target_image, (source_w, 0))

concat_w, concat_h = concat_image.size

def pixel_box_to_concat_normalized(cx_pixel, cy_pixel, w_pixel, h_pixel, 
                                     source_w, source_h, concat_w, concat_h):
    """
    将源图像中的像素坐标框转换为拼接图像的归一化坐标
    
    Args:
        cx_pixel, cy_pixel: 框中心点在源图像中的像素坐标
        w_pixel, h_pixel: 框的宽高（像素）
        source_w, source_h: 源图像尺寸
        concat_w, concat_h: 拼接图像尺寸
    
    Returns:
        [cx, cy, w, h]: 拼接图像中的归一化坐标
    """
    # 先归一化到源图像坐标系 [0, 1]
    cx_norm_source = cx_pixel / source_w
    cy_norm_source = cy_pixel / source_h
    w_norm_source = w_pixel / source_w
    h_norm_source = h_pixel / source_h
    
    # 再转换到拼接图像的归一化坐标
    cx_concat = (cx_norm_source * source_w) / concat_w
    cy_concat = (cy_norm_source * source_h) / concat_h
    w_concat = (w_norm_source * source_w) / concat_w
    h_concat = (h_norm_source * source_h) / concat_h
    
    return [cx_concat, cy_concat, w_concat, h_concat]

source_box_pixel = pixel_box_to_concat_normalized(
    # cx_pixel=1238,      # 中心点 x 坐标（像素）
    # cy_pixel=566,      # 中心点 y 坐标（像素）
    # w_pixel=228,       # 宽度（像素）
    # h_pixel=114,        # 高度（像素）
    
    cx_pixel=1030,      # 中心点 x 坐标（像素）
    cy_pixel=1685,      # 中心点 y 坐标（像素）
    w_pixel=372,       # 宽度（像素）
    h_pixel=318,        # 高度（像素）
    source_w=source_w,
    source_h=source_h,
    concat_w=concat_w,
    concat_h=concat_h
)

visual_prompt = {
    "box": source_box_pixel,
    "label": True,
    # "text": "bench",
    "text": None,
}

print(f"正在处理跨图像检索...")
print(f"源图像尺寸: {source_w}x{source_h}")
print(f"目标图像尺寸: {target_w}x{target_h}")
print(f"拼接图像尺寸: {concat_w}x{concat_h}")

# 在拼接图像上进行推理
inference_state = processor.set_image(concat_image)

if visual_prompt["text"]:
    processor.set_text_prompt(state=inference_state, prompt=visual_prompt["text"])

output = processor.add_geometric_prompt(
    box=visual_prompt["box"],
    label=visual_prompt["label"],
    state=inference_state
)

print(f"找到 {len(output['masks'])} 个相似实例")

# 可视化结果
plt.figure(figsize=(16, 8))
plot_results(concat_image, output)

# 添加分隔线标记源图像和目标图像的边界
plt.axvline(x=source_w, color='yellow', linestyle='--', linewidth=2, label='Source|Target')
plt.legend()

output_file = "result_cross_image_retrieval.png"
plt.savefig(output_file, bbox_inches='tight', dpi=300)
print(f"结果已保存到 {output_file}")
plt.close()

# # 分析结果：区分源图像和目标图像中的检测
# source_instances = []
# target_instances = []

# for i, box in enumerate(output["boxes"]):
#     # box 格式为 [x, y, w, h] 归一化坐标
#     box_center_x = box[0] + box[2] / 2
    
#     if box_center_x < (source_w / concat_w):
#         source_instances.append(i)
#     else:
#         target_instances.append(i)

# print(f"\n源图像中检测到 {len(source_instances)} 个实例")
# print(f"目标图像中检测到 {len(target_instances)} 个实例（检索结果）")

# # 可选：单独可视化目标图像中的检索结果
# if len(target_instances) > 0:
#     target_result = {
#         "masks": [output["masks"][i] for i in target_instances],
#         "boxes": [output["boxes"][i] for i in target_instances],
#         "scores": [output["scores"][i] for i in target_instances]
#     }
    
#     plt.figure(figsize=(12, 8))
#     plot_results(target_image, target_result)
#     output_file_target = "result_target_only.png"
#     plt.savefig(output_file_target, bbox_inches='tight', dpi=300)
#     print(f"目标图像检索结果已保存到 {output_file_target}")
#     plt.close()
