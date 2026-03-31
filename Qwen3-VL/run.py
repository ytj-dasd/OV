import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

# 模型路径换成你下载的路径
model_path = "Qwen/Qwen3-VL-8B-Instruct" 

# 加载模型
model = AutoModelForImageTextToText.from_pretrained(
    model_path, dtype="auto", device_map="auto"
)

# model = AutoModelForImageTextToText.from_pretrained(
#     model_path,
#     torch_dtype=torch.bfloat16, 
#     attn_implementation="flash_attention_2", 
#     device_map="auto"
# )

processor = AutoProcessor.from_pretrained(model_path)

# 构造输入
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "/home/guitu/文档/vector/Qwen3-VL/cookbooks/assets/spatial_understanding/drone_cars2.png"}, # 支持本地路径或URL
            {"type": "text", "text": "有路灯的话判断一下是单臂还是双臂"},
        ],
    }
]

# 处理输入
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
images, videos, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
inputs = processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt", **video_kwargs)
inputs = inputs.to(model.device)

# 推理
generated_ids = model.generate(**inputs, max_new_tokens=128)
output_text = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

# 打印结果（需要简单处理一下去掉输入部分的prompt）
print(output_text)