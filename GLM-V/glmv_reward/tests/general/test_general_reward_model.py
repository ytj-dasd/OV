# from glmv_reward.reward_system import RewardSystem


# reward_system = RewardSystem(config_path="/workspace/home/yuwenmeng/codes/vlm-train-config/train_config/reward_system/reward_think_answer.yaml")

# results = reward_system.get_reward(
#     prompts=["[gMASK]<sop><|user|>\n<|begin_of_image|><|end_of_image|>Please describe this image."] * 4,
#     answers=["<answer>This image depicts the life cycle of a ladybug (ladybird beetle), illustrated in six labeled stages (A to F) arranged in a circular sequence around a central letter “E”.</answer>"] * 3 + ["""<answer>This image depicts the life cycle of a ladybug (ladybird beetle), illustrated in six labeled stages (A to F) arranged in a circular sequence around a central letter “E”. Here’s a breakdown of each stage:

# ⸻

# Stages of the Ladybug Life Cycle:
# 	•	A. Egg stage:
# Ladybug eggs are small and oval-shaped, typically laid in clusters on leaves. However, in this image, A might instead represent the early larval stage due to its structure.
# 	•	B. Larva emerging:
# This stage shows a larva hatching from the egg. Ladybug larvae look very different from the adult and are elongated and spiny.
# 	•	C. Adult ladybug:
# The mature ladybug with its characteristic dome-shaped body and spotted wings.
# 	•	D. Pupa stage:
# After the larval stage, the ladybug enters the pupal stage, during which metamorphosis occurs.
# 	•	E. Central label – “E” for Egg (possibly):
# The central letter might stand for “Egg” or “Embryo,” indicating the beginning of the cycle.
# 	•	F. Growing larva:
# Shows the larva in an advanced stage before pupation, with more defined segments and legs.

# ⸻

# Life Cycle Order (Clockwise starting from E):
# 	1.	E – Egg
# 	2.	F – Larva (early to mid-stage)
# 	3.	D – Larva (late stage, pre-pupa)
# 	4.	B – Pupa
# 	5.	A – Emerging adult (from pupa)
# 	6.	C – Fully developed adult ladybug

# This life cycle is an example of complete metamorphosis, consisting of four main stages: egg → larva → pupa → adult.</answer>"""],
#     gt_answers=["<answer>This image depicts the life cycle of a ladybug (ladybird beetle), illustrated in six labeled stages (A to F) arranged in a circular sequence around a central letter “E”.</answer>"] * 4,
#     image_paths=[[['/workspace/image_general_data/yuwenmeng/yangsheng_sft_claude/图片内容识别/物体识别/TarFiles/meta_4_25.tar', 6561280, 'image']]] * 4,
#     datasources=["general_reward"] * 4,
#     debug=True,
#     return_extracted_answers=True
# )

# print(results)
