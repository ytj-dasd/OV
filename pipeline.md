# Zero-Shot 多模态 OV 要素提取（VLM + SAM + Attribute）

## 摘要
按一场景一目录组织多场景点云，主流程固定为：
1. 点云投影为多视角图像，并输出点-像素对应关系。
2. Qwen3VL/GLM-V 对每张图做类别筛选，减少 SAM 调用类别数。
3. SAM 输出每图 `.npz`（包含 `masks/boxes/scores`）。
4. Task5 将 2D 实例回投到 3D 点云，做合并、冲突消解、去噪、树木后处理。
5. Task5 额外输出杆状物聚类结果（6 类杆状物）与仅杆状物聚类 LAS。
6. Task6 执行属性提取：先 BEV 全局，再 Front 场景级，输出结构化属性结果。

## 目录结构与命名规范

```text
{data_root}/
  scene_{scene_id}/
    source/
      scene_{scene_id}.las
      scene_{scene_id}_stations.json
    projected_images/
      station_{sid}_cam_{cam}.png
      station_{sid}_cam_{cam}.npz
      num_points.txt
      effective_stations.json
    vlm_desc/
      station_{sid}_cam_{cam}.vlm.txt
      scene_vlm_summary.json
    sam_mask/
      station_{sid}_cam_{cam}.npz
    fusion/
      {scene_name}_instance_seg.las
      {scene_name}_instance_seg_refined.las
      {scene_name}_instance_seg_final.las
      {scene_name}_instance_seg.npz
      {scene_name}_instance_seg_final.npz
      {scene_name}_pole_groups.npz
      {scene_name}_pole_groups_merged.las
      {scene_name}_tree_metrics.npz
      {scene_name}_instance_seg_meta.json
    attributes/
      {scene_name}_task6_front_attributes.npz
      {scene_name}_task6_front_attributes.json

  bev/
    global_instances.npz
    global_rgb.png
    las_positions.txt
    bev_attributes_global.npz
    bev_attributes_global.json

  attributes_global/
    task6_front_attributes_merged.npz
    task6_front_attributes_merged.json
```

关键命名要求：
1. 点云投影目录固定为 `projected_images`。
2. 投影图像后缀固定为 `.png`。
3. 每张图的点-像素对应关系输出文件名固定为 `station_{sid}_cam_{cam}.npz`。
4. 对应关系 `.npz` 必含键：`dist_img`, `pts_img_indices`, `pts_indices`。
5. VLM 输出后缀固定为 `.vlm.txt`（每图一个）。
6. SAM 输出后缀固定为 `.npz`（每图一个）。
7. Task5 杆状物聚类输出固定为 `*_pole_groups.npz` 和 `*_pole_groups_merged.las`。
8. Task6 BEV 输出仅写入 `benchmark/bev`；Front 输出仅写入各场景 `attributes/`。

## class_vocab 规范
实现时必须提供 `class_vocab.yaml`，并包含以下 15 类（标准类别名）：
1. 电线杆
2. 路灯杆
3. 路牌
4. 交通标志
5. 红绿灯
6. 监控
7. 行道树
8. 果壳箱
9. 消防栓
10. 电箱
11. 雕塑
12. 座椅
13. 交通锥
14. 柱墩
15. 围栏

`class_vocab.yaml` 至少包含字段：
- `id`: 连续整数 ID（建议从 1 开始，0 保留为背景）。
- `name_zh`: 中文标准名。
- `aliases`: 同义词列表（用于 VLM 文本解析归一化）。

## 关键接口与中间数据
1. `vlm_desc/*.vlm.txt`：每行格式 `class_name<TAB>confidence`，仅保留命中类别。
2. `sam_mask/*.npz`：至少包含 `masks`, `boxes`, `scores`。
3. `fusion/{scene_name}_instance_seg_final.npz` 至少包含：
`scene_instance_id`, `scene_instance_class_id`, `scene_instance_class_name`, `scene_instance_confidence`, `scene_instance_point_indices`（仅 `class_id in 7..15`）；
`pole_group_id`, `pole_group_candidate_class_ids`, `pole_group_candidate_class_names`, `pole_group_member_instance_ids`, `pole_group_point_indices`（对应杆状物 1..6 合并结果）；
`point_scene_instance_id`, `point_pole_group_id`（点级回溯索引，未归属为 `-1`）。
4. `fusion/{scene_name}_pole_groups.npz`（可选调试）至少包含：
`pole_id`, `point_indices`, `member_instance_ids`, `candidate_class_ids`, `candidate_class_names`。
5. `fusion/{scene_name}_tree_metrics.npz` 至少包含：
`scene_instance_id`, `dbh_m`, `trunk_center_x`, `trunk_center_y`, `metric_source`。
6. `bev/bev_attributes_global.npz` 与 `{scene}/attributes/*front_attributes.npz` 语义与对应 JSON 一致，仅存储形态不同。

## 实施任务

### 任务1：多场景发现与配置装载
输入：`data_root/scene_*/source/*.las` 与同目录 `*_stations.json`。
输出：场景任务列表（`scene_id`, `las_path`, `station_path`, `output_paths`）。
要求：
1. 支持批量扫描，漏扫率为 0。
2. 支持断点续跑（已有结果可跳过）。

### 任务2：点云投影批处理
输入：LAS 点云 + stations。
处理：复用现有 `pc_proj_preprocess` 逐场景生成投影结果。
输出：`projected_images/*.png`、`projected_images/*.npz`、`projected_images/num_points.txt`、`projected_images/effective_stations.json`。
要求：
1. 每张 `.png` 必有同名 `.npz`。
2. `effective_stations.json` 保存实际用于投影的相机外参（包含地形修正后的外参）。
3. `.npz` 内 `pts_indices` 不得越界。
4. 图像渲染继续使用任务2既有策略（当前软融合/前景深度约束逻辑），Task6 不引入新渲染分支。

### 任务3：GLM-V/Qwen3VL 图像描述与类别筛选
输入：`projected_images/*.png` + 固定 15 类提示词。
输出：
1. 每图 `vlm_desc/{img_stem}.vlm.txt`。
2. 场景级汇总 `vlm_desc/scene_vlm_summary.json`。
要求：
1. 每图至少输出空集或命中类集合。
2. 解析失败支持重试并记录失败原因。
3. 若某图未命中任何类别，该图后续可跳过 SAM（按策略控制）。

### 任务4：SAM 批处理接入
输入：
1. 待推理图像（来自 `projected_images`）。
2. 类别来源可二选一：
   - 默认：每图固定 15 类文本全部推理（`--use-all-classes`）。
   - 可选：仅使用 `.vlm.txt` 命中类别（`--no-use-all-classes`）。
输出：`sam_mask/{img_stem}.npz`。
要求：
1. 无效实例（面积太小或分数过低）在回投阶段过滤。
2. 场景级跳过：若场景目录下已存在 `sam_mask/`，则该场景默认视为已处理并跳过。

### 任务5：2D->3D 回投 + 合并 + 冲突归属 + 去噪 + 杆状物聚类
输入：
1. `projected_images/{img_stem}.npz`（`pts_img_indices`, `pts_indices`, `dist_img`）。
2. `projected_images/effective_stations.json`（实际用于投影的相机外参）。
3. `sam_mask/{img_stem}.npz`（`masks/boxes/scores`，以及类别字段）。
4. 场景 LAS（用于输出实例点云）。

处理：
1. 将每个 2D mask 通过 `pts_img_indices -> pts_indices` 回投到 3D 点索引集合；回投时基于 `effective_stations.json` 计算候选点的相机前向深度，仅保留落在当前像素前景深度 `+0.2m` 范围内的点，形成 scene-level 候选实例。
2. 候选实例先做类别门控：同原类别允许合并；仅 `1 电线杆` 与 `2 路灯杆` 允许跨类合并；`3-6`（路牌/交通标志/红绿灯/监控）仅同类合并；`7 行道树` 仅与 `7 行道树` 合并；`8-15` 其余类别仅同类合并。在类别门控通过后，使用双通道合并：主通道为 `point IoU >= threshold`；补充通道用于“同类实例”以及 `1<->2` 组合（`XY` 中心距离足够小）；树木暂不启用该补充通道，先保持当前 `IoU` 合并强度。候选合并后，若 merged instance 点数小于 `min-merged-points`，则直接丢弃，不进入后续处理。
3. 对保留下来的 merged instance，先按 `w_angle * confidence` 累加类别得分，确定最终类别；点归属前执行冲突删点：若某点同时属于 `1/2` 与 `3/4/5/6`，优先保留 `3/4/5/6` 并从 `1/2` 删除；若某点同时属于树和围栏，则保留树点并从围栏侧剔除；若某点同时属于树实例和其他非树实例，则保留在非树实例中，并从树实例候选点中剔除。
4. 点级冲突归属：同一点属于多个实例时，按加权总分取最大实例。
5. 空间去噪（仅非树）：仅对 `class_id != 7` 的实例做空间聚类去噪，保留最大簇；若最大簇点数小于 `denoise-min-points`，则删除该实例。树木实例在该步跳过去噪，统一由第 8/9 步树干-树冠后处理完成拆分与清理。
6. 地面点剔除（非围栏）：对除围栏外的类别执行。按实例点云 `z` 值计算 `z_low = q05(z)` 与 `z_high = q95(z)`，并使用相对高度带 `[z_low + 0.2*(z_high-z_low), z_low + 0.8*(z_high-z_low)]` 的支撑点计算水平 `bbox`，向外扩 `2cm`。最终采用保守剔除：仅删除“低高度带（`z <= z_low + 0.2*(z_high-z_low)`）且位于 `bbox` 外”的点。
7. 围栏重聚类：围栏实例不做上述 bbox 地面点剔除；先对整场景点云执行一次全局 CSF 提取地面掩码，然后仅在围栏分支删除“CSF 判地面且位于围栏实例低高度带（`z <= q05 + r*(q95-q05)`，`r` 为 `--fence-csf-low-band-ratio`）”的点，再把所有围栏非地面点合并后按 `XY` 连通重新聚类；仅保留点数不少于 `500` 且稳健高度范围 `q95(z)-q05(z) >= 0.5m` 的围栏簇，其余簇直接丢弃。
8. 树木实例后处理：仅对 `class_id=7` 执行。先从 `effective_stations.json` 提取各 station 的位置与高度；对每棵树按 `XY` 质心匹配最近 station，并用该站点 `station_z - 2.0m` 作为该树的地面参考（station 信息异常时回退到全局默认值）。随后将所有树实例中离地 `0.8-1.4m` 的高度带点汇总，在 `XY` 平面执行 DBSCAN 生成树干候选簇。半径仍按该 `0.8-1.4m` 候选簇做 TaubinSVD 圆拟合（半径+残差门控）输出并固化 `dbh_m = 2 * r` 与 `trunk_center_xy`。结果写入 `tree_metrics.npz`，并与 `scene_instance_id` 一一对应；高度改为从独立高度带（例如 `0.8-1.8m`）中补点，但补点范围仅限于“参与该候选树干簇合并的树实例”内部，且补点 `XY` 邻域半径固定为 `0.05m`；高度判定直接使用 `z_max - z_min`，并要求其大于 `tree-trunk-min-height`。当前版本已关闭主方向竖直性门控。
9. 树冠拆分与挂接：树干锚点生成后，若某树实例的高度带点同时落入多个树干锚点，则按整实例点到各树干中心的 `XY` 最近距离拆分树冠；若仅落入一个树干锚点，则保留为单棵树；若没有树干锚点，则该实例记为“树冠待定”。对每个“树冠待定”实例，使用其所有点的水平质心，找到最近树干中心；若距离小于 `4m`，则把该树冠实例并入对应树木，否则丢弃该树冠待定实例（不进入 `final.las`）。树木挂接完成后，对每棵树实例再做一次 3D DBSCAN（`--tree-final-denoise-eps`，默认 `0.5m`），仅保留最大连通簇，以去除误投影到远处墙体等离散噪点。 
10. 杆状物聚类（新增）：聚类对象是“点”，不是“实例中心”。从最终实例中提取 `class_id in {1,2,3,4,5,6}` 的全部点，投到 XY 平面执行 DBSCAN。默认参数：`eps=0.3`, `min_samples=10`。每个簇记为一个 `pole_group`，汇总 `candidate_class_ids`（来自与簇点相交的成员实例类别）。
11. 杆状物聚类 LAS 导出（新增）：仅保存杆状物点。按 `pole_group_id` 着色（合并后的杆状物实例着色）。`classification=1`，`cls_id=1`，新增 extra dim：`pole_group_id`、`has_cls_1`、`has_cls_2`、`has_cls_3`、`has_cls_4`、`has_cls_5`、`has_cls_6`。其中 `has_cls_k`（`k=1..6`）规则为：若该合并实例（`pole_group`）包含类别 `k`，则写 `1`，否则写 `0`。
12. `instance_seg_final.npz` 重组（新增约束）：`scene_instance` 仅保留 `7..15` 类（不再保留 `1..6` 的原始 scene_instance 记录）。`1..6` 仅通过 `pole_group` 表达。Task6 回溯点集时统一从该文件读取：`scene_instance_point_indices` 或 `pole_group_point_indices`。

输出（每场景）：
1. `fusion/{scene_name}_instance_seg.las`：主流程实例点 LAS。
2. `fusion/{scene_name}_instance_seg_refined.las`：地面点剔除 + 围栏重聚类后结果。
3. `fusion/{scene_name}_instance_seg_final.las`：树木后处理后的最终结果。
4. `fusion/{scene_name}_instance_seg_final.npz`：统一封装 `scene_instance(7..15)` + `pole_group(1..6)`，并携带点级回溯索引。
5. `fusion/{scene_name}_tree_metrics.npz`（新增）：每棵树 `dbh_m` 与 `trunk_center_xy`。
6. `fusion/{scene_name}_pole_groups_merged.las`（新增）：仅杆状物聚类点云，按 `pole_group_id` 着色，并携带 `has_cls_1..has_cls_6` 六个二值字段。
7. 现有调试文件继续保留：`*_fence_csf_ground.las`、`*_instance_seg_tree_pre_denoise.las`、`*_instance_seg_tree_trunks_*.las`、`*_instance_seg_meta.json`。

默认场景级跳过：
- 若 `fusion/` 已存在且非空，且未启用 `--overwrite`，则直接跳过该场景。

示例（保留现有 Task5 完整参数，不删减；仅新增杆状物聚类参数）：
```bash
python ImgProject/pipeline/task5_scene_instance_seg.py \
  --data-root benchmark \
  --iou-threshold 0.25 \
  --merge-xy-distance 0.3 \
  --fov-deg 90 \
  --min-mask-points 100 \
  --min-merged-points 500 \
  --denoise-eps 0.35 \
  --denoise-min-points 500 \
  --denoise-dbscan-min-samples 5 \
  --ground-z-quantile 0.05 \
  --ground-support-height 0.20 \
  --ground-bbox-expand 0.02 \
  --ground-support-top-ratio 0.80 \
  --fence-recluster-eps 0.10 \
  --fence-min-cluster-points 500 \
  --fence-min-height 0.50 \
  --fence-dbscan-min-samples 5 \
  --fence-csf-rigidness 3 \
  --fence-csf-class-threshold 0.35 \
  --fence-csf-low-band-ratio 0.15 \
  --tree-trunk-band-min 0.80 \
  --tree-trunk-band-max 1.40 \
  --tree-trunk-height-band-min 0.80 \
  --tree-trunk-height-band-max 1.80 \
  --tree-trunk-dbscan-eps 0.05 \
  --tree-trunk-dbscan-min-samples 10 \
  --tree-trunk-min-points 50 \
  --tree-trunk-min-height 0.50 \
  --tree-trunk-max-radius 0.30 \
  --tree-trunk-max-residual 0.04 \
  --tree-crown-attach-distance 4.0 \
  --tree-final-denoise-eps 0.50 \
  --pole-cluster-eps 0.3 \
  --pole-cluster-min-samples 10 \
  --no-save-tree-pre-denoise-las \
  --save-tree-trunk-anchors \
  --save-pole-groups-las
```

### 任务5.1：汇总各场景 Final LAS
输入：`benchmark/{scene_name}/fusion/{scene_name}_instance_seg_final.las`。
输出：`benchmark/instance_seg_final_merged.las`。
要求：
1. 按场景目录顺序读取并拼接。
2. 默认保留点坐标、RGB、`classification`，尽量保留 `cls_id`。
3. 若输出已存在且未开启 `--overwrite`，直接报错避免误覆盖。

### 任务5.2：将 BEV prompt_results 转为独立 LAS
输入：
1. `sam3/benchmark/v1/prompt_results/las_positions.txt`。
2. `sam3/benchmark/v1/prompt_results/{concept}/*.npz`（如 `arrow`、`lane_line`、`manhole`）。
3. BEV 分辨率（默认 `0.02m/pixel`）。
输出：
1. `sam3/benchmark/v1/prompt_results/arrow_lane_line_manhole_z3p5.las`（示例）。

### 任务6：要素属性提取（BEV 全局 + Front 场景）
输入：
1. BEV 全局输入（固定）：
   - `benchmark/bev/global_instances.npz`
   - `benchmark/bev/global_rgb.png`
   - `benchmark/bev/las_positions.txt`
2. Front 场景输入：
   - `benchmark/{scene}/fusion/{scene}_instance_seg_final.npz`
   - `benchmark/{scene}/fusion/{scene}_tree_metrics.npz`
   - 场景 LAS + `projected_images/*.npz` + `effective_stations.json`

主流程：
1. `run_bev_global()`：全图裁剪、井盖属性提取与几何计算，输出 BEV 全局属性文件。
2. `run_front_by_scene()`：按场景生成 2 视图并做语义/几何属性提取，输出场景属性文件，再汇总全局。

实现约束：
1. 主流程文件仅做编排：`task6_scene_attribute_extract.py`。
2. GLM 调用/提示词/解析在 `task6_glm_utils.py`。
3. 几何计算在 `task6_geometry_utils.py`。

#### 任务6A：BEV 全局分支
处理：
1. 不按场景拆分，直接在全图中裁剪。
2. 井盖每实例输出两张图：局部图 + `500x500` 全局图。
3. GLM 仅判断井盖：`functional_type`, `shape`。
4. 井盖几何计算：最小外接圆，输出 `circle_center_xy`, `circle_radius_m`。

输出：
1. `benchmark/bev/bev_attributes_global.npz`
2. `benchmark/bev/bev_attributes_global.json`

#### 任务6B：Front 场景分支
处理：
1. 视角：每个目标只生成 2 张图：`0°(front)` 与 `+90°(side)`。
2. `0°` 方向定义为“最近有效站点到目标质心的水平向量”；`+90°` 为绕 `Z` 轴旋转 90°。
3. 成图策略复用任务2现有投影策略。
4. 自适应裁剪：目标占图比例固定 `0.8`。
   - `crop_w = bbox_w / 0.8`
   - `crop_h = bbox_h / 0.8`
5. 仅允许黑边补齐，不做 resize。

语义属性范围（只保留你明确需要的键）：
1. 杆状物组：
   - `contains_classes`
   - 若包含路灯杆：`arm_type`, `light_count`
   - 若包含路牌/交通标志：`sign_shape`, `sign_color`, `sign_content`
2. 树木：`tree_type`, `tree_trunk_visible`, `tree_pit`

几何属性范围：
1. 电线杆、路灯杆：`diameter_m`, `center_xy`, `height_m`
2. 路牌、交通标志、红绿灯、监控：`centroid_xy`, `height_m`
3. 行道树：读取 Task5 固化值 `dbh_m`, `trunk_center_xy`，并补 `height_m`
4. 果壳箱、电箱：`length_m`, `width_m`, `height_m`

几何计算口径：
1. 地面高程：优先 CSF 地面邻域估计，缺失时回退 `q02(z)`。
2. 高度：`height_m = q98(z - ground_z)`。
3. 直径/DBH：在离地 `0.8-1.4m` 高度带做 TaubinSVD 圆拟合，`diameter = 2r`。
4. 果壳箱/电箱长宽：XY 平面 PCA 主方向包围盒，定义 `length_m >= width_m`。

输出（场景与全局）：
1. 每场景：`benchmark/{scene}/attributes/{scene}_task6_front_attributes.npz|json`
2. 全局汇总：`benchmark/attributes_global/task6_front_attributes_merged.npz|json`

统一输出字段（NPZ/JSON 语义一致）：
- `record_id`
- `branch`（`bev`/`front`）
- `scene_name`
- `object_type`（`manhole`/`pole_group`/`scene_instance`）
- `object_id`
- `class_id`
- `candidate_class_ids`
- `semantic_attributes_json`
- `geometry_attributes_json`
- `evidence_json`
- `confidence`

回溯规则（Task6 -> 点云）：
1. Front 分支结果统一按 `(scene_name, object_type, object_id)` 回溯到同场景 `fusion/{scene}_instance_seg_final.npz`。
2. 当 `object_type=scene_instance` 时读取 `scene_instance_point_indices`。
3. 当 `object_type=pole_group` 时读取 `pole_group_point_indices`。

#### GLM 提示词示例
井盖（BEV，局部+全局）：
```text
Picture 1 是井盖局部俯视图，Picture 2 是井盖所在区域全局俯视图。
请判断：
1) Functional Type: Rain/Sewage/Electric/Telecom/Gas/Water
2) Shape: Round/Square
只输出 JSON：
{"functional_type":"...","shape":"...","confidence":0.0}
```

杆状物组（Front，front+side）：
```text
Picture 1 是同一杆状物的 front 视图，Picture 2 是 side90 视图。
候选类别仅限：[电线杆, 路灯杆, 路牌, 交通标志, 红绿灯, 监控]。
请输出包含的类别（可多选），并按命中类别补充属性：
- 路灯杆: arm_type, light_count
- 路牌/交通标志: sign_shape, sign_color, sign_content
只输出 JSON：
{
  "contains_classes": ["..."],
  "arm_type": null,
  "light_count": null,
  "sign_shape": null,
  "sign_color": null,
  "sign_content": null,
  "confidence": 0.0
}
```

树木（Front，front+side）：
```text
Picture 1 是树木 front 视图，Picture 2 是 side90 视图。
请判断 tree_type、tree_trunk_visible、tree_pit。
只输出 JSON：
{"tree_type":"...","tree_trunk_visible":"是/否","tree_pit":"是/否","confidence":0.0}
```

示例：
```bash
python ImgProject/pipeline/task6_scene_attribute_extract.py \
  --data-root benchmark \
  --task5-output-dir fusion \
  --bev-dir benchmark/bev \
  --target-fill-ratio 0.8 \
  --model-path ZhipuAI/GLM-4.6V-Flash
```
