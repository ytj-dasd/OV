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
  instance_seg_final_merged.las
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
      {scene_name}_instance_seg_final_pre_pole.las
      {scene_name}_instance_seg_final.las
      {scene_name}_instance_seg.npz
      {scene_name}_instance_seg_final.npz
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
    task6_bev_crops/
      manhole_00000_local.png
      manhole_00000_global.png
    task6_bev_annotated/
      manhole_00000_global_annotated.png
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
7. Task5 杆状物聚类点集统一封装到 `*_instance_seg_final.npz`；仅额外输出 `*_pole_groups_merged.las`（点云可视化）。
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
`pole_group_diameter_m`, `pole_group_center_x`, `pole_group_center_y`, `pole_group_metric_source`（Task5 预计算的杆状物几何指标）；
`point_scene_instance_id`, `point_pole_group_id`（点级回溯索引，未归属为 `-1`）。
4. `fusion/{scene_name}_instance_seg_final.las` 至少包含 extra dim：
`cls_id`, `point_scene_instance_id`, `point_pole_group_id`；其中杆状物点统一 `classification=1` / `cls_id=1`。
5. `fusion/{scene_name}_tree_metrics.npz` 至少包含：
`scene_instance_id`, `dbh_m`, `trunk_center_x`, `trunk_center_y`, `metric_source`。
6. `benchmark/instance_seg_final_merged.las`（Task5.1）新增点级 `global_gid(int32)`，支持跨场景单键查表。
7. `bev/bev_attributes_global.npz` 与 `{scene}/attributes/*front_attributes.npz` 语义与对应 JSON 一致，仅存储形态不同。

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
2. `effective_stations.json` 必须保存实际用于投影的相机外参；若启用了地形高度修正，这里保存修正后的外参。
3. `.npz` 内 `pts_indices` 不得越界。
4. 图像渲染采用两层逻辑：
   - 点像素对应关系 `.npz` 仍按“当前像素最近点 + buffer 扩张”生成，保持 2D->3D 回投语义不变。
   - 图像赋色改为 soft-splat：每个点以连续投影坐标向邻域像素发核，像素按空间距离与深度一致性对候选点做加权归一化。
5. 像素前景深度优先使用“当前像素自己的硬命中点深度”；如果该像素没有硬命中点，则用邻域候选点中的最小深度作为前景深度，再执行 soft-splat 赋色与补连续性。
6. 因此前景归属和颜色平滑解耦：前景边缘不被邻域核直接抢占，空像素则由邻域点在深度约束下自然补全。
7. 旧 render_img_gpu 是硬覆盖，谁最近就直接拿谁的颜色，边缘容易出现斑块和发硬的扩张感。新方案是软融合，过渡更自然。旧方案的补洞发生在像素层，已经丢失了子像素信息。新方案从连续 u/v 开始，天然保留子像素几何信息。

### 任务3：GLM-V/Qwen3VL 图像描述与类别筛选
输入：`projected_images/*.png` + 固定 15 类提示词。
输出：
1. 每图 `vlm_desc/{img_stem}.vlm.txt`。
2. 场景级汇总 `vlm_desc/scene_vlm_summary.json`。
要求：
1. 每图至少输出空集或命中类集合。
2. 解析失败支持重试并记录失败原因。
3. 若某图未命中任何类别，该图后续可跳过 SAM（按策略控制）。
```bash
# Qwen3-VL
python Qwen3-VL/infer/batch_scene_classify.py \
  --data-root benchmark \
  --model-path Qwen/Qwen3-VL-8B-Thinking \
  --max-new-tokens 512

# GLM-V
python GLM-V/inference/batch_scene_classify_glm.py \
  --data-root benchmark \
  --model-path ZhipuAI/GLM-4.6V-Flash \
  --max-new-tokens 1024
```


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
```bash
python sam3/preprocess/batch_scene_sam_from_vlm.py \
  --data-root benchmark \
  --alpha 0.45 \
  --checkpoint sam3/model/sam3.pt \
  --device cuda \
  --sam-resolution 1008 \
  --confidence-threshold 0

# 如需额外按类别保存单独结果（便于排查）
# python sam3/preprocess/batch_scene_sam_from_vlm.py --data-root benchmark --save-per-class-artifacts
```


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
10. 杆状物聚类：聚类对象是“点”，不是“实例中心”。从最终实例中提取 `class_id in {1,2,3,4,5,6}` 的全部点，在 3D 空间执行 DBSCAN。默认参数：`eps=0.3`, `min_samples=10`。每个簇记为一个 `pole_group`，汇总 `candidate_class_ids`（来自与簇点相交的成员实例类别）。新增三级过滤：聚类后若簇点数 `< 100` 直接丢弃；若该簇高度差 `z_max-z_min < 0.5m` 直接丢弃（抑制地面箭头等误分割成杆状物）；若拟合直径 `diameter_m > 5.0m` 直接丢弃（抑制明显异常杆体）。
11. 伪树筛除：先对每个 `pole_group` 按树木同口径（最近站点地面 + `0.8-1.4m` 高度带 + TaubinSVD）计算 `diameter_m/center_xy`。再将“有有效树干锚点”的树实例与 `pole_group` 做唯一匹配；匹配必须同时满足：`center_xy` 距离 `<= 0.35m` 且 `|tree_dbh - pole_diameter| <= 0.3m`。满足条件时判为伪树并直接删除该树实例（不并入杆状物，避免树冠噪点带入杆状物）。
12. 杆状物聚类 LAS 导出：仅保存杆状物点。按 `pole_group_id` 着色（合并后的杆状物实例着色）。`classification=1`，`cls_id=1`，新增 extra dim：`pole_group_id`、`has_cls_1`、`has_cls_2`、`has_cls_3`、`has_cls_4`、`has_cls_5`、`has_cls_6`。其中 `has_cls_k`（`k=1..6`）规则为：若该合并实例（`pole_group`）包含类别 `k`，则写 `1`，否则写 `0`。最终写盘顺序（调整）：`tree_metrics.npz`、`instance_seg_final.npz`、`instance_seg_final.las` 均放在“伪树筛除”之后统一写出，保证三者一致。`instance_seg_final.npz` 重组（新增约束）：`scene_instance` 仅保留 `7..15` 类（不再保留 `1..6` 的原始 scene_instance 记录）。`1..6` 仅通过 `pole_group` 表达。Task6 回溯点集时统一从该文件读取：`scene_instance_point_indices` 或 `pole_group_point_indices`。

输出（每场景）：
1. `fusion/{scene_name}_instance_seg.las`：主流程实例点 LAS。
2. `fusion/{scene_name}_instance_seg_refined.las`：地面点剔除 + 围栏重聚类后结果。
3. `fusion/{scene_name}_instance_seg_final_pre_pole.las`：树木后处理后、杆状物 compact 化之前的最终点云（保留原类）。
4. `fusion/{scene_name}_instance_seg_final.las`：与 compact NPZ 对齐的最终点云（`scene_instance(7..15)` + `pole_group`，杆状物统一类 `1`）。
5. `fusion/{scene_name}_instance_seg_final.npz`：统一封装 `scene_instance(7..15)` + `pole_group(1..6)`，并携带点级回溯索引。
6. `fusion/{scene_name}_tree_metrics.npz`：每棵树 `dbh_m` 与 `trunk_center_xy`。
7. `fusion/{scene_name}_pole_groups_merged.las`：仅杆状物聚类点云，按 `pole_group_id` 着色，并携带 `has_cls_1..has_cls_6` 六个二值字段。
8. 现有调试文件继续保留：`*_scene_csf_ground.las`、`*_instance_seg_tree_pre_denoise.las`、`*_instance_seg_tree_trunks_*.las`、`*_instance_seg_meta.json`。

默认场景级跳过：
- 若 `fusion/` 已存在且非空，且未启用 `--overwrite`，则直接跳过该场景。
- 支持场景级并行：`--num-workers` 默认 `2`；若内存紧张可改为 `1`（串行）。

示例（保留现有 Task5 完整参数，不删减；新增并行与杆状物相关参数）：
```bash
python ImgProject/pipeline/task5_scene_instance_seg.py \
  --data-root benchmark \
  --num-workers 2 \
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
  --pole-min-cluster-points 100 \
  --pole-min-height-diff 0.50 \
  --tree-pole-center-merge-distance 0.35 \
  --tree-pole-diameter-diff-max 0.30 \
  --pole-max-diameter-m 5.0 \
  --no-save-tree-pre-denoise-las \
  --save-tree-trunk-anchors \
  --save-pole-groups-las
```

### 任务5.1：汇总各场景 Final LAS
输入：`benchmark/{scene_name}/fusion/{scene_name}_instance_seg_final.las`。
输出：`benchmark/instance_seg_final_merged.las`。
要求：
1. 按场景目录顺序读取并拼接。
2. 默认保留点坐标、RGB、`classification`，并尽量保留 `cls_id`。
3. 新增点级键 `global_gid(int32)`，用于跨场景单键查表。
4. `global_gid` 位编码规则（v1）：
   - `scene_id`：10 bit，`[0, 1023]`
   - `type_code`：1 bit，`0=scene_instance`, `1=pole_group`
   - `object_id`：14 bit，`[0, 16383]`
   - `global_gid = (scene_id << 15) | (type_code << 14) | object_id`
5. `global_gid` 由 `point_scene_instance_id` / `point_pole_group_id` 计算，不再依赖额外 index 文件。
6. 若输出已存在且未开启 `--overwrite`，直接报错避免误覆盖。

示例：
```bash
python ImgProject/pipeline/task5_merge_instance_seg_final_las.py \
  --data-root benchmark \
  --output-name instance_seg_final_merged.las
```

### 任务5.2：将 BEV prompt_results 转为独立 LAS
输入：
1. `sam3/benchmark/v1/prompt_results/las_positions.txt`。
2. `sam3/benchmark/v1/prompt_results/{concept}/*.npz`（如 `arrow`、`lane_line`、`manhole`）。
3. BEV 分辨率（默认 `0.02m/pixel`）。
输出：
1. `sam3/benchmark/v1/prompt_results/arrow_lane_line_manhole_z3p5.las`（示例）。

### 任务5.3：按后缀合并场景 LAS
输入：
1. `benchmark/{scene_name}/fusion/*{suffix}`（默认后缀：`_pole_groups_merged.las`，即匹配如 `road1-1_pole_groups_merged.las`）。

输出：
1. `benchmark/pole_groups_merged_all.las`（默认，可通过 `--output-name` 自定义）。

要求：
1. 默认优先读取 `{scene_name}{suffix}`；若不存在则回退到该场景 `fusion/` 下任意 `*{suffix}` 的第一个匹配文件。
2. 场景间 LAS schema 必须一致（point format + extra dims），不一致直接报错。
3. 若输出已存在且未开启 `--overwrite`，直接报错避免误覆盖。

示例（默认合并 `*_pole_groups_merged.las`）：
```bash
python ImgProject/pipeline/task5_merge_scene_las_by_suffix.py \
  --data-root benchmark
```

示例（改为合并 `*_instance_seg_final.las`）：
```bash
python ImgProject/pipeline/task5_merge_scene_las_by_suffix.py \
  --data-root benchmark \
  --suffix _instance_seg_final.las \
  --output-name instance_seg_final_all.las
```


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

运行控制参数（Task6 主流程）：
1. 分支选择：`--run-branch`
   - `both`：同时跑任务6A+6B（默认）
   - `bev` / `a`：仅跑任务6A（BEV）
   - `front` / `b`：仅跑任务6B（Front）
2. 阶段选择：`--run-stage`
   - `both`：语义 VLM + 几何属性都执行（默认）
   - `geometry`：仅计算几何属性，不调用 VLM
   - `vlm`：仅调用 VLM，几何属性留空
3. VLM 后端选择：`--vlm-backend glm|qwen|gemma`，默认 `glm`。
4. 默认模型：`glm` 使用 `ZhipuAI/GLM-4.6V-Flash`；`qwen` 使用 `Qwen/Qwen3-VL-8B-Instruct`；`gemma` 使用 `google/gemma-4-E4B-it`。
5. `--model-path` 可覆盖默认模型路径。
6. 采样参数默认按后端自动切换（命令行显式传入会覆盖）：
   - `glm`: `temperature=0.2`, `repetition_penalty=1.1`, `top_p=0.8`, `top_k=2`
   - `qwen`: `temperature=0.2`, `repetition_penalty=1.1`, `top_p=0.8`, `top_k=2`
   - `gemma`: `temperature=1.0`, `repetition_penalty=1.0`, `top_p=0.95`, `top_k=64`
7. `--disable-vlm` 可跳过语义推理；旧参数 `--disable-glm` 仍兼容。

#### 任务6A：BEV 全局分支
处理：
1. 不按场景拆分，直接在全图中裁剪。
2. 井盖每实例输出两张图：局部图 + `500x500` 全局图。
3. GLM 仅判断井盖：`functional_type`, `shape`。
4. 井盖几何计算：最小外接圆，输出 `circle_center_xy`, `circle_radius_m`。
5. 新增每实例标注全局图：叠加 `functional_type/shape/circle_radius_m/circle_center_xy`，并绘制圆心与外接圆。

输出：
1. `benchmark/bev/bev_attributes_global.npz`
2. `benchmark/bev/bev_attributes_global.json`
3. `benchmark/bev/task6_bev_crops/manhole_{id}_local.png`
4. `benchmark/bev/task6_bev_crops/manhole_{id}_global.png`
5. `benchmark/bev/task6_bev_annotated/manhole_{id}_global_annotated.png`

#### 任务6B：Front 场景分支
处理：
1. 渲染范围：只渲染 `pole_group` 与 `scene_instance class_id=7(树木)`；其他 `scene_instance` 仅计算几何属性，不渲染图像。
2. 视角：每个目标只生成 2 张图：`0°(front)` 与 `+90°(side)`。
3. `0°` 方向定义为“最近有效站点到目标质心的水平向量”；`+90°` 为绕 `Z` 轴旋转 90°。
4. 成图策略：从源 LAS 提取实例点（含 RGB）后，使用 Task2 同源 `pc2img_soft` 进行实例级重投影。
5. 相机参数：固定 `FOV=90°`。`front` 视角通过迭代调整相机距离，优先保证目标完整投影在画布内（上下两端不能被截断），再尽量让完整投影范围接近占比 `0.8`；`side` 视角以 `front` 距离为初值，仅在完整投影仍出画时向外放大距离，不再单独追求占比 `0.8`。
6. `side` 兜底：若 `+90°` 失败，尝试 `-90°`；若仍失败，记录 `front-only` 状态并继续输出该对象属性，不因 side 失败丢记录。
7. 自适应裁剪：裁剪 bbox 使用全体实例点的几何投影范围，而不是渲染命中的像素范围；目标完整进入画布后，裁剪占图比例固定 `0.8`。
   - `crop_w = bbox_w / 0.8`
   - `crop_h = bbox_h / 0.8`
8. 允许黑边补齐；若裁剪后长边大于 `1024`，按比例下采样到 `1024`。
9. Front 渲染图只保留原始渲染图，不叠加 `semantic_attributes_json`，也不额外输出 `*_annotated.png`。

语义属性范围（只保留你明确需要的键）：
1. 杆状物组：
   - `contains_classes`
   - 若包含路灯杆：`arm_type`, `light_count`
   - 若包含路牌/交通标志：`sign_shape`, `sign_color`, `sign_content`
2. 树木：`tree_type`, `tree_trunk_visible`

几何属性范围：
1. 杆状物 6 类（电线杆、路灯杆、路牌、交通标志、红绿灯、监控）：`center_xy`, `diameter_m`, `height_m`
2. 行道树：`trunk_center_xy`, `dbh_m`, `height_m`
3. 果壳箱、电箱、消防栓、座椅、交通锥、柱墩：`center_xy`, `height_m`
4. 围栏：`control_points_xy`（5 控制点，含首尾点）, `height_m`

几何计算口径：
1. 地面高程：Task6 不再重新运行 CSF；直接复用 Task5 输出 `fusion/{scene}_scene_csf_ground.las` 作为地面点集，在目标 `XY` 邻域（默认半径 `2.0m`）取地面 `z` 的 `q10` 作为 `ground_z`；邻域点不足时回退 `q02(z)`。
2. 高度：`height_m = q98(z - ground_z)`。
3. 杆状物直径与中心：`pole_group` 的 `diameter_m/center_xy` 优先读取 Task5 预计算字段；若 `center_xy` 缺失则回退整实例质心，若 `diameter_m` 缺失则保持空值（`null`）。
4. 中心点类（果壳箱、电箱、消防栓、座椅、交通锥、柱墩）使用中部高度带质心：在实例 `q05~q95` 高程跨度内取 `30%~70%` 高度带求 `center_xy`；点数不足时回退全点质心。
5. 围栏控制点：先取中部高度带（默认 `30%~70%`），对 `XY` 栅格降采样后构图，提取最长路径中心线，再按弧长等距采样 5 点（`0%/25%/50%/75%/100%`，含首尾点）；若图路径不稳定则回退 PCA 中心线采样。

输出（场景与全局）：
1. 每场景：`benchmark/{scene}/attributes/{scene}_task6_front_attributes.npz|json`
2. 全局汇总：`benchmark/attributes_global/task6_front_attributes_merged.npz|json`

统一输出字段（NPZ/JSON 语义一致）：
- `record_id`
- `branch`（`bev`/`front`）
- `scene_name`
- `scene_id`（front 记录）
- `object_type`（`manhole`/`pole_group`/`scene_instance`）
- `object_id`
- `global_gid`（front 记录，和 Task5.1 一致编码）
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
4. 跨场景查询优先使用 `global_gid`；可通过解码得到 `(scene_id, object_type, object_id)` 后定位属性记录。

### 任务6.1：global_gid 查表小界面
输入：
1. `benchmark/attributes_global/task6_front_attributes_merged.json|npz`
2. 用户输入 `global_gid`

输出：
1. 命中记录的完整属性展示（语义/几何/evidence）
2. 解码信息：`scene_id`, `object_type`, `object_id`

实现：
1. `ImgProject/pipeline/task6_1_gid_lookup.py`：纯查表逻辑（可 CLI 查询）
2. `ImgProject/pipeline/task6_1_gid_lookup_gui.py`：`tkinter` 小界面（查询/清空/复制）

示例：
```bash
python ImgProject/pipeline/task6_1_gid_lookup.py \
  --front-attrs benchmark/attributes_global/task6_front_attributes_merged.json \
  --global-gid 409945
```

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
候选类别参考（非限制）：[电线杆, 路灯杆, 路牌, 交通标志, 红绿灯, 监控]。
最终 contains_classes 允许从以下 6 类中多选：[电线杆, 路灯杆, 路牌, 交通标志, 红绿灯, 监控]，不要受候选类别限制。
请输出包含的类别（可多选），并按命中类别补充属性：
- 路灯杆: arm_type, light_count；若包含路灯杆，arm_type 只能输出 single_arm、double_arm、no_arm 中的一个（single_arm=单臂，double_arm=双臂，no_arm=无臂）
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
请判断 tree_type、tree_trunk_visible。
只输出 JSON：
{"tree_type":"...","tree_trunk_visible":"是/否","confidence":0.0}
```

示例：
```bash
python ImgProject/pipeline/task6_scene_attribute_extract.py \
  --data-root benchmark \
  --run-branch front \
  --run-stage geometry \
  --task5-output-dir fusion \
  --bev-dir benchmark/bev \
  --vlm-backend glm \
  --model-path GLM-V/ZhipuAI/GLM-4.6V-Flash
```

Qwen 语义推理示例：
```bash
python ImgProject/pipeline/task6_scene_attribute_extract.py \
  --data-root benchmark \
  --run-branch front \
  --run-stage both \
  --task5-output-dir fusion \
  --bev-dir benchmark/bev \
  --vlm-backend qwen \
  --model-path Qwen3-VL/Qwen/Qwen3-VL-8B-Instruct
```

Gemma 语义推理示例：
```bash
python ImgProject/pipeline/task6_scene_attribute_extract.py \
  --data-root benchmark \
  --run-branch front \
  --run-stage both \
  --task5-output-dir fusion \
  --bev-dir benchmark/bev \
  --vlm-backend gemma \
  --model-path gemma4/gemma-4-E4B-it
```
