# Zero-Shot 多场景点云实例分割实施计划（VLM + SAM）

## 摘要
按一场景一目录组织多场景点云，流程固定为：
1. 点云投影为多视角图像，并输出点-像素对应关系。
2. Qwen3VL 对每张图做目标类别筛选，减少 SAM 调用类别数。
3. SAM 的 `.npz` 结果（包含全部 `mask/box/score`），不做适配层。
4. 将 2D 实例回投到 3D 点云，做多视角实例合并、类别加权、点级冲突消解。
5. 场景内实例去噪后，执行跨场景边界实例合并，输出全局实例结果。

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
    vlm_desc/
      station_{sid}_cam_{cam}.vlm.txt
      scene_vlm_summary.json
    sam_mask/
      station_{sid}_cam_{cam}.npz
    fusion/
      {scene_name}_instance_seg.las
      {scene_name}_instance_seg.npz
      {scene_name}_instance_seg_meta.json

  cross_scene/
    boundary_candidates.json
    global_instances.npz
    global_point_assignment.npz
    global_instance_meta.json
```

关键命名要求：
1. 点云投影目录固定为 `projected_images`。
2. 投影图像后缀固定为 `.png`。
3. 每张图的点-像素对应关系输出文件名固定为 `station_{sid}_cam_{cam}.npz`。
4. 对应关系 `.npz` 必含键：`dist_img`, `pts_img_indices`, `pts_indices`。
5. VLM 输出后缀固定为 `.vlm.txt`（每图一个）。
6. SAM 输出后缀固定为 `.npz`（每图一个）。

## class_vocab 规范
实现时必须提供 `class_vocab.yaml`，并包含且仅包含以下 14 类（标准类别名）：
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
- `name_zh`: 中文标准名（上面 15 类之一）。
- `aliases`: 同义词列表（用于 VLM 文本解析归一化）。

## 关键接口与中间数据
1. `vlm_desc/*.vlm.txt`：每行格式 `class_name<TAB>confidence`，仅保留命中类别。
2. `sam_mask/*.npz`：直接使用你现有格式，至少能读取：
`masks`, `boxes`, `scores`。
3. `fusion/{scene_name}_instance_seg.npz` 至少包含：
`scene_instance_id`, `class_id`, `class_name`, `confidence`, `point_indices`。
4. 同一 `.npz` 同时包含点级归属：
`point_instance_id (N,int32)`, `point_confidence (N,float32)`。

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
```bash
    uv run main.py
    SOFT_SPLAT_RADIUS_MIN/MAX 控制核大小
    SOFT_SPLAT_DEPTH_STEP 控制多远开始增大核
    SOFT_SPLAT_SIGMA_SCALE 控制空间扩散强度
    SOFT_SPLAT_DEPTH_GATE_BASE/SCALE 控制深度门控松紧
    SOFT_SPLAT_DEPTH_SIGMA_SCALE 控制通过门控后的深度软融合强度
```

### 任务3：GLM-V/Qwen3VL 图像描述与类别筛选
输入：`projected_images/*.png` + 固定 14 类提示词。
输出：
1. 每图 `vlm_desc/{img_stem}.vlm.txt`。
2. 场景级汇总 `vlm_desc/scene_vlm_summary.json`。
要求：
1. 每图至少输出空集或命中类集合。
2. 解析失败支持重试并记录失败原因。
3. 若某图未命中任何类别，该图后续跳过 SAM。
4. 场景级跳过：若场景目录下已存在 `vlm_desc/`，则该场景默认视为已处理并跳过。
```bash
# Qwen3-VL
python Qwen3-VL/infer/batch_scene_classify.py \
  --data-root benchmark \
  --model-path Qwen/Qwen3-VL-2B-Thinking \
  --max-new-tokens 1024 \
  --use-coarse-classes

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
3. 你的 SAM 推理代码（输出单个 `.npz`，含 `masks/boxes/scores`）。
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
  --confidence-threshold 0.3 \
  --use-all-classes

# 如需额外按类别保存单独结果（便于排查）
# python sam3/preprocess/batch_scene_sam_from_vlm.py --data-root benchmark --save-per-class-artifacts
# 如需回退为“仅使用 vlm_desc 命中类别”模式
# python sam3/preprocess/batch_scene_sam_from_vlm.py --data-root benchmark --no-use-all-classes
```

### 任务5：2D->3D 回投 + IoU合并 + 点冲突归属 + 去噪
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
5. 空间去噪：每实例做空间聚类，仅保留最大簇；若最大簇点数小于 `denoise-min-points`，则删除该实例。
6. 地面点剔除（非围栏）：对除围栏外的类别执行。按实例点云 `z` 值计算 `z_low = q05(z)` 与 `z_high = q95(z)`，并使用相对高度带 `[z_low + 0.2*(z_high-z_low), z_low + 0.8*(z_high-z_low)]` 的支撑点计算水平 `bbox`，向外扩 `2cm`。最终采用保守剔除：仅删除“低高度带（`z <= z_low + 0.2*(z_high-z_low)`）且位于 `bbox` 外”的点。
7. 围栏重聚类：围栏实例不做上述 bbox 地面点剔除；在重聚类前，对每个 `class_id=15` 围栏实例单独执行一次 CSF 去地面（实例级，默认开启），再把所有围栏非地面点合并后按 `XY` 连通重新聚类；仅保留点数不少于 `500` 且稳健高度范围 `q95(z)-q05(z) >= 0.5m` 的围栏簇，其余簇直接丢弃。
8. 树木实例后处理：仅对 `class_id=7` 执行。先从 `effective_stations.json` 提取各 station 的位置与高度；对每棵树按 `XY` 质心匹配最近 station，并用该站点 `station_z - 2.0m` 作为该树的地面参考（station 信息异常时回退到全局默认值）。随后将所有树实例中离地 `0.8-1.4m` 的高度带点汇总，在 `XY` 平面执行 DBSCAN 生成树干候选簇。半径仍按该 `0.8-1.4m` 候选簇做 TaubinSVD 圆拟合（半径+残差门控）；高度改为从独立高度带（例如 `0.8-1.8m`）中补点，但补点范围仅限于“参与该候选树干簇合并的树实例”内部，且补点 `XY` 邻域半径固定为 `0.05m`；高度判定直接使用 `z_max - z_min`，并要求其大于 `tree-trunk-min-height`。当前版本已关闭主方向竖直性门控。
9. 树冠拆分与挂接：树干锚点生成后，若某树实例的高度带点同时落入多个树干锚点，则按整实例点到各树干中心的 `XY` 最近距离拆分树冠；若仅落入一个树干锚点，则保留为单棵树；若没有树干锚点，则该实例记为“树冠待定”。对每个“树冠待定”实例，使用其所有点的水平质心，找到最近树干中心；若距离小于 `4m`，则把该树冠实例并入对应树木，否则丢弃该树冠待定实例（不进入 `final.las`）。


输出（每场景）：
1. `fusion/{scene_name}_instance_seg.las`：当前 Task5 主流程输出，只包含实例点，按场景内实例 ID 随机着色。
2. `fusion/{scene_name}_instance_seg_refined.las`：在主流程结果基础上，再经过地面点剔除和围栏重聚类后的 refined LAS。
3. `fusion/{scene_name}_instance_seg_final.las`：在 refined 结果基础上，再经过树木树干/树冠后处理后的最终 LAS。
4. `fusion/{scene_name}_instance_seg.npz`：主流程实例 ID、类别、置信度、实例点索引，以及点级唯一归属结果。
5. `fusion/{scene_name}_instance_seg_final.npz`：最终树木后处理结果对应的实例点索引、类别和点级唯一归属结果。
6. `fusion/{scene_name}_instance_seg_tree_trunks_height.las`：树干候选经过“稳健高度范围”过滤后的阶段 LAS。
7. `fusion/{scene_name}_instance_seg_tree_trunks_radius.las`：在高度阶段基础上，再经过“水平稳健半径”过滤后的阶段 LAS。
   以上 2 个 trunk LAS 都会写入额外字段：`trunk_height`、`trunk_radius`、`trunk_dir_x`、`trunk_dir_y`、`trunk_dir_z`、`trunk_dot_z`（以及 `cls_id`）。
8. `fusion/{scene_name}_instance_seg_meta.json`：去噪/过滤统计日志。

默认场景级跳过：
- 若 `fusion/` 已存在且非空，且未启用 `--overwrite`，则直接跳过该场景。

```bash
python ImgProject/pipeline/task5_scene_instance_seg.py \
  --data-root benchmark \
  --iou-threshold 0.25 \
  --merge-xy-distance 0.3 \
  --fov-deg 90 \
  --min-mask-points 100 \
  --min-merged-points 500 \
  --denoise-eps 0.25 \
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
  --save-tree-trunk-anchors
```


### 任务6：跨场景边界实例合并
处理：
1. 仅在相邻场景 bbox overlap + margin 区域生成候选对。
2. 候选实例按空间聚类与点云 IoU 判定是否合并。
3. 合并后重建全局实例 ID。
输出：
1. `cross_scene/boundary_candidates.json`
2. `cross_scene/global_instances.npz`
3. `cross_scene/global_point_assignment.npz`
4. `cross_scene/global_instance_meta.json`
要求：
1. 避免全局 N² 比对，只处理边界候选。
2. 全局实例 ID 稳定可复现。
```bash
python ImgProject/pipeline/task8_cross_scene_merge.py \
  --data-root benchmark \
  --boundary-margin 2.0
```

## 假设与默认值
1. VLM 固定使用 GLM。
2. SAM 输出固定为每图一个 `.npz`，且可直接读取 `masks/boxes/scores`。
3. 多场景 LAS 在同一坐标系下，可直接进行跨场景空间计算。
4. 合并与聚类阈值统一放到配置中可调。

## 围栏 CSF 备选方案（后续）
若实例级 CSF 在遮挡严重场景下仍不足以去掉“地面桥接”，后续可切换为“场景级全局 CSF”：每场景先跑一次全局地面分割，再在围栏分支按全局 ground mask 删除地面点，再做围栏重聚类。该方案通常更稳定，但计算开销更高。
