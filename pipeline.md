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

## 实施任务（可直接交给 AI 编码）

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
2. 每图命中类别（来自 `.vlm.txt`）。
3. 你的 SAM 推理代码（输出单个 `.npz`，含 `masks/boxes/scores`）。
输出：`sam_mask/{img_stem}.npz`。
要求：
1. 无效实例（面积太小或分数过低）在回投阶段过滤。
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

### 任务5：2D->3D 回投 + IoU合并 + 点冲突归属 + 去噪
输入：
1. `projected_images/{img_stem}.npz`（`pts_img_indices`, `pts_indices`, `dist_img`）。
2. `projected_images/effective_stations.json`（实际用于投影的相机外参）。
3. `sam_mask/{img_stem}.npz`（`masks/boxes/scores`，以及类别字段）。
4. 场景 LAS（用于输出实例点云）。

处理：
1. 将每个 2D mask 通过 `pts_img_indices -> pts_indices` 回投到 3D 点索引集合；回投时基于 `effective_stations.json` 计算候选点的相机前向深度，仅保留落在当前像素前景深度 `+0.2m` 范围内的点，形成 scene-level 候选实例。
2. 候选实例先做类别门控：同原类别允许合并；`1-6` 视为“杆状物大类”，彼此允许合并；`7 行道树` 仅与 `7 行道树` 合并；其他跨类组合暂不合并。
3. 在类别门控通过后，使用双通道合并：主通道为 `point IoU >= threshold`；补充通道用于 `1-6` 杆状物大类，以及 `8-15` 的同类实例，为 `XY` 中心距离足够小；树木暂不启用该补充通道（待树干拆分实现后再接入）。
4. 候选合并后，若 merged instance 点数小于 `min-merged-points`，则直接丢弃，不进入后续处理。
5. 对保留下来的 merged instance，先按 `w_angle * confidence` 累加类别得分，确定最终类别；点归属前执行“非树优先，树删点”，若某点同时属于树实例和非树实例，则保留在非树实例中，并从树实例候选点中剔除。
6. 树木类实例单独后处理（待实现）：在每个合并后的树实例内部先找树干锚点；若检测到多个树干，则按最近树干重新划分树冠；若只有一个树干，则保持单树。
7. 点级冲突归属：同一点属于多个实例时，按加权总分取最大实例。
8. 空间去噪：每实例做空间聚类，仅保留最大簇；若最大簇点数小于 `denoise-min-points`，则删除该实例。
9. 类特定几何去噪（待实现）：在上述结果上进一步剔除不符合类别形态的噪声点，例如围栏的贴地噪声、杆状物脚下的地面外扩点。

视角权重使用角度权重（仅角度，不用距离）：

\[
w_{angle} = \max\left(0,\ 1 - \frac{|\alpha_{max}-90^\circ|}{90^\circ}\right)
\]

输出（每场景）：
1. `fusion/{scene_name}_instance_seg.las`：只包含实例点，按场景内实例 ID 随机着色。
2. `fusion/{scene_name}_instance_seg.npz`：包含实例 ID、类别、置信度、实例点索引，以及点级唯一归属结果。
3. `fusion/{scene_name}_instance_seg_meta.json`：去噪/过滤统计日志。

```bash
python ImgProject/pipeline/task5_scene_instance_seg.py \
  --data-root benchmark \
  --iou-threshold 0.30 \
  --merge-xy-distance 0.50 \
  --fov-deg 90 \
  --min-mask-points 20 \
  --backproject-depth-threshold 0.20 \
  --min-merged-points 150 \
  --denoise-eps 0.60 \
  --denoise-min-points 30
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
1. VLM 固定使用 Qwen3VL。
2. SAM 输出固定为每图一个 `.npz`，且可直接读取 `masks/boxes/scores`。
3. 多场景 LAS 在同一坐标系下，可直接进行跨场景空间计算。
4. 合并与聚类阈值统一放到配置中可调。
