# SAM3 Preprocess

本目录用于点云转 BEV、分块推理、多 prompt 结果拼接。

## 主要脚本

### `pc2img.py`
- 输入 LAS 文件夹，逐个处理。
- 默认流程：中心化 -> CSF 地面筛选 -> BEV 投影。
- 输出四类图像到四个子目录：
  - `rgb_raw/`
  - `intensity_raw/`
  - `rgb_inpaint/`
  - `intensity_inpaint/`
- 同时输出 `las_positions.txt`（每个 LAS 的左上角三维原始坐标）。
- 强度模式可选：
  - `max_nn`：最高点取值 + 最近邻填充
  - `idw`：基于 KDTree 半径邻域的 IDW 插值

### `tile_infer.py`
- 输入单图或图像目录，支持一次传多个 prompt。
- 每个 prompt 会单独创建结果文件夹。
- 每张图输出：
  - `*_merged.png`（实例可视化，按实例 id 着色）
  - `*_merged.npz`（`masks/boxes/scores` + 元数据）

### `tile_infer_visual.py`
- 输入单图或图像目录 + 样例图 + 样例框，做基于视觉样例的分块检索分割。
- 检索方式：样例 patch 在左、当前 tile 在右，拼接后用几何框 prompt 做跨图检索。
- 支持黑块加速跳过：若 tile 黑色像素占比 > 80%，直接跳过推理。
- `--sample-box` 使用样例图像素坐标 `XYXY`（`X0 Y0 X1 Y1`），脚本内部会转换为 SAM3 所需的归一化 `[cx, cy, w, h]`。
- 每张图输出：
  - `*_merged.png`（实例可视化，按实例 id 着色）
  - `*_merged.npz`（`masks/boxes/scores` + 元数据，含样例图与样例框信息）

### `split_las_by_ranges.py`
- 输入单个 `.las` 文件。
- 输入多组分块参数（每组 `x` 范围 + `y` 范围），按范围裁切并输出子 LAS 文件（参数通过重复 `--range` 传入）。
- 只按范围裁切，不按点数分组。
- 内部使用分块读取大文件（chunk iterator）仅用于降低内存占用，不影响按范围分块逻辑。
- 支持重叠范围：同一个点可落入多个输出分块。
- 输出：
  - 每个范围一个 `.las`
  - `split_las_summary.csv`（每个分块的范围与输出文件路径）

### `stitch_bev_by_positions.py`
- 读取 `las_positions.txt`，将四类投影图按真实坐标拼成四张大图。

### `stitch_prompt_masks_to_global.py`
- 读取多个 prompt 结果文件夹和 `las_positions.txt`。
- 将所有 prompt 的 mask 贴到统一全局坐标。
- 输出：
  - `all_prompts_global_instances.npz`
  - `all_prompts_global_mask_black.png`（黑底贴图）
  - `all_prompts_global_overlay.png`（可选，贴到底图）

### 其他推理脚本
- `infer.py`：单文本提示
- `infer_multi_concept.py`：多概念文本提示
- `infer_visual_prompt.py`：视觉框提示
- `infer_cross_image_retrieval.py`：跨图像检索
- `tile_infer_visual.py`：分块视觉检索分割（样例 patch + tile 拼接）
- `split_las_by_ranges.py`：按多组 `x/y` 范围分块导出 LAS

## VS Code Launch 对应功能

当前 `.vscode/launch.json` 中的配置：

- `Python Debugger: Current File`：调试当前打开脚本
- `pc2img: Batch LAS -> BEV`：LAS 批量转 BEV（`max_nn` 强度）
- `pc2img: Batch LAS -> BEV (IDW Intensity)`：LAS 批量转 BEV（`idw` 强度）
- `tile_infer: Folder Images`：对目录图像做多 prompt 分块推理，按 prompt 分目录输出
- `tile_infer_visual: Folder Images`：对目录图像做分块视觉检索分割
- `split_las_by_ranges: Single LAS`：按多组范围分块导出 LAS
- `stitch_bev: By LAS Positions`：将四类 BEV 图拼成四张大图
- `stitch_prompt_masks: Global`：将多个 prompt 结果贴到全局图（含黑底图和可选底图叠加图）


## 最简流程

1. 跑 `pc2img.py` 生成四类 BEV 图和 `las_positions.txt`  
2. 跑 `tile_infer.py`（多个 prompt）生成 prompt 分目录结果  
3. 跑 `stitch_bev_by_positions.py` 生成四张全局 BEV 大图  
4. 跑 `stitch_prompt_masks_to_global.py` 生成全局 prompt 贴图  
