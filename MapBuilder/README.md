# landmark

- 当前实现的完整工作流与关键参数见 [WORKFLOW.md](WORKFLOW.md)
- 历史技术路线见 [TECH_ROUTE.md](TECH_ROUTE.md)
- 当前生产代码只保留在 `landmark/apps` 和 `landmark/tools`

## `landmark-main` 流程

`landmark-main` 的输入是原始 `ply` 点云，最终输出是 `road_arrow`、`laneline`、`crosswalk` 三类 `shp`。

执行顺序如下：

1. `pre-part`
   - 调 `landmark.tools.pc_process.pre_part`
   - 输出到 `pre-part/`
   - 产出整图 `bev_mask.png`、`bev_intensity.png`、`bev_rgb.png`
   - 默认只产出请求分辨率下的整图结果；当前主流程默认是 `0.08 mpp`
   - 产出 `geo_meta.json`、`geo_meta_mpp-08.json`
   - 若显式开启 `--render-extra-002`，才额外产出 `0.02 mpp` 的 `bev_intensity_mpp-02.png`、`bev_rgb_mpp-02.png` 和 `geo_meta_mpp-02.json`

2. `tile-part`
   - 调 `landmark.tools.pc_process.part.tile_part`
   - 默认输出到 `pre-part/parts.json`
   - 基于点云占据率生成规则分块
   - 同时在 `pre-part/parts_preview.png` 上给出分块示意图，底图为 `pre-part/bev_mask.png`

3. `part`
   - 调 `landmark.tools.pc_process.part.part`
   - 输出到 `parts/`
   - 按 `parts.json` 将整图点云切成多个 part PLY
   - 同时为每个 part 输出 `rgb/intensity` BEV 和 `parts/geo_meta.json`

4. `sam3`
   - 调 `landmark.tools.sam3.instance_seg`
   - 输入为 `parts/` 下的分块 BEV
   - 对 5 个固定 text prompt 做实例分割
   - 输出到 `objs/`
   - 每个 prompt/mode 的 `final_masks.npy` 统一为 `(H,W)` 的 `int32 label_map`

5. `masks`
   - 调 `landmark.tools.sam3.fuse_masks`
   - 对 `arrow`、`laneline`、`crosswalk` 三类目标做跨模态融合
   - 输出到 `masks/`
   - `masks/` 下保存融合后的 `*_label_map.npy`
   - 若 `landmark-main` 传 `--to-shp-mask-source intensity`，则 `to_shp` 直接使用 `objs/*/intensity/final_masks.npy`，跳过融合结果

6. `shp`
   - `road_arrow` 调 `landmark.apps.road_arrow`
   - `laneline` 调 `landmark.apps.laneline`
   - `crosswalk` 调 `landmark.apps.crosswalk`
   - 三个 app 的主输入统一为实例 `label_map`
   - 最终输出到 `shp/`

## 断点执行

`landmark-main` 支持用 `--stop-after` 在阶段间断点停止，可选值：

- `pre-part`
- `tile-part`
- `part`
- `sam3`
- `masks`
- `road-arrow`
- `laneline`
- `crosswalk`

示例：

```bash
uv run landmark-main input/sample.ply --out outputs/apps --stop-after sam3
```

## 常用命令

```bash
uv run landmark-tool-pre-part input/sample.ply -o outputs/pre-part
uv run landmark-tool-pre-part input/sample.ply -o outputs/pre-part --render-extra-002
uv run landmark-main input/sample.ply --out outputs/apps
uv run landmark-main input/sample.ply --out outputs/apps --to-shp-mask-source intensity
uv run landmark-full input/sample.ply --out outputs/map
```

## `landmark-full` 完整流程

`landmark-full` 是从原始 `ply` 到完整生产 `shp` 的总入口，不替换 `landmark-main`。

默认流程：

1. `pre-part`：输出 RGB、强度、高度等全局 BEV 和地理元数据。
2. `instance-seg-v2`：生成道路、车道线、箭头等七类文本实例分割结果。
3. `vectorize`：输出车道线、箭头、斑马线、人行道等 SHP。
4. `manhole`：复用 `objs/road/parts`，用多个视觉样例分别检索井盖，并输出最小外接圆 Polygon SHP。

井盖阶段将一个 `1008x1008` 样例分块放在左侧、一个目标分块放在右侧，
形成 `2016x1008` 输入。每个样例分别处理全部目标分块，结果按 IoU 合并。

```bash
uv run landmark-full input/sample.las --out outputs/map --stop-after manhole
uv run landmark-manhole \
  outputs/map/objs/road/parts/parts.json \
  outputs/map/pre-part/bev_pc_csf/bev_pc_csf_rgb_filled.png \
  outputs/map/pre-part/bev_pc_csf/pc_csf_geo_meta.json \
  --samples asserts/manhole_visual_samples/manifest.json \
  --out outputs/map/manhole
```

只使用已有原始 RGB BEV，以较小半径重建填充图和 road RGB 分块：

```bash
uv run landmark-rebuild-road-rgb-parts outputs/map --radius-px 3
```

该命令覆盖 `bev_pc_csf_rgb_filled.png` 与 `objs/road/parts`，不会执行 SAM3。

输出目录：

```text
<out>/shp/road_arrow/arrow.shp
<out>/shp/laneline/laneline.shp
<out>/shp/crosswalk/crosswalk.shp
<out>/shp/sidewalk_v2/sidewalk_boundary.shp
<out>/shp/belt_v2/belt.shp
<out>/shp/belt_v2/belt_boundary.shp
<out>/shp/manhole/manhole.shp
<out>/product/manhole.shp
<out>/manhole/result/objs.png
```
