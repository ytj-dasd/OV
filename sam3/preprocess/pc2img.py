import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

try:
    import laspy
except ImportError:  # pragma: no cover
    laspy = None

try:
    import CSF
except ImportError:  # pragma: no cover
    CSF = None


def _scale_to_uint8(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.size == 0:
        return arr.astype(np.uint8)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float64, copy=False)
    min_val = float(arr.min())
    max_val = float(arr.max())
    if max_val <= min_val:
        return np.zeros(arr.shape, dtype=np.uint8)
    if 0.0 <= min_val and max_val <= 255.0:
        return arr.astype(np.uint8)
    scaled = (arr - min_val) / (max_val - min_val) * 255.0
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)


def _scale_rgb_to_uint8(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
    rgb = np.stack([red, green, blue], axis=1)
    if rgb.dtype == np.uint8:
        return rgb
    rgb = rgb.astype(np.float64, copy=False)
    max_val = float(rgb.max()) if rgb.size > 0 else 0.0
    if max_val <= 0.0:
        return np.zeros(rgb.shape, dtype=np.uint8)
    if max_val <= 255.0:
        return rgb.astype(np.uint8)
    scale = 255.0 / max_val
    return np.clip(np.rint(rgb * scale), 0, 255).astype(np.uint8)


def read_las_point_cloud(las_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取 LAS 点云，提取 XYZ / RGB / intensity。"""
    if laspy is None:
        raise ImportError("laspy is required for LAS processing. Please install it first.")

    las = laspy.read(str(las_path))
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    points = np.stack([x, y, z], axis=1)
    if points.shape[0] == 0:
        raise ValueError(f"LAS has no points: {las_path}")

    has_rgb = all(hasattr(las, c) for c in ("red", "green", "blue"))
    if has_rgb:
        colors = _scale_rgb_to_uint8(
            np.asarray(las.red),
            np.asarray(las.green),
            np.asarray(las.blue),
        )
    else:
        if hasattr(las, "intensity"):
            gray = _scale_to_uint8(np.asarray(las.intensity))
        else:
            gray = np.zeros((points.shape[0],), dtype=np.uint8)
        colors = np.stack([gray, gray, gray], axis=1)

    if hasattr(las, "intensity"):
        intensity = _scale_to_uint8(np.asarray(las.intensity))
    else:
        intensity = np.zeros((points.shape[0],), dtype=np.uint8)

    return points, colors.astype(np.uint8), intensity.astype(np.uint8)


def center_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """坐标中心化：以包围盒中心作为原点。"""
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    return points - center, center


def csf_ground_mask(
    centered_points: np.ndarray,
    *,
    cloth_resolution: float = 1.0,
    rigidness: int = 1,
    time_step: float = 0.65,
    class_threshold: float = 1.2,
    iterations: int = 800,
    slope_smooth: bool = True,
) -> np.ndarray:
    """
    使用 CSF 提取地面点。
    参数默认偏宽松，优先保证地面不过滤丢失。
    """
    if CSF is None:
        raise ImportError(
            "CSF is required for ground filtering. Please install cloth-simulation-filter (CSF)."
        )
    if centered_points.shape[0] == 0:
        return np.zeros((0,), dtype=bool)

    csf = CSF.CSF()
    csf.setPointCloud(centered_points.astype(np.float64, copy=False))
    csf.params.cloth_resolution = float(cloth_resolution)
    csf.params.rigidness = int(rigidness)
    csf.params.time_step = float(time_step)
    csf.params.class_threshold = float(class_threshold)
    csf.params.interations = int(iterations)
    csf.params.bSloopSmooth = bool(slope_smooth)

    ground = CSF.VecInt()
    non_ground = CSF.VecInt()
    csf.do_filtering(ground, non_ground)

    mask = np.zeros((centered_points.shape[0],), dtype=bool)
    for idx in ground:
        mask[int(idx)] = True

    if not np.any(mask):
        # CSF 在极端参数/数据上可能返回空地面集，避免直接丢失全部结果。
        print("Warning: CSF returned empty ground set; fallback to all points.")
        mask[:] = True
    return mask


def generate_bev_images(
    points: np.ndarray,
    colors: np.ndarray,
    intensities: np.ndarray,
    resolution: float = 0.02,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, float] | None]:
    """
    生成俯视图 (BEV):
    1) 投影到栅格
    3) 每个像素保留 Z 最高点
    """
    if resolution <= 0:
        raise ValueError("resolution must be > 0")

    if points.shape[0] == 0:
        print("Warning: empty points for BEV projection.")
        return None, None, None

    min_x, max_x = float(np.min(points[:, 0])), float(np.max(points[:, 0]))
    min_y, max_y = float(np.min(points[:, 1])), float(np.max(points[:, 1]))
    min_z, max_z = float(np.min(points[:, 2])), float(np.max(points[:, 2]))

    width = max(1, int(np.floor((max_x - min_x) / resolution)) + 1)
    height = max(1, int(np.floor((max_y - min_y) / resolution)) + 1)
    print(f"Grid size: {width} x {height}")

    img_x = ((points[:, 0] - min_x) / resolution).astype(np.int64)
    img_y = ((max_y - points[:, 1]) / resolution).astype(np.int64)

    valid_mask = (img_x >= 0) & (img_x < width) & (img_y >= 0) & (img_y < height)
    img_x = img_x[valid_mask]
    img_y = img_y[valid_mask]
    z_vals = points[valid_mask, 2]
    colors = colors[valid_mask]
    intensities = intensities[valid_mask]

    flat_indices = img_y * width + img_x
    sort_idx = np.lexsort((-z_vals, flat_indices))
    flat_indices_sorted = flat_indices[sort_idx]
    _, unique_pos = np.unique(flat_indices_sorted, return_index=True)
    kept_indices = sort_idx[unique_pos]

    final_x = img_x[kept_indices]
    final_y = img_y[kept_indices]
    final_colors = colors[kept_indices]
    final_intensities = intensities[kept_indices]

    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    intensity_image = np.zeros((height, width), dtype=np.uint8)
    rgb_image[final_y, final_x] = final_colors
    intensity_image[final_y, final_x] = final_intensities.astype(np.uint8)

    meta = {
        "min_x": min_x,
        "min_y": min_y,
        "max_y": max_y,
        "min_z": min_z,
        "max_z": max_z,
        "width": int(width),
        "height": int(height),
        "resolution": float(resolution),
    }
    return rgb_image, intensity_image, meta


def find_nearest_pixel_inpaint(img: np.ndarray, max_dist: int = 5) -> np.ndarray:
    """最近邻填充黑色空洞像素。"""
    if img.ndim == 3:
        mask = np.all(img == 0, axis=2)
    else:
        mask = img == 0
    valid_mask = ~mask

    if not np.any(valid_mask) or not np.any(mask):
        return img.copy()

    valid_coords = np.argwhere(valid_mask)
    missing_coords = np.argwhere(mask)
    tree = cKDTree(valid_coords)
    dists, indices = tree.query(missing_coords, k=1, distance_upper_bound=max_dist)
    found_mask = dists != float("inf")

    out_img = img.copy()
    points_to_fill = missing_coords[found_mask]
    source_coords = valid_coords[indices[found_mask]]
    out_img[points_to_fill[:, 0], points_to_fill[:, 1]] = img[source_coords[:, 0], source_coords[:, 1]]
    return out_img


def stretch_intensity_for_inpaint(
    intensity_img: np.ndarray,
    *,
    low: float = 2.0,
    high: float = 100.0,
) -> np.ndarray:
    """
    Clamp + stretch intensity for inpaint preprocessing:
    - < low  -> 0
    - > high -> 255
    - [low, high] linearly mapped to [0, 255]
    """
    if high <= low:
        raise ValueError("high must be greater than low")
    x = intensity_img.astype(np.float32, copy=False)
    x = np.clip(x, low, high)
    x = (x - low) / (high - low) * 255.0
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def generate_idw_intensity_image(
    points: np.ndarray,
    intensities: np.ndarray,
    *,
    min_x: float,
    max_y: float,
    width: int,
    height: int,
    resolution: float,
    idw_radius: float,
    idw_epsilon: float,
) -> np.ndarray:
    """
    使用 KDTree + IDW 计算强度图：
    - 像素中心坐标:
      Xc = min_x + (col + 0.5) * resolution
      Yc = max_y - (row + 0.5) * resolution
    - 邻域半径内无点则保持 0
    """
    if idw_radius <= 0:
        raise ValueError("idw_radius must be > 0")
    if idw_epsilon <= 0:
        raise ValueError("idw_epsilon must be > 0")
    if points.shape[0] == 0:
        return np.zeros((height, width), dtype=np.uint8)

    xy = points[:, :2].astype(np.float64, copy=False)
    val = intensities.astype(np.float32, copy=False)
    tree = cKDTree(xy)

    out = np.zeros((height, width), dtype=np.float32)
    x_centers = min_x + (np.arange(width, dtype=np.float64) + 0.5) * resolution

    for row in tqdm(range(height), desc="IDW intensity", unit="row", leave=False):
        y_center = max_y - (row + 0.5) * resolution
        centers = np.column_stack((x_centers, np.full((width,), y_center, dtype=np.float64)))
        neighbors_per_pixel = tree.query_ball_point(centers, r=idw_radius)

        for col, neighbors in enumerate(neighbors_per_pixel):
            if not neighbors:
                continue

            idx = np.asarray(neighbors, dtype=np.int64)
            px = centers[col]
            nbr_xy = xy[idx]
            dx = nbr_xy[:, 0] - px[0]
            dy = nbr_xy[:, 1] - px[1]
            dist = np.sqrt(dx * dx + dy * dy)

            close_mask = dist <= idw_epsilon
            if np.any(close_mask):
                out[row, col] = float(np.mean(val[idx[close_mask]]))
                continue

            weights = 1.0 / np.maximum(dist, idw_epsilon)
            vv = val[idx]
            out[row, col] = float(np.sum(weights * vv) / np.sum(weights))

    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def cv2_imwrite_chinese(filename: Path, img: np.ndarray) -> bool:
    """兼容中文路径保存。"""
    try:
        cv2.imencode(".png", img)[1].tofile(str(filename))
        return True
    except Exception as exc:  # pragma: no cover
        print(f"Save failed: {exc}")
        return False


def process_single_las(
    las_path: Path,
    output_dir: Path,
    *,
    resolution: float,
    inpaint_radius: int,
    csf_cloth_resolution: float,
    csf_rigidness: int,
    csf_time_step: float,
    csf_class_threshold: float,
    csf_iterations: int,
    csf_slope_smooth: bool,
    intensity_mode: str,
    idw_radius: float,
    idw_epsilon: float,
    intensity_stretch_low: float,
    intensity_stretch_high: float,
) -> dict[str, float] | None:
    print(f"\nProcessing LAS: {las_path}")
    points, colors, intensities = read_las_point_cloud(las_path)
    centered_points, center_xyz = center_points(points)
    ground_mask = csf_ground_mask(
        centered_points,
        cloth_resolution=csf_cloth_resolution,
        rigidness=csf_rigidness,
        time_step=csf_time_step,
        class_threshold=csf_class_threshold,
        iterations=csf_iterations,
        slope_smooth=csf_slope_smooth,
    )
    print(f"CSF ground points: {int(np.count_nonzero(ground_mask))}/{centered_points.shape[0]}")

    ground_points = centered_points[ground_mask]
    ground_colors = colors[ground_mask]
    ground_intensities = intensities[ground_mask]
    # ground_intensities_stretched = stretch_intensity_for_inpaint(
    #     ground_intensities,
    #     low=intensity_stretch_low,
    #     high=intensity_stretch_high,
    # )

    rgb_raw, int_raw, meta = generate_bev_images(
        ground_points,
        ground_colors,
        ground_intensities,
        resolution=resolution,
    )
    if rgb_raw is None or int_raw is None or meta is None:
        return None

    rgb_raw_bgr = cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR)
    rgb_inpainted = find_nearest_pixel_inpaint(rgb_raw_bgr, max_dist=inpaint_radius)

    if intensity_mode == "max_nn":
        int_inpainted = find_nearest_pixel_inpaint(int_raw, max_dist=inpaint_radius)
    elif intensity_mode == "idw":
        print("Using IDW intensity mode...")
        int_idw = generate_idw_intensity_image(
            ground_points,
            ground_intensities_stretched,
            min_x=float(meta["min_x"]),
            max_y=float(meta["max_y"]),
            width=int(meta["width"]),
            height=int(meta["height"]),
            resolution=float(meta["resolution"]),
            idw_radius=idw_radius,
            idw_epsilon=idw_epsilon,
        )
        int_raw = int_idw
        int_inpainted = int_idw.copy()
    else:  # pragma: no cover
        raise ValueError(f"Unsupported intensity_mode: {intensity_mode}")

    stem = las_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_raw_dir = output_dir / "rgb_raw"
    int_raw_dir = output_dir / "intensity_raw"
    rgb_fill_dir = output_dir / "rgb_inpaint"
    int_fill_dir = output_dir / "intensity_inpaint"
    rgb_raw_dir.mkdir(parents=True, exist_ok=True)
    int_raw_dir.mkdir(parents=True, exist_ok=True)
    rgb_fill_dir.mkdir(parents=True, exist_ok=True)
    int_fill_dir.mkdir(parents=True, exist_ok=True)

    rgb_raw_path = rgb_raw_dir / f"{stem}.png"
    int_raw_path = int_raw_dir / f"{stem}.png"
    rgb_fill_path = rgb_fill_dir / f"{stem}.png"
    int_fill_path = int_fill_dir / f"{stem}.png"

    cv2_imwrite_chinese(rgb_raw_path, rgb_raw_bgr)
    cv2_imwrite_chinese(int_raw_path, int_raw)
    cv2_imwrite_chinese(rgb_fill_path, rgb_inpainted)
    cv2_imwrite_chinese(int_fill_path, int_inpainted)
    print(f"Saved: {rgb_raw_path.name}, {int_raw_path.name}, {rgb_fill_path.name}, {int_fill_path.name}")

    left_top_orig = np.array(
        [meta["min_x"], meta["max_y"], meta["min_z"]],
        dtype=np.float64,
    ) + center_xyz

    return {
        "las_name": las_path.name,
        "left_top_x": float(left_top_orig[0]),
        "left_top_y": float(left_top_orig[1]),
        "left_top_z": float(left_top_orig[2]),
    }


def _collect_las_files(input_dir: Path, recursive: bool) -> list[Path]:
    if recursive:
        files = sorted(
            p for p in input_dir.rglob("*")
            if p.is_file() and p.suffix.lower() == ".las"
        )
    else:
        files = sorted(
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".las"
        )
    return files


def batch_las_to_bev(
    input_dir: Path,
    output_dir: Path,
    *,
    resolution: float,
    inpaint_radius: int,
    csf_cloth_resolution: float,
    csf_rigidness: int,
    csf_time_step: float,
    csf_class_threshold: float,
    csf_iterations: int,
    csf_slope_smooth: bool,
    intensity_mode: str,
    idw_radius: float,
    idw_epsilon: float,
    intensity_stretch_low: float,
    intensity_stretch_high: float,
    recursive: bool,
    positions_txt: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    las_files = _collect_las_files(input_dir, recursive=recursive)
    if not las_files:
        raise FileNotFoundError(f"No .las files found in: {input_dir}")

    positions: list[dict[str, float]] = []
    for las_path in tqdm(las_files, desc="LAS -> BEV", unit="las"):
        pos = process_single_las(
            las_path,
            output_dir,
            resolution=resolution,
            inpaint_radius=inpaint_radius,
            csf_cloth_resolution=csf_cloth_resolution,
            csf_rigidness=csf_rigidness,
            csf_time_step=csf_time_step,
            csf_class_threshold=csf_class_threshold,
            csf_iterations=csf_iterations,
            csf_slope_smooth=csf_slope_smooth,
            intensity_mode=intensity_mode,
            idw_radius=idw_radius,
            idw_epsilon=idw_epsilon,
            intensity_stretch_low=intensity_stretch_low,
            intensity_stretch_high=intensity_stretch_high,
        )
        if pos is not None:
            positions.append(pos)

    pos_file = output_dir / positions_txt
    with pos_file.open("w", encoding="utf-8") as f:
        f.write("las_name left_top_x left_top_y left_top_z\n")
        for item in positions:
            f.write(
                f"{item['las_name']} "
                f"{item['left_top_x']:.6f} "
                f"{item['left_top_y']:.6f} "
                f"{item['left_top_z']:.6f}\n"
            )
    print(f"\nSaved LAS positions to: {pos_file}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch convert LAS folder to BEV images (center -> CSF ground -> BEV).")
    parser.add_argument("input_dir", type=str, help="Input folder containing .las files")
    parser.add_argument("output_dir", type=str, help="Output folder for generated images and position txt")
    parser.add_argument("--resolution", type=float, default=0.02, help="BEV resolution (meters/pixel)")
    parser.add_argument("--inpaint-radius", type=int, default=4, help="Nearest-neighbor inpaint radius")
    parser.add_argument(
        "--csf-cloth-resolution",
        type=float,
        default=2.0,
        help="CSF cloth resolution (looser for better ground recall)",
    )
    parser.add_argument(
        "--csf-rigidness",
        type=int,
        default=2,
        help="CSF rigidness (1-3); lower is looser",
    )
    parser.add_argument("--csf-time-step", type=float, default=0.65, help="CSF time step")
    parser.add_argument(
        "--csf-class-threshold",
        type=float,
        default=1.2,
        help="CSF class threshold; higher tends to keep more ground",
    )
    parser.add_argument("--csf-iterations", type=int, default=800, help="CSF iterations")
    parser.add_argument(
        "--intensity-mode",
        type=str,
        choices=["max_nn", "idw"],
        default="max_nn",
        help="Intensity generation mode: max-height+nearest-fill or IDW interpolation",
    )
    parser.add_argument("--idw-radius", type=float, default=0.08, help="IDW neighborhood radius in XY")
    parser.add_argument("--idw-epsilon", type=float, default=1e-6, help="IDW epsilon to avoid division by zero")
    parser.add_argument("--intensity-stretch-low", type=float, default=2.0, help="Stretch low threshold for max_nn mode")
    parser.add_argument("--intensity-stretch-high", type=float, default=100.0, help="Stretch high threshold for max_nn mode")
    parser.add_argument(
        "--no-csf-slope-smooth",
        action="store_true",
        help="Disable CSF slope smoothing (enabled by default)",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively search .las files")
    parser.add_argument(
        "--positions-txt",
        type=str,
        default="las_positions.txt",
        help="Output txt name for per-LAS left-top 3D coordinates",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_dir = Path(args.input_dir).expanduser().absolute()
    output_dir = Path(args.output_dir).expanduser().absolute()
    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"Input folder not found: {input_dir}")
    batch_las_to_bev(
        input_dir=input_dir,
        output_dir=output_dir,
        resolution=args.resolution,
        inpaint_radius=args.inpaint_radius,
        csf_cloth_resolution=args.csf_cloth_resolution,
        csf_rigidness=args.csf_rigidness,
        csf_time_step=args.csf_time_step,
        csf_class_threshold=args.csf_class_threshold,
        csf_iterations=args.csf_iterations,
        csf_slope_smooth=not args.no_csf_slope_smooth,
        intensity_mode=args.intensity_mode,
        idw_radius=args.idw_radius,
        idw_epsilon=args.idw_epsilon,
        intensity_stretch_low=args.intensity_stretch_low,
        intensity_stretch_high=args.intensity_stretch_high,
        recursive=args.recursive,
        positions_txt=args.positions_txt,
    )


if __name__ == "__main__":
    main()
