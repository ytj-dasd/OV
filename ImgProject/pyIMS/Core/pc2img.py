import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple
from numpy.typing import NDArray

try:
    import cupy as cp  # type: ignore
    HAS_CUPY = True
except ModuleNotFoundError:  # pragma: no cover - fallback for non-GPU test environments
    cp = np  # type: ignore
    HAS_CUPY = False


# Soft-splat tuning knobs. Adjust these values first when tuning image smoothness.
SOFT_SPLAT_RADIUS_MIN = 1
SOFT_SPLAT_RADIUS_MAX = 3  # was 3
SOFT_SPLAT_DEPTH_STEP = 10.0  # was 10.0
SOFT_SPLAT_SIGMA_SCALE = 0.6  # was 0.6
SOFT_SPLAT_DEPTH_GATE_BASE = 0.05  # was 0.05
SOFT_SPLAT_DEPTH_GATE_SCALE = 0.01  # was 0.01
SOFT_SPLAT_DEPTH_SIGMA_SCALE = 0.5  # was 0.5
SOFT_SPLAT_MIN_SIGMA = 1e-3
SOFT_SPLAT_CHUNK_SIZE = 50000


def _asnumpy(arr):
    if HAS_CUPY:
        return cp.asnumpy(arr)
    return np.asarray(arr)


def _get_array_module(arr):
    if HAS_CUPY:
        return cp.get_array_module(arr)
    return np


def _compute_soft_splat_radius(depths):
    xp = _get_array_module(depths)
    radius = xp.floor(depths / SOFT_SPLAT_DEPTH_STEP).astype(xp.int32) + SOFT_SPLAT_RADIUS_MIN
    radius = xp.clip(radius, SOFT_SPLAT_RADIUS_MIN, SOFT_SPLAT_RADIUS_MAX)
    return radius.astype(xp.int32)


def _compute_depth_gate_threshold(front_depth):
    xp = _get_array_module(front_depth)
    gate = SOFT_SPLAT_DEPTH_GATE_BASE + SOFT_SPLAT_DEPTH_GATE_SCALE * front_depth
    return xp.asarray(gate, dtype=xp.float32)


def _compute_soft_splat_sigma(radius):
    xp = _get_array_module(radius)
    sigma = SOFT_SPLAT_SIGMA_SCALE * radius.astype(xp.float32)
    return xp.maximum(sigma, xp.float32(0.5))


def _resolve_front_depth(base_front_depth, candidate_front_depth):
    xp = _get_array_module(base_front_depth)
    base_front_depth = xp.asarray(base_front_depth, dtype=xp.float32)
    candidate_front_depth = xp.asarray(candidate_front_depth, dtype=xp.float32)
    valid_base_mask = base_front_depth >= 0
    return xp.where(valid_base_mask, base_front_depth, candidate_front_depth).astype(xp.float32)


def _build_intrinsic(img_shape: Tuple[int, int], fov: float) -> np.ndarray:
    img_height, img_width = img_shape
    f = img_width / 2 / np.tan(fov / 2 * np.pi / 180)
    cx, cy = img_width / 2, img_height / 2
    return np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1],
    ], dtype=np.float32)


def _prepare_render_inputs(colors, intensities):
    if colors.dtype != cp.uint8:
        colors = (colors * 255).astype(cp.uint8)

    intensity_rgb = None
    if intensities is not None:
        if intensities.dtype != cp.uint8:
            intensities = intensities.astype(cp.uint8)
        intensity_rgb = cp.stack([intensities, intensities, intensities], axis=-1)

    return colors, intensity_rgb


def project_point_cloud(points, colors, intrinsic, extrinsic, img_shape):
    """
    Projects a 3D point cloud onto a 2D image plane.

    Args:
        points (np.ndarray): Nx3 array of 3D points.
        colors (np.ndarray, optional): Nx3 array of colors for each point.
        intrinsic (np.ndarray): 3x3 camera intrinsic matrix.
        extrinsic (np.ndarray): 3x4 affine transformation matrix [R | -R*t]
        img_shape (tuple): (height, width) of the output image.

    Returns:
        np.ndarray: An image with the projected points drawn.
    """

    if colors.dtype != np.uint8:
        colors = (colors * 255).astype(np.uint8)
    intrinsic = np.asarray(intrinsic)
    extrinsic = np.asarray(extrinsic)

    # Transform points to the camera coordinate system using the affine transformation.
    num_points = points.shape[0]
    points_hom = np.hstack([points, np.ones((num_points, 1), dtype=np.float32)])
    trans_mat = np.linalg.inv(extrinsic)[:3, :]
    pts_cam = (trans_mat @ points_hom.T).T

    # Keep only the points in front of the camera (positive Z).
    mask = pts_cam[:, 2] > 0
    pts_in_front = pts_cam[mask]
    pt_idx_in_front = np.arange(num_points)[mask]
    colors_in_front = colors[mask]

    # Transform points to the image coordinate system using the affine transformation.
    depths_in_front = pts_in_front[:, 2].reshape(-1, 1)
    normalized_points = pts_in_front / depths_in_front    # (x, y, z) => (x/z, y/z, 1)
    homogeneous_img_pts = intrinsic @ normalized_points.T
    pts_img = homogeneous_img_pts[:2, :].T  # the x, y coordinates in the image plane.

    # Round and clip to integer coordinates
    pts_indices = pts_img.astype(np.int32)

    # Filter points within view
    valid_mask = (pts_indices[:, 0] >= 0) & (pts_indices[:, 0] < img_shape[1]) & \
                 (pts_indices[:, 1] >= 0) & (pts_indices[:, 1] < img_shape[0])
    
    pts_inices_in_view = pts_indices[valid_mask]
    depths_in_view = depths_in_front[valid_mask]
    pt_idx_in_view = pt_idx_in_front[valid_mask]
    colors_in_view = colors_in_front[valid_mask]

    # Flatten indices for 2D array indexing of every points_in_view
    pts_flat_indices = pts_inices_in_view[:, 1] * img_shape[1] + pts_inices_in_view[:, 0]

    # Create GPU arrays for storing results
    height, width = img_shape
    rgb_img = np.zeros((height, width, 3), dtype=np.uint8)
    depth_img = np.full((height, width), np.inf, dtype=np.float32)
    pt_idx_img = np.full((height, width), -1, dtype=np.int32)

    # Use atomic operations to ensure depth consistency
    flat_depth_img = depth_img.ravel()
    np.minimum.at(flat_depth_img, pts_flat_indices, depths_in_view.ravel())

    # Compare to create a mask of valid minimum depth points
    min_depth_mask = np.isclose(flat_depth_img[pts_flat_indices], depths_in_view.ravel(), atol= 1e-6)

    min_depth_pts_flat_inices = pts_flat_indices[min_depth_mask]

    rgb_img.reshape(-1, 3)[min_depth_pts_flat_inices] = colors_in_view[min_depth_mask]
    pt_idx_img.ravel()[min_depth_pts_flat_inices] = pt_idx_in_view[min_depth_mask]

    return rgb_img, depth_img, pt_idx_img


# def project_point_cloud_gpu(points: NDArray, colors: NDArray, intrinsic: NDArray, extrinsic: NDArray, img_shape: Tuple[int, int], buffer_size: float):
#     """
#     Projects a 3D point cloud onto a 2D image plane using GPU for faster processing.

#     Args:
#         points (np.ndarray): Nx3 array of 3D points.
#         intrinsic (np.ndarray): 3x3 camera intrinsic matrix.
#         extrinsic (np.ndarray): 3x4 affine transformation matrix [R | -R*t]
#         img_shape (tuple): (height, width) of the output image.
#         colors (np.ndarray, optional): Nx3 array of colors for each point.

#     Returns:
#         np.ndarray: rgb_img, depth_img, info 
#     """

#     # Transfer data to GPU
#     if colors.dtype != cp.uint8:
#         colors = (colors * 255).astype(cp.uint8)
#     intrinsic = cp.asarray(intrinsic)
#     extrinsic = cp.asarray(extrinsic)

#     height, width = img_shape

#     # Transform points to the camera coordinate system using the affine transformation.
#     num_points = points.shape[0]
#     pts_hom = cp.hstack([points, cp.ones((num_points, 1), dtype=cp.float32)])
#     trans_mat = cp.linalg.inv(extrinsic)[:3, :]
#     pts_cam = (trans_mat @ pts_hom.T).T

#     # Keep only the points in front of the camera (positive Z).
#     front_mask = pts_cam[:, 2] > 0
#     pts_cam = pts_cam[front_mask]
#     pts_dist = cp.power(pts_cam, 2).sum(axis= 1)
#     pts_depth = pts_cam[:, 2]
#     pts_colors = colors[front_mask]
#     pts_indices = cp.arange(num_points, dtype= cp.int32)[front_mask]

#     # Transform points to the image coordinate system using the affine transformation.
#     pts_cam_normalized = pts_cam / pts_depth.reshape(-1, 1)
#     pts_img_hom = intrinsic @ pts_cam_normalized.T
#     pts_img = pts_img_hom[:2, :].T  # the x, y coordinates in the image plane.

#     # Round and clip to integer coordinates
#     pts_img_indices = cp.round(pts_img).astype(cp.int32)

#     # Filter points within view
#     valid_mask = (pts_img_indices[:, 0] >= 0) & (pts_img_indices[:, 0] < width) & \
#                  (pts_img_indices[:, 1] >= 0) & (pts_img_indices[:, 1] < height)
    
#     pts_img_indices = pts_img_indices[valid_mask]    # (n x 2)
#     pts_dist = pts_dist[valid_mask]    # (n)
#     pts_depth = pts_depth[valid_mask]
#     pts_colors = pts_colors[valid_mask]    # (n x 3)
#     pts_indices = pts_indices[valid_mask]    # (n)

#     # Flatten indices for 2D array indexing of every points_in_view
#     pts_img_indices = pts_img_indices[:, 1] * width + pts_img_indices[:, 0]   # * (n) for store every points' img index which in the camera's view.

#     # Create GPU arrays for storing results
#     dist_img = cp.full((height * width), cp.inf, dtype=cp.float32)
#     depth_img = cp.full((height * width), -1, dtype=cp.float32)
#     rgb_img = cp.zeros((height * width, 3), dtype=cp.uint8)
#     pts_indices_img = cp.full((height * width), -1, dtype=cp.int32)

#     # Use atomic operations to ensure depth consistency
#     cp.minimum.at(dist_img, pts_img_indices, pts_dist)

#     # Mask to update only the closest points, get the min depth point mask for pts_img_indices
#     min_dist_mask = cp.isclose(dist_img[pts_img_indices], pts_dist, atol= 1e-6)   # * mask for pts in the view which are the cloest point.

#     min_dist_pts_img_inices = pts_img_indices[min_dist_mask]

#     depth_img[min_dist_pts_img_inices] = pts_depth[min_dist_mask]
#     rgb_img[min_dist_pts_img_inices] = pts_colors[min_dist_mask]
#     pts_indices_img[min_dist_pts_img_inices] = pts_indices[min_dist_mask]

#     # for buffer 
#     buffer = pts_dist - dist_img[pts_img_indices]    # * distance for every pts in the view from the corresponding cloest point in the same line.
#     buffer_mask = buffer < buffer_size

#     buffer_pts_img_indices = pts_img_indices[buffer_mask]     # * buffer pts flat indices
#     buffer_pts_indices = pts_indices[buffer_mask]

#     # Transfer results back to CPU
#     rgb_img = cp.asnumpy(rgb_img).reshape(height, width, 3)
#     dist_img = cp.asnumpy(dist_img).reshape(height, width)
#     depth_img = cp.asnumpy(depth_img).reshape(height, width)
#     buffer_pts_img_indices = cp.asnumpy(buffer_pts_img_indices)
#     buffer_pts_indices = cp.asnumpy(buffer_pts_indices)

#     return rgb_img, (dist_img, depth_img), (buffer_pts_img_indices, buffer_pts_indices)    # (height x width x 3), (height * width), ((buffered_points_num), (buffered_points_num))

def project_point_cloud_gpu(points: NDArray, colors: NDArray, intensities: NDArray, intrinsic: NDArray, extrinsic: NDArray, img_shape: Tuple[int, int], buffer_size: float):
    """
    Projects a 3D point cloud onto a 2D image plane using GPU for faster processing.

    Args:
        points (np.ndarray): Nx3 array of 3D points.
        colors (np.ndarray): Nx3 array of colors for each point.
        intensities (np.ndarray): Nx1 array of intensities for each point.
        intrinsic (np.ndarray): 3x3 camera intrinsic matrix.
        extrinsic (np.ndarray): 3x4 affine transformation matrix [R | -R*t]
        img_shape (tuple): (height, width) of the output image.
        buffer_size (float): buffer size for occlusion handling.

    Returns:
        np.ndarray: rgb_img, intensity_img, depth_img, info 
    """

    # Transfer data to GPU
    if colors.dtype != cp.uint8:
        colors = (colors * 255).astype(cp.uint8)
    
    # Convert intensity to 3-channel uint8
    if intensities is not None:
        if intensities.dtype != cp.uint8:
            intensities = intensities.astype(cp.uint8)
        # INTENSITY_LO = 10.0
        # INTENSITY_HI = 60.0
        
        # # print(f"Original intensity: min={float(intensities.min()):.1f}, max={float(intensities.max()):.1f}, mean={float(intensities.mean()):.1f}")
        
        # stretched_intensities = cp.where(
        #     intensities <= INTENSITY_LO, 
        #     0.0,
        #     cp.where(
        #         intensities >= INTENSITY_HI,
        #         255.0,
        #         (intensities - INTENSITY_LO) * 255.0 / (INTENSITY_HI - INTENSITY_LO)
        #     )
        # )
        
        # # Ensure values are in valid range and convert to uint8
        # intensities = cp.clip(stretched_intensities, 0, 255).astype(cp.uint8)
        
        # # Expand to 3 channels
        intensities = cp.stack([intensities, intensities, intensities], axis=-1)
    
    intrinsic = cp.asarray(intrinsic)
    extrinsic = cp.asarray(extrinsic)

    height, width = img_shape

    # Transform points to the camera coordinate system using the affine transformation.
    num_points = points.shape[0]
    pts_hom = cp.hstack([points, cp.ones((num_points, 1), dtype=cp.float32)])
    trans_mat = cp.linalg.inv(extrinsic)[:3, :]
    pts_cam = (trans_mat @ pts_hom.T).T

    # Keep only the points in front of the camera (positive Z).
    front_mask = pts_cam[:, 2] > 0
    pts_cam = pts_cam[front_mask]
    pts_dist = cp.power(pts_cam, 2).sum(axis= 1)
    pts_depth = pts_cam[:, 2]
    pts_colors = colors[front_mask]
    pts_intensities = intensities[front_mask] if intensities is not None else None
    pts_indices = cp.arange(num_points, dtype= cp.int32)[front_mask]

    # Transform points to the image coordinate system using the affine transformation.
    pts_cam_normalized = pts_cam / pts_depth.reshape(-1, 1)
    pts_img_hom = intrinsic @ pts_cam_normalized.T
    pts_img = pts_img_hom[:2, :].T  # the x, y coordinates in the image plane.

    # Round and clip to integer coordinates
    pts_img_indices = cp.round(pts_img).astype(cp.int32)

    # Filter points within view
    valid_mask = (pts_img_indices[:, 0] >= 0) & (pts_img_indices[:, 0] < width) & \
                 (pts_img_indices[:, 1] >= 0) & (pts_img_indices[:, 1] < height)
    
    pts_img_indices = pts_img_indices[valid_mask]    # (n x 2)
    pts_dist = pts_dist[valid_mask]    # (n)
    pts_depth = pts_depth[valid_mask]
    pts_colors = pts_colors[valid_mask]    # (n x 3)
    pts_intensities = pts_intensities[valid_mask] if pts_intensities is not None else None    # (n x 3)
    pts_indices = pts_indices[valid_mask]    # (n)

    # Flatten indices for 2D array indexing of every points_in_view
    pts_img_indices = pts_img_indices[:, 1] * width + pts_img_indices[:, 0]   # * (n) for store every points' img index which in the camera's view.

    # Create GPU arrays for storing results
    dist_img = cp.full((height * width), cp.inf, dtype=cp.float32)
    depth_img = cp.full((height * width), -1, dtype=cp.float32)
    rgb_img = cp.zeros((height * width, 3), dtype=cp.uint8)
    intensity_img = cp.zeros((height * width, 3), dtype=cp.uint8) if pts_intensities is not None else None
    pts_indices_img = cp.full((height * width), -1, dtype=cp.int32)

    # Use atomic operations to ensure depth consistency
    cp.minimum.at(dist_img, pts_img_indices, pts_dist)

    # Mask to update only the closest points, get the min depth point mask for pts_img_indices
    min_dist_mask = cp.isclose(dist_img[pts_img_indices], pts_dist, atol= 1e-6)   # * mask for pts in the view which are the cloest point.

    min_dist_pts_img_inices = pts_img_indices[min_dist_mask]

    depth_img[min_dist_pts_img_inices] = pts_depth[min_dist_mask]
    rgb_img[min_dist_pts_img_inices] = pts_colors[min_dist_mask]
    if intensity_img is not None:
        intensity_img[min_dist_pts_img_inices] = pts_intensities[min_dist_mask]
    pts_indices_img[min_dist_pts_img_inices] = pts_indices[min_dist_mask]

    # for buffer 
    buffer = pts_dist - dist_img[pts_img_indices]    # * distance for every pts in the view from the corresponding cloest point in the same line.
    buffer_mask = buffer < buffer_size

    buffer_pts_img_indices = pts_img_indices[buffer_mask]     # * buffer pts flat indices
    buffer_pts_indices = pts_indices[buffer_mask]

    # Transfer results back to CPU
    rgb_img = _asnumpy(rgb_img).reshape(height, width, 3)
    intensity_img = _asnumpy(intensity_img).reshape(height, width, 3) if intensity_img is not None else None
    dist_img = _asnumpy(dist_img).reshape(height, width)
    depth_img = _asnumpy(depth_img).reshape(height, width)
    buffer_pts_img_indices = _asnumpy(buffer_pts_img_indices)
    buffer_pts_indices = _asnumpy(buffer_pts_indices)

    return rgb_img, intensity_img, (dist_img, depth_img), (buffer_pts_img_indices, buffer_pts_indices)


def pc2img(points, colors, intensities, extrinsic, img_shape= (1080, 1980), fov= 90, buffer_size= .05):
    img_height, img_width = img_shape
    f = img_width / 2 / np.tan(fov / 2 * np.pi / 180)
    cx, cy = img_width / 2, img_height / 2
    intrinsic = np.array([
        [f,  0, cx],
        [ 0, f, cy],
        [ 0,  0,  1]
    ])

    points = cp.asarray(points)
    colors = cp.asarray(colors)
    intensities = cp.asarray(intensities) if intensities is not None else None
    res = project_point_cloud_gpu(points, colors, intensities, intrinsic, extrinsic, img_shape, buffer_size)
    return res


def _project_points_continuous_gpu(
    points,
    colors,
    intensities,
    intrinsic,
    extrinsic,
    img_shape: Tuple[int, int],
):
    height, width = img_shape
    num_points = points.shape[0]
    pts_hom = cp.hstack([points, cp.ones((num_points, 1), dtype=cp.float32)])
    trans_mat = cp.linalg.inv(extrinsic)[:3, :]
    pts_cam = (trans_mat @ pts_hom.T).T

    front_mask = pts_cam[:, 2] > 0
    pts_cam = pts_cam[front_mask]
    pts_depth = pts_cam[:, 2]
    pts_colors = colors[front_mask]
    pts_intensities = intensities[front_mask] if intensities is not None else None
    pts_indices = cp.arange(num_points, dtype=cp.int32)[front_mask]

    if pts_cam.shape[0] == 0:
        return (
            cp.zeros((0, 2), dtype=cp.float32),
            cp.zeros((0,), dtype=cp.float32),
            cp.zeros((0, 3), dtype=cp.uint8),
            None if pts_intensities is None else cp.zeros((0, 3), dtype=cp.uint8),
            cp.zeros((0,), dtype=cp.int32),
        )

    pts_cam_normalized = pts_cam / pts_depth.reshape(-1, 1)
    pts_img_hom = intrinsic @ pts_cam_normalized.T
    pts_img = pts_img_hom[:2, :].T.astype(cp.float32, copy=False)

    margin = float(SOFT_SPLAT_RADIUS_MAX)
    valid_mask = (
        (pts_img[:, 0] >= -margin)
        & (pts_img[:, 0] < width + margin)
        & (pts_img[:, 1] >= -margin)
        & (pts_img[:, 1] < height + margin)
    )
    return (
        pts_img[valid_mask],
        pts_depth[valid_mask].astype(cp.float32, copy=False),
        pts_colors[valid_mask],
        pts_intensities[valid_mask] if pts_intensities is not None else None,
        pts_indices[valid_mask],
    )


def _build_soft_splat_candidates(
    pts_img,
    radii,
    img_shape: Tuple[int, int],
    offsets_x,
    offsets_y,
):
    height, width = img_shape
    if pts_img.shape[0] == 0:
        return (
            cp.zeros((0,), dtype=cp.int32),
            cp.zeros((0,), dtype=cp.int32),
            cp.zeros((0,), dtype=cp.float32),
            cp.zeros((0,), dtype=cp.int32),
        )

    base_x = cp.floor(pts_img[:, 0]).astype(cp.int32, copy=False)
    base_y = cp.floor(pts_img[:, 1]).astype(cp.int32, copy=False)

    cand_x = base_x[:, None] + offsets_x[None, :]
    cand_y = base_y[:, None] + offsets_y[None, :]

    dx = (cand_x.astype(cp.float32) + 0.5) - pts_img[:, 0:1]
    dy = (cand_y.astype(cp.float32) + 0.5) - pts_img[:, 1:2]
    dist2 = dx * dx + dy * dy

    valid_mask = (
        (cand_x >= 0)
        & (cand_x < width)
        & (cand_y >= 0)
        & (cand_y < height)
        & (dist2 <= cp.square(radii.astype(cp.float32))[:, None])
    )
    if not bool(valid_mask.any()):
        return (
            cp.zeros((0,), dtype=cp.int32),
            cp.zeros((0,), dtype=cp.int32),
            cp.zeros((0,), dtype=cp.float32),
            cp.zeros((0,), dtype=cp.int32),
        )

    chunk_point_ids = cp.broadcast_to(
        cp.arange(pts_img.shape[0], dtype=cp.int32)[:, None],
        valid_mask.shape,
    )[valid_mask]
    pixel_indices = (cand_y[valid_mask] * width + cand_x[valid_mask]).astype(cp.int32, copy=False)
    distances2 = dist2[valid_mask].astype(cp.float32, copy=False)
    candidate_radii = cp.broadcast_to(radii[:, None], valid_mask.shape)[valid_mask].astype(cp.int32, copy=False)
    return chunk_point_ids, pixel_indices, distances2, candidate_radii


def _soft_splat_render_gpu(
    pts_img,
    pts_depth,
    pts_colors,
    pts_intensities,
    hard_depth_img,
    img_shape: Tuple[int, int],
):
    height, width = img_shape
    flat_size = height * width
    rgb_acc = cp.zeros((flat_size, 3), dtype=cp.float32)
    intensity_acc = cp.zeros((flat_size, 3), dtype=cp.float32) if pts_intensities is not None else None
    weight_sum = cp.zeros((flat_size,), dtype=cp.float32)
    candidate_front_depth = cp.full((flat_size,), cp.inf, dtype=cp.float32)

    max_radius = int(SOFT_SPLAT_RADIUS_MAX)
    offsets = cp.arange(-max_radius, max_radius + 1, dtype=cp.int32)
    offsets_y, offsets_x = cp.meshgrid(offsets, offsets, indexing="ij")
    offsets_x = offsets_x.reshape(-1)
    offsets_y = offsets_y.reshape(-1)

    radii = _compute_soft_splat_radius(pts_depth)
    chunk_size = int(SOFT_SPLAT_CHUNK_SIZE)

    for start in range(0, int(pts_img.shape[0]), chunk_size):
        end = min(start + chunk_size, int(pts_img.shape[0]))
        chunk_point_ids, pixel_indices, _, _ = _build_soft_splat_candidates(
            pts_img[start:end],
            radii[start:end],
            img_shape,
            offsets_x,
            offsets_y,
        )
        if pixel_indices.size == 0:
            continue
        depth_values = pts_depth[start:end][chunk_point_ids]
        cp.minimum.at(candidate_front_depth, pixel_indices, depth_values)

    base_front_depth = cp.asarray(hard_depth_img, dtype=cp.float32).reshape(-1)
    front_depth = _resolve_front_depth(base_front_depth, candidate_front_depth)

    if not bool(cp.isfinite(front_depth).any()):
        rgb_img = np.zeros((height, width, 3), dtype=np.uint8)
        intensity_img = None if pts_intensities is None else np.zeros((height, width, 3), dtype=np.uint8)
        return rgb_img, intensity_img

    for start in range(0, int(pts_img.shape[0]), chunk_size):
        end = min(start + chunk_size, int(pts_img.shape[0]))
        chunk_point_ids, pixel_indices, distances2, candidate_radii = _build_soft_splat_candidates(
            pts_img[start:end],
            radii[start:end],
            img_shape,
            offsets_x,
            offsets_y,
        )
        if pixel_indices.size == 0:
            continue

        depth_values = pts_depth[start:end][chunk_point_ids]
        local_front_depth = front_depth[pixel_indices]
        finite_mask = cp.isfinite(local_front_depth)
        if not bool(finite_mask.any()):
            continue

        chunk_point_ids = chunk_point_ids[finite_mask]
        pixel_indices = pixel_indices[finite_mask]
        distances2 = distances2[finite_mask]
        candidate_radii = candidate_radii[finite_mask]
        depth_values = depth_values[finite_mask]
        local_front_depth = local_front_depth[finite_mask]

        delta_depth = depth_values - local_front_depth
        depth_gate = _compute_depth_gate_threshold(local_front_depth)
        depth_mask = delta_depth <= depth_gate
        if not bool(depth_mask.any()):
            continue

        chunk_point_ids = chunk_point_ids[depth_mask]
        pixel_indices = pixel_indices[depth_mask]
        distances2 = distances2[depth_mask]
        candidate_radii = candidate_radii[depth_mask]
        delta_depth = delta_depth[depth_mask]
        depth_gate = depth_gate[depth_mask]

        sigma_xy = _compute_soft_splat_sigma(candidate_radii)
        sigma_z = cp.maximum(depth_gate * SOFT_SPLAT_DEPTH_SIGMA_SCALE, SOFT_SPLAT_MIN_SIGMA)
        spatial_weight = cp.exp(-distances2 / (2.0 * sigma_xy * sigma_xy))
        depth_weight = cp.exp(-(delta_depth * delta_depth) / (2.0 * sigma_z * sigma_z))
        weights = (spatial_weight * depth_weight).astype(cp.float32, copy=False)

        colors_chunk = pts_colors[start:end][chunk_point_ids].astype(cp.float32, copy=False)
        cp.add.at(weight_sum, pixel_indices, weights)
        for channel in range(3):
            cp.add.at(rgb_acc[:, channel], pixel_indices, weights * colors_chunk[:, channel])

        if intensity_acc is not None and pts_intensities is not None:
            intensity_chunk = pts_intensities[start:end][chunk_point_ids].astype(cp.float32, copy=False)
            for channel in range(3):
                cp.add.at(intensity_acc[:, channel], pixel_indices, weights * intensity_chunk[:, channel])

    valid_pixels = weight_sum > 0
    rgb_img = cp.zeros((flat_size, 3), dtype=cp.uint8)
    if bool(valid_pixels.any()):
        rgb_values = rgb_acc[valid_pixels] / weight_sum[valid_pixels, None]
        rgb_img[valid_pixels] = cp.clip(rgb_values, 0, 255).astype(cp.uint8)

    intensity_img = None
    if intensity_acc is not None:
        intensity_img = cp.zeros((flat_size, 3), dtype=cp.uint8)
        if bool(valid_pixels.any()):
            intensity_values = intensity_acc[valid_pixels] / weight_sum[valid_pixels, None]
            intensity_img[valid_pixels] = cp.clip(intensity_values, 0, 255).astype(cp.uint8)

    rgb_img = _asnumpy(rgb_img).reshape(height, width, 3)
    intensity_img = _asnumpy(intensity_img).reshape(height, width, 3) if intensity_img is not None else None
    return rgb_img, intensity_img


def pc2img_soft(points, colors, intensities, extrinsic, img_shape=(1080, 1980), fov=90, buffer_size=.05):
    intrinsic = _build_intrinsic(img_shape, fov)

    points_gpu = cp.asarray(points)
    colors_gpu = cp.asarray(colors)
    intensities_gpu = cp.asarray(intensities) if intensities is not None else None

    _, _, (dist_img, depth_img), (pts_img_indices, pts_indices) = project_point_cloud_gpu(
        points_gpu,
        colors_gpu,
        intensities_gpu,
        cp.asarray(intrinsic),
        cp.asarray(extrinsic),
        img_shape,
        buffer_size,
    )

    render_colors, render_intensities = _prepare_render_inputs(colors_gpu, intensities_gpu)
    pts_img, pts_depth, pts_colors, pts_intensities, _ = _project_points_continuous_gpu(
        points_gpu,
        render_colors,
        render_intensities,
        cp.asarray(intrinsic),
        cp.asarray(extrinsic),
        img_shape,
    )
    rgb_img, intensity_img = _soft_splat_render_gpu(
        pts_img,
        pts_depth,
        pts_colors,
        pts_intensities,
        depth_img,
        img_shape,
    )
    return rgb_img, intensity_img, (dist_img, depth_img), (pts_img_indices, pts_indices)


def patch(flat_img_indices: NDArray[np.float32], width: int, radius: int = 1) -> NDArray[np.float32]:
    patch_flat_depth_img = []
    for row in range(-radius, radius + 1):
        tmp_img = flat_img_indices + row * width
        for col in range(-radius, radius + 1):
            patch_flat_depth_img.append(tmp_img + col)
    
    if isinstance(flat_img_indices, np.ndarray):
        patch_flat_depth_img = np.concatenate(patch_flat_depth_img)
    elif isinstance(flat_img_indices, cp.ndarray):
        patch_flat_depth_img = cp.concatenate(patch_flat_depth_img)

    return patch_flat_depth_img


def render_img(rgb_img: NDArray, depth_img: NDArray, patch_size: int):
    """
    for handeling the erroneous prediction caused by holes in the point cloud.
    """
    height, width = depth_img.shape
    radius = patch_size // 2
    flat_rgb_img = rgb_img.reshape(-1, 3)
    flat_depth_img = depth_img.ravel()
    flat_img_shape = flat_depth_img.shape
    flat_img_indices = np.arange(flat_img_shape[0])

    patch_flat_img_indices = patch(flat_img_indices, width, radius)   # * (radius^2 * height * width)
    patch_flat_depth_img = np.concatenate([flat_depth_img] * patch_size**2)
    patch_flat_rgb_img = np.concatenate([flat_rgb_img] * patch_size**2)

    within_mask: NDArray[np.bool] = (patch_flat_img_indices > 0) & (patch_flat_img_indices < flat_img_indices.shape[0])
    patch_flat_img_indices = patch_flat_img_indices[within_mask]
    patch_flat_depth_img = patch_flat_depth_img[within_mask]
    patch_flat_rgb_img = patch_flat_rgb_img[within_mask]

    flat_rendered_depth_img = np.full(flat_img_shape, np.inf, dtype= np.float32)
    np.minimum.at(flat_rendered_depth_img, patch_flat_img_indices, patch_flat_depth_img)
    
    rendered_mask = np.isclose(flat_rendered_depth_img[patch_flat_img_indices], patch_flat_depth_img)
    flat_rendered_rgb_img = np.zeros(flat_rgb_img.shape, dtype= np.uint8)
    flat_rendered_rgb_img[patch_flat_img_indices[rendered_mask]] = patch_flat_rgb_img[rendered_mask]

    rendered_rgb_img = flat_rendered_rgb_img.reshape(rgb_img.shape)
    # rendered_depth_img = flat_rendered_depth_img.reshape(depth_img.shape)

    # patch_blocked_mask = np.isclose(depth_img, rendered_depth_img)
    # depth_img[~patch_blocked_mask] = np.inf
    # depth_img = np.asnumpy(depth_img)
    # rendered_depth_img = np.asnumpy(rendered_depth_img)

    return rendered_rgb_img


def render_img_gpu(target_imgs: NDArray, according_img: NDArray, patch_size: int) -> List[NDArray]:
    """
    for handeling the erroneous prediction caused by holes in the point cloud.
    """

    target_imgs = [cp.asarray(img) for img in target_imgs]
    according_img = cp.asarray(according_img)

    height, width = according_img.shape
    according_img = according_img.ravel()
    img_indices = cp.arange(height * width)
    for idx in range(len(target_imgs)):
        target_imgs[idx] = target_imgs[idx].reshape(height * width, -1)    # TODO different target imgs

    # * get patch img
    radius = patch_size // 2
    patch_img_indices = patch(img_indices, width, radius)   # * (radius^2 * height * width)
    patch_according_img = cp.concatenate([according_img] * patch_size**2)
    patch_target_imgs = []
    for idx in range(len(target_imgs)):
        patch_target_imgs.append(cp.concatenate([target_imgs[idx]] * patch_size**2))

    # * get rad of patch point not within the img
    within_mask = (patch_img_indices > 0) & (patch_img_indices < height * width)
    patch_img_indices = patch_img_indices[within_mask]
    patch_according_img = patch_according_img[within_mask]

    for idx in range(len(patch_target_imgs)):
        patch_target_imgs[idx] = patch_target_imgs[idx][within_mask]

    # * get right pathched according_img
    rendered_according_img = cp.full(height * width, cp.inf, dtype= cp.float32)
    cp.minimum.at(rendered_according_img, patch_img_indices, patch_according_img)
    
    rendered_mask = cp.isclose(rendered_according_img[patch_img_indices], patch_according_img)    # * (rendered_num), for patch_img
    rendered_img_indices = patch_img_indices[rendered_mask]

    rendered_target_imgs = []
    for idx in range(len(target_imgs)):
        rendered_target_img = cp.zeros(target_imgs[idx].shape, dtype= target_imgs[idx].dtype)
        rendered_target_img[rendered_img_indices] = patch_target_imgs[idx][rendered_mask]
        rendered_target_img = _asnumpy(rendered_target_img).reshape(height, width, -1)
        rendered_target_imgs.append(rendered_target_img)


    # patch_blocked_mask = cp.isclose(according_img, rendered_according_img)
    # according_img[~patch_blocked_mask] = cp.inf
    # according_img = cp.asnumpy(according_img)

    return rendered_target_imgs
