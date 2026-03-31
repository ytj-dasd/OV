from typing import List, Any, Tuple, Optional, Sequence
from .typing import PathLike
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

from plyfile import PlyData, PlyElement
import laspy
from PIL import Image, ImageDraw, ImageFont
from .label import color_mapping


def read_las(file_path):
    # Read LAS file
    las = laspy.read(file_path)

    # Get XYZ coordinates
    points = np.vstack((las.x, las.y, las.z)).T  # shape: (N, 3)

    # Intensity (normalize to uint8 for downstream image projection usage)
    intensity = None
    if hasattr(las, "intensity"):
        intensity = las.intensity.astype(np.uint16)
        if intensity.size > 0 and intensity.max() > 255:
            intensity = (intensity / 256).astype(np.uint8)
        else:
            intensity = intensity.astype(np.uint8)

    # Check for RGB color fields
    has_color = all(hasattr(las, attr) for attr in ['red', 'green', 'blue'])

    if has_color:
        # Normalize color from 16-bit to 8-bit if necessary
        r = las.red.astype(np.uint16)
        g = las.green.astype(np.uint16)
        b = las.blue.astype(np.uint16)

        # Some LAS files store color in 0-65535, normalize if needed
        if r.max() > 255:
            r = (r / 256).astype(np.uint8)
            g = (g / 256).astype(np.uint8)
            b = (b / 256).astype(np.uint8)
        else:
            r = r.astype(np.uint8)
            g = g.astype(np.uint8)
            b = b.astype(np.uint8)

        colors = np.vstack((r, g, b)).T  # shape: (N, 3)
    else:
        colors = np.zeros((points.shape[0], 3), dtype=np.uint8)  # No color data in LAS

    return points, colors, intensity


def read_ply(fp, is_property: bool = False):
    plydata = PlyData.read(fp)
    vertex_data = plydata['vertex']
    xyz = (vertex_data['x'], vertex_data['y'], vertex_data['z'])
    points = np.stack(xyz, axis=-1)
    
    if all(field in vertex_data.data.dtype.names for field in ['red', 'green', 'blue']):
        rgb = (vertex_data['red'], vertex_data['green'], vertex_data['blue'])
        colors = np.stack(rgb, axis=-1)
    else:
        colors = np.zeros((points.shape[0], 3), dtype=np.uint8)
    
    intensity = None
    if 'scalar_Intensity' in vertex_data.data.dtype.names:
        intensity = vertex_data['scalar_Intensity']
    elif 'intensity' in vertex_data.data.dtype.names:
        intensity = vertex_data['intensity']
    
    if is_property:
        properties = {}
        for prop in vertex_data.data.dtype.names:
            if prop not in ['x', 'y', 'z', 'red', 'green', 'blue']:
                properties[prop] = vertex_data[prop]
        return points, colors, properties
    else:
        return points, colors, intensity


# class PointCloud:
#     def __init__(self, points: NDArray[np.float32], colors: Optional[NDArray[np.uint8]] = None, properties: Optional[dict] = None):
#         """
#         Initialize a PointCloud object.

#         Parameters
#         ----------
#         points : np.ndarray
#             An array of shape (N, 3) with dtype float32, where each row is [X, Y, Z].
#         colors : np.ndarray, optional
#             An array of shape (N, 3) with dtype uint8, where each row is [R, G, B].
#         properties : dict, optional
#             A dictionary containing additional properties of the point cloud.
#             Each key is a property name and the value is an array of shape (N,).
#             If not provided, no properties are stored.
#         """
#         self.points = points.astype(np.float32)
#         if colors is not None:
#             self.colors = colors.astype(np.uint8) if colors is not None else None
#         if properties is not None:
#             self.properties = {}

#     @staticmethod
#     def from_ply(fp) -> 'PointCloud':
#         plydata = PlyData.read(fp)

#     def 


def update_pc_colors(pc: PlyData, colors: NDArray= None, colormap: List[List[int]]= None, color_indices: NDArray= None) -> PlyData:
    """
    Update the colors of a PointCloud object.

    Parameters
    ----------
    pcd : PointCloud
        The PointCloud object to update.
    colors : np.ndarray, optional
        An array of shape (N, 3) with dtype uint8, where each row is [R, G, B].
    colormap : List[List[int]], optional
        A list of RGB color values.
    color_indices : np.ndarray, optional
        An array of shape (N,) with dtype int, where each element is an index into the colormap.

    Returns
    -------
    None
    """

    if colors is not None:
        assert colors.ndim == 2, "colors must be a 3D array"
    elif colormap is not None and color_indices is not None:
        colors = color_mapping(color_indices, colormap)

    for channel, name in enumerate(['red', 'green', 'blue']):
        if name not in pc['vertex'].data.dtype.names:
            raise KeyError(f"Vertex property '{name}' is missing from the ply file.")
        pc['vertex'].data[name] = colors[:, channel].astype(pc['vertex'].data[name].dtype)

    return pc


def update_pc_property(pd: PlyData, property_name: str, property_arr: NDArray) -> PlyData:
    """
    """

    if property_name not in pd['vertex'].data.dtype.names:
        raise KeyError(f"Vertex property '{property_name}' is missing from the ply file.")
    
    pd['vertex'].data[property_name] = property_arr.astype(pd['vertex'].data[property_name].dtype)

    return pd


def pack_rgb(rgb: NDArray[np.uint]) -> NDArray[np.float32]:
    """
    Pack an (N,3) array of uint8 RGB values into an (N,) float32 array
    for PCD rgb storage.

    Parameters
    ----------
    rgb : np.ndarray
        An array of shape (N, 3), with dtype uint8, where each row is [R, G, B].

    Returns
    -------
    np.ndarray
        A float32 array of shape (N,), where each element is the packed RGB
        bit‐pattern reinterpret‐cast as a float32.
    """
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError("Input must be of shape (N, 3)")
    # Ensure unsigned 8‐bit
    arr = rgb.astype(np.uint8)
    # Promote to 32‐bit integers for bit‐shifting
    r = arr[:, 0].astype(np.uint32)
    g = arr[:, 1].astype(np.uint32)
    b = arr[:, 2].astype(np.uint32)
    # Pack into a single 32‐bit word: R in bits 16–23, G in 8–15, B in 0–7
    rgb_u = (r << 16) | (g << 8) | b
    # Reinterpret the 32‐bit integer as a 32‐bit float (no copy)
    rgb_f = rgb_u.view(np.float32)
    return rgb_f


def unpack_rgb(rgb_f: NDArray[np.float32]) -> NDArray[np.uint8]:
    """
    Unpack a (N,) float32 array of packed RGB into an (N,3) uint8 array.

    Parameters
    ----------
    rgb_f : np.ndarray
        Array of shape (N,) with dtype float32, where each element
        holds 0xRRGGBB packed into the mantissa+exponent bits.

    Returns
    -------
    np.ndarray
        Array of shape (N,3), dtype uint8, where each row is [R, G, B].
    """
    if rgb_f.dtype != np.float32:
        raise ValueError("Input array must have dtype float32")
    # Reinterpret the float32 bits as uint32 (no copy) :contentReference[oaicite:0]{index=0}
    rgb_u = rgb_f.view(np.uint32)
    # Extract channels via bit‐shifts & masks (PCL style unpack) :contentReference[oaicite:1]{index=1}
    r = (rgb_u >> 16) & 0xFF
    g = (rgb_u >>  8) & 0xFF
    b =  rgb_u        & 0xFF
    # Stack and cast to uint8
    return np.stack([r, g, b], axis=1).astype(np.uint8)


def _ensure_supported_dtype(arr: np.ndarray) -> np.ndarray:
    """Return *arr* with a plyfile‑supported dtype, casting if loss‑less.

    plyfile supports at most 32‑bit integers, so we down‑cast int64/uint64 if the
    values fit. Otherwise we raise a ValueError so the caller can decide what to
    do (e.g. scale, clip, or quantise).
    """
    supported = {
        ("i", 1): np.int8,
        ("i", 2): np.int16,
        ("i", 4): np.int32,
        ("u", 1): np.uint8,
        ("u", 2): np.uint16,
        ("u", 4): np.uint32,
        ("f", 4): np.float32,
        ("f", 8): np.float64,
    }

    kind, itemsize = arr.dtype.kind, arr.dtype.itemsize
    if (kind, itemsize) in supported:
        return arr  # already fine

    # Attempt loss‑less down‑cast for 64‑bit ints.
    if kind == "i":
        target = np.int32
    elif kind == "u":
        target = np.uint32
    else:
        raise ValueError(
            f"Unsupported dtype {arr.dtype}. plyfile allows int/uint ≤32‑bit "
            "or float32/float64."
        )

    info = np.iinfo(target)
    if arr.min() < info.min or arr.max() > info.max:
        raise ValueError(
            f"Values in array range [{arr.min()}, {arr.max()}] exceed {target} "
            "bounds; cannot safely cast."
        )
    return arr.astype(target, copy=False)


def _new_vertex_dtype(old_dtype: np.dtype, new_field: str, new_dtype: np.dtype) -> np.dtype:
    """Create a *new* structured dtype with *new_field* appended as *new_dtype*."""
    if old_dtype.names is None:
        raise ValueError("Vertex element has no named properties to extend.")
    if new_field in old_dtype.names:
        raise ValueError(f"Property '{new_field}' already exists in vertex data.")

    descr: list[Any] = old_dtype.descr + [(new_field, new_dtype.str)]
    return np.dtype(descr)


def add_pc_property(pc: PlyData, property_name: str, property_data: np.ndarray) -> PlyData:  # noqa: N802
    """Return a *new* ``PlyData`` where the *vertex* element gains *property_name*.

    Parameters
    ----------
    pc
        Input point cloud.
    property_name
        Name of the new property.
    property_data
        1‑D ``numpy.ndarray`` whose length equals ``pc['vertex'].count``.

    Returns
    -------
    PlyData
        New point cloud including the extra property.

    Raises
    ------
    ValueError
        On length mismatch, duplicate property name, or unsupported dtype.
    """

    if "vertex" not in pc:
        raise ValueError("Input PLY has no 'vertex' element to extend.")

    vertex_el: PlyElement = pc["vertex"]
    nverts: int = vertex_el.count

    if property_data.shape[0] != nverts:
        raise ValueError(
            f"property_data length {property_data.shape[0]} does not match vertex "
            f"count {nverts}."
        )

    property_data = np.ascontiguousarray(property_data)
    property_data = _ensure_supported_dtype(property_data)

    new_dtype = _new_vertex_dtype(vertex_el.data.dtype, property_name, property_data.dtype)

    # Allocate and copy existing + new column
    new_vertex = np.empty(nverts, dtype=new_dtype)
    for name in vertex_el.data.dtype.names:  # type: ignore[attr-defined]
        new_vertex[name] = vertex_el.data[name]
    new_vertex[property_name] = property_data

    is_text = pc.text  # keep original text/binary mode
    new_vertex_el = PlyElement.describe(new_vertex, "vertex", is_text)
    new_elements = [new_vertex_el] + [el for el in pc.elements if el.name != "vertex"]

    return PlyData(
        new_elements,
        text=pc.text,
        byte_order=pc.byte_order,
        comments=list(pc.comments),
        obj_info=list(pc.obj_info),
    )


def select_points_by_mask(ply: PlyData, mask: NDArray[np.bool_]) -> PlyData:
    """
    Return a new PlyData object that retains only the points where mask is True.

    Args:
        ply (PlyData): Original point cloud data.
        mask (NDArray[np.bool_]): Boolean mask of shape (N,), where N is the number of points.

    Returns:
        PlyData: A new PlyData object with points filtered by the mask.
    """

    original_vertices = ply['vertex'].data
    if len(original_vertices) != mask.shape[0]:
        raise ValueError("Mask length must match number of vertices")

    # Filter the structured array
    filtered_vertices = original_vertices[mask]

    # Create new PlyElement and PlyData
    new_vertex_element = PlyElement.describe(filtered_vertices, 'vertex')
    return PlyData([new_vertex_element], text=ply.text)


def load_obj_vertices_edges(file: PathLike) -> tuple[np.ndarray, np.ndarray]:
    """Read *vertices* and *edge* index pairs from a Wavefront OBJ file.

    Only two record types are recognised:

    * **v x y z** – 3‑D vertex positions (additional components are ignored)
    * **l i j [k …]** – poly‑line connecting listed 1‑based vertex indices.

    Parameters
    ----------
    file
        Path to the *.obj* file.

    Returns
    -------
    (N, 3) float32 ndarray
        Vertices in XYZ order.
    (M, 2) int32 ndarray
        Zero‑based vertex‑index pairs representing all edges in the file.
    """
    verts: list[list[float]] = []
    edges: list[list[int]] = []

    with Path(file).expanduser().open("r", encoding="utf‑8", errors="ignore") as fh:
        for line in fh:
            line = line.lstrip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("v "):  # vertex
                parts: Sequence[str] = line.split()
                if len(parts) >= 4:
                    x, y, z = map(float, parts[1:4])
                    verts.append([x, y, z])

            elif line.startswith("l "):  # line/poly‑line
                idx = list(map(int, line.split()[1:]))
                for a, b in zip(idx[:-1], idx[1:]):
                    # OBJ indices are 1‑based; convert to 0‑based
                    edges.append([a - 1, b - 1])

    return (
        np.asarray(verts, dtype=np.float32),
        np.asarray(edges, dtype=np.int32),
    )


def _bresenham_line(x0: int, y0: int, x1: int, y1: int):
    """Yield integer coordinates along a Bresenham line from *(x0, y0)* → *(x1, y1)*."""
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def render_point_cloud_ortho(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    *,
    lines: Optional[np.ndarray] = None,
    line_color: Tuple[int, int, int] = (255, 0, 0),
    img_size: Tuple[int, int] = (800, 800),
    margin: float = 0.05,
    background: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Project 3‑D points/edges onto an orthographic top‑down RGB image.

    Parameters
    ----------
    points
        *(N, 3)* float array with XYZ coordinates.
    colors
        Optional *(N, 3)* uint8 array of per‑point RGB values.  Defaults to black.
    lines
        Optional *(M, 2)* int array of vertex‑index pairs (0‑based) to draw.
    line_color
        RGB colour for all edges.
    img_size
        *(width, height)* of the output image in pixels.
    margin
        Uniform fractional border around the projected data (0 ≤ *margin* < 0.5).
    background
        RGB background colour.

    Returns
    -------
    np.ndarray
        *(H, W, 3)* uint8 image array in RGB order.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if colors is not None and colors.shape != (points.shape[0], 3):
        raise ValueError("colors must match points shape (N, 3)")
    if lines is not None and (lines.ndim != 3 or lines.shape[-1] != 3):
        raise ValueError("lines must have shape (M, 2)")

    w, h = map(int, img_size)

    # ----- Compute view transform -----
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span_xy = maxs[:2] - mins[:2]
    span_xy[span_xy == 0] = 1e-9  # avoid division by zero

    scale = min(
        (w * (1 - 2 * margin)) / span_xy[0],
        (h * (1 - 2 * margin)) / span_xy[1],
    )
    offset_x = margin * w - mins[0] * scale
    offset_y = margin * h + maxs[1] * scale  # invert Y

    # ----- Project vertices -----
    proj = np.empty((points.shape[0], 2), dtype=np.int32)
    proj[:, 0] = np.round(points[:, 0] * scale + offset_x).astype(np.int32)
    proj[:, 1] = np.round(-points[:, 1] * scale + offset_y).astype(np.int32)

    img = np.full((h, w, 3), background, dtype=np.uint8)
    default_color = np.array([0, 0, 0], dtype=np.uint8)

    # Depth sort – far→near so closer points overwrite
    for idx in np.argsort(points[:, 2]):
        x, y = proj[idx]
        if 0 <= x < w and 0 <= y < h:
            img[y, x] = colors[idx] if colors is not None else default_color

    # ----- Draw edges -----
    if lines is not None and lines.size:
        num_lines, _, _ = lines.shape
        lines_proj = np.empty((num_lines, 2, 2), dtype= np.int32)
        lines_proj[:, :, 0] = np.round(lines[:, :, 0] * scale + offset_x).astype(np.int32)
        lines_proj[:, :, 1] = np.round(-lines[:, :, 1] * scale + offset_y).astype(np.int32)
        col = np.array(line_color, dtype=np.uint8)
        for v0, v1 in lines_proj:
            x0, y0 = v0
            x1, y1 = v1
            for x, y in _bresenham_line(x0, y0, x1, y1):
                if 0 <= x < w and 0 <= y < h:
                    img[y, x] = col

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default()

    if lines is not None and lines.size:
        groups = (len(lines) + 11) // 12
        for gi in range(groups):
            start = gi * 12
            x, y = lines_proj[start][0]
            draw.text((x - 8 , y - 6), str(gi + 1), fill=line_color, font=font, anchor="mm")

    return np.asarray(pil_img)
