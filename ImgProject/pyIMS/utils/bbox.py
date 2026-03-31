import numpy as np
from numpy.typing import NDArray
from .typing import PathLike
from typing import Iterable
from plyfile import PlyData, PlyElement
from math import acos, degrees

from pathlib import Path
from typing import Iterable, List

import numpy as np


_REL_EDGES: List[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 0),        # bottom
    (4, 5), (5, 6), (6, 7), (7, 4),        # top
    (0, 4), (1, 5), (2, 6), (3, 7),        # verticals
]


def obj_aabb_from_numpy(points: NDArray) -> str:

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be (N, 3)")

    mins = points.min(0)
    maxs = points.max(0)
    x0, y0, z0 = mins
    x1, y1, z1 = maxs
    corners = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=np.float32)

    lines: list[str] = []
    for x, y, z in corners:
        lines.append(f"v {x} {y} {z}")

    # write edges – +1 so they’re 1‑based
    for a, b in _REL_EDGES:
        lines.append(f"l {a+1} {b+1}")

    # add a blank line for readability
    lines.append("")
    return "\n".join(lines)


def merge_obj_strings(objs: Iterable[str]) -> str:
    out: list[str] = []
    vertex_offset = 0  # total vertices already written

    for obj_text in objs:
        lines = [ln.strip() for ln in obj_text.splitlines() if ln.strip()]
        if not lines:
            continue

        verts = [ln for ln in lines if ln.startswith("v ")]
        edges = [ln for ln in lines if ln.startswith("l ")]

        # write vertices first (no offset needed)
        out.extend(verts)

        # now write edges, adding base offset to every index
        for edge_line in edges:
            idxs = list(map(int, edge_line.split()[1:]))
            idxs = [i + vertex_offset for i in idxs]
            out.append("l " + " ".join(map(str, idxs)))

        vertex_offset += len(verts)
        out.append("")  # spacer between blocks

    return "\n".join(out)


def save_obj(text: str, path: PathLike) -> None:
    """Write *text* to *path* (parents created automatically)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def aabbs_from_instance_id(instance_id: NDArray[np.int32], points: NDArray[np.float32]) -> str:
    """Create an AABB OBJ string from instance IDs and point cloud."""

    unique_ids = np.unique(instance_id)
    obj_strings = []

    for inst_id in unique_ids:
        mask = instance_id == inst_id
        if not np.any(mask):
            continue  # skip empty instances

        pc = points[mask]
        obj_strings.append(obj_aabb_from_numpy(pc))

    return merge_obj_strings(obj_strings)


def create_cylinder_obj(p1, p2, radius, segments=36, height_segments=1, filename="cylinder.obj"):
    """
    Create a cylinder between two points and export it to an OBJ file.

    Parameters:
        p1 (tuple): Start point (x, y, z)
        p2 (tuple): End point (x, y, z)
        radius (float): Radius of the cylinder
        segments (int): Number of segments for the circular base
        height_segments (int): Number of segments along the height
        filename (str): Output OBJ file name
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    axis = p2 - p1
    height = np.linalg.norm(axis)

    # Normalize the axis
    axis_norm = axis / height

    # Find rotation axis and angle to align with Z
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(z_axis, axis_norm)
    rotation_angle = acos(np.dot(z_axis, axis_norm))

    def rotate(points):
        if np.allclose(rotation_angle, 0):
            return points
        rot_matrix = rotation_matrix(rotation_axis, rotation_angle)
        return np.dot(points, rot_matrix.T)

    def rotation_matrix(axis, theta):
        axis = axis / np.linalg.norm(axis)
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        return np.array([
            [a*a + b*b - c*c - d*d, 2*(b*c - a*d),     2*(b*d + a*c)],
            [2*(b*c + a*d),     a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
            [2*(b*d - a*c),     2*(c*d + a*b),     a*a + d*d - b*b - c*c]
        ])

    # Create cylinder vertices
    verts = []
    faces = []
    for j in range(height_segments + 1):
        z = j * height / height_segments
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            verts.append([x, y, z])

    # Rotate and translate
    verts = rotate(np.array(verts)) + p1

    # Side faces
    for j in range(height_segments):
        for i in range(segments):
            next_i = (i + 1) % segments
            idx1 = j * segments + i
            idx2 = j * segments + next_i
            idx3 = (j + 1) * segments + next_i
            idx4 = (j + 1) * segments + i
            faces.append([idx1, idx2, idx3])
            faces.append([idx1, idx3, idx4])

    # Write to OBJ
    with open(filename, 'w') as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

    print(f"Cylinder saved to {filename}")