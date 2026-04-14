import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import remove_small_holes, remove_small_objects, skeletonize
from sklearn.cluster import DBSCAN
from tqdm import tqdm

try:
    from shapely.affinity import scale as scale_geometry
    from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
    from shapely.ops import substring
except ModuleNotFoundError as exc:
    if exc.name == "shapely":
        raise ModuleNotFoundError(
            "Missing dependency 'shapely'. Install it in the Python interpreter "
            "used to run this script:\n"
            f"  {sys.executable} -m pip install shapely\n"
            "If running from VS Code, select the same interpreter in "
            "'Python: Select Interpreter'."
        ) from exc
    raise

Pixel = tuple[int, int]  # (y, x)

NEIGHBOR_OFFSETS: tuple[Pixel, ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


@dataclass
class RoadNode:
    node_id: str
    kind: str  # junction | endpoint
    y: float
    x: float
    pixels: list[Pixel]
    incident_edge_ids: list[str] = field(default_factory=list)


@dataclass
class RoadEdge:
    edge_id: str
    u: str
    v: str
    pixels: list[Pixel]
    length_px: float


@dataclass
class RoadGraph:
    nodes: list[RoadNode]
    edges: list[RoadEdge]
    resolution: float


@dataclass
class SlicePolygon:
    slice_id: str
    kind: str  # junction | road_segment
    source_id: str
    geometry: Polygon | MultiPolygon


@dataclass
class SegmentPartition:
    partition_id: str
    edge_id: str
    segment_index: int
    centerline: LineString
    length_m: float


def cv2_imwrite_any(path: Path, img: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".png"
    ok, encoded = cv2.imencode(ext, img)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def cv2_imread_any(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def _pixel_step(a: Pixel, b: Pixel) -> float:
    dy = abs(a[0] - b[0])
    dx = abs(a[1] - b[1])
    return float(np.hypot(dy, dx))


def _stable_color_bgr(key: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    rng = np.random.default_rng(seed)
    color = rng.integers(64, 256, size=3, dtype=np.uint8)
    return int(color[0]), int(color[1]), int(color[2])


def _meters_to_pixels(value_m: float, resolution: float) -> float:
    return float(value_m / resolution)


def _area_m2_to_px(area_m2: float, resolution: float) -> int:
    return int(max(1.0, round(area_m2 / (resolution * resolution))))


def downsample_mask_for_topology(
    mask: np.ndarray,
    source_resolution: float,
    topology_resolution: float | None,
) -> tuple[np.ndarray, float, float, float]:
    """
    Optionally downsample mask for topology processing only.
    Returns:
      topo_mask, topo_resolution_m_per_px, scale_x_topo_to_source, scale_y_topo_to_source
    """
    src = np.asarray(mask, dtype=bool)
    if topology_resolution is None or topology_resolution <= source_resolution:
        return src, float(source_resolution), 1.0, 1.0

    src_h, src_w = src.shape[:2]
    scale = float(source_resolution / topology_resolution)  # <1 for downsampling
    dst_w = max(1, int(round(src_w * scale)))
    dst_h = max(1, int(round(src_h * scale)))

    src_u8 = (src.astype(np.uint8) * 255).astype(np.uint8)
    resized = cv2.resize(src_u8, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
    topo_mask = resized > 0

    scale_x = float(src_w / dst_w)
    scale_y = float(src_h / dst_h)
    topo_res_x = float(source_resolution * scale_x)
    topo_res_y = float(source_resolution * scale_y)
    topo_resolution_eff = float(0.5 * (topo_res_x + topo_res_y))

    return topo_mask, topo_resolution_eff, scale_x, scale_y


def _resize_bool_mask(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    src = (np.asarray(mask, dtype=np.uint8) * 255).astype(np.uint8)
    resized = cv2.resize(src, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return resized > 0


def load_binary_mask(mask_path: Path, mask_key: str = "union_mask") -> np.ndarray:
    if not mask_path.exists():
        raise FileNotFoundError(f"mask file not found: {mask_path}")

    if mask_path.suffix.lower() == ".npz":
        data = np.load(mask_path)
        if mask_key in data.files:
            arr = np.asarray(data[mask_key])
        elif "mask" in data.files:
            arr = np.asarray(data["mask"])
        elif "union_mask" in data.files:
            arr = np.asarray(data["union_mask"])
        elif "masks" in data.files:
            arr = np.asarray(data["masks"])
        else:
            raise KeyError(
                f"No usable mask array in {mask_path}. Available keys: {list(data.files)}"
            )
    else:
        img = cv2_imread_any(mask_path, flags=cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read image mask: {mask_path}")
        arr = np.asarray(img)

    if arr.ndim == 4:
        if arr.shape[1] == 1:
            arr = arr[:, 0]
        arr = np.any(arr > 0, axis=0)
    elif arr.ndim == 3:
        arr = np.any(arr > 0, axis=2)
    elif arr.ndim == 2:
        arr = arr > 0
    else:
        raise ValueError(f"Unsupported mask shape: {arr.shape}")

    return np.asarray(arr, dtype=bool)


def clean_road_mask(
    mask: np.ndarray,
    closing_radius_px: int,
    min_component_area_px: int,
    max_hole_area_px: int,
) -> np.ndarray:
    clean = np.asarray(mask, dtype=bool)

    if closing_radius_px > 0:
        kernel_size = int(2 * closing_radius_px + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        clean = cv2.morphologyEx(clean.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0

    if min_component_area_px > 1:
        clean = remove_small_objects(clean, min_size=int(min_component_area_px))

    if max_hole_area_px > 1:
        clean = remove_small_holes(clean, area_threshold=int(max_hole_area_px))

    return np.asarray(clean, dtype=bool)


def _build_skeleton_adjacency(
    skeleton: np.ndarray,
) -> tuple[set[Pixel], dict[Pixel, list[Pixel]], dict[Pixel, int]]:
    ys, xs = np.nonzero(skeleton)
    pixels: set[Pixel] = {(int(y), int(x)) for y, x in zip(ys, xs)}

    adjacency: dict[Pixel, list[Pixel]] = {}
    degrees: dict[Pixel, int] = {}
    for p in pixels:
        py, px = p
        neighbors: list[Pixel] = []
        for dy, dx in NEIGHBOR_OFFSETS:
            q = (py + dy, px + dx)
            if q in pixels:
                neighbors.append(q)
        adjacency[p] = neighbors
        degrees[p] = len(neighbors)

    return pixels, adjacency, degrees


def prune_short_spurs(skeleton: np.ndarray, prune_length_px: float, max_iters: int = 10) -> np.ndarray:
    if prune_length_px <= 0:
        return np.asarray(skeleton, dtype=bool)

    pruned = np.asarray(skeleton, dtype=bool).copy()

    for _ in range(max_iters):
        pixels, adjacency, degrees = _build_skeleton_adjacency(pruned)
        if not pixels:
            break

        endpoints = [p for p in pixels if degrees[p] == 1]
        to_remove: set[Pixel] = set()

        for endpoint in endpoints:
            path: list[Pixel] = [endpoint]
            prev: Pixel | None = None
            cur: Pixel = endpoint
            length_px = 0.0

            while True:
                candidates = [q for q in adjacency[cur] if q != prev]
                if not candidates:
                    break
                nxt = candidates[0]
                length_px += _pixel_step(cur, nxt)
                path.append(nxt)
                prev, cur = cur, nxt
                if degrees[cur] != 2:
                    break

            cur_degree = degrees.get(cur, 0)
            is_short = length_px <= prune_length_px
            ends_at_branch = cur_degree >= 3
            ends_at_endpoint = cur_degree <= 1

            if not is_short:
                continue

            if ends_at_branch:
                to_remove.update(path[:-1])
            elif ends_at_endpoint:
                to_remove.update(path)

        if not to_remove:
            break

        for y, x in to_remove:
            pruned[y, x] = False

    return pruned


def skeletonize_and_prune(mask: np.ndarray, prune_length_px: float) -> np.ndarray:
    skeleton = skeletonize(np.asarray(mask, dtype=bool))
    return prune_short_spurs(skeleton, prune_length_px=prune_length_px)


def _cluster_junction_pixels(
    junction_pixels: list[Pixel],
    junction_cluster_eps_px: float,
    junction_min_samples: int,
) -> tuple[list[list[Pixel]], dict[Pixel, int]]:
    if not junction_pixels:
        return [], {}

    coords_xy = np.asarray([(float(x), float(y)) for y, x in junction_pixels], dtype=np.float32)
    eps = max(0.5, float(junction_cluster_eps_px))
    min_samples = max(1, int(junction_min_samples))
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(coords_xy).labels_

    clusters: list[list[Pixel]] = []
    pixel_to_cluster: dict[Pixel, int] = {}

    unique_labels = sorted(set(int(v) for v in labels if int(v) >= 0))
    for label in unique_labels:
        pixels = [junction_pixels[i] for i, lb in enumerate(labels) if int(lb) == label]
        cluster_idx = len(clusters)
        clusters.append(pixels)
        for p in pixels:
            pixel_to_cluster[p] = cluster_idx

    for idx, lb in enumerate(labels):
        if int(lb) >= 0:
            continue
        p = junction_pixels[idx]
        cluster_idx = len(clusters)
        clusters.append([p])
        pixel_to_cluster[p] = cluster_idx

    return clusters, pixel_to_cluster


def _trace_paths_between_critical(
    adjacency: dict[Pixel, list[Pixel]],
    critical: set[Pixel],
) -> list[list[Pixel]]:
    visited_links: set[tuple[Pixel, Pixel]] = set()
    paths: list[list[Pixel]] = []

    def _edge_key(a: Pixel, b: Pixel) -> tuple[Pixel, Pixel]:
        return (a, b) if a <= b else (b, a)

    for start in tqdm(sorted(critical), desc="Trace skeleton paths", unit="node"):
        for nbr in adjacency.get(start, []):
            key = _edge_key(start, nbr)
            if key in visited_links:
                continue

            path = [start, nbr]
            visited_links.add(key)
            prev = start
            cur = nbr

            while cur not in critical:
                next_candidates = [q for q in adjacency.get(cur, []) if q != prev]
                if not next_candidates:
                    break
                nxt = next_candidates[0]
                key2 = _edge_key(cur, nxt)
                if key2 in visited_links:
                    break
                visited_links.add(key2)
                path.append(nxt)
                prev, cur = cur, nxt

            if len(path) >= 2:
                paths.append(path)

    return paths


def _edge_pixels_oriented(edge: RoadEdge, start_node_id: str) -> list[Pixel]:
    if edge.u == start_node_id:
        return list(edge.pixels)
    if edge.v == start_node_id:
        return list(reversed(edge.pixels))
    raise ValueError(f"Edge {edge.edge_id} is not incident to node {start_node_id}")


def _next_edge_index(edges: list[RoadEdge]) -> int:
    max_idx = 0
    for e in edges:
        if e.edge_id.startswith("R") and e.edge_id[1:].isdigit():
            max_idx = max(max_idx, int(e.edge_id[1:]))
    return max_idx + 1


def _normalize_junction_nodes_by_branch_lengths(
    nodes: list[RoadNode],
    edges: list[RoadEdge],
    *,
    resolution: float,
    min_edge_length_m: float,
    min_branches: int,
    min_branch_len_m: float,
) -> tuple[list[RoadNode], list[RoadEdge]]:
    if not nodes or not edges:
        return nodes, edges

    node_lookup = {n.node_id: n for n in nodes}
    node_kind = {n.node_id: n.kind for n in nodes}
    removed_nodes: set[str] = set()
    active_edges: dict[str, RoadEdge] = {e.edge_id: e for e in edges}
    candidate_node_ids = [n.node_id for n in nodes]
    next_edge_idx = _next_edge_index(edges)

    def _active_incident(nid: str) -> list[str]:
        if nid not in node_lookup:
            return []
        # IMPORTANT: compute incident edges from the current active edge set,
        # not from stale node.incident_edge_ids, so newly merged edges are visible.
        out: list[str] = []
        for eid, edge in active_edges.items():
            if edge.u == nid or edge.v == nid:
                out.append(eid)
        out.sort()
        return out

    changed = True
    while changed:
        changed = False
        for nid in candidate_node_ids:
            if nid in removed_nodes:
                continue

            incident = _active_incident(nid)
            if not incident:
                removed_nodes.add(nid)
                node_kind[nid] = "removed"
                changed = True
                continue

            long_incident = [
                eid
                for eid in incident
                if float(active_edges[eid].length_px * resolution) >= min_branch_len_m
            ]
            long_count = len(long_incident)
            if long_count >= min_branches:
                if node_kind.get(nid) != "junction":
                    node_kind[nid] = "junction"
                    changed = True
                continue
            if long_count >= 3:
                # Even when min_branches is set >3, treat 3+ valid branches as
                # a stable junction to avoid over-deleting true intersections.
                if node_kind.get(nid) != "junction":
                    node_kind[nid] = "junction"
                    changed = True
                continue

            if long_count == 2:
                e1 = active_edges[long_incident[0]]
                e2 = active_edges[long_incident[1]]
                other1 = e1.v if e1.u == nid else e1.u
                other2 = e2.v if e2.u == nid else e2.u

                for eid in incident:
                    if eid in active_edges:
                        active_edges.pop(eid, None)
                        changed = True

                removed_nodes.add(nid)
                node_kind[nid] = "removed"

                if other1 == other2:
                    continue

                p1 = _edge_pixels_oriented(e1, other1)  # other1 -> nid
                p2 = _edge_pixels_oriented(e2, nid)  # nid -> other2
                merged_pixels = p1 + p2[1:] if p2 else p1
                if len(merged_pixels) < 2:
                    continue

                merged_len_px = float(
                    sum(_pixel_step(merged_pixels[i], merged_pixels[i + 1]) for i in range(len(merged_pixels) - 1))
                )
                if merged_len_px * resolution < float(min_edge_length_m):
                    continue

                new_edge_id = f"R{next_edge_idx:05d}"
                next_edge_idx += 1
                active_edges[new_edge_id] = RoadEdge(
                    edge_id=new_edge_id,
                    u=other1,
                    v=other2,
                    pixels=merged_pixels,
                    length_px=merged_len_px,
                )
                changed = True
                continue

            if long_count == 1:
                keep = long_incident[0]
                for eid in incident:
                    if eid == keep:
                        continue
                    if eid in active_edges:
                        active_edges.pop(eid, None)
                        changed = True
                if node_kind.get(nid) != "endpoint":
                    node_kind[nid] = "endpoint"
                    changed = True
                continue

            for eid in incident:
                if eid in active_edges:
                    active_edges.pop(eid, None)
                    changed = True
            removed_nodes.add(nid)
            node_kind[nid] = "removed"

    new_edges = list(active_edges.values())
    new_incident: dict[str, list[str]] = {n.node_id: [] for n in nodes}
    for e in new_edges:
        if e.u in new_incident:
            new_incident[e.u].append(e.edge_id)
        if e.v in new_incident and e.v != e.u:
            new_incident[e.v].append(e.edge_id)

    new_nodes: list[RoadNode] = []
    for n in nodes:
        nid = n.node_id
        if nid in removed_nodes:
            continue
        if node_kind.get(nid) == "removed":
            continue
        incident = new_incident.get(nid, [])
        if not incident:
            continue
        new_nodes.append(
            RoadNode(
                node_id=n.node_id,
                kind=node_kind.get(nid, n.kind),
                y=float(n.y),
                x=float(n.x),
                pixels=list(n.pixels),
                incident_edge_ids=list(incident),
            )
        )

    return new_nodes, new_edges


def build_road_graph(
    skeleton: np.ndarray,
    junction_cluster_eps_px: float,
    junction_min_samples: int,
    resolution: float,
    min_edge_length_m: float = 0.0,
    junction_min_valid_branches: int = 3,
    junction_min_branch_length_m: float = 10.0,
) -> RoadGraph:
    if resolution <= 0:
        raise ValueError("resolution must be > 0")

    pixels, adjacency, degrees = _build_skeleton_adjacency(np.asarray(skeleton, dtype=bool))

    junction_pixels = [p for p in pixels if degrees[p] >= 3]
    clusters, pixel_to_cluster = _cluster_junction_pixels(
        junction_pixels=junction_pixels,
        junction_cluster_eps_px=junction_cluster_eps_px,
        junction_min_samples=junction_min_samples,
    )

    nodes: list[RoadNode] = []
    pixel_to_node: dict[Pixel, str] = {}

    for idx, cluster_pixels in enumerate(clusters):
        ys = np.asarray([p[0] for p in cluster_pixels], dtype=np.float32)
        xs = np.asarray([p[1] for p in cluster_pixels], dtype=np.float32)
        node_id = f"J{idx + 1:04d}"
        node = RoadNode(
            node_id=node_id,
            kind="junction",
            y=float(np.mean(ys)),
            x=float(np.mean(xs)),
            pixels=list(cluster_pixels),
        )
        nodes.append(node)
        for p in cluster_pixels:
            pixel_to_node[p] = node_id

    endpoint_pixels = [p for p in pixels if degrees[p] <= 1 and p not in pixel_to_node]
    for idx, p in enumerate(sorted(endpoint_pixels)):
        node_id = f"E{idx + 1:04d}"
        node = RoadNode(
            node_id=node_id,
            kind="endpoint",
            y=float(p[0]),
            x=float(p[1]),
            pixels=[p],
        )
        nodes.append(node)
        pixel_to_node[p] = node_id

    critical = {p for p in pixels if degrees[p] != 2}
    if not critical and pixels:
        anchor = min(pixels)
        if anchor not in pixel_to_node:
            node_id = f"E{len([n for n in nodes if n.kind == 'endpoint']) + 1:04d}"
            nodes.append(
                RoadNode(
                    node_id=node_id,
                    kind="endpoint",
                    y=float(anchor[0]),
                    x=float(anchor[1]),
                    pixels=[anchor],
                )
            )
            pixel_to_node[anchor] = node_id
        critical = {anchor}

    raw_paths = _trace_paths_between_critical(adjacency=adjacency, critical=critical)

    node_lookup: dict[str, RoadNode] = {n.node_id: n for n in nodes}

    def _fallback_node_for_pixel(pixel: Pixel) -> str:
        if pixel in pixel_to_node:
            return pixel_to_node[pixel]
        node_id = f"E{len([n for n in nodes if n.kind == 'endpoint']) + 1:04d}"
        node = RoadNode(
            node_id=node_id,
            kind="endpoint",
            y=float(pixel[0]),
            x=float(pixel[1]),
            pixels=[pixel],
        )
        nodes.append(node)
        node_lookup[node_id] = node
        pixel_to_node[pixel] = node_id
        return node_id

    edges: list[RoadEdge] = []
    for path in tqdm(raw_paths, desc="Build graph edges", unit="path"):
        start = path[0]
        end = path[-1]
        u = _fallback_node_for_pixel(start)
        v = _fallback_node_for_pixel(end)

        if u == v and node_lookup[u].kind == "junction":
            continue

        length_px = float(sum(_pixel_step(path[i], path[i + 1]) for i in range(len(path) - 1)))
        if length_px <= 0:
            continue
        if length_px * resolution < float(min_edge_length_m):
            continue

        edge_id = f"R{len(edges) + 1:05d}"
        edge = RoadEdge(edge_id=edge_id, u=u, v=v, pixels=path, length_px=length_px)
        edges.append(edge)

        node_lookup[u].incident_edge_ids.append(edge_id)
        if v != u:
            node_lookup[v].incident_edge_ids.append(edge_id)

    if edges:
        valid_nodes = {e.u for e in edges} | {e.v for e in edges}
        nodes = [n for n in nodes if n.node_id in valid_nodes]
    else:
        nodes = []

    return RoadGraph(nodes=nodes, edges=edges, resolution=resolution)


def normalize_graph_junctions_by_branch_lengths(
    graph: RoadGraph,
    *,
    min_edge_length_m: float,
    junction_min_valid_branches: int,
    junction_min_branch_length_m: float,
) -> RoadGraph:
    # Junction validation / normalization by long-branch count:
    # >=3 keep junction; ==2 remove node and merge two long edges;
    # ==1 keep as endpoint; ==0 remove.
    if not graph.nodes or not graph.edges:
        return graph

    min_branches = max(1, int(junction_min_valid_branches))
    min_branch_len_m = max(0.0, float(junction_min_branch_length_m))
    nodes, edges = _normalize_junction_nodes_by_branch_lengths(
        nodes=graph.nodes,
        edges=graph.edges,
        resolution=graph.resolution,
        min_edge_length_m=min_edge_length_m,
        min_branches=min_branches,
        min_branch_len_m=min_branch_len_m,
    )

    if edges:
        valid_nodes = {e.u for e in edges} | {e.v for e in edges}
        nodes = [n for n in nodes if n.node_id in valid_nodes]
    else:
        nodes = []

    return RoadGraph(nodes=nodes, edges=edges, resolution=graph.resolution)


def merge_close_junctions(
    graph: RoadGraph,
    merge_distance_m: float,
) -> tuple[RoadGraph, int]:
    if merge_distance_m <= 0 or not graph.nodes or not graph.edges:
        return graph, 0
    if graph.resolution <= 0:
        return graph, 0

    node_lookup = {n.node_id: n for n in graph.nodes}
    junction_ids = [n.node_id for n in graph.nodes if n.kind == "junction"]
    if len(junction_ids) < 2:
        return graph, 0

    eps_px = max(0.5, float(merge_distance_m / graph.resolution))
    coords_xy = np.asarray(
        [[float(node_lookup[nid].x), float(node_lookup[nid].y)] for nid in junction_ids],
        dtype=np.float32,
    )
    labels = DBSCAN(eps=eps_px, min_samples=1).fit(coords_xy).labels_

    clusters: dict[int, list[str]] = {}
    for nid, label in zip(junction_ids, labels):
        clusters.setdefault(int(label), []).append(nid)

    merge_to_keep: dict[str, str] = {}
    merged_nodes: dict[str, RoadNode] = {}

    for cluster_node_ids in clusters.values():
        if len(cluster_node_ids) == 1:
            nid = cluster_node_ids[0]
            n = node_lookup[nid]
            merged_nodes[nid] = RoadNode(
                node_id=n.node_id,
                kind=n.kind,
                y=float(n.y),
                x=float(n.x),
                pixels=list(n.pixels),
            )
            continue

        keep_id = sorted(
            cluster_node_ids,
            key=lambda nid: (-len(node_lookup[nid].incident_edge_ids), nid),
        )[0]
        keep_node = node_lookup[keep_id]

        merged_pixels: list[Pixel] = []
        x_values: list[float] = []
        y_values: list[float] = []
        for nid in cluster_node_ids:
            n = node_lookup[nid]
            merged_pixels.extend(n.pixels)
            x_values.append(float(n.x))
            y_values.append(float(n.y))
            if nid != keep_id:
                merge_to_keep[nid] = keep_id

        merged_nodes[keep_id] = RoadNode(
            node_id=keep_node.node_id,
            kind="junction",
            y=float(np.mean(np.asarray(y_values, dtype=np.float32))),
            x=float(np.mean(np.asarray(x_values, dtype=np.float32))),
            pixels=merged_pixels,
        )

    # Keep non-junction nodes as-is.
    for n in graph.nodes:
        if n.kind == "junction":
            continue
        merged_nodes[n.node_id] = RoadNode(
            node_id=n.node_id,
            kind=n.kind,
            y=float(n.y),
            x=float(n.x),
            pixels=list(n.pixels),
        )

    rewired_edges: list[RoadEdge] = []
    for e in graph.edges:
        new_u = merge_to_keep.get(e.u, e.u)
        new_v = merge_to_keep.get(e.v, e.v)
        # Remove internal connector between merged junctions.
        if new_u == new_v:
            continue
        rewired_edges.append(
            RoadEdge(
                edge_id=e.edge_id,
                u=new_u,
                v=new_v,
                pixels=list(e.pixels),
                length_px=float(e.length_px),
            )
        )

    valid_nodes = {e.u for e in rewired_edges} | {e.v for e in rewired_edges}
    for n in merged_nodes.values():
        n.incident_edge_ids = []
    for e in rewired_edges:
        if e.u in merged_nodes:
            merged_nodes[e.u].incident_edge_ids.append(e.edge_id)
        if e.v in merged_nodes and e.v != e.u:
            merged_nodes[e.v].incident_edge_ids.append(e.edge_id)

    ordered_node_ids: list[str] = []
    seen: set[str] = set()
    for old_n in graph.nodes:
        mapped = merge_to_keep.get(old_n.node_id, old_n.node_id)
        if mapped in valid_nodes and mapped not in seen and mapped in merged_nodes:
            seen.add(mapped)
            ordered_node_ids.append(mapped)

    final_nodes = [merged_nodes[nid] for nid in ordered_node_ids]
    merged_junction_nodes = int(len(merge_to_keep))
    return (
        RoadGraph(nodes=final_nodes, edges=rewired_edges, resolution=graph.resolution),
        merged_junction_nodes,
    )


def compute_distance_map_m(mask: np.ndarray, resolution: float) -> np.ndarray:
    return distance_transform_edt(np.asarray(mask, dtype=bool)) * float(resolution)


def _sample_distance_values_on_pixels(
    pixels: list[Pixel],
    distance_map_m: np.ndarray,
    resolution: float,
    start_m: float,
    end_m: float,
) -> list[float]:
    if not pixels:
        return []

    h, w = distance_map_m.shape[:2]
    values: list[float] = []
    cum_m = 0.0

    prev = pixels[0]
    for idx, p in enumerate(pixels):
        if idx > 0:
            cum_m += _pixel_step(prev, p) * resolution
            prev = p

        if cum_m < start_m:
            continue
        if cum_m > end_m:
            break

        y, x = p
        if 0 <= y < h and 0 <= x < w:
            v = float(distance_map_m[y, x])
            if v > 0:
                values.append(v)

    return values


def estimate_junction_radii_m(
    graph: RoadGraph,
    distance_map_m: np.ndarray,
    sample_start_m: float,
    sample_end_m: float,
    radius_scale: float,
    radius_margin_m: float,
    radius_min_m: float,
    radius_max_m: float,
) -> dict[str, float]:
    edge_map = {e.edge_id: e for e in graph.edges}

    start_m = max(0.0, float(sample_start_m))
    end_m = max(start_m, float(sample_end_m))

    radii: dict[str, float] = {}
    for node in tqdm(graph.nodes, desc="Estimate junction radii", unit="node"):
        if node.kind != "junction":
            continue

        branch_values: list[float] = []
        for edge_id in node.incident_edge_ids:
            edge = edge_map[edge_id]
            if edge.u == node.node_id:
                ordered_pixels = edge.pixels
            else:
                ordered_pixels = list(reversed(edge.pixels))
            branch_values.extend(
                _sample_distance_values_on_pixels(
                    pixels=ordered_pixels,
                    distance_map_m=distance_map_m,
                    resolution=graph.resolution,
                    start_m=start_m,
                    end_m=end_m,
                )
            )

        if not branch_values:
            for y, x in node.pixels:
                v = float(distance_map_m[y, x])
                if v > 0:
                    branch_values.append(v)

        if branch_values:
            half_width_m = float(np.median(np.asarray(branch_values, dtype=np.float32)))
        else:
            half_width_m = float(max(0.0, radius_min_m / max(1e-6, radius_scale)))

        radius = float(radius_scale * half_width_m + radius_margin_m)
        radius = float(np.clip(radius, radius_min_m, radius_max_m))
        radii[node.node_id] = radius

    return radii


def split_line_with_overlap(line: LineString, max_length: float, overlap: float) -> list[LineString]:
    if max_length <= 0:
        raise ValueError("max_length must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= max_length:
        raise ValueError("overlap must be smaller than max_length")

    length = float(line.length)
    if length <= 0:
        return []
    if length <= max_length:
        return [line]

    pieces: list[LineString] = []
    stride = max_length - overlap
    start = 0.0

    while start < length:
        end = min(start + max_length, length)
        segment = substring(line, start, end)
        if isinstance(segment, LineString) and segment.length > 0:
            pieces.append(segment)
        if end >= length:
            break
        start += stride

    return pieces


def build_segment_partitions(
    graph: RoadGraph,
    resolution: float,
    segment_max_length_m: float,
    segment_overlap_m: float,
    endpoint_extension_m: float = 0.0,
) -> list[SegmentPartition]:
    if resolution <= 0:
        raise ValueError("resolution must be > 0")

    max_len_px = _meters_to_pixels(segment_max_length_m, resolution)
    overlap_px = _meters_to_pixels(segment_overlap_m, resolution)
    endpoint_extension_px = max(0.0, _meters_to_pixels(endpoint_extension_m, resolution))
    node_kind = {n.node_id: n.kind for n in graph.nodes}

    partitions: list[SegmentPartition] = []
    for edge in graph.edges:
        coords_xy = [(float(x), float(y)) for y, x in edge.pixels]
        if len(coords_xy) < 2:
            continue
        line = LineString(coords_xy)
        if line.length <= 0:
            continue

        pieces = split_line_with_overlap(line=line, max_length=max_len_px, overlap=overlap_px)
        for seg_idx, piece in enumerate(pieces):
            if piece.length <= 0:
                continue
            extend_start_px = (
                endpoint_extension_px
                if seg_idx == 0 and node_kind.get(edge.u, "endpoint") == "endpoint"
                else 0.0
            )
            extend_end_px = (
                endpoint_extension_px
                if seg_idx == (len(pieces) - 1) and node_kind.get(edge.v, "endpoint") == "endpoint"
                else 0.0
            )
            piece = extend_linestring_at_ends(
                line=piece,
                extend_start_px=extend_start_px,
                extend_end_px=extend_end_px,
            )
            if piece.length <= 0:
                continue
            partition_id = f"P{len(partitions) + 1:04d}"
            partitions.append(
                SegmentPartition(
                    partition_id=partition_id,
                    edge_id=edge.edge_id,
                    segment_index=seg_idx,
                    centerline=piece,
                    length_m=float(piece.length * resolution),
                )
            )
    return partitions


def _sample_distance_values_on_line(
    line: LineString,
    distance_map_m: np.ndarray,
) -> list[float]:
    length = float(line.length)
    if length <= 0:
        return []

    h, w = distance_map_m.shape[:2]
    n_samples = max(8, int(np.ceil(length)))
    dists = np.linspace(0.0, length, num=n_samples)
    values: list[float] = []
    for d in dists:
        p = line.interpolate(float(d))
        x = int(round(float(p.x)))
        y = int(round(float(p.y)))
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))
        v = float(distance_map_m[y, x])
        if v > 0:
            values.append(v)
    return values


def _endpoint_extension_direction(
    coords: list[tuple[float, float]],
    *,
    at_start: bool,
) -> tuple[float, float] | None:
    if len(coords) < 2:
        return None

    if at_start:
        anchor = np.asarray(coords[0], dtype=np.float64)
        for i in range(1, len(coords)):
            ref = np.asarray(coords[i], dtype=np.float64)
            vec = anchor - ref
            norm = float(np.hypot(vec[0], vec[1]))
            if norm > 1e-6:
                return float(vec[0] / norm), float(vec[1] / norm)
        return None

    anchor = np.asarray(coords[-1], dtype=np.float64)
    for i in range(len(coords) - 2, -1, -1):
        ref = np.asarray(coords[i], dtype=np.float64)
        vec = anchor - ref
        norm = float(np.hypot(vec[0], vec[1]))
        if norm > 1e-6:
            return float(vec[0] / norm), float(vec[1] / norm)
    return None


def _global_direction_from_all_points(
    coords: list[tuple[float, float]],
) -> tuple[float, float] | None:
    """
    Estimate one global direction from all skeleton points in this segment.

    We fit a principal axis (PCA on XY coordinates), then align its sign with
    the start->end direction of the polyline.
    """
    if len(coords) < 2:
        return None

    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return None

    center = np.mean(arr, axis=0)
    centered = arr - center
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    norm = float(np.hypot(axis[0], axis[1]))
    if norm <= 1e-6:
        return None
    axis = axis / norm

    end_vec = arr[-1] - arr[0]
    end_norm = float(np.hypot(end_vec[0], end_vec[1]))
    if end_norm > 1e-6 and float(np.dot(axis, end_vec)) < 0:
        axis = -axis

    return float(axis[0]), float(axis[1])


def extend_linestring_at_ends(
    line: LineString,
    *,
    extend_start_px: float,
    extend_end_px: float,
) -> LineString:
    if line.length <= 0:
        return line

    start_ext = max(0.0, float(extend_start_px))
    end_ext = max(0.0, float(extend_end_px))
    if start_ext <= 0 and end_ext <= 0:
        return line

    coords = [(float(x), float(y)) for x, y in line.coords]
    if len(coords) < 2:
        return line

    out_coords = list(coords)
    global_dir = _global_direction_from_all_points(coords)

    if start_ext > 0:
        # Start endpoint extends opposite to the global segment direction.
        d = (
            (-global_dir[0], -global_dir[1]) if global_dir is not None else _endpoint_extension_direction(coords, at_start=True)
        )
        if d is not None:
            sx, sy = coords[0]
            out_coords = [(sx + d[0] * start_ext, sy + d[1] * start_ext)] + out_coords

    if end_ext > 0:
        # End endpoint extends along the global segment direction.
        d = global_dir if global_dir is not None else _endpoint_extension_direction(coords, at_start=False)
        if d is not None:
            ex, ey = coords[-1]
            out_coords = out_coords + [(ex + d[0] * end_ext, ey + d[1] * end_ext)]

    ext_line = LineString(out_coords)
    if ext_line.length <= 0:
        return line
    return ext_line


def build_slice_polygons(
    graph: RoadGraph,
    distance_map_m: np.ndarray,
    resolution: float,
    junction_radii_m: dict[str, float],
    segment_max_length_m: float,
    segment_overlap_m: float,
    segment_width_margin_m: float,
    min_segment_half_width_m: float,
    endpoint_extension_m: float = 0.0,
) -> list[SlicePolygon]:
    if resolution <= 0:
        raise ValueError("resolution must be > 0")

    slices: list[SlicePolygon] = []

    for node in tqdm(graph.nodes, desc="Build junction polygons", unit="node"):
        if node.kind != "junction":
            continue
        radius_m = float(junction_radii_m.get(node.node_id, max(1.0, min_segment_half_width_m * 1.5)))
        radius_px = _meters_to_pixels(radius_m, resolution)
        if radius_px <= 0:
            continue
        # Use rectangle slice for junctions (instead of circle) to improve
        # downstream readability and slicing consistency.
        poly = box(
            float(node.x) - float(radius_px),
            float(node.y) - float(radius_px),
            float(node.x) + float(radius_px),
            float(node.y) + float(radius_px),
        )
        if poly.is_empty:
            continue
        slices.append(
            SlicePolygon(
                slice_id=f"junction_{node.node_id}",
                kind="junction",
                source_id=node.node_id,
                geometry=poly,
            )
        )

    max_len_px = _meters_to_pixels(segment_max_length_m, resolution)
    overlap_px = _meters_to_pixels(segment_overlap_m, resolution)
    endpoint_extension_px = max(0.0, _meters_to_pixels(endpoint_extension_m, resolution))
    node_kind = {n.node_id: n.kind for n in graph.nodes}

    for edge in tqdm(graph.edges, desc="Build road-segment polygons", unit="edge"):
        coords_xy = [(float(x), float(y)) for y, x in edge.pixels]
        if len(coords_xy) < 2:
            continue

        line = LineString(coords_xy)
        if line.length <= 0:
            continue

        pieces = split_line_with_overlap(line=line, max_length=max_len_px, overlap=overlap_px)
        for idx, piece in enumerate(pieces):
            if piece.length <= 0:
                continue
            extend_start_px = (
                endpoint_extension_px
                if idx == 0 and node_kind.get(edge.u, "endpoint") == "endpoint"
                else 0.0
            )
            extend_end_px = (
                endpoint_extension_px
                if idx == (len(pieces) - 1) and node_kind.get(edge.v, "endpoint") == "endpoint"
                else 0.0
            )
            piece = extend_linestring_at_ends(
                line=piece,
                extend_start_px=extend_start_px,
                extend_end_px=extend_end_px,
            )
            if piece.length <= 0:
                continue

            local_half_width_values = _sample_distance_values_on_line(piece, distance_map_m)
            if local_half_width_values:
                half_width_m = float(np.median(np.asarray(local_half_width_values, dtype=np.float32)))
            else:
                half_width_m = float(min_segment_half_width_m)
            # User rule: slice width = 2 * (half_road_width + extra_margin),
            # so buffer half-width is (half_road_width + extra_margin).
            half_width_m = max(min_segment_half_width_m, half_width_m + segment_width_margin_m)

            half_width_px = _meters_to_pixels(half_width_m, resolution)
            # Square-cap buffer: straight segments are rendered as rectangle-like
            # road strips with clearer boundaries in the final overlay.
            poly = piece.buffer(half_width_px, cap_style=3, join_style=2)
            if poly.is_empty:
                continue

            slices.append(
                SlicePolygon(
                    slice_id=f"road_{edge.edge_id}_{idx:03d}",
                    kind="road_segment",
                    source_id=edge.edge_id,
                    geometry=poly,
                )
            )

    return slices


def _iter_polygons(geometry: Polygon | MultiPolygon):
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        for g in geometry.geoms:
            if isinstance(g, Polygon):
                yield g


def _coords_to_cv2_points(coords: list[tuple[float, float]], w: int, h: int) -> np.ndarray:
    arr = np.asarray(coords, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 3:
        return np.zeros((0, 1, 2), dtype=np.int32)
    xs = np.clip(np.round(arr[:, 0]), 0, w - 1).astype(np.int32)
    ys = np.clip(np.round(arr[:, 1]), 0, h - 1).astype(np.int32)
    pts = np.stack([xs, ys], axis=1)
    return pts.reshape(-1, 1, 2)


def _draw_star_nodes(
    canvas: np.ndarray,
    nodes: list[RoadNode],
    *,
    scale_x: float,
    scale_y: float,
    draw_node_labels: bool,
    junction_marker_size: int,
    endpoint_marker_size: int,
) -> None:
    h, w = canvas.shape[:2]
    for node in nodes:
        cx = int(round(node.x * scale_x))
        cy = int(round(node.y * scale_y))
        if not (0 <= cx < w and 0 <= cy < h):
            continue

        outer_size = junction_marker_size if node.kind == "junction" else endpoint_marker_size
        inner_size = max(16, outer_size - 10)
        inner_color = (0, 0, 255) if node.kind == "junction" else (0, 255, 255)
        ring_color = (255, 255, 255) if node.kind == "junction" else (210, 210, 210)
        marker_thickness = max(4, outer_size // 8)
        label_font_scale = float(np.clip(outer_size / 10.0, 2.0, 14.0))
        label_thickness = max(4, int(round(label_font_scale * 2.0)))

        # Filled disk + white ring improves visibility on dense overlays.
        cv2.circle(
            canvas,
            (cx, cy),
            max(3, outer_size // 2),
            inner_color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            canvas,
            (cx, cy),
            max(4, outer_size // 2),
            ring_color,
            thickness=max(2, marker_thickness),
            lineType=cv2.LINE_AA,
        )

        cv2.drawMarker(
            canvas,
            (cx, cy),
            (255, 255, 255),
            markerType=cv2.MARKER_STAR,
            markerSize=outer_size,
            thickness=marker_thickness,
            line_type=cv2.LINE_AA,
        )
        cv2.drawMarker(
            canvas,
            (cx, cy),
            inner_color,
            markerType=cv2.MARKER_STAR,
            markerSize=inner_size,
            thickness=marker_thickness,
            line_type=cv2.LINE_AA,
        )

        if draw_node_labels:
            label_dx = max(10, outer_size // 2)
            label_dy = max(10, outer_size // 2)
            cv2.putText(
                canvas,
                node.node_id,
                (cx + label_dx, cy - label_dy),
                cv2.FONT_HERSHEY_SIMPLEX,
                label_font_scale,
                (0, 0, 0),
                label_thickness + 3,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                node.node_id,
                (cx + label_dx, cy - label_dy),
                cv2.FONT_HERSHEY_SIMPLEX,
                label_font_scale,
                inner_color,
                label_thickness,
                cv2.LINE_AA,
            )


def _draw_graph_legend(canvas: np.ndarray) -> None:
    h, _ = canvas.shape[:2]
    origin_x, origin_y = 24, 24
    box_w, box_h = 360, 112

    overlay = canvas.copy()
    cv2.rectangle(
        overlay,
        (origin_x, origin_y),
        (origin_x + box_w, origin_y + box_h),
        (25, 25, 25),
        thickness=-1,
    )
    cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, dst=canvas)
    cv2.rectangle(
        canvas,
        (origin_x, origin_y),
        (origin_x + box_w, origin_y + box_h),
        (180, 180, 180),
        thickness=1,
    )

    y1 = origin_y + 32
    cv2.line(canvas, (origin_x + 18, y1), (origin_x + 90, y1), (80, 220, 80), 6, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "Edge",
        (origin_x + 110, y1 + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )

    y2 = origin_y + 68
    cv2.circle(canvas, (origin_x + 54, y2), 10, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
    cv2.putText(
        canvas,
        "Junction Node",
        (origin_x + 110, y2 + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )

    y3 = origin_y + 100
    cv2.circle(canvas, (origin_x + 54, y3), 10, (0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
    cv2.putText(
        canvas,
        "Endpoint Node",
        (origin_x + 110, y3 + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )


def draw_clean_mask_black(cleaned_mask: np.ndarray, output_path: Path) -> None:
    h, w = cleaned_mask.shape[:2]
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[cleaned_mask] = (220, 220, 220)
    if not cv2_imwrite_any(output_path, vis):
        raise RuntimeError(f"Failed to save visualization: {output_path}")


def draw_skeleton_black(
    skeleton_mask: np.ndarray,
    nodes: list[RoadNode],
    output_path: Path,
    *,
    scale_x: float,
    scale_y: float,
    out_h: int,
    out_w: int,
    line_thickness_px: int = 4,
    endpoint_radius_px: int = 8,
) -> None:
    vis = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    skel_big = _resize_bool_mask(skeleton_mask, out_h=out_h, out_w=out_w)
    if line_thickness_px > 1:
        k = max(1, int(line_thickness_px))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        skel_big = cv2.dilate(skel_big.astype(np.uint8), kernel, iterations=1) > 0
    vis[skel_big] = (255, 200, 0)

    for node in nodes:
        if node.kind != "endpoint":
            continue
        cx = int(round(node.x * scale_x))
        cy = int(round(node.y * scale_y))
        if 0 <= cx < out_w and 0 <= cy < out_h:
            cv2.circle(vis, (cx, cy), int(endpoint_radius_px), (0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

    if not cv2_imwrite_any(output_path, vis):
        raise RuntimeError(f"Failed to save visualization: {output_path}")


def draw_graph_black(
    graph: RoadGraph,
    output_path: Path,
    *,
    scale_x: float,
    scale_y: float,
    out_h: int,
    out_w: int,
    draw_node_labels: bool,
    edge_thickness_px: int = 10,
) -> None:
    vis = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    auto_junction_marker = int(np.clip(max(out_h, out_w) / 26.0, 36, 120))
    auto_endpoint_marker = int(np.clip(max(out_h, out_w) / 32.0, 28, 96))

    for edge in graph.edges:
        pts = []
        for y, x in edge.pixels:
            px = int(round(float(x) * scale_x))
            py = int(round(float(y) * scale_y))
            if 0 <= px < out_w and 0 <= py < out_h:
                pts.append((px, py))
        if len(pts) >= 2:
            arr = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                vis,
                [arr],
                isClosed=False,
                color=(80, 220, 80),
                thickness=max(3, int(edge_thickness_px)),
                lineType=cv2.LINE_AA,
            )

    _draw_star_nodes(
        canvas=vis,
        nodes=graph.nodes,
        scale_x=scale_x,
        scale_y=scale_y,
        draw_node_labels=draw_node_labels,
        junction_marker_size=auto_junction_marker,
        endpoint_marker_size=auto_endpoint_marker,
    )
    _draw_graph_legend(vis)

    if not cv2_imwrite_any(output_path, vis):
        raise RuntimeError(f"Failed to save visualization: {output_path}")


def draw_partition_centerlines_overlay(
    base_bgr: np.ndarray,
    partitions: list[SegmentPartition],
    nodes: list[RoadNode],
    output_path: Path,
    draw_node_labels: bool,
    *,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    line_thickness_px: int = 30,
    show_partition_labels: bool = True,
) -> None:
    if base_bgr.ndim != 3 or base_bgr.shape[2] != 3:
        raise ValueError(f"Expected base image shape (H,W,3), got {base_bgr.shape}")

    h, w = base_bgr.shape[:2]
    vis = base_bgr.copy()
    auto_junction_marker = int(np.clip(max(h, w) / 10.0, 120, 520))
    auto_endpoint_marker = int(np.clip(max(h, w) / 13.0, 100, 420))

    for part in tqdm(partitions, desc="Draw partition centerlines", unit="line"):
        geom = part.centerline
        if scale_x != 1.0 or scale_y != 1.0:
            geom = scale_geometry(geom, xfact=scale_x, yfact=scale_y, origin=(0.0, 0.0))
        if not isinstance(geom, LineString) or geom.length <= 0:
            continue

        pts = _coords_to_cv2_points(list(geom.coords), w=w, h=h)
        if pts.shape[0] < 2:
            continue

        color = _stable_color_bgr(part.partition_id)
        cv2.polylines(
            vis,
            [pts],
            isClosed=False,
            color=(255, 255, 255),
            thickness=max(12, int(line_thickness_px) + 12),
            lineType=cv2.LINE_AA,
        )
        cv2.polylines(
            vis,
            [pts],
            isClosed=False,
            color=color,
            thickness=max(8, int(line_thickness_px)),
            lineType=cv2.LINE_AA,
        )

        if show_partition_labels:
            mid = geom.interpolate(float(geom.length * 0.5))
            mx = int(np.clip(round(float(mid.x)), 0, w - 1))
            my = int(np.clip(round(float(mid.y)), 0, h - 1))
            label = part.partition_id
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = float(np.clip(max(h, w) / 1200.0, 3.2, 8.0))
            thickness = max(6, int(round(font_scale * 3.0)))
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            pad = max(14, int(round(font_scale * 8)))
            x0 = int(np.clip(mx - tw // 2 - pad, 0, max(0, w - (tw + 2 * pad))))
            y0 = int(np.clip(my - th - (2 * pad), 0, max(0, h - (th + baseline + 2 * pad))))
            cv2.rectangle(
                vis,
                (x0, y0),
                (x0 + tw + 2 * pad, y0 + th + baseline + 2 * pad),
                (0, 0, 0),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
            cv2.rectangle(
                vis,
                (x0, y0),
                (x0 + tw + 2 * pad, y0 + th + baseline + 2 * pad),
                color,
                thickness=max(3, int(round(font_scale * 1.3))),
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                label,
                (x0 + pad, y0 + th + pad // 2),
                font,
                font_scale,
                (0, 0, 0),
                thickness + 3,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                label,
                (x0 + pad, y0 + th + pad // 2),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

    _draw_star_nodes(
        canvas=vis,
        nodes=nodes,
        scale_x=scale_x,
        scale_y=scale_y,
        draw_node_labels=draw_node_labels,
        junction_marker_size=auto_junction_marker,
        endpoint_marker_size=auto_endpoint_marker,
    )

    if not cv2_imwrite_any(output_path, vis):
        raise RuntimeError(f"Failed to save visualization: {output_path}")


def _save_bool_mask_png(mask: np.ndarray, output_path: Path) -> None:
    vis = (np.asarray(mask, dtype=np.uint8) * 255).astype(np.uint8)
    if not cv2_imwrite_any(output_path, vis):
        raise RuntimeError(f"Failed to save image: {output_path}")


def _save_graph_tables(graph: RoadGraph, output_dir: Path) -> None:
    nodes_csv = output_dir / "nodes.csv"
    edges_csv = output_dir / "edges.csv"

    with nodes_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "node_id",
                "kind",
                "x_px",
                "y_px",
                "x_m",
                "y_m",
                "pixel_count",
                "degree",
            ]
        )
        for n in graph.nodes:
            writer.writerow(
                [
                    n.node_id,
                    n.kind,
                    f"{n.x:.3f}",
                    f"{n.y:.3f}",
                    f"{n.x * graph.resolution:.3f}",
                    f"{n.y * graph.resolution:.3f}",
                    len(n.pixels),
                    len(n.incident_edge_ids),
                ]
            )

    with edges_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "edge_id",
                "u",
                "v",
                "length_px",
                "length_m",
                "num_pixels",
            ]
        )
        for e in graph.edges:
            writer.writerow(
                [
                    e.edge_id,
                    e.u,
                    e.v,
                    f"{e.length_px:.3f}",
                    f"{e.length_px * graph.resolution:.3f}",
                    len(e.pixels),
                ]
            )


def _save_slice_json(slices: list[SlicePolygon], resolution: float, output_path: Path) -> None:
    records = []
    for sp in slices:
        records.append(
            {
                "slice_id": sp.slice_id,
                "kind": sp.kind,
                "source_id": sp.source_id,
                "area_px2": float(sp.geometry.area),
                "area_m2": float(sp.geometry.area * resolution * resolution),
                "geometry_wkt": sp.geometry.wkt,
            }
        )
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_base_canvas(
    cleaned_mask: np.ndarray,
    base_image_path: Path | None,
) -> np.ndarray:
    h, w = cleaned_mask.shape[:2]

    if base_image_path is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[cleaned_mask] = (60, 60, 60)
        return canvas

    base = cv2_imread_any(base_image_path, flags=cv2.IMREAD_UNCHANGED)
    if base is None:
        raise RuntimeError(f"Failed to read base image: {base_image_path}")

    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    elif base.ndim == 3 and base.shape[2] == 4:
        base = base[:, :, :3]

    if base.shape[0] != h or base.shape[1] != w:
        raise ValueError(
            f"base image shape {base.shape[:2]} does not match mask shape {(h, w)}"
        )

    return np.asarray(base, dtype=np.uint8)

