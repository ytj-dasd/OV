from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, label
from shapely.affinity import scale as scale_geometry
from shapely.geometry import LineString, MultiPolygon, Polygon, box
from shapely.ops import substring
from skimage.morphology import remove_small_holes, remove_small_objects, skeletonize


DEFAULT_MPP = 0.08

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


def _meters_to_pixels(value_m: float, resolution: float) -> float:
    return float(value_m / resolution)


def _area_m2_to_px(area_m2: float, resolution: float) -> int:
    return int(max(1.0, round(area_m2 / (resolution * resolution))))


def downsample_mask_for_topology(
    mask: np.ndarray,
    *,
    source_resolution: float,
    topology_resolution: float,
) -> tuple[np.ndarray, float, float, float]:
    src = np.asarray(mask, dtype=bool)
    if topology_resolution <= source_resolution:
        return src, float(source_resolution), 1.0, 1.0

    src_h, src_w = src.shape[:2]
    scale = float(source_resolution / topology_resolution)
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


def _pixel_step(a: Pixel, b: Pixel) -> float:
    dy = abs(a[0] - b[0])
    dx = abs(a[1] - b[1])
    return float(np.hypot(dy, dx))


def _cv2_read_any(path: Path) -> np.ndarray | None:
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None


def load_binary_mask(mask_path: Path | str, mask_key: str = "union_mask") -> np.ndarray:
    mask_path = Path(mask_path).expanduser()
    if not mask_path.exists():
        raise FileNotFoundError(f"mask file not found: {mask_path}")

    if mask_path.suffix.lower() == ".npz":
        data = np.load(mask_path, allow_pickle=True)
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
        img = _cv2_read_any(mask_path)
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

            if length_px > prune_length_px:
                continue
            cur_degree = degrees.get(cur, 0)
            if cur_degree >= 3:
                to_remove.update(path[:-1])
            else:
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
    image_shape: tuple[int, int],
    junction_cluster_eps_px: float,
) -> tuple[list[list[Pixel]], dict[Pixel, int]]:
    if not junction_pixels:
        return [], {}

    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for y, x in junction_pixels:
        mask[y, x] = 1

    radius = max(0, int(round(junction_cluster_eps_px)))
    if radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)

    labeled, num = label(mask > 0, structure=np.ones((3, 3), dtype=np.uint8))
    clusters: list[list[Pixel]] = []
    pixel_to_cluster: dict[Pixel, int] = {}
    buckets: dict[int, list[Pixel]] = {}
    for p in junction_pixels:
        lb = int(labeled[p[0], p[1]])
        if lb <= 0:
            continue
        buckets.setdefault(lb, []).append(p)

    for _, pixels in sorted(buckets.items()):
        idx = len(clusters)
        clusters.append(pixels)
        for p in pixels:
            pixel_to_cluster[p] = idx
    return clusters, pixel_to_cluster


def _trace_paths_between_critical(
    adjacency: dict[Pixel, list[Pixel]],
    critical: set[Pixel],
) -> list[list[Pixel]]:
    visited_links: set[tuple[Pixel, Pixel]] = set()
    paths: list[list[Pixel]] = []

    def edge_key(a: Pixel, b: Pixel) -> tuple[Pixel, Pixel]:
        return (a, b) if a <= b else (b, a)

    for start in sorted(critical):
        for nbr in adjacency.get(start, []):
            key = edge_key(start, nbr)
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
                key2 = edge_key(cur, nxt)
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

    def active_incident(nid: str) -> list[str]:
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

            incident = active_incident(nid)
            if not incident:
                removed_nodes.add(nid)
                node_kind[nid] = "removed"
                changed = True
                continue

            long_incident = [
                eid for eid in incident if float(active_edges[eid].length_px * resolution) >= min_branch_len_m
            ]
            long_count = len(long_incident)

            if long_count >= max(3, min_branches):
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
                    active_edges.pop(eid, None)
                removed_nodes.add(nid)
                node_kind[nid] = "removed"
                changed = True

                if other1 == other2:
                    continue
                p1 = _edge_pixels_oriented(e1, other1)
                p2 = _edge_pixels_oriented(e2, nid)
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
                continue

            if long_count == 1:
                keep = long_incident[0]
                for eid in incident:
                    if eid != keep:
                        active_edges.pop(eid, None)
                if node_kind.get(nid) != "endpoint":
                    node_kind[nid] = "endpoint"
                    changed = True
                continue

            for eid in incident:
                active_edges.pop(eid, None)
            removed_nodes.add(nid)
            node_kind[nid] = "removed"
            changed = True

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
        if nid in removed_nodes or node_kind.get(nid) == "removed":
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
    resolution: float,
    *,
    min_edge_length_m: float = 0.0,
) -> RoadGraph:
    pixels, adjacency, degrees = _build_skeleton_adjacency(np.asarray(skeleton, dtype=bool))
    junction_pixels = [p for p in pixels if degrees[p] >= 3]
    clusters, _ = _cluster_junction_pixels(junction_pixels, skeleton.shape[:2], junction_cluster_eps_px)

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
        node = RoadNode(node_id=node_id, kind="endpoint", y=float(p[0]), x=float(p[1]), pixels=[p])
        nodes.append(node)
        pixel_to_node[p] = node_id

    critical = {p for p in pixels if degrees[p] != 2}
    if not critical and pixels:
        anchor = min(pixels)
        node_id = "E0001"
        nodes.append(RoadNode(node_id=node_id, kind="endpoint", y=float(anchor[0]), x=float(anchor[1]), pixels=[anchor]))
        pixel_to_node[anchor] = node_id
        critical = {anchor}

    raw_paths = _trace_paths_between_critical(adjacency, critical)
    node_lookup: dict[str, RoadNode] = {n.node_id: n for n in nodes}

    def fallback_node_for_pixel(pixel: Pixel) -> str:
        if pixel in pixel_to_node:
            return pixel_to_node[pixel]
        node_id = f"E{len([n for n in nodes if n.kind == 'endpoint']) + 1:04d}"
        node = RoadNode(node_id=node_id, kind="endpoint", y=float(pixel[0]), x=float(pixel[1]), pixels=[pixel])
        nodes.append(node)
        node_lookup[node_id] = node
        pixel_to_node[pixel] = node_id
        return node_id

    edges: list[RoadEdge] = []
    for path in raw_paths:
        start = path[0]
        end = path[-1]
        u = fallback_node_for_pixel(start)
        v = fallback_node_for_pixel(end)
        if u == v and node_lookup[u].kind == "junction":
            continue

        length_px = float(sum(_pixel_step(path[i], path[i + 1]) for i in range(len(path) - 1)))
        if length_px <= 0 or length_px * resolution < float(min_edge_length_m):
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
    if not graph.nodes or not graph.edges:
        return graph
    nodes, edges = _normalize_junction_nodes_by_branch_lengths(
        nodes=graph.nodes,
        edges=graph.edges,
        resolution=graph.resolution,
        min_edge_length_m=min_edge_length_m,
        min_branches=max(1, int(junction_min_valid_branches)),
        min_branch_len_m=max(0.0, float(junction_min_branch_length_m)),
    )
    if edges:
        valid_nodes = {e.u for e in edges} | {e.v for e in edges}
        nodes = [n for n in nodes if n.node_id in valid_nodes]
    else:
        nodes = []
    return RoadGraph(nodes=nodes, edges=edges, resolution=graph.resolution)


def prune_graph_short_leaf_edges(graph: RoadGraph, max_leaf_length_m: float) -> RoadGraph:
    if max_leaf_length_m <= 0 or not graph.nodes or not graph.edges:
        return graph

    node_lookup = {
        n.node_id: RoadNode(
            node_id=n.node_id,
            kind=n.kind,
            y=float(n.y),
            x=float(n.x),
            pixels=list(n.pixels),
            incident_edge_ids=list(n.incident_edge_ids),
        )
        for n in graph.nodes
    }
    active_edges: dict[str, RoadEdge] = {
        e.edge_id: RoadEdge(
            edge_id=e.edge_id,
            u=e.u,
            v=e.v,
            pixels=list(e.pixels),
            length_px=float(e.length_px),
        )
        for e in graph.edges
    }

    def build_incident_map() -> dict[str, list[str]]:
        incident = {nid: [] for nid in node_lookup}
        for eid, edge in active_edges.items():
            if edge.u in incident:
                incident[edge.u].append(eid)
            if edge.v in incident and edge.v != edge.u:
                incident[edge.v].append(eid)
        for eids in incident.values():
            eids.sort()
        return incident

    while True:
        incident = build_incident_map()
        leaf_edge_ids: set[str] = set()
        for node_id, edge_ids in incident.items():
            if len(edge_ids) < 3:
                continue
            sorted_edge_ids = sorted(
                edge_ids,
                key=lambda eid: (-float(active_edges[eid].length_px), eid),
            )
            trunk_edge_ids = set(sorted_edge_ids[:2])
            for edge_id in edge_ids:
                if edge_id in trunk_edge_ids:
                    continue
                edge = active_edges.get(edge_id)
                if edge is None:
                    continue
                other_node_id = edge.v if edge.u == node_id else edge.u
                other_degree = len(incident.get(other_node_id, []))
                if other_degree != 1:
                    continue
                if float(edge.length_px * graph.resolution) <= float(max_leaf_length_m):
                    leaf_edge_ids.add(edge.edge_id)

        if not leaf_edge_ids:
            break
        for edge_id in leaf_edge_ids:
            active_edges.pop(edge_id, None)

    if not active_edges:
        return RoadGraph(nodes=[], edges=[], resolution=graph.resolution)

    final_incident = build_incident_map()
    ordered_edges = [active_edges[e.edge_id] for e in graph.edges if e.edge_id in active_edges]
    valid_nodes = {e.u for e in ordered_edges} | {e.v for e in ordered_edges}
    ordered_nodes: list[RoadNode] = []
    for node in graph.nodes:
        if node.node_id not in valid_nodes:
            continue
        degree = len(final_incident.get(node.node_id, []))
        if degree <= 0:
            continue
        ordered_nodes.append(
            RoadNode(
                node_id=node.node_id,
                kind="junction" if degree >= 3 else "endpoint",
                y=float(node.y),
                x=float(node.x),
                pixels=list(node.pixels),
                incident_edge_ids=list(final_incident[node.node_id]),
            )
        )
    return RoadGraph(nodes=ordered_nodes, edges=ordered_edges, resolution=graph.resolution)


def merge_close_junctions(graph: RoadGraph, merge_distance_m: float) -> tuple[RoadGraph, int]:
    if merge_distance_m <= 0 or not graph.nodes or not graph.edges or graph.resolution <= 0:
        return graph, 0

    node_lookup = {n.node_id: n for n in graph.nodes}
    junction_ids = [n.node_id for n in graph.nodes if n.kind == "junction"]
    if len(junction_ids) < 2:
        return graph, 0

    groups: list[list[str]] = []
    visited: set[str] = set()
    threshold_px = float(merge_distance_m / graph.resolution)
    for nid in junction_ids:
        if nid in visited:
            continue
        seed = [nid]
        group: list[str] = []
        while seed:
            cur = seed.pop()
            if cur in visited:
                continue
            visited.add(cur)
            group.append(cur)
            cx, cy = node_lookup[cur].x, node_lookup[cur].y
            for other in junction_ids:
                if other in visited:
                    continue
                ox, oy = node_lookup[other].x, node_lookup[other].y
                if math.hypot(cx - ox, cy - oy) <= threshold_px:
                    seed.append(other)
        groups.append(group)

    merge_to_keep: dict[str, str] = {}
    merged_nodes: dict[str, RoadNode] = {}
    for cluster_node_ids in groups:
        if len(cluster_node_ids) == 1:
            nid = cluster_node_ids[0]
            n = node_lookup[nid]
            merged_nodes[nid] = RoadNode(node_id=n.node_id, kind=n.kind, y=float(n.y), x=float(n.x), pixels=list(n.pixels))
            continue

        keep_id = sorted(cluster_node_ids, key=lambda nid: (-len(node_lookup[nid].incident_edge_ids), nid))[0]
        keep_node = node_lookup[keep_id]
        merged_pixels: list[Pixel] = []
        xs: list[float] = []
        ys: list[float] = []
        for nid in cluster_node_ids:
            n = node_lookup[nid]
            merged_pixels.extend(n.pixels)
            xs.append(float(n.x))
            ys.append(float(n.y))
            if nid != keep_id:
                merge_to_keep[nid] = keep_id
        merged_nodes[keep_id] = RoadNode(
            node_id=keep_node.node_id,
            kind="junction",
            y=float(np.mean(np.asarray(ys, dtype=np.float32))),
            x=float(np.mean(np.asarray(xs, dtype=np.float32))),
            pixels=merged_pixels,
        )

    for n in graph.nodes:
        if n.kind == "junction":
            continue
        merged_nodes[n.node_id] = RoadNode(node_id=n.node_id, kind=n.kind, y=float(n.y), x=float(n.x), pixels=list(n.pixels))

    rewired_edges: list[RoadEdge] = []
    for e in graph.edges:
        new_u = merge_to_keep.get(e.u, e.u)
        new_v = merge_to_keep.get(e.v, e.v)
        if new_u == new_v:
            continue
        rewired_edges.append(RoadEdge(edge_id=e.edge_id, u=new_u, v=new_v, pixels=list(e.pixels), length_px=float(e.length_px)))

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
    return RoadGraph(nodes=final_nodes, edges=rewired_edges, resolution=graph.resolution), int(len(merge_to_keep))


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
    *,
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

    for node in graph.nodes:
        if node.kind != "junction":
            continue
        branch_values: list[float] = []
        for edge_id in node.incident_edge_ids:
            edge = edge_map[edge_id]
            ordered_pixels = edge.pixels if edge.u == node.node_id else list(reversed(edge.pixels))
            branch_values.extend(
                _sample_distance_values_on_pixels(
                    ordered_pixels,
                    distance_map_m,
                    graph.resolution,
                    start_m,
                    end_m,
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
        radii[node.node_id] = float(np.clip(radius, radius_min_m, radius_max_m))
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


def _sample_distance_values_on_line(line: LineString, distance_map_m: np.ndarray) -> list[float]:
    length = float(line.length)
    if length <= 0:
        return []
    h, w = distance_map_m.shape[:2]
    n_samples = max(8, int(np.ceil(length)))
    dists = np.linspace(0.0, length, num=n_samples)
    values: list[float] = []
    for d in dists:
        p = line.interpolate(float(d))
        x = int(np.clip(round(float(p.x)), 0, w - 1))
        y = int(np.clip(round(float(p.y)), 0, h - 1))
        v = float(distance_map_m[y, x])
        if v > 0:
            values.append(v)
    return values


def _global_direction_from_all_points(coords: list[tuple[float, float]]) -> tuple[float, float] | None:
    if len(coords) < 2:
        return None
    arr = np.asarray(coords, dtype=np.float64)
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


def _endpoint_extension_direction(coords: list[tuple[float, float]], *, at_start: bool) -> tuple[float, float] | None:
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
        d = (-global_dir[0], -global_dir[1]) if global_dir is not None else _endpoint_extension_direction(coords, at_start=True)
        if d is not None:
            sx, sy = coords[0]
            out_coords = [(sx + d[0] * start_ext, sy + d[1] * start_ext)] + out_coords

    if end_ext > 0:
        d = global_dir if global_dir is not None else _endpoint_extension_direction(coords, at_start=False)
        if d is not None:
            ex, ey = coords[-1]
            out_coords = out_coords + [(ex + d[0] * end_ext, ey + d[1] * end_ext)]

    ext_line = LineString(out_coords)
    return ext_line if ext_line.length > 0 else line


def _iter_polygons(geometry: Polygon | MultiPolygon):
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        for g in geometry.geoms:
            if isinstance(g, Polygon):
                yield g


def _geometry_to_open_ring_pixels(geometry: Polygon | MultiPolygon) -> list[list[list[float]]]:
    rings: list[list[list[float]]] = []
    for poly in _iter_polygons(geometry):
        coords = [[float(x), float(y)] for x, y in list(poly.exterior.coords)[:-1]]
        if len(coords) >= 3:
            rings.append(coords)
    return rings


def _load_geo_meta(geo_meta_path: Path | str) -> dict[str, Any]:
    path = Path(geo_meta_path).expanduser()
    with path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    required = {"min_xy", "max_xy", "meters_per_pixel", "width", "height"}
    missing = required - set(meta)
    if missing:
        raise KeyError(f"geo_meta missing keys: {sorted(missing)}")
    return meta


def _pixel_ring_to_world_xy(ring: list[list[float]], geo_meta: dict[str, Any]) -> list[list[float]]:
    min_x = float(geo_meta["min_xy"][0])
    min_y = float(geo_meta["min_xy"][1])
    mpp = float(geo_meta["meters_per_pixel"])
    height = int(geo_meta["height"])
    out: list[list[float]] = []
    for x, y in ring:
        world_x = min_x + float(x) * mpp
        world_y = min_y + float(height - 1 - y) * mpp
        out.append([world_x, world_y])
    return out


def _rings_to_geometry_wkt_xy(rings: list[list[list[float]]]) -> str:
    polygons: list[Polygon] = []
    for ring in rings:
        if len(ring) < 3:
            continue
        closed = ring if ring[0] == ring[-1] else [*ring, ring[0]]
        poly = Polygon(closed)
        if not poly.is_empty:
            polygons.append(poly)
    if not polygons:
        return Polygon().wkt
    if len(polygons) == 1:
        return polygons[0].wkt
    return MultiPolygon(polygons).wkt


def _part_records_from_graph(
    graph: RoadGraph,
    distance_map_m: np.ndarray,
    *,
    resolution: float,
    scale_x_to_source: float,
    scale_y_to_source: float,
    source_geo_meta: dict[str, Any] | None,
    junction_radii_m: dict[str, float],
    segment_max_length_m: float,
    segment_overlap_m: float,
    segment_width_margin_m: float,
    min_segment_half_width_m: float,
    endpoint_extension_m: float,
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    next_part_id = 1

    for node in graph.nodes:
        if node.kind != "junction":
            continue
        radius_m = float(junction_radii_m.get(node.node_id, max(1.0, min_segment_half_width_m * 1.5)))
        radius_px = _meters_to_pixels(radius_m, resolution)
        if radius_px <= 0:
            continue
        geometry = box(
            float(node.x) - float(radius_px),
            float(node.y) - float(radius_px),
            float(node.x) + float(radius_px),
            float(node.y) + float(radius_px),
        )
        geometry_px = scale_geometry(
            geometry,
            xfact=scale_x_to_source,
            yfact=scale_y_to_source,
            origin=(0.0, 0.0),
        )
        geometry_m = scale_geometry(geometry, xfact=resolution, yfact=resolution, origin=(0.0, 0.0))
        polygon_pixels = _geometry_to_open_ring_pixels(geometry_px)
        if source_geo_meta is not None:
            polygon_xy = [_pixel_ring_to_world_xy(ring, source_geo_meta) for ring in polygon_pixels]
            geometry_wkt_xy = _rings_to_geometry_wkt_xy(polygon_xy)
        else:
            polygon_xy = _geometry_to_open_ring_pixels(geometry_m)
            geometry_wkt_xy = geometry_m.wkt
        parts.append(
            {
                "part_id": next_part_id,
                "kind": "junction",
                "source_id": node.node_id,
                "segment_index": None,
                "area_px2": float(geometry_px.area),
                "area_m2": float(geometry_m.area),
                "polygon_pixels": polygon_pixels,
                "polygon_xy": polygon_xy,
                "geometry_wkt_pixels": geometry_px.wkt,
                "geometry_wkt_xy": geometry_wkt_xy,
            }
        )
        next_part_id += 1

    max_len_px = _meters_to_pixels(segment_max_length_m, resolution)
    overlap_px = _meters_to_pixels(segment_overlap_m, resolution)
    endpoint_extension_px = max(0.0, _meters_to_pixels(endpoint_extension_m, resolution))
    node_kind = {n.node_id: n.kind for n in graph.nodes}

    for edge in graph.edges:
        coords_xy = [(float(x), float(y)) for y, x in edge.pixels]
        if len(coords_xy) < 2:
            continue
        line = LineString(coords_xy)
        if line.length <= 0:
            continue

        pieces = split_line_with_overlap(line, max_length=max_len_px, overlap=overlap_px)
        for idx, piece in enumerate(pieces):
            extend_start_px = endpoint_extension_px if idx == 0 and node_kind.get(edge.u, "endpoint") == "endpoint" else 0.0
            extend_end_px = endpoint_extension_px if idx == (len(pieces) - 1) and node_kind.get(edge.v, "endpoint") == "endpoint" else 0.0
            piece = extend_linestring_at_ends(piece, extend_start_px=extend_start_px, extend_end_px=extend_end_px)
            if piece.length <= 0:
                continue

            local_half_width_values = _sample_distance_values_on_line(piece, distance_map_m)
            if local_half_width_values:
                half_width_m = float(np.median(np.asarray(local_half_width_values, dtype=np.float32)))
            else:
                half_width_m = float(min_segment_half_width_m)
            half_width_m = max(min_segment_half_width_m, half_width_m + segment_width_margin_m)
            half_width_px = _meters_to_pixels(half_width_m, resolution)
            geometry = piece.buffer(half_width_px, cap_style=3, join_style=2)
            if geometry.is_empty:
                continue
            geometry_px = scale_geometry(
                geometry,
                xfact=scale_x_to_source,
                yfact=scale_y_to_source,
                origin=(0.0, 0.0),
            )
            geometry_m = scale_geometry(geometry, xfact=resolution, yfact=resolution, origin=(0.0, 0.0))
            polygon_pixels = _geometry_to_open_ring_pixels(geometry_px)
            if source_geo_meta is not None:
                polygon_xy = [_pixel_ring_to_world_xy(ring, source_geo_meta) for ring in polygon_pixels]
                geometry_wkt_xy = _rings_to_geometry_wkt_xy(polygon_xy)
            else:
                polygon_xy = _geometry_to_open_ring_pixels(geometry_m)
                geometry_wkt_xy = geometry_m.wkt
            parts.append(
                {
                    "part_id": next_part_id,
                    "kind": "road_segment",
                    "source_id": edge.edge_id,
                    "segment_index": idx,
                    "length_m": float(piece.length * resolution),
                    "half_width_m": float(half_width_m),
                    "area_px2": float(geometry_px.area),
                    "area_m2": float(geometry_m.area),
                    "polygon_pixels": polygon_pixels,
                    "polygon_xy": polygon_xy,
                    "geometry_wkt_pixels": geometry_px.wkt,
                    "geometry_wkt_xy": geometry_wkt_xy,
                }
            )
            next_part_id += 1
    return parts


def _draw_parts_preview(mask: np.ndarray, parts: list[dict[str, Any]], output_path: Path) -> Path:
    mask_u8 = (np.asarray(mask, dtype=np.uint8) * 255).astype(np.uint8)
    canvas = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)

    for part in parts:
        color = (0, 165, 255) if part["kind"] == "junction" else (0, 200, 80)
        for ring in part.get("polygon_pixels", []):
            pts = np.asarray([[round(p[0]), round(p[1])] for p in ring], dtype=np.int32).reshape(-1, 1, 2)
            if pts.shape[0] < 3:
                continue
            cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)
            m = pts.reshape(-1, 2).mean(axis=0)
            cv2.putText(
                canvas,
                str(part["part_id"]),
                (int(m[0]), int(m[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError(f"Failed to encode preview image: {output_path}")
    encoded.tofile(str(output_path))
    return output_path


def _graph_to_source_backbone_mask(
    graph: RoadGraph,
    *,
    topology_shape: tuple[int, int],
    source_shape: tuple[int, int],
) -> np.ndarray:
    topo_canvas = np.zeros(topology_shape, dtype=np.uint8)
    for edge in graph.edges:
        for y, x in edge.pixels:
            iy = int(round(y))
            ix = int(round(x))
            if 0 <= iy < topology_shape[0] and 0 <= ix < topology_shape[1]:
                topo_canvas[iy, ix] = 255
    for node in graph.nodes:
        for y, x in node.pixels:
            iy = int(round(y))
            ix = int(round(x))
            if 0 <= iy < topology_shape[0] and 0 <= ix < topology_shape[1]:
                topo_canvas[iy, ix] = 255

    source_h, source_w = source_shape
    source_canvas = cv2.resize(topo_canvas, (source_w, source_h), interpolation=cv2.INTER_NEAREST)
    if np.any(source_canvas):
        source_canvas = cv2.dilate(source_canvas, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return source_canvas > 0


def _draw_backbone_preview(mask: np.ndarray, backbone_mask: np.ndarray, output_path: Path) -> Path:
    mask_u8 = (np.asarray(mask, dtype=np.uint8) * 255).astype(np.uint8)
    canvas = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
    canvas[np.asarray(backbone_mask, dtype=bool)] = (0, 0, 255)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError(f"Failed to encode backbone preview image: {output_path}")
    encoded.tofile(str(output_path))
    return output_path


def _build_scene_parts_artifacts(
    mask_path: Path | str,
    *,
    geo_meta_path: Path | str | None = None,
    mpp: float = DEFAULT_MPP,
    topology_mpp: float = 0.2,
    closing_radius_m: float = 5.0,
    min_component_area_m2: float = 30.0,
    max_hole_area_m2: float = 200.0,
    spur_prune_length_m: float = 30.0,
    junction_cluster_eps_m: float = 3.0,
    min_edge_length_m: float = 0.0,
    junction_min_valid_branches: int = 3,
    junction_min_branch_length_m: float = 5.0,
    junction_merge_distance_m: float = 12.0,
    branch_sample_start_m: float = 2.0,
    branch_sample_end_m: float = 6.0,
    junction_radius_scale: float = 1.5,
    junction_radius_margin_m: float = 0.8,
    junction_radius_min_m: float = 3.0,
    junction_radius_max_m: float = 20.0,
    segment_max_length_m: float = 100.0,
    segment_overlap_m: float = 10.0,
    segment_width_margin_m: float = 10.0,
    min_segment_half_width_m: float = 1.2,
    endpoint_extension_m: float = 20.0,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if mpp <= 0:
        raise ValueError("mpp must be > 0")

    mask_path = Path(mask_path).expanduser()
    orig_mask = load_binary_mask(mask_path)
    source_geo_meta = _load_geo_meta(geo_meta_path) if geo_meta_path is not None else None
    if source_geo_meta is not None:
        if int(source_geo_meta["width"]) != int(orig_mask.shape[1]) or int(source_geo_meta["height"]) != int(orig_mask.shape[0]):
            raise ValueError(
                "geo_meta width/height do not match mask shape: "
                f"meta=({source_geo_meta['width']}, {source_geo_meta['height']}), "
                f"mask=({orig_mask.shape[1]}, {orig_mask.shape[0]})"
            )
        meta_mpp = float(source_geo_meta["meters_per_pixel"])
        if not np.isclose(meta_mpp, float(mpp)):
            raise ValueError(f"geo_meta meters_per_pixel={meta_mpp} does not match mpp={mpp}")
    topo_mask, topo_resolution, scale_x_to_source, scale_y_to_source = downsample_mask_for_topology(
        orig_mask,
        source_resolution=mpp,
        topology_resolution=topology_mpp,
    )

    cleaned = clean_road_mask(
        topo_mask,
        closing_radius_px=int(round(_meters_to_pixels(closing_radius_m, topo_resolution))),
        min_component_area_px=_area_m2_to_px(min_component_area_m2, topo_resolution),
        max_hole_area_px=_area_m2_to_px(max_hole_area_m2, topo_resolution),
    )
    skeleton = skeletonize_and_prune(cleaned, prune_length_px=_meters_to_pixels(spur_prune_length_m, topo_resolution))
    graph = build_road_graph(
        skeleton,
        junction_cluster_eps_px=_meters_to_pixels(junction_cluster_eps_m, topo_resolution),
        resolution=topo_resolution,
        min_edge_length_m=min_edge_length_m,
    )
    graph = prune_graph_short_leaf_edges(graph, max_leaf_length_m=spur_prune_length_m)
    graph, merged_junction_nodes = merge_close_junctions(graph, merge_distance_m=junction_merge_distance_m)
    graph = normalize_graph_junctions_by_branch_lengths(
        graph,
        min_edge_length_m=min_edge_length_m,
        junction_min_valid_branches=junction_min_valid_branches,
        junction_min_branch_length_m=junction_min_branch_length_m,
    )
    graph = prune_graph_short_leaf_edges(graph, max_leaf_length_m=spur_prune_length_m)

    distance_map_m = compute_distance_map_m(cleaned, resolution=topo_resolution)
    junction_radii_m = estimate_junction_radii_m(
        graph,
        distance_map_m,
        sample_start_m=branch_sample_start_m,
        sample_end_m=branch_sample_end_m,
        radius_scale=junction_radius_scale,
        radius_margin_m=junction_radius_margin_m,
        radius_min_m=junction_radius_min_m,
        radius_max_m=junction_radius_max_m,
    )
    parts = _part_records_from_graph(
        graph,
        distance_map_m,
        resolution=topo_resolution,
        scale_x_to_source=scale_x_to_source,
        scale_y_to_source=scale_y_to_source,
        source_geo_meta=source_geo_meta,
        junction_radii_m=junction_radii_m,
        segment_max_length_m=segment_max_length_m,
        segment_overlap_m=segment_overlap_m,
        segment_width_margin_m=segment_width_margin_m,
        min_segment_half_width_m=min_segment_half_width_m,
        endpoint_extension_m=endpoint_extension_m,
    )
    backbone_mask = _graph_to_source_backbone_mask(
        graph,
        topology_shape=topo_mask.shape,
        source_shape=orig_mask.shape,
    )

    payload = {
        "source_mask": str(mask_path),
        "meters_per_pixel": float(mpp),
        "mask_height": int(orig_mask.shape[0]),
        "mask_width": int(orig_mask.shape[1]),
        "geo_meta_path": str(Path(geo_meta_path).expanduser()) if geo_meta_path is not None else None,
        "topology_meters_per_pixel": float(topo_resolution),
        "topology_height": int(topo_mask.shape[0]),
        "topology_width": int(topo_mask.shape[1]),
        "topology_to_source_scale_x": float(scale_x_to_source),
        "topology_to_source_scale_y": float(scale_y_to_source),
        "num_nodes": len(graph.nodes),
        "num_edges": len(graph.edges),
        "merged_junction_nodes": int(merged_junction_nodes),
        "num_parts": len(parts),
        "parts": parts,
    }
    return payload, orig_mask, backbone_mask


def build_scene_parts_payload(
    mask_path: Path | str,
    *,
    geo_meta_path: Path | str | None = None,
    mpp: float = DEFAULT_MPP,
    topology_mpp: float = 0.2,
    closing_radius_m: float = 5.0,
    min_component_area_m2: float = 30.0,
    max_hole_area_m2: float = 200.0,
    spur_prune_length_m: float = 30.0,
    junction_cluster_eps_m: float = 3.0,
    min_edge_length_m: float = 0.0,
    junction_min_valid_branches: int = 3,
    junction_min_branch_length_m: float = 5.0,
    junction_merge_distance_m: float = 12.0,
    branch_sample_start_m: float = 2.0,
    branch_sample_end_m: float = 6.0,
    junction_radius_scale: float = 1.5,
    junction_radius_margin_m: float = 0.8,
    junction_radius_min_m: float = 3.0,
    junction_radius_max_m: float = 20.0,
    segment_max_length_m: float = 100.0,
    segment_overlap_m: float = 10.0,
    segment_width_margin_m: float = 10.0,
    min_segment_half_width_m: float = 1.2,
    endpoint_extension_m: float = 20.0,
) -> tuple[dict[str, Any], np.ndarray]:
    payload, orig_mask, _ = _build_scene_parts_artifacts(
        mask_path,
        geo_meta_path=geo_meta_path,
        mpp=mpp,
        topology_mpp=topology_mpp,
        closing_radius_m=closing_radius_m,
        min_component_area_m2=min_component_area_m2,
        max_hole_area_m2=max_hole_area_m2,
        spur_prune_length_m=spur_prune_length_m,
        junction_cluster_eps_m=junction_cluster_eps_m,
        min_edge_length_m=min_edge_length_m,
        junction_min_valid_branches=junction_min_valid_branches,
        junction_min_branch_length_m=junction_min_branch_length_m,
        junction_merge_distance_m=junction_merge_distance_m,
        branch_sample_start_m=branch_sample_start_m,
        branch_sample_end_m=branch_sample_end_m,
        junction_radius_scale=junction_radius_scale,
        junction_radius_margin_m=junction_radius_margin_m,
        junction_radius_min_m=junction_radius_min_m,
        junction_radius_max_m=junction_radius_max_m,
        segment_max_length_m=segment_max_length_m,
        segment_overlap_m=segment_overlap_m,
        segment_width_margin_m=segment_width_margin_m,
        min_segment_half_width_m=min_segment_half_width_m,
        endpoint_extension_m=endpoint_extension_m,
    )
    return payload, orig_mask


def write_scene_parts_json(
    mask_path: Path | str,
    output_json_path: Path | str,
    *,
    geo_meta_path: Path | str | None = None,
    preview_path: Path | str | None = None,
    backbone_preview_path: Path | str | None = None,
    mpp: float = DEFAULT_MPP,
    **kwargs: Any,
) -> dict[str, Any]:
    payload, mask, backbone_mask = _build_scene_parts_artifacts(
        mask_path,
        geo_meta_path=geo_meta_path,
        mpp=mpp,
        **kwargs,
    )
    output_json_path = Path(output_json_path).expanduser()
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if preview_path is None:
        preview_path = output_json_path.with_name(f"{output_json_path.stem}_preview.png")
    _draw_parts_preview(mask, payload["parts"], Path(preview_path).expanduser())
    if backbone_preview_path is None:
        backbone_preview_path = output_json_path.with_name(f"{output_json_path.stem}_backbone.png")
    _draw_backbone_preview(mask, backbone_mask, Path(backbone_preview_path).expanduser())
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build road-scene parts.json from an intensity BEV mask image.",
    )
    parser.add_argument("mask_path", help="Input BEV mask image or npz.")
    parser.add_argument("geo_meta_path", help="Input geo_meta.json path.")
    parser.add_argument("output_json_path", help="Output parts.json path.")
    parser.add_argument("--preview", default=None, help="Optional preview PNG path.")
    parser.add_argument("--backbone-preview", default=None, help="Optional backbone preview PNG path.")
    args = parser.parse_args()

    payload = write_scene_parts_json(
        args.mask_path,
        args.output_json_path,
        geo_meta_path=args.geo_meta_path,
        preview_path=args.preview,
        backbone_preview_path=args.backbone_preview,
        mpp=DEFAULT_MPP,
    )
    print(f"Wrote {payload['num_parts']} scene parts to {Path(args.output_json_path).expanduser()}")


if __name__ == "__main__":
    main()
