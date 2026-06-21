"""Convert polygon SHP features to simple centerline polyline SHP features."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import shapefile
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Point, Polygon


def _iter_polygon_parts(geom) -> list[Polygon]:
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom] if float(geom.area) > 0 else []
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        polygons: list[Polygon] = []
        for part in geom.geoms:
            polygons.extend(_iter_polygon_parts(part))
        return polygons
    return []


def _shape_to_polygons(shape) -> list[Polygon]:
    points = shape.points
    if len(points) < 4:
        return []
    parts = list(shape.parts) + [len(points)]
    polygons: list[Polygon] = []
    for idx in range(len(parts) - 1):
        ring = points[int(parts[idx]) : int(parts[idx + 1])]
        if len(ring) < 4:
            continue
        try:
            poly = Polygon(ring)
        except Exception:
            continue
        if poly.is_empty:
            continue
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or float(poly.area) <= 0:
            continue
        polygons.extend(_iter_polygon_parts(poly))
    return polygons


def _long_axis_from_rect(poly: Polygon) -> tuple[tuple[float, float], tuple[float, float], float, float]:
    rect = poly.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)[:4]
    if len(coords) < 4:
        raise ValueError("minimum rotated rectangle has fewer than four points")
    edges = []
    for idx in range(4):
        p0 = coords[idx]
        p1 = coords[(idx + 1) % 4]
        dx = float(p1[0] - p0[0])
        dy = float(p1[1] - p0[1])
        edges.append((math.hypot(dx, dy), dx, dy))
    length, dx, dy = max(edges, key=lambda item: item[0])
    width = min(edge[0] for edge in edges)
    norm = math.hypot(dx, dy)
    if norm <= 1e-9:
        return (1.0, 0.0), (0.0, 1.0), float(length), float(width)
    ux, uy = dx / norm, dy / norm
    return (ux, uy), (-uy, ux), float(length), float(width)


def _rect_short_edge_midpoint_centerline(poly: Polygon) -> tuple[list[list[float]] | None, dict[str, Any]]:
    if poly.is_empty:
        return None, {"method": "empty"}
    if not poly.is_valid:
        poly = poly.buffer(0)
    parts = _iter_polygon_parts(poly)
    if not parts:
        return None, {"method": "invalid"}
    poly = max(parts, key=lambda item: float(item.area))

    rect = poly.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)[:4]
    if len(coords) < 4:
        return None, {"method": "invalid_rect"}

    edges = []
    for idx in range(4):
        p0 = coords[idx]
        p1 = coords[(idx + 1) % 4]
        edges.append((math.hypot(float(p1[0] - p0[0]), float(p1[1] - p0[1])), idx, p0, p1))
    short_edges = sorted(edges, key=lambda item: item[0])[:2]
    if len(short_edges) < 2:
        return None, {"method": "invalid_rect"}
    midpoints = []
    for _length, _idx, p0, p1 in short_edges:
        midpoints.append(Point((float(p0[0]) + float(p1[0])) / 2.0, (float(p0[1]) + float(p1[1])) / 2.0))
    if midpoints[0].distance(midpoints[1]) <= 1e-9:
        return None, {"method": "invalid_rect"}

    line = [[float(midpoints[0].x), float(midpoints[0].y)], [float(midpoints[1].x), float(midpoints[1].y)]]
    length = max(edge[0] for edge in edges)
    width = min(edge[0] for edge in edges)
    return line, {
        "method": "short_edge_midpoints",
        "line_len_m": float(midpoints[0].distance(midpoints[1])),
        "rect_length_m": float(length),
        "rect_width_m": float(width),
    }


def _projection_bounds(poly: Polygon, axis: tuple[float, float]) -> tuple[float, float]:
    ux, uy = axis
    vals = [float(x) * ux + float(y) * uy for x, y in poly.exterior.coords]
    return min(vals), max(vals)


def _middle_point_of_intersection(geom, fallback: Point) -> Point:
    if geom.is_empty:
        return fallback
    if isinstance(geom, Point):
        return geom
    if isinstance(geom, LineString):
        return geom.interpolate(0.5, normalized=True)
    if isinstance(geom, MultiLineString):
        lines = [part for part in geom.geoms if not part.is_empty and float(part.length) > 0]
        if not lines:
            return fallback
        return max(lines, key=lambda line: float(line.length)).interpolate(0.5, normalized=True)
    if isinstance(geom, GeometryCollection):
        lines = [part for part in geom.geoms if isinstance(part, LineString) and not part.is_empty and float(part.length) > 0]
        if lines:
            return max(lines, key=lambda line: float(line.length)).interpolate(0.5, normalized=True)
        points = [part for part in geom.geoms if isinstance(part, Point) and not part.is_empty]
        if points:
            return points[0]
    return fallback


def polygon_to_centerline(poly: Polygon, *, cross_scale: float = 3.0) -> tuple[list[list[float]] | None, dict[str, Any]]:
    if poly.is_empty:
        return None, {"method": "empty"}
    if not poly.is_valid:
        poly = poly.buffer(0)
    parts = _iter_polygon_parts(poly)
    if not parts:
        return None, {"method": "invalid"}
    poly = max(parts, key=lambda item: float(item.area))

    axis, normal, rect_length, rect_width = _long_axis_from_rect(poly)
    center = poly.centroid
    center_u = float(center.x) * axis[0] + float(center.y) * axis[1]
    min_u, max_u = _projection_bounds(poly, axis)
    ux, uy = axis
    nx, ny = normal
    half_cross = max(rect_width * float(cross_scale), rect_width + 1.0, 1.0)
    points: list[Point] = []
    for u in (min_u, max_u):
        offset_u = float(u) - center_u
        base_x = float(center.x) + ux * offset_u
        base_y = float(center.y) + uy * offset_u
        line = LineString(
            [
                (base_x - nx * half_cross, base_y - ny * half_cross),
                (base_x + nx * half_cross, base_y + ny * half_cross),
            ]
        )
        fallback = Point(base_x, base_y)
        points.append(_middle_point_of_intersection(poly.intersection(line), fallback))

    if len(points) != 2 or points[0].distance(points[1]) <= 1e-9:
        cx, cy = center.x, center.y
        half = rect_length / 2.0
        points = [Point(cx - ux * half, cy - uy * half), Point(cx + ux * half, cy + uy * half)]
        method = "rect_axis"
    else:
        method = "normal_midpoints"

    line = [[float(points[0].x), float(points[0].y)], [float(points[1].x), float(points[1].y)]]
    line_len = float(points[0].distance(points[1]))
    return line, {"method": method, "line_len_m": line_len, "rect_length_m": rect_length, "rect_width_m": rect_width}


def polygon_to_laneline_centerline(
    poly: Polygon,
    *,
    source: str = "",
    cross_scale: float = 3.0,
) -> tuple[list[list[float]] | None, dict[str, Any]]:
    if source == "box":
        return _rect_short_edge_midpoint_centerline(poly)
    return polygon_to_centerline(poly, cross_scale=cross_scale)


def polygon_shp_to_centerline_shp(
    polygon_shp_path: Path | str,
    output_path: Path | str,
    *,
    cross_scale: float = 3.0,
) -> Path:
    polygon_shp_path = Path(polygon_shp_path).expanduser()
    output_base = Path(output_path).expanduser().with_suffix("")
    output_base.parent.mkdir(parents=True, exist_ok=True)

    reader = shapefile.Reader(str(polygon_shp_path))
    writer = shapefile.Writer(str(output_base))
    writer.shapeType = shapefile.POLYLINE
    writer.field("id", "N", decimal=0)
    writer.field("source", "C", size=12)
    writer.field("length_m", "F", decimal=3)
    writer.field("width_m", "F", decimal=3)
    writer.field("yaw", "F", decimal=6)
    writer.field("cx", "F", decimal=3)
    writer.field("cy", "F", decimal=3)
    writer.field("line_len_m", "F", decimal=3)
    writer.field("method", "C", size=32)

    out_id = 0
    for shape_record in reader.iterShapeRecords():
        rec = shape_record.record.as_dict()
        for poly in _shape_to_polygons(shape_record.shape):
            line, info = polygon_to_laneline_centerline(poly, source=str(rec.get("source", "")), cross_scale=cross_scale)
            if line is None:
                continue
            writer.line([line])
            writer.record(
                id=out_id,
                source=str(rec.get("source", "")),
                length_m=float(rec.get("length_m", rec.get("length", 0.0))),
                width_m=float(rec.get("width_m", rec.get("width", 0.0))),
                yaw=float(rec.get("yaw", 0.0)),
                cx=float(rec.get("cx", poly.centroid.x)),
                cy=float(rec.get("cy", poly.centroid.y)),
                line_len_m=float(info["line_len_m"]),
                method=str(info["method"]),
            )
            out_id += 1

    writer.close()
    return Path(f"{output_base}.shp")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert polygon SHP features to centerline polyline SHP features.")
    parser.add_argument("polygon_shp", help="Input polygon SHP path.")
    parser.add_argument("-o", "--output", required=True, help="Output polyline SHP path.")
    parser.add_argument("--cross-scale", type=float, default=3.0)
    args = parser.parse_args()
    polygon_shp_to_centerline_shp(args.polygon_shp, args.output, cross_scale=args.cross_scale)


if __name__ == "__main__":
    main()
