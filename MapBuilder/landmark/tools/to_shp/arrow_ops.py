"""Arrow template matching via Chamfer distance."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Arrow outline vertices (analytical, from parametric geometry)
# ---------------------------------------------------------------------------

# Outline vertices in cm for each arrow type (union of filled polygons).
# Coordinate system: x right, y up; origin at arrow anchor point.
_ARROW_OUTLINES_CM: dict[str, list[list[float]]] = {
    "straight": [
        [-15, -360], [15, -360], [15, 0], [45, 0],
        [0, 240], [-45, 0], [-15, 0],
    ],
    "straight_left": [
        [-15, -360], [15, -360], [15, 0], [45, 0], [0, 240], [-45, 0],
        [-15, 0], [-15, -200], [-95, -120], [-95, -20], [-135, -180],
        [-95, -330], [-95, -240], [-15, -320],
    ],
    "straight_right": [
        [-15, -360], [15, -360], [15, -320], [95, -240], [95, -330],
        [135, -180], [95, -20], [95, -120], [15, -200], [15, 0],
        [45, 0], [0, 240], [-45, 0], [-15, 0],
    ],
    "left": [
        [-15, -300], [15, -300], [15, 90], [-95, 200], [-95, 300],
        [-135, 140], [-95, -10], [-95, 80], [-15, 0],
    ],
    "right": [
        [-15, -300], [15, -300], [15, 0], [95, 80], [95, -10],
        [135, 140], [95, 300], [95, 200], [-15, 90],
    ],
}

# Bounding box params for cm→pixel conversion (padding_cm=5, res=2.0 cm/px).
_ARROW_BBOX: dict[str, dict] = {
    "straight":       {"min_x": -50.0, "max_y": 245.0, "size": [305, 50]},
    "straight_left":  {"min_x": -140.0, "max_y": 245.0, "size": [305, 95]},
    "straight_right": {"min_x": -50.0, "max_y": 245.0, "size": [305, 95]},
    "left":           {"min_x": -140.0, "max_y": 305.0, "size": [305, 80]},
    "right":          {"min_x": -20.0, "max_y": 305.0, "size": [305, 80]},
}

_RES_CM_PER_PX = 2.0


def _cm_to_px(
    vertices_cm: list[list[float]], min_x: float, max_y: float,
) -> np.ndarray:
    """Convert cm vertices to template pixel coordinates."""
    v = np.asarray(vertices_cm, dtype=np.float64)
    px = np.empty_like(v)
    px[:, 0] = (v[:, 0] - min_x) / _RES_CM_PER_PX
    px[:, 1] = (max_y - v[:, 1]) / _RES_CM_PER_PX
    return px


def _clip_polygon_top(vertices: np.ndarray, max_y: float) -> np.ndarray:
    """Clip polygon, keeping vertices where pixel-y < *max_y* (tail crop)."""
    n = len(vertices)
    if n == 0:
        return vertices
    clipped: list[list[float]] = []
    for i in range(n):
        curr = vertices[i]
        nxt = vertices[(i + 1) % n]
        c_in = curr[1] < max_y
        n_in = nxt[1] < max_y
        if c_in:
            clipped.append(curr.tolist())
            if not n_in:
                dy = nxt[1] - curr[1]
                if abs(dy) > 1e-9:
                    t = (max_y - curr[1]) / dy
                    clipped.append([curr[0] + t * (nxt[0] - curr[0]), float(max_y)])
        elif n_in:
            dy = nxt[1] - curr[1]
            if abs(dy) > 1e-9:
                t = (max_y - curr[1]) / dy
                clipped.append([curr[0] + t * (nxt[0] - curr[0]), float(max_y)])
    return np.asarray(clipped, dtype=np.float64) if clipped else np.empty((0, 2))


def load_arrow_vertices(
    vertex_dir: Path | str = "asserts/arrow_line",
) -> dict[str, dict]:
    """Load arrow vertex templates from JSON."""
    p = Path(vertex_dir) / "arrow_vertices.json"
    if not p.exists():
        raise FileNotFoundError(f"Arrow vertices not found: {p}. Run generate_arrow_line_templates() first.")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def reconstruct_arrow_contour(
    arrow: dict,
    arrow_vertices: dict[str, dict],
) -> np.ndarray | None:
    """Reconstruct arrow outline in BEV pixel coords from vector vertices.

    *arrow_vertices* maps type name → ``{"size": [h, w], "vertices_px": [...]}``.
    Returns (N, 2) float64 array of transformed outline points, or *None*.
    """
    name = arrow.get("type")
    if name not in arrow_vertices:
        return None
    data = arrow_vertices[name]
    verts = np.asarray(data["vertices_px"], dtype=np.float64)
    t_h, t_w = data["size"]

    theta = float(arrow["theta"])
    tail_f = float(arrow.get("tail_frac", 1.0))
    tx = float(arrow["tx"])
    ty = float(arrow["ty"])

    crop_h = int(round(t_h * tail_f))
    if crop_h < 1:
        return None

    if tail_f < 1.0:
        verts = _clip_polygon_top(verts, float(crop_h))
        if len(verts) < 3:
            return None

    # Same affine as chamfer_match rendering
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cx, cy = t_w / 2.0, crop_h / 2.0
    mat = np.array([
        [cos_t, -sin_t, tx - cos_t * cx + sin_t * cy],
        [sin_t,  cos_t, ty - sin_t * cx - cos_t * cy],
    ], dtype=np.float64)

    ones = np.ones((len(verts), 1), dtype=np.float64)
    return (mat @ np.hstack([verts, ones]).T).T  # (N, 2)


def generate_arrow_line_templates(
    output_dir: Path | str = "./asserts/arrow_line",
) -> dict[str, dict]:
    """Generate arrow outline vector images and vertex JSON in *output_dir*."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, dict] = {}
    for name, verts_cm in _ARROW_OUTLINES_CM.items():
        bb = _ARROW_BBOX[name]
        h, w = bb["size"]
        verts_px = _cm_to_px(verts_cm, bb["min_x"], bb["max_y"])

        # Save polyline image
        img = np.zeros((h, w), dtype=np.uint8)
        pts_i = np.round(verts_px).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts_i], isClosed=True, color=255, thickness=1)
        cv2.imwrite(str(output_dir / f"{name}.png"), img)

        result[name] = {"size": [h, w], "vertices_px": verts_px.tolist()}

    with (output_dir / "arrow_vertices.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[arrow] line templates saved to {output_dir} ({len(result)} types)")
    return result


# ---------------------------------------------------------------------------
# Chamfer matching
# ---------------------------------------------------------------------------
def _extract_edge_points(binary: np.ndarray) -> np.ndarray:
    """Extract edge pixel coordinates (x, y) from a binary mask."""
    edge = cv2.Canny(binary.astype(np.uint8) * 255, 50, 150)
    ys, xs = np.nonzero(edge)
    return np.stack([xs, ys], axis=-1).astype(np.float64)


def _transform_points(
    pts: np.ndarray, *, cx: float, cy: float, theta: float, scale: float,
    tx: float, ty: float,
) -> np.ndarray:
    """Rotate *pts* around (cx, cy) by *theta*, scale, then translate by (tx, ty)."""
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    centered = pts - np.array([cx, cy])
    rot = centered @ np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    return rot * scale + np.array([tx, ty])


def generate_edge_templates(
    template_dir: Path | str = "asserts/arrow_templates",
    edge_dir: Path | str = "asserts/arrow_edges",
) -> None:
    """Pre-compute Canny edges for each arrow template and save as PNGs."""
    template_dir = Path(template_dir)
    edge_dir = Path(edge_dir)
    edge_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(template_dir.glob("*.png")):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        binary = (img > 0).astype(np.uint8) * 255
        edge = cv2.Canny(binary, 50, 150)
        cv2.imwrite(str(edge_dir / p.name), edge)
    print(f"[arrow] edge templates saved to {edge_dir}")


def chamfer_match(
    mask: np.ndarray,
    templates: dict[str, np.ndarray],
    *,
    edge_dir: Path | str | None = None,
    angle_steps: int = 72,
    tail_steps: int = 6,
    fill_threshold: float = 0.5,
    overflow_threshold: float = 0.5,
    refine_iters: int = 60,
) -> dict[str, object]:
    """Match the best arrow template to *mask* using Chamfer distance.

    If *edge_dir* is provided, pre-computed edge PNGs are loaded from that
    directory instead of running Canny on each template at runtime.

    Scale is fixed at 1.0 (template and BEV share the same mpp).
    After Phase-1 full-template matching, two metrics decide whether to
    try tail-cropping (Phase 2):

    * **fill ratio** – mask area / mask-bbox area.  Low values indicate the
      SAM3 mask may be incomplete.
    * **overflow ratio** – fraction of the fitted template edge points that
      land *outside* the mask.  High values mean the template is longer than
      the mask.

    Tail search is triggered only when *both* fill_ratio < *fill_threshold*
    **and** overflow > *overflow_threshold*.

    Returns dict with keys:
        type, score, theta, tail_frac, tx, ty, overflow, fill_ratio,
        fitted_mask (H×W bool)
    """
    from scipy.optimize import minimize  # lazy import

    h, w = mask.shape
    mask_u8 = mask.astype(np.uint8) * 255

    # Distance transform of mask edges
    edge = cv2.Canny(mask_u8, 50, 150)
    dist = cv2.distanceTransform((255 - edge), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    # Mask centroid & fill ratio (computed once)
    ys, xs = np.nonzero(mask)
    if ys.size < 10:
        raise ValueError("mask too small for chamfer match")
    m_cx, m_cy = float(xs.mean()), float(ys.mean())
    bbox_area = float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
    fill_ratio = float(mask.sum()) / bbox_area

    best_global: dict[str, object] = {"score": float("inf")}

    # Load or compute template edge points
    _edge_dir = Path(edge_dir) if edge_dir is not None else None
    tmpl_edges: dict[str, tuple[np.ndarray, float, float]] = {}
    for name, tmpl in templates.items():
        edge_img = None
        if _edge_dir is not None:
            edge_path = _edge_dir / f"{name}.png"
            if edge_path.exists():
                edge_img = cv2.imread(str(edge_path), cv2.IMREAD_GRAYSCALE)
        if edge_img is not None:
            ey, ex = np.nonzero(edge_img)
            pts = np.stack([ex, ey], axis=-1).astype(np.float64)
        else:
            pts = _extract_edge_points(tmpl)
        if pts.shape[0] < 5:
            continue
        tmpl_edges[name] = (pts, float(tmpl.shape[1]), float(tmpl.shape[0]))

    thetas = np.linspace(0, 2 * np.pi, angle_steps, endpoint=False)

    for name, (pts_full, t_w, t_h) in tmpl_edges.items():
        cx_full = t_w / 2.0
        cy_full = t_h / 2.0

        # ---------- Phase 1: full template (tail_frac = 1.0) ----------
        def _cost_full(params: np.ndarray) -> float:
            theta, tx, ty = params
            transformed = _transform_points(
                pts_full, cx=cx_full, cy=cy_full,
                theta=theta, scale=1.0, tx=tx, ty=ty,
            )
            xi = np.clip(transformed[:, 0], 0, w - 1).astype(np.int32)
            yi = np.clip(transformed[:, 1], 0, h - 1).astype(np.int32)
            return float(np.mean(dist[yi, xi]))

        best_cost = float("inf")
        best_params = (0.0, m_cx, m_cy)
        for theta in thetas:
            c = _cost_full(np.array([theta, m_cx, m_cy]))
            if c < best_cost:
                best_cost = c
                best_params = (theta, m_cx, m_cy)

        res_full = minimize(
            _cost_full, np.array(best_params), method="Nelder-Mead",
            options={"maxiter": refine_iters, "xatol": 0.5, "fatol": 0.1},
        )
        score_full = float(res_full.fun)

        # Compute overflow: fraction of template edge points outside mask
        tf_pts = _transform_points(
            pts_full, cx=cx_full, cy=cy_full,
            theta=float(res_full.x[0]), scale=1.0,
            tx=float(res_full.x[1]), ty=float(res_full.x[2]),
        )
        xi = np.clip(tf_pts[:, 0], 0, w - 1).astype(np.int32)
        yi = np.clip(tf_pts[:, 1], 0, h - 1).astype(np.int32)
        overflow = float(np.mean(~mask[yi, xi]))

        result_full = {
            "type": name, "score": score_full, "tail_frac": 1.0,
            "theta": float(res_full.x[0]),
            "tx": float(res_full.x[1]), "ty": float(res_full.x[2]),
            "overflow": overflow, "fill_ratio": fill_ratio,
        }

        # ---------- Phase 2: tail-cropped ----------
        # triggered only when fill ratio is low AND template overflows mask
        result_best = result_full
        if fill_ratio < fill_threshold and overflow > overflow_threshold:
            tail_fracs = np.linspace(0.5, 0.95, tail_steps)

            def _cost_tail(params: np.ndarray) -> float:
                theta, tail_f, tx, ty = params
                tail_f = float(np.clip(tail_f, 0.4, 1.0))
                max_y = t_h * tail_f
                keep = pts_full[:, 1] < max_y
                pts = pts_full[keep]
                if pts.shape[0] < 5:
                    return 1e6
                cy = max_y / 2.0
                transformed = _transform_points(
                    pts, cx=cx_full, cy=cy,
                    theta=theta, scale=1.0, tx=tx, ty=ty,
                )
                xi = np.clip(transformed[:, 0], 0, w - 1).astype(np.int32)
                yi = np.clip(transformed[:, 1], 0, h - 1).astype(np.int32)
                return float(np.mean(dist[yi, xi]))

            best_cost_t = float("inf")
            best_params_t = (0.0, 0.75, m_cx, m_cy)
            for theta in thetas:
                for tf in tail_fracs:
                    c = _cost_tail(np.array([theta, tf, m_cx, m_cy]))
                    if c < best_cost_t:
                        best_cost_t = c
                        best_params_t = (theta, tf, m_cx, m_cy)

            res_tail = minimize(
                _cost_tail, np.array(best_params_t), method="Nelder-Mead",
                options={"maxiter": refine_iters, "xatol": 0.5, "fatol": 0.1},
            )
            score_tail = float(res_tail.fun)
            if score_tail < score_full:
                tail_f = float(np.clip(res_tail.x[1], 0.4, 1.0))
                result_best = {
                    "type": name, "score": score_tail, "tail_frac": tail_f,
                    "theta": float(res_tail.x[0]),
                    "tx": float(res_tail.x[2]), "ty": float(res_tail.x[3]),
                    "overflow": overflow, "fill_ratio": fill_ratio,
                }

        if result_best["score"] < float(best_global["score"]):
            best_global = {**result_best, "_tmpl_name": name}

    if best_global.get("type") is None:
        raise RuntimeError("chamfer match failed – no valid template")

    # ---- render the best-fit template as a mask ----
    name = str(best_global["_tmpl_name"])
    tmpl = templates[name]
    t_h_t, t_w_t = tmpl.shape
    theta = float(best_global["theta"])
    tail_f = float(best_global["tail_frac"])
    tx = float(best_global["tx"])
    ty = float(best_global["ty"])

    crop_h = int(round(t_h_t * tail_f))
    tmpl_cropped = tmpl[:crop_h, :]

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cx = t_w_t / 2.0
    cy = crop_h / 2.0
    a, b = cos_t, -sin_t
    c_val, d = sin_t, cos_t
    tx_aff = tx - a * cx - b * cy
    ty_aff = ty - c_val * cx - d * cy
    mat = np.array([[a, b, tx_aff], [c_val, d, ty_aff]], dtype=np.float32)
    warped = cv2.warpAffine(
        tmpl_cropped.astype(np.uint8) * 255, mat, (w, h),
        flags=cv2.INTER_NEAREST, borderValue=0,
    )
    fitted_mask = warped > 0
    best_global["fitted_mask"] = fitted_mask
    del best_global["_tmpl_name"]
    return best_global


def normalize_masks(masks: np.ndarray) -> np.ndarray:
    masks = np.asarray(masks)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    elif masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim != 3:
        raise ValueError(f"Unexpected masks shape: {masks.shape}")
    return masks.astype(bool)


# ---------------------------------------------------------------------------
# Arrow pattern canvas & parametric drawing (from arrow_pattern.py)
# ---------------------------------------------------------------------------

from dataclasses import dataclass


class ArrowCanvas:
    def __init__(self, res_cm_per_px: float = 2.0):
        self.res = res_cm_per_px
        self.polygons: list[np.ndarray] = []

    def clear(self) -> None:
        self.polygons = []

    def add_polygon(self, vertices) -> None:
        self.polygons.append(np.array(vertices, dtype=np.float32))

    def add_rect(self, x: float, y: float, w: float, h: float) -> None:
        pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        self.add_polygon(pts)

    def add_parallelogram(self, x: float, y: float, base_w: float, h: float, skew_x: float) -> None:
        pts = [(x, y), (x + base_w, y), (x + base_w + skew_x, y + h), (x + skew_x, y + h)]
        self.add_polygon(pts)

    def add_triangle(self, p1, p2, p3) -> None:
        self.add_polygon([p1, p2, p3])

    def export(self, save_path, padding_cm: float = 10.0) -> None:
        if not self.polygons:
            print("画板是空的，无法导出。")
            return
        all_pts = np.vstack(self.polygons)
        min_x, min_y = np.min(all_pts, axis=0)
        max_x, max_y = np.max(all_pts, axis=0)
        min_x -= padding_cm
        max_x += padding_cm
        min_y -= padding_cm
        max_y += padding_cm
        width_px = int((max_x - min_x) / self.res)
        height_px = int((max_y - min_y) / self.res)
        img = np.zeros((height_px, width_px), dtype=np.uint8)
        for poly in self.polygons:
            pts_px = np.zeros_like(poly)
            pts_px[:, 0] = (poly[:, 0] - min_x) / self.res
            pts_px[:, 1] = (max_y - poly[:, 1]) / self.res
            cv2.fillPoly(img, [pts_px.astype(np.int32)], 255)
        cv2.imwrite(str(save_path), img)
        print(f"图像已导出至: {save_path} (尺寸: {width_px}x{height_px} px)")


@dataclass
class Tri:
    width: float
    height: float
    mid: float


@dataclass
class Rect:
    width: float
    height: float


@dataclass
class Straight:
    main_arrow: Tri
    main_stem: Rect


@dataclass
class LeftStraight:
    main_arrow: Tri
    main_stem: Rect
    bot_gap: float
    par_width: float
    par_heigth: float
    turn_arrow: Tri
    arrow_gap: float


@dataclass
class TurnArrow:
    main_stem: Rect
    par_height: float
    par_width: float
    main_arrow: Tri
    arrow_gap: float
    is_left: bool


def draw_straight(canvas: ArrowCanvas, tar: Straight, save_path) -> None:
    canvas.clear()
    canvas.add_rect(
        x=-tar.main_stem.width / 2, y=-tar.main_stem.height,
        w=tar.main_stem.width, h=tar.main_stem.height,
    )
    canvas.add_triangle(
        (-tar.main_arrow.width / 2, 0),
        (tar.main_arrow.width / 2, 0),
        (0, tar.main_arrow.height),
    )
    canvas.export(save_path, padding_cm=5)


def draw_left_straight(canvas: ArrowCanvas, tar: LeftStraight, save_path) -> None:
    canvas.clear()
    canvas.add_rect(
        x=-tar.main_stem.width / 2, y=-tar.main_stem.height,
        w=tar.main_stem.width, h=tar.main_stem.height,
    )
    canvas.add_triangle(
        (-tar.main_arrow.width / 2, 0),
        (tar.main_arrow.width / 2, 0),
        (0, tar.main_arrow.height),
    )
    par_left_top = (-(tar.main_stem.width / 2 + tar.par_heigth), -(tar.main_stem.height - tar.bot_gap - tar.par_width - tar.par_heigth))
    par_left_bot = (-(tar.main_stem.width / 2 + tar.par_heigth), -(tar.main_stem.height - tar.bot_gap - tar.par_heigth))
    par_pts = [
        (-tar.main_stem.width / 2, -(tar.main_stem.height - tar.bot_gap)),
        (-tar.main_stem.width / 2, -(tar.main_stem.height - tar.bot_gap - tar.par_width)),
        par_left_top,
        par_left_bot,
    ]
    canvas.add_polygon(par_pts)
    left_arrow_pts = [
        (par_left_bot[0], -(-par_left_bot[1] + tar.arrow_gap)),
        (par_left_bot[0], -(-par_left_bot[1] + tar.arrow_gap - tar.turn_arrow.width)),
        (par_left_bot[0] - tar.turn_arrow.height, -(-par_left_bot[1] + tar.arrow_gap - tar.turn_arrow.mid)),
    ]
    canvas.add_polygon(left_arrow_pts)
    canvas.export(save_path, padding_cm=5.0)


def draw_right_straight(canvas: ArrowCanvas, tar: LeftStraight, save_path) -> None:
    canvas.clear()
    canvas.add_rect(
        x=-tar.main_stem.width / 2, y=-tar.main_stem.height,
        w=tar.main_stem.width, h=tar.main_stem.height,
    )
    canvas.add_triangle(
        (-tar.main_arrow.width / 2, 0),
        (tar.main_arrow.width / 2, 0),
        (0, tar.main_arrow.height),
    )
    par_left_top = ((tar.main_stem.width / 2 + tar.par_heigth), -(tar.main_stem.height - tar.bot_gap - tar.par_width - tar.par_heigth))
    par_left_bot = ((tar.main_stem.width / 2 + tar.par_heigth), -(tar.main_stem.height - tar.bot_gap - tar.par_heigth))
    par_pts = [
        (tar.main_stem.width / 2, -(tar.main_stem.height - tar.bot_gap)),
        (tar.main_stem.width / 2, -(tar.main_stem.height - tar.bot_gap - tar.par_width)),
        par_left_top,
        par_left_bot,
    ]
    canvas.add_polygon(par_pts)
    left_arrow_pts = [
        (par_left_bot[0], -(-par_left_bot[1] + tar.arrow_gap)),
        (par_left_bot[0], -(-par_left_bot[1] + tar.arrow_gap - tar.turn_arrow.width)),
        (par_left_bot[0] + tar.turn_arrow.height, -(-par_left_bot[1] + tar.arrow_gap - tar.turn_arrow.mid)),
    ]
    canvas.add_polygon(left_arrow_pts)
    canvas.export(save_path, padding_cm=5.0)


def draw_turn(canvas: ArrowCanvas, tar: TurnArrow, save_path) -> None:
    canvas.clear()
    main_stem_pts = [
        (-tar.main_stem.width / 2, 0),
        (tar.main_stem.width / 2, 0),
        (tar.main_stem.width / 2, -tar.main_stem.height),
        (-tar.main_stem.width / 2, -tar.main_stem.height),
    ]
    canvas.add_polygon(main_stem_pts)
    par_pts = [
        (tar.main_stem.width / 2, -tar.main_stem.width),
        (tar.main_stem.width / 2, -tar.main_stem.width + tar.par_width),
        (-(tar.par_height - tar.main_stem.width / 2), -tar.main_stem.width + tar.par_width + tar.par_height),
        (-(tar.par_height - tar.main_stem.width / 2), -tar.main_stem.width + tar.par_width + tar.par_height - tar.par_width),
    ]
    par_pts_ = []
    if not tar.is_left:
        for pt in par_pts:
            par_pts_.append((-pt[0], pt[1]))
    else:
        par_pts_ = par_pts
    canvas.add_polygon(par_pts_)
    par_left_bot_y = -tar.main_stem.width + tar.par_width + tar.par_height - tar.par_width
    arrow_pts = [
        (-(tar.par_height - tar.main_stem.width / 2), par_left_bot_y - tar.arrow_gap + tar.main_arrow.width),
        (-(tar.par_height - tar.main_stem.width / 2), par_left_bot_y - tar.arrow_gap),
        (-(tar.par_height - tar.main_stem.width / 2 + tar.main_arrow.height), par_left_bot_y - tar.arrow_gap + tar.main_arrow.mid),
    ]
    arrow_pts_ = []
    if not tar.is_left:
        for pt in arrow_pts:
            arrow_pts_.append((-pt[0], pt[1]))
    else:
        arrow_pts_ = arrow_pts
    canvas.add_polygon(arrow_pts_)
    canvas.export(save_path, padding_cm=5)


def generate_default_arrow_templates(output_dir: Path | str = "./asserts/arrow_templates") -> None:
    """Generate the default set of arrow template images."""
    home_dir = Path(output_dir)
    home_dir.mkdir(parents=True, exist_ok=True)
    canvas = ArrowCanvas(res_cm_per_px=2.0)

    straight = Straight(Tri(90, 240, 45), Rect(30, 360))
    draw_straight(canvas, straight, str(home_dir / "straight.png"))

    straight_turn = LeftStraight(Tri(90, 240, 45), Rect(30, 360), 40, 120, 80, Tri(310, 40, 150), 90)
    draw_left_straight(canvas, straight_turn, str(home_dir / "straight_left.png"))
    draw_right_straight(canvas, straight_turn, str(home_dir / "straight_right.png"))

    left_turn = TurnArrow(Rect(30, 300), 110, 120, Tri(310, 40, 150), 90, True)
    draw_turn(canvas, left_turn, str(home_dir / "left.png"))

    right_turn = TurnArrow(Rect(30, 300), 110, 120, Tri(310, 40, 150), 90, False)
    draw_turn(canvas, right_turn, str(home_dir / "right.png"))
