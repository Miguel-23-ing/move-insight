"""
core/drawing.py
Parametrizable drawing utilities for pose overlay visualization.
Refactored and extended from process_all_videos.py – reusable by all modules.

Color palette uses BGR (OpenCV order).
  Teal accent  #00B4D8 → BGR (216, 180,   0)
  Green normal #2ECC71 → BGR (113, 204,  46)
  Orange warn  #F39C12 → BGR ( 18, 156, 243)
  Red alert    #E74C3C → BGR ( 60,  76, 231)
"""
from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

# ── BGR color palette ─────────────────────────────────────────────────────────
COLOR_TEAL   = (216, 180,   0)   # primary accent (#00B4D8)
COLOR_NORMAL = (113, 204,  46)   # normal status  (#2ECC71)
COLOR_WARN   = ( 18, 156, 243)   # attention      (#F39C12)
COLOR_ALERT  = ( 60,  76, 231)   # alert          (#E74C3C)
COLOR_MUTED  = (110, 110, 110)   # non-highlighted elements
COLOR_WHITE  = (255, 255, 255)
COLOR_BLACK  = (  0,   0,   0)
COLOR_PANEL  = ( 20,  20,  30)   # panel background

STATUS_COLORS = {
    "normal":   COLOR_NORMAL,
    "atencion": COLOR_WARN,
    "alerta":   COLOR_ALERT,
}

# ── Font constants ────────────────────────────────────────────────────────────
FONT             = cv2.FONT_HERSHEY_SIMPLEX
PANEL_ALPHA      = 0.40
PANEL_PAD        = 6
PANEL_FSCALE     = 0.46
PANEL_LINE_THICK = 1
PANEL_OUTLINE_T  = 2

# ── Full YOLOv8-Pose skeleton connections (no head) ──────────────────────────
ALL_CONNECTIONS: list[tuple[int, int]] = [
    (5,  6),                  # shoulders
    (5,  7), (7,  9),         # left arm
    (6,  8), (8, 10),         # right arm
    (5, 11), (6, 12),         # torso
    (11, 12),                 # hip belt
    (11, 13), (13, 15),       # left leg
    (12, 14), (14, 16),       # right leg
]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _pt_ok(pt: dict, w: int, h: int, thr: float) -> bool:
    return (
        pt["conf"] >= thr
        and 0 <= int(pt["x"]) < w
        and 0 <= int(pt["y"]) < h
    )


def _put_outlined(
    frame: np.ndarray,
    text: str,
    org: tuple[int, int],
    font_scale: float,
    text_color: tuple,
    text_thick: int,
    outline_thick: int = 2,
) -> None:
    cv2.putText(frame, text, org, FONT, font_scale, COLOR_BLACK, outline_thick, cv2.LINE_AA)
    cv2.putText(frame, text, org, FONT, font_scale, text_color,  text_thick,    cv2.LINE_AA)


def _rects_overlap(a: tuple, b: tuple) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


# ── Public API ────────────────────────────────────────────────────────────────

def draw_skeleton(
    frame: np.ndarray,
    pts: list[dict],
    highlight_kps: Optional[list[int]] = None,
    visibility_threshold: float = 0.10,
    line_thickness: int = 3,
    point_radius: int = 6,
) -> None:
    """Draw skeleton with highlighted zone (teal) and rest in muted gray.

    Parameters
    ----------
    frame                : BGR frame – drawn on in-place
    pts                  : 17 smoothed keypoint dicts {x, y, conf}
    highlight_kps        : keypoint indices for the zone of interest (drawn in teal)
    visibility_threshold : minimum confidence to draw a point/line
    """
    h, w = frame.shape[:2]
    hi_set = set(highlight_kps or [])

    # Determine which connections to highlight
    hi_conns = {
        (a, b) for a, b in ALL_CONNECTIONS
        if a in hi_set and b in hi_set
    }

    # Draw connections
    for a, b in ALL_CONNECTIONS:
        pa, pb = pts[a], pts[b]
        if not (_pt_ok(pa, w, h, visibility_threshold) and _pt_ok(pb, w, h, visibility_threshold)):
            continue
        p1 = (int(pa["x"]), int(pa["y"]))
        p2 = (int(pb["x"]), int(pb["y"]))
        is_hi = (a, b) in hi_conns or (b, a) in hi_conns
        color = COLOR_TEAL  if is_hi else COLOR_MUTED
        thick = line_thickness if is_hi else max(1, line_thickness - 2)
        cv2.line(frame, p1, p2, color, thick, cv2.LINE_AA)

    # Draw keypoints on top of connections
    for i, pt in enumerate(pts):
        if not _pt_ok(pt, w, h, visibility_threshold):
            continue
        x, y   = int(pt["x"]), int(pt["y"])
        is_hi  = i in hi_set
        color  = COLOR_TEAL if is_hi else COLOR_MUTED
        r      = point_radius if is_hi else max(3, point_radius - 3)
        cv2.circle(frame, (x, y), r + 1, COLOR_BLACK, -1)
        cv2.circle(frame, (x, y), r, color, -1)
        if is_hi:
            cv2.circle(frame, (x, y), max(1, r - 2), COLOR_WHITE, 1)


def draw_angle_arc(
    frame: np.ndarray,
    vertex: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    angle_deg: float,
    color: tuple = COLOR_TEAL,
    radius: int = 28,
    label: Optional[str] = None,
) -> None:
    """Draw a colored arc at *vertex* representing the angle p1–vertex–p2."""
    vx, vy = int(vertex[0]), int(vertex[1])
    v1 = p1 - vertex
    v2 = p2 - vertex
    a1 = math.degrees(math.atan2(float(v1[1]), float(v1[0])))
    a2 = math.degrees(math.atan2(float(v2[1]), float(v2[0])))
    start_a, end_a = sorted([a1, a2])
    # Always draw the shorter arc
    if end_a - start_a > 180:
        start_a, end_a = end_a, start_a + 360.0
    try:
        cv2.ellipse(frame, (vx, vy), (radius, radius), 0.0, start_a, end_a, color, 2, cv2.LINE_AA)
    except Exception:
        pass

    # Angle label near midpoint of arc
    mid_rad = math.radians((start_a + end_a) / 2.0)
    lx = int(vx + (radius + 14) * math.cos(mid_rad))
    ly = int(vy + (radius + 14) * math.sin(mid_rad))
    text = label if label is not None else f"{angle_deg:.0f}\u00b0"
    _put_outlined(frame, text, (lx, ly), 0.44, color, 1)


def draw_metric_panel(
    frame: np.ndarray,
    lines: list[tuple[str, tuple]],
    origin: tuple[int, int] = (10, 10),
) -> tuple[int, int, int, int] | None:
    """Draw a semi-transparent text panel on *frame*.

    Parameters
    ----------
    lines  : list of (text, BGR_color) tuples
    origin : (x0, y0) top-left corner of the panel

    Returns
    -------
    (x0, y0, x1, y1) bounding box of the drawn panel, or None.
    """
    if not lines:
        return None

    h, w = frame.shape[:2]
    x0, y0 = origin

    sizes     = [cv2.getTextSize(t, FONT, PANEL_FSCALE, PANEL_LINE_THICK) for t, _ in lines]
    widths    = [s[0][0] for s in sizes]
    txt_h     = max(s[0][1] for s in sizes)
    base_h    = max(s[1]    for s in sizes)
    line_h    = txt_h + base_h + 4

    panel_w = int(max(widths) + 2 * PANEL_PAD)
    panel_h = int(len(lines) * line_h + 2 * PANEL_PAD)

    x1 = min(w - 1, x0 + panel_w)
    y1 = min(h - 1, y0 + panel_h)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), COLOR_PANEL, -1)
    cv2.addWeighted(overlay, PANEL_ALPHA, frame, 1.0 - PANEL_ALPHA, 0.0, frame)

    y = y0 + PANEL_PAD + txt_h
    for text, color in lines:
        _put_outlined(frame, text, (x0 + PANEL_PAD, y), PANEL_FSCALE, color, PANEL_LINE_THICK)
        y += line_h

    return (x0, y0, x1, y1)


def draw_joint_tag(
    frame: np.ndarray,
    pt: dict,
    text: str,
    side: str,                              # "L" | "R"
    occupied: list[tuple[int, int, int, int]],
    threshold: float = 0.10,
    color: tuple = COLOR_TEAL,
) -> None:
    """Draw a small labeled tag near a joint, avoiding overlaps.

    Adapted from _draw_tag() in process_all_videos.py.
    """
    h, w = frame.shape[:2]
    if pt["conf"] < threshold:
        return
    x, y = int(pt["x"]), int(pt["y"])
    if not (0 <= x < w and 0 <= y < h):
        return

    (tw, th), base = cv2.getTextSize(text, FONT, 0.38, 1)
    pad = 4
    if side == "L":
        candidates = [(-tw - 16, -th - 8), (-tw - 16, 6), (-tw - 16, -2 * th - 12)]
    else:
        candidates = [(16, -th - 8), (16, 6), (16, -2 * th - 12)]

    for dx, dy in candidates:
        bx1 = max(0, x + dx)
        by1 = max(0, y + dy)
        bx2 = min(w - 1, bx1 + tw + 2 * pad)
        by2 = min(h - 1, by1 + th + base + 2 * pad)
        box = (bx1, by1, bx2, by2)
        if any(_rects_overlap(box, o) for o in occupied):
            continue
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), COLOR_BLACK, -1)
        cv2.putText(frame, text, (bx1 + pad, by1 + pad + th), FONT, 0.38, color, 1, cv2.LINE_AA)
        occupied.append(box)
        return
