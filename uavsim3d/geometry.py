"""Geometry helpers for continuous 3D path planning and rendering."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(v: Vec3, s: float) -> Vec3:
    return (v[0] * s, v[1] * s, v[2] * s)


def dot(a: Vec3, b: Vec3) -> float:
    return (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2])


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def length(v: Vec3) -> float:
    return math.sqrt(dot(v, v))


def distance(a: Vec3, b: Vec3) -> float:
    return length(sub(a, b))


def normalize(v: Vec3) -> Vec3:
    mag = length(v)
    if mag <= 1e-9:
        return (0.0, 0.0, 0.0)
    return scale(v, 1.0 / mag)


def lerp(a: Vec3, b: Vec3, t: float) -> Vec3:
    return (
        a[0] + t * (b[0] - a[0]),
        a[1] + t * (b[1] - a[1]),
        a[2] + t * (b[2] - a[2]),
    )


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def polygon_signed_area(vertices: Sequence[Vec2]) -> float:
    area = 0.0
    for i in range(len(vertices)):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % len(vertices)]
        area += (x1 * y2) - (x2 * y1)
    return 0.5 * area


def ensure_ccw(vertices: Sequence[Vec2]) -> List[Vec2]:
    pts = list(vertices)
    return pts if polygon_signed_area(pts) >= 0.0 else list(reversed(pts))


def polygon_centroid(vertices: Sequence[Vec2]) -> Vec2:
    area = polygon_signed_area(vertices)
    if abs(area) <= 1e-9:
        x = sum(v[0] for v in vertices) / len(vertices)
        y = sum(v[1] for v in vertices) / len(vertices)
        return (x, y)
    cx = 0.0
    cy = 0.0
    for i in range(len(vertices)):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % len(vertices)]
        cross_term = (x1 * y2) - (x2 * y1)
        cx += (x1 + x2) * cross_term
        cy += (y1 + y2) * cross_term
    factor = 1.0 / (6.0 * area)
    return (cx * factor, cy * factor)


def point_in_polygon(point: Vec2, vertices: Sequence[Vec2]) -> bool:
    x, y = point
    inside = False
    for i in range(len(vertices)):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % len(vertices)]
        intersects = ((y1 > y) != (y2 > y)) and (x < ((x2 - x1) * (y - y1) / max(1e-9, y2 - y1)) + x1)
        if intersects:
            inside = not inside
    return inside


def distance_point_to_segment_2d(point: Vec2, a: Vec2, b: Vec2) -> float:
    ax, ay = a
    bx, by = b
    px, py = point
    abx = bx - ax
    aby = by - ay
    denom = (abx * abx) + (aby * aby)
    if denom <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * abx + (py - ay) * aby) / denom
    t = clamp(t, 0.0, 1.0)
    closest = (ax + t * abx, ay + t * aby)
    return math.hypot(px - closest[0], py - closest[1])


def distance_point_to_polygon_2d(point: Vec2, vertices: Sequence[Vec2]) -> float:
    if point_in_polygon(point, vertices):
        return 0.0
    return min(
        distance_point_to_segment_2d(point, vertices[i], vertices[(i + 1) % len(vertices)])
        for i in range(len(vertices))
    )


def _is_convex(prev: Vec2, curr: Vec2, nxt: Vec2) -> bool:
    return ((curr[0] - prev[0]) * (nxt[1] - curr[1])) - ((curr[1] - prev[1]) * (nxt[0] - curr[0])) > 0.0


def _point_in_triangle(point: Vec2, a: Vec2, b: Vec2, c: Vec2) -> bool:
    def sign(p1: Vec2, p2: Vec2, p3: Vec2) -> float:
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(point, a, b)
    d2 = sign(point, b, c)
    d3 = sign(point, c, a)
    has_neg = (d1 < 0.0) or (d2 < 0.0) or (d3 < 0.0)
    has_pos = (d1 > 0.0) or (d2 > 0.0) or (d3 > 0.0)
    return not (has_neg and has_pos)


def triangulate_polygon(vertices: Sequence[Vec2]) -> List[Tuple[int, int, int]]:
    pts = ensure_ccw(vertices)
    indices = list(range(len(pts)))
    triangles: List[Tuple[int, int, int]] = []
    guard = 0
    while len(indices) > 3 and guard < 10_000:
        ear_found = False
        guard += 1
        for i in range(len(indices)):
            prev_idx = indices[i - 1]
            curr_idx = indices[i]
            next_idx = indices[(i + 1) % len(indices)]
            prev_pt = pts[prev_idx]
            curr_pt = pts[curr_idx]
            next_pt = pts[next_idx]
            if not _is_convex(prev_pt, curr_pt, next_pt):
                continue
            if any(
                other not in {prev_idx, curr_idx, next_idx}
                and _point_in_triangle(pts[other], prev_pt, curr_pt, next_pt)
                for other in indices
            ):
                continue
            triangles.append((prev_idx, curr_idx, next_idx))
            del indices[i]
            ear_found = True
            break
        if not ear_found:
            break
    if len(indices) == 3:
        triangles.append((indices[0], indices[1], indices[2]))
    return triangles


def average(points: Iterable[Vec3]) -> Vec3:
    pts = list(points)
    if not pts:
        return (0.0, 0.0, 0.0)
    inv = 1.0 / len(pts)
    return (
        sum(p[0] for p in pts) * inv,
        sum(p[1] for p in pts) * inv,
        sum(p[2] for p in pts) * inv,
    )
