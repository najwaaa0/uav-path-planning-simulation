"""Obstacle primitives for 2D continuous environments."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Tuple


Point = Tuple[float, float]


@dataclass
class Obstacle:
    """Base class for obstacles with geometry queries."""

    def contains(self, p: Point) -> bool:
        raise NotImplementedError

    def distance(self, p: Point) -> float:
        raise NotImplementedError

    def segment_intersects(self, a: Point, b: Point) -> bool:
        # Sample-based intersection check keeps code simple and robust for 2D thesis use.
        steps = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / 0.5))
        for i in range(steps + 1):
            t = i / steps
            p = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            if self.contains(p):
                return True
        return False

    def boundary_points(self, count: int = 16) -> Iterable[Point]:
        raise NotImplementedError


@dataclass
class CircleObstacle(Obstacle):
    center: Point
    radius: float

    def contains(self, p: Point) -> bool:
        return math.hypot(p[0] - self.center[0], p[1] - self.center[1]) <= self.radius

    def distance(self, p: Point) -> float:
        return max(0.0, math.hypot(p[0] - self.center[0], p[1] - self.center[1]) - self.radius)

    def boundary_points(self, count: int = 16) -> Iterable[Point]:
        for i in range(count):
            ang = 2.0 * math.pi * i / count
            yield (
                self.center[0] + self.radius * math.cos(ang),
                self.center[1] + self.radius * math.sin(ang),
            )


@dataclass
class RectObstacle(Obstacle):
    # Axis-aligned rectangle
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def contains(self, p: Point) -> bool:
        return self.x_min <= p[0] <= self.x_max and self.y_min <= p[1] <= self.y_max

    def distance(self, p: Point) -> float:
        dx = max(self.x_min - p[0], 0.0, p[0] - self.x_max)
        dy = max(self.y_min - p[1], 0.0, p[1] - self.y_max)
        return math.hypot(dx, dy)

    def boundary_points(self, count: int = 16) -> Iterable[Point]:
        per_side = max(2, count // 4)
        xs = [self.x_min + i * (self.x_max - self.x_min) / (per_side - 1) for i in range(per_side)]
        ys = [self.y_min + i * (self.y_max - self.y_min) / (per_side - 1) for i in range(per_side)]
        for x in xs:
            yield (x, self.y_min)
            yield (x, self.y_max)
        for y in ys:
            yield (self.x_min, y)
            yield (self.x_max, y)


@dataclass
class DynamicObstacle:
    """Dynamic obstacle with simple motion model (velocity or waypoint-based)."""

    base: Obstacle
    velocity: Point | None = None
    waypoints: list[Point] | None = None
    _waypoint_idx: int = 0

    def update(self, dt: float) -> None:
        if self.velocity is not None:
            dx = self.velocity[0] * dt
            dy = self.velocity[1] * dt
            self._translate(dx, dy)
        elif self.waypoints:
            target = self.waypoints[self._waypoint_idx]
            cx, cy = self._center()
            vec = (target[0] - cx, target[1] - cy)
            dist = math.hypot(vec[0], vec[1])
            if dist < 0.5:
                self._waypoint_idx = (self._waypoint_idx + 1) % len(self.waypoints)
                return
            step = 1.0 * dt
            dx = step * vec[0] / dist
            dy = step * vec[1] / dist
            self._translate(dx, dy)

    def _center(self) -> Point:
        if isinstance(self.base, CircleObstacle):
            return self.base.center
        return ((self.base.x_min + self.base.x_max) / 2.0, (self.base.y_min + self.base.y_max) / 2.0)

    def _translate(self, dx: float, dy: float) -> None:
        if isinstance(self.base, CircleObstacle):
            self.base.center = (self.base.center[0] + dx, self.base.center[1] + dy)
        else:
            self.base.x_min += dx
            self.base.x_max += dx
            self.base.y_min += dy
            self.base.y_max += dy

    def contains(self, p: Point) -> bool:
        return self.base.contains(p)

    def distance(self, p: Point) -> float:
        return self.base.distance(p)

    def boundary_points(self, count: int = 16) -> Iterable[Point]:
        return self.base.boundary_points(count)

    def segment_intersects(self, a: Point, b: Point) -> bool:
        return self.base.segment_intersects(a, b)
