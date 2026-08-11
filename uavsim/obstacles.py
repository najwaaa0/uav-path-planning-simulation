"""Obstacle primitives for 2D continuous environments."""

from __future__ import annotations

from dataclasses import dataclass, field
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

    def translated(self, dx: float, dy: float) -> "Obstacle":
        raise NotImplementedError

    def bounds(self) -> tuple[float, float, float, float]:
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

    def translated(self, dx: float, dy: float) -> "CircleObstacle":
        return CircleObstacle((self.center[0] + dx, self.center[1] + dy), self.radius)

    def bounds(self) -> tuple[float, float, float, float]:
        return (
            self.center[0] - self.radius,
            self.center[0] + self.radius,
            self.center[1] - self.radius,
            self.center[1] + self.radius,
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

    def translated(self, dx: float, dy: float) -> "RectObstacle":
        return RectObstacle(
            self.x_min + dx,
            self.x_max + dx,
            self.y_min + dy,
            self.y_max + dy,
        )

    def bounds(self) -> tuple[float, float, float, float]:
        return (self.x_min, self.x_max, self.y_min, self.y_max)


@dataclass
class DynamicObstacle:
    """Dynamic obstacle with simple motion model (velocity or waypoint-based)."""

    base: Obstacle
    velocity: Point | None = None
    waypoints: list[Point] | None = None
    waypoint_speed: float = 1.0
    name: str = "dynamic obstacle"
    color: str = "tomato"
    current_velocity: Point = (0.0, 0.0)
    trail: list[Point] = field(default_factory=list)
    trail_limit: int = 30
    _waypoint_idx: int = 0

    def __post_init__(self) -> None:
        self._remember_position()

    def update(self, dt: float, world_bounds: tuple[float, float, float, float] | None = None) -> None:
        if dt <= 0.0:
            self.current_velocity = self.velocity_vector
            self._remember_position()
            return
        if self.velocity is not None:
            self.current_velocity = self.velocity
            dx = self.velocity[0] * dt
            dy = self.velocity[1] * dt
            self._translate(dx, dy)
            if world_bounds is not None:
                self._reflect_off_bounds(world_bounds)
        elif self.waypoints:
            target = self.waypoints[self._waypoint_idx]
            cx, cy = self._center()
            vec = (target[0] - cx, target[1] - cy)
            dist = math.hypot(vec[0], vec[1])
            if dist < 1e-6:
                self._waypoint_idx = (self._waypoint_idx + 1) % len(self.waypoints)
                self.current_velocity = (0.0, 0.0)
                self._remember_position()
                return
            step = min(self.waypoint_speed * dt, dist)
            dx = step * vec[0] / dist
            dy = step * vec[1] / dist
            self.current_velocity = (dx / dt, dy / dt)
            self._translate(dx, dy)
            if step >= dist - 1e-6:
                self._waypoint_idx = (self._waypoint_idx + 1) % len(self.waypoints)
        else:
            self.current_velocity = (0.0, 0.0)
        self._remember_position()

    def _center(self) -> Point:
        if isinstance(self.base, CircleObstacle):
            return self.base.center
        return ((self.base.x_min + self.base.x_max) / 2.0, (self.base.y_min + self.base.y_max) / 2.0)

    @property
    def position(self) -> Point:
        return self._center()

    @property
    def velocity_vector(self) -> Point:
        if self.velocity is not None:
            return self.velocity
        if self.waypoints:
            target = self.waypoints[self._waypoint_idx]
            cx, cy = self._center()
            dx = target[0] - cx
            dy = target[1] - cy
            dist = math.hypot(dx, dy)
            if dist > 1e-6:
                return (self.waypoint_speed * dx / dist, self.waypoint_speed * dy / dist)
        return self.current_velocity

    def footprint_at_time(self, dt: float) -> Obstacle:
        vx, vy = self.velocity_vector
        return self.base.translated(vx * dt, vy * dt)

    def distance_at_time(self, p: Point, dt: float) -> float:
        return self.footprint_at_time(dt).distance(p)

    def _translate(self, dx: float, dy: float) -> None:
        if isinstance(self.base, CircleObstacle):
            self.base.center = (self.base.center[0] + dx, self.base.center[1] + dy)
        else:
            self.base.x_min += dx
            self.base.x_max += dx
            self.base.y_min += dy
            self.base.y_max += dy

    def _reflect_off_bounds(self, world_bounds: tuple[float, float, float, float]) -> None:
        if self.velocity is None:
            return
        x_min, x_max, y_min, y_max = world_bounds
        obs_x_min, obs_x_max, obs_y_min, obs_y_max = self.base.bounds()
        vx, vy = self.velocity
        if obs_x_min < x_min:
            self._translate(x_min - obs_x_min, 0.0)
            vx = abs(vx)
        elif obs_x_max > x_max:
            self._translate(x_max - obs_x_max, 0.0)
            vx = -abs(vx)
        if obs_y_min < y_min:
            self._translate(0.0, y_min - obs_y_min)
            vy = abs(vy)
        elif obs_y_max > y_max:
            self._translate(0.0, y_max - obs_y_max)
            vy = -abs(vy)
        self.velocity = (vx, vy)
        self.current_velocity = (vx, vy)

    def _remember_position(self) -> None:
        self.trail.append(self.position)
        if len(self.trail) > self.trail_limit:
            del self.trail[0]

    def contains(self, p: Point) -> bool:
        return self.base.contains(p)

    def distance(self, p: Point) -> float:
        return self.base.distance(p)

    def boundary_points(self, count: int = 16) -> Iterable[Point]:
        return self.base.boundary_points(count)

    def segment_intersects(self, a: Point, b: Point) -> bool:
        return self.base.segment_intersects(a, b)
