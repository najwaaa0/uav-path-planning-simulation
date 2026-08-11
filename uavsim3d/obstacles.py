"""3D obstacle primitives, including arbitrary extruded footprints."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List, Sequence, Tuple

from .geometry import Vec2, Vec3, distance, distance_point_to_polygon_2d, point_in_polygon, polygon_centroid, triangulate_polygon

RGB = Tuple[int, int, int]


@dataclass
class Obstacle3D:
    label: str = "obstacle"
    color: RGB = (90, 110, 125)

    def contains(self, point: Vec3) -> bool:
        raise NotImplementedError

    def distance(self, point: Vec3) -> float:
        raise NotImplementedError

    def centroid(self) -> Vec3:
        raise NotImplementedError


@dataclass
class SphereObstacle3D(Obstacle3D):
    center: Vec3 = (0.0, 0.0, 0.0)
    radius: float = 1.0

    def contains(self, point: Vec3) -> bool:
        return distance(self.center, point) <= self.radius

    def distance(self, point: Vec3) -> float:
        return max(0.0, distance(self.center, point) - self.radius)

    def centroid(self) -> Vec3:
        return self.center


@dataclass
class MovingSphereObstacle3D(SphereObstacle3D):
    velocity: Vec3 = (0.0, 0.0, 0.0)
    trail: List[Vec3] = field(default_factory=list)
    trail_limit: int = 25

    def __post_init__(self) -> None:
        self._remember()

    def update(self, dt: float, bounds: tuple[float, float, float, float, float, float]) -> None:
        if dt <= 0.0:
            return
        self.center = (
            self.center[0] + self.velocity[0] * dt,
            self.center[1] + self.velocity[1] * dt,
            self.center[2] + self.velocity[2] * dt,
        )
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        cx, cy, cz = self.center
        vx, vy, vz = self.velocity
        if cx - self.radius < x_min:
            cx = x_min + self.radius
            vx = abs(vx)
        elif cx + self.radius > x_max:
            cx = x_max - self.radius
            vx = -abs(vx)
        if cy - self.radius < y_min:
            cy = y_min + self.radius
            vy = abs(vy)
        elif cy + self.radius > y_max:
            cy = y_max - self.radius
            vy = -abs(vy)
        if cz - self.radius < z_min:
            cz = z_min + self.radius
            vz = abs(vz)
        elif cz + self.radius > z_max:
            cz = z_max - self.radius
            vz = -abs(vz)
        self.center = (cx, cy, cz)
        self.velocity = (vx, vy, vz)
        self._remember()

    def distance_at_time(self, point: Vec3, dt: float) -> float:
        center = (
            self.center[0] + self.velocity[0] * dt,
            self.center[1] + self.velocity[1] * dt,
            self.center[2] + self.velocity[2] * dt,
        )
        return max(0.0, distance(center, point) - self.radius)

    def _remember(self) -> None:
        self.trail.append(self.center)
        if len(self.trail) > self.trail_limit:
            del self.trail[0]


@dataclass
class ExtrudedPolygonObstacle(Obstacle3D):
    footprint: Sequence[Vec2] = ()
    z_min: float = 0.0
    z_max: float = 1.0

    def contains(self, point: Vec3) -> bool:
        return self.z_min <= point[2] <= self.z_max and point_in_polygon((point[0], point[1]), self.footprint)

    def distance(self, point: Vec3) -> float:
        horizontal = distance_point_to_polygon_2d((point[0], point[1]), self.footprint)
        vertical = max(self.z_min - point[2], 0.0, point[2] - self.z_max)
        if horizontal <= 1e-9 and vertical <= 1e-9:
            return 0.0
        return math.hypot(horizontal, vertical)

    def centroid(self) -> Vec3:
        cx, cy = polygon_centroid(self.footprint)
        return (cx, cy, (self.z_min + self.z_max) / 2.0)

    def surface_triangles(self) -> List[Tuple[Vec3, Vec3, Vec3]]:
        tris: List[Tuple[Vec3, Vec3, Vec3]] = []
        top = [(x, y, self.z_max) for x, y in self.footprint]
        bottom = [(x, y, self.z_min) for x, y in self.footprint]
        for i, j, k in triangulate_polygon(self.footprint):
            tris.append((top[i], top[j], top[k]))
            tris.append((bottom[k], bottom[j], bottom[i]))
        for i in range(len(self.footprint)):
            next_i = (i + 1) % len(self.footprint)
            p0 = bottom[i]
            p1 = bottom[next_i]
            p2 = top[next_i]
            p3 = top[i]
            tris.append((p0, p1, p2))
            tris.append((p0, p2, p3))
        return tris
