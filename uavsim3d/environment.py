"""3D environment with extruded buildings, moving spheres, and boid flocks."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, List

from .boids import BoidFlock
from .geometry import Vec3, lerp
from .obstacles import ExtrudedPolygonObstacle, MovingSphereObstacle3D, Obstacle3D


@dataclass
class Environment3D:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    static_obstacles: List[ExtrudedPolygonObstacle] = field(default_factory=list)
    dynamic_spheres: List[MovingSphereObstacle3D] = field(default_factory=list)
    flocks: List[BoidFlock] = field(default_factory=list)
    _cached_dynamic_obstacles: List[MovingSphereObstacle3D] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._refresh_dynamic_cache()

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        return (self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max)

    def in_bounds(self, point: Vec3) -> bool:
        return (
            self.x_min <= point[0] <= self.x_max
            and self.y_min <= point[1] <= self.y_max
            and self.z_min <= point[2] <= self.z_max
        )

    def update(self, dt: float) -> None:
        for obstacle in self.dynamic_spheres:
            obstacle.update(dt, self.bounds)
        for flock in self.flocks:
            flock.update(dt, self.bounds, self.static_obstacles)
        self._refresh_dynamic_cache()

    def dynamic_obstacles(self) -> List[MovingSphereObstacle3D]:
        return list(self._cached_dynamic_obstacles)

    def all_obstacles(self) -> Iterable[Obstacle3D]:
        return list(self.static_obstacles) + self.dynamic_obstacles()

    def collides(self, point: Vec3, radius: float = 0.0, time_offset: float = 0.0, safety_margin: float = 0.0) -> bool:
        if not self.in_bounds(point):
            return True
        clearance = radius + safety_margin
        for obstacle in self.static_obstacles:
            if obstacle.distance(point) <= clearance:
                return True
        for obstacle in self.dynamic_obstacles():
            if obstacle.distance_at_time(point, time_offset) <= clearance:
                return True
        return False

    def distance_to_obstacles(
        self,
        point: Vec3,
        time_horizon: float = 0.0,
        samples: int = 1,
        safety_margin: float = 0.0,
    ) -> float:
        distances: List[float] = []
        for obstacle in self.static_obstacles:
            distances.append(max(0.0, obstacle.distance(point) - safety_margin))
        sample_count = max(1, samples)
        for obstacle in self.dynamic_obstacles():
            closest = math.inf
            for i in range(sample_count):
                tau = 0.0 if sample_count == 1 else time_horizon * i / (sample_count - 1)
                closest = min(closest, max(0.0, obstacle.distance_at_time(point, tau) - safety_margin))
            distances.append(closest)
        return min(distances) if distances else math.inf

    def segment_collides(self, start: Vec3, end: Vec3, radius: float = 0.0, dt: float = 0.0, safety_margin: float = 0.0) -> bool:
        spatial_steps = int(math.dist(start, end) / 0.7)
        temporal_steps = int(dt / 0.08) if dt > 0.0 else 0
        steps = max(5, spatial_steps, temporal_steps)
        for i in range(steps + 1):
            tau = i / steps
            point = lerp(start, end, tau)
            if self.collides(point, radius=radius, time_offset=tau * dt, safety_margin=safety_margin):
                return True
        return False

    def _refresh_dynamic_cache(self) -> None:
        boid_obstacles: List[MovingSphereObstacle3D] = []
        for flock in self.flocks:
            boid_obstacles.extend(flock.obstacle_obstacles())
        self._cached_dynamic_obstacles = list(self.dynamic_spheres) + boid_obstacles
