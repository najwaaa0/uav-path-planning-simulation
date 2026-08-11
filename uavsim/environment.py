"""Environment definition with static/dynamic obstacles and world bounds."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, List, Tuple

from .obstacles import DynamicObstacle, Obstacle

Point = Tuple[float, float]


@dataclass
class Environment:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    static_obstacles: List[Obstacle] = field(default_factory=list)
    dynamic_obstacles: List[DynamicObstacle] = field(default_factory=list)

    def in_bounds(self, p: Point) -> bool:
        return self.x_min <= p[0] <= self.x_max and self.y_min <= p[1] <= self.y_max

    def all_obstacles(self) -> Iterable[Obstacle | DynamicObstacle]:
        return list(self.static_obstacles) + list(self.dynamic_obstacles)

    def update(self, dt: float) -> None:
        bounds = (self.x_min, self.x_max, self.y_min, self.y_max)
        for obs in self.dynamic_obstacles:
            obs.update(dt, bounds)

    def distance_to_obstacles(
        self,
        p: Point,
        time_horizon: float = 0.0,
        samples: int = 1,
        safety_margin: float = 0.0,
    ) -> float:
        distances: list[float] = []
        for obs in self.static_obstacles:
            distances.append(max(0.0, obs.distance(p) - safety_margin))
        horizon_samples = max(1, samples)
        for obs in self.dynamic_obstacles:
            closest = math.inf
            for i in range(horizon_samples):
                tau = 0.0 if horizon_samples == 1 else time_horizon * i / (horizon_samples - 1)
                closest = min(closest, max(0.0, obs.distance_at_time(p, tau) - safety_margin))
            distances.append(closest)
        if not distances:
            return math.inf
        return min(distances)

    def collides(self, p: Point, radius: float = 0.0, time_offset: float = 0.0, safety_margin: float = 0.0) -> bool:
        if not self.in_bounds(p):
            return True
        clearance = radius + safety_margin
        for obs in self.static_obstacles:
            if obs.distance(p) <= clearance:
                return True
        for obs in self.dynamic_obstacles:
            if obs.distance_at_time(p, time_offset) <= clearance:
                return True
        return False

    def segment_collides(self, a: Point, b: Point, radius: float = 0.0, dt: float = 0.0, safety_margin: float = 0.0) -> bool:
        spatial_steps = int(math.hypot(b[0] - a[0], b[1] - a[1]) / 0.25)
        temporal_steps = int(dt / 0.05) if dt > 0.0 else 0
        steps = max(4, spatial_steps, temporal_steps)
        for i in range(steps + 1):
            t = i / steps
            p = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            if self.collides(p, radius=radius, time_offset=t * dt, safety_margin=safety_margin):
                return True
        return False
