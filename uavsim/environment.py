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
        for obs in self.dynamic_obstacles:
            obs.update(dt)

    def distance_to_obstacles(self, p: Point) -> float:
        distances = [obs.distance(p) for obs in self.all_obstacles()]
        if not distances:
            return math.inf
        return min(distances)

    def collides(self, p: Point, radius: float = 0.0) -> bool:
        if not self.in_bounds(p):
            return True
        for obs in self.all_obstacles():
            if obs.distance(p) <= radius:
                return True
        return False

    def segment_collides(self, a: Point, b: Point, radius: float = 0.0) -> bool:
        # Simple inflation by checking points along the segment.
        steps = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / 0.5))
        for i in range(steps + 1):
            t = i / steps
            p = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            if self.collides(p, radius=radius):
                return True
        return False
