"""Limited-FOV perception and partial map maintenance."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Tuple

import numpy as np

from .environment import Environment
from .obstacles import Obstacle, DynamicObstacle

Point = Tuple[float, float]

UNKNOWN = 0
FREE = 1
OCCUPIED = 2


@dataclass
class KnownMap:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    resolution: float

    def __post_init__(self) -> None:
        self.width = int(math.ceil((self.x_max - self.x_min) / self.resolution))
        self.height = int(math.ceil((self.y_max - self.y_min) / self.resolution))
        self.grid = np.zeros((self.height, self.width), dtype=np.uint8)

    def world_to_grid(self, p: Point) -> Tuple[int, int]:
        gx = int((p[0] - self.x_min) / self.resolution)
        gy = int((p[1] - self.y_min) / self.resolution)
        return gx, gy

    def grid_to_world(self, g: Tuple[int, int]) -> Point:
        return (
            self.x_min + (g[0] + 0.5) * self.resolution,
            self.y_min + (g[1] + 0.5) * self.resolution,
        )

    def in_bounds(self, g: Tuple[int, int]) -> bool:
        return 0 <= g[0] < self.width and 0 <= g[1] < self.height

    def mark_free(self, p: Point) -> None:
        g = self.world_to_grid(p)
        if self.in_bounds(g):
            if self.grid[g[1], g[0]] == UNKNOWN:
                self.grid[g[1], g[0]] = FREE

    def mark_occupied(self, p: Point) -> None:
        g = self.world_to_grid(p)
        if self.in_bounds(g):
            self.grid[g[1], g[0]] = OCCUPIED


@dataclass
class Perception:
    fov_deg: float
    range_max: float
    ray_count: int
    noise_std: float = 0.0

    def sense(self, env: Environment, pose: Tuple[float, float, float], known_map: KnownMap) -> None:
        x, y, heading = pose
        half_fov = math.radians(self.fov_deg) / 2.0
        ray_angles = np.linspace(-half_fov, half_fov, self.ray_count)

        # Ray casting to mark free space and detect obstacles within FOV/range.
        for ang in ray_angles:
            theta = heading + ang
            hit_point = None
            for r in np.linspace(0.0, self.range_max, int(self.range_max / known_map.resolution)):
                px = x + r * math.cos(theta)
                py = y + r * math.sin(theta)
                if not env.in_bounds((px, py)):
                    break
                if any(obs.contains((px, py)) for obs in env.all_obstacles()):
                    hit_point = (px, py)
                    break
                known_map.mark_free((px, py))
            if hit_point:
                known_map.mark_occupied(self._apply_noise(hit_point))

        # Also add boundary points for visible obstacles to improve map quality.
        for obs in env.all_obstacles():
            if self._visible(obs, pose):
                for p in obs.boundary_points():
                    if self._in_fov(p, pose) and self._in_range(p, pose):
                        known_map.mark_occupied(self._apply_noise(p))

    def _visible(self, obs: Obstacle | DynamicObstacle, pose: Tuple[float, float, float]) -> bool:
        # Simple visibility: check nearest boundary point.
        for p in obs.boundary_points(8):
            if self._in_fov(p, pose) and self._in_range(p, pose):
                return True
        return False

    def _in_range(self, p: Point, pose: Tuple[float, float, float]) -> bool:
        dx = p[0] - pose[0]
        dy = p[1] - pose[1]
        return math.hypot(dx, dy) <= self.range_max

    def _in_fov(self, p: Point, pose: Tuple[float, float, float]) -> bool:
        dx = p[0] - pose[0]
        dy = p[1] - pose[1]
        ang = math.atan2(dy, dx)
        rel = self._wrap(ang - pose[2])
        return abs(rel) <= math.radians(self.fov_deg) / 2.0

    def _apply_noise(self, p: Point) -> Point:
        if self.noise_std <= 0.0:
            return p
        return (p[0] + np.random.normal(0.0, self.noise_std), p[1] + np.random.normal(0.0, self.noise_std))

    @staticmethod
    def _wrap(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
