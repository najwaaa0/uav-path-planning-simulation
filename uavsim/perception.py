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
    dynamic_memory_steps: int = 4

    def __post_init__(self) -> None:
        self.width = int(math.ceil((self.x_max - self.x_min) / self.resolution))
        self.height = int(math.ceil((self.y_max - self.y_min) / self.resolution))
        self.static_grid = np.zeros((self.height, self.width), dtype=np.uint8)
        self.dynamic_grid = np.zeros((self.height, self.width), dtype=np.uint8)
        self.free_grid = np.zeros((self.height, self.width), dtype=np.uint8)
        self.dynamic_age = np.full((self.height, self.width), -1, dtype=np.int16)
        self.grid = np.zeros((self.height, self.width), dtype=np.uint8)

    def world_to_grid(self, p: Point) -> Tuple[int, int]:
        gx = int(math.floor((p[0] - self.x_min) / self.resolution))
        gy = int(math.floor((p[1] - self.y_min) / self.resolution))
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
            self.free_grid[g[1], g[0]] = FREE
            self._sync_cell(g)

    def mark_occupied(self, p: Point, dynamic: bool = False) -> None:
        g = self.world_to_grid(p)
        if self.in_bounds(g):
            self.free_grid[g[1], g[0]] = 0
            if dynamic:
                self.dynamic_grid[g[1], g[0]] = OCCUPIED
                self.dynamic_age[g[1], g[0]] = 0
            else:
                self.static_grid[g[1], g[0]] = OCCUPIED
            self._sync_cell(g)

    def begin_sensor_cycle(self, pose: Tuple[float, float, float], fov_deg: float, range_max: float) -> None:
        self._decay_dynamic_cells()
        self._clear_visible_region(pose, fov_deg, range_max)

    def _decay_dynamic_cells(self) -> None:
        active = self.dynamic_age >= 0
        self.dynamic_age[active] += 1
        stale = self.dynamic_age > self.dynamic_memory_steps
        if not np.any(stale):
            return
        self.dynamic_grid[stale] = 0
        self.dynamic_age[stale] = -1
        for gy, gx in np.argwhere(stale):
            self._sync_cell((gx, gy))

    def _clear_visible_region(self, pose: Tuple[float, float, float], fov_deg: float, range_max: float) -> None:
        x, y, heading = pose
        gx_min = max(0, int(math.floor((x - range_max - self.x_min) / self.resolution)))
        gx_max = min(self.width - 1, int(math.floor((x + range_max - self.x_min) / self.resolution)))
        gy_min = max(0, int(math.floor((y - range_max - self.y_min) / self.resolution)))
        gy_max = min(self.height - 1, int(math.floor((y + range_max - self.y_min) / self.resolution)))
        for gy in range(gy_min, gy_max + 1):
            for gx in range(gx_min, gx_max + 1):
                p = self.grid_to_world((gx, gy))
                if math.hypot(p[0] - x, p[1] - y) > range_max:
                    continue
                if not self._in_fov(p, heading, (x, y), fov_deg):
                    continue
                self.dynamic_grid[gy, gx] = 0
                self.dynamic_age[gy, gx] = -1
                self.free_grid[gy, gx] = 0
                self._sync_cell((gx, gy))

    def _sync_cell(self, g: Tuple[int, int]) -> None:
        gx, gy = g
        if self.static_grid[gy, gx] == OCCUPIED or self.dynamic_grid[gy, gx] == OCCUPIED:
            self.grid[gy, gx] = OCCUPIED
        elif self.free_grid[gy, gx] == FREE:
            self.grid[gy, gx] = FREE
        else:
            self.grid[gy, gx] = UNKNOWN

    @staticmethod
    def _in_fov(p: Point, heading: float, origin: Point, fov_deg: float) -> bool:
        dx = p[0] - origin[0]
        dy = p[1] - origin[1]
        ang = math.atan2(dy, dx)
        rel = KnownMap._wrap(ang - heading)
        return abs(rel) <= math.radians(fov_deg) / 2.0

    @staticmethod
    def _wrap(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


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
        known_map.begin_sensor_cycle(pose, self.fov_deg, self.range_max)

        # Ray casting to mark free space and detect obstacles within FOV/range.
        for ang in ray_angles:
            theta = heading + ang
            hit_point = None
            hit_dynamic = False
            ray_steps = max(2, int(self.range_max / known_map.resolution))
            for r in np.linspace(0.0, self.range_max, ray_steps + 1):
                px = x + r * math.cos(theta)
                py = y + r * math.sin(theta)
                if not env.in_bounds((px, py)):
                    break
                hit_obs = self._hit_obstacle(env, (px, py))
                if hit_obs is not None:
                    hit_point = (px, py)
                    hit_dynamic = isinstance(hit_obs, DynamicObstacle)
                    break
                known_map.mark_free((px, py))
            if hit_point:
                known_map.mark_occupied(self._apply_noise(hit_point), dynamic=hit_dynamic)

        # Also add boundary points for visible obstacles to improve map quality.
        for obs in env.all_obstacles():
            if self._visible(obs, pose):
                for p in obs.boundary_points():
                    if self._in_fov(p, pose) and self._in_range(p, pose):
                        known_map.mark_occupied(self._apply_noise(p), dynamic=isinstance(obs, DynamicObstacle))

    @staticmethod
    def _hit_obstacle(env: Environment, p: Point) -> Obstacle | DynamicObstacle | None:
        for obs in env.all_obstacles():
            if obs.contains(p):
                return obs
        return None

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
