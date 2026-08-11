"""Simple 2D kinematic UAV model."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Tuple

from .obstacles import Obstacle, DynamicObstacle


@dataclass
class UAVState:
    x: float
    y: float
    heading: float

    def as_point(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class UAV:
    state: UAVState
    max_speed: float
    max_turn_rate: float
    body_radius: float

    def predict_step(self, v_cmd: float, w_cmd: float, dt: float) -> Tuple[float, float, float]:
        v = max(-self.max_speed, min(self.max_speed, v_cmd))
        w = max(-self.max_turn_rate, min(self.max_turn_rate, w_cmd))
        heading_mid = self.state.heading + (0.5 * w * dt)
        x = self.state.x + v * dt * math.cos(heading_mid)
        y = self.state.y + v * dt * math.sin(heading_mid)
        heading = self._wrap(self.state.heading + w * dt)
        return x, y, heading

    def step(self, v_cmd: float, w_cmd: float, dt: float) -> None:
        self.state.x, self.state.y, self.state.heading = self.predict_step(v_cmd, w_cmd, dt)

    @staticmethod
    def _wrap(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def control_to_waypoint(self, waypoint: Tuple[float, float], speed: float) -> Tuple[float, float]:
        dx = waypoint[0] - self.state.x
        dy = waypoint[1] - self.state.y
        target = math.atan2(dy, dx)
        err = self._wrap(target - self.state.heading)
        w_cmd = 2.0 * err  # proportional heading controller
        v_cmd = speed * max(0.0, math.cos(err))  # slow down if heading error is large
        return v_cmd, w_cmd


def compute_velocity(
    position: Tuple[float, float],
    obstacles: Iterable[Obstacle | DynamicObstacle],
    max_speed: float = 5.0,
    min_speed: float = 1.0,
    slow_down_radius: float = 8.0,
    stop_radius: float = 1.5,
    safety_margin: float = 0.0,
    prediction_horizon: float = 0.0,
    prediction_steps: int = 1,
) -> float:
    nearest = math.inf
    horizon_steps = max(1, prediction_steps)
    for obs in obstacles:
        closest = math.inf
        for i in range(horizon_steps):
            tau = 0.0 if horizon_steps == 1 else prediction_horizon * i / (horizon_steps - 1)
            if isinstance(obs, DynamicObstacle):
                dist = obs.distance_at_time(position, tau)
            else:
                dist = obs.distance(position)
            closest = min(closest, max(0.0, dist - safety_margin))
        nearest = min(nearest, closest)
    if not math.isfinite(nearest):
        return max_speed
    if nearest <= stop_radius:
        return min_speed
    if nearest >= slow_down_radius:
        return max_speed
    ratio = (nearest - stop_radius) / max(1e-6, slow_down_radius - stop_radius)
    ratio = max(0.0, min(1.0, ratio))
    smooth_ratio = ratio * ratio * (3.0 - 2.0 * ratio)
    return min_speed + smooth_ratio * (max_speed - min_speed)
