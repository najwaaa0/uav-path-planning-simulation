"""Simple 2D kinematic UAV model."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple


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

    def step(self, v_cmd: float, w_cmd: float, dt: float) -> None:
        v = max(-self.max_speed, min(self.max_speed, v_cmd))
        w = max(-self.max_turn_rate, min(self.max_turn_rate, w_cmd))
        self.state.x += v * dt * math.cos(self.state.heading)
        self.state.y += v * dt * math.sin(self.state.heading)
        self.state.heading = self._wrap(self.state.heading + w * dt)

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
