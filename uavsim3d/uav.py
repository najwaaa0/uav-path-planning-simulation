"""3D UAV state and adaptive velocity control."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from .geometry import Vec3, distance, normalize, sub
from .obstacles import MovingSphereObstacle3D, Obstacle3D


@dataclass
class UAV3D:
    position: Vec3
    max_speed: float
    body_radius: float
    max_acceleration: float = 3.2
    max_deceleration: float = 4.4
    max_turn_rate: float = 0.95
    max_climb_rate: float = 3.0
    max_descent_rate: float = 3.4
    speed: float = 0.0
    heading: float = 0.0
    vertical_speed: float = 0.0

    def predict_move(self, target: Vec3, speed: float, dt: float) -> Vec3:
        proposal, _, _, _ = self._predict_state(target, speed, dt)
        return proposal

    def move_towards(self, target: Vec3, speed: float, dt: float) -> Vec3:
        self.position, self.speed, self.heading, self.vertical_speed = self._predict_state(target, speed, dt)
        return self.position

    def _predict_state(self, target: Vec3, speed_command: float, dt: float) -> tuple[Vec3, float, float, float]:
        delta = sub(target, self.position)
        remaining = distance(self.position, target)
        if remaining <= 1e-9 or dt <= 0.0:
            next_speed = max(0.0, self.speed - (self.max_deceleration * max(dt, 0.0)))
            return self.position, next_speed, self.heading, 0.0

        dx = target[0] - self.position[0]
        dy = target[1] - self.position[1]
        dz = target[2] - self.position[2]
        horizontal_distance = math.hypot(dx, dy)

        desired_heading = self.heading if horizontal_distance <= 1e-9 else math.atan2(dy, dx)
        heading_error = self._wrap(desired_heading - self.heading)
        max_yaw_step = self.max_turn_rate * dt
        next_heading = self._wrap(self.heading + self._clamp(heading_error, -max_yaw_step, max_yaw_step))

        target_speed = self._clamp(speed_command, 0.0, self.max_speed)
        if target_speed >= self.speed:
            speed_delta = min(self.max_acceleration * dt, target_speed - self.speed)
        else:
            speed_delta = -min(self.max_deceleration * dt, self.speed - target_speed)
        next_speed = self._clamp(self.speed + speed_delta, 0.0, self.max_speed)

        desired_vertical = next_speed * dz / max(remaining, 1e-9)
        desired_vertical = self._clamp(desired_vertical, -self.max_descent_rate, self.max_climb_rate)
        vertical_component = self._clamp(desired_vertical, -next_speed, next_speed)
        horizontal_speed = math.sqrt(max(0.0, (next_speed * next_speed) - (vertical_component * vertical_component)))

        if horizontal_distance <= 1e-9:
            horizontal_step = 0.0
        else:
            horizontal_step = min(horizontal_speed * dt, horizontal_distance)
        vertical_step = self._clamp(vertical_component * dt, -abs(dz), abs(dz))

        proposal = (
            self.position[0] + horizontal_step * math.cos(next_heading),
            self.position[1] + horizontal_step * math.sin(next_heading),
            self.position[2] + vertical_step,
        )
        step_length = distance(self.position, proposal)
        if step_length > remaining:
            proposal = target
        return proposal, next_speed, next_heading, vertical_component

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    @staticmethod
    def _wrap(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


def compute_velocity_3d(
    position: Vec3,
    obstacles: Iterable[Obstacle3D | MovingSphereObstacle3D],
    max_speed: float,
    min_speed: float,
    slow_down_radius: float,
    stop_radius: float,
    safety_margin: float,
    prediction_horizon: float = 0.0,
    prediction_steps: int = 1,
) -> float:
    nearest = math.inf
    steps = max(1, prediction_steps)
    for obstacle in obstacles:
        closest = math.inf
        for i in range(steps):
            tau = 0.0 if steps == 1 else prediction_horizon * i / (steps - 1)
            if isinstance(obstacle, MovingSphereObstacle3D):
                dist = obstacle.distance_at_time(position, tau)
            else:
                dist = obstacle.distance(position)
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
    smooth = ratio * ratio * (3.0 - 2.0 * ratio)
    return min_speed + smooth * (max_speed - min_speed)
