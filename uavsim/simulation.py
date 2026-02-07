"""Simulation loop orchestrating perception, planning, and motion."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import List, Optional, Tuple

from .config import CostWeights, SimConfig
from .environment import Environment
from .metrics import Metrics
from .perception import KnownMap, Perception
from .planners.hybrid import HybridPlanner
from .uav import UAV

Point = Tuple[float, float]


@dataclass
class Simulation:
    env: Environment
    uav: UAV
    perception: Perception
    planner: HybridPlanner
    known_map: KnownMap
    sim_config: SimConfig
    cost_weights: CostWeights

    def run(self, start: Point, goal: Point, visualize=None) -> Metrics:
        metrics = Metrics(trajectory=[start])
        path = None
        waypoint_idx = 0
        replan_cooldown = 0
        lookahead_steps = 3

        for step in range(self.sim_config.max_steps):
            self.env.update(self.sim_config.dt)

            # Perception update.
            self.perception.sense(self.env, (self.uav.state.x, self.uav.state.y, self.uav.state.heading), self.known_map)

            # Planning update.
            t0 = time.time()
            if replan_cooldown <= 0:
                new_path = self.planner.replan_if_blocked(self.known_map, self.uav.state.as_point(), goal, path)
                if new_path != path and new_path is not None:
                    metrics.record_replan()
                    path = new_path
                    waypoint_idx = 0
                    replan_cooldown = self.sim_config.replan_cooldown_steps
                    metrics.record_path_cost(self._path_cost(path))
            replan_cooldown -= 1
            metrics.record_compute_time(time.time() - t0)

            # Select next waypoint.
            if path and waypoint_idx < len(path):
                # Progress along path by tracking the nearest point and adding a small lookahead.
                current = self.uav.state.as_point()
                nearest_idx = min(range(len(path)), key=lambda i: self._distance(current, path[i]))
                waypoint_idx = max(waypoint_idx, nearest_idx)
                if self._distance(current, path[waypoint_idx]) <= self.sim_config.waypoint_tolerance:
                    waypoint_idx = min(waypoint_idx + 1, len(path) - 1)
                target_idx = min(waypoint_idx + lookahead_steps, len(path) - 1)
                waypoint = path[target_idx]
            else:
                waypoint = goal

            v_cmd, w_cmd = self.uav.control_to_waypoint(waypoint, self.uav.max_speed * self.sim_config.speed_scale)
            prev = self.uav.state.as_point()
            self.uav.step(v_cmd, w_cmd, self.sim_config.dt)
            curr = self.uav.state.as_point()
            metrics.trajectory.append(curr)
            metrics.update_path_length(prev, curr)
            metrics.update_clearance(self.env.distance_to_obstacles(curr))

            # Collision check against true environment.
            if self.env.collides(curr, radius=self.uav.body_radius):
                metrics.record_collision()
                break

            # Goal check.
            if self._distance(curr, goal) <= self.sim_config.waypoint_tolerance:
                break

            if visualize:
                visualize.update((self.uav.state.x, self.uav.state.y, self.uav.state.heading), metrics.trajectory, path, self.perception.fov_deg, self.perception.range_max)

        return metrics

    @staticmethod
    def _distance(a: Point, b: Point) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def _path_cost(self, path: List[Point]) -> float:
        if len(path) < 2:
            return 0.0
        length = 0.0
        smoothness = 0.0
        clearance = 0.0
        for i in range(len(path) - 1):
            length += self._distance(path[i], path[i + 1])
            clearance += self.env.distance_to_obstacles(path[i])
        for i in range(1, len(path) - 1):
            v1 = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
            v2 = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
            a1 = self._angle(v1)
            a2 = self._angle(v2)
            smoothness += abs(self._wrap(a2 - a1))
        avg_clearance = clearance / max(1, len(path) - 1)
        return (self.cost_weights.length * length) + (self.cost_weights.smoothness * smoothness) + (self.cost_weights.clearance / max(0.1, avg_clearance))

    @staticmethod
    def _angle(v: Point) -> float:
        return math.atan2(v[1], v[0])

    @staticmethod
    def _wrap(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
