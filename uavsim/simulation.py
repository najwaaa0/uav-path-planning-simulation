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
from .uav import UAV, compute_velocity

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
        lookahead_steps = 3
        sim_time = 0.0
        stall_steps = 0
        metrics.update_clearance(self.env.distance_to_obstacles(start))

        for step in range(self.sim_config.max_steps):
            # Perception update.
            self.perception.sense(self.env, (self.uav.state.x, self.uav.state.y, self.uav.state.heading), self.known_map)

            # Planning update.
            t0 = time.time()
            if self.sim_config.replan_every_step:
                new_path = self.planner.plan(self.known_map, self.uav.state.as_point(), goal)
            else:
                new_path = self.planner.replan_if_blocked(self.known_map, self.uav.state.as_point(), goal, path)
            if new_path != path and new_path is not None:
                metrics.record_replan()
                path = new_path
                waypoint_idx = 0
                metrics.record_path_cost(self._path_cost(path))
            metrics.record_compute_time(time.time() - t0)

            # Select next waypoint.
            if path and waypoint_idx < len(path):
                # Progress along path by tracking the nearest point and adding a small lookahead.
                current = self.uav.state.as_point()
                nearest_idx = min(range(len(path)), key=lambda i: self._distance(current, path[i]))
                waypoint_idx = max(waypoint_idx, nearest_idx)
                if self._distance(current, path[waypoint_idx]) <= self.sim_config.waypoint_tolerance:
                    waypoint_idx = min(waypoint_idx + 1, len(path) - 1)
                waypoint = self._select_tracking_waypoint(current, path, waypoint_idx, lookahead_steps)
            else:
                waypoint = goal

            max_speed = self.uav.max_speed * self.sim_config.speed_scale
            min_speed = self.uav.max_speed * self.sim_config.min_speed_scale
            prediction_horizon = max(self.sim_config.dt, self.sim_config.velocity_prediction_horizon)
            nearest_distance = self.env.distance_to_obstacles(
                self.uav.state.as_point(),
                time_horizon=prediction_horizon,
                samples=self.sim_config.dynamic_obstacle_lookahead_steps,
                safety_margin=self.sim_config.safety_margin,
            )
            speed = compute_velocity(
                self.uav.state.as_point(),
                self.env.all_obstacles(),
                max_speed=max_speed,
                min_speed=min_speed,
                slow_down_radius=self.sim_config.slow_down_radius,
                stop_radius=self.sim_config.stop_radius,
                safety_margin=self.sim_config.safety_margin,
                prediction_horizon=prediction_horizon,
                prediction_steps=self.sim_config.dynamic_obstacle_lookahead_steps,
            )
            v_cmd, w_cmd = self.uav.control_to_waypoint(waypoint, speed)
            prev = self.uav.state.as_point()
            v_safe = v_cmd
            clearance_with_margin = self.env.distance_to_obstacles(prev)
            active_margin = self.sim_config.safety_margin if clearance_with_margin >= (self.uav.body_radius + self.sim_config.safety_margin) else 0.0
            v_safe = self._apply_safety_backoff(prev, v_safe, w_cmd, active_margin)
            if v_safe == 0.0 and active_margin > 0.0:
                # If we are trapped on the inflated boundary, retry with body-only clearance.
                v_safe = self._apply_safety_backoff(prev, v_cmd, w_cmd, 0.0)
            if v_safe == 0.0:
                v_safe, w_cmd = self._find_recovery_motion(prev, speed, stall_steps)
            self.uav.step(v_safe, w_cmd, self.sim_config.dt)
            self.env.update(self.sim_config.dt)
            sim_time += self.sim_config.dt
            curr = self.uav.state.as_point()
            if self._distance(prev, curr) < 1e-4:
                stall_steps += 1
            else:
                stall_steps = 0
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
                visualize.update(
                    (self.uav.state.x, self.uav.state.y, self.uav.state.heading),
                    metrics.trajectory,
                    path,
                    self.perception.fov_deg,
                    self.perception.range_max,
                    sim_time=sim_time,
                    current_speed=abs(v_safe),
                    nearest_distance=nearest_distance,
                    planner_mode=self.planner.last_mode,
                )

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

    def _select_tracking_waypoint(self, current: Point, path: List[Point], waypoint_idx: int, lookahead_steps: int) -> Point:
        target_idx = min(waypoint_idx + lookahead_steps, len(path) - 1)
        for idx in range(target_idx, waypoint_idx - 1, -1):
            candidate = path[idx]
            if self._segment_safe_for_tracking(current, candidate):
                return candidate
        return path[waypoint_idx]

    def _segment_safe_for_tracking(self, start: Point, end: Point) -> bool:
        return not self.env.segment_collides(
            start,
            end,
            radius=self.uav.body_radius,
            safety_margin=self.sim_config.safety_margin,
        )

    def _apply_safety_backoff(self, start: Point, speed: float, turn_rate: float, safety_margin: float) -> float:
        v_safe = speed
        for _ in range(self.sim_config.safety_backoff_steps):
            proposal_x, proposal_y, _ = self.uav.predict_step(v_safe, turn_rate, self.sim_config.dt)
            proposal = (proposal_x, proposal_y)
            if not self.env.segment_collides(
                start,
                proposal,
                radius=self.uav.body_radius,
                dt=self.sim_config.dt,
                safety_margin=safety_margin,
            ):
                return v_safe
            v_safe *= self.sim_config.safety_backoff_ratio
        return 0.0

    def _find_recovery_motion(self, position: Point, nominal_speed: float, stall_steps: int) -> Tuple[float, float]:
        crawl_speed = max(0.25, min(nominal_speed, self.uav.max_speed * 0.35))
        heading_candidates = [0.0, 0.35, -0.35, 0.7, -0.7, 1.05, -1.05, math.pi, math.pi / 2.0, -math.pi / 2.0]
        if stall_steps >= 4:
            heading_candidates.extend([1.4, -1.4, 2.2, -2.2])

        for delta in heading_candidates:
            desired_heading = self._wrap(self.uav.state.heading + delta)
            heading_error = self._wrap(desired_heading - self.uav.state.heading)
            w_cmd = 2.0 * heading_error
            test_speed = crawl_speed if abs(delta) < (math.pi * 0.75) else crawl_speed * 0.6
            proposal_x, proposal_y, _ = self.uav.predict_step(test_speed, w_cmd, self.sim_config.dt)
            proposal = (proposal_x, proposal_y)
            if not self.env.segment_collides(
                position,
                proposal,
                radius=self.uav.body_radius,
                dt=self.sim_config.dt,
                safety_margin=0.0,
            ):
                return test_speed, w_cmd

        # Last resort: rotate in place to search for a better heading on the next replanning cycle.
        return 0.0, self.uav.max_turn_rate

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
