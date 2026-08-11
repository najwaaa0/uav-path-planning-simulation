"""Main 3D simulation loop with dynamic replanning."""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Sim3DConfig
from .environment import Environment3D
from .geometry import Vec3, distance, lerp
from .metrics import Metrics3D
from .rrt import RRTStarPlanner3D
from .uav import UAV3D, compute_velocity_3d


@dataclass
class Simulation3D:
    env: Environment3D
    uav: UAV3D
    planner: RRTStarPlanner3D
    config: Sim3DConfig
    goal: Vec3
    path: list[Vec3] | None = None
    sim_time: float = 0.0
    current_speed: float = 0.0
    nearest_distance: float = float("inf")
    finished: bool = False
    collided: bool = False
    step_count: int = 0
    metrics: Metrics3D = field(default_factory=Metrics3D)

    def __post_init__(self) -> None:
        if not self.metrics.trajectory:
            self.metrics.trajectory.append(self.uav.position)
            self.metrics.update_clearance(self.env.distance_to_obstacles(self.uav.position))

    def step(self) -> bool:
        if self.finished:
            return True

        previous_path = list(self.path) if self.path else None
        should_replan = (
            self.path is None
            or (
                self.config.enable_dynamic_replanning
                and (
                    self.step_count % self.config.replan_interval_steps == 0
                    or self._path_blocked(self.path)
                )
            )
        )
        if should_replan:
            new_path = self.planner.plan(
                self.env,
                self.uav.position,
                self.goal,
                safety_radius=self.uav.body_radius + self.config.safety_margin,
            )
            if new_path is not None:
                self.path = new_path
                if previous_path != self.path:
                    self.metrics.record_replan()

        target = self.goal
        if self.path:
            while len(self.path) > 1 and distance(self.uav.position, self.path[1]) <= self.config.waypoint_tolerance:
                self.path = self.path[1:]
            target_idx = min(self.config.lookahead_points, len(self.path) - 1)
            target = self.path[target_idx]

        prediction_horizon = max(self.config.dt, self.config.velocity_prediction_horizon)
        self.nearest_distance = self.env.distance_to_obstacles(
            self.uav.position,
            time_horizon=prediction_horizon,
            samples=self.config.dynamic_obstacle_lookahead_steps,
            safety_margin=self.config.safety_margin,
        )
        if self.config.enable_adaptive_velocity:
            commanded_speed = compute_velocity_3d(
                self.uav.position,
                self.env.all_obstacles(),
                max_speed=self.uav.max_speed,
                min_speed=self.uav.max_speed * self.config.min_speed_scale,
                slow_down_radius=self.config.slow_down_radius,
                stop_radius=self.config.stop_radius,
                safety_margin=self.config.safety_margin,
                prediction_horizon=prediction_horizon,
                prediction_steps=self.config.dynamic_obstacle_lookahead_steps,
            )
        else:
            commanded_speed = self.uav.max_speed * self.config.fixed_speed_scale

        safe_speed = commanded_speed
        active_margin = self.config.safety_margin if self.env.distance_to_obstacles(self.uav.position) >= (self.uav.body_radius + self.config.safety_margin) else 0.0
        for _ in range(self.config.safety_backoff_steps):
            proposal = self.uav.predict_move(target, safe_speed, self.config.dt)
            if not self.env.segment_collides(
                self.uav.position,
                proposal,
                radius=self.uav.body_radius,
                dt=self.config.dt,
                safety_margin=active_margin,
            ):
                break
            safe_speed *= self.config.safety_backoff_ratio
        else:
            safe_speed = 0.0

        current = self.uav.move_towards(target, safe_speed, self.config.dt)
        self.current_speed = self.uav.speed
        self.env.update(self.config.dt)
        self.sim_time += self.config.dt
        self.step_count += 1
        self.metrics.append_position(current)
        self.metrics.record_speed(self.current_speed)
        self.metrics.update_clearance(self.env.distance_to_obstacles(current))

        if self.env.collides(current, radius=self.uav.body_radius):
            self.metrics.record_collision()
            self.collided = True
            self.finished = True
            self.metrics.finalize(self.sim_time, self.step_count, goal_reached=False)
            return True

        if distance(current, self.goal) <= self.config.waypoint_tolerance:
            self.finished = True
            self.metrics.finalize(self.sim_time, self.step_count, goal_reached=True)
            return True

        if len(self.metrics.trajectory) >= self.config.max_steps:
            self.finished = True
            self.metrics.finalize(self.sim_time, self.step_count, goal_reached=False)
            return True

        return False

    def run(self) -> Metrics3D:
        while not self.finished:
            self.step()
        if self.metrics.flight_time <= 0.0 and self.step_count > 0:
            self.metrics.finalize(self.sim_time, self.step_count, goal_reached=(not self.collided and distance(self.uav.position, self.goal) <= self.config.waypoint_tolerance))
        return self.metrics

    def _path_blocked(self, path: list[Vec3]) -> bool:
        if len(path) < 2:
            return True
        anchor = self.uav.position
        preview = path[1 : min(len(path), self.config.path_preview_points + 1)]
        cruise_speed = max(self.uav.max_speed * 0.55, self.uav.max_speed * self.config.min_speed_scale, 1.0)
        elapsed_time = 0.0
        for point in preview:
            segment_distance = distance(anchor, point)
            segment_time = max(self.config.dt, segment_distance / cruise_speed)
            segment_time = min(segment_time, self.config.path_block_prediction_horizon)
            if self._segment_collides_future(
                anchor,
                point,
                start_time_offset=elapsed_time,
                travel_time=segment_time,
            ):
                return True
            elapsed_time += segment_time
            if elapsed_time >= self.config.path_block_prediction_horizon:
                break
            anchor = point
        return False

    def _segment_collides_future(
        self,
        start: Vec3,
        end: Vec3,
        start_time_offset: float,
        travel_time: float,
    ) -> bool:
        segment_distance = distance(start, end)
        spatial_steps = int(segment_distance / 0.7)
        temporal_steps = int(travel_time / 0.08) if travel_time > 0.0 else 0
        steps = max(5, spatial_steps, temporal_steps)
        for i in range(steps + 1):
            tau = i / steps
            point = lerp(start, end, tau)
            time_offset = start_time_offset + (tau * travel_time)
            if self.env.collides(
                point,
                radius=self.uav.body_radius,
                time_offset=time_offset,
                safety_margin=self.config.safety_margin,
            ):
                return True
        return False
