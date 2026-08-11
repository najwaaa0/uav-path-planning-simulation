"""Configuration defaults for the 3D UAV simulator."""

from dataclasses import dataclass


@dataclass
class World3DConfig:
    x_min: float = 0.0
    x_max: float = 100.0
    y_min: float = 0.0
    y_max: float = 100.0
    z_min: float = 0.0
    z_max: float = 50.0


@dataclass
class UAV3DConfig:
    max_speed: float = 9.0
    body_radius: float = 1.2
    max_acceleration: float = 3.2
    max_deceleration: float = 4.4
    max_turn_rate: float = 0.95
    max_climb_rate: float = 3.0
    max_descent_rate: float = 3.4


@dataclass
class Planner3DConfig:
    step: float = 8.0
    max_iter: int = 220
    goal_sample_rate: float = 0.18
    rewire_radius: float = 18.0
    goal_tolerance: float = 5.0
    use_corridor_guidance: bool = True
    enable_pso_refinement: bool = True
    pso_particles: int = 10
    pso_iterations: int = 12
    pso_inertia: float = 0.58
    pso_cognitive: float = 1.35
    pso_social: float = 1.45


@dataclass
class Sim3DConfig:
    dt: float = 0.18
    max_steps: int = 900
    waypoint_tolerance: float = 2.5
    lookahead_points: int = 1
    replan_interval_steps: int = 24
    path_preview_points: int = 5
    path_block_prediction_horizon: float = 4.2
    enable_dynamic_replanning: bool = True
    enable_adaptive_velocity: bool = True
    fixed_speed_scale: float = 0.72
    min_speed_scale: float = 0.22
    slow_down_radius: float = 15.0
    stop_radius: float = 3.0
    safety_margin: float = 1.0
    safety_backoff_ratio: float = 0.55
    safety_backoff_steps: int = 5
    velocity_prediction_horizon: float = 1.4
    dynamic_obstacle_lookahead_steps: int = 7
