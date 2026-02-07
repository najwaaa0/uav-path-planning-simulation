"""Central configuration defaults for the UAV simulation platform."""

from dataclasses import dataclass


@dataclass
class WorldConfig:
    x_min: float = 0.0
    x_max: float = 100.0
    y_min: float = 0.0
    y_max: float = 100.0
    grid_resolution: float = 1.0


@dataclass
class UAVConfig:
    max_speed: float = 5.0
    max_turn_rate: float = 1.2
    body_radius: float = 0.8


@dataclass
class SensorConfig:
    fov_deg: float = 120.0
    range_max: float = 25.0
    ray_count: int = 61
    noise_std: float = 0.2


@dataclass
class PlannerConfig:
    astar_allow_unknown: bool = False
    rrt_max_iter: int = 800
    rrt_step: float = 3.0
    rrt_goal_sample_rate: float = 0.15
    rrt_allow_unknown: bool = True


@dataclass
class SimConfig:
    dt: float = 0.2
    max_steps: int = 1200
    waypoint_tolerance: float = 1.5
    replan_cooldown_steps: int = 5
    speed_scale: float = 0.6


@dataclass
class CostWeights:
    length: float = 1.0
    clearance: float = 2.0
    smoothness: float = 0.5
