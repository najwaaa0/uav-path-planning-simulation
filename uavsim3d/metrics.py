"""Metrics for 3D simulation runs."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import List

from .geometry import Vec3, distance


@dataclass
class Metrics3D:
    path_length: float = 0.0
    min_obstacle_distance: float = math.inf
    collisions: int = 0
    replans: int = 0
    goal_reached: bool = False
    flight_time: float = 0.0
    step_count: int = 0
    speed_samples: List[float] = field(default_factory=list)
    trajectory: List[Vec3] = field(default_factory=list)

    def append_position(self, point: Vec3) -> None:
        if self.trajectory:
            self.path_length += distance(self.trajectory[-1], point)
        self.trajectory.append(point)

    def update_clearance(self, clearance: float) -> None:
        self.min_obstacle_distance = min(self.min_obstacle_distance, clearance)

    def record_collision(self) -> None:
        self.collisions += 1

    def record_replan(self) -> None:
        self.replans += 1

    def record_speed(self, speed: float) -> None:
        self.speed_samples.append(speed)

    def finalize(self, flight_time: float, step_count: int, goal_reached: bool) -> None:
        self.flight_time = flight_time
        self.step_count = step_count
        self.goal_reached = goal_reached

    def summary(self) -> dict:
        avg_speed = sum(self.speed_samples) / len(self.speed_samples) if self.speed_samples else 0.0
        return {
            "path_length": self.path_length,
            "min_obstacle_distance": self.min_obstacle_distance,
            "collision_rate": self.collisions,
            "replanning_frequency": self.replans,
            "goal_reached": self.goal_reached,
            "flight_time": self.flight_time,
            "step_count": self.step_count,
            "avg_speed": avg_speed,
        }

    def save(self, out_dir: str, run_name: str) -> None:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        with (path / f"{run_name}_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(self.summary(), handle, indent=2)
        with (path / f"{run_name}_trajectory.csv").open("w", encoding="utf-8") as handle:
            handle.write("x,y,z\n")
            for x, y, z in self.trajectory:
                handle.write(f"{x},{y},{z}\n")
