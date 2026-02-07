"""Performance metrics logging for simulation runs."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import json
from pathlib import Path
from typing import List, Tuple

Point = Tuple[float, float]


@dataclass
class Metrics:
    path_length: float = 0.0
    min_obstacle_distance: float = math.inf
    collisions: int = 0
    compute_times: List[float] = field(default_factory=list)
    replans: int = 0
    path_costs: List[float] = field(default_factory=list)
    trajectory: List[Point] = field(default_factory=list)

    def update_path_length(self, prev: Point, curr: Point) -> None:
        self.path_length += math.hypot(curr[0] - prev[0], curr[1] - prev[1])

    def update_clearance(self, distance: float) -> None:
        self.min_obstacle_distance = min(self.min_obstacle_distance, distance)

    def record_collision(self) -> None:
        self.collisions += 1

    def record_compute_time(self, t: float) -> None:
        self.compute_times.append(t)

    def record_replan(self) -> None:
        self.replans += 1

    def record_path_cost(self, cost: float) -> None:
        self.path_costs.append(cost)

    def summary(self) -> dict:
        avg_time = sum(self.compute_times) / len(self.compute_times) if self.compute_times else 0.0
        avg_cost = sum(self.path_costs) / len(self.path_costs) if self.path_costs else 0.0
        return {
            "path_length": self.path_length,
            "min_obstacle_distance": self.min_obstacle_distance,
            "collision_rate": self.collisions,
            "avg_compute_time": avg_time,
            "replanning_frequency": self.replans,
            "avg_path_cost": avg_cost,
        }

    def save(self, out_dir: str, run_name: str) -> None:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        summary = self.summary()
        with (path / f"{run_name}_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with (path / f"{run_name}_trajectory.csv").open("w", encoding="utf-8") as f:
            f.write("x,y\n")
            for x, y in self.trajectory:
                f.write(f"{x},{y}\n")
