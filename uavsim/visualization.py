"""Matplotlib-based 2D visualization for the simulation."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Wedge

from .environment import Environment
from .obstacles import CircleObstacle, RectObstacle

Point = Tuple[float, float]


class Visualizer:
    def __init__(self, env: Environment) -> None:
        self.env = env
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.ax.set_aspect("equal")
        self.ax.set_xlim(env.x_min, env.x_max)
        self.ax.set_ylim(env.y_min, env.y_max)
        self.ax.set_title("UAV Path Planning with Limited Perception")
        self.traj_line, = self.ax.plot([], [], "b-", linewidth=1.5, label="trajectory")
        self.path_line, = self.ax.plot([], [], "g--", linewidth=1.2, label="planned path")
        self._draw_static()

    def _draw_static(self) -> None:
        for obs in self.env.static_obstacles:
            if isinstance(obs, CircleObstacle):
                self.ax.add_patch(Circle(obs.center, obs.radius, color="gray", alpha=0.6))
            elif isinstance(obs, RectObstacle):
                self.ax.add_patch(
                    Rectangle((obs.x_min, obs.y_min), obs.x_max - obs.x_min, obs.y_max - obs.y_min, color="gray", alpha=0.6)
                )
        self.ax.legend(loc="upper right")

    def update(self, uav_pose: Tuple[float, float, float], trajectory: List[Point], planned_path: Optional[List[Point]], fov_deg: float, range_max: float) -> None:
        for patch in list(self.ax.patches):
            if getattr(patch, "_dynamic", False):
                patch.remove()

        # Draw dynamic obstacles each frame.
        for obs in self.env.dynamic_obstacles:
            base = obs.base
            patch = None
            if isinstance(base, CircleObstacle):
                patch = Circle(base.center, base.radius, color="tomato", alpha=0.6)
            elif isinstance(base, RectObstacle):
                patch = Rectangle((base.x_min, base.y_min), base.x_max - base.x_min, base.y_max - base.y_min, color="tomato", alpha=0.6)
            if patch:
                patch._dynamic = True
                self.ax.add_patch(patch)

        # UAV marker and FOV wedge.
        uav_patch = Circle((uav_pose[0], uav_pose[1]), 0.8, color="navy")
        uav_patch._dynamic = True
        self.ax.add_patch(uav_patch)
        wedge = Wedge((uav_pose[0], uav_pose[1]), range_max, math.degrees(uav_pose[2] - math.radians(fov_deg) / 2.0), math.degrees(uav_pose[2] + math.radians(fov_deg) / 2.0), alpha=0.08, color="blue")
        wedge._dynamic = True
        self.ax.add_patch(wedge)

        if trajectory:
            xs, ys = zip(*trajectory)
            self.traj_line.set_data(xs, ys)
        if planned_path:
            xs, ys = zip(*planned_path)
            self.path_line.set_data(xs, ys)
        else:
            self.path_line.set_data([], [])

        plt.pause(0.001)

    def show(self) -> None:
        plt.show()
