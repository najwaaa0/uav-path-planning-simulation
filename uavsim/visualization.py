"""Matplotlib-based 2D visualization for the simulation."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, Wedge

from .environment import Environment
from .obstacles import CircleObstacle, DynamicObstacle, RectObstacle

Point = Tuple[float, float]


class Visualizer:
    def __init__(self, env: Environment, uav_radius: float = 0.8, safety_margin: float = 0.6, max_speed: float = 5.0) -> None:
        self.env = env
        self.uav_radius = uav_radius
        self.safety_margin = safety_margin
        self.max_speed = max_speed
        self.dynamic_artists: List[object] = []
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.patch.set_facecolor("#f4efe6")
        self.ax.set_facecolor("#fbf8f2")
        self.ax.set_aspect("equal")
        self.ax.set_xlim(env.x_min, env.x_max)
        self.ax.set_ylim(env.y_min, env.y_max)
        self.ax.set_title("Dynamic UAV Path Planning")
        self.ax.grid(color="#d9d2c7", linewidth=0.7, alpha=0.45)
        self.traj_line, = self.ax.plot([], [], color="#1d4ed8", linewidth=1.8, label="trajectory")
        self.path_line, = self.ax.plot([], [], color="#0f766e", linestyle="--", linewidth=1.4, label="planned path")
        self.status_text = self.ax.text(
            0.02,
            0.98,
            "",
            transform=self.ax.transAxes,
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#d6d3d1"},
        )
        self._draw_static()
        self.fig.tight_layout()

    def _draw_static(self) -> None:
        for obs in self.env.static_obstacles:
            if isinstance(obs, CircleObstacle):
                self.ax.add_patch(Circle(obs.center, obs.radius, color="#6b7280", alpha=0.7))
            elif isinstance(obs, RectObstacle):
                self.ax.add_patch(
                    Rectangle(
                        (obs.x_min, obs.y_min),
                        obs.x_max - obs.x_min,
                        obs.y_max - obs.y_min,
                        facecolor="#475569",
                        edgecolor="#1f2937",
                        linewidth=1.0,
                        alpha=0.75,
                    )
                )
        self.ax.legend(loc="upper right")

    def update(
        self,
        uav_pose: Tuple[float, float, float],
        trajectory: List[Point],
        planned_path: Optional[List[Point]],
        fov_deg: float,
        range_max: float,
        sim_time: float = 0.0,
        current_speed: float = 0.0,
        nearest_distance: float = math.inf,
        planner_mode: str = "rrt_star",
    ) -> None:
        for artist in self.dynamic_artists:
            artist.remove()
        self.dynamic_artists.clear()

        # Draw dynamic obstacles each frame.
        for obs in self.env.dynamic_obstacles:
            self._draw_dynamic_obstacle(obs)

        # UAV marker and FOV wedge.
        speed_ratio = 0.0 if self.max_speed <= 0.0 else min(1.0, current_speed / self.max_speed)
        uav_patch = Circle((uav_pose[0], uav_pose[1]), self.uav_radius, color=plt.cm.viridis(speed_ratio), zorder=6)
        self.ax.add_patch(uav_patch)
        self.dynamic_artists.append(uav_patch)
        wedge = Wedge(
            (uav_pose[0], uav_pose[1]),
            range_max,
            math.degrees(uav_pose[2] - math.radians(fov_deg) / 2.0),
            math.degrees(uav_pose[2] + math.radians(fov_deg) / 2.0),
            alpha=0.09,
            color="#2563eb",
        )
        self.ax.add_patch(wedge)
        self.dynamic_artists.append(wedge)

        if trajectory:
            xs, ys = zip(*trajectory)
            self.traj_line.set_data(xs, ys)
        if planned_path:
            xs, ys = zip(*planned_path)
            self.path_line.set_data(xs, ys)
        else:
            self.path_line.set_data([], [])

        clearance_text = "clear" if math.isinf(nearest_distance) else f"{nearest_distance:.2f} m"
        self.ax.set_title(f"Dynamic UAV Path Planning  |  t = {sim_time:5.1f} s")
        self.status_text.set_text(
            f"planner: {planner_mode}\n"
            f"speed: {current_speed:.2f} m/s\n"
            f"nearest obstacle: {clearance_text}"
        )
        plt.pause(0.001)

    def show(self) -> None:
        plt.show()

    def _draw_dynamic_obstacle(self, obs: DynamicObstacle) -> None:
        base = obs.base
        dynamic_color = obs.color
        if len(obs.trail) > 1:
            xs, ys = zip(*obs.trail)
            trail_line, = self.ax.plot(xs, ys, color=dynamic_color, alpha=0.28, linewidth=1.0)
            self.dynamic_artists.append(trail_line)
        if isinstance(base, CircleObstacle):
            halo = Circle(
                base.center,
                base.radius + self.uav_radius + self.safety_margin,
                color=dynamic_color,
                alpha=0.10,
                zorder=2,
            )
            patch = Circle(base.center, base.radius, facecolor=dynamic_color, edgecolor="white", linewidth=1.0, alpha=0.85, zorder=4)
        elif isinstance(base, RectObstacle):
            halo = Rectangle(
                (base.x_min - self.uav_radius - self.safety_margin, base.y_min - self.uav_radius - self.safety_margin),
                (base.x_max - base.x_min) + 2.0 * (self.uav_radius + self.safety_margin),
                (base.y_max - base.y_min) + 2.0 * (self.uav_radius + self.safety_margin),
                facecolor=dynamic_color,
                edgecolor="none",
                alpha=0.10,
                zorder=2,
            )
            patch = Rectangle(
                (base.x_min, base.y_min),
                base.x_max - base.x_min,
                base.y_max - base.y_min,
                facecolor=dynamic_color,
                edgecolor="white",
                linewidth=1.0,
                alpha=0.8,
                zorder=4,
            )
        else:
            return
        self.ax.add_patch(halo)
        self.ax.add_patch(patch)
        self.dynamic_artists.extend([halo, patch])
        vx, vy = obs.velocity_vector
        speed = math.hypot(vx, vy)
        if speed > 1e-3:
            arrow = FancyArrowPatch(
                posA=obs.position,
                posB=(obs.position[0] + 0.7 * vx, obs.position[1] + 0.7 * vy),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.5,
                color=dynamic_color,
                alpha=0.9,
                zorder=5,
            )
            self.ax.add_patch(arrow)
            self.dynamic_artists.append(arrow)
