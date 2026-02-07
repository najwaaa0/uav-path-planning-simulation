"""Example scenarios with static and dynamic obstacles."""

from __future__ import annotations

from typing import Tuple

from uavsim.config import WorldConfig
from uavsim.environment import Environment
from uavsim.obstacles import CircleObstacle, RectObstacle, DynamicObstacle

Point = Tuple[float, float]


def static_corridor(world: WorldConfig) -> tuple[Environment, Point, Point]:
    env = Environment(world.x_min, world.x_max, world.y_min, world.y_max)
    env.static_obstacles.extend(
        [
            RectObstacle(20, 80, 40, 45),
            RectObstacle(20, 80, 55, 60),
            CircleObstacle((30, 25), 4),
            CircleObstacle((70, 75), 4),
        ]
    )
    start = (10.0, 10.0)
    goal = (90.0, 90.0)
    return env, start, goal


def dynamic_crossing(world: WorldConfig) -> tuple[Environment, Point, Point]:
    env = Environment(world.x_min, world.x_max, world.y_min, world.y_max)
    env.static_obstacles.extend(
        [
            RectObstacle(10, 15, 10, 90),
            RectObstacle(85, 90, 10, 90),
            RectObstacle(30, 70, 10, 15),
            RectObstacle(30, 70, 85, 90),
        ]
    )
    moving_circle = DynamicObstacle(CircleObstacle((20, 50), 3.5), velocity=(2.0, 0.0))
    moving_rect = DynamicObstacle(RectObstacle(50, 55, 20, 30), waypoints=[(50, 20), (50, 70)])
    env.dynamic_obstacles.extend([moving_circle, moving_rect])
    start = (20.0, 20.0)
    goal = (80.0, 80.0)
    return env, start, goal
