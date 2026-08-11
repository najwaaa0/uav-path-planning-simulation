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
            RectObstacle(12, 26, 14, 34),
            RectObstacle(12, 26, 52, 82),
            RectObstacle(28, 34, 36, 78),
            RectObstacle(38, 54, 12, 30),
            RectObstacle(40, 60, 44, 64),
            RectObstacle(62, 68, 26, 70),
            RectObstacle(72, 88, 18, 38),
            RectObstacle(72, 88, 56, 84),
            CircleObstacle((50, 82), 4.0),
        ]
    )
    env.dynamic_obstacles.extend(
        [
            DynamicObstacle(
                CircleObstacle((16, 22), 1.3),
                velocity=(3.2, 1.0),
                name="bird flock eastbound",
                color="#d97706",
            ),
            DynamicObstacle(
                CircleObstacle((84, 28), 1.5),
                velocity=(-2.8, 1.6),
                name="bird flock westbound",
                color="#ea580c",
            ),
            DynamicObstacle(
                CircleObstacle((54, 18), 2.0),
                waypoints=[(54, 18), (54, 76)],
                waypoint_speed=2.4,
                name="inspection balloon",
                color="#f59e0b",
            ),
            DynamicObstacle(
                CircleObstacle((36, 72), 1.2),
                waypoints=[(36, 72), (64, 52), (82, 70), (58, 86)],
                waypoint_speed=3.1,
                name="bird flock orbit",
                color="#ef4444",
            ),
        ]
    )
    start = (8.0, 10.0)
    goal = (94.0, 92.0)
    return env, start, goal


def urban_birds(world: WorldConfig) -> tuple[Environment, Point, Point]:
    return dynamic_crossing(world)
