"""Core deterministic tests for the 2D simulation components."""

from __future__ import annotations

import random

from uavsim.environment import Environment
from uavsim.obstacles import CircleObstacle, RectObstacle
from uavsim.perception import FREE, KnownMap
from uavsim.planners.astar import AStarPlanner
from uavsim.planners.rrt import RRTPlanner


def _free_known_map(width: int = 20, height: int = 20, resolution: float = 1.0) -> KnownMap:
    known_map = KnownMap(0.0, float(width), 0.0, float(height), resolution)
    known_map.grid[:, :] = FREE
    known_map.free_grid[:, :] = FREE
    return known_map


def _path_segments_are_free(planner: RRTPlanner, known_map: KnownMap, path: list[tuple[float, float]]) -> bool:
    return all(planner._segment_free(known_map, path[i], path[i + 1]) for i in range(len(path) - 1))


def test_astar_finds_valid_path_around_obstacle() -> None:
    known_map = _free_known_map()
    for y in range(20):
        if y == 10:
            continue
        known_map.mark_occupied((10.5, y + 0.5))

    path = AStarPlanner(allow_unknown=False).plan(known_map, (1.5, 10.5), (18.5, 10.5))

    assert path is not None
    assert len(path) > 2
    assert path[0] == known_map.grid_to_world(known_map.world_to_grid((1.5, 10.5)))
    assert path[-1] == known_map.grid_to_world(known_map.world_to_grid((18.5, 10.5)))
    assert all(known_map.grid[known_map.world_to_grid(point)[1], known_map.world_to_grid(point)[0]] == FREE for point in path)


def test_rrt_star_produces_valid_path_when_route_is_possible() -> None:
    random.seed(7)
    known_map = _free_known_map(width=30, height=30)
    for y in range(30):
        if 13 <= y <= 16:
            continue
        known_map.mark_occupied((15.5, y + 0.5))

    planner = RRTPlanner(step=2.5, max_iter=600, goal_sample_rate=0.25, allow_unknown=False, rewire_radius=8.0)
    path = planner.plan(known_map, (2.0, 15.0), (28.0, 15.0))

    assert path is not None
    assert path[0] == (2.0, 15.0)
    assert path[-1] == (28.0, 15.0)
    assert _path_segments_are_free(planner, known_map, path)


def test_environment_collision_checks_detect_collision_and_clearance() -> None:
    env = Environment(0.0, 20.0, 0.0, 20.0)
    env.static_obstacles.extend(
        [
            CircleObstacle((5.0, 5.0), 2.0),
            RectObstacle(10.0, 12.0, 10.0, 12.0),
        ]
    )

    assert env.collides((5.0, 5.0))
    assert env.collides((11.0, 11.0))
    assert not env.collides((2.0, 18.0))
    assert env.segment_collides((0.0, 5.0), (8.0, 5.0))
    assert not env.segment_collides((0.0, 18.0), (8.0, 18.0))
