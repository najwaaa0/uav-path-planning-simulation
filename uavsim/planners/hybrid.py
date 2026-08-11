"""Hybrid planner that tries RRT* first and falls back to A*."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from .astar import AStarPlanner
from .rrt import RRTPlanner
from ..perception import KnownMap

Point = Tuple[float, float]


class HybridPlanner:
    def __init__(self, astar: AStarPlanner, rrt: RRTPlanner) -> None:
        self.astar = astar
        self.rrt = rrt
        self.last_plan: Optional[List[Point]] = None
        self.last_mode: str = "none"

    def plan(self, known_map: KnownMap, start: Point, goal: Point) -> Optional[List[Point]]:
        path = self.rrt.plan(known_map, start, goal)
        if path:
            self.last_plan = path
            self.last_mode = "rrt_star"
            return path
        path = self.astar.plan(known_map, start, goal)
        if path:
            self.last_plan = path
            self.last_mode = "astar"
            return path
        self.last_plan = None
        self.last_mode = "none"
        return path

    def replan_if_blocked(self, known_map: KnownMap, start: Point, goal: Point, path: Optional[List[Point]]) -> Optional[List[Point]]:
        if not path or self._path_blocked(known_map, path):
            return self.plan(known_map, start, goal)
        return path

    def _path_blocked(self, known_map: KnownMap, path: List[Point]) -> bool:
        for i in range(len(path) - 1):
            if not self.rrt._segment_free(known_map, path[i], path[i + 1]):
                return True
        return False

    @staticmethod
    def path_cost(path: List[Point]) -> float:
        if len(path) < 2:
            return 0.0
        return sum(math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]) for i in range(len(path) - 1))
