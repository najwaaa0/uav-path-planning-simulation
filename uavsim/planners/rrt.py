"""Continuous-space RRT local planner."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import List, Optional, Tuple

from ..perception import FREE, OCCUPIED, UNKNOWN, KnownMap

Point = Tuple[float, float]


@dataclass
class RRTNode:
    point: Point
    parent: int | None


class RRTPlanner:
    def __init__(self, step: float = 3.0, max_iter: int = 800, goal_sample_rate: float = 0.1, allow_unknown: bool = True) -> None:
        self.step = step
        self.max_iter = max_iter
        self.goal_sample_rate = goal_sample_rate
        self.allow_unknown = allow_unknown

    def plan(self, known_map: KnownMap, start: Point, goal: Point) -> Optional[List[Point]]:
        nodes: List[RRTNode] = [RRTNode(start, None)]
        for _ in range(self.max_iter):
            sample = goal if random.random() < self.goal_sample_rate else self._random_point(known_map)
            idx = self._nearest(nodes, sample)
            new_pt = self._steer(nodes[idx].point, sample)
            if not self._segment_free(known_map, nodes[idx].point, new_pt):
                continue
            nodes.append(RRTNode(new_pt, idx))
            if math.hypot(new_pt[0] - goal[0], new_pt[1] - goal[1]) <= self.step:
                if self._segment_free(known_map, new_pt, goal):
                    nodes.append(RRTNode(goal, len(nodes) - 1))
                    return self._extract_path(nodes)
        return None

    def _random_point(self, known_map: KnownMap) -> Point:
        return (
            random.uniform(known_map.x_min, known_map.x_max),
            random.uniform(known_map.y_min, known_map.y_max),
        )

    @staticmethod
    def _nearest(nodes: List[RRTNode], p: Point) -> int:
        best = 0
        best_d = math.inf
        for i, node in enumerate(nodes):
            d = math.hypot(node.point[0] - p[0], node.point[1] - p[1])
            if d < best_d:
                best_d = d
                best = i
        return best

    def _steer(self, from_pt: Point, to_pt: Point) -> Point:
        dx = to_pt[0] - from_pt[0]
        dy = to_pt[1] - from_pt[1]
        dist = math.hypot(dx, dy)
        if dist <= self.step:
            return to_pt
        return (from_pt[0] + self.step * dx / dist, from_pt[1] + self.step * dy / dist)

    def _segment_free(self, known_map: KnownMap, a: Point, b: Point) -> bool:
        steps = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / (known_map.resolution / 2)))
        for i in range(steps + 1):
            t = i / steps
            px = a[0] + t * (b[0] - a[0])
            py = a[1] + t * (b[1] - a[1])
            if not (known_map.x_min <= px <= known_map.x_max and known_map.y_min <= py <= known_map.y_max):
                return False
            gx, gy = known_map.world_to_grid((px, py))
            if not known_map.in_bounds((gx, gy)):
                return False
            cell = known_map.grid[gy, gx]
            if cell == OCCUPIED:
                return False
            if cell == UNKNOWN and not self.allow_unknown:
                return False
        return True

    @staticmethod
    def _extract_path(nodes: List[RRTNode]) -> List[Point]:
        path = []
        idx = len(nodes) - 1
        while idx is not None:
            node = nodes[idx]
            path.append(node.point)
            idx = node.parent
        path.reverse()
        return path
