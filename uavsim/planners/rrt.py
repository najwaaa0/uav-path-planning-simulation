"""Continuous-space RRT* local planner."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import List, Optional, Tuple

from ..perception import OCCUPIED, UNKNOWN, KnownMap

Point = Tuple[float, float]


@dataclass
class RRTNode:
    point: Point
    parent: int | None
    cost: float = 0.0


class RRTPlanner:
    def __init__(
        self,
        step: float = 3.0,
        max_iter: int = 800,
        goal_sample_rate: float = 0.1,
        allow_unknown: bool = True,
        rewire_radius: float = 10.0,
    ) -> None:
        self.step = step
        self.max_iter = max_iter
        self.goal_sample_rate = goal_sample_rate
        self.allow_unknown = allow_unknown
        self.rewire_radius = rewire_radius

    def plan(self, known_map: KnownMap, start: Point, goal: Point) -> Optional[List[Point]]:
        if self._segment_free(known_map, start, goal):
            return [start, goal]

        nodes: List[RRTNode] = [RRTNode(start, None, 0.0)]
        best_goal_parent: int | None = None
        best_goal_cost = math.inf
        for _ in range(self.max_iter):
            sample = goal if random.random() < self.goal_sample_rate else self._random_point(known_map)
            idx = self._nearest(nodes, sample)
            new_pt = self._steer(nodes[idx].point, sample)
            if not self._segment_free(known_map, nodes[idx].point, new_pt):
                continue
            near_indices = self._near(nodes, new_pt)
            parent_idx, node_cost = self._choose_parent(nodes, near_indices, idx, new_pt, known_map)
            nodes.append(RRTNode(new_pt, parent_idx, node_cost))
            new_idx = len(nodes) - 1
            self._rewire(nodes, near_indices, new_idx, known_map)
            if math.hypot(new_pt[0] - goal[0], new_pt[1] - goal[1]) <= self.step:
                if self._segment_free(known_map, new_pt, goal):
                    goal_cost = nodes[new_idx].cost + self._distance(new_pt, goal)
                    if goal_cost < best_goal_cost:
                        best_goal_cost = goal_cost
                        best_goal_parent = new_idx
        if best_goal_parent is None:
            return None
        path = self._extract_path(nodes, best_goal_parent)
        path.append(goal)
        return path

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

    def _near(self, nodes: List[RRTNode], p: Point) -> List[int]:
        if len(nodes) <= 1:
            return [0]
        adaptive_radius = self.rewire_radius * math.sqrt(math.log(len(nodes) + 1) / (len(nodes) + 1))
        radius = max(self.step * 2.0, adaptive_radius)
        near = [i for i, node in enumerate(nodes) if self._distance(node.point, p) <= radius]
        return near or [self._nearest(nodes, p)]

    def _choose_parent(
        self,
        nodes: List[RRTNode],
        near_indices: List[int],
        nearest_idx: int,
        new_pt: Point,
        known_map: KnownMap,
    ) -> tuple[int, float]:
        parent_idx = nearest_idx
        best_cost = nodes[nearest_idx].cost + self._distance(nodes[nearest_idx].point, new_pt)
        for idx in near_indices:
            if not self._segment_free(known_map, nodes[idx].point, new_pt):
                continue
            candidate_cost = nodes[idx].cost + self._distance(nodes[idx].point, new_pt)
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                parent_idx = idx
        return parent_idx, best_cost

    def _rewire(self, nodes: List[RRTNode], near_indices: List[int], new_idx: int, known_map: KnownMap) -> None:
        new_node = nodes[new_idx]
        for idx in near_indices:
            if idx == new_idx or idx == new_node.parent:
                continue
            candidate_cost = new_node.cost + self._distance(new_node.point, nodes[idx].point)
            if candidate_cost + 1e-6 >= nodes[idx].cost:
                continue
            if not self._segment_free(known_map, new_node.point, nodes[idx].point):
                continue
            nodes[idx].parent = new_idx
            nodes[idx].cost = candidate_cost
            self._propagate_costs(nodes, idx)

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

    def _propagate_costs(self, nodes: List[RRTNode], parent_idx: int) -> None:
        pending = [parent_idx]
        while pending:
            current_idx = pending.pop()
            current = nodes[current_idx]
            for child_idx, child in enumerate(nodes):
                if child.parent != current_idx:
                    continue
                nodes[child_idx].cost = current.cost + self._distance(current.point, child.point)
                pending.append(child_idx)

    @staticmethod
    def _distance(a: Point, b: Point) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _extract_path(nodes: List[RRTNode], idx: int) -> List[Point]:
        path = []
        while idx is not None:
            node = nodes[idx]
            path.append(node.point)
            idx = node.parent
        path.reverse()
        return path
