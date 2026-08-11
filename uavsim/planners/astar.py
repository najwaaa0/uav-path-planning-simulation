"""Grid-based A* planner operating on the partial known map."""

from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Tuple

from ..perception import OCCUPIED, UNKNOWN, KnownMap

Grid = Tuple[int, int]


class AStarPlanner:
    def __init__(self, allow_unknown: bool = False) -> None:
        self.allow_unknown = allow_unknown

    def plan(self, known_map: KnownMap, start: Tuple[float, float], goal: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        start_g = known_map.world_to_grid(start)
        goal_g = known_map.world_to_grid(goal)
        if not known_map.in_bounds(start_g) or not known_map.in_bounds(goal_g):
            return None

        open_set: List[Tuple[float, Grid]] = []
        heapq.heappush(open_set, (0.0, start_g))
        came_from: Dict[Grid, Grid] = {}
        g_cost: Dict[Grid, float] = {start_g: 0.0}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal_g:
                return self._reconstruct_path(known_map, came_from, current)

            for nb in self._neighbors(known_map, current):
                if not self._cell_traversable(known_map, nb):
                    continue
                tentative = g_cost[current] + self._move_cost(current, nb, known_map)
                if nb not in g_cost or tentative < g_cost[nb]:
                    came_from[nb] = current
                    g_cost[nb] = tentative
                    f = tentative + self._heuristic(nb, goal_g)
                    heapq.heappush(open_set, (f, nb))
        return None

    def _neighbors(self, known_map: KnownMap, g: Grid) -> List[Grid]:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        result = []
        for dx, dy in offsets:
            nb = (g[0] + dx, g[1] + dy)
            if known_map.in_bounds(nb):
                result.append(nb)
        return result

    def _cell_traversable(self, known_map: KnownMap, g: Grid) -> bool:
        val = known_map.grid[g[1], g[0]]
        if val == OCCUPIED:
            return False
        if val == UNKNOWN:
            return self.allow_unknown
        return True

    def _move_cost(self, a: Grid, b: Grid, known_map: KnownMap) -> float:
        base = math.hypot(b[0] - a[0], b[1] - a[1])
        # Clearance penalty based on nearby occupied cells (simple local density).
        penalty = 0.0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                gx = b[0] + dx
                gy = b[1] + dy
                if 0 <= gx < known_map.width and 0 <= gy < known_map.height:
                    if known_map.grid[gy, gx] == OCCUPIED:
                        penalty += 0.5
        return base + penalty

    @staticmethod
    def _heuristic(a: Grid, b: Grid) -> float:
        return math.hypot(b[0] - a[0], b[1] - a[1])

    @staticmethod
    def _reconstruct_path(known_map: KnownMap, came_from: Dict[Grid, Grid], current: Grid) -> List[Tuple[float, float]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return [known_map.grid_to_world(g) for g in path]
