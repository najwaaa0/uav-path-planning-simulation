"""Continuous-space 3D RRT* planner."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import random
from typing import Dict, List, Optional, Tuple

from .environment import Environment3D
from .geometry import Vec3, distance, lerp


@dataclass
class RRTNode3D:
    point: Vec3
    parent: int | None
    cost: float = 0.0


class RRTStarPlanner3D:
    def __init__(
        self,
        step: float = 6.0,
        max_iter: int = 1200,
        goal_sample_rate: float = 0.15,
        rewire_radius: float = 16.0,
        goal_tolerance: float = 4.0,
        use_corridor_guidance: bool = True,
        enable_pso_refinement: bool = True,
        pso_particles: int = 10,
        pso_iterations: int = 12,
        pso_inertia: float = 0.58,
        pso_cognitive: float = 1.35,
        pso_social: float = 1.45,
    ) -> None:
        self.step = step
        self.max_iter = max_iter
        self.goal_sample_rate = goal_sample_rate
        self.rewire_radius = rewire_radius
        self.goal_tolerance = goal_tolerance
        self.use_corridor_guidance = use_corridor_guidance
        self.enable_pso_refinement = enable_pso_refinement
        self.pso_particles = pso_particles
        self.pso_iterations = pso_iterations
        self.pso_inertia = pso_inertia
        self.pso_cognitive = pso_cognitive
        self.pso_social = pso_social

    def plan(self, env: Environment3D, start: Vec3, goal: Vec3, safety_radius: float = 0.0) -> Optional[List[Vec3]]:
        if not env.segment_collides(start, goal, radius=safety_radius):
            return [start, goal]
        preferred_altitude = self._preferred_altitude(env, start, goal, safety_radius)
        if self.use_corridor_guidance:
            street_path = self._street_canyon_path(env, start, goal, safety_radius, preferred_altitude)
            if street_path is not None:
                return self._refine_path_with_pso(street_path, env, safety_radius, preferred_altitude)
        path = self._plan_attempt(env, start, goal, safety_radius, preferred_altitude, self.max_iter)
        if path is not None:
            return self._refine_path_with_pso(path, env, safety_radius, preferred_altitude)
        relaxed_altitude = self._relaxed_altitude(env, preferred_altitude, safety_radius)
        fallback = self._plan_attempt(env, start, goal, safety_radius, relaxed_altitude, int(self.max_iter * 1.35))
        if fallback is None:
            return None
        return self._refine_path_with_pso(fallback, env, safety_radius, relaxed_altitude)

    def _plan_attempt(
        self,
        env: Environment3D,
        start: Vec3,
        goal: Vec3,
        safety_radius: float,
        preferred_altitude: float,
        max_iter: int,
    ) -> Optional[List[Vec3]]:
        if not env.segment_collides(start, goal, radius=safety_radius):
            return [start, goal]

        nodes: List[RRTNode3D] = [RRTNode3D(start, None, 0.0)]
        best_goal_parent: int | None = None
        best_goal_cost = math.inf

        for _ in range(max_iter):
            sample = self._sample(env, start, goal, preferred_altitude)
            nearest_idx = self._nearest(nodes, sample)
            new_point = self._steer(nodes[nearest_idx].point, sample)
            if env.segment_collides(nodes[nearest_idx].point, new_point, radius=safety_radius):
                continue
            near_indices = self._near(nodes, new_point)
            parent_idx, node_cost = self._choose_parent(
                nodes,
                near_indices,
                nearest_idx,
                new_point,
                env,
                safety_radius,
                preferred_altitude,
            )
            nodes.append(RRTNode3D(new_point, parent_idx, node_cost))
            new_idx = len(nodes) - 1
            self._rewire(nodes, near_indices, new_idx, env, safety_radius, preferred_altitude)

            if distance(new_point, goal) <= self.goal_tolerance and not env.segment_collides(new_point, goal, radius=safety_radius):
                goal_cost = nodes[new_idx].cost + self._edge_cost(new_point, goal, preferred_altitude)
                if goal_cost < best_goal_cost:
                    best_goal_cost = goal_cost
                    best_goal_parent = new_idx

        if best_goal_parent is None:
            return None
        path = self._extract_path(nodes, best_goal_parent)
        path.append(goal)
        return self._shortcut_path(path, env, safety_radius)

    def _random_point(self, env: Environment3D, preferred_altitude: float | None = None) -> Vec3:
        if preferred_altitude is None or random.random() < 0.24:
            z = random.uniform(env.z_min, env.z_max)
        else:
            z_band = max(2.5, min(8.0, 0.18 * (env.z_max - env.z_min)))
            z = min(env.z_max, max(env.z_min, random.gauss(preferred_altitude, z_band)))
        return (
            random.uniform(env.x_min, env.x_max),
            random.uniform(env.y_min, env.y_max),
            z,
        )

    def _sample(self, env: Environment3D, start: Vec3, goal: Vec3, preferred_altitude: float) -> Vec3:
        if random.random() < self.goal_sample_rate:
            return goal
        guide_roll = random.random()
        if guide_roll < 0.38:
            return self._guided_sample(env, start, goal)
        if guide_roll < 0.66:
            return self._street_corridor_sample(env, start, goal, preferred_altitude)
        if guide_roll < 0.86:
            return self._building_corner_sample(env, preferred_altitude)
        return self._random_point(env, preferred_altitude)

    def _guided_sample(self, env: Environment3D, start: Vec3, goal: Vec3) -> Vec3:
        t = random.random()
        anchor = lerp(start, goal, t)
        span = max(self.step * 2.0, distance(start, goal) * 0.16)
        return (
            min(env.x_max, max(env.x_min, anchor[0] + random.uniform(-span, span))),
            min(env.y_max, max(env.y_min, anchor[1] + random.uniform(-span, span))),
            min(env.z_max, max(env.z_min, anchor[2] + random.uniform(-span * 0.35, span * 0.35))),
        )

    def _street_corridor_sample(self, env: Environment3D, start: Vec3, goal: Vec3, preferred_altitude: float) -> Vec3:
        t = random.random()
        anchor = lerp((start[0], start[1], preferred_altitude), (goal[0], goal[1], preferred_altitude), t)
        return (
            min(env.x_max, max(env.x_min, anchor[0] + random.uniform(-11.0, 11.0))),
            min(env.y_max, max(env.y_min, anchor[1] + random.uniform(-11.0, 11.0))),
            min(env.z_max, max(env.z_min, preferred_altitude + random.uniform(-3.5, 4.0))),
        )

    def _building_corner_sample(self, env: Environment3D, preferred_altitude: float) -> Vec3:
        if not env.static_obstacles:
            return self._random_point(env, preferred_altitude)
        obstacle = random.choice(env.static_obstacles)
        corner_x, corner_y = random.choice(list(obstacle.footprint))
        centroid = obstacle.centroid()
        dx = corner_x - centroid[0]
        dy = corner_y - centroid[1]
        span = math.hypot(dx, dy)
        if span <= 1e-6:
            return self._random_point(env, preferred_altitude)
        offset = random.uniform(3.5, 8.5)
        sample_x = corner_x + (dx / span) * offset
        sample_y = corner_y + (dy / span) * offset
        sample_z = preferred_altitude + random.uniform(-2.5, 3.0)
        return (
            min(env.x_max, max(env.x_min, sample_x)),
            min(env.y_max, max(env.y_min, sample_y)),
            min(env.z_max, max(env.z_min, sample_z)),
        )

    @staticmethod
    def _nearest(nodes: List[RRTNode3D], point: Vec3) -> int:
        return min(range(len(nodes)), key=lambda idx: distance(nodes[idx].point, point))

    def _near(self, nodes: List[RRTNode3D], point: Vec3) -> List[int]:
        if len(nodes) <= 1:
            return [0]
        adaptive_radius = self.rewire_radius * math.sqrt(math.log(len(nodes) + 1) / (len(nodes) + 1))
        radius = max(self.step * 2.0, adaptive_radius)
        near = [idx for idx, node in enumerate(nodes) if distance(node.point, point) <= radius]
        return near or [self._nearest(nodes, point)]

    def _choose_parent(
        self,
        nodes: List[RRTNode3D],
        near_indices: List[int],
        nearest_idx: int,
        point: Vec3,
        env: Environment3D,
        safety_radius: float,
        preferred_altitude: float,
    ) -> tuple[int, float]:
        parent_idx = nearest_idx
        best_cost = nodes[nearest_idx].cost + self._edge_cost(nodes[nearest_idx].point, point, preferred_altitude)
        for idx in near_indices:
            if env.segment_collides(nodes[idx].point, point, radius=safety_radius):
                continue
            candidate_cost = nodes[idx].cost + self._edge_cost(nodes[idx].point, point, preferred_altitude)
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                parent_idx = idx
        return parent_idx, best_cost

    def _rewire(
        self,
        nodes: List[RRTNode3D],
        near_indices: List[int],
        new_idx: int,
        env: Environment3D,
        safety_radius: float,
        preferred_altitude: float,
    ) -> None:
        for idx in near_indices:
            if idx == new_idx or idx == nodes[new_idx].parent:
                continue
            candidate_cost = nodes[new_idx].cost + self._edge_cost(nodes[new_idx].point, nodes[idx].point, preferred_altitude)
            if candidate_cost + 1e-6 >= nodes[idx].cost:
                continue
            if env.segment_collides(nodes[new_idx].point, nodes[idx].point, radius=safety_radius):
                continue
            nodes[idx].parent = new_idx
            nodes[idx].cost = candidate_cost
            self._propagate_costs(nodes, idx)

    @staticmethod
    def _propagate_costs(nodes: List[RRTNode3D], parent_idx: int) -> None:
        pending = [parent_idx]
        while pending:
            current_idx = pending.pop()
            current = nodes[current_idx]
            for idx, child in enumerate(nodes):
                if child.parent != current_idx:
                    continue
                nodes[idx].cost = current.cost + distance(current.point, child.point)
                pending.append(idx)

    def _steer(self, start: Vec3, target: Vec3) -> Vec3:
        d = distance(start, target)
        if d <= self.step:
            return target
        return lerp(start, target, self.step / max(d, 1e-9))

    @staticmethod
    def _extract_path(nodes: List[RRTNode3D], idx: int) -> List[Vec3]:
        path: List[Vec3] = []
        while idx is not None:
            path.append(nodes[idx].point)
            idx = nodes[idx].parent
        path.reverse()
        return path

    @staticmethod
    def _shortcut_path(path: List[Vec3], env: Environment3D, safety_radius: float) -> List[Vec3]:
        if len(path) <= 2:
            return path
        simplified = [path[0]]
        anchor = 0
        while anchor < len(path) - 1:
            next_idx = len(path) - 1
            while next_idx > anchor + 1:
                if not env.segment_collides(path[anchor], path[next_idx], radius=safety_radius):
                    break
                next_idx -= 1
            simplified.append(path[next_idx])
            anchor = next_idx
        return simplified

    @staticmethod
    def _preferred_altitude(env: Environment3D, start: Vec3, goal: Vec3, safety_radius: float) -> float:
        base_altitude = max(start[2], goal[2], env.z_min + safety_radius + 2.5)
        return min(env.z_max - safety_radius - 1.5, base_altitude + 4.0)

    @staticmethod
    def _relaxed_altitude(env: Environment3D, preferred_altitude: float, safety_radius: float) -> float:
        return min(env.z_max - safety_radius - 1.5, preferred_altitude + 8.0)

    @staticmethod
    def _edge_cost(a: Vec3, b: Vec3, preferred_altitude: float) -> float:
        segment_length = distance(a, b)
        mean_altitude = 0.5 * (a[2] + b[2])
        climb = abs(b[2] - a[2])
        excess_altitude = max(0.0, mean_altitude - preferred_altitude)
        return segment_length + (0.18 * climb) + (1.35 * excess_altitude)

    def _street_canyon_path(
        self,
        env: Environment3D,
        start: Vec3,
        goal: Vec3,
        safety_radius: float,
        preferred_altitude: float,
    ) -> Optional[List[Vec3]]:
        grid_step = max(4.0, self.step)
        start_grid = self._world_to_grid(env, start[0], start[1], grid_step)
        goal_grid = self._world_to_grid(env, goal[0], goal[1], grid_step)
        if not self._grid_cell_free(env, start_grid, grid_step, preferred_altitude, safety_radius):
            return None
        if not self._grid_cell_free(env, goal_grid, grid_step, preferred_altitude, safety_radius):
            return None

        open_set: List[Tuple[float, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0.0, start_grid))
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_cost: Dict[Tuple[int, int], float] = {start_grid: 0.0}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal_grid:
                points_2d = self._reconstruct_grid_path(came_from, current, env, grid_step)
                lifted_path = [start, (start[0], start[1], preferred_altitude)]
                lifted_path.extend((x, y, preferred_altitude) for x, y in points_2d[1:-1])
                lifted_path.extend([(goal[0], goal[1], preferred_altitude), goal])
                return self._shortcut_path(self._dedupe_path(lifted_path), env, safety_radius)

            for neighbor in self._grid_neighbors(current):
                if not self._grid_in_bounds(env, neighbor, grid_step):
                    continue
                if not self._grid_cell_free(env, neighbor, grid_step, preferred_altitude, safety_radius):
                    continue
                current_point = self._grid_to_world(env, current, grid_step)
                neighbor_point = self._grid_to_world(env, neighbor, grid_step)
                seg_a = (current_point[0], current_point[1], preferred_altitude)
                seg_b = (neighbor_point[0], neighbor_point[1], preferred_altitude)
                if env.segment_collides(seg_a, seg_b, radius=safety_radius):
                    continue
                move_cost = math.hypot(neighbor[0] - current[0], neighbor[1] - current[1]) * grid_step
                tentative = g_cost[current] + move_cost
                if tentative >= g_cost.get(neighbor, math.inf):
                    continue
                came_from[neighbor] = current
                g_cost[neighbor] = tentative
                heuristic = math.hypot(goal_grid[0] - neighbor[0], goal_grid[1] - neighbor[1]) * grid_step
                heapq.heappush(open_set, (tentative + heuristic, neighbor))
        return None

    @staticmethod
    def _grid_neighbors(cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        offsets = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]
        return [(cell[0] + dx, cell[1] + dy) for dx, dy in offsets]

    @staticmethod
    def _world_to_grid(env: Environment3D, x: float, y: float, grid_step: float) -> Tuple[int, int]:
        gx = int(round((x - env.x_min) / grid_step))
        gy = int(round((y - env.y_min) / grid_step))
        return gx, gy

    @staticmethod
    def _grid_to_world(env: Environment3D, cell: Tuple[int, int], grid_step: float) -> Tuple[float, float]:
        x = env.x_min + (cell[0] * grid_step)
        y = env.y_min + (cell[1] * grid_step)
        return x, y

    @staticmethod
    def _grid_in_bounds(env: Environment3D, cell: Tuple[int, int], grid_step: float) -> bool:
        x, y = RRTStarPlanner3D._grid_to_world(env, cell, grid_step)
        return env.x_min <= x <= env.x_max and env.y_min <= y <= env.y_max

    @staticmethod
    def _grid_cell_free(
        env: Environment3D,
        cell: Tuple[int, int],
        grid_step: float,
        altitude: float,
        safety_radius: float,
    ) -> bool:
        x, y = RRTStarPlanner3D._grid_to_world(env, cell, grid_step)
        return not env.collides((x, y, altitude), radius=safety_radius)

    @staticmethod
    def _reconstruct_grid_path(
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        current: Tuple[int, int],
        env: Environment3D,
        grid_step: float,
    ) -> List[Tuple[float, float]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return [RRTStarPlanner3D._grid_to_world(env, cell, grid_step) for cell in path]

    @staticmethod
    def _dedupe_path(path: List[Vec3]) -> List[Vec3]:
        if not path:
            return path
        deduped = [path[0]]
        for point in path[1:]:
            if distance(point, deduped[-1]) > 1e-6:
                deduped.append(point)
        return deduped

    def _refine_path_with_pso(
        self,
        path: List[Vec3],
        env: Environment3D,
        safety_radius: float,
        preferred_altitude: float,
    ) -> List[Vec3]:
        base_path = self._dedupe_path(path)
        if not self.enable_pso_refinement or len(base_path) <= 2 or self.pso_particles <= 0 or self.pso_iterations <= 0:
            return base_path

        interior_count = len(base_path) - 2
        if interior_count <= 0:
            return base_path

        base_vector = [coord for point in base_path[1:-1] for coord in point]
        base_cost = self._path_objective(base_path, env, safety_radius, preferred_altitude, base_path)
        global_best = list(base_vector)
        global_best_cost = base_cost
        velocity_limits = (6.0, 6.0, 2.5)
        swarm: List[dict[str, object]] = []

        for particle_idx in range(self.pso_particles):
            if particle_idx == 0:
                position = list(base_vector)
            else:
                position = self._randomized_waypoints(base_path, env, preferred_altitude)
            velocity = []
            for dim_idx in range(len(position)):
                axis_limit = velocity_limits[dim_idx % 3]
                velocity.append(random.uniform(-axis_limit, axis_limit))
            particle_path = self._vector_to_path(position, base_path)
            particle_path = self._clamp_path(particle_path, env, preferred_altitude)
            position = [coord for point in particle_path[1:-1] for coord in point]
            cost = self._path_objective(particle_path, env, safety_radius, preferred_altitude, base_path)
            swarm.append(
                {
                    "position": position,
                    "velocity": velocity,
                    "best_position": list(position),
                    "best_cost": cost,
                }
            )
            if cost < global_best_cost:
                global_best = list(position)
                global_best_cost = cost

        for _ in range(self.pso_iterations):
            for particle in swarm:
                position = particle["position"]
                velocity = particle["velocity"]
                personal_best = particle["best_position"]
                for idx in range(len(position)):
                    r1 = random.random()
                    r2 = random.random()
                    velocity[idx] = (
                        self.pso_inertia * velocity[idx]
                        + self.pso_cognitive * r1 * (personal_best[idx] - position[idx])
                        + self.pso_social * r2 * (global_best[idx] - position[idx])
                    )
                    axis_limit = velocity_limits[idx % 3]
                    velocity[idx] = max(-axis_limit, min(axis_limit, velocity[idx]))
                    position[idx] += velocity[idx]

                candidate_path = self._vector_to_path(position, base_path)
                candidate_path = self._clamp_path(candidate_path, env, preferred_altitude)
                position[:] = [coord for point in candidate_path[1:-1] for coord in point]
                cost = self._path_objective(candidate_path, env, safety_radius, preferred_altitude, base_path)
                if cost < particle["best_cost"]:
                    particle["best_cost"] = cost
                    particle["best_position"] = list(position)
                if cost < global_best_cost:
                    global_best_cost = cost
                    global_best = list(position)

        optimized = self._shortcut_path(self._vector_to_path(global_best, base_path), env, safety_radius)
        optimized_cost = self._path_objective(optimized, env, safety_radius, preferred_altitude, base_path)
        if optimized_cost + 1e-6 < base_cost:
            return optimized
        return base_path

    @staticmethod
    def _vector_to_path(vector: List[float], template_path: List[Vec3]) -> List[Vec3]:
        if len(template_path) <= 2:
            return template_path
        interior_count = len(template_path) - 2
        interior = []
        for idx in range(interior_count):
            interior.append((vector[3 * idx], vector[3 * idx + 1], vector[3 * idx + 2]))
        return [template_path[0]] + interior + [template_path[-1]]

    def _randomized_waypoints(self, path: List[Vec3], env: Environment3D, preferred_altitude: float) -> List[float]:
        randomized: List[float] = []
        z_upper = min(env.z_max, preferred_altitude + 7.0)
        for point in path[1:-1]:
            randomized.extend(
                [
                    self._clamp_axis(point[0] + random.uniform(-3.5, 3.5), env.x_min, env.x_max),
                    self._clamp_axis(point[1] + random.uniform(-3.5, 3.5), env.y_min, env.y_max),
                    self._clamp_axis(point[2] + random.uniform(-1.8, 1.8), env.z_min, z_upper),
                ]
            )
        return randomized

    def _clamp_path(self, path: List[Vec3], env: Environment3D, preferred_altitude: float) -> List[Vec3]:
        if len(path) <= 2:
            return path
        clamped = [path[0]]
        z_upper = min(env.z_max, max(preferred_altitude + 7.0, path[-1][2] + 3.0))
        for point in path[1:-1]:
            clamped.append(
                (
                    self._clamp_axis(point[0], env.x_min, env.x_max),
                    self._clamp_axis(point[1], env.y_min, env.y_max),
                    self._clamp_axis(point[2], env.z_min, z_upper),
                )
            )
        clamped.append(path[-1])
        return clamped

    @staticmethod
    def _clamp_axis(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    def _path_objective(
        self,
        path: List[Vec3],
        env: Environment3D,
        safety_radius: float,
        preferred_altitude: float,
        reference_path: List[Vec3],
    ) -> float:
        if len(path) < 2:
            return math.inf

        total = 0.0
        min_clearance = math.inf
        for idx in range(len(path) - 1):
            a = path[idx]
            b = path[idx + 1]
            if env.segment_collides(a, b, radius=safety_radius):
                return 100000.0 + (2000.0 * idx)
            total += self._edge_cost(a, b, preferred_altitude)
            min_clearance = min(min_clearance, self._segment_clearance(a, b, env, safety_radius))

        smoothness = 0.0
        for prev_point, current_point, next_point in zip(path, path[1:], path[2:]):
            v1 = (
                current_point[0] - prev_point[0],
                current_point[1] - prev_point[1],
                current_point[2] - prev_point[2],
            )
            v2 = (
                next_point[0] - current_point[0],
                next_point[1] - current_point[1],
                next_point[2] - current_point[2],
            )
            n1 = math.sqrt((v1[0] * v1[0]) + (v1[1] * v1[1]) + (v1[2] * v1[2]))
            n2 = math.sqrt((v2[0] * v2[0]) + (v2[1] * v2[1]) + (v2[2] * v2[2]))
            if n1 <= 1e-6 or n2 <= 1e-6:
                continue
            cosine = ((v1[0] * v2[0]) + (v1[1] * v2[1]) + (v1[2] * v2[2])) / (n1 * n2)
            cosine = max(-1.0, min(1.0, cosine))
            smoothness += (1.0 - cosine)

        deviation = 0.0
        for point, ref_point in zip(path[1:-1], reference_path[1:-1]):
            deviation += distance(point, ref_point)

        altitude_band = 0.0
        for point in path[1:-1]:
            altitude_band += max(0.0, abs(point[2] - preferred_altitude) - 4.0)

        clearance_penalty = 0.0 if not math.isfinite(min_clearance) else 14.0 / max(0.35, min_clearance)
        return total + (7.5 * smoothness) + (0.45 * deviation) + (2.4 * altitude_band) + clearance_penalty

    def _segment_clearance(self, a: Vec3, b: Vec3, env: Environment3D, safety_radius: float) -> float:
        min_clearance = math.inf
        for step_idx in range(5):
            tau = step_idx / 4.0
            sample = lerp(a, b, tau)
            clearance = env.distance_to_obstacles(sample, safety_margin=safety_radius)
            min_clearance = min(min_clearance, clearance)
        return min_clearance
