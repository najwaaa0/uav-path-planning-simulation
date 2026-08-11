"""Simple boids flocking model used for moving bird obstacles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from .geometry import Vec3, add, average, clamp, distance, normalize, scale, sub
from .obstacles import ExtrudedPolygonObstacle, MovingSphereObstacle3D


@dataclass
class Boid:
    position: Vec3
    velocity: Vec3
    radius: float = 1.0

    def as_obstacle(self, color: tuple[int, int, int], flock_name: str, idx: int) -> MovingSphereObstacle3D:
        return MovingSphereObstacle3D(
            center=self.position,
            radius=self.radius,
            velocity=self.velocity,
            color=color,
            label=f"{flock_name}-{idx}",
        )


@dataclass
class BoidFlock:
    name: str
    boids: List[Boid] = field(default_factory=list)
    color: tuple[int, int, int] = (232, 116, 39)
    min_speed: float = 3.5
    max_speed: float = 7.5
    neighbor_radius: float = 16.0
    separation_radius: float = 5.0
    obstacle_avoid_radius: float = 8.0
    alignment_weight: float = 0.7
    cohesion_weight: float = 0.45
    separation_weight: float = 1.6
    obstacle_weight: float = 1.3
    boundary_weight: float = 1.1
    cruise_weight: float = 0.35
    cruise_direction: Vec3 = (1.0, 0.0, 0.0)

    def update(
        self,
        dt: float,
        bounds: tuple[float, float, float, float, float, float],
        static_obstacles: Sequence[ExtrudedPolygonObstacle],
    ) -> None:
        next_velocities: List[Vec3] = []
        for index, boid in enumerate(self.boids):
            neighbors = [other for j, other in enumerate(self.boids) if j != index and distance(other.position, boid.position) <= self.neighbor_radius]
            alignment = (0.0, 0.0, 0.0)
            cohesion = (0.0, 0.0, 0.0)
            separation = (0.0, 0.0, 0.0)
            if neighbors:
                avg_velocity = average(other.velocity for other in neighbors)
                avg_position = average(other.position for other in neighbors)
                alignment = normalize(sub(avg_velocity, boid.velocity))
                cohesion = normalize(sub(avg_position, boid.position))
                for other in neighbors:
                    d = distance(other.position, boid.position)
                    if d <= 1e-6 or d > self.separation_radius:
                        continue
                    push = scale(normalize(sub(boid.position, other.position)), (self.separation_radius - d) / self.separation_radius)
                    separation = add(separation, push)
            obstacle_force = self._obstacle_force(boid.position, static_obstacles)
            boundary_force = self._boundary_force(boid.position, bounds)
            cruise = normalize(self.cruise_direction)
            steer = (0.0, 0.0, 0.0)
            steer = add(steer, scale(alignment, self.alignment_weight))
            steer = add(steer, scale(cohesion, self.cohesion_weight))
            steer = add(steer, scale(separation, self.separation_weight))
            steer = add(steer, scale(obstacle_force, self.obstacle_weight))
            steer = add(steer, scale(boundary_force, self.boundary_weight))
            steer = add(steer, scale(cruise, self.cruise_weight))
            velocity = add(boid.velocity, scale(steer, dt * self.max_speed))
            speed = max(1e-6, distance((0.0, 0.0, 0.0), velocity))
            target_speed = clamp(speed, self.min_speed, self.max_speed)
            next_velocities.append(scale(normalize(velocity), target_speed))

        for boid, velocity in zip(self.boids, next_velocities):
            boid.velocity = velocity
            boid.position = add(boid.position, scale(velocity, dt))
            boid.position = self._clip_position(boid.position, bounds, boid.radius)

    def obstacle_obstacles(self) -> List[MovingSphereObstacle3D]:
        return [boid.as_obstacle(self.color, self.name, idx) for idx, boid in enumerate(self.boids)]

    def _obstacle_force(self, position: Vec3, static_obstacles: Sequence[ExtrudedPolygonObstacle]) -> Vec3:
        force = (0.0, 0.0, 0.0)
        for obstacle in static_obstacles:
            d = obstacle.distance(position)
            if d >= self.obstacle_avoid_radius:
                continue
            away = normalize(sub(position, obstacle.centroid()))
            strength = (self.obstacle_avoid_radius - d) / max(1e-6, self.obstacle_avoid_radius)
            force = add(force, scale(away, strength))
        return force

    def _boundary_force(self, position: Vec3, bounds: tuple[float, float, float, float, float, float]) -> Vec3:
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        margin = 8.0
        fx = 0.0
        fy = 0.0
        fz = 0.0
        if position[0] < x_min + margin:
            fx += (x_min + margin - position[0]) / margin
        elif position[0] > x_max - margin:
            fx -= (position[0] - (x_max - margin)) / margin
        if position[1] < y_min + margin:
            fy += (y_min + margin - position[1]) / margin
        elif position[1] > y_max - margin:
            fy -= (position[1] - (y_max - margin)) / margin
        if position[2] < z_min + margin:
            fz += (z_min + margin - position[2]) / margin
        elif position[2] > z_max - margin:
            fz -= (position[2] - (z_max - margin)) / margin
        return (fx, fy, fz)

    @staticmethod
    def _clip_position(position: Vec3, bounds: tuple[float, float, float, float, float, float], radius: float) -> Vec3:
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        return (
            clamp(position[0], x_min + radius, x_max - radius),
            clamp(position[1], y_min + radius, y_max - radius),
            clamp(position[2], z_min + radius, z_max - radius),
        )
