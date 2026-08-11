"""3D scenarios with extruded buildings, moving obstacles, and flocking birds."""

from __future__ import annotations

import random
from typing import Callable

from .boids import Boid, BoidFlock
from .config import World3DConfig
from .environment import Environment3D
from .geometry import Vec3
from .obstacles import ExtrudedPolygonObstacle, MovingSphereObstacle3D

ScenarioBuilder = Callable[[World3DConfig, int | None], tuple[Environment3D, Vec3, Vec3]]


def urban_canyon(world: World3DConfig, seed: int | None = None) -> tuple[Environment3D, Vec3, Vec3]:
    env = Environment3D(world.x_min, world.x_max, world.y_min, world.y_max, world.z_min, world.z_max)
    env.static_obstacles.extend(
        [
            ExtrudedPolygonObstacle(footprint=[(12, 12), (26, 12), (26, 28), (12, 28)], z_min=0.0, z_max=27.0, label="building-a", color=(74, 85, 104)),
            ExtrudedPolygonObstacle(footprint=[(12, 42), (26, 42), (26, 58), (12, 58)], z_min=0.0, z_max=32.0, label="building-b", color=(67, 91, 118)),
            ExtrudedPolygonObstacle(footprint=[(12, 70), (28, 70), (28, 86), (12, 86)], z_min=0.0, z_max=29.0, label="building-c", color=(82, 99, 122)),
            ExtrudedPolygonObstacle(footprint=[(40, 16), (54, 16), (54, 32), (40, 32)], z_min=0.0, z_max=24.0, label="building-d", color=(88, 102, 126)),
            ExtrudedPolygonObstacle(footprint=[(40, 44), (56, 44), (58, 56), (48, 62), (40, 58)], z_min=0.0, z_max=36.0, label="building-e", color=(72, 94, 116)),
            ExtrudedPolygonObstacle(footprint=[(42, 72), (56, 72), (56, 88), (42, 88)], z_min=0.0, z_max=30.0, label="building-f", color=(63, 82, 105)),
            ExtrudedPolygonObstacle(footprint=[(70, 12), (84, 12), (84, 28), (70, 28)], z_min=0.0, z_max=25.0, label="building-g", color=(78, 96, 119)),
            ExtrudedPolygonObstacle(footprint=[(70, 44), (86, 44), (88, 54), (80, 62), (70, 58)], z_min=0.0, z_max=31.0, label="building-h", color=(68, 86, 108)),
        ]
    )
    env.dynamic_spheres.extend(
        [
            MovingSphereObstacle3D(center=(30.0, 66.0, 16.0), radius=2.2, velocity=(2.8, 0.0, 0.35), label="inspection-balloon", color=(245, 158, 11)),
        ]
    )

    rng = _rng(seed, 7)
    env.flocks.extend(
        [
            _make_flock(
                rng,
                name="eastbound-birds",
                center=(20.0, 34.0, 11.5),
                count=6,
                spread=(2.2, 4.5, 1.0),
                base_velocity=(5.4, 0.9, 0.0),
                velocity_jitter=(0.4, 0.3, 0.12),
                radius=1.15,
                color=(234, 88, 12),
                cruise_direction=(1.0, 0.15, 0.02),
            ),
            _make_flock(
                rng,
                name="crosswind-birds",
                center=(72.0, 60.0, 12.8),
                count=7,
                spread=(3.0, 3.0, 1.2),
                base_velocity=(-4.0, -2.4, 0.0),
                velocity_jitter=(0.5, 0.4, 0.12),
                radius=1.2,
                color=(220, 38, 38),
                cruise_direction=(-0.8, -0.45, 0.0),
            ),
        ]
    )

    start = (6.0, 8.0, 6.0)
    goal = (94.0, 92.0, 10.0)
    return env, start, goal


def dense_downtown(world: World3DConfig, seed: int | None = None) -> tuple[Environment3D, Vec3, Vec3]:
    env = Environment3D(world.x_min, world.x_max, world.y_min, world.y_max, world.z_min, world.z_max)
    env.static_obstacles.extend(
        [
            ExtrudedPolygonObstacle(footprint=[(10, 10), (22, 10), (22, 25), (10, 25)], z_min=0.0, z_max=24.0, label="tower-a", color=(66, 80, 96)),
            ExtrudedPolygonObstacle(footprint=[(10, 34), (22, 34), (22, 49), (10, 49)], z_min=0.0, z_max=35.0, label="tower-b", color=(77, 89, 108)),
            ExtrudedPolygonObstacle(footprint=[(10, 58), (24, 58), (24, 74), (10, 74)], z_min=0.0, z_max=30.0, label="tower-c", color=(72, 87, 103)),
            ExtrudedPolygonObstacle(footprint=[(28, 16), (38, 16), (40, 28), (30, 32), (26, 24)], z_min=0.0, z_max=33.0, label="tower-d", color=(88, 100, 120)),
            ExtrudedPolygonObstacle(footprint=[(30, 44), (42, 44), (42, 58), (30, 58)], z_min=0.0, z_max=26.0, label="tower-e", color=(94, 107, 128)),
            ExtrudedPolygonObstacle(footprint=[(28, 70), (42, 70), (42, 86), (28, 86)], z_min=0.0, z_max=38.0, label="tower-f", color=(73, 91, 112)),
            ExtrudedPolygonObstacle(footprint=[(48, 12), (62, 12), (62, 26), (48, 26)], z_min=0.0, z_max=28.0, label="tower-g", color=(84, 97, 118)),
            ExtrudedPolygonObstacle(footprint=[(48, 34), (64, 34), (64, 48), (48, 48)], z_min=0.0, z_max=24.0, label="tower-h", color=(66, 84, 105)),
            ExtrudedPolygonObstacle(footprint=[(48, 56), (64, 56), (64, 70), (48, 70)], z_min=0.0, z_max=36.0, label="tower-i", color=(76, 95, 114)),
            ExtrudedPolygonObstacle(footprint=[(70, 18), (84, 18), (84, 32), (70, 32)], z_min=0.0, z_max=26.0, label="tower-j", color=(91, 103, 124)),
            ExtrudedPolygonObstacle(footprint=[(70, 42), (86, 42), (86, 58), (70, 58)], z_min=0.0, z_max=34.0, label="tower-k", color=(72, 89, 110)),
            ExtrudedPolygonObstacle(footprint=[(72, 68), (88, 68), (88, 86), (72, 86)], z_min=0.0, z_max=29.0, label="tower-l", color=(83, 97, 116)),
        ]
    )
    env.dynamic_spheres.extend(
        [
            MovingSphereObstacle3D(center=(34.0, 60.0, 14.0), radius=2.1, velocity=(1.8, -1.2, 0.15), label="survey-drone", color=(251, 146, 60)),
            MovingSphereObstacle3D(center=(60.0, 22.0, 12.5), radius=1.9, velocity=(-1.6, 1.8, 0.2), label="helium-balloon", color=(245, 158, 11)),
        ]
    )

    rng = _rng(seed, 17)
    env.flocks.extend(
        [
            _make_flock(
                rng,
                name="delivery-birds",
                center=(18.0, 52.0, 12.0),
                count=8,
                spread=(2.8, 5.0, 1.0),
                base_velocity=(4.8, 0.6, 0.0),
                velocity_jitter=(0.5, 0.4, 0.12),
                radius=1.1,
                color=(249, 115, 22),
                cruise_direction=(1.0, 0.05, 0.0),
            ),
            _make_flock(
                rng,
                name="square-crossing",
                center=(56.0, 52.0, 13.5),
                count=9,
                spread=(4.0, 4.0, 1.2),
                base_velocity=(-3.6, 2.8, 0.0),
                velocity_jitter=(0.6, 0.5, 0.1),
                radius=1.15,
                color=(220, 38, 38),
                cruise_direction=(-0.7, 0.55, 0.0),
            ),
            _make_flock(
                rng,
                name="roofline-birds",
                center=(78.0, 62.0, 15.0),
                count=7,
                spread=(3.0, 3.5, 0.8),
                base_velocity=(-2.4, -3.0, 0.0),
                velocity_jitter=(0.4, 0.5, 0.08),
                radius=1.0,
                color=(239, 68, 68),
                cruise_direction=(-0.6, -0.75, 0.02),
            ),
        ]
    )

    start = (6.0, 8.0, 6.0)
    goal = (94.0, 90.0, 12.0)
    return env, start, goal


def crosswind_port(world: World3DConfig, seed: int | None = None) -> tuple[Environment3D, Vec3, Vec3]:
    env = Environment3D(world.x_min, world.x_max, world.y_min, world.y_max, world.z_min, world.z_max)
    env.static_obstacles.extend(
        [
            ExtrudedPolygonObstacle(footprint=[(14, 18), (30, 18), (30, 34), (14, 34)], z_min=0.0, z_max=22.0, label="hangar-a", color=(79, 94, 114)),
            ExtrudedPolygonObstacle(footprint=[(14, 44), (30, 44), (30, 62), (14, 62)], z_min=0.0, z_max=27.0, label="hangar-b", color=(69, 84, 104)),
            ExtrudedPolygonObstacle(footprint=[(16, 72), (34, 72), (34, 90), (16, 90)], z_min=0.0, z_max=23.0, label="hangar-c", color=(88, 104, 123)),
            ExtrudedPolygonObstacle(footprint=[(42, 12), (58, 12), (58, 26), (42, 26)], z_min=0.0, z_max=20.0, label="warehouse-a", color=(76, 93, 112)),
            ExtrudedPolygonObstacle(footprint=[(44, 38), (58, 38), (58, 54), (44, 54)], z_min=0.0, z_max=34.0, label="warehouse-b", color=(66, 82, 101)),
            ExtrudedPolygonObstacle(footprint=[(42, 66), (60, 66), (60, 84), (42, 84)], z_min=0.0, z_max=28.0, label="warehouse-c", color=(83, 99, 118)),
            ExtrudedPolygonObstacle(footprint=[(70, 18), (86, 18), (86, 34), (70, 34)], z_min=0.0, z_max=25.0, label="terminal-a", color=(90, 108, 126)),
            ExtrudedPolygonObstacle(footprint=[(70, 46), (88, 46), (88, 66), (70, 66)], z_min=0.0, z_max=31.0, label="terminal-b", color=(74, 91, 111)),
        ]
    )
    env.dynamic_spheres.extend(
        [
            MovingSphereObstacle3D(center=(24.0, 40.0, 11.5), radius=1.8, velocity=(2.2, 1.2, 0.0), label="cargo-drone", color=(250, 204, 21)),
            MovingSphereObstacle3D(center=(68.0, 72.0, 15.0), radius=2.4, velocity=(-2.0, -1.6, -0.1), label="inspection-balloon", color=(245, 158, 11)),
            MovingSphereObstacle3D(center=(56.0, 58.0, 13.0), radius=1.6, velocity=(0.0, -2.6, 0.18), label="tower-crane-hook", color=(251, 113, 133)),
        ]
    )

    rng = _rng(seed, 31)
    env.flocks.extend(
        [
            _make_flock(
                rng,
                name="harbor-crosswind",
                center=(32.0, 52.0, 11.8),
                count=7,
                spread=(3.5, 5.0, 1.0),
                base_velocity=(3.2, 2.8, 0.0),
                velocity_jitter=(0.5, 0.6, 0.08),
                radius=1.15,
                color=(249, 115, 22),
                cruise_direction=(0.7, 0.65, 0.02),
            ),
            _make_flock(
                rng,
                name="pier-gulls",
                center=(78.0, 40.0, 12.5),
                count=10,
                spread=(4.0, 4.5, 1.2),
                base_velocity=(-3.8, 1.2, 0.0),
                velocity_jitter=(0.5, 0.4, 0.08),
                radius=1.0,
                color=(239, 68, 68),
                cruise_direction=(-0.85, 0.28, 0.0),
            ),
        ]
    )

    start = (8.0, 12.0, 7.0)
    goal = (94.0, 88.0, 11.0)
    return env, start, goal


SCENARIOS: dict[str, ScenarioBuilder] = {
    "urban_canyon": urban_canyon,
    "dense_downtown": dense_downtown,
    "crosswind_port": crosswind_port,
}


def build_scenario(name: str, world: World3DConfig, seed: int | None = None) -> tuple[Environment3D, Vec3, Vec3]:
    try:
        builder = SCENARIOS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown 3D scenario: {name}") from exc
    return builder(world, seed)


def list_scenarios() -> list[str]:
    return sorted(SCENARIOS.keys())


def _rng(seed: int | None, default_seed: int) -> random.Random:
    return random.Random(default_seed if seed is None else seed)


def _make_flock(
    rng: random.Random,
    *,
    name: str,
    center: Vec3,
    count: int,
    spread: Vec3,
    base_velocity: Vec3,
    velocity_jitter: Vec3,
    radius: float,
    color: tuple[int, int, int],
    cruise_direction: Vec3,
) -> BoidFlock:
    boids = [
        Boid(
            position=(
                center[0] + rng.uniform(-spread[0], spread[0]),
                center[1] + rng.uniform(-spread[1], spread[1]),
                center[2] + rng.uniform(-spread[2], spread[2]),
            ),
            velocity=(
                base_velocity[0] + rng.uniform(-velocity_jitter[0], velocity_jitter[0]),
                base_velocity[1] + rng.uniform(-velocity_jitter[1], velocity_jitter[1]),
                base_velocity[2] + rng.uniform(-velocity_jitter[2], velocity_jitter[2]),
            ),
            radius=radius,
        )
        for _ in range(count)
    ]
    return BoidFlock(name=name, boids=boids, color=color, cruise_direction=cruise_direction)
