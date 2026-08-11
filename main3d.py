"""Entry point for the 3D UAV simulator."""

from __future__ import annotations

import argparse
import random

from uavsim3d.config import Planner3DConfig, Sim3DConfig, UAV3DConfig, World3DConfig
from uavsim3d.metrics import Metrics3D
from uavsim3d.rrt import RRTStarPlanner3D
from uavsim3d.scenarios import build_scenario, list_scenarios
from uavsim3d.simulation import Simulation3D
from uavsim3d.uav import UAV3D


def build_simulation(
    scenario_name: str = "urban_canyon",
    *,
    seed: int | None = None,
    world_cfg: World3DConfig | None = None,
    uav_cfg: UAV3DConfig | None = None,
    planner_cfg: Planner3DConfig | None = None,
    sim_cfg: Sim3DConfig | None = None,
) -> Simulation3D:
    if seed is not None:
        random.seed(seed)

    world_cfg = world_cfg or World3DConfig()
    uav_cfg = uav_cfg or UAV3DConfig()
    planner_cfg = planner_cfg or Planner3DConfig()
    sim_cfg = sim_cfg or Sim3DConfig()

    env, start, goal = build_scenario(scenario_name, world_cfg, seed=seed)
    uav = UAV3D(
        position=start,
        max_speed=uav_cfg.max_speed,
        body_radius=uav_cfg.body_radius,
        max_acceleration=uav_cfg.max_acceleration,
        max_deceleration=uav_cfg.max_deceleration,
        max_turn_rate=uav_cfg.max_turn_rate,
        max_climb_rate=uav_cfg.max_climb_rate,
        max_descent_rate=uav_cfg.max_descent_rate,
    )
    planner = RRTStarPlanner3D(
        step=planner_cfg.step,
        max_iter=planner_cfg.max_iter,
        goal_sample_rate=planner_cfg.goal_sample_rate,
        rewire_radius=planner_cfg.rewire_radius,
        goal_tolerance=planner_cfg.goal_tolerance,
        use_corridor_guidance=planner_cfg.use_corridor_guidance,
        enable_pso_refinement=planner_cfg.enable_pso_refinement,
        pso_particles=planner_cfg.pso_particles,
        pso_iterations=planner_cfg.pso_iterations,
        pso_inertia=planner_cfg.pso_inertia,
        pso_cognitive=planner_cfg.pso_cognitive,
        pso_social=planner_cfg.pso_social,
    )
    return Simulation3D(env=env, uav=uav, planner=planner, config=sim_cfg, goal=goal)


def run_headless(
    log_dir: str,
    run_name: str,
    *,
    scenario_name: str = "urban_canyon",
    seed: int | None = None,
    planner_cfg: Planner3DConfig | None = None,
    sim_cfg: Sim3DConfig | None = None,
    uav_cfg: UAV3DConfig | None = None,
) -> Metrics3D:
    simulation = build_simulation(
        scenario_name=scenario_name,
        seed=seed,
        planner_cfg=planner_cfg,
        sim_cfg=sim_cfg,
        uav_cfg=uav_cfg,
    )
    metrics = simulation.run()
    metrics.save(log_dir, run_name)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="3D UAV path planning with RRT*, boids, and extruded polygon buildings")
    parser.add_argument("--scenario", choices=list_scenarios(), default="urban_canyon", help="3D environment to run")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible planning and flock initialization")
    parser.add_argument("--no-viz", action="store_true", help="Run without the pyglet real-time renderer")
    parser.add_argument("--disable-pso", action="store_true", help="Disable PSO path refinement")
    parser.add_argument("--disable-adaptive-velocity", action="store_true", help="Use a fixed cruise speed instead of obstacle-aware velocity adaptation")
    parser.add_argument("--disable-dynamic-replanning", action="store_true", help="Disable periodic and path-blockage-driven replanning after the first path")
    parser.add_argument("--log-dir", default="outputs", help="Directory to save logs")
    parser.add_argument("--run-name", default="run3d", help="Prefix for output files")
    args = parser.parse_args()

    planner_cfg = Planner3DConfig()
    sim_cfg = Sim3DConfig()
    if args.disable_pso:
        planner_cfg.enable_pso_refinement = False
    if args.disable_adaptive_velocity:
        sim_cfg.enable_adaptive_velocity = False
    if args.disable_dynamic_replanning:
        sim_cfg.enable_dynamic_replanning = False

    if args.no_viz:
        metrics = run_headless(
            args.log_dir,
            args.run_name,
            scenario_name=args.scenario,
            seed=args.seed,
            planner_cfg=planner_cfg,
            sim_cfg=sim_cfg,
        )
        print("3D simulation summary:")
        for key, value in metrics.summary().items():
            print(f"  {key}: {value}")
        return

    from uavsim3d.visualization import Pyglet3DVisualizer

    simulation = build_simulation(
        scenario_name=args.scenario,
        seed=args.seed,
        planner_cfg=planner_cfg,
        sim_cfg=sim_cfg,
    )
    visualizer = Pyglet3DVisualizer(simulation)
    visualizer.run()
    simulation.metrics.save(args.log_dir, args.run_name)


if __name__ == "__main__":
    main()
