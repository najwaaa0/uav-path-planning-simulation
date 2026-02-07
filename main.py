"""Entry point to run UAV path planning simulations."""

from __future__ import annotations

import argparse

from uavsim.config import CostWeights, PlannerConfig, SensorConfig, SimConfig, UAVConfig, WorldConfig
from uavsim.environment import Environment
from uavsim.metrics import Metrics
from uavsim.perception import KnownMap, Perception
from uavsim.planners.astar import AStarPlanner
from uavsim.planners.rrt import RRTPlanner
from uavsim.planners.hybrid import HybridPlanner
from uavsim.simulation import Simulation
from uavsim.uav import UAV, UAVState
from uavsim.visualization import Visualizer

import scenarios


def build_sim(env: Environment, start, goal) -> tuple[Simulation, Visualizer | None]:
    world_cfg = WorldConfig(x_min=env.x_min, x_max=env.x_max, y_min=env.y_min, y_max=env.y_max, grid_resolution=1.0)
    uav_cfg = UAVConfig()
    sensor_cfg = SensorConfig()
    planner_cfg = PlannerConfig()
    sim_cfg = SimConfig()
    cost_cfg = CostWeights()

    uav = UAV(UAVState(start[0], start[1], 0.0), uav_cfg.max_speed, uav_cfg.max_turn_rate, uav_cfg.body_radius)
    known_map = KnownMap(world_cfg.x_min, world_cfg.x_max, world_cfg.y_min, world_cfg.y_max, world_cfg.grid_resolution)
    perception = Perception(sensor_cfg.fov_deg, sensor_cfg.range_max, sensor_cfg.ray_count, sensor_cfg.noise_std)
    astar = AStarPlanner(allow_unknown=planner_cfg.astar_allow_unknown)
    rrt = RRTPlanner(step=planner_cfg.rrt_step, max_iter=planner_cfg.rrt_max_iter, goal_sample_rate=planner_cfg.rrt_goal_sample_rate, allow_unknown=planner_cfg.rrt_allow_unknown)
    hybrid = HybridPlanner(astar, rrt)

    sim = Simulation(env, uav, perception, hybrid, known_map, sim_cfg, cost_cfg)
    viz = Visualizer(env)
    return sim, viz


def run_scenario(name: str, visualize: bool = True) -> Metrics:
    world_cfg = WorldConfig()
    if name == "static":
        env, start, goal = scenarios.static_corridor(world_cfg)
    elif name == "dynamic":
        env, start, goal = scenarios.dynamic_crossing(world_cfg)
    else:
        raise ValueError(f"Unknown scenario: {name}")

    sim, viz = build_sim(env, start, goal)
    metrics = sim.run(start, goal, visualize=viz if visualize else None)
    if visualize:
        viz.show()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="2D UAV path planning simulation with limited perception")
    parser.add_argument("--scenario", choices=["static", "dynamic"], default="static")
    parser.add_argument("--no-viz", action="store_true", help="Run without visualization")
    parser.add_argument("--log-dir", default="outputs", help="Directory to save metrics logs")
    parser.add_argument("--run-name", default="run", help="Prefix for log files")
    args = parser.parse_args()

    metrics = run_scenario(args.scenario, visualize=not args.no_viz)
    metrics.save(args.log_dir, f"{args.run_name}_{args.scenario}")
    print("Simulation summary:")
    for k, v in metrics.summary().items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
