# UAV Path Planning Simulation Platform

This repository contains a Python simulation platform for UAV path planning in 2D and 3D environments. The main focus is the 3D urban navigation simulator, where a UAV plans and replans paths through extruded building obstacles, moving spherical obstacles, and Boids-based bird flocks.

The project is intended for studying path planning behavior, collision avoidance, dynamic replanning, and trajectory/metrics generation in simulated UAV missions.

## Starting point

UAVs operating in urban environments must navigate around buildings and moving obstacles while maintaining safe clearance. This project simulates that problem by combining environment modeling, collision checking, path planning, replanning, motion constraints, adaptive speed control, and metrics logging.

## Main Features

- 3D urban simulation with extruded polygon buildings
- Moving 3D spherical obstacles
- Boids-style flocking model for groups of moving bird obstacles
- 3D RRT* path planning
- Corridor-guided 3D planning using a grid-based street-canyon guide
- PSO-based path refinement for intermediate waypoints
- Dynamic replanning when paths become blocked or at periodic intervals
- Adaptive velocity control based on obstacle clearance and prediction horizon
- 3D UAV motion constraints for acceleration, deceleration, turn rate, climb rate, and descent rate
- Collision checking for points and path segments
- Headless simulation runs with JSON summary and CSV trajectory outputs
- Optional real-time 3D visualization with Pyglet
- 2D simulation package with limited field-of-view perception, partial occupancy mapping, A*, RRT*, hybrid planning, dynamic obstacles, adaptive velocity, Matplotlib visualization, and metrics logging

## Implemented Algorithms and Components

### 3D Simulator

- **RRT\***: `uavsim3d/rrt.py` grows and rewires a continuous 3D planning tree.
- **Corridor-guided RRT\***: the 3D planner first attempts a grid-based street-canyon guide before falling back to general sampling.
- **PSO path refinement**: intermediate waypoints can be optimized by particle swarm optimization when enabled.
- **Dynamic replanning**: `uavsim3d/simulation.py` replans when no path exists, on configured intervals, or when future path collision checks indicate blockage.
- **Adaptive velocity control**: `uavsim3d/uav.py` computes speed from obstacle clearance, safety margin, and a prediction horizon.
- **Boids/flocking**: `uavsim3d/boids.py` models flock motion using alignment, cohesion, separation, obstacle avoidance, boundary force, and cruise direction.
- **Collision checking**: `uavsim3d/environment.py` checks point and segment collisions against static and dynamic obstacles.

### 2D Simulator

- **A\***: `uavsim/planners/astar.py` plans on the partial occupancy grid.
- **RRT\***: `uavsim/planners/rrt.py` performs continuous-space local planning with rewiring.
- **Hybrid planning**: `uavsim/planners/hybrid.py` tries RRT* first and falls back to A*.
- **Limited field-of-view perception**: `uavsim/perception.py` updates a known occupancy map from simulated ray casting.
- **Dynamic obstacles**: `uavsim/obstacles.py` supports velocity-based and waypoint-based moving obstacles.
- **Adaptive velocity control**: `uavsim/uav.py` adjusts speed from obstacle clearance.

## System Architecture

```text
main3d.py
└── builds 3D scenario, UAV, planner, and simulation
    ├── uavsim3d/config.py        configuration dataclasses
    ├── uavsim3d/scenarios.py     urban_canyon, dense_downtown, crosswind_port
    ├── uavsim3d/environment.py   bounds, obstacles, collision queries
    ├── uavsim3d/obstacles.py     extruded buildings and moving spheres
    ├── uavsim3d/boids.py         flocking dynamic obstacles
    ├── uavsim3d/rrt.py           RRT*, corridor guidance, PSO refinement
    ├── uavsim3d/uav.py           3D UAV motion and velocity control
    ├── uavsim3d/simulation.py    planning, movement, replanning loop
    ├── uavsim3d/metrics.py       summaries and trajectory export
    └── uavsim3d/visualization.py optional Pyglet renderer
```

The 2D simulator follows the same pattern with `main.py`, `scenarios.py`, and the `uavsim/` package.

## Project Structure

```text
.
├── main.py                  # 2D simulation command-line entry point
├── main3d.py                # 3D simulation command-line entry point
├── study3d.py               # 3D ablation and PSO tuning runner
├── scenarios.py             # 2D example scenarios
├── requirements.txt
├── uavsim/                  # 2D simulation package
│   ├── config.py
│   ├── environment.py
│   ├── obstacles.py
│   ├── perception.py
│   ├── uav.py
│   ├── simulation.py
│   ├── metrics.py
│   ├── visualization.py
│   └── planners/
│       ├── astar.py
│       ├── rrt.py
│       └── hybrid.py
└── uavsim3d/                # 3D simulation package
    ├── config.py
    ├── environment.py
    ├── geometry.py
    ├── obstacles.py
    ├── boids.py
    ├── rrt.py
    ├── simulation.py
    ├── scenarios.py
    ├── metrics.py
    ├── uav.py
    └── visualization.py
```

## Technologies

- Python 3.9+
- NumPy
- Matplotlib
- Pyglet

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run the 3D Simulator

Run a headless 3D simulation:

```bash
python main3d.py --scenario urban_canyon --seed 1 --no-viz
```

Available 3D scenarios:

```bash
python main3d.py --scenario urban_canyon --no-viz
python main3d.py --scenario dense_downtown --no-viz
python main3d.py --scenario crosswind_port --no-viz
```

Run with the optional real-time 3D renderer:

```bash
python main3d.py --scenario urban_canyon
```

Disable selected 3D components for comparison:

```bash
python main3d.py --scenario urban_canyon --no-viz --disable-pso
python main3d.py --scenario urban_canyon --no-viz --disable-adaptive-velocity
python main3d.py --scenario urban_canyon --no-viz --disable-dynamic-replanning
```

Run 3D study workflows:

```bash
python study3d.py ablation --scenarios urban_canyon dense_downtown
python study3d.py pso-tuning --scenario urban_canyon
```

## How to Run the 2D Simulator

Run a headless 2D simulation:

```bash
python main.py --scenario dynamic --no-viz
```

Available 2D scenarios:

```bash
python main.py --scenario static --no-viz
python main.py --scenario dynamic --no-viz
python main.py --scenario urban --no-viz
```

Run with Matplotlib visualization:

```bash
python main.py --scenario dynamic
```

## Example Output

Headless runs print a summary and save files under `outputs/` by default.

3D summary fields include:

- `path_length`
- `min_obstacle_distance`
- `collision_rate`
- `replanning_frequency`
- `goal_reached`
- `flight_time`
- `step_count`
- `avg_speed`

2D summary fields include:

- `path_length`
- `min_obstacle_distance`
- `collision_rate`
- `avg_compute_time`
- `replanning_frequency`
- `avg_path_cost`

Each run also writes a trajectory CSV:

```text
outputs/<run_name>_summary.json
outputs/<run_name>_trajectory.csv
```

## Reproducibility Notes

- Use `--seed` with `main3d.py` for reproducible 3D scenario initialization and stochastic planning behavior.
- Headless mode is recommended for automated runs and environments without display support.
- Generated logs, figures, caches, local virtual environments, and thesis draft files are intentionally excluded from version control.

## Limitations

- This is a simulation platform, not a real UAV flight controller.
- Obstacle sensing, dynamics, and collision checking are simplified for thesis simulation purposes.
- The 3D renderer uses software-projected geometry through Pyglet rather than a physics engine or full 3D game engine.
- Planner performance and success can vary with scenario, seed, and configuration because sampling-based planning is stochastic.
- The 2D partial map and perception model are simplified approximations of UAV sensing.

## Future Improvements

- Add automated tests for planners, geometry helpers, metrics, and scenario builders.
- Add benchmark scripts with fixed seeds and documented expected outputs.
- Add optional saved screenshots or demo media generated from the existing simulations.
- Add more scenario configuration through external files.
- Compare planner variants with clearer experiment reports.

## Author

Najwa Aouaj
