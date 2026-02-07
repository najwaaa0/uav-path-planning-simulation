# UAV Path Planning Simulation (2D, Limited Perception)

This repository provides a modular 2D UAV simulation platform for evaluating path planning algorithms under limited perception.

## Features
- 2D continuous world with static and dynamic obstacles
- UAV kinematic model (state: `x, y, heading`)
- Limited FOV and range perception with a partial known map
- Hybrid planning (A* global + RRT local) with dynamic replanning
- Multi-objective cost considerations (length, clearance, smoothness)
- Metrics logging and visualization

## Folder Structure
- `uavsim/`
  - `environment.py`: world bounds and obstacles
  - `obstacles.py`: circle/rectangle + dynamic obstacle wrappers
  - `uav.py`: kinematic model
  - `perception.py`: limited FOV sensing + partial map
  - `planners/astar.py`: grid-based A*
  - `planners/rrt.py`: continuous RRT
  - `planners/hybrid.py`: planner interface + replanning
  - `simulation.py`: main loop (sense → plan → move)
  - `metrics.py`: performance logging
  - `visualization.py`: matplotlib 2D display
- `scenarios.py`: example scenarios
- `main.py`: run experiments

## Design Notes (for Thesis Writing)
- **Perception-first planning**: the planner only sees the partial known map. Unknown space is treated conservatively by A* and optionally by RRT to model limited sensing.
- **Hybrid logic**: A* provides an efficient global route on the known grid; when the route is blocked by newly detected obstacles, RRT is triggered to recover in continuous space.
- **Dynamic obstacles**: simple velocity or waypoint models create non-deterministic disruptions that require replanning.
- **Metrics**: the simulation logs path length, clearance, collisions, computation time, and replanning frequency for comparative studies.
- **Cost function**: length, clearance (inverse), and smoothness (heading change) are combined into a single score for analysis.

## Run
```bash
python3 main.py --scenario static
python3 main.py --scenario dynamic
```

Use `--no-viz` to run without matplotlib visualization.
Logs are saved to `outputs/` by default; use `--log-dir` and `--run-name` to customize.

## Requirements
- Python 3.9+
- `numpy`
- `matplotlib`

## Extension Ideas
- Add additional planners behind the unified interface
- Implement risk-aware cost functions
- Add more dynamic obstacle models and behaviors
