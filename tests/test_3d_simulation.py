"""Smoke tests for the 3D UAV simulation."""

from __future__ import annotations

from main3d import build_simulation
from uavsim3d.config import Planner3DConfig, Sim3DConfig


def test_basic_3d_simulation_initializes_and_runs_to_completion() -> None:
    planner_cfg = Planner3DConfig(
        max_iter=120,
        enable_pso_refinement=False,
        use_corridor_guidance=True,
    )
    sim_cfg = Sim3DConfig(
        max_steps=450,
        enable_adaptive_velocity=True,
        enable_dynamic_replanning=True,
    )

    simulation = build_simulation(
        scenario_name="urban_canyon",
        seed=1,
        planner_cfg=planner_cfg,
        sim_cfg=sim_cfg,
    )
    metrics = simulation.run()
    summary = metrics.summary()

    assert simulation.finished
    assert summary["step_count"] > 0
    assert len(metrics.trajectory) > 1
    assert summary["collision_rate"] == 0
    assert summary["goal_reached"] is True
