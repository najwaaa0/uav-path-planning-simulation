"""Experiment runner for 3D UAV ablations and PSO tuning."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import time
from typing import Iterable

from main3d import build_simulation
from uavsim3d.config import Planner3DConfig, Sim3DConfig, UAV3DConfig, World3DConfig
from uavsim3d.scenarios import list_scenarios


ABLATION_VARIANTS: dict[str, dict[str, bool]] = {
    "baseline_rrt_star": {
        "enable_pso_refinement": False,
        "enable_adaptive_velocity": False,
        "enable_dynamic_replanning": False,
    },
    "no_pso": {
        "enable_pso_refinement": False,
        "enable_adaptive_velocity": True,
        "enable_dynamic_replanning": True,
    },
    "no_adaptive_velocity": {
        "enable_pso_refinement": True,
        "enable_adaptive_velocity": False,
        "enable_dynamic_replanning": True,
    },
    "no_dynamic_replanning": {
        "enable_pso_refinement": True,
        "enable_adaptive_velocity": True,
        "enable_dynamic_replanning": False,
    },
    "full_system": {
        "enable_pso_refinement": True,
        "enable_adaptive_velocity": True,
        "enable_dynamic_replanning": True,
    },
}


def _parse_list(raw: str, caster) -> list:
    return [caster(item.strip()) for item in raw.split(",") if item.strip()]


def _run_trial(
    *,
    scenario: str,
    seed: int,
    planner_cfg: Planner3DConfig,
    sim_cfg: Sim3DConfig,
    uav_cfg: UAV3DConfig | None = None,
    world_cfg: World3DConfig | None = None,
) -> dict[str, object]:
    t0 = time.perf_counter()
    simulation = build_simulation(
        scenario_name=scenario,
        seed=seed,
        planner_cfg=planner_cfg,
        sim_cfg=sim_cfg,
        uav_cfg=uav_cfg,
        world_cfg=world_cfg,
    )
    metrics = simulation.run()
    wall_time = time.perf_counter() - t0
    result = dict(metrics.summary())
    result.update(
        {
            "scenario": scenario,
            "seed": seed,
            "wall_time_sec": wall_time,
            "enable_pso_refinement": planner_cfg.enable_pso_refinement,
            "enable_adaptive_velocity": sim_cfg.enable_adaptive_velocity,
            "enable_dynamic_replanning": sim_cfg.enable_dynamic_replanning,
            "pso_particles": planner_cfg.pso_particles,
            "pso_iterations": planner_cfg.pso_iterations,
            "pso_inertia": planner_cfg.pso_inertia,
            "pso_cognitive": planner_cfg.pso_cognitive,
            "pso_social": planner_cfg.pso_social,
        }
    )
    return result


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: Iterable[dict[str, object]], group_keys: list[str]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[group_key] for group_key in group_keys)
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, object]] = []
    numeric_keys = [
        "path_length",
        "min_obstacle_distance",
        "collision_rate",
        "replanning_frequency",
        "flight_time",
        "step_count",
        "avg_speed",
        "wall_time_sec",
    ]
    for key, items in grouped.items():
        summary: dict[str, object] = {group_key: key[idx] for idx, group_key in enumerate(group_keys)}
        summary["runs"] = len(items)
        summary["success_rate"] = sum(1.0 for item in items if item.get("goal_reached")) / len(items)
        for metric_key in numeric_keys:
            summary[f"mean_{metric_key}"] = sum(float(item[metric_key]) for item in items) / len(items)
        summary_rows.append(summary)
    return summary_rows


def _composite_score(row: dict[str, object]) -> float:
    clearance = max(0.35, float(row["min_obstacle_distance"]))
    penalty = 0.0
    if not bool(row["goal_reached"]):
        penalty += 12.0
    penalty += 24.0 * float(row["collision_rate"])
    return (
        float(row["path_length"])
        + (8.0 / clearance)
        + (0.5 * float(row["replanning_frequency"]))
        + (0.8 * float(row["wall_time_sec"]))
        + penalty
    )


def run_ablation(args: argparse.Namespace) -> None:
    scenarios = args.scenarios
    seeds = args.seeds
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, object]] = []
    for scenario, seed, variant_name in itertools.product(scenarios, seeds, ABLATION_VARIANTS.keys()):
        planner_cfg = Planner3DConfig()
        sim_cfg = Sim3DConfig()
        variant = ABLATION_VARIANTS[variant_name]
        planner_cfg.enable_pso_refinement = variant["enable_pso_refinement"]
        sim_cfg.enable_adaptive_velocity = variant["enable_adaptive_velocity"]
        sim_cfg.enable_dynamic_replanning = variant["enable_dynamic_replanning"]
        row = _run_trial(
            scenario=scenario,
            seed=seed,
            planner_cfg=planner_cfg,
            sim_cfg=sim_cfg,
        )
        row["variant"] = variant_name
        raw_rows.append(row)

    summary_rows = _aggregate(raw_rows, ["scenario", "variant"])
    _write_csv(out_dir / "ablation_runs.csv", raw_rows)
    _write_csv(out_dir / "ablation_summary.csv", summary_rows)
    with (out_dir / "ablation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2)
    print(f"Ablation study complete. Results saved to {out_dir}")


def run_pso_tuning(args: argparse.Namespace) -> None:
    particles = _parse_list(args.particles, int)
    iterations = _parse_list(args.iterations, int)
    inertia_values = _parse_list(args.inertia, float)
    cognitive_values = _parse_list(args.cognitive, float)
    social_values = _parse_list(args.social, float)
    seeds = args.seeds
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, object]] = []
    for combo_idx, (particles_count, iteration_count, inertia, cognitive, social) in enumerate(
        itertools.product(particles, iterations, inertia_values, cognitive_values, social_values),
        start=1,
    ):
        for seed in seeds:
            planner_cfg = Planner3DConfig(
                enable_pso_refinement=True,
                pso_particles=particles_count,
                pso_iterations=iteration_count,
                pso_inertia=inertia,
                pso_cognitive=cognitive,
                pso_social=social,
            )
            sim_cfg = Sim3DConfig()
            row = _run_trial(
                scenario=args.scenario,
                seed=seed,
                planner_cfg=planner_cfg,
                sim_cfg=sim_cfg,
            )
            row["combo_id"] = combo_idx
            row["score"] = _composite_score(row)
            raw_rows.append(row)

    summary_rows = _aggregate(raw_rows, ["combo_id", "scenario", "pso_particles", "pso_iterations", "pso_inertia", "pso_cognitive", "pso_social"])
    for summary in summary_rows:
        matching_rows = [row for row in raw_rows if int(row["combo_id"]) == int(summary["combo_id"])]
        summary["mean_score"] = sum(float(row["score"]) for row in matching_rows) / len(matching_rows)
    summary_rows.sort(key=lambda row: (float(row["mean_score"]), -float(row["success_rate"]), float(row["mean_wall_time_sec"])))

    _write_csv(out_dir / "pso_tuning_runs.csv", raw_rows)
    _write_csv(out_dir / "pso_tuning_summary.csv", summary_rows)
    with (out_dir / "pso_tuning_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2)

    print(f"PSO tuning complete. Results saved to {out_dir}")
    if summary_rows:
        best = summary_rows[0]
        print("Best configuration:")
        print(
            f"  particles={best['pso_particles']}, iterations={best['pso_iterations']}, inertia={best['pso_inertia']}, "
            f"cognitive={best['pso_cognitive']}, social={best['pso_social']}, mean_score={best['mean_score']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 3D UAV ablation studies and PSO tuning experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ablation = subparsers.add_parser("ablation", help="Run feature ablation studies across one or more scenarios")
    ablation.add_argument("--scenarios", nargs="+", choices=list_scenarios(), default=["urban_canyon", "dense_downtown"])
    ablation.add_argument("--seeds", nargs="+", type=int, default=[3, 7, 11], help="Seeds used to average stochastic planning behavior")
    ablation.add_argument("--out-dir", default="outputs/studies/ablation", help="Directory for CSV and JSON study outputs")
    ablation.set_defaults(func=run_ablation)

    tuning = subparsers.add_parser("pso-tuning", help="Run a PSO parameter sweep for one scenario")
    tuning.add_argument("--scenario", choices=list_scenarios(), default="urban_canyon")
    tuning.add_argument("--seeds", nargs="+", type=int, default=[5, 9, 13], help="Seeds used to average stochastic planning behavior")
    tuning.add_argument("--particles", default="8,10", help="Comma-separated PSO particle counts")
    tuning.add_argument("--iterations", default="10,12", help="Comma-separated PSO iteration counts")
    tuning.add_argument("--inertia", default="0.52,0.58", help="Comma-separated inertia coefficients")
    tuning.add_argument("--cognitive", default="1.25,1.35", help="Comma-separated cognitive coefficients")
    tuning.add_argument("--social", default="1.35,1.45", help="Comma-separated social coefficients")
    tuning.add_argument("--out-dir", default="outputs/studies/pso_tuning", help="Directory for CSV and JSON study outputs")
    tuning.set_defaults(func=run_pso_tuning)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
