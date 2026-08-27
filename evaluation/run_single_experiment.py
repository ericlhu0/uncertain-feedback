"""Run one (approach, benchmark, seed) evaluation with hydra.

Single run:
    uv run python evaluation/run_single_experiment.py approach=full benchmark=smoke

Sweep (hydra multirun):
    uv run python evaluation/run_single_experiment.py -m seed=0,1,2 \
        approach=full,no_steering benchmark=abstraction_sweep
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from evaluation.approaches.base import Approach
from evaluation.benchmarks.base import Benchmark
from evaluation.episode import run_episode
from uncertain_feedback.planners.rig import build_rig
from uncertain_feedback.simulated_users import get_persona

_REPO_ROOT = Path(__file__).resolve().parent.parent


@hydra.main(version_base=None, config_name="config", config_path="conf")
def _main(cfg: DictConfig) -> None:
    # The MDM loader os.chdir()s into its submodule and never restores, which
    # would poison relative paths in every later job of a multirun.
    os.chdir(_REPO_ROOT)
    out_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    approach = hydra.utils.instantiate(cfg.approach)
    assert isinstance(approach, Approach)
    benchmark = hydra.utils.instantiate(cfg.benchmark)
    assert isinstance(benchmark, Benchmark)
    mpc_config = Path(cfg.mpc_config)
    if not mpc_config.is_absolute():
        mpc_config = _REPO_ROOT / mpc_config
    seed = int(cfg.seed)
    logging.info(
        "seed=%s approach=%s benchmark=%s mpc_config=%s",
        seed,
        approach.name,
        benchmark.name,
        mpc_config,
    )

    rig = build_rig(mpc_config, seed=seed, load_generator=approach.requires_generator)
    if cfg.sim_chooser is not None:
        rig = replace(
            rig,
            cfg=replace(
                rig.cfg,
                simulated_user=replace(
                    rig.cfg.simulated_user, chooser=str(cfg.sim_chooser)
                ),
            ),
        )
    tasks = benchmark.generate_tasks(seed, rig.cfg)
    if cfg.max_tasks is not None:
        tasks = tasks[: int(cfg.max_tasks)]

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for task_id, task in enumerate(tasks):
        user = get_persona(task.persona)
        episode_dir = out_dir / f"task_{task_id:02d}_{task.persona}_{task.verbalizer}"
        approach.reset(rig, user, task, episode_dir)
        result = run_episode(rig, user, task, approach, episode_dir)
        for row in result["rows"]:
            row.update(
                {
                    "task_id": task_id,
                    "seed": seed,
                    "approach": approach.name,
                    "benchmark": benchmark.name,
                }
            )
        rows.extend(result["rows"])
        summary = dict(result["summary"])
        summary.update({"task_id": task_id, "benchmark": benchmark.name})
        summaries.append(summary)
        pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)

    with open(out_dir / "episodes.json", "w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=2, default=str)
    episodes = pd.DataFrame(
        [
            {
                "task_id": summary["task_id"],
                "approach": summary["approach"],
                "benchmark": summary["benchmark"],
                "seed": summary["seed"],
                "persona": summary["persona"],
                "verbalizer": summary["verbalizer"],
                "feedback_events": summary["feedback_events"],
                "all_goals_resolved": summary["all_goals_resolved"],
                "all_goals_reached": summary["all_goals_reached"],
                "executed_mean_violation": summary["executed_metrics"][
                    "mean_violation"
                ],
            }
            for summary in summaries
        ]
    )
    episodes.to_csv(out_dir / "episodes.csv", index=False)
    print(episodes)
    print(f"[evaluation] wrote {out_dir / 'results.csv'}")


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
