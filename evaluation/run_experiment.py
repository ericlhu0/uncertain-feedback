"""Run one (approach, benchmark, seed) evaluation with hydra.

Single run:
    uv run python evaluation/run_experiment.py approach=oracle_no_learning benchmark=smoke

Sweep (hydra multirun):
    uv run python evaluation/run_experiment.py -m seed=0,1,2 \\
        approach=oracle_language,nominal_language,mdm_language,oracle_no_learning \\
        benchmark=cost_learning \\
        mpc_config=src/uncertain_feedback/planners/mpc/configs/mdm_llm_transfer.yaml

Parallel sweep (one process per episode; ``tasks`` picks the benchmark's task
indices so each (approach, task) pair is its own job):
    uv run python evaluation/run_experiment.py -m hydra/launcher=joblib \\
        hydra.launcher.n_jobs=8 approach=... tasks=0,1,2,3 benchmark=procedural ...

Every arm plans on the person the config describes (its ``pose:`` body with the
``arm:`` start); the motion generator is loaded only for approaches that need it.

Each task's episode lands in ``task_NN_<persona>_<verbalizer>/`` under the hydra
run dir with its ``interactions.pkl``; ``goals.csv`` at the run root is the goal
table for a quick look. Aggregate runs with ``analyze_results.py``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from evaluation.approaches.base import Approach
from evaluation.benchmarks.base import Benchmark
from evaluation.benchmarks.episode import run_episode
from evaluation.benchmarks.structs import Interaction
from evaluation.metrics.cost_learning.success import goal_table
from uncertain_feedback.motion_generators import make_motion_generator
from uncertain_feedback.planners.mpc.config import load_mpc_config
from uncertain_feedback.planners.mpc.goal_spaces import GoalStallConfig
from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.simulated_users import get_persona

_REPO_ROOT = Path(__file__).resolve().parents[1]


@hydra.main(version_base=None, config_name="config", config_path="conf")
def _main(cfg: DictConfig) -> None:
    # The MDM loader os.chdir()s into its submodule and never restores. Hydra
    # creates the next multirun job's output dir relative to the cwd before
    # calling us, so the cwd has to be back at the repo root when we return.
    os.chdir(_REPO_ROOT)
    try:
        _run(cfg)
    finally:
        os.chdir(_REPO_ROOT)


def _run(cfg: DictConfig) -> None:
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

    run_cfg = replace(load_mpc_config(mpc_config), seed=seed)
    if run_cfg.cartesian is not None:
        stall = (
            (run_cfg.cartesian.stall or GoalStallConfig()) if cfg.goal_stall else None
        )
        run_cfg = replace(run_cfg, cartesian=replace(run_cfg.cartesian, stall=stall))
        logging.info("goal stall: %s", stall)
    human = Human(pose=run_cfg.pose, arm=run_cfg.arm)
    gen = (
        make_motion_generator(run_cfg.motion_generator, None, seed=seed)
        if approach.requires_generator
        else None
    )
    if cfg.sim_chooser is not None:
        run_cfg = replace(
            run_cfg,
            simulated_user=replace(
                run_cfg.simulated_user, chooser=str(cfg.sim_chooser)
            ),
        )
    tasks = benchmark.generate_tasks(seed, run_cfg)
    if cfg.max_tasks is not None:
        tasks = tasks[: int(cfg.max_tasks)]
    task_ids = list(range(len(tasks)))
    if cfg.tasks is not None:
        selected = (
            [int(cfg.tasks)]
            if isinstance(cfg.tasks, int)
            else [int(i) for i in cfg.tasks]
        )
        task_ids = [i for i in task_ids if i in selected]
    if not task_ids:
        print(f"[evaluation] no tasks selected for {out_dir}")
        return

    interactions: list[Interaction] = []
    for task_id in task_ids:
        task = tasks[task_id]
        user = task.user if task.user is not None else get_persona(task.persona)
        episode_dir = out_dir / f"task_{task_id:02d}_{task.persona}_{task.verbalizer}"
        approach.reset(run_cfg, human, gen, user, task.seed, episode_dir)
        interactions.extend(
            run_episode(run_cfg, human, user, task, approach, episode_dir)
        )
        goals = goal_table(interactions)
        goals.to_csv(out_dir / "goals.csv", index=False)
    print(goals.to_string(index=False))
    print(f"[evaluation] wrote {out_dir}")


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
