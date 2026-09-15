"""Run the bound-transfer evaluation for one (approach, benchmark, seed) with hydra.

    uv run python evaluation/run_bound_transfer.py -m seed=0 \\
        approach=mdm_language,full \\
        benchmark=bound_transfer load_generator=true \\
        mpc_config=src/uncertain_feedback/planners/mpc/configs/mdm_llm_transfer.yaml \\
        hydra.sweep.dir=outputs/bound_transfer/seed0

Per task the approach learns on the first goal (a full episode under
``task_NN_.../learn``), then every later goal is probed: the unlearned plan
provokes feedback, the approach proposes a menu, and each candidate is scored
by its hidden-bound violation. ``menu.csv`` at the run root has one row per
candidate; pool runs with ``analyze_results.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import hydra
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from evaluation.approaches.base import Approach
from evaluation.benchmarks.base import Benchmark
from evaluation.benchmarks.structs import MenuProbe
from evaluation.benchmarks.transfer import run_transfer
from evaluation.metrics.cost_learning.menu import menu_rows
from uncertain_feedback.planners.rig import build_rig
from uncertain_feedback.simulated_users import get_persona

_REPO_ROOT = Path(__file__).resolve().parents[1]


@hydra.main(version_base=None, config_name="config", config_path="conf")
def _main(cfg: DictConfig) -> None:
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
    load_generator = (
        approach.requires_generator
        if cfg.load_generator is None
        else bool(cfg.load_generator)
    )
    rig = build_rig(mpc_config, seed=seed, load_generator=load_generator)
    tasks = benchmark.generate_tasks(seed, rig.cfg)
    if cfg.max_tasks is not None:
        tasks = tasks[: int(cfg.max_tasks)]

    probes: list[MenuProbe] = []
    for task_id, task in enumerate(tasks):
        user = task.user if task.user is not None else get_persona(task.persona)
        episode_dir = out_dir / f"task_{task_id:02d}_{task.persona}_{task.verbalizer}"
        approach.reset(rig, user, task.seed, episode_dir)
        probes.extend(run_transfer(rig, user, task, approach, episode_dir))
        menu = pd.DataFrame([row for probe in probes for row in menu_rows(probe)])
        menu.to_csv(out_dir / "menu.csv", index=False)
    if probes:
        print(
            menu.groupby("goal_index")["violation"].agg(["mean", "count"]).to_string()
        )
    print(f"[transfer] wrote {out_dir}")


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
