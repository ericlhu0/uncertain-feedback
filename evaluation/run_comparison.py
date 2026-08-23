"""Compare grounding methods on one scenario, with the utterance given as text.

Unlike ``run_single_experiment.py`` (which synthesizes the utterance from the
persona's hidden intent at whatever abstraction level the benchmark asks for),
this runs one chosen sentence through every approach and writes a side-by-side
video plus a per-approach table. The persona still scores the candidates against
its own hidden intent, so the numbers stay meaningful — only the words the
grounding methods read are yours.

Every arm shares one rig, so the nominal rollout and the trigger step are
identical and the comparison isolates grounding.

    uv run python evaluation/run_comparison.py \
        --feedback "keep my arm closer to my body and keep my arm lower" \
        --persona triceps_long_head_contracture \
        --goal 0.25 0.3 0.18 \
        --mpc-config evaluation/conf/mpc_demo_base1.yaml \
        --out outputs/comparison

Add ``--approaches agent_waypoint cost_only full`` to pick the arms (any YAML
name in ``evaluation/conf/approach/``), or ``--no-video`` for the table alone.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from evaluation.approaches.base import Approach
from evaluation.approaches.cost_gen import COST_GEN_MODES
from evaluation.benchmarks.base import InteractionBenchmark
from evaluation.comparison import comparison_table, render_comparison, run_arms
from evaluation.rig import build_rig
from uncertain_feedback.simulated_users import get_persona

_REPO_ROOT = Path(__file__).resolve().parent.parent
_APPROACH_DIR = _REPO_ROOT / "evaluation" / "conf" / "approach"
_DEFAULT_APPROACHES = (
    "agent_waypoint",
    "agent_sparse_waypoints",
    "agent_dense_positions",
    "agent_dense_anatomical",
    "cost_only",
    "full",
)


def _build_approach(name: str, learning: str) -> Approach:
    """Instantiate one approach from its hydra config, forcing the cost-gen mode."""
    if learning not in COST_GEN_MODES:
        raise ValueError(f"--learning must be one of {COST_GEN_MODES}.")
    with initialize_config_dir(config_dir=str(_APPROACH_DIR), version_base=None):
        cfg = compose(config_name=name, overrides=[f"cost_gen={learning}"])
    approach = instantiate(cfg)
    assert isinstance(approach, Approach)
    return approach


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", required=True, help="the utterance to ground")
    parser.add_argument("--persona", default="triceps_long_head_contracture")
    parser.add_argument("--goal", type=float, nargs=3, default=[0.25, 0.3, 0.18])
    parser.add_argument(
        "--mpc-config", type=Path, default=Path("evaluation/conf/mpc_demo_base1.yaml")
    )
    parser.add_argument("--approaches", nargs="+", default=list(_DEFAULT_APPROACHES))
    parser.add_argument("--out", type=Path, default=Path("outputs/comparison"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument(
        "--learning",
        default="none",
        help="learning mode forced on every arm, so they differ only in grounding",
    )
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run every approach on one scripted utterance and report the comparison."""
    args = _parse_args()
    approaches = [_build_approach(name, args.learning) for name in args.approaches]
    mpc_config = args.mpc_config
    if not mpc_config.is_absolute():
        mpc_config = _REPO_ROOT / mpc_config
    # One shared rig: mixing generator and T-pose rigs would give the arms
    # different torso geometry and different nominal rollouts.
    rig = build_rig(
        mpc_config,
        seed=args.seed,
        load_generator=any(a.requires_generator for a in approaches),
    )
    # The MDM loader chdir()s into its submodule and never restores.
    os.chdir(_REPO_ROOT)

    benchmark = InteractionBenchmark(
        name="comparison",
        personas=[args.persona],
        verbalizers=["scripted"],
        goals=[list(args.goal)],
        max_rounds=args.max_rounds,
        feedback_text=args.feedback,
    )
    task = benchmark.generate_tasks(args.seed, rig.cfg)[0]
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = run_arms(rig, get_persona(args.persona), task, approaches, out_dir)
    rows.to_csv(out_dir / "rounds.csv", index=False)
    table = comparison_table(rows)
    table.to_csv(out_dir / "comparison.csv", index=False)
    print(table.to_string(index=False))

    if not args.no_video:
        video = render_comparison(
            rig,
            get_persona(args.persona),
            np.asarray(args.goal, dtype=np.float64),
            out_dir,
            [approach.name for approach in approaches],
        )
        print(f"[comparison] wrote {video}")
    print(f"[comparison] wrote {out_dir / 'comparison.csv'}")


if __name__ == "__main__":
    main()
