"""Render the oracle correction the grounding methods are evaluated against.

Sampled evaluation cases (random start / goal / trigger, with a hidden bound
sampled to be crossed at the trigger):

    uv run python evaluation/visualize_oracle.py \
        --clips-dir src/uncertain_feedback/data_collection/data/dataset_auto_correction/clips_auto500_s1 \
        --n-cases 6 --out-dir outputs/oracle_viz

Fixed-persona interaction benchmark cases:

    uv run python evaluation/visualize_oracle.py \
        --mpc-config src/uncertain_feedback/planners/mpc/configs/mdm_llm_transfer.yaml \
        --out-dir outputs/oracle_viz_persona
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.benchmarks.oracle_viz import (
    build_persona_case,
    build_sampled_case,
    case_summary,
    render_case,
)
from uncertain_feedback.data_collection.dataset_auto_correction.clips import (
    clip_source_from_dir,
)
from uncertain_feedback.planners.rig import build_rig
from uncertain_feedback.simulated_users import PERSONAS, get_persona


def _sampled(args: argparse.Namespace) -> list[dict[str, object]]:
    source = clip_source_from_dir(args.clips_dir)
    summaries = []
    for index in range(args.first_case, args.first_case + args.n_cases):
        case = build_sampled_case(source, index, oracle_steps=args.oracle_steps)
        render_case(case, source.context, source.body_pos, args.out_dir)
        summaries.append(case_summary(case))
    return summaries


def _persona(args: argparse.Namespace) -> list[dict[str, object]]:
    rig = build_rig(
        args.mpc_config, seed=args.seed, load_generator=not args.no_generator
    )
    names = args.personas or [
        name
        for name, user in PERSONAS.items()
        if user.bounds and name in rig.cfg.persona_goals
    ]
    summaries = []
    for name in names:
        goal = np.asarray(rig.cfg.persona_goals[name].cartesian[0], dtype=np.float64)
        case = build_persona_case(rig, get_persona(name), goal, seed=args.seed)
        if case is None:
            print(f"[oracle-viz] {name}: nominal plan never violates; no correction.")
            continue
        render_case(case, rig.context, rig.body_pos, args.out_dir)
        summaries.append(case_summary(case))
    return summaries


def main() -> None:
    """Build each case's oracle correction and render it against the nominal."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--clips-dir",
        type=Path,
        default=None,
        help="Clip set supplying the body geometry and sampling config; "
        "selects the sampled-case source.",
    )
    parser.add_argument("--n-cases", type=int, default=6)
    parser.add_argument(
        "--oracle-steps",
        type=int,
        default=2000,
        help="Step budget for the oracle replan, overriding the clip config's "
        "300 so a detour is not cut off before it reaches the goal.",
    )
    parser.add_argument("--first-case", type=int, default=0)
    parser.add_argument("--mpc-config", type=Path, default=None)
    parser.add_argument("--personas", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-generator", action="store_true")
    args = parser.parse_args()

    if (args.clips_dir is None) == (args.mpc_config is None):
        parser.error("pass exactly one of --clips-dir (sampled) or --mpc-config.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = _sampled(args) if args.clips_dir is not None else _persona(args)
    with open(args.out_dir / "oracle_cases.json", "w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=2)
    print(f"[oracle-viz] wrote {args.out_dir / 'oracle_cases.json'}")


if __name__ == "__main__":
    main()
