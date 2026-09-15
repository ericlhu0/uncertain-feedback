"""Score a grounder on sampled oracle cases, as the simulated user would use it.

    uv run python evaluation/run_grounding.py --grounder llm \\
        --clips-dir src/uncertain_feedback/data_collection/data/dataset_auto_correction/clips_auto200_s5 \\
        --n-cases 20 --out-dir outputs/grounding_llm_auto200_s5

    uv run python evaluation/run_grounding.py --grounder mdm \\
        --model-path <checkpoint.pt> \\
        --clips-dir ... --n-cases 20 --out-dir outputs/grounding_mdm_auto200_s5

Case ``i`` is run ``i`` of the clip set (same seed, same draw order), so its
first VLM caption is the utterance and the clip's hidden bound and oracle window
are the ground truth. The correction starts one frame before the bound crossing
(see :func:`build_sampled_case`); the grounder proposes a candidate menu from the
naive continuation there, and the simulated user's chooser (``oracle_progress``
by default, from the planner config) picks a candidate and magnitude, and that
scaled correction is scored with the violation and progress metrics, the naive
continuation alongside it as the ``nominal`` reference. Expressivity is scored
over the whole menu, in anatomical-feature space and in elbow/wrist position space. A case whose ``trajectories.npz`` already exists reuses
its menu, shifted onto the case's start frame, instead of grounding again.

``llm`` needs ``OPENAI_API_KEY`` only; ``mdm`` loads the generator (GPU) and
checks its body against the clip set's.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.approaches.grounders.base import Grounder
from evaluation.approaches.grounders.llm_trajectory import LlmTrajectoryGrounder
from evaluation.approaches.grounders.mdm import MdmGrounder
from evaluation.benchmarks.oracle_viz import OracleCase, build_sampled_case, case_summary
from evaluation.metrics.grounding.score import candidate_row, case_row
from uncertain_feedback.data_collection.dataset_auto_correction.clips import (
    ClipSource,
    clip_source_from_dir,
)
from uncertain_feedback.planners.mpc.kinematics import q_to_arm_aa
from uncertain_feedback.planners.rig import PlanningRig, build_rig
from uncertain_feedback.simulated_users import attribute_correction, choose_correction
from uncertain_feedback.uncertainty.cluster_picker import scale_trajectory

_LOG = "[grounding]"
_METRICS = ("violation", "arc_progress", "alignment")


def _rig(args: argparse.Namespace, source: ClipSource) -> PlanningRig:
    if args.grounder == "mdm":
        rig = build_rig(
            source.cfg.config_path,
            seed=args.seed,
            load_generator=True,
            model_path=args.model_path,
        )
        assert rig.body_pos is not None
        assert np.allclose(rig.spine3_pos, source.context.spine3_pos, atol=1e-6)
        assert np.allclose(rig.body_pos, source.body_pos, atol=1e-6)
        return rig
    return PlanningRig(
        cfg=source.run_cfg,
        fk=source.context.fk,
        context=source.context,
        q0=np.zeros(7),
        spine3_pos=source.context.spine3_pos,
        spine3_aa=source.context.spine3_aa,
        body_pos=source.body_pos,
        gen=None,
        initial_hml_pose=None,
    )


def _grounder(args: argparse.Namespace) -> Grounder:
    if args.grounder == "mdm":
        return MdmGrounder()
    return LlmTrajectoryGrounder(
        n_interpretations=4, output_space="positions", n_waypoints=5, n_frames=50
    )


def _start_at(rig: PlanningRig, arm_aa: np.ndarray, q_start: np.ndarray) -> np.ndarray:
    q = rig.fk.arm_aa_to_q_batch(arm_aa, rig.spine3_aa)
    return q_to_arm_aa(q - q[0] + q_start, rig.fk.elbow_hinge_axis)


def _menu(
    grounder: Grounder,
    rig: PlanningRig,
    case: OracleCase,
    utterance: str,
    case_dir: Path,
    seed: int,
) -> dict[int, np.ndarray]:
    path = case_dir / "trajectories.npz"
    if path.exists():
        # A cached menu may have been proposed from a different start frame;
        # shift it onto this case's start, keeping every per-frame displacement.
        saved = np.load(path)
        return {
            int(key.split("_")[1]): _start_at(rig, saved[key], case.q_feedback)
            for key in saved.files
            if key.startswith("candidate_")
        }
    grounder.reset(rig, case.user, seed, case_dir)
    result = grounder.ground(
        utterance, case.q_feedback, case.nominal_continuation, lambda _: (0, 1.0)
    )
    np.savez(
        path,
        oracle=case.oracle_correction,
        nominal=case.nominal_continuation,
        **{f"candidate_{label}": traj for label, traj in result.candidates.items()},
    )
    return result.candidates


def main() -> None:
    """Ground every case's caption, choose as the user would, score, tabulate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grounder", choices=("llm", "mdm"), default="llm")
    parser.add_argument("--model-path", type=Path, default=None, help="mdm weights")
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-cases", type=int, default=20)
    parser.add_argument("--first-case", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    clips_dir = args.clips_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.model_path is not None:
        args.model_path = args.model_path.resolve()
    source = clip_source_from_dir(clips_dir)
    runs = json.loads((clips_dir / "manifest.json").read_text(encoding="utf-8"))["runs"]
    sim_cfg = source.run_cfg.simulated_user
    rig = _rig(args, source)
    grounder = _grounder(args)
    context = source.context
    rng = np.random.default_rng(args.seed)

    rows = []
    for index in range(args.first_case, args.first_case + args.n_cases):
        case = build_sampled_case(source, index)
        run = runs[index]
        assert run["feature"] == case.detail["feature"], (index, run["feature"])
        assert run["trigger_step"] == case.trigger_step, (index, run["trigger_step"])
        utterance = run["captions"][0]
        case_dir = out_dir / case.label
        case_dir.mkdir(parents=True, exist_ok=True)
        menu = _menu(grounder, rig, case, utterance, case_dir, args.seed)

        intent = attribute_correction(
            case.oracle_correction, case.nominal_continuation, case.q_feedback, context
        )
        choice = choose_correction(
            case.user,
            context,
            menu,
            case.oracle_correction,
            threshold=source.threshold,
            magnitudes=sim_cfg.magnitudes,
            mode=sim_cfg.chooser,
            intent=intent,
            rng=rng,
        )
        chosen = scale_trajectory(menu[choice.label], choice.magnitude)
        chosen_scores = candidate_row(case.user, case.oracle_correction, chosen, context)
        nominal_scores = candidate_row(
            case.user, case.oracle_correction, case.nominal_continuation, context
        )
        rows.append(
            {
                **case_summary(case),
                "utterance": utterance,
                "chosen_label": choice.label,
                "magnitude": choice.magnitude,
                "no_acceptable_cluster": choice.no_acceptable_cluster,
                **{f"chosen_{k}": v for k, v in chosen_scores.items()},
                **{f"nominal_{k}": v for k, v in nominal_scores.items()},
                **case_row(menu, context),
            }
        )
        print(
            f"{_LOG} {case.label} {utterance!r}: chose {choice.label} x{choice.magnitude}"
            f"{' (none acceptable)' if choice.no_acceptable_cluster else ''}, "
            f"violation {chosen_scores['violation']:.3f} vs nominal "
            f"{nominal_scores['violation']:.3f}, progress "
            f"{chosen_scores['arc_progress']:.2f}, diversity {rows[-1]['diversity']:.2f}",
            flush=True,
        )
        pd.DataFrame(rows).to_csv(out_dir / "cases.csv", index=False)

    cases = pd.DataFrame(rows)
    summary = pd.DataFrame(
        {
            args.grounder: [cases[f"chosen_{m}"].mean() for m in _METRICS],
            "nominal": [cases[f"nominal_{m}"].mean() for m in _METRICS],
        },
        index=list(_METRICS),
    )
    summary.loc["diversity", args.grounder] = cases["diversity"].mean()
    summary.loc["position_diversity", args.grounder] = cases["position_diversity"].mean()
    summary.loc["acceptable_rate", args.grounder] = 1 - cases["no_acceptable_cluster"].mean()
    summary.to_csv(out_dir / "summary.csv")
    print(f"{_LOG} {len(rows)} cases\n{summary.to_string()}")


if __name__ == "__main__":
    main()
