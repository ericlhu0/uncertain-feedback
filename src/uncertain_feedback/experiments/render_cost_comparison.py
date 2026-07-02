"""Render a cost rollout against the target correction — for the ``agent`` backend.

The ``agent`` (codex) cost generator writes its answer as ``response.json`` (the
shared ``description`` / ``code`` / ``params`` / ... schema) and runs this script to
see how the cost it just authored actually steers the arm: it rolls the goal-seeking
MPC with that cost installed and overlays the result against the user's correction,
writing ``--out`` and printing the L2 score. Codex inspects the image and the score,
refines ``response.json``, and repeats.

Run as::

    uv run python src/uncertain_feedback/experiments/render_cost_comparison.py \
        --state state.pkl --response response.json --out comparison.png
"""

from __future__ import annotations

import json
import shutil

import argparse
from pathlib import Path

from uncertain_feedback.planners.mpc.costs import (
    EvalState,
    GeneratedPythonCost,
    evaluate_and_render,
    parse_llm_cost_response,
)


def _next_candidate_dir(archive_dir: Path) -> Path:
    archive_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    while True:
        candidate_dir = archive_dir / f"candidate_{idx:03d}"
        if not candidate_dir.exists():
            candidate_dir.mkdir()
            return candidate_dir
        idx += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state",
        required=True,
        type=Path,
        help="Pickled EvalState produced by the agent cost generator.",
    )
    parser.add_argument(
        "--response",
        required=True,
        type=Path,
        help="response.json holding the candidate cost (code + params).",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output PNG path for the rollout-vs-correction overlay.",
    )
    parser.add_argument(
        "--angles-out",
        type=Path,
        default=None,
        help="Optional output PNG path for the joint-angle-over-time comparison.",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=None,
        help="Optional directory for numbered candidate artifacts.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save the candidate rollout video alongside archived artifacts.",
    )
    args = parser.parse_args()

    state = EvalState.load(args.state)
    context = state.make_generated_context()
    rollout_fn = state.make_rollout_fn()

    raw_response = args.response.read_text(encoding="utf-8")
    response = parse_llm_cost_response(raw_response)
    cost = GeneratedPythonCost(
        code=response.code,
        params=response.params,
        description=response.description,
        context=context,
    )

    candidate_dir = (
        _next_candidate_dir(args.archive_dir)
        if args.archive_dir is not None
        else None
    )
    rollout_path = None
    video_path = None
    if args.save_video:
        if candidate_dir is not None:
            rollout_path = candidate_dir / "rollout.npy"
            video_path = candidate_dir / "rollout.mp4"
        else:
            rollout_path = args.out.with_suffix(".npy")
            video_path = args.out.with_suffix(".mp4")

    score, image_path, _ = evaluate_and_render(
        context,
        cost,
        rollout_fn,
        args.out,
        angle_path=args.angles_out,
        rollout_path=rollout_path,
        video_path=video_path,
    )
    if candidate_dir is not None:
        (candidate_dir / "response.json").write_text(raw_response, encoding="utf-8")
        (candidate_dir / "cost.py").write_text(response.code, encoding="utf-8")
        with open(candidate_dir / "score.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "score": score,
                    "comparison_path": str(candidate_dir / "comparison.png"),
                    "angles_path": (
                        str(candidate_dir / "angles.png") if args.angles_out else None
                    ),
                    "rollout_path": str(rollout_path) if rollout_path else None,
                    "video_path": str(video_path) if video_path else None,
                },
                f,
                indent=2,
                sort_keys=True,
            )
        if image_path is not None:
            archive_image = candidate_dir / "comparison.png"
            if image_path.resolve() != archive_image.resolve():
                shutil.copyfile(image_path, archive_image)
            if args.angles_out is not None:
                archive_angles = candidate_dir / "angles.png"
                if args.angles_out.resolve() != archive_angles.resolve():
                    shutil.copyfile(args.angles_out, archive_angles)
    if image_path is None:
        print(
            "[cost-compare] no rollout available for this planner "
            "(no Cartesian goal); cannot render a comparison."
        )
        return
    print(f"[cost-compare] L2 score (lower is better): {score:.4f}")
    print(f"[cost-compare] comparison image: {image_path}")
    if args.angles_out is not None:
        print(f"[cost-compare] joint-angle image: {args.angles_out}")


if __name__ == "__main__":
    main()
