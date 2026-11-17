"""Grounding comparison on held-out cases: ours pooled over MDM seeds against a reference run.

    uv run python evaluation/paper_tables/grounding_table.py \\
        --reference <run dir> --ours <run dir> [<run dir> ...] [--clips-dir <clip set>]

Every run dir is a ``run_grounding.py`` output (``cases.csv`` and a
``trajectories.npz`` per case). The ``--ours`` runs are the same cases sampled
with different MDM seeds: a case's value is its mean over the seeds that define
the metric (direction is undefined where the oracle barely moves); the
reference is one run, the LLM grounder or a parent checkpoint. Metrics:
direction (net-change cosine with the oracle), arc progress, violation,
acceptable (the menu held an option the user accepts at some magnitude) and
position diversity. Per metric: both means over cases, the paired difference
ours − reference with a case-bootstrap 95% CI, and a Wilcoxon p. A run that
predates the stored ``chosen_direction`` column has it recomputed from the
chosen trajectory and the oracle, which needs ``--clips-dir`` for the body.
Produced the paper's held-out grounding table (ours over 8 seeds) in the
2026-10-20 audit; one or two seeds sat in the low tail of the model's spread.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from uncertain_feedback.data_collection.dataset_auto_correction.clips import (
    clip_source_from_dir,
)
from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.simulated_users.progress import correction_progress
from uncertain_feedback.uncertainty.cluster_picker import scale_trajectory

METRICS = ("direction", "arc_progress", "violation", "acceptable", "position_diversity")


def _recomputed_direction(
    run_dir: Path, cases: pd.DataFrame, human: Human
) -> list[float]:
    direction = []
    for label, case in cases.iterrows():
        saved = np.load(run_dir / str(label) / "trajectories.npz")
        chosen = scale_trajectory(
            saved[f"candidate_{int(case['chosen_label'])}"], float(case["magnitude"])
        )
        direction.append(correction_progress(saved["oracle"], chosen, human).direction)
    return direction


def load_run(run_dir: Path, human: Human | None = None) -> pd.DataFrame:
    """Per-case metrics of one run, indexed by case label."""
    cases = pd.read_csv(run_dir / "cases.csv").set_index("label")
    if "chosen_direction" in cases.columns:
        direction = cases["chosen_direction"].to_numpy(dtype=float)
    elif human is not None:
        direction = np.asarray(_recomputed_direction(run_dir, cases, human))
    else:
        raise ValueError(f"{run_dir} has no chosen_direction column; pass --clips-dir")
    return pd.DataFrame(
        {
            "direction": direction,
            "arc_progress": cases["chosen_arc_progress"].to_numpy(dtype=float),
            "violation": cases["chosen_violation"].to_numpy(dtype=float),
            "acceptable": 1.0 - cases["no_acceptable_cluster"].to_numpy(dtype=float),
            "position_diversity": cases["position_diversity"].to_numpy(dtype=float),
        },
        index=cases.index,
    )


def pool_seeds(runs: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Per-case mean over runs, on the cases every run has, skipping undefined values."""
    index = runs[0].index
    for run in runs[1:]:
        index = index.intersection(run.index)
    stacked = {
        metric: pd.concat([run.loc[index, metric] for run in runs], axis=1).mean(axis=1)
        for metric in METRICS
    }
    return pd.DataFrame(stacked, index=index)


def compare(
    reference: pd.DataFrame,
    ours: pd.DataFrame,
    n_boot: int = 4000,
    seed: int = 0,
) -> list[str]:
    """One line per metric: means, paired Δ ours − reference, bootstrap CI, Wilcoxon p."""
    rng = np.random.default_rng(seed)
    index = ours.index.intersection(reference.index)
    lines = []
    for metric in METRICS:
        delta = (ours.loc[index, metric] - reference.loc[index, metric]).dropna()
        values = delta.to_numpy()
        boot = np.array(
            [
                values[rng.integers(0, len(values), len(values))].mean()
                for _ in range(n_boot)
            ]
        )
        p = float(wilcoxon(values).pvalue) if (values != 0).any() else 1.0
        lines.append(
            f"{metric:19s} reference {reference.loc[delta.index, metric].mean():.3f}  "
            f"ours {ours.loc[delta.index, metric].mean():.3f}  "
            f"Δ {values.mean():+.3f} [{np.percentile(boot, 2.5):+.3f}, "
            f"{np.percentile(boot, 97.5):+.3f}]  Wilcoxon p {p:.2g}  (n {len(values)})"
        )
    return lines


def report(reference: pd.DataFrame, ours_runs: Sequence[pd.DataFrame]) -> str:
    """Per-seed means, then the pooled comparison against the reference."""
    ours = pool_seeds(ours_runs)
    lines = [
        f"{len(ours)} cases; per-seed direction "
        + " ".join(f"{run['direction'].mean():.3f}" for run in ours_runs),
        "per-seed acceptable "
        + " ".join(f"{run['acceptable'].mean():.3f}" for run in ours_runs),
    ]
    lines.extend(compare(reference, ours))
    return "\n".join(lines)


def main() -> None:
    """Compare seed-pooled runs of ours against one reference run on their shared cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--ours", type=Path, nargs="+", required=True)
    parser.add_argument("--clips-dir", type=Path, default=None)
    args = parser.parse_args()
    human = (
        None
        if args.clips_dir is None
        else clip_source_from_dir(args.clips_dir.resolve()).human
    )
    print(
        report(
            load_run(args.reference.resolve(), human),
            [load_run(run.resolve(), human) for run in args.ours],
        )
    )


if __name__ == "__main__":
    main()
