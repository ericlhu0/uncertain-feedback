"""Aggregate evaluation runs into per-approach tables and per-round plots.

uv run python evaluation/analyze_results.py outputs/ multirun/ --out analysis/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import pandas as pd  # pylint: disable=wrong-import-position


def _collect(roots: list[Path], filename: str) -> pd.DataFrame:
    frames = []
    for root in roots:
        for path in sorted(root.rglob(filename)):
            frame = pd.read_csv(path)
            if not frame.empty:
                frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _episode_table(episodes: pd.DataFrame) -> pd.DataFrame:
    grouped = episodes.groupby(["approach", "benchmark"])
    return pd.DataFrame(
        {
            "episodes": grouped.size(),
            "resolved_rate": grouped["all_goals_resolved"].mean(),
            "reached_rate": grouped["all_goals_reached"].mean(),
            "mean_feedback_events": grouped["feedback_events"].mean(),
            "mean_executed_violation": grouped["executed_mean_violation"].mean(),
        }
    ).reset_index()


def _plot_by_event(rows: pd.DataFrame, columns: dict[str, str], path: Path) -> None:
    fig, axes = plt.subplots(1, len(columns), figsize=(5.5 * len(columns), 4))
    if len(columns) == 1:
        axes = [axes]
    for axis, (column, label) in zip(axes, columns.items()):
        for approach, group in rows.groupby("approach"):
            series = group.groupby("event_index")[column].mean()
            axis.plot(series.index, series.values, marker="o", label=str(approach))
        axis.set_xlabel("feedback event")
        axis.set_ylabel(label)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Aggregate results.csv/episodes.csv trees into tables and plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="+", help="Run/multirun dirs")
    parser.add_argument("--out", type=Path, default=Path("evaluation_analysis"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    episodes = _collect(args.roots, "episodes.csv")
    rows = _collect(args.roots, "results.csv")
    if episodes.empty:
        raise SystemExit("No episodes.csv found under the given roots.")

    table = _episode_table(episodes)
    table.to_csv(args.out / "aggregate_episodes.csv", index=False)
    print(table.to_string(index=False))

    if not rows.empty:
        rows.to_csv(args.out / "all_rounds.csv", index=False)
        _plot_by_event(
            rows,
            {
                "any_acceptable": "candidate coverage",
                "candidate_hidden_mean": "mean candidate hidden cost",
            },
            args.out / "grounding_vs_events.png",
        )
        _plot_by_event(
            rows,
            {"continuation_mean_violation": "continuation mean violation (rad)"},
            args.out / "violation_vs_events.png",
        )
    print(f"[evaluation] analysis written to {args.out}")


if __name__ == "__main__":
    main()
