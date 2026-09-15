"""Aggregate evaluation runs into per-approach tables and plots.

uv run python evaluation/analyze_results.py outputs/ multirun/ --out analysis/

Loads every ``interactions.pkl`` an episode wrote under the roots and runs the
cost-learning metric on the pooled :class:`Interaction` list: success within k
feedback rounds per approach and, for goal sequences, per goal index; the
breakdown of goal results; per-round grounding quality against feedback events.
Every ``menu.csv`` a bound-transfer run wrote is pooled into ``all_menus.csv``
with mean menu violation per approach and goal index.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import pandas as pd  # pylint: disable=wrong-import-position

from evaluation.benchmarks.structs import (
    Interaction,
)  # pylint: disable=wrong-import-position
from evaluation.metrics.cost_learning.success import (  # pylint: disable=wrong-import-position
    goal_table,
    success_at_k,
)
from evaluation.metrics.cost_learning.rounds import (
    round_rows,
)  # pylint: disable=wrong-import-position


def _collect(roots: list[Path]) -> list[Interaction]:
    interactions: list[Interaction] = []
    for root in roots:
        for path in sorted(root.rglob("interactions.pkl")):
            with open(path, "rb") as file:
                interactions.extend(pickle.load(file))
    return interactions


def _plot_success_at_k(curve: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(5.5, 4))
    for approach, group in curve.groupby("approach"):
        axis.plot(group["k"], group["success"], marker="o", label=str(approach))
    axis.set_xlabel("feedback rounds k")
    axis.set_xticks(sorted(curve["k"].unique()))
    axis.set_ylabel("fraction of goals resolved within k")
    axis.set_ylim(0, 1.02)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_by_goal(curve: pd.DataFrame, max_k: int, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, k, label in (
        (axes[0], 0, "zero-shot success (no feedback)"),
        (axes[1], max_k, f"success within {max_k} rounds"),
    ):
        at_k = curve[curve["k"] == k]
        for approach, group in at_k.groupby("approach"):
            axis.plot(
                group["goal_index"], group["success"], marker="o", label=str(approach)
            )
        axis.set_ylabel(label)
        axis.set_ylim(0, 1.02)
        axis.set_xlabel("goal index in sequence")
        axis.set_xticks(sorted(curve["goal_index"].unique()))
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
    """Pool every interactions.pkl under the roots into tables and plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="+", help="Run/multirun dirs")
    parser.add_argument("--out", type=Path, default=Path("evaluation_analysis"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    interactions = _collect(args.roots)
    if not interactions:
        raise SystemExit("No interactions.pkl found under the given roots.")

    goals = goal_table(interactions)
    goals.to_csv(args.out / "all_goals.csv", index=False)
    results = pd.crosstab(goals["approach"], goals["result"], normalize="index")
    results.to_csv(args.out / "goal_results.csv")
    print(results.to_string())

    max_k = max(item.task.max_rounds for item in interactions)
    curve = success_at_k(interactions, max_k)
    curve.to_csv(args.out / "success_at_k.csv", index=False)
    print(curve.pivot(index="k", columns="approach", values="success").to_string())
    _plot_success_at_k(curve, args.out / "success_at_k.png")

    if len({item.goal_index for item in interactions}) > 1:
        by_goal = success_at_k(interactions, max_k, by=("approach", "goal_index"))
        by_goal.to_csv(args.out / "success_at_k_by_goal.csv", index=False)
        _plot_by_goal(by_goal, max_k, args.out / "success_by_goal.png")

    rows = pd.DataFrame([row for item in interactions for row in round_rows(item)])
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
    menus = [
        pd.read_csv(path)
        for root in args.roots
        for path in sorted(root.rglob("menu.csv"))
    ]
    if menus:
        menu = pd.concat(menus, ignore_index=True)
        menu.to_csv(args.out / "all_menus.csv", index=False)
        print(
            menu.groupby(["approach", "goal_index"])["violation"]
            .agg(["mean", "count"])
            .to_string()
        )
    print(f"[evaluation] analysis written to {args.out}")


if __name__ == "__main__":
    main()
