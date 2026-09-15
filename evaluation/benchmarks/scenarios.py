"""Benchmark over generated correction scenarios (see ``generate_scenarios.py``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from evaluation.benchmarks.base import Benchmark
from evaluation.benchmarks.structs import InteractionTask
from uncertain_feedback.planners.mpc.config import MpcRunConfig
from uncertain_feedback.simulated_users import (
    Bound,
    CoupledBound,
    FeatureCondition,
    HiddenBound,
    JointBoxLimit,
    SimulatedUser,
    get_persona,
)

# Resolved against the repo root: the MDM loader os.chdir()s into its submodule
# before the runner asks for tasks.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _bound_from_dict(data: dict[str, Any]) -> Bound:
    if "cond_feature" in data:
        return CoupledBound(**data)
    condition = data.get("condition")
    return HiddenBound(
        feature=data["feature"],
        bound_type=data["bound_type"],
        low=data.get("low"),
        high=data.get("high"),
        condition=FeatureCondition(**condition) if condition is not None else None,
    )


def user_from_dict(data: dict[str, Any]) -> SimulatedUser:
    """Rebuild a serialized ``SimulatedUser`` from a selection audit row."""
    return SimulatedUser(
        name=data["name"],
        description=data["description"],
        feedback_text=data["feedback_text"],
        bounds=tuple(_bound_from_dict(bound) for bound in data["bounds"]),
        joint_limits=tuple(
            JointBoxLimit(
                joint=limit["joint"],
                low=tuple(limit["low"]),
                high=tuple(limit["high"]),
            )
            for limit in data["joint_limits"]
        ),
    )


class ScenarioBenchmark(Benchmark):
    """One task per accepted scenario NPZ: its user, start pose and goal.

    Sampled-bound cases carry their synthetic user in the audit row; fixed-
    persona cases resolve the persona by name. ``recommended_only`` keeps the
    cases the audit's ``visual_review`` marked recommended.
    """

    def __init__(
        self,
        name: str,
        scenario_dir: str,
        verbalizers: Sequence[str] = ("everyday",),
        max_rounds: int = 3,
        recommended_only: bool = False,
    ) -> None:
        super().__init__(name)
        self.scenario_dir = _REPO_ROOT / scenario_dir
        self.verbalizers = list(verbalizers)
        self.max_rounds = max_rounds
        self.recommended_only = recommended_only

    def generate_tasks(self, seed: int, cfg: MpcRunConfig) -> list[InteractionTask]:
        del cfg
        audit = json.loads((self.scenario_dir / "selection.json").read_text())
        accepted = {
            str(row["persona"]): row
            for row in audit["attempts"]
            if row.get("rejected") == ""
        }
        review = audit.get("visual_review", {})
        tasks: list[InteractionTask] = []
        for case, row in accepted.items():
            if self.recommended_only and not review.get(case, {}).get("recommended"):
                continue
            data = np.load(self.scenario_dir / f"{case}.npz")
            user = user_from_dict(row["user"]) if "user" in row else get_persona(case)
            goal = tuple(float(v) for v in data["goal"])
            start = tuple(float(v) for v in data["naive"][0])
            for verbalizer in self.verbalizers:
                tasks.append(
                    InteractionTask(
                        persona=case,
                        verbalizer=verbalizer,
                        goals=(goal,),
                        max_rounds=self.max_rounds,
                        seed=seed,
                        user=user,
                        start_q=start,
                    )
                )
        return tasks
