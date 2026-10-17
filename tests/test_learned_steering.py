"""Tests for steering MDM with the learned preference instead of hidden limits."""

# pylint: disable=missing-function-docstring,protected-access

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from evaluation.approaches import (
    Approach,
    ConsolidateCostGen,
    CostGen,
    ImmediateCostGen,
    MdmGrounder,
)
from evaluation.approaches.cost_gen.bounds import admissible_bounds, bounds_from_terms
from evaluation.approaches.cost_gen.structs import RoundContext
from evaluation.approaches.steering import LearnedSteering
from uncertain_feedback.evaluation_mechanism import EvalState
from uncertain_feedback.motion_generators.mdm.mdm_api import MdmMotionGenerator
from uncertain_feedback.motion_generators.steering import SteeringConfig
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    GeneratedPythonCost,
)
from uncertain_feedback.planners.mpc.goal_spaces import PointRegion
from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.simulated_users import get_persona
from uncertain_feedback.simulated_users.base import (
    CoupledBound,
    HiddenBound,
    feature_series,
)

_ZERO_COST = "def cost(q_trajs, context, params):\n    return np.zeros(len(q_trajs))\n"


class _RecordingGenerator(MdmMotionGenerator):
    """Records the user each steering cost is compiled for; loads no model."""

    def __init__(self) -> None:  # pylint: disable=super-init-not-called
        self.compiled: list = []

    def build_user_steering_cost(self, user):
        self.compiled.append(user)
        return lambda x0: x0.sum()


def _elbow_trajectory(human: Human, elbow: float) -> np.ndarray:
    trajectory = np.repeat(human.q[None], 3, axis=0)
    trajectory[:, 6] = elbow
    return trajectory


def _elbow_flexion(human: Human, elbow: float) -> float:
    return float(
        feature_series(human, _elbow_trajectory(human, elbow))["elbow_flexion"][0]
    )


def _state(human: Human, chosen_elbow: float, interrupted_elbow: float) -> EvalState:
    cfg = SimpleNamespace(
        steps=3,
        horizon=2,
        n_mpc_samples=4,
        max_angle_delta=0.01,
        seed=0,
        cartesian=None,
        constraints={},
    )
    return EvalState(
        cfg=cfg,  # type: ignore[arg-type]
        human=human,
        correction_traj=_elbow_trajectory(human, chosen_elbow),
        window=3,
        base_extra_costs=CompositeTrajectoryCost(),
        reference_traj=_elbow_trajectory(human, interrupted_elbow),
    )


def _learn_round(
    cost_gen: CostGen, tmp_path: Path, terms: list[dict], state: EvalState
) -> None:
    human = state.human
    ctx = RoundContext(
        round_dir=tmp_path / f"round_{len(cost_gen._cost_rounds):02d}",
        goal=PointRegion((0.0, 0.0, 0.0)),
        utterance_text="keep my elbow straighter",
        grounding=None,  # type: ignore[arg-type]
        human=human,
        event_index=len(cost_gen._cost_rounds),
        rejected_labels=frozenset(),
    )
    ctx.round_dir.mkdir(parents=True)
    generation = SimpleNamespace(
        generated_cost=GeneratedPythonCost(
            _ZERO_COST, {}, state.make_generated_context()
        ),
        eval_state=state,
        summaries={},
        images={},
        description="",
        explanation="",
        interpretation="",
        grounding=json.dumps({"terms": terms}),
    )
    cost_gen.record(ctx, generation)  # type: ignore[arg-type]


def test_learned_steering_compiles_only_the_learned_bounds() -> None:
    user = get_persona("elbow_contracture")
    assert user.bounds, "the persona must have hidden limits to leave unread"
    steering = LearnedSteering()
    gen = _RecordingGenerator()

    assert steering.spec(gen, user, SteeringConfig(), seed=0) is None
    assert not gen.compiled

    learned = (HiddenBound("shoulder_elevation", "upper_bound", high=1.0),)
    steering.learned_bounds = lambda: learned
    assert steering.spec(gen, user, SteeringConfig(), seed=0) is not None
    assert len(gen.compiled) == 1
    target = gen.compiled[0]
    assert target.bounds == learned
    assert target.joint_limits == ()


def test_terms_become_bounds_and_inadmissible_ones_are_dropped() -> None:
    human = Human()
    straight, bent = _elbow_flexion(human, 0.3), _elbow_flexion(human, 1.2)
    middle = 0.5 * (straight + bent)
    bounds = bounds_from_terms(
        [
            {
                "feature": "elbow_flexion",
                "bound_type": "upper_bound",
                "values": {"threshold": middle},
            },
            {
                "feature": "elbow_flexion",
                "bound_type": "band",
                "values": {"low": 0.0, "high": 3.0},
            },
            {
                "feature": "elbow_flexion",
                "bound_type": "lower_bound",
                "values": {
                    "cond_feature": "shoulder_elevation",
                    "points": [[0.0, 0.1], [1.0, 0.3]],
                },
            },
            {"feature": "elbow_flexion", "bound_type": "upper_bound", "values": {}},
        ]
    )
    upper = HiddenBound("elbow_flexion", "upper_bound", high=middle)
    assert bounds[0] == upper
    assert bounds[1:3] == (
        HiddenBound("elbow_flexion", "lower_bound", low=0.0),
        HiddenBound("elbow_flexion", "upper_bound", high=3.0),
    )
    coupled = bounds[3]
    assert isinstance(coupled, CoupledBound)
    assert np.isclose(coupled.intercept, 0.1) and np.isclose(coupled.slope, 0.2)
    assert len(bounds) == 4  # the malformed term is skipped

    # Straight is chosen and bent was interrupted: only the upper bound both
    # admits the choice and excludes something the person turned away from.
    states = [_state(human, chosen_elbow=0.3, interrupted_elbow=1.2)]
    assert admissible_bounds(bounds[:3], states) == (upper,)
    flipped = [_state(human, chosen_elbow=1.2, interrupted_elbow=0.3)]
    assert not admissible_bounds([upper], flipped)


def test_immediate_learning_steers_from_its_stacked_rounds(tmp_path: Path) -> None:
    human = Human()
    straight, bent = _elbow_flexion(human, 0.3), _elbow_flexion(human, 1.2)
    threshold = 0.5 * (straight + bent)
    first = [
        {
            "feature": "elbow_flexion",
            "bound_type": "upper_bound",
            "values": {"threshold": threshold},
        }
    ]
    second = [{**first[0], "values": {"threshold": threshold + 0.02}}]
    cost_gen = ImmediateCostGen()
    steering = LearnedSteering()
    Approach("full", MdmGrounder(), cost_gen, steering)
    _learn_round(cost_gen, tmp_path, first, _state(human, 0.3, 1.2))
    _learn_round(cost_gen, tmp_path, second, _state(human, 0.3, 1.2))
    assert steering.learned_bounds() == (
        HiddenBound("elbow_flexion", "upper_bound", high=threshold),
        HiddenBound("elbow_flexion", "upper_bound", high=threshold + 0.02),
    )

    # A retired round's cost no longer plans, so it no longer steers either.
    cost_gen.retire([cost_gen._generated[0]], "too strict")
    assert steering.learned_bounds() == (
        HiddenBound("elbow_flexion", "upper_bound", high=threshold + 0.02),
    )
