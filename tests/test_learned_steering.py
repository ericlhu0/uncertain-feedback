"""Tests for steering MDM with the learned costs instead of hidden limits."""

# pylint: disable=missing-function-docstring,protected-access

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from evaluation.approaches import Approach, ImmediateCostGen, MdmGrounder
from evaluation.approaches.cost_gen.structs import RoundContext
from evaluation.approaches.steering import LearnedSteering
from uncertain_feedback.evaluation_mechanism import EvalState
from uncertain_feedback.motion_generators.mdm.mdm_api import (
    N_PREFIX_FRAMES,
    MdmMotionGenerator,
)
from uncertain_feedback.motion_generators.mdm.torch_features import (
    build_generated_cost,
    features_from_hml,
)
from uncertain_feedback.motion_generators.steering import SteeringConfig
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    GeneratedPythonCost,
    build_generated_cost_context,
)
from uncertain_feedback.planners.mpc.costs.generated import interp
from uncertain_feedback.planners.mpc.goal_spaces import PointRegion
from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.simulated_users import get_persona
from uncertain_feedback.simulated_users.base import feature_series

_COUPLED = (
    "def path_margins(features, params):\n"
    "    floor = interp(features['shoulder_elevation'], params['xs'], params['ys'])\n"
    "    return {'floor': floor - features['elbow_flexion']}\n"
)
_COUPLED_PARAMS = {"xs": [0.2, 0.6, 1.4], "ys": [0.4, 1.6, 1.8]}
_ROTATION = (
    "def path_margins(features, params):\n"
    "    return {'twist': features['shoulder_internal_external_rotation']}\n"
)
_SPEED = (
    "def speed_margins(features, params):\n"
    "    return {'elbow': features['elbow_speed'] - 0.01}\n"
)


class _RecordingGenerator(MdmMotionGenerator):
    """Records the costs each steering cost is compiled from; loads no model."""

    def __init__(self) -> None:  # pylint: disable=super-init-not-called
        self.compiled: list = []
        self.speed_scale: float | None = None

    def build_generated_steering_cost(self, costs, speed_scale):
        self.compiled.append(list(costs))
        self.speed_scale = speed_scale
        return lambda x0: x0.sum()


def _cost(code: str, params: dict | None = None) -> GeneratedPythonCost:
    context = build_generated_cost_context(Human(), np.zeros((3, 3, 3)), window=3)
    return GeneratedPythonCost(code, params or {}, context)


def test_learned_steering_compiles_the_learned_costs() -> None:
    user = get_persona("elbow_contracture")
    assert user.bounds, "the persona must have hidden limits to leave unread"
    steering = LearnedSteering()
    gen = _RecordingGenerator()

    assert steering.spec(gen, user, SteeringConfig(), seed=0) is None
    assert not gen.compiled

    learned = [_cost(_COUPLED, _COUPLED_PARAMS)]
    steering.learned_costs = lambda: learned
    config = SteeringConfig(guidance_weight=1e5, speed_guidance_weight=1e3)
    assert steering.spec(gen, user, config, seed=0) is not None
    assert gen.compiled == [learned]
    # Speed limits steer at their own weight, as a ratio to the path weight.
    assert gen.speed_scale == 1e-2


def test_interp_matches_numpy_and_holds_the_end_anchors() -> None:
    x = np.linspace(-0.5, 2.0, 41)
    xs, ys = [0.2, 0.6, 1.4], [0.4, 1.6, 1.8]
    np.testing.assert_allclose(
        interp(torch.from_numpy(x), xs, ys).numpy(), np.interp(x, xs, ys)
    )


def test_steering_reads_the_planners_margins_on_differentiable_features() -> None:
    human = Human()
    cost = _cost(_COUPLED, _COUPLED_PARAMS)
    q_trajs = np.repeat(human.q[None, None], 4, axis=1).repeat(2, axis=0)
    q_trajs[0, :, 6] = np.linspace(1.5, 1.2, 4)  # straightens under the floor
    q_trajs[1, :, 6] = 1.5  # holds

    planned = cost.path_excess(q_trajs)
    assert planned[0] > planned[1] == 0.0

    features = {
        name: torch.tensor(series, requires_grad=True)
        for name, series in feature_series(human, q_trajs).items()
    }
    path, speed = cost.margins(features)
    assert not speed
    excess = path["floor"].clamp(min=0.0)
    np.testing.assert_allclose(
        (excess[:, 1:] - excess[:, :1]).clamp(min=0.0).max(dim=1).values.detach(),
        planned,
    )
    excess.pow(2).mean(dim=1).sum().backward()  # type: ignore[no-untyped-call]
    grad = features["elbow_flexion"].grad
    assert grad is not None and grad[0].abs().sum() > 0.0


def test_generated_steering_cost_skips_features_positions_cannot_give() -> None:
    mean, std = torch.zeros(263), torch.ones(263)
    assert build_generated_cost([_cost(_ROTATION)], mean, std, 1.0) is None

    steer = build_generated_cost(
        [_cost(_COUPLED, _COUPLED_PARAMS), _cost(_ROTATION)], mean, std, 1.0
    )
    assert steer is not None
    x0 = torch.randn(3, 263, 1, N_PREFIX_FRAMES + 6, requires_grad=True)
    value = steer(x0)
    assert value.shape == (3,)
    value.sum().backward()  # type: ignore[no-untyped-call]
    assert x0.grad is not None and torch.isfinite(x0.grad).all()


def test_generated_steering_cost_has_a_fixed_scale_per_kind_of_limit() -> None:
    mean, std = torch.zeros(263), torch.ones(263)
    x0 = torch.randn(3, 263, 1, N_PREFIX_FRAMES + 6)
    path, speed = _cost(_COUPLED, _COUPLED_PARAMS), _cost(_SPEED)

    def steer(costs: list, speed_scale: float) -> torch.Tensor:
        compiled = build_generated_cost(costs, mean, std, speed_scale)
        assert compiled is not None
        return compiled(x0)

    # Mean over time of the squared excess, in the margin's own units: no
    # weight of the generated code's making enters.
    features = features_from_hml(x0, mean, std)
    expected = speed.margins(features)[1]["elbow"].clamp(min=0.0).pow(2).mean(dim=1)
    torch.testing.assert_close(steer([speed], 1.0), expected)
    assert (expected > 0.0).all()
    torch.testing.assert_close(
        steer([path, speed], 0.01), steer([path], 1.0) + 0.01 * expected
    )


def _learn_round(cost_gen: ImmediateCostGen, tmp_path: Path, human: Human) -> None:
    trajectory = np.repeat(human.q[None], 3, axis=0)
    state = EvalState(
        cfg=SimpleNamespace(  # type: ignore[arg-type]
            steps=3,
            horizon=2,
            n_mpc_samples=4,
            max_angle_delta=0.01,
            seed=0,
            cartesian=None,
            constraints={},
            llm_cost=SimpleNamespace(
                limit_weight=1000.0, pullback_weight=10.0, enforcement="hard"
            ),
        ),
        human=human,
        correction_traj=trajectory,
        window=3,
        base_extra_costs=CompositeTrajectoryCost(),
        reference_traj=trajectory,
    )
    ctx = RoundContext(
        round_dir=tmp_path / f"round_{len(cost_gen._cost_rounds):02d}",
        goal=PointRegion((0.0, 0.0, 0.0)),
        utterance_text="keep my elbow bent",
        grounding=None,  # type: ignore[arg-type]
        human=human,
        event_index=len(cost_gen._cost_rounds),
        rejected_labels=frozenset(),
    )
    ctx.round_dir.mkdir(parents=True)
    generation = SimpleNamespace(
        generated_cost=GeneratedPythonCost(
            _COUPLED, _COUPLED_PARAMS, state.make_generated_context()
        ),
        eval_state=state,
        summaries={},
        images={},
        description="",
        explanation="",
        interpretation="",
        grounding=json.dumps({"terms": []}),
    )
    cost_gen.record(ctx, generation)  # type: ignore[arg-type]


def test_learned_steering_follows_the_costs_planning(tmp_path: Path) -> None:
    human = Human()
    cost_gen = ImmediateCostGen()
    steering = LearnedSteering()
    Approach("full", MdmGrounder(), cost_gen, steering)
    assert not steering.learned_costs()
    _learn_round(cost_gen, tmp_path, human)
    _learn_round(cost_gen, tmp_path, human)
    assert steering.learned_costs() == cost_gen._generated

    # A retired round's cost no longer plans, so it no longer steers either.
    cost_gen.retire([cost_gen._generated[0]], "too strict")
    assert steering.learned_costs() == [cost_gen._generated[1]]
