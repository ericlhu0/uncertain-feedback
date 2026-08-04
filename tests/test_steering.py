"""Tests for cost-guided diffusion steering.

Covers the sampler-agnostic pieces: the particle resampler and its diagnostics
(``steering.py``), the torch mirror of the hidden-bound math that produces the
steering cost (``mdm/torch_features.py``), and the YAML plumbing that turns a
``feedback.uq.steering`` block into a :class:`SteeringConfig`.
"""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import numpy as np
import pytest

from uncertain_feedback.motion_generators.steering import (
    SteeringConfig,
    SteeringEvent,
    conflict_warning,
    resample_indices,
)
from uncertain_feedback.planners.mpc.config import load_mpc_config
from uncertain_feedback.simulated_users.base import (
    CoupledBound,
    FeatureCondition,
    HiddenBound,
    JointBoxLimit,
    SimulatedUser,
)
from uncertain_feedback.simulated_users.personas import get_persona


def _features(seed: int = 0, n: int = 4, t: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "elbow_flexion": rng.uniform(0.0, 2.8, size=(n, t)),
        "shoulder_elevation": rng.uniform(0.0, 3.0, size=(n, t)),
    }


def _write_uq_config(tmp_path, steering_block: str):
    path = tmp_path / "mpc.yaml"
    path.write_text(
        f"""
steps: 2
horizon: 3
n_mpc_samples: 4
max_angle_delta: 0.0025
feedback:
  uq:
    diffusion_samples: 8
{steering_block}
""",
        encoding="utf-8",
    )
    return path


# --- resampler --------------------------------------------------------------


def test_resample_indices_is_deterministic_for_a_fixed_rng() -> None:
    costs = np.linspace(0.0, 1.0, 32)

    first, first_ess = resample_indices(costs, 0.5, np.random.default_rng(3))
    second, second_ess = resample_indices(costs, 0.5, np.random.default_rng(3))

    np.testing.assert_array_equal(first, second)
    assert first_ess == second_ess


def test_resample_indices_is_the_identity_for_a_degenerate_population() -> None:
    costs = np.full(16, 0.25)

    indices, ess = resample_indices(costs, 0.5, np.random.default_rng(0))

    np.testing.assert_array_equal(indices, np.arange(16))
    assert ess == 16.0


def test_resample_indices_over_represents_low_cost_chains() -> None:
    costs = np.concatenate([np.zeros(50), np.ones(50)])

    indices, ess = resample_indices(costs, 0.5, np.random.default_rng(1))

    assert indices.shape == (100,)
    assert (indices < 50).mean() > 0.9
    assert ess < 100.0


def test_resample_indices_ess_matches_the_softmax_weights() -> None:
    costs = np.random.default_rng(4).normal(size=64)
    temperature = 0.7

    _, ess = resample_indices(costs, temperature, np.random.default_rng(0))

    logits = -(costs - costs.mean()) / (costs.std() * temperature)
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    assert ess == pytest.approx(1.0 / (weights**2).sum())


# --- bound math parity ------------------------------------------------------


@pytest.mark.parametrize(
    "bound",
    [
        HiddenBound(feature="elbow_flexion", bound_type="upper_bound", high=1.4),
        HiddenBound(feature="elbow_flexion", bound_type="lower_bound", low=0.9),
        HiddenBound(
            feature="shoulder_elevation",
            bound_type="avoid_band",
            low=0.8,
            high=1.9,
        ),
        HiddenBound(
            feature="elbow_flexion",
            bound_type="upper_bound",
            high=1.2,
            condition=FeatureCondition(feature="shoulder_elevation", low=1.0, high=2.2),
        ),
        CoupledBound(
            feature="elbow_flexion",
            bound_type="upper_bound",
            cond_feature="shoulder_elevation",
            intercept=2.6,
            slope=-0.65,
        ),
        CoupledBound(
            feature="elbow_flexion",
            bound_type="lower_bound",
            cond_feature="shoulder_elevation",
            intercept=0.4,
            slope=0.3,
        ),
    ],
)
def test_torch_bound_violation_matches_numpy(bound) -> None:
    import torch

    from uncertain_feedback.motion_generators.mdm.torch_features import (
        bound_violation,
    )

    features = _features(seed=11)
    torch_features = {
        name: torch.tensor(value, dtype=torch.float64)
        for name, value in features.items()
    }

    violation = bound_violation(bound, torch_features).numpy()

    np.testing.assert_allclose(violation, bound.violation(features), atol=1e-12)


# --- cost construction ------------------------------------------------------


def test_supported_bounds_skips_unsupported_features_and_joint_boxes() -> None:
    from uncertain_feedback.motion_generators.mdm.torch_features import (
        supported_bounds,
    )

    coupled = CoupledBound(
        feature="elbow_flexion",
        bound_type="upper_bound",
        cond_feature="shoulder_elevation",
        intercept=2.6,
        slope=-0.65,
    )
    unsupported_feature = HiddenBound(
        feature="shoulder_abduction_adduction", bound_type="upper_bound", high=1.0
    )
    unsupported_condition = HiddenBound(
        feature="elbow_flexion",
        bound_type="upper_bound",
        high=1.0,
        condition=FeatureCondition(
            feature="shoulder_internal_external_rotation", low=0.0, high=1.0
        ),
    )
    user = SimulatedUser(
        name="mixed",
        description="",
        feedback_text="",
        bounds=(coupled, unsupported_feature, unsupported_condition),
        joint_limits=(
            JointBoxLimit(
                joint="left_elbow", low=(-1.0, -1.0, -1.0), high=(1.0, 1.0, 1.0)
            ),
        ),
    )

    assert supported_bounds(user) == (coupled,)


def test_build_user_bound_cost_returns_none_without_supported_bounds() -> None:
    import torch

    from uncertain_feedback.motion_generators.mdm.torch_features import (
        build_user_bound_cost,
    )

    user = SimulatedUser(
        name="rotation_only",
        description="",
        feedback_text="",
        bounds=(
            HiddenBound(
                feature="shoulder_internal_external_rotation",
                bound_type="upper_bound",
                high=1.0,
            ),
        ),
    )
    zeros = torch.zeros(263, dtype=torch.float32)

    assert build_user_bound_cost(user, zeros, torch.ones_like(zeros)) is None


def test_build_user_bound_cost_scores_the_persona_bound_over_x0() -> None:
    import torch

    from uncertain_feedback.motion_generators.mdm.torch_features import (
        bound_violation,
        build_user_bound_cost,
        features_from_hml,
    )

    user = get_persona("triceps_long_head_contracture")
    mean = torch.zeros(263, dtype=torch.float32)
    std = torch.ones(263, dtype=torch.float32)
    x0 = torch.tensor(
        np.random.default_rng(7).normal(size=(5, 263, 1, 12)), dtype=torch.float32
    )

    cost = build_user_bound_cost(user, mean, std)
    assert cost is not None
    values = cost(x0)

    expected = (
        bound_violation(user.bounds[0], features_from_hml(x0, mean, std)) ** 2
    ).mean(dim=1)
    assert values.shape == (5,)
    assert (values >= 0.0).all()
    torch.testing.assert_close(values, expected)


def test_build_user_bound_cost_is_differentiable_wrt_x0() -> None:
    import torch

    from uncertain_feedback.motion_generators.mdm.torch_features import (
        build_user_bound_cost,
    )

    user = get_persona("triceps_long_head_contracture")
    mean = torch.zeros(263, dtype=torch.float32)
    std = torch.ones(263, dtype=torch.float32)
    x0 = torch.tensor(
        np.random.default_rng(8).normal(size=(3, 263, 1, 12)), dtype=torch.float32
    ).requires_grad_()

    cost = build_user_bound_cost(user, mean, std)
    assert cost is not None
    (grad,) = torch.autograd.grad(cost(x0).sum(), x0)

    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0.0


# --- diagnostics and config -------------------------------------------------


def test_conflict_warning_fires_when_every_sample_violates_and_ess_collapses() -> None:
    event = SteeringEvent(
        step=15, cost_mean=0.4, frac_violating=1.0, ess=2.0, unique_ancestors=7
    )

    message = conflict_warning(event, 500)

    assert message is not None
    assert "cg" in message


def test_conflict_warning_is_silent_for_a_healthy_event() -> None:
    event = SteeringEvent(
        step=15, cost_mean=0.02, frac_violating=0.2, ess=310.0, unique_ancestors=180
    )

    assert conflict_warning(event, 500) is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "guided"},
        {"temperature": 0.0},
        {"temperature": -0.5},
        {"resample_steps": (25, 15)},
        {"resample_steps": (15, 15)},
        {"resample_steps": (-1, 15)},
    ],
)
def test_steering_config_rejects_invalid_settings(kwargs) -> None:
    with pytest.raises(ValueError):
        SteeringConfig(**kwargs)


def test_uq_config_steering_defaults_to_off(tmp_path) -> None:
    cfg = load_mpc_config(_write_uq_config(tmp_path, ""))

    assert cfg.feedback is not None and cfg.feedback.uq is not None
    assert cfg.feedback.uq.steering == SteeringConfig()
    assert cfg.feedback.uq.steering.mode == "off"


def test_uq_config_parses_an_explicit_steering_block(tmp_path) -> None:
    path = _write_uq_config(
        tmp_path,
        """    steering:
      mode: cg
      resample_steps: [10, 20]
      temperature: 0.25
      guide_from: 5
      guidance_weight: 1.0e4
""",
    )

    cfg = load_mpc_config(path)

    assert cfg.feedback is not None and cfg.feedback.uq is not None
    assert cfg.feedback.uq.steering == SteeringConfig(
        mode="cg",
        resample_steps=(10, 20),
        temperature=0.25,
        guide_from=5,
        guidance_weight=1e4,
    )
