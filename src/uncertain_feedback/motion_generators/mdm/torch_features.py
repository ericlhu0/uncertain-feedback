"""Differentiable anatomical features and hidden-bound costs read off ``x̂0``.

Used to build the steering cost passed to
:class:`~uncertain_feedback.motion_generators.steering.SteeringSpec`: the whole
point is to score a diffusion sample *inside* the denoising loop, which rules
out the numpy IK path (``arm_aa_from_positions_batch`` costs seconds per call).
Instead the two features the personas' bounds need are computed straight from
the RIC joint positions in the HML263 block, in torch, on device.

The features mirror ``arm_features.arm_feature_series`` geometrically but read
world-frame positions: elbow flexion is frame-invariant and matches exactly,
while shoulder elevation is measured from world-down rather than torso-down, so
the two agree only for an unleaned torso. That approximation is fine for
steering (it never leaves the sampler); trajectory *evaluation* keeps using the
exact IK-based oracle.

This module imports torch at module scope — import it lazily from anywhere that
must stay torch-free (config parsing, the abstract generator interface).
"""

from __future__ import annotations

from typing import Callable

import torch

from uncertain_feedback.motion_generators.mdm.mdm_api import N_PREFIX_FRAMES
from uncertain_feedback.simulated_users.base import (
    Bound,
    CoupledBound,
    SimulatedUser,
)

SUPPORTED_FEATURES = ("elbow_flexion", "shoulder_elevation")

_RIC_OFFSET = 4  # root rotation/velocity block preceding the RIC positions
_N_RIC_JOINTS = 21  # SMPL joints minus the root
# SMPL left shoulder/elbow/wrist (16/18/20) shifted into the root-less RIC block.
_L_SHOULDER, _L_ELBOW, _L_WRIST = 15, 17, 19

_ACOS_EPS = 1e-6


def flexion_elevation(
    shoulder: torch.Tensor, elbow: torch.Tensor, wrist: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(elbow_flexion, shoulder_elevation)`` for arm joint positions.

    Flexion is the angle between the upper arm and the forearm (0 = straight);
    elevation is the upper arm's angle from world-down, and so is yaw-invariant.
    The dot products are clamped inside ``±1`` because ``acos`` has infinite
    gradient at the endpoints.
    """
    upper = torch.nn.functional.normalize(elbow - shoulder, dim=-1)
    forearm = torch.nn.functional.normalize(wrist - elbow, dim=-1)
    flexion = torch.acos(
        (upper * forearm).sum(-1).clamp(-1.0 + _ACOS_EPS, 1.0 - _ACOS_EPS)
    )
    elevation = torch.acos((-upper[..., 1]).clamp(-1.0 + _ACOS_EPS, 1.0 - _ACOS_EPS))
    return flexion, elevation


def features_from_hml(
    x0: torch.Tensor, hml_mean: torch.Tensor, hml_std: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Return the supported features for a normalized ``(N, 263, 1, T)`` batch.

    The frames pinned to the start pose by inpainting are dropped: the trigger
    pose the correction starts from may itself violate the user's bounds, and
    nothing the sampler does can change it.
    """
    hml = x0[:, :, 0, :].permute(0, 2, 1)
    denorm = hml * hml_std + hml_mean
    ric = denorm[..., _RIC_OFFSET : _RIC_OFFSET + _N_RIC_JOINTS * 3].reshape(
        *denorm.shape[:2], _N_RIC_JOINTS, 3
    )
    ric = ric[:, N_PREFIX_FRAMES:]
    flexion, elevation = flexion_elevation(
        ric[:, :, _L_SHOULDER], ric[:, :, _L_ELBOW], ric[:, :, _L_WRIST]
    )
    return {"elbow_flexion": flexion, "shoulder_elevation": elevation}


def bound_violation(bound: Bound, features: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return per-frame violation magnitudes (radians), zero when satisfied.

    Torch mirror of :meth:`HiddenBound.violation` / :meth:`CoupledBound.violation`.
    """
    values = features[bound.feature]
    if isinstance(bound, CoupledBound):
        threshold = bound.intercept + bound.slope * features[bound.cond_feature]
        if bound.bound_type == "upper_bound":
            return (values - threshold).clamp(min=0.0)
        return (threshold - values).clamp(min=0.0)

    if bound.bound_type == "upper_bound":
        violation = (values - float(bound.high)).clamp(min=0.0)  # type: ignore[arg-type]
    elif bound.bound_type == "lower_bound":
        violation = (float(bound.low) - values).clamp(min=0.0)  # type: ignore[arg-type]
    else:
        violation = torch.minimum(
            values - float(bound.low),  # type: ignore[arg-type]
            float(bound.high) - values,  # type: ignore[arg-type]
        ).clamp(min=0.0)
    if bound.condition is not None:
        cond = features[bound.condition.feature]
        active = (cond >= bound.condition.low) & (cond <= bound.condition.high)
        violation = torch.where(active, violation, torch.zeros_like(violation))
    return violation


def supported_bounds(user: SimulatedUser) -> tuple[Bound, ...]:
    """Return the persona's bounds this module can score from positions.

    ``JointBoxLimit`` is structurally excluded: it acts on the raw controlled
    axis-angles, which positions alone cannot recover without IK.
    """
    return tuple(bound for bound in user.bounds if _is_supported(bound))


def _is_supported(bound: Bound) -> bool:
    features = [bound.feature]
    if isinstance(bound, CoupledBound):
        features.append(bound.cond_feature)
    elif bound.condition is not None:
        features.append(bound.condition.feature)
    return all(feature in SUPPORTED_FEATURES for feature in features)


def build_user_bound_cost(
    user: SimulatedUser, hml_mean: torch.Tensor, hml_std: torch.Tensor
) -> Callable[[torch.Tensor], torch.Tensor] | None:
    """Compile a persona's hidden bounds into a steering cost over ``x̂0``.

    Matches :class:`HiddenCostTerm`'s shape — the time-averaged square of the
    summed per-frame violation — without its weight: resampling is scale-free
    and classifier guidance absorbs the scale into its guidance weight.

    Returns ``None`` when the persona has no bounds this module can score.
    """
    bounds = supported_bounds(user)
    skipped = [
        f"{type(bound).__name__}({bound.feature})"
        for bound in user.bounds
        if not _is_supported(bound)
    ] + [f"JointBoxLimit({limit.joint})" for limit in user.joint_limits]
    if skipped:
        print(
            f"steering cost for {user.name}: skipping unsupported terms "
            f"({', '.join(skipped)})"
        )
    if not bounds:
        return None

    def cost(x0: torch.Tensor) -> torch.Tensor:
        features = features_from_hml(x0, hml_mean, hml_std)
        total = torch.stack([bound_violation(bound, features) for bound in bounds]).sum(
            dim=0
        )
        return (total**2).mean(dim=1)

    return cost
