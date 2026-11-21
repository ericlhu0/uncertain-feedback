"""Differentiable anatomical features and hidden-bound costs read off ``x̂0``.

Used to build the steering cost passed to
:class:`~uncertain_feedback.motion_generators.steering.SteeringSpec`: the whole
point is to score a diffusion sample *inside* the denoising loop, which rules
out the numpy IK path (``arm_aa_from_positions_batch`` costs seconds per call).
Instead the features the personas' bounds need are computed straight from
the RIC joint positions in the HML263 block, in torch, on device.

The features mirror ``arm_features.arm_feature_series`` geometrically but read
positions in the root's facing frame: elbow flexion is frame-invariant and
matches exactly, shoulder elevation is measured from world-down rather than
torso-down, and shoulder abduction and flexion/extension from the root's lateral
and forward axes rather than spine3's, so the two agree only for an unleaned,
untwisted torso. That approximation is fine for steering (it never leaves the
sampler); trajectory *evaluation* keeps using the exact IK-based oracle.

TODO(frame): shoulder abduction and flexion/extension are the features here
that are not yaw-invariant, and they are read in the root's facing frame (HML263
RIC positions are rotated so the pelvis faces +z) while ``arm_feature_series``
reads them in the spine3 frame. Any pelvis-to-spine3 twist becomes a constant
offset, so a bound on either (``cross_body_pain``, the sampled coupled bounds,
learned bounds) is steered toward a slightly displaced threshold. Fix: define the
lateral axis per frame from the RIC shoulder pair (joints 16/17, left minus
right) and measure both against it, which also makes elevation torso-relative;
then tighten ``test_torch_position_features_match_arm_feature_series`` to a
twisted ``spine3_aa``. Introduced 2026-09-11 with abduction support.

This module imports torch at module scope — import it lazily from anywhere that
must stay torch-free (config parsing, the abstract generator interface).
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch

from uncertain_feedback.motion_generators.mdm.mdm_api import _FPS, N_PREFIX_FRAMES
from uncertain_feedback.planners.mpc.costs.generated import (
    GeneratedPythonCost,
    generated_cost_feature_dependencies,
)
from uncertain_feedback.simulated_users.base import (
    Bound,
    CoupledBound,
    SimulatedUser,
)

_POSE_FEATURES = (
    "elbow_flexion",
    "shoulder_elevation",
    "shoulder_abduction_adduction",
    "shoulder_flexion_extension",
)
SUPPORTED_FEATURES = (
    *_POSE_FEATURES,
    "wrist_speed",
    "elbow_speed",
    *(f"{name}_rate" for name in _POSE_FEATURES),
)

_RIC_OFFSET = 4  # root rotation/velocity block preceding the RIC positions
_N_RIC_JOINTS = 21  # SMPL joints minus the root
# SMPL spine3 and left shoulder/elbow/wrist (9/16/18/20) shifted into the
# root-less RIC block.
_SPINE3, _L_SHOULDER, _L_ELBOW, _L_WRIST = 8, 15, 17, 19

_ACOS_EPS = 1e-6
_SPEED_EPS = 1e-12  # a norm has no gradient at zero


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


def shoulder_abduction(shoulder: torch.Tensor, elbow: torch.Tensor) -> torch.Tensor:
    """Return the upper arm's signed angle out of the sagittal plane.

    The arcsine of the upper arm's lateral component, as in
    ``arm_feature_series`` (positive = abducted, for the left arm +x).
    """
    upper = torch.nn.functional.normalize(elbow - shoulder, dim=-1)
    return torch.asin(upper[..., 0].clamp(-1.0 + _ACOS_EPS, 1.0 - _ACOS_EPS))


def shoulder_flexion(shoulder: torch.Tensor, elbow: torch.Tensor) -> torch.Tensor:
    """Return the upper arm's signed angle out of the frontal plane.

    The arcsine of the upper arm's forward component, as in
    ``arm_feature_series`` (positive = forward, +z).
    """
    upper = torch.nn.functional.normalize(elbow - shoulder, dim=-1)
    return torch.asin(upper[..., 2].clamp(-1.0 + _ACOS_EPS, 1.0 - _ACOS_EPS))


def motion_features(
    spine3: torch.Tensor,
    shoulder: torch.Tensor,
    elbow: torch.Tensor,
    wrist: torch.Tensor,
    dt: float,
) -> dict[str, torch.Tensor]:
    """Return pose and speed features for every frame after the first.

    Torch mirror of ``arm_features.arm_motion_series`` for ``(N, T, 3)`` joint
    positions ``dt`` seconds apart; every series is ``(N, T - 1)``. Speeds are
    taken relative to ``spine3``, so a swaying body does not count as a moving
    arm.
    """
    flexion, elevation = flexion_elevation(shoulder, elbow, wrist)
    pose = {
        "elbow_flexion": flexion,
        "shoulder_elevation": elevation,
        "shoulder_abduction_adduction": shoulder_abduction(shoulder, elbow),
        "shoulder_flexion_extension": shoulder_flexion(shoulder, elbow),
    }
    features = {name: values[:, 1:] for name, values in pose.items()}
    for name, joint in (("wrist_speed", wrist), ("elbow_speed", elbow)):
        step = (joint - spine3).diff(dim=1)
        features[name] = (step.pow(2).sum(-1) + _SPEED_EPS).sqrt() / dt
    for name, values in pose.items():
        features[f"{name}_rate"] = values.diff(dim=1) / dt
    return features


def features_from_hml(
    x0: torch.Tensor, hml_mean: torch.Tensor, hml_std: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Return the supported features for a normalized ``(N, 263, 1, T)`` batch.

    The frames pinned to the start pose by inpainting are dropped: the trigger
    pose the correction starts from may itself violate the user's bounds, and
    nothing the sampler does can change it. The last pinned frame is kept as
    the reference the first free frame's speed is measured from, so a jump out
    of the prefix counts as speed (:func:`build_generated_cost` leaves that
    step out of speed limits).
    """
    hml = x0[:, :, 0, :].permute(0, 2, 1)
    denorm = hml * hml_std + hml_mean
    ric = denorm[..., _RIC_OFFSET : _RIC_OFFSET + _N_RIC_JOINTS * 3].reshape(
        *denorm.shape[:2], _N_RIC_JOINTS, 3
    )
    ric = ric[:, N_PREFIX_FRAMES - 1 :]
    return motion_features(
        ric[:, :, _SPINE3],
        ric[:, :, _L_SHOULDER],
        ric[:, :, _L_ELBOW],
        ric[:, :, _L_WRIST],
        1.0 / _FPS,
    )


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


def build_generated_cost(
    costs: Sequence[GeneratedPythonCost],
    hml_mean: torch.Tensor,
    hml_std: torch.Tensor,
    speed_scale: float,
) -> Callable[[torch.Tensor], torch.Tensor] | None:
    """Compile generated limits into one steering cost over ``x̂0``.

    The limits are the planner's, evaluated on the sample's features. The
    planner's penalty (linear, worst step) has no useful gradient, so steering
    uses a smooth one of fixed scale: each margin's squared excess averaged
    over time, in rad² for path limits and, times ``speed_scale``, in (rad/s)²
    or (m/s)² for speed limits. Speed limits skip the step out of the pinned
    prefix: a correction is played re-anchored on the live pose
    (``anchor_q_trajectory``), which drops that step, and charging it pushed
    guidance against a jump that never plays, to divergence at high weights.
    A cost reading a feature positions cannot give is left out. Returns
    ``None`` when no cost is left.
    """
    steerable = [
        cost
        for cost in costs
        if set(generated_cost_feature_dependencies(cost.code))
        <= set(SUPPORTED_FEATURES)
    ]
    if len(steerable) < len(costs):
        print(
            f"steering: skipping {len(costs) - len(steerable)} learned cost(s) "
            "that read features steering cannot compute"
        )
    if not steerable:
        return None

    def cost(x0: torch.Tensor) -> torch.Tensor:
        features = features_from_hml(x0, hml_mean, hml_std)
        total = torch.zeros(x0.shape[0], dtype=x0.dtype, device=x0.device)
        for term in steerable:
            path, speed = term.margins(features)
            for margin in path.values():
                total = total + margin.clamp(min=0.0).pow(2).mean(dim=1)
            for margin in speed.values():
                excess = margin[:, 1:].clamp(min=0.0)
                total = total + speed_scale * excess.pow(2).mean(dim=1)
        return total

    return cost
