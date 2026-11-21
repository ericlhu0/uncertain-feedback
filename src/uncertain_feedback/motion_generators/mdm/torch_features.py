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

Shoulder internal/external rotation is read off the forearm: with no twist, the
forearm's component perpendicular to the upper arm points where the T-pose bend
points, carried along by the swing that took the T-pose upper-arm axis to the
current one; the twist is the angle from that reference to the actual
perpendicular, about the upper arm. It equals ``arm_feature_series``'s
swing-twist angle exactly for a T-pose collar and clavicle; the seated demo
pose's fixed collar and clavicle shift it by a median 0.04 rad (at most 0.15 rad
over random arm poses), the same kind of frame offset as the two features
above. A straight elbow leaves no perpendicular to read, so a limit on rotation
is inactive below ``ROTATION_DEAD_ZONE`` of elbow flexion; the planner's q-based
feature has no such blind spot.

This module imports torch at module scope — import it lazily from anywhere that
must stay torch-free (config parsing, the abstract generator interface).
"""

from __future__ import annotations

import functools
from typing import Callable, Sequence

import numpy as np
import torch

from uncertain_feedback.motion_generators.mdm.mdm_api import _FPS, N_PREFIX_FRAMES
from uncertain_feedback.planners.mpc.costs.generated import (
    GeneratedPythonCost,
    generated_cost_feature_dependencies,
)
from uncertain_feedback.planners.mpc.kinematics import (
    ELBOW_CHAIN_IDX,
    SHOULDER_CHAIN_IDX,
    WRIST_CHAIN_IDX,
    SmplLeftArmFK,
)
from uncertain_feedback.simulated_users.base import (
    Bound,
    CoupledBound,
    SimulatedUser,
)

_ROTATION = "shoulder_internal_external_rotation"
# Elbow flexion (radians) below which the forearm lies too close to the upper arm
# for its direction to tell the humerus twist: rotation limits are inactive there.
ROTATION_DEAD_ZONE = 0.3

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


@functools.lru_cache(maxsize=None)
def _tpose_axes() -> tuple[torch.Tensor, torch.Tensor]:
    """T-pose upper-arm axis and the forearm bend perpendicular to it (float64)."""
    tpose = SmplLeftArmFK().tpose_joints
    upper = tpose[ELBOW_CHAIN_IDX] - tpose[SHOULDER_CHAIN_IDX]
    upper = upper / np.linalg.norm(upper)
    forearm = tpose[WRIST_CHAIN_IDX] - tpose[ELBOW_CHAIN_IDX]
    bend = forearm - (forearm @ upper) * upper
    bend = bend / np.linalg.norm(bend)
    return torch.tensor(upper), torch.tensor(bend)


def shoulder_rotation(
    shoulder: torch.Tensor, elbow: torch.Tensor, wrist: torch.Tensor
) -> torch.Tensor:
    """Return the upper arm's signed twist about its own axis (internal = +).

    The swing is the shortest rotation taking the T-pose upper-arm axis onto
    the current one; it carries the T-pose bend direction to the zero-twist
    reference, and the twist is the angle from that reference to the forearm's
    actual perpendicular, about the upper arm (see the module docstring for
    what this matches and where it is blind). Rodrigues' formula is written
    with ``1 / (1 + cos)`` so the no-swing case is regular; only an arm pointing
    exactly opposite the T-pose axis is singular, and that is clamped.
    """
    axis, bend = (t.to(shoulder) for t in _tpose_axes())
    upper = torch.nn.functional.normalize(elbow - shoulder, dim=-1)
    forearm = torch.nn.functional.normalize(wrist - elbow, dim=-1)
    cos = (upper * axis).sum(-1, keepdim=True)
    swing = torch.cross(axis.expand_as(upper), upper, dim=-1)
    reference = (
        cos * bend
        + torch.cross(swing, bend.expand_as(upper), dim=-1)
        + swing
        * (swing * bend).sum(-1, keepdim=True)
        / (1.0 + cos).clamp(min=_ACOS_EPS)
    )
    perpendicular = forearm - (forearm * upper).sum(-1, keepdim=True) * upper
    return torch.atan2(
        (upper * torch.cross(reference, perpendicular, dim=-1)).sum(-1),
        (reference * perpendicular).sum(-1),
    )


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
        _ROTATION: shoulder_rotation(shoulder, elbow, wrist),
    }
    features = {name: values[:, 1:] for name, values in pose.items()}
    for name, joint in (("wrist_speed", wrist), ("elbow_speed", elbow)):
        step = (joint - spine3).diff(dim=1)
        features[name] = (step.pow(2).sum(-1) + _SPEED_EPS).sqrt() / dt
    for name, values in pose.items():
        step = values.diff(dim=1)
        if name == _ROTATION:
            step = torch.remainder(step + torch.pi, 2.0 * torch.pi) - torch.pi
        features[f"{name}_rate"] = step / dt
    return features


def features_from_hml(
    x0: torch.Tensor, hml_mean: torch.Tensor, hml_std: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Return the supported features for a normalized ``(N, 263, 1, T)`` batch.

    The frames pinned to the start pose by inpainting are dropped: the trigger
    pose the correction starts from may itself violate the user's bounds, and
    nothing the sampler does can change it. The last pinned frame is kept as
    the reference the first free frame's speed is measured from, so a jump out
    of the prefix counts as speed.
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

    Torch mirror of :meth:`HiddenBound.violation` / :meth:`CoupledBound.violation`,
    except that a bound reading shoulder rotation is inactive while the elbow is
    too straight to read it (``ROTATION_DEAD_ZONE``).
    """
    values = features[bound.feature]
    if isinstance(bound, CoupledBound):
        threshold = bound.intercept + bound.slope * features[bound.cond_feature]
        if bound.bound_type == "upper_bound":
            violation = (values - threshold).clamp(min=0.0)
        else:
            violation = (threshold - values).clamp(min=0.0)
        if _ROTATION in (bound.feature, bound.cond_feature):
            violation = _outside_dead_zone(violation, features)
        return violation

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
    if _ROTATION in (
        bound.feature,
        bound.condition.feature if bound.condition is not None else None,
    ):
        violation = _outside_dead_zone(violation, features)
    return violation


def _outside_dead_zone(
    excess: torch.Tensor, features: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Zero ``excess`` on frames whose elbow is too straight to read rotation."""
    bent = features["elbow_flexion"] >= ROTATION_DEAD_ZONE
    return torch.where(bent, excess, torch.zeros_like(excess))


def build_user_bound_cost(
    user: SimulatedUser, hml_mean: torch.Tensor, hml_std: torch.Tensor
) -> Callable[[torch.Tensor], torch.Tensor] | None:
    """Compile a persona's hidden bounds into a steering cost over ``x̂0``.

    Matches :class:`HiddenCostTerm`'s shape — the time-averaged square of the
    summed per-frame violation — without its weight: resampling is scale-free
    and classifier guidance absorbs the scale into its guidance weight.

    ``JointBoxLimit`` is structurally excluded: it acts on the raw controlled
    axis-angles, which positions alone cannot recover without IK. Returns
    ``None`` when the persona has no feature bounds.
    """
    bounds = user.bounds
    if user.joint_limits:
        skipped = ", ".join(
            f"JointBoxLimit({limit.joint})" for limit in user.joint_limits
        )
        print(f"steering cost for {user.name}: skipping unsupported terms ({skipped})")
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
    or (m/s)² for speed limits. A cost reading shoulder rotation or its rate is
    inactive, every margin of it, on frames whose elbow is too straight to read
    rotation (``ROTATION_DEAD_ZONE``). Returns ``None`` when there is no cost.
    """
    if not costs:
        return None
    reads_rotation = [
        any(
            name.startswith(_ROTATION)
            for name in generated_cost_feature_dependencies(term.code)
        )
        for term in costs
    ]

    def cost(x0: torch.Tensor) -> torch.Tensor:
        features = features_from_hml(x0, hml_mean, hml_std)
        total = torch.zeros(x0.shape[0], dtype=x0.dtype, device=x0.device)
        for term, gated in zip(costs, reads_rotation):
            for scale, margins in zip((1.0, speed_scale), term.margins(features)):
                for margin in margins.values():
                    excess = margin.clamp(min=0.0)
                    if gated:
                        excess = _outside_dead_zone(excess, features)
                    total = total + scale * excess.pow(2).mean(dim=1)
        return total

    return cost
