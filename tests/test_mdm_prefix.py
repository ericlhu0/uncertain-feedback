"""Tests for multi-frame inpainting prefixes in ``mdm_api``.

Covers the pure, checkpoint-free pieces behind ``MdmMotionGenerator``:
``build_inpainting_tensors`` / ``build_prefix_tensor`` (the tensor construction
inside ``_sample_hml``) and ``_resolve_total_frames`` (the additive frame
budget).
"""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from uncertain_feedback.motion_generators.mdm.mdm_api import (  # noqa: E402  pylint: disable=wrong-import-position
    _MAX_FRAMES,
    N_PREFIX_FRAMES,
    _resolve_total_frames,
    build_inpainting_tensors,
    build_prefix_tensor,
)

_N_FRAMES = 12
_N_SAMPLES = 3


def _prefix(k: int) -> "torch.Tensor":
    """(k, 263) prefix whose frame i is filled with the value i + 1."""
    return torch.arange(1, k + 1, dtype=torch.float32).unsqueeze(-1).repeat(1, 263)


def _no_body_mask() -> np.ndarray:
    return np.zeros(263, dtype=bool)


class TestBuildInpaintingTensors:
    """Shape, pinning and continuation behavior of the inpainting inputs."""

    def test_shapes(self) -> None:
        motion, mask = build_inpainting_tensors(
            torch, _prefix(4), 4, _N_FRAMES, _N_SAMPLES, _no_body_mask()
        )
        assert motion.shape == (_N_SAMPLES, 263, 1, _N_FRAMES)
        assert mask.shape == (_N_SAMPLES, 263, 1, _N_FRAMES)
        assert mask.dtype == torch.bool

    def test_prefix_frames_are_pinned_in_order(self) -> None:
        k = 4
        motion, _ = build_inpainting_tensors(
            torch, _prefix(k), k, _N_FRAMES, _N_SAMPLES, _no_body_mask()
        )
        # Frame i of every sample must hold prefix frame i, oldest first.
        for i in range(k):
            assert torch.allclose(
                motion[:, :, 0, i], torch.full((_N_SAMPLES, 263), float(i + 1))
            )

    def test_frames_past_prefix_hold_last_prefix_frame(self) -> None:
        k = 4
        motion, _ = build_inpainting_tensors(
            torch, _prefix(k), k, _N_FRAMES, _N_SAMPLES, _no_body_mask()
        )
        tail = motion[:, :, 0, k:]
        assert torch.allclose(tail, torch.full_like(tail, float(k)))

    def test_mask_locks_exactly_the_prefix(self) -> None:
        k = 4
        _, mask = build_inpainting_tensors(
            torch, _prefix(k), k, _N_FRAMES, _N_SAMPLES, _no_body_mask()
        )
        assert mask[..., :k].all()
        assert not mask[..., k:].any()

    def test_body_mask_freezes_channels_past_the_prefix(self) -> None:
        k = 3
        body_mask = np.zeros(263, dtype=bool)
        body_mask[:100] = True
        _, mask = build_inpainting_tensors(
            torch, _prefix(k), k, _N_FRAMES, _N_SAMPLES, body_mask
        )
        assert mask[:, :100, :, k:].all()
        assert not mask[:, 100:, :, k:].any()
        assert mask[..., :k].all()

    def test_single_frame_matches_default_prefix_length(self) -> None:
        """A (1, 263) prefix reproduces today's single pinned frame."""
        motion, mask = build_inpainting_tensors(
            torch, _prefix(1), N_PREFIX_FRAMES, _N_FRAMES, _N_SAMPLES, _no_body_mask()
        )
        assert torch.allclose(motion, torch.ones_like(motion))
        assert mask[..., :N_PREFIX_FRAMES].all()
        assert not mask[..., N_PREFIX_FRAMES:].any()


class TestBuildPrefixTensor:
    """Expansion of a single pose and validation of an explicit prefix."""

    def test_flat_pose_expands_to_static_prefix(self) -> None:
        pose = np.arange(263, dtype=np.float64)
        prefix = build_prefix_tensor(torch, pose, device="cpu")
        assert prefix.shape == (N_PREFIX_FRAMES, 263)
        assert torch.allclose(prefix, prefix[0].expand_as(prefix))

    def test_flat_pose_builds_the_same_tensors_as_a_static_prefix(self) -> None:
        """A (263,) pose and the same pose repeated K times are equivalent."""
        pose = np.arange(263, dtype=np.float64)
        expanded = build_prefix_tensor(torch, pose, device="cpu")
        explicit = build_prefix_tensor(
            torch, np.repeat(pose[np.newaxis, :], N_PREFIX_FRAMES, axis=0), device="cpu"
        )
        args = (N_PREFIX_FRAMES, _N_FRAMES, _N_SAMPLES, _no_body_mask())
        motion_a, mask_a = build_inpainting_tensors(torch, expanded, *args)
        motion_b, mask_b = build_inpainting_tensors(torch, explicit, *args)
        assert torch.equal(motion_a, motion_b)
        assert torch.equal(mask_a, mask_b)

    def test_wrong_prefix_length_raises(self) -> None:
        bad = np.zeros((N_PREFIX_FRAMES + 1, 263))
        with pytest.raises(ValueError, match="must have"):
            build_prefix_tensor(torch, bad, device="cpu")


class TestResolveTotalFrames:
    """The prefix is additive to the requested output length."""

    def test_prefix_is_added_to_the_request(self) -> None:
        assert _resolve_total_frames(0.0, 50) == 50 + N_PREFIX_FRAMES - 1

    def test_seconds_are_converted_then_extended(self) -> None:
        assert _resolve_total_frames(2.5, None) == 50 + N_PREFIX_FRAMES - 1

    def test_over_budget_request_raises(self) -> None:
        with pytest.raises(ValueError, match="pinned frames"):
            _resolve_total_frames(0.0, _MAX_FRAMES - N_PREFIX_FRAMES + 2)
