"""The clean-prefix contract: what the training input, the loss and the sampler see."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import random

import torch

from uncertain_feedback.motion_generators.mdm.clean_prefix import (
    CropFromStart,
    loss_mask_without_prefix,
    noised_prefix,
    prefix_timesteps,
    present_prefix,
    splice_prefix,
)

_K = 3


def _batch(fill: float) -> torch.Tensor:
    """(4 clips, 2 joints, 1 feat, 6 frames), frame f holding fill + f."""
    return (
        torch.arange(6, dtype=torch.float32).view(1, 1, 1, 6).repeat(4, 2, 1, 1) + fill
    )


def test_splice_prefix_replaces_only_the_prefix_frames_of_the_chosen_rows() -> None:
    noisy, clean = _batch(100.0), _batch(0.0)
    rows = torch.tensor([True, False, True, False])

    out = splice_prefix(noisy, clean, _K, rows)

    assert torch.equal(out[rows, :, :, :_K], clean[rows, :, :, :_K])
    assert torch.equal(out[rows, :, :, _K:], noisy[rows, :, :, _K:])
    assert torch.equal(out[~rows], noisy[~rows])
    assert torch.equal(noisy, _batch(100.0))  # untouched: the caller's tensor
    assert torch.equal(
        splice_prefix(noisy, clean, _K)[:, :, :, :_K], clean[:, :, :, :_K]
    )


def test_loss_mask_drops_the_prefix_frames_of_the_chosen_rows() -> None:
    mask = torch.ones(4, 1, 1, 6, dtype=torch.bool)
    mask[:, :, :, 5] = False  # a padded frame stays off
    rows = torch.tensor([False, True, False, True])

    out = loss_mask_without_prefix(mask, _K, rows)

    assert not out[rows, :, :, :_K].any()
    assert out[rows, :, :, _K:5].all()
    assert torch.equal(out[~rows], mask[~rows])
    assert not out[:, :, :, 5].any()
    assert not loss_mask_without_prefix(mask, _K)[:, :, :, :_K].any()


def test_prefix_timesteps_cap_the_step() -> None:
    t = torch.tensor([0, 10, 25, 49])
    assert prefix_timesteps(t, 25).tolist() == [0, 10, 25, 25]


def test_noised_prefix_is_clean_without_a_cap_and_capped_with_one() -> None:
    calls: list[torch.Tensor] = []

    def q_sample(x_start: torch.Tensor, t: torch.Tensor, noise=None) -> torch.Tensor:
        calls.append(t)
        return x_start + t.view(-1, 1, 1, 1).float() + noise

    x_start, noise = _batch(0.0), torch.zeros(4, 2, 1, 6)
    t = torch.tensor([3, 30, 49, 0])

    assert noised_prefix(q_sample, x_start, t, noise, None) is x_start
    assert not calls
    out = noised_prefix(q_sample, x_start, t, noise, 25)
    assert calls[0].tolist() == [3, 25, 25, 0]
    assert torch.equal(
        out, x_start + torch.tensor([3.0, 25.0, 25.0, 0.0]).view(-1, 1, 1, 1)
    )


def test_present_prefix_feeds_the_sampler_the_prefix_at_the_capped_step() -> None:
    seen: list[torch.Tensor] = []

    def q_sample(x_start: torch.Tensor, t: torch.Tensor, noise=None) -> torch.Tensor:
        seen.append(t)
        return x_start + noise

    x_t, prefix = _batch(100.0), _batch(0.0)
    t = torch.tensor([40] * 4)
    torch.manual_seed(0)

    out = present_prefix(q_sample, x_t, prefix, t, _K, 25)

    assert seen[0].tolist() == [25] * 4
    assert torch.equal(out[:, :, :, _K:], x_t[:, :, :, _K:])
    assert not torch.equal(
        out[:, :, :, :_K], prefix[:, :, :, :_K]
    )  # noised, fresh draw
    clean = present_prefix(q_sample, x_t, prefix, t, _K, None)
    assert torch.equal(clean[:, :, :, :_K], prefix[:, :, :, :_K])
    assert len(seen) == 1


def test_crop_from_start_keeps_frame_zero_and_delegates_the_rest() -> None:
    stand_in = CropFromStart(random)
    assert stand_in.randint(0, 7) == 0
    assert stand_in.randint(2, 7) == 2
    assert stand_in.choice([4]) == 4
    assert stand_in.random is random.random
