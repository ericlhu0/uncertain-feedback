"""Clean-prefix conditioning: train and sample with the pinned history frames un-noised.

MDM's training noises every frame of a clip, and sampling pins the first
``N_PREFIX_FRAMES`` by overwriting the model's clean-motion prediction there, so
inside the denoising loop the model only ever sees a *noised* history and commits
to a start of its own: a catch-up rush of 10-30 mm/frame over the first ~20
generated frames, whatever the text says (CODEBASE_MAP, "Clean-prefix
conditioning"). With a clean prefix the history frames of the training input
``x_t`` are noised to at most diffusion step ``noise_cap`` (``None`` keeps them
exactly clean), those frames are left out of the loss, the loader's crop starts
at frame 0 so window frames never slide into the prefix, and sampling presents the
pinned prefix the same way before every model call. Only clips that carry an
executed-history prefix take part (the correction sources, ``train_leftarm.py``
tells them apart by id); HumanML3D clips are noised throughout, as text-to-motion
sampling without a prefix presents them.

Reimplemented 2026-11-14 from the PC speed sweep's description, whose own patch
was unreachable; ``train_leftarm.py --clean_prefix K --prefix_noise N`` installs
it for training and ``mdm_api.MdmMotionGenerator`` for sampling, from the
checkpoint's ``args.json``. Everything here is plain tensor arithmetic so the
contract can be unit-tested without MDM.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

QSample = Callable[..., torch.Tensor]


def prefix_timesteps(t: torch.Tensor, noise_cap: int) -> torch.Tensor:
    """The step the history frames are noised to at step ``t``: ``min(t, noise_cap)``."""
    return t.clamp(max=noise_cap)


def splice_prefix(
    x_t: torch.Tensor,
    x_prefix: torch.Tensor,
    n_prefix: int,
    rows: torch.Tensor | None = None,
) -> torch.Tensor:
    """``x_t`` with its first ``n_prefix`` frames taken from ``x_prefix``.

    Both are ``(bs, joints, feats, frames)``; ``rows`` is a ``(bs,)`` bool mask of
    the clips that carry a prefix (every clip when ``None``).
    """
    out = x_t.clone()
    if rows is None:
        out[:, :, :, :n_prefix] = x_prefix[:, :, :, :n_prefix]
    else:
        out[rows, :, :, :n_prefix] = x_prefix[rows, :, :, :n_prefix]
    return out


def loss_mask_without_prefix(
    mask: torch.Tensor, n_prefix: int, rows: torch.Tensor | None = None
) -> torch.Tensor:
    """The ``(bs, 1, 1, frames)`` loss mask with the prefix frames switched off on ``rows``."""
    out = mask.clone()
    if rows is None:
        out[:, :, :, :n_prefix] = False
    else:
        out[rows, :, :, :n_prefix] = False
    return out


def noised_prefix(
    q_sample: QSample,
    x_start: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
    noise_cap: int | None,
) -> torch.Tensor:
    """``x_start`` noised to step ``min(t, noise_cap)``, or ``x_start`` itself (clean)."""
    if noise_cap is None:
        return x_start
    return q_sample(x_start, prefix_timesteps(t, noise_cap), noise=noise)


def present_prefix(
    q_sample: QSample,
    x_t: torch.Tensor,
    prefix: torch.Tensor,
    t: torch.Tensor,
    n_prefix: int,
    noise_cap: int | None,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """The sampling side: ``x_t`` with the pinned ``prefix`` presented as training did.

    Called before every model call of the denoising loop; a fresh noise draw per
    step unless ``noise`` is given.
    """
    if noise_cap is None:
        return splice_prefix(x_t, prefix, n_prefix)
    if noise is None:
        noise = torch.randn_like(prefix)
    return splice_prefix(
        x_t, noised_prefix(q_sample, prefix, t, noise, noise_cap), n_prefix
    )


class CropFromStart:
    """Stand-in for the loader's ``random`` module whose ``randint`` returns its lower bound.

    MDM's humanml loader crops every motion to a multiple of 4 frames starting at
    ``random.randint(0, slack)``, 0-7 frames on these clips and non-zero on most
    draws; from here a crop only trims the end. Everything else is the real module's.
    """

    def __init__(self, real_random: Any) -> None:
        self._real = real_random

    def randint(self, a: int, b: int) -> int:
        """The lower bound, so the crop keeps frame 0."""
        del b
        return a

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)
