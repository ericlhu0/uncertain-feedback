"""Motion-generation backends and their selection registry.

Mirrors the ``COST_BUILDERS`` pattern in
``planners/mpc/costs/base.py``: a name → builder mapping plus a
:func:`make_motion_generator` factory. Builders import their backend lazily so
importing this registry never forces the heavy MDM / kimodo dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from uncertain_feedback.motion_generators.base import MotionGenerator


def _build_mdm(
    model_path: Path | None,
    num_denoising_steps: int | None,
    seed: int | None,
    lock_seed: bool,
) -> MotionGenerator:
    from uncertain_feedback.motion_generators.mdm.mdm_api import (  # pylint: disable=import-outside-toplevel
        MdmMotionGenerator,
    )

    del num_denoising_steps
    return MdmMotionGenerator(
        model_path=model_path,
        seed=10 if seed is None else seed,
        lock_seed=lock_seed,
    )


def _build_kimodo(
    model_path: Path | None,
    num_denoising_steps: int | None,
    seed: int | None,
    lock_seed: bool,
) -> MotionGenerator:
    from uncertain_feedback.motion_generators.kimodo.kimodo_api import (  # pylint: disable=import-outside-toplevel
        KimodoMotionGenerator,
    )

    del seed, lock_seed
    if num_denoising_steps is None:
        return KimodoMotionGenerator(model_path=model_path)
    return KimodoMotionGenerator(
        model_path=model_path,
        num_denoising_steps=num_denoising_steps,
    )


MOTION_GENERATOR_BUILDERS: dict[
    str, Callable[[Path | None, int | None, int | None, bool], MotionGenerator]
] = {
    "mdm": _build_mdm,
    "kimodo": _build_kimodo,
}


def make_motion_generator(
    name: str,
    model_path: Path | None,
    num_denoising_steps: int | None = None,
    seed: int | None = None,
    lock_seed: bool = False,
) -> MotionGenerator:
    """Construct the motion generator selected by ``name``.

    Backend-specific options are forwarded only to backends that support them.
    MDM uses ``seed`` and can reset it before every generation when
    ``lock_seed`` is true; kimodo uses ``num_denoising_steps``.
    """
    if name not in MOTION_GENERATOR_BUILDERS:
        raise ValueError(
            f"Unknown motion_generator '{name}'. "
            f"Available: {sorted(MOTION_GENERATOR_BUILDERS)}"
        )
    return MOTION_GENERATOR_BUILDERS[name](
        model_path, num_denoising_steps, seed, lock_seed
    )
