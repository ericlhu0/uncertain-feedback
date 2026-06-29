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


def _build_mdm(model_path: Path | None) -> MotionGenerator:
    from uncertain_feedback.motion_generators.mdm.mdm_api import (  # pylint: disable=import-outside-toplevel
        MdmMotionGenerator,
    )

    return MdmMotionGenerator(model_path=model_path)


def _build_kimodo(model_path: Path | None) -> MotionGenerator:
    from uncertain_feedback.motion_generators.kimodo.kimodo_api import (  # pylint: disable=import-outside-toplevel
        KimodoMotionGenerator,
    )

    return KimodoMotionGenerator(model_path=model_path)


MOTION_GENERATOR_BUILDERS: dict[str, Callable[[Path | None], MotionGenerator]] = {
    "mdm": _build_mdm,
    "kimodo": _build_kimodo,
}


def make_motion_generator(name: str, model_path: Path | None) -> MotionGenerator:
    """Construct the motion generator selected by ``name``."""
    if name not in MOTION_GENERATOR_BUILDERS:
        raise ValueError(
            f"Unknown motion_generator '{name}'. "
            f"Available: {sorted(MOTION_GENERATOR_BUILDERS)}"
        )
    return MOTION_GENERATOR_BUILDERS[name](model_path)
