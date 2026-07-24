"""Execution environments and their selection registry.

Mirrors the ``MOTION_GENERATOR_BUILDERS`` pattern in
``motion_generators/__init__.py``: a name → builder mapping plus a
:func:`make_env` factory. Builders import their env lazily so importing this
registry never forces sim or hardware dependencies.
"""

from __future__ import annotations

from typing import Any, Callable

from uncertain_feedback.envs.base import ExecutionEnv


def _build_kinematic(**params: Any) -> ExecutionEnv:
    from uncertain_feedback.envs.kinematic import (  # pylint: disable=import-outside-toplevel
        KinematicEnv,
    )

    return KinematicEnv(**params)


def _build_sim_robot_visual(**params: Any) -> ExecutionEnv:
    from uncertain_feedback.envs.sim_robot_visual import (  # pylint: disable=import-outside-toplevel
        SimRobotVisualEnv,
    )

    return SimRobotVisualEnv(**params)


def _build_sim_mannequin(**params: Any) -> ExecutionEnv:
    from uncertain_feedback.envs.sim_mannequin import (  # pylint: disable=import-outside-toplevel
        SimMannequinEnv,
    )

    return SimMannequinEnv(**params)


ENV_BUILDERS: dict[str, Callable[..., ExecutionEnv]] = {
    "kinematic": _build_kinematic,
    "sim_robot_visual": _build_sim_robot_visual,
    "sim_mannequin": _build_sim_mannequin,
}


def make_env(name: str, **params: Any) -> ExecutionEnv:
    """Construct the execution env selected by ``name``.

    ``params`` (the YAML ``env_params`` mapping) is forwarded to the env
    constructor; envs that take no parameters reject unknown keys.
    """
    if name not in ENV_BUILDERS:
        raise ValueError(f"Unknown env '{name}'. Available: {sorted(ENV_BUILDERS)}")
    return ENV_BUILDERS[name](**params)
