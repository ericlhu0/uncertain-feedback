"""The MDM grounder's menu is judged exactly as the chosen option is tracked."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from evaluation.approaches.grounders.mdm import MdmGrounder
from uncertain_feedback.planners.mpc.config import load_mpc_config
from uncertain_feedback.planners.mpc.feedback.mdm import FeedbackConfig
from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.planners.mpc.kinematics import LEFT_ARM_CHAIN_INDICES
from uncertain_feedback.simulated_users import get_persona
from uncertain_feedback.uncertainty import UqConfig
from uncertain_feedback.uncertainty.cluster_picker import scale_trajectory

_SMOKE_MPC = (
    Path(__file__).resolve().parents[1] / "evaluation" / "conf" / "mpc_smoke.yaml"
)
_FRAMES = 6


class _SeamGenerator:
    """Fake MDM: every sample echoes the current arm at frame 0, then jumps away."""

    def __init__(self, samples: np.ndarray) -> None:
        self.samples = samples

    def generate_positions(self, text: str, human: Human, **_kwargs: Any) -> np.ndarray:
        del text, human
        return self.samples


def _samples(human: Human) -> np.ndarray:
    """Two groups of motions, each with a large frame-0 to frame-1 seam."""
    seam = np.array([0.0, 0.0, 0.0, 0.15, 0.1, -0.1, 0.2])
    samples = []
    for sign in (1.0, 1.0, -1.0, -1.0):
        q = np.repeat(human.q[np.newaxis], _FRAMES, axis=0)
        ramp = np.arange(1, _FRAMES)[:, np.newaxis]
        q[1:] += seam + ramp * sign * np.array([0.0, 0.0, 0.0, 0.08, 0.0, 0.05, 0.06])
        positions = np.repeat(human.posture[np.newaxis], _FRAMES, axis=0)
        positions[:, LEFT_ARM_CHAIN_INDICES] = human.fk_positions_from_q(q)
        samples.append(positions)
    return np.stack(samples)


def test_the_chosen_option_is_tracked_as_the_person_judged_it(tmp_path: Path) -> None:
    base = Human()
    q = base.q
    q[3:6] += [0.2, -0.1, 0.3]
    q[6] += 1.0
    human = base.reset_human_with_q(q)
    cfg = replace(
        load_mpc_config(_SMOKE_MPC),
        feedback=FeedbackConfig(
            frames=_FRAMES, uq=UqConfig(diffusion_samples=4, n_clusters=2)
        ),
    )
    grounder = MdmGrounder()
    gen = _SeamGenerator(_samples(human))
    grounder.reset(cfg, cast(Any, gen), get_persona("elbow_contracture"), 0, tmp_path)
    judged: dict[int, np.ndarray] = {}

    def select(menu: dict[int, np.ndarray]) -> tuple[int, float]:
        judged.update(menu)
        return min(menu), 1.5

    result = grounder.ground("raise it", human, human.q[np.newaxis], select)

    assert len(judged) == 2
    for option in judged.values():
        q_option = human.q_from_arm_aa(option)
        steps = np.linalg.norm(np.diff(q_option, axis=0), axis=1)
        np.testing.assert_allclose(q_option[0], human.q, atol=1e-9)
        np.testing.assert_allclose(steps, steps[0], rtol=1e-6)
    assert result.candidates.keys() == judged.keys()
    np.testing.assert_allclose(
        result.correction_traj, scale_trajectory(judged[result.chosen_label], 1.5)
    )
