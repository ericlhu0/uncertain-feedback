"""Bound-transfer harness: resuming a learned state, raw draws, goal stalls."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from evaluation.approaches import Approach, BridgePotentialFieldGrounder, NoCostGen
from evaluation.approaches.base import LEARNED_STATE_FILE
from evaluation.benchmarks import transfer as transfer_module
from evaluation.benchmarks.episode import with_goal_stall
from evaluation.benchmarks.structs import InteractionTask
from evaluation.benchmarks.transfer import probe_menu, run_transfer
from evaluation.metrics.grounding.structs import GroundingResult
from uncertain_feedback.planners.mpc.config import load_mpc_config
from uncertain_feedback.planners.mpc.goal_spaces import GoalStallConfig, PointRegion
from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.simulated_users import get_persona

_SMOKE_MPC = (
    Path(__file__).resolve().parents[1] / "evaluation" / "conf" / "mpc_smoke.yaml"
)
_GOAL = PointRegion((-0.18, 0.40, 0.34))


def _task() -> InteractionTask:
    return InteractionTask(
        persona="elbow_contracture",
        verbalizer="joint_resolved",
        goals=(_GOAL, _GOAL),
        max_rounds=1,
        seed=0,
    )


class _SamplingGrounder(BridgePotentialFieldGrounder):
    """The bridge baseline's menu, reported as if drawn from a generator."""

    def ground(self, *args, **kwargs) -> GroundingResult:  # type: ignore[no-untyped-def]
        result = super().ground(*args, **kwargs)
        return replace(
            result,
            samples=np.ones((4, 5, 22, 3)),
            sample_labels=np.array([0, 0, 1, 1]),
        )


def test_resumed_transfer_probes_from_the_earlier_runs_learned_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No learning goal is played; the probes start where that run's oracle ended."""
    cfg = replace(load_mpc_config(_SMOKE_MPC), seed=0)
    human = Human(pose=cfg.pose, arm=cfg.arm)
    user = get_persona("elbow_contracture")
    earlier = tmp_path / "earlier" / "task_00"
    first = Approach("full", BridgePotentialFieldGrounder(), NoCostGen())
    first.reset(cfg, human, None, user, 0, earlier / "learn")
    first.grounder._history = ["user: keep my elbow down", "robot: ok"]
    (earlier / "learn" / "goal_00").mkdir(parents=True)
    first.save(earlier / "learn" / LEARNED_STATE_FILE)
    oracle_path = np.linspace(0.0, 1.0, 21).reshape(3, 7)
    np.save(earlier / "learn" / "goal_00" / "oracle_path.npy", oracle_path)

    def no_learning(*_args: object) -> None:
        raise AssertionError("a resumed task must not learn again")

    starts: list[np.ndarray] = []

    def probe(*args: object) -> tuple[np.ndarray, None]:
        q_start = np.asarray(args[6])
        starts.append(q_start)
        return q_start[None], None

    monkeypatch.setattr(transfer_module, "run_episode", no_learning)
    monkeypatch.setattr(transfer_module, "probe_menu", probe)
    later = Approach("full", BridgePotentialFieldGrounder(), NoCostGen())
    later_dir = tmp_path / "later" / "task_00"
    later.reset(cfg, human, None, user, 0, later_dir)

    assert not run_transfer(cfg, human, user, _task(), later, later_dir, earlier)
    assert len(starts) == 1
    np.testing.assert_array_equal(starts[0], oracle_path[-1])
    assert later.grounder._history == first.grounder._history
    assert (later_dir / "probes.pkl").exists()


def test_probe_saves_the_raw_draws(tmp_path: Path) -> None:
    """``menu.npz`` keeps the generator's samples and labels next to the menu."""
    cfg = replace(load_mpc_config(_SMOKE_MPC), seed=0)
    human = Human(pose=cfg.pose, arm=cfg.arm)
    user = get_persona("elbow_contracture")
    approach = Approach("sampling", _SamplingGrounder(), NoCostGen())
    approach.reset(cfg, human, None, user, 0, tmp_path)

    _, probe = probe_menu(
        cfg, human, user, _task(), approach, 1, human.q, tmp_path / "probe_01"
    )

    assert probe is not None
    menu = np.load(tmp_path / "probe_01" / "menu.npz")
    assert menu["samples"].dtype == np.float32
    assert menu["samples"].shape == (4, 5, 22, 3)
    np.testing.assert_array_equal(menu["sample_labels"], [0, 0, 1, 1])
    assert {f"candidate_{label}" for label in probe.candidates} <= set(menu.files)


def test_goal_stall_switch() -> None:
    """On keeps a config's own stall settings or adds the defaults; off drops them."""
    cfg = load_mpc_config(_SMOKE_MPC)
    assert cfg.cartesian is not None and cfg.cartesian.stall is None
    defaulted = with_goal_stall(cfg, True).cartesian
    assert defaulted is not None and defaulted.stall == GoalStallConfig()
    own = replace(
        cfg, cartesian=replace(cfg.cartesian, stall=GoalStallConfig(window=5))
    )
    kept = with_goal_stall(own, True).cartesian
    assert kept is not None and kept.stall == GoalStallConfig(window=5)
    dropped = with_goal_stall(own, False).cartesian
    assert dropped is not None and dropped.stall is None
