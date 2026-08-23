"""Every approach yaml instantiates, and invalid axis compositions fail fast."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from evaluation.approaches import (
    Approach,
    ClassifierGuidanceSteering,
    ConsolidateCostGen,
    ImmediateCostGen,
    MdmGrounder,
    NoCostGen,
    NominalGrounder,
    NoSteering,
    ParameterizedEditGrounder,
)
from evaluation.benchmarks.base import InteractionBenchmark
from evaluation.episode import run_episode
from evaluation.rig import build_rig
from uncertain_feedback.simulated_users import get_persona

_APPROACH_DIR = (
    Path(__file__).resolve().parents[1] / "evaluation" / "conf" / "approach"
)
_SMOKE_MPC = _APPROACH_DIR.parent / "mpc_smoke.yaml"
_NAMES = sorted(path.stem for path in _APPROACH_DIR.glob("*.yaml"))


def _instantiate(name: str, overrides: list[str] | None = None) -> Approach:
    with initialize_config_dir(config_dir=str(_APPROACH_DIR), version_base=None):
        cfg = compose(config_name=name, overrides=overrides or [])
    approach = instantiate(cfg)
    assert isinstance(approach, Approach)
    return approach


@pytest.mark.parametrize("name", _NAMES)
def test_every_approach_yaml_instantiates(name: str) -> None:
    approach = _instantiate(name)
    assert approach.name == name


def test_axis_overrides_compose_from_any_approach() -> None:
    approach = _instantiate(
        "edit_baseline", overrides=["grounder=none", "cost_gen=language_only"]
    )
    assert isinstance(approach.grounder, NominalGrounder)
    assert isinstance(approach.cost_gen, ImmediateCostGen)
    assert approach.cost_gen.source == "nominal"


def test_full_composes_mdm_consolidate_cg() -> None:
    approach = _instantiate("full")
    assert isinstance(approach.grounder, MdmGrounder)
    assert isinstance(approach.cost_gen, ConsolidateCostGen)
    assert isinstance(approach.steering, ClassifierGuidanceSteering)
    assert approach.grounder.steering is approach.steering
    assert approach.requires_generator


def test_steering_axis_override_disables_cg() -> None:
    approach = _instantiate("full", overrides=["steering=none"])
    assert isinstance(approach.steering, NoSteering)
    assert approach.grounder.steering is approach.steering


def test_steering_requires_the_mdm_grounder() -> None:
    with pytest.raises(ValueError, match="mdm grounder"):
        Approach(
            name="bad",
            grounder=ParameterizedEditGrounder(),
            cost_gen=NoCostGen(),
            steering=ClassifierGuidanceSteering(),
        )


def test_nominal_grounder_rejects_chosen_source() -> None:
    with pytest.raises(ValueError, match="nominal"):
        Approach(
            name="bad",
            grounder=NominalGrounder(),
            cost_gen=ImmediateCostGen(source="chosen"),
        )


def test_nominal_grounder_episode_smoke(tmp_path: Path) -> None:
    rig = build_rig(_SMOKE_MPC, seed=0, load_generator=False)
    user = get_persona("elbow_contracture")
    bench = InteractionBenchmark(
        name="smoke",
        personas=["elbow_contracture"],
        verbalizers=["joint_resolved"],
        goals=[[-0.18, 0.40, 0.34]],
        max_rounds=1,
    )
    task = bench.generate_tasks(0, rig.cfg)[0]
    approach = Approach(
        name="cost_only", grounder=NominalGrounder(), cost_gen=NoCostGen()
    )
    approach.reset(rig, user, task, tmp_path / "episode")
    result = run_episode(rig, user, task, approach, tmp_path / "episode")
    assert (tmp_path / "episode" / "episode_summary.json").exists()
    assert result["summary"]["goal_results"], "episode recorded no goal results"
    rows = result["rows"]
    assert all(row["n_candidates"] == 1 for row in rows)
