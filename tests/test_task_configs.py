"""Config ``base:`` composition and the evaluation's care-task configs."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from uncertain_feedback.planners.mpc.config import load_mpc_config
from uncertain_feedback.planners.mpc.goal_spaces import DisplacementRegion

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_base_sections_merge_in_order(tmp_path) -> None:
    learning = tmp_path / "learning.yaml"
    learning.write_text("steps: 1\nsimulated_user:\n  nominal_steps: 40\n")
    task = tmp_path / "task.yaml"
    task.write_text(
        "steps: 2\nhorizon: 3\nn_mpc_samples: 4\nmax_angle_delta: 0.0025\n"
        "cartesian:\n  goals:\n    - [0.1, 0.2, 0.3]\n"
    )
    combined = tmp_path / "combined.yaml"
    combined.write_text(f"base:\n  - {learning}\n  - {task}\nhorizon: 9\n")
    cfg = load_mpc_config(combined)
    assert cfg.steps == 2  # a later base replaces an earlier one
    assert cfg.horizon == 9  # the file's own sections replace every base's
    assert cfg.simulated_user.nominal_steps == 40
    assert cfg.cartesian is not None
    assert cfg.cartesian.goals == [[0.1, 0.2, 0.3]]


def test_conf_folders_do_not_shadow_run_options() -> None:
    # Hydra reads every folder under evaluation/conf as a config group, so a
    # folder named like a run option (e.g. tasks/) breaks `tasks=0` overrides.
    conf = _REPO_ROOT / "evaluation/conf"
    options = set(yaml.safe_load((conf / "config.yaml").read_text())) - {
        "defaults",
        "hydra",
    }
    folders = {path.name for path in conf.iterdir() if path.is_dir()}
    assert not options & folders


def test_base_cycle_is_rejected(tmp_path) -> None:
    first, second = tmp_path / "first.yaml", tmp_path / "second.yaml"
    first.write_text(f"base: {second}\n")
    second.write_text(f"base: {first}\n")
    with pytest.raises(ValueError, match="cycle"):
        load_mpc_config(first)


@pytest.mark.parametrize(
    "task",
    [
        "bathing",
        "transfer",
        "dressing_first_sleeve",
        "dressing_second_sleeve",
        "pressure_sore_relief",
    ],
)
def test_task_configs_add_the_learning_stack_to_the_planner_task(task: str) -> None:
    cfg = load_mpc_config(_REPO_ROOT / "evaluation/conf/care_tasks" / f"{task}.yaml")
    planner = load_mpc_config(
        _REPO_ROOT / "src/uncertain_feedback/planners/mpc/configs" / f"{task}.yaml"
    )
    # Seated only: MDM's direction words were trained for a sitting body.
    assert cfg.pose == Path(
        "src/uncertain_feedback/motion_generators/mdm/transfer_seated_pose.pt"
    )
    assert cfg.feedback is not None and cfg.feedback.uq is not None
    assert cfg.llm_cost.enabled and cfg.llm_cost.model
    assert cfg.simulated_user.nominal_steps == 40
    assert cfg.cartesian is not None and planner.cartesian is not None
    assert repr(cfg.cartesian) == repr(planner.cartesian)
    assert (cfg.pose, cfg.costs, cfg.max_angle_delta) == (
        planner.pose,
        planner.costs,
        planner.max_angle_delta,
    )


def test_pressure_sore_goal_is_a_start_relative_elbow_displacement() -> None:
    cfg = load_mpc_config(
        _REPO_ROOT
        / "src/uncertain_feedback/planners/mpc/configs/pressure_sore_relief.yaml"
    )
    assert cfg.cartesian is not None
    assert cfg.cartesian.goals == [DisplacementRegion(joint="elbow", min_distance=0.10)]
