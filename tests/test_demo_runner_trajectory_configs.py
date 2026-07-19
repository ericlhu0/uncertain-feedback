"""Tests for saving and reloading named initial poses and goals."""

# pylint: disable=missing-function-docstring

import json
from types import SimpleNamespace

import numpy as np

from uncertain_feedback.demo_runner import server
from uncertain_feedback.demo_runner.core import DemoRig


def _rig(tmp_path) -> DemoRig:
    rig = DemoRig.__new__(DemoRig)
    rig.trajectory_configs_path = tmp_path / "trajectory_configs.json"
    rig.trajectory_configs = {"initial_poses": [], "goals": []}
    return rig


def test_named_trajectory_configs_persist_and_update(tmp_path) -> None:
    rig = _rig(tmp_path)
    first_pose = np.arange(9, dtype=np.float64).reshape(3, 3)

    rig.upsert_trajectory_config(
        "initial_poses", {"name": "ready", "arm_aa": first_pose.tolist()}
    )
    rig.upsert_trajectory_config("goals", {"name": "high", "goal": [0.1, 0.5, -0.2]})
    rig.upsert_trajectory_config(
        "initial_poses",
        {"name": "ready", "arm_aa": np.ones((3, 3)).tolist()},
    )

    saved = json.loads(rig.trajectory_configs_path.read_text(encoding="utf-8"))
    assert saved == {
        "initial_poses": [{"name": "ready", "arm_aa": np.ones((3, 3)).tolist()}],
        "goals": [{"name": "high", "goal": [0.1, 0.5, -0.2]}],
    }

    loaded = _rig(tmp_path)
    loaded._load_trajectory_configs()
    assert loaded.trajectory_configs_payload() == saved


def test_named_trajectory_config_rejects_wrong_value_shape(tmp_path) -> None:
    rig = _rig(tmp_path)

    with np.testing.assert_raises_regex(ValueError, "three axis-angle joints"):
        rig.upsert_trajectory_config(
            "initial_poses", {"name": "bad", "arm_aa": [[0.0, 0.0, 0.0]]}
        )
    with np.testing.assert_raises_regex(ValueError, "three Cartesian coordinates"):
        rig.upsert_trajectory_config("goals", {"name": "bad", "goal": [0.0]})


def test_trajectory_config_endpoint_returns_updated_library(monkeypatch) -> None:
    expected = {
        "initial_poses": [],
        "goals": [{"name": "high", "goal": [0.1, 0.5, -0.2]}],
    }
    fake_rig = SimpleNamespace(upsert_trajectory_config=lambda kind, data: expected)
    monkeypatch.setattr(server, "rig", fake_rig)

    response = server.app.test_client().post(
        "/api/trajectory-configs/goals",
        json={"name": "high", "goal": [0.1, 0.5, -0.2]},
    )

    assert response.status_code == 200
    assert response.get_json() == {"trajectory_configs": expected}
