"""A clip set generated elsewhere resolves its planner config to this checkout's copy."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from pathlib import Path

from uncertain_feedback.data_collection.dataset_auto_correction.clips import (
    local_config_path,
)

_CONF = Path(__file__).resolve().parents[1] / "evaluation" / "conf"


def test_an_existing_path_is_kept(tmp_path: Path) -> None:
    stored = tmp_path / "mpc_demo_low1.yaml"
    stored.write_text("pose: {}\n")
    assert local_config_path(stored) == stored


def test_a_foreign_path_falls_back_to_the_checkout_copy(capsys) -> None:
    foreign = Path(
        "/home/nobody/safe-contact/uncertain-feedback/evaluation/conf/mpc_demo_low1.yaml"
    )
    assert local_config_path(foreign) == _CONF / "mpc_demo_low1.yaml"
    assert "is not on this host" in capsys.readouterr().out


def test_an_unknown_file_is_returned_as_stored() -> None:
    foreign = Path("/home/nobody/conf/no_such_config.yaml")
    assert local_config_path(foreign) == foreign
