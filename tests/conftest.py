"""Shared pytest configuration.

``SmplLeftArmFK`` loads ``SMPL_NEUTRAL.pkl`` from inside the MDM submodule. That
file is a licensed asset, gitignored by upstream, so it is absent on CI and on
any fresh clone. Decoding a pose file (``decode_hml_pose``) likewise reads the
HumanML3D ``Mean.npy``/``Std.npy`` from the MDM submodule, which CI does not
check out. Rather than mark whole modules — which would also skip the tests in
them that need neither — translate those specific missing-asset failures into a
skip.
"""

import pytest

from uncertain_feedback.motion_generators.mdm.hml_smpl_conversion import HML_STATS_DIR
from uncertain_feedback.planners.mpc.kinematics import _SMPL_PKL_DEFAULT


def _missing_asset(exc: BaseException) -> str | None:
    """The MDM-submodule asset whose absence raised ``exc``, if that is what did."""
    if not isinstance(exc, FileNotFoundError):
        return None
    message = str(exc)
    if not _SMPL_PKL_DEFAULT.exists() and "SMPL_NEUTRAL.pkl" in message:
        return str(_SMPL_PKL_DEFAULT)
    if not (HML_STATS_DIR / "Mean.npy").exists() and str(HML_STATS_DIR) in message:
        return str(HML_STATS_DIR)
    return None


def _force_skip_if_missing_asset(outcome) -> None:
    excinfo = outcome.excinfo
    missing = None if excinfo is None else _missing_asset(excinfo[1])
    if missing is not None:
        outcome.force_exception(pytest.skip.Exception(f"{missing} not available"))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item: pytest.Item):  # pylint: disable=unused-argument
    """Turn a missing-SMPL-model failure during fixture setup into a skip."""
    outcome = yield
    _force_skip_if_missing_asset(outcome)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item):  # pylint: disable=unused-argument
    """Turn a missing-SMPL-model failure into a skip."""
    outcome = yield
    _force_skip_if_missing_asset(outcome)
