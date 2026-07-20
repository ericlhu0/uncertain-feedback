"""Shared pytest configuration.

``SmplLeftArmFK`` loads ``SMPL_NEUTRAL.pkl`` from inside the MDM submodule. That
file is a licensed asset, gitignored by upstream, so it is absent on CI and on
any fresh clone. Rather than mark whole modules — which would also skip the
tests in them that need no body model — translate the specific missing-model
failure into a skip.
"""

import pytest

from uncertain_feedback.planners.mpc.kinematics import _SMPL_PKL_DEFAULT


def _is_missing_smpl_model(exc: BaseException) -> bool:
    if _SMPL_PKL_DEFAULT.exists():
        return False
    return isinstance(exc, FileNotFoundError) and "SMPL_NEUTRAL.pkl" in str(exc)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item):  # pylint: disable=unused-argument
    """Turn a missing-SMPL-model failure into a skip."""
    outcome = yield
    excinfo = outcome.excinfo
    if excinfo is not None and _is_missing_smpl_model(excinfo[1]):
        outcome.force_exception(
            pytest.skip.Exception(
                f"SMPL_NEUTRAL.pkl not available at {_SMPL_PKL_DEFAULT}"
            )
        )
