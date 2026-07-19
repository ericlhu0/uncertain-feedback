"""Debug visualization for hidden bounds.

Renders one panel per bound so a pose-dependent limit is inspectable at a
glance: pose-dependent bounds (a :class:`CoupledBound` or a gated
:class:`HiddenBound`) are drawn in the conditioning-feature vs bounded-feature
plane with the forbidden region shaded and each trajectory traced through it;
simple bounds are drawn as the feature over time with the forbidden range
shaded. Violation is exactly the plotted region — the shading is computed by
evaluating ``bound.violation`` on a grid, so the picture cannot drift from the
code.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from uncertain_feedback.planners.mpc.costs.base import MpcCostContext
from uncertain_feedback.simulated_users.base import (
    Bound,
    CoupledBound,
    HiddenBound,
    SimulatedUser,
    feature_series,
)

_FORBIDDEN_COLOR = "tab:red"
_GRID_N = 200


def _condition_feature(bound: Bound) -> str | None:
    if isinstance(bound, CoupledBound):
        return bound.cond_feature
    if isinstance(bound, HiddenBound) and bound.condition is not None:
        return bound.condition.feature
    return None


def _padded_range(values: list[np.ndarray], extra: list[float]) -> tuple[float, float]:
    stacked = np.concatenate([np.asarray(v, dtype=np.float64).ravel() for v in values])
    lo = min([float(stacked.min()), *extra])
    hi = max([float(stacked.max()), *extra])
    pad = 0.15 * max(hi - lo, 0.2)
    return lo - pad, hi + pad


def _draw_plane(
    ax: plt.Axes,
    bound: Bound,
    cond_name: str,
    features_by_traj: dict[str, dict[str, np.ndarray]],
) -> None:
    finite_thresholds = [
        v
        for v in (
            getattr(bound, "low", None),
            getattr(bound, "high", None),
            getattr(bound, "intercept", None),
        )
        if v is not None
    ]
    x_lo, x_hi = _padded_range(
        [f[cond_name] for f in features_by_traj.values()], extra=[]
    )
    y_lo, y_hi = _padded_range(
        [f[bound.feature] for f in features_by_traj.values()], extra=finite_thresholds
    )
    xs, ys = np.meshgrid(
        np.linspace(x_lo, x_hi, _GRID_N), np.linspace(y_lo, y_hi, _GRID_N)
    )
    violation = bound.violation({bound.feature: ys, cond_name: xs})
    ax.contourf(
        xs,
        ys,
        (violation > 0.0).astype(float),
        levels=[0.5, 1.5],
        colors=[_FORBIDDEN_COLOR],
        alpha=0.25,
    )
    for name, features in features_by_traj.items():
        x = features[cond_name]
        y = features[bound.feature]
        (line,) = ax.plot(x, y, label=name, linewidth=1.5)
        ax.plot(x[0], y[0], "o", color=line.get_color(), markersize=6)
        ax.plot(x[-1], y[-1], "s", color=line.get_color(), markersize=6)
    ax.set_xlabel(f"{cond_name} (rad)")
    ax.set_ylabel(f"{bound.feature} (rad)")
    ax.set_title(
        f"{bound.bound_type} on {bound.feature}\nvs {cond_name} (shaded = forbidden)"
    )


def _draw_series(
    ax: plt.Axes,
    bound: HiddenBound,
    features_by_traj: dict[str, dict[str, np.ndarray]],
) -> None:
    thresholds = [v for v in (bound.low, bound.high) if v is not None]
    y_lo, y_hi = _padded_range(
        [f[bound.feature] for f in features_by_traj.values()], extra=thresholds
    )
    if bound.bound_type == "upper_bound":
        ax.axhspan(float(bound.high), y_hi, color=_FORBIDDEN_COLOR, alpha=0.25)  # type: ignore[arg-type]
    elif bound.bound_type == "lower_bound":
        ax.axhspan(y_lo, float(bound.low), color=_FORBIDDEN_COLOR, alpha=0.25)  # type: ignore[arg-type]
    else:
        ax.axhspan(float(bound.low), float(bound.high), color=_FORBIDDEN_COLOR, alpha=0.25)  # type: ignore[arg-type]
    for name, features in features_by_traj.items():
        ax.plot(features[bound.feature], label=name, linewidth=1.5)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("frame")
    ax.set_ylabel(f"{bound.feature} (rad)")
    ax.set_title(f"{bound.bound_type} on {bound.feature} (shaded = forbidden)")


def render_hidden_bounds(
    user: SimulatedUser,
    context: MpcCostContext,
    trajectories: dict[str, np.ndarray],
    path: Path,
) -> Path:
    """Render one panel per hidden bound with ``trajectories`` traced through it.

    ``trajectories`` maps a legend label to a ``(T, 3, 3)`` arm trajectory
    (e.g. ``{"base": ..., "generated": ..., "correction": ...}``). Start frames
    are circles, end frames squares.
    """
    features_by_traj = {
        name: feature_series(context, traj) for name, traj in trajectories.items()
    }
    n = len(user.bounds)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 5.0), squeeze=False)
    for ax, bound in zip(axes[0], user.bounds):
        cond_name = _condition_feature(bound)
        if cond_name is not None:
            _draw_plane(ax, bound, cond_name, features_by_traj)
        else:
            assert isinstance(bound, HiddenBound)
            _draw_series(ax, bound, features_by_traj)
        ax.legend(fontsize=8)
    fig.suptitle(f"hidden bounds: {user.name}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
