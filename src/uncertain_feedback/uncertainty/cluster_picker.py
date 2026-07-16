"""Interactive cluster picker for MDM trajectory uncertainty quantification.

Shows one 3D panel per cluster, each displaying the full body (T-pose
grey) with the mean arm pose overlaid in blue, plus a wrist trace.  The
user clicks a panel to select it, then clicks "Confirm" to return the
chosen cluster label.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

from uncertain_feedback.planners.mpc.kinematics import (
    LEFT_ARM_BONE_PAIRS_22,
    LEFT_ARM_CHAIN_INDICES,
    LEFT_ARM_JOINT_INDICES_22,
    SmplLeftArmFK,
)

if TYPE_CHECKING:
    from uncertain_feedback.utils.plot import ArmVisualizer

if TYPE_CHECKING:
    import matplotlib
    from matplotlib.figure import Figure

_COLOR_ARM = "#4878CF"
_COLOR_SELECTED = "#E87722"
_COLOR_TRACE = "#888888"  # wrist trace
_COLOR_CURRENT = "#AAAAAA"  # current MPC arm state

_SCALE_MIN = 0.0
_SCALE_MAX = 2.0
_SCALE_INIT = 1.0


@dataclass(frozen=True)
class ClusterPickResult:
    """Final selection from a potentially recursive cluster-picker session."""

    root_label: int
    sample_indices: np.ndarray
    scale: float


@dataclass
class _ClusterLevel:
    sample_indices: np.ndarray
    labels: np.ndarray
    selected_label: int | None
    scales: dict[int, float]
    path: tuple[int, ...]


@dataclass(frozen=True)
class _LevelPickResult:
    action: Literal["confirm", "refine", "back"]
    selected_label: int | None
    scales: dict[int, float]


def scale_trajectory(traj: np.ndarray, scale: float) -> np.ndarray:
    """Scale a trajectory's per-timestep motion about its first frame.

    Keeps the direction of motion at every timestep and multiplies its
    magnitude by ``scale``, anchored at ``traj[0]`` so ``scale == 1.0`` is a
    no-op and ``scale == 0.0`` holds the start pose.  Works on any array whose
    first axis is time (e.g. ``(n_frames, 3, 3)`` axis-angle).
    """
    anchor = traj[0:1]
    return anchor + scale * (traj - anchor)


def _update_cluster_artists(
    arm_lines_by_view: list[list],
    arm_scats_by_view: list,
    wrist_lines_by_view: list,
    body_cutoff: np.ndarray,  # (22, 3)
    wrist_trace: np.ndarray,  # (n_frames, 3)
) -> None:
    """Rewrite one cluster's mean-arm and wrist artists after rescaling."""
    for view_idx, view in enumerate(_PANEL_VIEWS):
        for (pi, ci), ln in zip(LEFT_ARM_BONE_PAIRS_22, arm_lines_by_view[view_idx]):
            seg = body_cutoff[[pi, ci]]
            ln.set_data(seg[:, view.hi], seg[:, view.vi])
        arm_scats_by_view[view_idx].set_offsets(
            np.column_stack(
                (
                    body_cutoff[LEFT_ARM_JOINT_INDICES_22, view.hi],
                    body_cutoff[LEFT_ARM_JOINT_INDICES_22, view.vi],
                )
            )
        )
        wrist_lines_by_view[view_idx].set_data(
            wrist_trace[:, view.hi], wrist_trace[:, view.vi]
        )


class _OrthoView(NamedTuple):
    title: str
    hi: int  # horizontal axis index into (x, y, z)
    vi: int  # vertical axis index
    hl: str  # horizontal label
    vl: str  # vertical label


_PANEL_VIEWS = [
    _OrthoView("Front (XY)", 0, 1, "X (m)", "Y (m)"),
    _OrthoView("Side (ZY)", 2, 1, "Z (m)", "Y (m)"),
    _OrthoView("Top (XZ)", 0, 2, "X (m)", "Z (m)"),
]


def _draw_bones_2d(
    ax: "plt.Axes",
    positions: np.ndarray,
    bone_pairs: list[tuple[int, int]],
    hi: int,
    vi: int,
    color: str,
    alpha: float = 1.0,
    lw: float = 1.5,
    linestyle: str = "-",
) -> None:
    for pi, ci in bone_pairs:
        seg = positions[[pi, ci]]
        ax.plot(seg[:, hi], seg[:, vi], color=color, alpha=alpha, linewidth=lw, linestyle=linestyle)


def _merge_arm(arm_full: np.ndarray, body_pos: np.ndarray | None) -> np.ndarray:
    """Combine FK arm joints with a reference body pose.

    Returns ``body_pos`` with the arm joints (collar/shoulder/elbow/wrist)
    replaced by the positions in ``arm_full``.  Falls back to ``arm_full``
    when ``body_pos`` is ``None``, which leaves non-arm joints at T-pose.
    """
    if body_pos is None:
        return arm_full
    result = body_pos.copy()
    result[LEFT_ARM_JOINT_INDICES_22] = arm_full[LEFT_ARM_JOINT_INDICES_22]
    return result


def _align_arm_to_spine(
    frame_positions: np.ndarray,
    display_spine_pos: np.ndarray,
) -> np.ndarray:
    """Translate an SMPL frame so its arm chain is anchored at display spine3."""
    aligned = np.asarray(frame_positions, dtype=np.float64).copy()
    spine3_j = LEFT_ARM_CHAIN_INDICES[0]
    offset = np.asarray(display_spine_pos, dtype=np.float64) - aligned[spine3_j]
    aligned[LEFT_ARM_JOINT_INDICES_22] += offset
    return aligned


def _full_body_positions_for_arm(
    fk: SmplLeftArmFK,
    arm_aa: np.ndarray,
    spine_pos: np.ndarray | None,
    spine_aa: np.ndarray | None,
) -> np.ndarray:
    return fk.full_body_positions(
        np.asarray(arm_aa, dtype=np.float64), spine_pos, spine_aa
    )


def _fk_batch_for_arm(
    fk: SmplLeftArmFK,
    arm_aa: np.ndarray,
) -> np.ndarray:
    return fk.fk_batch(np.asarray(arm_aa, dtype=np.float64))


def _draw_body(
    ax: "plt.Axes", body_pos: np.ndarray, arm_color: str, hi: int, vi: int
) -> tuple[list, object]:
    """Draw full body on a 2D axes.

    Returns (arm_bone_lines, arm_joint_scatter).
    """
    from uncertain_feedback.utils.plot import ArmVisualizer  # pylint: disable=import-outside-toplevel
    # Grey non-arm skeleton
    _draw_bones_2d(ax, body_pos, ArmVisualizer.BODY_BONES, hi, vi, ArmVisualizer.BODY_COLOR, alpha=0.45, lw=1.2)
    ax.scatter(
        body_pos[ArmVisualizer.BODY_JOINTS, hi],
        body_pos[ArmVisualizer.BODY_JOINTS, vi],
        color=ArmVisualizer.BODY_COLOR,
        s=14,
        alpha=0.45,
    )

    # Coloured arm skeleton (mutable for highlight)
    arm_lines = []
    for pi, ci in LEFT_ARM_BONE_PAIRS_22:
        seg = body_pos[[pi, ci]]
        (ln,) = ax.plot(seg[:, hi], seg[:, vi], color=arm_color, linewidth=2.2)
        arm_lines.append(ln)
    arm_scat = ax.scatter(
        body_pos[LEFT_ARM_JOINT_INDICES_22, hi],
        body_pos[LEFT_ARM_JOINT_INDICES_22, vi],
        c=arm_color,
        s=35,
        zorder=5,
    )
    return arm_lines, arm_scat


def _build_figure(  # pylint: disable=too-many-locals,redefined-outer-name
    unique_labels: list[int],
    cluster_body_cutoffs: list[np.ndarray],  # each (22, 3) — mean arm at cutoff
    cluster_wrist_traces: list[np.ndarray],  # each (n_frames, 3)
    cluster_counts: list[int],
    lims: list[tuple[float, float]],
    cluster_individual_previews: (
        list[list[np.ndarray]] | None
    ) = None,  # each list of (22, 3)
    current_body: np.ndarray | None = None,  # (22, 3) current MPC arm state
    init_scales: dict[int, float] | None = None,
) -> tuple["Figure", list[list], list[list], list[list], list[list], list[Slider]]:
    n_clusters = len(unique_labels)
    n_views = len(_PANEL_VIEWS)
    fig_w = max(4 * n_clusters, 8)
    fig = plt.figure(figsize=(fig_w, 3.0 * n_views + 1.0))
    fig.patch.set_facecolor("#F5F5F5")

    left, right = 0.06, 0.98
    gs = fig.add_gridspec(
        n_views,
        n_clusters,
        bottom=0.17,
        top=0.92,
        left=left,
        right=right,
        wspace=0.08,
        hspace=0.30,
    )
    axes_by_cluster = [
        [fig.add_subplot(gs[row, col]) for row in range(n_views)]
        for col in range(n_clusters)
    ]

    # Per-cluster arm/scatter/wrist handles, grouped by view for live rescaling.
    panel_arm_lines: list[list] = []  # [cluster][view] -> list of bone Line2D
    panel_arm_scats: list = []  # [cluster][view] -> arm joint scatter
    panel_wrist_lines: list[list] = []  # [cluster][view] -> wrist trace Line2D

    for idx, (k, body_cutoff, wrist_trace, count) in enumerate(
        zip(unique_labels, cluster_body_cutoffs, cluster_wrist_traces, cluster_counts)
    ):
        cluster_lines: list = []
        cluster_scats: list = []
        cluster_wrists: list = []
        for view_idx, view in enumerate(_PANEL_VIEWS):
            ax = axes_by_cluster[idx][view_idx]
            ax.set_aspect("equal")
            ax.set_xlim(*lims[view.hi])
            ax.set_ylim(*lims[view.vi])
            ax.set_xlabel(view.hl, fontsize=7)
            ax.set_ylabel(view.vl, fontsize=7)
            ax.tick_params(labelsize=6)
            if view_idx == 0:
                ax.set_title(
                    f"Cluster {k} ({count} samples)\n{view.title}",
                    fontsize=9,
                    pad=4,
                )
            else:
                ax.set_title(view.title, fontsize=9, pad=4)

            # Wrist trace of the (scaled) cluster mean — mutable for rescaling
            (wrist_ln,) = ax.plot(
                wrist_trace[:, view.hi],
                wrist_trace[:, view.vi],
                linestyle=":",
                color=_COLOR_TRACE,
                linewidth=1.0,
                alpha=0.7,
            )
            cluster_wrists.append(wrist_ln)

            # Individual ghost arms (one per sample in cluster), very faint
            if cluster_individual_previews is not None:
                for body_ind in cluster_individual_previews[idx]:
                    _draw_bones_2d(
                        ax,
                        body_ind,
                        LEFT_ARM_BONE_PAIRS_22,
                        view.hi,
                        view.vi,
                        _COLOR_ARM,
                        alpha=0.12,
                        lw=1.2,
                    )

            # Current MPC arm state (grey, drawn behind the cluster arm)
            if current_body is not None:
                _draw_bones_2d(
                    ax,
                    current_body,
                    LEFT_ARM_BONE_PAIRS_22,
                    view.hi,
                    view.vi,
                    _COLOR_CURRENT,
                    alpha=0.9,
                    lw=2.2,
                )
                ax.scatter(
                    current_body[LEFT_ARM_JOINT_INDICES_22, view.hi],
                    current_body[LEFT_ARM_JOINT_INDICES_22, view.vi],
                    color=_COLOR_CURRENT,
                    s=28,
                    zorder=4,
                )

            # Solid mean arm at trajectory-fraction cutoff (the pose that will be enqueued)
            arm_lines, arm_scat = _draw_body(ax, body_cutoff, _COLOR_ARM, view.hi, view.vi)
            cluster_lines.append(arm_lines)
            cluster_scats.append(arm_scat)
        panel_arm_lines.append(cluster_lines)
        panel_arm_scats.append(cluster_scats)
        panel_wrist_lines.append(cluster_wrists)

    # Per-column magnitude sliders, one under each cluster panel.
    sliders: list[Slider] = []
    col_w = (right - left) / n_clusters
    for idx, k in enumerate(unique_labels):
        col_x0 = left + idx * col_w
        slider_ax = fig.add_axes(
            [col_x0 + 0.15 * col_w, 0.085, 0.7 * col_w, 0.025]
        )
        slider = Slider(
            slider_ax,
            "magnitude",
            _SCALE_MIN,
            _SCALE_MAX,
            valinit=(init_scales or {}).get(k, _SCALE_INIT),
            valfmt="%.2f",
        )
        slider.label.set_fontsize(7)
        slider.valtext.set_fontsize(7)
        sliders.append(slider)

    return (
        fig,
        axes_by_cluster,
        panel_arm_lines,
        panel_arm_scats,
        panel_wrist_lines,
        sliders,
    )


def _pick_cluster_level(  # pylint: disable=too-many-locals,redefined-outer-name,too-many-statements
    trajectories: np.ndarray,
    labels: np.ndarray,
    fk: SmplLeftArmFK | None = None,
    save_path: str | Path | None = None,
    trajectory_fraction: float = 0.75,
    spine_pos: np.ndarray | None = None,
    spine_aa: np.ndarray | None = None,
    body_pos: np.ndarray | None = None,
    current_arm_aa: np.ndarray | None = None,
    init_scales: dict[int, float] | None = None,
    selected_label: int | None = None,
    can_go_back: bool = False,
    n_clusters: int = 1,
    path: tuple[int, ...] = (),
) -> _LevelPickResult:
    """Show one blocking trajectory-cluster level.

    Each cluster panel shows:

    * Faint individual arms at the ``trajectory_fraction`` cutoff frame —
      one per sample, showing within-cluster spread.
    * Current MPC arm state (grey), so the user can see where the arm is now.
    * Solid mean arm at the ``trajectory_fraction`` cutoff frame — the pose
      that will be enqueued when this cluster is selected.
    * Dotted wrist trace of the cluster mean trajectory.

    Args:
        trajectories:        ``(num_samples, n_frames, 3, 3)`` arm axis-angle batch.
        labels:              ``(num_samples,)`` integer cluster labels (0-based).
        fk:                  FK instance.  Defaults to :class:`SmplLeftArmFK`.
        save_path:           If given, save a PNG of the initial window here
                             before showing the interactive display.
        trajectory_fraction: Fraction of trajectory frames that will be
                             enqueued; arms are drawn at this timestep.
                             Should match
                             :attr:`~LeftArmMPCMDM.trajectory_fraction`.
        spine_pos:           ``(3,)`` world position of spine3.
        spine_aa:            ``(3,)`` world axis-angle of spine3.
        body_pos:            ``(22, 3)`` reference body joint positions (e.g.
                             from the initial seated pose decode).  When
                             provided, non-arm joints are drawn from this pose
                             instead of T-pose.
        current_arm_aa:      ``(3, 3)`` current MPC arm axis-angles.  When
                             provided, the current arm pose is drawn in grey
                             on every cluster panel.
        init_scales:         Initial magnitude-slider values by local label.
        selected_label:      Local label to restore as selected.
        can_go_back:         Whether the Back action is available.
        n_clusters:          Number of clusters used for child refinement.
        path:                Ancestor labels displayed in the title.
    """
    if fk is None:
        fk = SmplLeftArmFK()

    trajectories = np.asarray(trajectories, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.intp)
    unique_labels = sorted(set(labels.tolist()))

    # ------------------------------------------------------------------
    # Precompute per-cluster mean trajectories
    # ------------------------------------------------------------------
    cluster_body_cutoffs: list[np.ndarray] = []  # (22, 3) mean arm at cutoff frame
    cluster_individual_previews: list[list[np.ndarray]] = (
        []
    )  # per-sample (22, 3) at cutoff
    cluster_wrist_traces: list[np.ndarray] = []  # (n_frames, 3)
    cluster_mean_trajs: list[np.ndarray] = []  # (n_frames, 3, 3) unscaled mean aa
    cluster_counts: list[int] = []

    precompute_t0 = time.perf_counter()
    for k in unique_labels:
        mask = labels == k
        mean_traj = trajectories[mask].mean(axis=0)  # (n_frames, 3, 3)
        n_frames = mean_traj.shape[0]
        # Mean body at trajectory-fraction cutoff frame (the pose that gets enqueued)
        preview_idx = max(0, round(n_frames * trajectory_fraction) - 1)
        body_cutoff = _merge_arm(
            _full_body_positions_for_arm(
                fk, mean_traj[preview_idx], spine_pos, spine_aa
            ),
            body_pos,
        )  # (22, 3)
        # Per-sample ghost arm body poses at the cutoff frame
        individual_previews = [
            _merge_arm(
                _full_body_positions_for_arm(fk, traj[preview_idx], spine_pos, spine_aa),
                body_pos,
            )  # (22, 3)
            for traj in trajectories[mask]
        ]
        # Wrist trace from arm-chain FK
        arm_positions = _fk_batch_for_arm(fk, mean_traj)
        wrist_trace = arm_positions[:, -1, :]  # (n_frames, 3)

        cluster_body_cutoffs.append(body_cutoff)
        cluster_individual_previews.append(individual_previews)
        cluster_wrist_traces.append(wrist_trace)
        cluster_mean_trajs.append(mean_traj)
        cluster_counts.append(int(mask.sum()))
    print(
        f"[timing] cluster picker precompute: {time.perf_counter() - precompute_t0:.3f}s"
    )

    # Current MPC arm state (same on every cluster panel)
    current_body: np.ndarray | None = None
    if current_arm_aa is not None:
        current_body = _merge_arm(
            _full_body_positions_for_arm(fk, current_arm_aa, spine_pos, spine_aa),
            body_pos,
        )

    # Shared axis limits: cutoffs, individual previews, wrist traces, current arm
    all_cutoffs = np.stack(cluster_body_cutoffs, axis=0).reshape(-1, 3)
    all_ind_previews = np.vstack([p for ip in cluster_individual_previews for p in ip])
    all_wrists = np.concatenate(cluster_wrist_traces, axis=0)
    extra = [current_body.reshape(-1, 3)] if current_body is not None else []
    all_pts = np.vstack([all_cutoffs, all_ind_previews, all_wrists, *extra])
    margin = 0.05
    lims = [
        (float(all_pts[:, d].min()) - margin, float(all_pts[:, d].max()) + margin)
        for d in range(3)
    ]

    # ------------------------------------------------------------------
    # Build figure
    # ------------------------------------------------------------------
    figure_t0 = time.perf_counter()
    (
        fig,
        axes_by_cluster,
        panel_arm_lines,
        panel_arm_scats,
        panel_wrist_lines,
        sliders,
    ) = _build_figure(
        unique_labels,
        cluster_body_cutoffs,
        cluster_wrist_traces,
        cluster_counts,
        lims,
        cluster_individual_previews=cluster_individual_previews,
        current_body=current_body,
        init_scales=init_scales,
    )
    print(
        f"[timing] cluster picker figure build: {time.perf_counter() - figure_t0:.3f}s"
    )

    # ------------------------------------------------------------------
    # State + interaction
    # ------------------------------------------------------------------
    state: dict = {"selected": None, "action": None}

    def _rescale(idx: int, scale: float) -> None:
        traj = scale_trajectory(cluster_mean_trajs[idx], scale)
        n_f = traj.shape[0]
        pidx = max(0, round(n_f * trajectory_fraction) - 1)
        body_cutoff = _merge_arm(
            _full_body_positions_for_arm(fk, traj[pidx], spine_pos, spine_aa),
            body_pos,
        )
        wrist_trace = _fk_batch_for_arm(fk, traj)[:, -1, :]
        _update_cluster_artists(
            panel_arm_lines[idx],
            panel_arm_scats[idx],
            panel_wrist_lines[idx],
            body_cutoff,
            wrist_trace,
        )
        fig.canvas.draw_idle()

    for i, slider in enumerate(sliders):
        slider.on_changed(lambda val, i=i: _rescale(i, val))
    for i, slider in enumerate(sliders):
        if slider.val != _SCALE_INIT:
            _rescale(i, float(slider.val))

    def _set_selected(idx: int) -> None:
        state["selected"] = idx
        for i, (arm_lines_by_view, arm_scats) in enumerate(
            zip(panel_arm_lines, panel_arm_scats)
        ):
            color = _COLOR_SELECTED if i == idx else _COLOR_ARM
            for view_lines in arm_lines_by_view:
                for ln in view_lines:
                    ln.set_color(color)
            for arm_scat in arm_scats:
                arm_scat.set_color(color)
            for ax in axes_by_cluster[i]:
                ax.set_facecolor("#FFF3E0" if i == idx else "white")
        refine_btn.set_active(n_clusters >= 2)
        fig.canvas.draw_idle()

    def _on_click(event: "matplotlib.backend_bases.MouseEvent") -> None:
        if event.inaxes is None:
            return
        for i, cluster_axes in enumerate(axes_by_cluster):
            if event.inaxes in cluster_axes:
                _set_selected(i)
                return

    fig.canvas.mpl_connect("button_press_event", _on_click)

    back_ax = fig.add_axes([0.19, 0.02, 0.18, 0.045])
    back_btn = Button(back_ax, "Back", color="#DDDDDD", hovercolor="#BBBBBB")
    back_btn.set_active(can_go_back)
    refine_ax = fig.add_axes([0.41, 0.02, 0.18, 0.045])
    refine_btn = Button(
        refine_ax, "Refine selected", color="#DDDDDD", hovercolor="#BBBBBB"
    )
    refine_btn.set_active(False)
    confirm_ax = fig.add_axes([0.63, 0.02, 0.18, 0.045])
    confirm_btn = Button(
        confirm_ax, "Confirm", color="#DDDDDD", hovercolor="#BBBBBB"
    )

    def _finish(action: Literal["confirm", "refine", "back"]) -> None:
        state["action"] = action
        plt.close(fig)

    def _on_confirm(_event: object) -> None:
        if state["selected"] is None:
            return
        _finish("confirm")

    def _on_refine(_event: object) -> None:
        idx = state["selected"]
        if idx is None or n_clusters < 2:
            return
        _finish("refine")

    def _on_back(_event: object) -> None:
        if can_go_back:
            _finish("back")

    confirm_btn.on_clicked(_on_confirm)
    refine_btn.on_clicked(_on_refine)
    back_btn.on_clicked(_on_back)
    if selected_label is not None and selected_label in unique_labels:
        _set_selected(unique_labels.index(selected_label))
    path_text = "Root" if not path else "Root > " + " > ".join(
        f"cluster {label}" for label in path
    )
    fig.suptitle(
        f"{path_text} — select a cluster, refine it, or confirm its mean",
        fontsize=10,
        y=0.97,
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    interaction_t0 = time.perf_counter()
    plt.show(block=True)
    print(
        "[timing] cluster picker interaction/display: "
        f"{time.perf_counter() - interaction_t0:.3f}s"
    )

    if state["action"] is None:
        raise RuntimeError("Window closed without confirming or navigating.")
    chosen_label = (
        None
        if state["selected"] is None
        else unique_labels[int(state["selected"])]
    )
    return _LevelPickResult(
        action=state["action"],
        selected_label=chosen_label,
        scales={
            label: float(slider.val)
            for label, slider in zip(unique_labels, sliders)
        },
    )


def _navigate_cluster_levels(
    samples: np.ndarray,
    labels: np.ndarray,
    show_level: Callable[[_ClusterLevel], _LevelPickResult],
    recluster: Callable[[np.ndarray], np.ndarray] | None,
    init_scale: float,
    n_clusters: int | None = None,
) -> ClusterPickResult:
    """Navigate recursive cluster levels while retaining global sample indices."""
    unique_labels = sorted(int(value) for value in np.unique(labels))
    levels = [
        _ClusterLevel(
            sample_indices=np.arange(samples.shape[0], dtype=np.intp),
            labels=np.asarray(labels, dtype=np.intp),
            selected_label=None,
            scales={label: init_scale for label in unique_labels},
            path=(),
        )
    ]
    while levels:
        level = levels[-1]
        result = show_level(level)
        level.scales = result.scales
        if result.selected_label is not None:
            level.selected_label = result.selected_label

        if result.action == "back":
            levels.pop()
            continue
        if level.selected_label is None:
            raise RuntimeError("Choose a cluster before confirming or refining.")

        selected_indices = level.sample_indices[
            level.labels == level.selected_label
        ]
        if result.action == "refine":
            if recluster is None:
                raise RuntimeError("Recursive clustering is not configured.")
            child_labels = (
                np.arange(selected_indices.size, dtype=np.intp)
                if n_clusters is not None and selected_indices.size < n_clusters
                else np.asarray(recluster(samples[selected_indices]), dtype=np.intp)
            )
            child_unique = sorted(int(value) for value in np.unique(child_labels))
            child_scale = level.scales[level.selected_label]
            levels.append(
                _ClusterLevel(
                    sample_indices=selected_indices,
                    labels=child_labels,
                    selected_label=None,
                    scales={label: child_scale for label in child_unique},
                    path=(*level.path, level.selected_label),
                )
            )
            continue

        root_label = levels[0].selected_label
        if root_label is None:
            raise RuntimeError("Root cluster selection was lost.")
        return ClusterPickResult(
            root_label=root_label,
            sample_indices=selected_indices,
            scale=level.scales[level.selected_label],
        )
    raise RuntimeError("Cluster navigation ended without a selection.")


def pick_cluster(
    trajectories: np.ndarray,
    labels: np.ndarray,
    fk: SmplLeftArmFK | None = None,
    save_path: str | Path | None = None,
    trajectory_fraction: float = 0.75,
    spine_pos: np.ndarray | None = None,
    spine_aa: np.ndarray | None = None,
    body_pos: np.ndarray | None = None,
    current_arm_aa: np.ndarray | None = None,
    init_scale: float = _SCALE_INIT,
    recluster: Callable[[np.ndarray], np.ndarray] | None = None,
    n_clusters: int = 1,
) -> ClusterPickResult:
    """Let the user recursively refine and select axis-angle trajectories."""
    trajectories = np.asarray(trajectories, dtype=np.float64)

    def _show(level: _ClusterLevel) -> _LevelPickResult:
        return _pick_cluster_level(
            trajectories[level.sample_indices],
            level.labels,
            fk=fk,
            save_path=save_path if not level.path else None,
            trajectory_fraction=trajectory_fraction,
            spine_pos=spine_pos,
            spine_aa=spine_aa,
            body_pos=body_pos,
            current_arm_aa=current_arm_aa,
            init_scales=level.scales,
            selected_label=level.selected_label,
            can_go_back=bool(level.path),
            n_clusters=n_clusters if recluster is not None else 1,
            path=level.path,
        )

    return _navigate_cluster_levels(
        trajectories, labels, _show, recluster, init_scale, n_clusters
    )


def _pick_cluster_positions_level(  # pylint: disable=too-many-locals,redefined-outer-name,too-many-statements
    positions: np.ndarray,
    labels: np.ndarray,
    fk: SmplLeftArmFK | None = None,
    save_path: str | Path | None = None,
    trajectory_fraction: float = 0.75,
    spine_pos: np.ndarray | None = None,
    spine_aa: np.ndarray | None = None,
    body_pos: np.ndarray | None = None,
    current_arm_aa: np.ndarray | None = None,
    init_scales: dict[int, float] | None = None,
    selected_label: int | None = None,
    can_go_back: bool = False,
    n_clusters: int = 1,
    path: tuple[int, ...] = (),
) -> _LevelPickResult:
    """Show one blocking cluster level from precomputed SMPL XYZ positions.

    Args:
        positions: ``(num_samples, n_frames, 22, 3)`` global SMPL joint
            positions.
        labels: ``(num_samples,)`` integer cluster labels.
        init_scales: Initial magnitude-slider values by local label.
    """
    if fk is None:
        fk = SmplLeftArmFK()

    positions = np.asarray(positions, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.intp)
    unique_labels = sorted(set(labels.tolist()))

    cluster_body_cutoffs: list[np.ndarray] = []
    cluster_individual_previews: list[list[np.ndarray]] = []
    cluster_wrist_traces: list[np.ndarray] = []
    cluster_mean_trajs: list[np.ndarray] = []  # (n_frames, 3, 3) unscaled mean aa
    cluster_counts: list[int] = []

    precompute_t0 = time.perf_counter()
    spine3_j = LEFT_ARM_CHAIN_INDICES[0]
    display_spine_pos = (
        np.asarray(body_pos[spine3_j], dtype=np.float64)
        if body_pos is not None
        else (
            np.asarray(spine_pos, dtype=np.float64)
            if spine_pos is not None
            else fk.tpose_spine3_pos
        )
    )
    for k in unique_labels:
        mask = labels == k
        mean_positions = positions[mask].mean(axis=0)  # (n_frames, 22, 3)
        n_frames = mean_positions.shape[0]
        preview_idx = max(0, round(n_frames * trajectory_fraction) - 1)
        mean_arm_aa = fk.arm_aa_from_positions_batch(
            mean_positions,
            spine3_aa=spine_aa,
        )

        body_cutoff = _merge_arm(
            _full_body_positions_for_arm(
                fk, mean_arm_aa[preview_idx], display_spine_pos, spine_aa
            ),
            body_pos,
        )
        individual_previews = [
            _merge_arm(
                _full_body_positions_for_arm(
                    fk,
                    fk.arm_aa_from_positions(
                        sample_positions[preview_idx],
                        spine3_aa=spine_aa,
                    ),
                    display_spine_pos,
                    spine_aa,
                ),
                body_pos,
            )
            for sample_positions in positions[mask]
        ]
        wrist_trace = fk.fk_batch(
            mean_arm_aa, display_spine_pos, spine_aa
        )[:, -1, :]

        cluster_body_cutoffs.append(body_cutoff)
        cluster_individual_previews.append(individual_previews)
        cluster_wrist_traces.append(wrist_trace)
        cluster_mean_trajs.append(mean_arm_aa)
        cluster_counts.append(int(mask.sum()))
    print(
        "[timing] position cluster picker precompute: "
        f"{time.perf_counter() - precompute_t0:.3f}s"
    )

    current_body: np.ndarray | None = None
    if current_arm_aa is not None:
        current_body = _merge_arm(
            _full_body_positions_for_arm(fk, current_arm_aa, spine_pos, spine_aa),
            body_pos,
        )

    all_cutoffs = np.stack(cluster_body_cutoffs, axis=0).reshape(-1, 3)
    all_ind_previews = np.vstack([p for ip in cluster_individual_previews for p in ip])
    all_wrists = np.concatenate(cluster_wrist_traces, axis=0)
    extra = [current_body.reshape(-1, 3)] if current_body is not None else []
    all_pts = np.vstack([all_cutoffs, all_ind_previews, all_wrists, *extra])
    margin = 0.05
    lims = [
        (float(all_pts[:, d].min()) - margin, float(all_pts[:, d].max()) + margin)
        for d in range(3)
    ]

    figure_t0 = time.perf_counter()
    (
        fig,
        axes_by_cluster,
        panel_arm_lines,
        panel_arm_scats,
        panel_wrist_lines,
        sliders,
    ) = _build_figure(
        unique_labels,
        cluster_body_cutoffs,
        cluster_wrist_traces,
        cluster_counts,
        lims,
        cluster_individual_previews=cluster_individual_previews,
        current_body=current_body,
        init_scales=init_scales,
    )
    print(
        "[timing] position cluster picker figure build: "
        f"{time.perf_counter() - figure_t0:.3f}s"
    )

    state: dict = {"selected": None, "action": None}

    def _rescale(idx: int, scale: float) -> None:
        traj = scale_trajectory(cluster_mean_trajs[idx], scale)
        n_f = traj.shape[0]
        pidx = max(0, round(n_f * trajectory_fraction) - 1)
        body_cutoff = _merge_arm(
            _full_body_positions_for_arm(fk, traj[pidx], display_spine_pos, spine_aa),
            body_pos,
        )
        wrist_trace = fk.fk_batch(traj, display_spine_pos, spine_aa)[:, -1, :]
        _update_cluster_artists(
            panel_arm_lines[idx],
            panel_arm_scats[idx],
            panel_wrist_lines[idx],
            body_cutoff,
            wrist_trace,
        )
        fig.canvas.draw_idle()

    for i, slider in enumerate(sliders):
        slider.on_changed(lambda val, i=i: _rescale(i, val))
    for i, slider in enumerate(sliders):
        if slider.val != _SCALE_INIT:
            _rescale(i, float(slider.val))

    def _set_selected(idx: int) -> None:
        state["selected"] = idx
        for i, (arm_lines_by_view, arm_scats) in enumerate(
            zip(panel_arm_lines, panel_arm_scats)
        ):
            color = _COLOR_SELECTED if i == idx else _COLOR_ARM
            for view_lines in arm_lines_by_view:
                for ln in view_lines:
                    ln.set_color(color)
            for arm_scat in arm_scats:
                arm_scat.set_color(color)
            for ax in axes_by_cluster[i]:
                ax.set_facecolor("#FFF3E0" if i == idx else "white")
        refine_btn.set_active(n_clusters >= 2)
        fig.canvas.draw_idle()

    def _on_click(event: "matplotlib.backend_bases.MouseEvent") -> None:
        if event.inaxes is None:
            return
        for i, cluster_axes in enumerate(axes_by_cluster):
            if event.inaxes in cluster_axes:
                _set_selected(i)
                return

    fig.canvas.mpl_connect("button_press_event", _on_click)

    back_ax = fig.add_axes([0.19, 0.02, 0.18, 0.045])
    back_btn = Button(back_ax, "Back", color="#DDDDDD", hovercolor="#BBBBBB")
    back_btn.set_active(can_go_back)
    refine_ax = fig.add_axes([0.41, 0.02, 0.18, 0.045])
    refine_btn = Button(
        refine_ax, "Refine selected", color="#DDDDDD", hovercolor="#BBBBBB"
    )
    refine_btn.set_active(False)
    confirm_ax = fig.add_axes([0.63, 0.02, 0.18, 0.045])
    confirm_btn = Button(
        confirm_ax, "Confirm", color="#DDDDDD", hovercolor="#BBBBBB"
    )

    def _finish(action: Literal["confirm", "refine", "back"]) -> None:
        state["action"] = action
        plt.close(fig)

    def _on_confirm(_event: object) -> None:
        if state["selected"] is None:
            return
        _finish("confirm")

    def _on_refine(_event: object) -> None:
        idx = state["selected"]
        if idx is None or n_clusters < 2:
            return
        _finish("refine")

    def _on_back(_event: object) -> None:
        if can_go_back:
            _finish("back")

    confirm_btn.on_clicked(_on_confirm)
    refine_btn.on_clicked(_on_refine)
    back_btn.on_clicked(_on_back)
    if selected_label is not None and selected_label in unique_labels:
        _set_selected(unique_labels.index(selected_label))
    path_text = "Root" if not path else "Root > " + " > ".join(
        f"cluster {label}" for label in path
    )
    fig.suptitle(
        f"{path_text} — select a cluster, refine it, or confirm its mean",
        fontsize=10,
        y=0.97,
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    interaction_t0 = time.perf_counter()
    plt.show(block=True)
    print(
        "[timing] position cluster picker interaction/display: "
        f"{time.perf_counter() - interaction_t0:.3f}s"
    )

    if state["action"] is None:
        raise RuntimeError("Window closed without confirming or navigating.")
    chosen_label = (
        None
        if state["selected"] is None
        else unique_labels[int(state["selected"])]
    )
    return _LevelPickResult(
        action=state["action"],
        selected_label=chosen_label,
        scales={
            label: float(slider.val)
            for label, slider in zip(unique_labels, sliders)
        },
    )


def pick_cluster_positions(
    positions: np.ndarray,
    labels: np.ndarray,
    fk: SmplLeftArmFK | None = None,
    save_path: str | Path | None = None,
    trajectory_fraction: float = 0.75,
    spine_pos: np.ndarray | None = None,
    spine_aa: np.ndarray | None = None,
    body_pos: np.ndarray | None = None,
    current_arm_aa: np.ndarray | None = None,
    init_scale: float = _SCALE_INIT,
    recluster: Callable[[np.ndarray], np.ndarray] | None = None,
    n_clusters: int = 1,
) -> ClusterPickResult:
    """Let the user recursively refine and select position trajectories."""
    positions = np.asarray(positions, dtype=np.float64)

    def _show(level: _ClusterLevel) -> _LevelPickResult:
        return _pick_cluster_positions_level(
            positions[level.sample_indices],
            level.labels,
            fk=fk,
            save_path=save_path if not level.path else None,
            trajectory_fraction=trajectory_fraction,
            spine_pos=spine_pos,
            spine_aa=spine_aa,
            body_pos=body_pos,
            current_arm_aa=current_arm_aa,
            init_scales=level.scales,
            selected_label=level.selected_label,
            can_go_back=bool(level.path),
            n_clusters=n_clusters if recluster is not None else 1,
            path=level.path,
        )

    return _navigate_cluster_levels(
        positions, labels, _show, recluster, init_scale, n_clusters
    )


# ---------------------------------------------------------------------------
# Demo / screenshot entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pylint: disable=redefined-outer-name
    import sys

    rng = np.random.default_rng(42)
    num_samples, n_frames = 12, 60

    # Three synthetic clusters with distinct arm directions
    raw = rng.standard_normal((num_samples, n_frames, 4, 3)) * 0.15
    offsets = [
        np.array([0.0, 0.3, 0.0]),  # arm up
        np.array([0.3, 0.0, 0.0]),  # arm forward
        np.array([0.0, 0.0, -0.3]),  # arm down
    ]
    demo_labels = np.array([i % 3 for i in range(num_samples)], dtype=np.intp)
    for i in range(num_samples):
        # Ramp the offset across frames so the wrist trace is visible
        ramp = np.linspace(0, 1, n_frames)[:, None, None]
        raw[i] += ramp * offsets[demo_labels[i]][None, None, :]

    save = Path("cluster_picker_preview.png")
    print(f"Saving preview to {save.resolve()} …")

    backend = plt.get_backend()
    if backend.lower() == "agg":
        fk = SmplLeftArmFK()
        unique_labels = sorted(set(demo_labels.tolist()))
        n_clusters = len(unique_labels)

        cluster_body_finals = []
        cluster_wrist_traces = []
        cluster_counts = []
        for k in unique_labels:
            mask = demo_labels == k
            mean_traj = raw[mask].mean(axis=0)
            cluster_body_finals.append(fk.full_body_positions(mean_traj[-1]))
            arm_pos = fk.fk_batch(mean_traj)
            cluster_wrist_traces.append(arm_pos[:, -1, :])
            cluster_counts.append(int(mask.sum()))

        all_body = np.stack(cluster_body_finals).reshape(-1, 3)
        all_wrists = np.concatenate(cluster_wrist_traces)
        all_pts = np.vstack([all_body, all_wrists])
        margin = 0.05  # pylint: disable=invalid-name
        lims = [
            (float(all_pts[:, d].min()) - margin, float(all_pts[:, d].max()) + margin)
            for d in range(3)
        ]

        fig, axes, _, _, _, _ = _build_figure(
            unique_labels,
            cluster_body_finals,
            cluster_wrist_traces,
            cluster_counts,
            lims,
        )
        for x, label in (
            (0.19, "Back"),
            (0.41, "Refine selected"),
            (0.63, "Confirm"),
        ):
            btn_ax = fig.add_axes([x, 0.02, 0.18, 0.045])
            btn_ax.set_facecolor("#DDDDDD")
            btn_ax.text(0.5, 0.5, label, ha="center", va="center", fontsize=10)
            btn_ax.set_xticks([])
            btn_ax.set_yticks([])
        fig.suptitle(
            "Root — select a cluster, refine it, or confirm its mean",
            fontsize=10,
            y=0.97,
        )
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Saved → {save.resolve()}")
        sys.exit(0)

    chosen = pick_cluster(raw, demo_labels, save_path=save)
    print(
        f"User chose cluster {chosen.root_label} at magnitude {chosen.scale:.2f} "
        f"from {len(chosen.sample_indices)} samples"
    )
