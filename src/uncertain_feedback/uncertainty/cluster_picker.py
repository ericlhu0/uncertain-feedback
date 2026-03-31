"""Interactive cluster picker for MDM trajectory uncertainty quantification.

Shows one 3D panel per cluster, each displaying the full body (T-pose
grey) with the mean arm pose overlaid in blue, plus a wrist trace.  The
user clicks a panel to select it, then clicks "Confirm" to return the
chosen cluster label.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

from uncertain_feedback.planners.mpc.kinematics import (
    LEFT_ARM_BONE_PAIRS_22,
    LEFT_ARM_JOINT_INDICES_22,
    SmplLeftArmFK,
)
from uncertain_feedback.planners.mpc.visualizer import (
    _BODY_BONES,
    _BODY_COLOR,
    _BODY_JOINTS,
    _draw_bones_3d,
)

if TYPE_CHECKING:
    import matplotlib
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.axes3d import Axes3D  # type: ignore[import-untyped]

_COLOR_ARM = "#4878CF"
_COLOR_SELECTED = "#E87722"
_COLOR_TRACE = "#888888"  # wrist trace
_COLOR_CURRENT = "#AAAAAA"  # current MPC arm state
_ELEV = 120
_AZIM = -90


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


def _draw_body(
    ax: "Axes3D", body_pos: np.ndarray, arm_color: str
) -> tuple[list, object]:
    """Draw full body on ax.

    Returns (arm_bone_lines, arm_joint_scatter).
    """
    # Grey non-arm skeleton
    _draw_bones_3d(ax, body_pos, _BODY_BONES, _BODY_COLOR, alpha=0.45, lw=1.2)
    ax.scatter(
        *body_pos[_BODY_JOINTS].T,
        color=_BODY_COLOR,
        s=14,
        alpha=0.45,
        depthshade=False,
    )

    # Coloured arm skeleton (mutable for highlight)
    arm_lines = []
    for pi, ci in LEFT_ARM_BONE_PAIRS_22:
        seg = body_pos[[pi, ci]]
        (ln,) = ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=arm_color, linewidth=2.2)
        arm_lines.append(ln)
    arm_scat = ax.scatter(
        *body_pos[LEFT_ARM_JOINT_INDICES_22].T,
        c=arm_color,
        s=35,
        depthshade=False,
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
) -> tuple["Figure", list["Axes3D"], list[list], list]:
    n_clusters = len(unique_labels)
    fig_w = max(4 * n_clusters, 8)
    fig = plt.figure(figsize=(fig_w, 5.5))
    fig.patch.set_facecolor("#F5F5F5")

    gs = fig.add_gridspec(
        1, n_clusters, bottom=0.18, top=0.92, left=0.04, right=0.96, wspace=0.05
    )
    axes = [fig.add_subplot(gs[0, i], projection="3d") for i in range(n_clusters)]

    panel_arm_lines: list[list] = []
    panel_arm_scats: list = []

    for idx, (k, body_cutoff, wrist_trace, count) in enumerate(
        zip(unique_labels, cluster_body_cutoffs, cluster_wrist_traces, cluster_counts)
    ):
        ax = axes[idx]
        ax.view_init(elev=_ELEV, azim=_AZIM)  # type: ignore[attr-defined]
        ax.set_xlim(*lims[0])
        ax.set_ylim(*lims[1])
        ax.set_zlim(*lims[2])  # type: ignore[attr-defined]
        ax.set_xlabel("X", fontsize=7)
        ax.set_ylabel("Y", fontsize=7)
        ax.set_zlabel("Z", fontsize=7)  # type: ignore[attr-defined]
        ax.tick_params(labelsize=6)
        ax.set_title(f"Cluster {k}\n({count} samples)", fontsize=9, pad=4)

        # Wrist trace (static grey)
        ax.plot(
            wrist_trace[:, 0],
            wrist_trace[:, 1],
            wrist_trace[:, 2],
            linestyle=":",
            color=_COLOR_TRACE,
            linewidth=1.0,
            alpha=0.7,
        )

        # Individual ghost arms (one per sample in cluster), very faint
        if cluster_individual_previews is not None:
            for body_ind in cluster_individual_previews[idx]:
                _draw_bones_3d(
                    ax,
                    body_ind,
                    LEFT_ARM_BONE_PAIRS_22,
                    _COLOR_ARM,
                    alpha=0.12,
                    lw=1.2,
                )

        # Current MPC arm state (grey, drawn behind the cluster arm)
        if current_body is not None:
            _draw_bones_3d(
                ax,
                current_body,
                LEFT_ARM_BONE_PAIRS_22,
                _COLOR_CURRENT,
                alpha=0.9,
                lw=2.2,
            )
            ax.scatter(  # type: ignore[misc]
                *current_body[LEFT_ARM_JOINT_INDICES_22].T,
                color=_COLOR_CURRENT,
                s=28,
                depthshade=False,
            )

        # Solid mean arm at trajectory-fraction cutoff (the pose that will be enqueued)
        arm_lines, arm_scat = _draw_body(ax, body_cutoff, _COLOR_ARM)
        panel_arm_lines.append(arm_lines)
        panel_arm_scats.append(arm_scat)

    return fig, axes, panel_arm_lines, panel_arm_scats


def pick_cluster(  # pylint: disable=too-many-locals,redefined-outer-name,too-many-statements
    trajectories: np.ndarray,
    labels: np.ndarray,
    fk: SmplLeftArmFK | None = None,
    save_path: str | Path | None = None,
    trajectory_fraction: float = 0.75,
    spine_pos: np.ndarray | None = None,
    spine_aa: np.ndarray | None = None,
    body_pos: np.ndarray | None = None,
    current_arm_aa: np.ndarray | None = None,
) -> int:
    """Show a blocking window to let the user pick a trajectory cluster.

    Each cluster panel shows:

    * Faint individual arms at the ``trajectory_fraction`` cutoff frame —
      one per sample, showing within-cluster spread.
    * Current MPC arm state (grey), so the user can see where the arm is now.
    * Solid mean arm at the ``trajectory_fraction`` cutoff frame — the pose
      that will be enqueued when this cluster is selected.
    * Dotted wrist trace of the cluster mean trajectory.

    Args:
        trajectories:        ``(num_samples, n_frames, 4, 3)`` axis-angle batch.
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
        current_arm_aa:      ``(4, 3)`` current MPC arm axis-angles.  When
                             provided, the current arm pose is drawn in grey
                             on every cluster panel.

    Returns:
        The integer cluster label chosen by the user.
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
    cluster_counts: list[int] = []

    for k in unique_labels:
        mask = labels == k
        mean_traj = trajectories[mask].mean(axis=0)  # (n_frames, 4, 3)
        n_frames = mean_traj.shape[0]
        # Mean body at trajectory-fraction cutoff frame (the pose that gets enqueued)
        preview_idx = max(0, round(n_frames * trajectory_fraction) - 1)
        body_cutoff = _merge_arm(
            fk.full_body_positions(mean_traj[preview_idx], spine_pos, spine_aa),
            body_pos,
        )  # (22, 3)
        # Per-sample ghost arm body poses at the cutoff frame
        individual_previews = [
            _merge_arm(
                fk.full_body_positions(traj[preview_idx], spine_pos, spine_aa), body_pos
            )  # (22, 3)
            for traj in trajectories[mask]
        ]
        # Wrist trace from arm-chain FK
        arm_positions = fk.fk_batch(mean_traj)  # (n_frames, 5, 3)
        wrist_trace = arm_positions[:, -1, :]  # (n_frames, 3)

        cluster_body_cutoffs.append(body_cutoff)
        cluster_individual_previews.append(individual_previews)
        cluster_wrist_traces.append(wrist_trace)
        cluster_counts.append(int(mask.sum()))

    # Current MPC arm state (same on every cluster panel)
    current_body: np.ndarray | None = None
    if current_arm_aa is not None:
        current_body = _merge_arm(
            fk.full_body_positions(current_arm_aa, spine_pos, spine_aa), body_pos
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
    fig, axes, panel_arm_lines, panel_arm_scats = _build_figure(
        unique_labels,
        cluster_body_cutoffs,
        cluster_wrist_traces,
        cluster_counts,
        lims,
        cluster_individual_previews=cluster_individual_previews,
        current_body=current_body,
    )

    # ------------------------------------------------------------------
    # State + interaction
    # ------------------------------------------------------------------
    state: dict = {"selected": None}

    def _set_selected(idx: int) -> None:
        state["selected"] = idx
        for i, (arm_lines, arm_scat) in enumerate(
            zip(panel_arm_lines, panel_arm_scats)
        ):
            color = _COLOR_SELECTED if i == idx else _COLOR_ARM
            for ln in arm_lines:
                ln.set_color(color)
            arm_scat.set_color(color)
            axes[i].set_facecolor("#FFF3E0" if i == idx else "white")
        fig.canvas.draw_idle()

    def _on_click(event: "matplotlib.backend_bases.MouseEvent") -> None:
        if event.inaxes is None:
            return
        for i, ax in enumerate(axes):
            if event.inaxes is ax:
                _set_selected(i)
                return

    fig.canvas.mpl_connect("button_press_event", _on_click)

    btn_ax = fig.add_axes([0.38, 0.04, 0.24, 0.08])
    btn = Button(btn_ax, "Confirm", color="#DDDDDD", hovercolor="#BBBBBB")

    def _on_confirm(_event: object) -> None:
        if state["selected"] is None:
            return
        plt.close(fig)

    btn.on_clicked(_on_confirm)
    fig.suptitle(
        "Click a cluster to select it, then click Confirm", fontsize=10, y=0.97
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show(block=True)

    if state["selected"] is None:
        raise RuntimeError("Window closed without selecting a cluster.")

    return unique_labels[state["selected"]]


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

        fig, axes, _, _ = _build_figure(
            unique_labels,
            cluster_body_finals,
            cluster_wrist_traces,
            cluster_counts,
            lims,
        )
        btn_ax = fig.add_axes([0.38, 0.04, 0.24, 0.08])
        btn_ax.set_facecolor("#DDDDDD")
        btn_ax.text(0.5, 0.5, "Confirm", ha="center", va="center", fontsize=10)
        btn_ax.set_xticks([])
        btn_ax.set_yticks([])
        fig.suptitle(
            "Click a cluster to select it, then click Confirm", fontsize=10, y=0.97
        )
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Saved → {save.resolve()}")
        sys.exit(0)

    chosen = pick_cluster(raw, demo_labels, save_path=save)
    print(f"User chose cluster {chosen}")
