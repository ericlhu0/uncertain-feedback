"""Plotting utilities for the SMPL left arm.

All plotting functionality lives on :class:`ArmVisualizer`, either as instance
methods for stateful MPC visualization or as static methods for one-shot drawing.

Static utilities (no instantiation required)::

    import matplotlib.pyplot as plt
    import numpy as np
    from uncertain_feedback.utils.plot import ArmVisualizer

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ArmVisualizer.draw_smpl_skeleton(ax, positions)          # (22, 3) joint positions
    ArmVisualizer.format_3d_axis(ax, positions[:, [0, 2, 1]])
    ArmVisualizer.draw_bones_3d(ax, positions, bone_pairs, color="blue")

Live MPC visualization::

    from uncertain_feedback.planners.mpc import SmplLeftArmMPC, SmplLeftArmFK
    from uncertain_feedback.utils.plot import ArmVisualizer

    fk  = SmplLeftArmFK()
    mpc = SmplLeftArmMPC(horizon=10, n_mpc_samples=512)
    vis = ArmVisualizer(fk)

    initial_q = np.zeros((3, 3))
    target_q  = np.array([[0, 0.5, 0], [0, 0, 0.4], [0, 0, 0]])

    fig, anim = vis.animate(mpc, initial_q, target_q, n_steps=40)
    plt.show()
    # or: anim.save("arm.gif", writer="pillow")
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import warnings
from typing import TYPE_CHECKING, NamedTuple

import matplotlib
import numpy as np
from pathlib import Path

# Select an interactive backend when running as a script; skip if one is
# already active (e.g. when imported inside Jupyter).
if matplotlib.get_backend().lower() in ("agg", ""):
    for _backend in ("Qt5Agg", "TkAgg", "Qt6Agg", "WXAgg", "MacOSX"):
        try:
            matplotlib.use(_backend, force=True)
            if matplotlib.get_backend().lower() not in ("agg", ""):
                break
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    else:
        print(
            "WARNING: no interactive matplotlib backend found — run: uv add pyqt5",
            file=sys.stderr,
        )
# pylint: disable=wrong-import-position,ungrouped-imports
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from uncertain_feedback.planners.mpc.kinematics import (
    LEFT_ARM_BONE_PAIRS_22,
    LEFT_ARM_JOINT_INDICES_22,
    SMPL_BONE_PAIRS_22,
    SmplLeftArmFK,
)

# pylint: enable=wrong-import-position

if TYPE_CHECKING:
    from uncertain_feedback.planners.mpc.arm_mpc import SmplLeftArmMPC

# Internal style constants
_TRACE_COLOR = "cornflowerblue"
_ELBOW_RANGE_COLOR = "red"
_WRIST_IDX = 20  # left_wrist in the 22-joint array

# 3D camera angles: (title, elev, azim)
_3D_VIEWS = [
    ("Perspective", 45, -60),
    ("p2", 45, 60),
    ("p3", 45, -120),
]

# Single view used in compact (1-panel) mode — overhead diagonal
_COMPACT_VIEW = ("Perspective", 65, -90)


# 2D orthographic projections
class _OrthoView(NamedTuple):
    title: str
    hi: int  # horizontal axis index
    vi: int  # vertical axis index
    hl: str  # horizontal label
    vl: str  # vertical label


# Axis labels carry the body-relative meaning of each world axis (SMPL convention,
# upright body: +X = person's left, +Y = up, +Z = person's front) so an LLM reading
# the overlays can resolve "out to the side" vs "across the body" without guessing
# the handedness of the projection.
_ORTHO_VIEWS = [
    _OrthoView(
        "Front (XY)", 0, 1, "X (m), + = person's left", "Y (m), + = up"
    ),
    _OrthoView(
        "Side (ZY)", 2, 1, "Z (m), + = person's front", "Y (m), + = up"
    ),
    _OrthoView(
        "Top (XZ)", 0, 2, "X (m), + = person's left", "Z (m), + = person's front"
    ),
]


@dataclasses.dataclass
class _LiveState:  # pylint: disable=too-many-instance-attributes
    """Mutable state for the interactive live window."""

    fig: plt.Figure
    artists3d: list[dict]
    artists2d: list[dict]
    spine_pos: np.ndarray | None
    spine_aa: np.ndarray | None
    elbow_height_range: tuple[float, float] | None = None
    wrist_trace: list = dataclasses.field(default_factory=list)
    recorded_frames: list = dataclasses.field(default_factory=list)
    step: int = 0


class ArmVisualizer:  # pylint: disable=too-many-instance-attributes
    """Animate the full SMPL skeleton with the left arm driven by MPC.

    All plotting utilities are available as static methods and do not require
    an instance.  The instance methods handle stateful live visualization.

    Args:
        fk:            :class:`SmplLeftArmFK` instance.  If ``None``, one is
                       created with the default SMPL model path.
        smpl_pkl_path: Passed to :class:`SmplLeftArmFK` when ``fk`` is ``None``.
    """

    # Public color / geometry constants
    BODY_COLOR: str = "#aaaaaa"
    TARGET_COLOR: str = "royalblue"
    MDM_COLOR: str = "darkorange"
    BODY_BONES: list = [p for p in SMPL_BONE_PAIRS_22 if p not in LEFT_ARM_BONE_PAIRS_22]
    BODY_JOINTS: list = [j for j in range(22) if j not in LEFT_ARM_JOINT_INDICES_22]

    # ------------------------------------------------------------------
    # Static drawing utilities
    # ------------------------------------------------------------------

    @staticmethod
    def draw_bones_3d(
        ax: Axes3D,
        positions: np.ndarray,
        bone_pairs: list[tuple[int, int]],
        color: str,
        alpha: float = 1.0,
        lw: float = 2.0,
        linestyle: str = "-",
        label: str | None = None,
    ) -> None:
        """Draw line segments connecting joints in 3D."""
        for i, (pi, ci) in enumerate(bone_pairs):
            seg = positions[[pi, ci]]
            ax.plot(
                seg[:, 0],
                seg[:, 1],
                seg[:, 2],
                color=color,
                alpha=alpha,
                linewidth=lw,
                linestyle=linestyle,
                label=label if i == 0 else None,
            )

    @staticmethod
    def format_3d_axis(ax: Axes3D, points: np.ndarray) -> None:
        """Set equal-aspect cube limits on a 3D axes to fit *points*.

        Args:
            ax:     Matplotlib 3D axes to update.
            points: ``(N, 3)`` array of points in the axes' coordinate system.
        """
        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)
        center = (mins + maxs) / 2.0
        radius = max(float(np.max(maxs - mins)) / 2.0, 0.05)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)

    @staticmethod
    def draw_smpl_skeleton(
        ax: Axes3D,
        positions: np.ndarray,
        title: str = "",
        highlight_joints: set | None = None,
    ) -> None:
        """Draw a 22-joint SMPL skeleton on a 3D axes.

        Positions are in SMPL Y-up world coordinates; the plot displays
        with Z up (``xlabel="X"``, ``ylabel="Z"``, ``zlabel="Y"``).

        Args:
            ax:               Matplotlib 3D axes.
            positions:        ``(22, 3)`` joint world positions (SMPL Y-up).
            title:            Axes title.
            highlight_joints: Joint indices to draw in arm color ``#e05c2a``.
                              All others use body color ``#4a90d9``.
        """
        highlight_joints = highlight_joints or set()
        arm_color = "#e05c2a"
        body_color = "#4a90d9"

        # Draw bones
        for parent, child in SMPL_BONE_PAIRS_22:
            is_arm = parent in highlight_joints or child in highlight_joints
            color = arm_color if is_arm else body_color
            lw = 2.5 if is_arm else 1.5
            ax.plot(
                [positions[parent, 0], positions[child, 0]],
                [positions[parent, 2], positions[child, 2]],
                [positions[parent, 1], positions[child, 1]],
                color=color,
                linewidth=lw,
            )

        # Draw joints
        for j in range(22):
            is_arm = j in highlight_joints
            ax.scatter(
                positions[j, 0],
                positions[j, 2],
                positions[j, 1],
                c=arm_color if is_arm else body_color,
                s=40 if is_arm else 20,
                zorder=5,
            )

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y")
        ax.view_init(elev=10, azim=-60)

        # Equal-aspect ratio using XZY-reordered points to match plot axes
        ArmVisualizer.format_3d_axis(ax, positions[:, [0, 2, 1]])

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(
        self,
        fk: SmplLeftArmFK | None = None,
        smpl_pkl_path: str | None = None,
    ) -> None:
        self.fk = fk if fk is not None else SmplLeftArmFK(smpl_pkl_path)
        # Live-window state; populated by open_live()
        self._live: _LiveState | None = None
        # Pre-captured RGB frames for fast video saving
        self._capture_video: bool = False
        self._frame_bufs: list = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_capture(self) -> None:
        """Enable per-step frame buffering so finish_live() can use imageio."""
        self._capture_video = True
        self._frame_bufs = []

    def prepend_frames(self, frames: list) -> None:
        """Prepend pre-captured frames (e.g. from before MDM generation)."""
        self._frame_bufs = list(frames) + self._frame_bufs

    def close(self) -> None:
        """Close the live window if open."""
        if self._live is not None:
            plt.close(self._live.fig)

    def animate(
        self,
        mpc: SmplLeftArmMPC,
        initial_q: np.ndarray,
        target_q: np.ndarray,
        n_steps: int = 50,
        spine3_pos: np.ndarray | None = None,
        spine3_aa: np.ndarray | None = None,
        interval: int = 120,
        save_path: str | None = None,
    ) -> tuple[plt.Figure, FuncAnimation]:
        """Run the MPC loop and return a matplotlib animation.

        Args:
            mpc:        Configured :class:`SmplLeftArmMPC`.
            initial_q:  ``(3, 3)`` controlled arm axis-angles.
            target_q:   ``(3, 3)`` target controlled arm axis-angles.
            n_steps:    Number of MPC steps to simulate.
            spine3_pos: ``(3,)`` world position of spine3.
            spine3_aa:  ``(3,)`` world axis-angle of spine3.
            interval:   Milliseconds between animation frames.
            save_path:  If given, save to this path (e.g. ``"arm.gif"``).

        Returns:
            ``(fig, anim)`` — the matplotlib Figure and FuncAnimation.
        """
        frames = self._run_mpc(mpc, initial_q, target_q, n_steps, spine3_pos, spine3_aa)

        fig, artists_3d, artists_2d = self._build_figure(
            target_q,
            _compute_lims(frames, self.fk, target_q, spine3_pos, spine3_aa),
            spine3_pos,
            spine3_aa,
        )

        update = _make_frame_updater(frames, artists_3d, artists_2d, n_steps)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.tight_layout()

        anim = FuncAnimation(
            fig, update, frames=len(frames), interval=interval, blit=False
        )

        if save_path is not None:
            _save(anim, save_path)

        return fig, anim

    def _full_body_positions(
        self,
        q: np.ndarray,
        spine3_pos: np.ndarray | None = None,
        spine3_aa: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute full 22-joint body positions from (3,3) arm axis-angles."""
        return self.fk.full_body_positions(
            np.asarray(q, dtype=np.float64), spine3_pos, spine3_aa
        )

    def open_live(
        self,
        target_q: np.ndarray,
        spine3_pos: np.ndarray | None = None,
        spine3_aa: np.ndarray | None = None,
        body_pos: np.ndarray | None = None,
        compact: bool = False,
        elbow_height_range: tuple[float, float] | None = None,
        show_target_arm: bool = True,
    ) -> None:
        """Open an interactive window for live step-by-step visualization.

        Call this once before the MPC loop, then call :meth:`update_step` each
        iteration.  Uses matplotlib's interactive mode so the window stays
        responsive while Python keeps running.

        Args:
            target_q:   ``(3, 3)`` target controlled arm axis-angles.
            spine3_pos: ``(3,)`` world position of spine3.
            spine3_aa:  ``(3,)`` world axis-angle of spine3.
            body_pos:   ``(22, 3)`` world positions for the static body
                        background skeleton.  When provided (e.g. sitting pose)
                        it replaces the default T-pose for the grey backdrop.
            compact:    If ``True``, build a single 3-D panel instead of the
                        full 6-panel layout.  Faster to render and encode.
            elbow_height_range: Optional ``(min_y, max_y)`` world-space Y bounds
                        for acceptable elbow height, shown as red planes.
            show_target_arm: If ``False``, omit the static dashed blue
                        joint-space target arm.  Cartesian planners use this
                        when the blue target should be the wrist star only.
        """
        target_full = self._full_body_positions(target_q, spine3_pos, spine3_aa)
        ref_body = body_pos if body_pos is not None else self.fk.tpose_all_joints

        # Use body reference + target to set axis limits
        all_pts = np.vstack([ref_body, target_full])
        mg = 0.15
        lims = [(all_pts[:, i].min() - mg, all_pts[:, i].max() + mg) for i in range(3)]
        if elbow_height_range is not None:
            low_y, high_y = elbow_height_range
            lims[1] = (
                min(lims[1][0], low_y - mg),
                max(lims[1][1], high_y + mg),
            )

        plt.ion()
        fig, artists_3d, artists_2d = self._build_figure(
            target_q,
            lims,
            spine3_pos,
            spine3_aa,
            body_pos=body_pos,
            compact=compact,
            elbow_height_range=elbow_height_range,
            show_target_arm=show_target_arm,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.tight_layout()
        plt.show(block=False)
        fig.canvas.draw()
        fig.canvas.flush_events()

        self._live = _LiveState(
            fig=fig,
            artists3d=artists_3d,
            artists2d=artists_2d,
            spine_pos=spine3_pos,
            spine_aa=spine3_aa,
            elbow_height_range=elbow_height_range,
        )

    def update_step(
        self,
        q: np.ndarray,
        dist: float,
        color: str = TARGET_COLOR,
    ) -> None:
        """Update the live window with the current joint configuration.

        Args:
            q:     ``(3, 3)`` current controlled arm axis-angle joint angles.
            dist:  Distance to target (for title).
            color: Arm color.  Pass ``ArmVisualizer.MDM_COLOR`` when following
                   an MDM trajectory so the arm is visually distinct.
        """
        assert self._live is not None
        pos = self._full_body_positions(q, self._live.spine_pos, self._live.spine_aa)
        arm_pts = pos[LEFT_ARM_JOINT_INDICES_22]
        self._live.wrist_trace.append(pos[_WRIST_IDX])
        trace_color = ArmVisualizer.MDM_COLOR if color == ArmVisualizer.MDM_COLOR else _TRACE_COLOR
        self._live.recorded_frames.append(
            {
                "positions": pos.copy(),
                "dist": dist,
                "color": color,
                "trace_color": trace_color,
            }
        )

        _update_artists(
            self._live.artists3d,
            self._live.artists2d,
            pos,
            arm_pts,
            np.array(self._live.wrist_trace),
            step=self._live.step,
            n_steps=None,
            dist=dist,
            color=color,
            trace_color=trace_color,
        )
        self._live.step += 1

        self._live.fig.canvas.draw_idle()
        self._live.fig.canvas.flush_events()
        plt.pause(0.0001)

        if self._capture_video:
            self._live.fig.canvas.draw()
            w, h = self._live.fig.canvas.get_width_height()
            buf = np.frombuffer(self._live.fig.canvas.buffer_rgba(), dtype=np.uint8)
            self._frame_bufs.append(buf.reshape(h, w, 4)[..., :3].copy())

    def update_mdm_goal(self, goal_q: np.ndarray) -> None:
        """Draw (or update) the MDM end-of-trajectory goal marker.

        Computes full body positions for ``goal_q`` and updates the dashed
        arm skeleton artists in every panel.  Safe to call multiple times.

        Args:
            goal_q: ``(3, 3)`` axis-angle joint angles for the MDM trajectory's
                    last frame ``[left_shoulder, left_elbow, left_wrist]``.
        """
        assert self._live is not None, "update_mdm_goal() called before open_live()"
        goal_full = self._full_body_positions(
            goal_q, self._live.spine_pos, self._live.spine_aa
        )
        arm_pts = goal_full[LEFT_ARM_JOINT_INDICES_22]

        for a3 in self._live.artists3d:
            a3["mdm_goal_scat"]._offsets3d = (  # pylint: disable=protected-access
                arm_pts[:, 0],
                arm_pts[:, 1],
                arm_pts[:, 2],
            )
            for line, (pi, ci) in zip(a3["mdm_goal_lines"], LEFT_ARM_BONE_PAIRS_22):
                seg = goal_full[[pi, ci]]
                line.set_data(seg[:, 0], seg[:, 1])
                line.set_3d_properties(seg[:, 2])

        for a2 in self._live.artists2d:
            a2["mdm_goal_scat"].set_offsets(arm_pts[:, [a2["hi"], a2["vi"]]])
            for line, (pi, ci) in zip(a2["mdm_goal_lines"], LEFT_ARM_BONE_PAIRS_22):
                seg = goal_full[[pi, ci]]
                line.set_data(seg[:, a2["hi"]], seg[:, a2["vi"]])

        self._live.fig.canvas.draw_idle()
        self._live.fig.canvas.flush_events()

    def update_trajectory_preview(self, preview_q: np.ndarray) -> None:
        """Draw (or update) a semi-transparent ghost arm at the trajectory
        cutoff frame.

        Called automatically by
        :meth:`~uncertain_feedback.planners.mpc.arm_mpc_mdm.LeftArmMPCMDM.push_trajectory`
        to show the arm pose at the enqueued cutoff timestep (e.g. 75 % through
        the generated trajectory).  Safe to call multiple times.

        Args:
            preview_q: ``(3, 3)`` axis-angle joint angles for the cutoff frame
                       ``[left_shoulder, left_elbow, left_wrist]``.
        """
        assert (
            self._live is not None
        ), "update_trajectory_preview() called before open_live()"
        preview_full = self._full_body_positions(
            preview_q, self._live.spine_pos, self._live.spine_aa
        )
        arm_pts = preview_full[LEFT_ARM_JOINT_INDICES_22]

        for a3 in self._live.artists3d:
            a3["preview_scat"]._offsets3d = (  # pylint: disable=protected-access
                arm_pts[:, 0],
                arm_pts[:, 1],
                arm_pts[:, 2],
            )
            for line, (pi, ci) in zip(a3["preview_lines"], LEFT_ARM_BONE_PAIRS_22):
                seg = preview_full[[pi, ci]]
                line.set_data(seg[:, 0], seg[:, 1])
                line.set_3d_properties(seg[:, 2])

        for a2 in self._live.artists2d:
            a2["preview_scat"].set_offsets(arm_pts[:, [a2["hi"], a2["vi"]]])
            for line, (pi, ci) in zip(a2["preview_lines"], LEFT_ARM_BONE_PAIRS_22):
                seg = preview_full[[pi, ci]]
                line.set_data(seg[:, a2["hi"]], seg[:, a2["vi"]])

        self._live.fig.canvas.draw_idle()
        self._live.fig.canvas.flush_events()

    def update_cartesian_target(self, world_pos: np.ndarray) -> None:
        """Draw (or move) the blue star marking the Cartesian wrist goal.

        Args:
            world_pos: ``(3,)`` world-space position of the target wrist point.
        """
        assert (
            self._live is not None
        ), "update_cartesian_target() called before open_live()"
        for a3 in self._live.artists3d:
            a3["cartesian_goal_scat"]._offsets3d = (  # pylint: disable=protected-access
                [world_pos[0]],
                [world_pos[1]],
                [world_pos[2]],
            )
        for a2 in self._live.artists2d:
            a2["cartesian_goal_scat"].set_offsets(
                [[world_pos[a2["hi"]], world_pos[a2["vi"]]]]
            )
        self._live.fig.canvas.draw_idle()
        self._live.fig.canvas.flush_events()

    def update_elbow_height_range(
        self,
        elbow_height_range: tuple[float, float] | None,
    ) -> None:
        """Draw or update red planes marking acceptable elbow-height bounds.

        Args:
            elbow_height_range: ``(min_y, max_y)`` world-space Y bounds, or
                ``None`` to hide the range.
        """
        assert (
            self._live is not None
        ), "update_elbow_height_range() called before open_live()"
        self._live.elbow_height_range = elbow_height_range
        _update_elbow_height_artists(
            self._live.artists3d,
            self._live.artists2d,
            elbow_height_range,
        )
        self._live.fig.canvas.draw_idle()
        self._live.fig.canvas.flush_events()

    def render_rollout_video(
        self,
        rollout: np.ndarray,
        save_path: str | Path,
        spine3_pos: np.ndarray | None = None,
        spine3_aa: np.ndarray | None = None,
        body_pos: np.ndarray | None = None,
        cartesian_goal: np.ndarray | None = None,
        frame_colors: list[str] | None = None,
        mdm_goal_q: np.ndarray | None = None,
        fps: int = 20,
    ) -> None:
        """Render a pre-computed rollout to video using the 6-panel MPC layout.

        Args:
            rollout:         ``(T, 3, 3)`` arm axis-angle frames.
            save_path:       Output path (.mp4 or .gif).
            spine3_pos:      ``(3,)`` spine3 world position.
            spine3_aa:       ``(3,)`` spine3 world axis-angle.
            body_pos:        ``(22, 3)`` static body backdrop positions.
            cartesian_goal:  ``(3,)`` Cartesian wrist goal offset relative to
                             spine3 (same convention as the MPC planner).
            frame_colors:    Per-frame arm color strings (len == len(rollout)).
                             Defaults to ``TARGET_COLOR`` for all frames.
            mdm_goal_q:      ``(3, 3)`` arm axis-angles for the MDM trajectory
                             end-frame, shown as an orange dashed skeleton.
            fps:             Frames per second.
        """
        import imageio  # pylint: disable=import-outside-toplevel
        from matplotlib.backends.backend_agg import FigureCanvasAgg  # pylint: disable=import-outside-toplevel
        from matplotlib.figure import Figure as _MplFigure  # pylint: disable=import-outside-toplevel

        save_path = Path(save_path)
        try:
            all_pos = np.stack([
                self.fk.full_body_positions(rollout[i], spine3_pos, spine3_aa)
                for i in range(len(rollout))
            ])  # (T, 22, 3)
            ref_body = body_pos if body_pos is not None else self.fk.tpose_all_joints
            mg = 0.15
            all_pts = np.vstack([all_pos.reshape(-1, 3), ref_body])
            lims = [(float(all_pts[:, i].min()) - mg, float(all_pts[:, i].max()) + mg) for i in range(3)]

            # Create figure with Agg canvas directly — avoids touching global pyplot
            # state (no matplotlib.use() call), making this safe to call from threads.
            agg_fig = _MplFigure(figsize=(20, 9))
            FigureCanvasAgg(agg_fig)

            # Use MDM goal as the static dashed reference (matches live visualizer),
            # or the initial pose when no MDM goal is given (invisible at frame 0).
            target_q = mdm_goal_q if mdm_goal_q is not None else rollout[0]
            fig, artists_3d, artists_2d = self._build_figure(
                target_q,
                lims,
                spine3_pos,
                spine3_aa,
                body_pos=body_pos,
                fig=agg_fig,
                show_target_arm=cartesian_goal is None,
            )

            if cartesian_goal is not None:
                s3 = np.asarray(spine3_pos, dtype=np.float64) if spine3_pos is not None else self.fk.tpose_spine3_pos
                world_goal = s3 + np.asarray(cartesian_goal, dtype=np.float64)
                for a3 in artists_3d:
                    a3["cartesian_goal_scat"]._offsets3d = (  # pylint: disable=protected-access
                        [world_goal[0]], [world_goal[1]], [world_goal[2]]
                    )
                for a2 in artists_2d:
                    a2["cartesian_goal_scat"].set_offsets([[world_goal[a2["hi"]], world_goal[a2["vi"]]]])

            if mdm_goal_q is not None:
                goal_full = self._full_body_positions(mdm_goal_q, spine3_pos, spine3_aa)
                arm_pts = goal_full[LEFT_ARM_JOINT_INDICES_22]
                for a3 in artists_3d:
                    a3["mdm_goal_scat"]._offsets3d = (  # pylint: disable=protected-access
                        arm_pts[:, 0], arm_pts[:, 1], arm_pts[:, 2]
                    )
                    for line, (pi, ci) in zip(a3["mdm_goal_lines"], LEFT_ARM_BONE_PAIRS_22):
                        seg = goal_full[[pi, ci]]
                        line.set_data(seg[:, 0], seg[:, 1])
                        line.set_3d_properties(seg[:, 2])
                for a2 in artists_2d:
                    a2["mdm_goal_scat"].set_offsets(arm_pts[:, [a2["hi"], a2["vi"]]])
                    for line, (pi, ci) in zip(a2["mdm_goal_lines"], LEFT_ARM_BONE_PAIRS_22):
                        seg = goal_full[[pi, ci]]
                        line.set_data(seg[:, a2["hi"]], seg[:, a2["vi"]])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig.tight_layout()

            wrist_trace: list[np.ndarray] = []
            frames_out: list[np.ndarray] = []
            n_steps = len(rollout) - 1
            for i, q in enumerate(rollout):
                pos = all_pos[i]
                arm_pts = pos[LEFT_ARM_JOINT_INDICES_22]
                wrist_trace.append(pos[_WRIST_IDX])
                color = frame_colors[i] if frame_colors is not None else ArmVisualizer.TARGET_COLOR
                trace_color = ArmVisualizer.MDM_COLOR if color == ArmVisualizer.MDM_COLOR else _TRACE_COLOR
                _update_artists(
                    artists_3d, artists_2d,
                    pos, arm_pts,
                    np.array(wrist_trace),
                    step=i, n_steps=n_steps,
                    dist=float(np.linalg.norm(q - rollout[-1])),
                    color=color,
                    trace_color=trace_color,
                )
                fig.canvas.draw()
                w, h = fig.canvas.get_width_height()
                buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                frames_out.append(buf.reshape(h, w, 4)[..., :3].copy())

            save_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(str(save_path), frames_out, fps=fps)
            print(f"[rollout-video] saved {save_path}")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[rollout-video] failed to render {save_path}: {exc}")

    def render_body_trajectory_video(
        self,
        positions: np.ndarray,
        save_path: str | Path,
        fps: int = 30,
        highlight_joints: set | None = None,
    ) -> None:
        """Render a full-body SMPL skeleton trajectory to video.

        Unlike :meth:`render_rollout_video` (left arm only, over a static body
        backdrop), every joint is animated so the whole body moves.

        Args:
            positions:        ``(T, 22, 3)`` SMPL joint world positions (Y-up).
            save_path:        Output path (.mp4 or .gif).
            fps:              Frames per second.
            highlight_joints: Joints drawn in arm color (defaults to left arm).
        """
        import imageio  # pylint: disable=import-outside-toplevel
        from matplotlib.backends.backend_agg import FigureCanvasAgg  # pylint: disable=import-outside-toplevel
        from matplotlib.figure import Figure as _MplFigure  # pylint: disable=import-outside-toplevel

        positions = np.asarray(positions, dtype=np.float64)
        if highlight_joints is None:
            highlight_joints = set(LEFT_ARM_JOINT_INDICES_22)
        save_path = Path(save_path)

        # Fixed limits across all frames (XZY order matches draw_smpl_skeleton)
        # so the camera/scale stays steady while the body moves.
        pts = positions.reshape(-1, 3)[:, [0, 2, 1]]
        center = (pts.min(0) + pts.max(0)) / 2.0
        radius = max(float((pts.max(0) - pts.min(0)).max()) / 2.0, 0.05)

        agg_fig = _MplFigure(figsize=(7, 7))
        FigureCanvasAgg(agg_fig)
        ax = agg_fig.add_subplot(111, projection="3d")

        frames_out: list[np.ndarray] = []
        for t in range(len(positions)):
            ax.clear()
            ArmVisualizer.draw_smpl_skeleton(
                ax, positions[t], title=f"frame {t}", highlight_joints=highlight_joints
            )
            ax.set_xlim(center[0] - radius, center[0] + radius)
            ax.set_ylim(center[1] - radius, center[1] + radius)
            ax.set_zlim(center[2] - radius, center[2] + radius)
            agg_fig.canvas.draw()
            w, h = agg_fig.canvas.get_width_height()
            buf = np.frombuffer(agg_fig.canvas.buffer_rgba(), dtype=np.uint8)
            frames_out.append(buf.reshape(h, w, 4)[..., :3].copy())

        save_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(save_path), frames_out, fps=fps)
        print(f"[body-video] saved {save_path}")

    def render_trajectory_overlay(
        self,
        save_path: str | Path,
        *,
        mdm_traj: np.ndarray,
        current_q: np.ndarray,
        spine3_pos: np.ndarray,
        spine3_aa: np.ndarray,
        body_pos: np.ndarray | None = None,
    ) -> None:
        """Render the 3-view MDM-trajectory overlay used to ground LLM cost prompts.

        Args:
            save_path:  Output image path (.png).
            mdm_traj:   ``(T, 3, 3)`` MDM arm axis-angle frames.
            current_q:  ``(3, 3)`` current arm axis-angle state.
            spine3_pos: ``(3,)`` spine3 world position.
            spine3_aa:  ``(3,)`` spine3 world axis-angle.
            body_pos:   ``(22, 3)`` reference body positions; falls back to a
                        translated T-pose when ``None``.
        """
        save_path = Path(save_path)
        positions = self.fk.fk_batch(
            mdm_traj, spine3_pos, spine3_aa
        )  # (T, 5, 3) arm chain — wrist path / markers
        current_positions = self.fk.fk(current_q, spine3_pos, spine3_aa)

        # Reference body: actual body pose if available, else translated T-pose.
        if body_pos is not None:
            ref_body = body_pos
        else:
            ref_body = self.fk.tpose_all_joints + (spine3_pos - self.fk.tpose_spine3_pos)

        cur_full = self.fk.full_body_positions(current_q, spine3_pos, spine3_aa)

        # Equal-square axis limits across all three axes (matches format_3d_axis).
        all_pts = np.concatenate(
            [ref_body, positions.reshape(-1, 3), current_positions], axis=0
        )
        mins = np.min(all_pts, axis=0)
        maxs = np.max(all_pts, axis=0)
        center = (mins + maxs) / 2.0
        radius = max(float(np.max(maxs - mins)) / 2.0, 0.05)
        lims = [(center[i] - radius, center[i] + radius) for i in range(3)]

        n_samples = min(12, positions.shape[0])
        sample_indices = (
            np.linspace(0, positions.shape[0] - 1, n_samples).round().astype(int)
        )
        cmap = plt.get_cmap("Blues")
        denom = max(1, positions.shape[0] - 1)
        wrist_chain_idx = 4  # left_wrist in the 5-joint arm chain
        wrist_path = positions[:, wrist_chain_idx]
        start_w = positions[0, wrist_chain_idx]
        end_w = positions[-1, wrist_chain_idx]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, view in zip(axes, _ORTHO_VIEWS):
            ax.set_aspect("equal")
            ax.set_title(view.title, fontsize=9)
            ax.set_xlabel(view.hl, fontsize=8)
            ax.set_ylabel(view.vl, fontsize=8)
            ax.set_xlim(*lims[view.hi])
            ax.set_ylim(*lims[view.vi])
            ax.tick_params(labelsize=7)

            # Static reference body (grey)
            _draw_bones_2d(ax, ref_body, ArmVisualizer.BODY_BONES, view.hi, view.vi,
                           ArmVisualizer.BODY_COLOR, alpha=0.45, lw=1.2)

            # MDM trajectory arm bones (blue gradient, sampled frames)
            for frame_idx in sample_indices:
                t = 0.3 + 0.7 * (frame_idx / denom)
                full = self.fk.full_body_positions(
                    mdm_traj[frame_idx], spine3_pos, spine3_aa
                )
                _draw_bones_2d(ax, full, LEFT_ARM_BONE_PAIRS_22, view.hi, view.vi,
                               cmap(t), alpha=0.5, lw=1.2)

            # Wrist path and start/end markers
            ax.plot(wrist_path[:, view.hi], wrist_path[:, view.vi],
                    color="steelblue", alpha=0.5, linewidth=1.0)
            ax.scatter(start_w[view.hi], start_w[view.vi], marker="o", color="lime", s=55, zorder=5)
            ax.scatter(end_w[view.hi], end_w[view.vi], marker="X", color="red", s=65, zorder=5)

            # Current pose arm (orange)
            _draw_bones_2d(ax, cur_full, LEFT_ARM_BONE_PAIRS_22, view.hi, view.vi,
                           "tab:orange", alpha=1.0, lw=2.2)

        scalar_mappable = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(vmin=0, vmax=positions.shape[0] - 1)
        )
        scalar_mappable.set_array([])
        fig.colorbar(scalar_mappable, ax=axes[-1], shrink=0.8, pad=0.04,
                     label="frame (light=early, dark=late)")
        axes[0].legend(
            handles=[
                plt.Line2D([0], [0], color="tab:orange", linewidth=2, label="current"),
                plt.Line2D([0], [0], marker="o", color="lime", linestyle="", markersize=7, label="traj start"),
                plt.Line2D([0], [0], marker="X", color="red", linestyle="", markersize=7, label="traj end"),
            ],
            fontsize=7, loc="upper left",
        )
        fig.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    def render_cluster_contrast_overlay(
        self,
        save_path: str | Path,
        *,
        mdm_trajs: dict[int, np.ndarray],
        highlight_label: int,
        current_q: np.ndarray,
        spine3_pos: np.ndarray,
        spine3_aa: np.ndarray,
        body_pos: np.ndarray | None = None,
        reference_traj: np.ndarray | None = None,
        goal_pos: np.ndarray | None = None,
        include_others: bool = True,
        include_reference: bool = True,
    ) -> None:
        """Render a shared-axis overlay anchored on the highlighted cluster.

        The ``highlight_label`` cluster is always drawn as the full blue-gradient arm
        (sampled frames + wrist path); its start/end markers are intentionally omitted.
        The orange current pose (the shared start every candidate departs from) and the
        gold goal star are always drawn.

        ``include_others`` adds only the terminal full-arm pose and wrist marker for
        every OTHER cluster. ``include_reference`` adds the original-goal reference as
        a green-gradient full arm plus a green dashed wrist path. Splitting these layers
        onto separate images keeps each one readable.

        Each image computes its own equal-square axis limits from only the layers it
        draws, so it is individually legible (the highlighted cluster may therefore sit
        at a slightly different scale across images).

        Args:
            save_path:         Output image path (.png).
            mdm_trajs:         ``{label: (T, 3, 3)}`` arm trajectories per cluster.
            highlight_label:   Key in ``mdm_trajs`` to render in full blue detail.
            current_q:         ``(3, 3)`` shared current arm axis-angle state.
            spine3_pos:        ``(3,)`` spine3 world position.
            spine3_aa:         ``(3,)`` spine3 world axis-angle.
            body_pos:          ``(22, 3)`` reference body; falls back to a translated
                               T-pose when ``None``.
            reference_traj:    ``(T, 3, 3)`` original-goal arm trajectory, or ``None``.
            goal_pos:          ``(3,)`` spine3-relative wrist goal, or ``None``.
            include_others:    Draw the other clusters' terminal full-arm poses
                               (default ``True``).
            include_reference: Draw the original-goal reference's full arm (default
                               ``True``); ignored when ``reference_traj`` is ``None``.
        """
        save_path = Path(save_path)
        wrist_chain_idx = 4  # left_wrist in the 5-joint arm chain
        draw_reference = include_reference and reference_traj is not None
        ref_traj_arr = (
            np.asarray(reference_traj, dtype=np.float64) if draw_reference else None
        )
        ref_positions = (
            self.fk.fk_batch(ref_traj_arr, spine3_pos, spine3_aa)
            if draw_reference
            else None
        )
        goal_world = (
            spine3_pos + np.asarray(goal_pos, dtype=np.float64)
            if goal_pos is not None
            else None
        )
        other_labels = (
            [label for label in mdm_trajs if label != highlight_label]
            if include_others
            else []
        )
        hi_traj = np.asarray(mdm_trajs[highlight_label], dtype=np.float64)
        hi_positions = self.fk.fk_batch(hi_traj, spine3_pos, spine3_aa)
        other_end_positions = {
            label: self.fk.fk(
                np.asarray(mdm_trajs[label][-1], dtype=np.float64),
                spine3_pos,
                spine3_aa,
            )
            for label in other_labels
        }
        current_positions = self.fk.fk(current_q, spine3_pos, spine3_aa)

        if body_pos is not None:
            ref_body = body_pos
        else:
            ref_body = self.fk.tpose_all_joints + (spine3_pos - self.fk.tpose_spine3_pos)

        cur_full = self.fk.full_body_positions(current_q, spine3_pos, spine3_aa)

        # Per-image equal-square axis limits from only the layers this image draws.
        extra_pts = [hi_positions.reshape(-1, 3)]
        extra_pts.extend(other_end_positions.values())
        if ref_positions is not None:
            extra_pts.append(ref_positions[:, wrist_chain_idx])
        if goal_world is not None:
            extra_pts.append(goal_world.reshape(1, 3))
        all_pts = np.concatenate([ref_body, current_positions] + extra_pts, axis=0)
        mins = np.min(all_pts, axis=0)
        maxs = np.max(all_pts, axis=0)
        center = (mins + maxs) / 2.0
        radius = max(float(np.max(maxs - mins)) / 2.0, 0.05)
        lims = [(center[i] - radius, center[i] + radius) for i in range(3)]

        def _sampled_arm_frames(ax, traj, cmap, view, alpha):
            """Draw up to 12 full-arm frames sampled along ``traj`` in a colour gradient."""
            n_total = traj.shape[0]
            n_samples = min(12, n_total)
            sample_indices = np.linspace(0, n_total - 1, n_samples).round().astype(int)
            denom = max(1, n_total - 1)
            for frame_idx in sample_indices:
                t = 0.3 + 0.7 * (frame_idx / denom)
                full = self.fk.full_body_positions(traj[frame_idx], spine3_pos, spine3_aa)
                _draw_bones_2d(ax, full, LEFT_ARM_BONE_PAIRS_22, view.hi, view.vi,
                               cmap(t), alpha=alpha, lw=1.2)

        hi_cmap = plt.get_cmap("Blues")
        other_cmap = plt.get_cmap("Greys")
        ref_cmap = plt.get_cmap("Greens")
        hi_wrist = hi_positions[:, wrist_chain_idx]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, view in zip(axes, _ORTHO_VIEWS):
            ax.set_aspect("equal")
            ax.set_title(view.title, fontsize=9)
            ax.set_xlabel(view.hl, fontsize=8)
            ax.set_ylabel(view.vl, fontsize=8)
            ax.set_xlim(*lims[view.hi])
            ax.set_ylim(*lims[view.vi])
            ax.tick_params(labelsize=7)

            # Static reference body (grey)
            _draw_bones_2d(ax, ref_body, ArmVisualizer.BODY_BONES, view.hi, view.vi,
                           ArmVisualizer.BODY_COLOR, alpha=0.45, lw=1.2)

            # Other clusters: terminal grey arm pose + wrist end marker only.
            for label in other_labels:
                positions = other_end_positions[label]
                terminal_full = self.fk.full_body_positions(
                    np.asarray(mdm_trajs[label][-1], dtype=np.float64),
                    spine3_pos,
                    spine3_aa,
                )
                _draw_bones_2d(
                    ax, terminal_full, LEFT_ARM_BONE_PAIRS_22, view.hi, view.vi,
                    other_cmap(0.75), alpha=0.65, lw=1.4,
                )
                other_wrist = positions[wrist_chain_idx]
                ax.scatter(other_wrist[view.hi], other_wrist[view.vi],
                           marker="x", color="dimgrey", s=40, alpha=0.8, zorder=4)

            # Original-goal reference: green full arm + green dashed wrist path.
            if ref_positions is not None:
                _sampled_arm_frames(ax, ref_traj_arr, ref_cmap, view, alpha=0.4)
                ref_wrist = ref_positions[:, wrist_chain_idx]
                ax.plot(ref_wrist[:, view.hi], ref_wrist[:, view.vi],
                        color="green", alpha=0.7, linewidth=1.3, linestyle="--")
            if goal_world is not None:
                ax.scatter(goal_world[view.hi], goal_world[view.vi],
                           marker="*", color="gold", edgecolors="black",
                           linewidths=0.5, s=180, zorder=6)

            # Highlighted cluster: blue-gradient arm + wrist path (no start/end markers).
            _sampled_arm_frames(ax, hi_traj, hi_cmap, view, alpha=0.5)
            ax.plot(hi_wrist[:, view.hi], hi_wrist[:, view.vi],
                    color="steelblue", alpha=0.6, linewidth=1.2)

            # Current pose arm (orange, shared start)
            _draw_bones_2d(ax, cur_full, LEFT_ARM_BONE_PAIRS_22, view.hi, view.vi,
                           "tab:orange", alpha=1.0, lw=2.2)

        scalar_mappable = plt.cm.ScalarMappable(
            cmap=hi_cmap, norm=plt.Normalize(vmin=0, vmax=hi_positions.shape[0] - 1)
        )
        scalar_mappable.set_array([])
        fig.colorbar(scalar_mappable, ax=axes[-1], shrink=0.8, pad=0.04,
                     label="chosen frame (light=early, dark=late)")
        legend_handles = [
            plt.Line2D([0], [0], color="steelblue", linewidth=2, label="chosen path"),
            plt.Line2D([0], [0], color="tab:orange", linewidth=2, label="current"),
        ]
        if other_labels:
            legend_handles.append(
                plt.Line2D([0], [0], color="darkgrey", linewidth=1.5,
                           marker="x", markeredgecolor="dimgrey",
                           label="other candidate end poses")
            )
        if ref_positions is not None:
            legend_handles.append(
                plt.Line2D([0], [0], color="green", linewidth=1.5, linestyle="--",
                           label="goal path (pre-correction)")
            )
        if goal_world is not None:
            legend_handles.append(
                plt.Line2D([0], [0], marker="*", color="gold", markeredgecolor="black",
                           linestyle="", markersize=11, label="original goal")
            )
        axes[0].legend(handles=legend_handles, fontsize=7, loc="upper left")
        fig.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    def render_cost_feedback_overlay(
        self,
        save_path: str | Path,
        *,
        rollout_traj: np.ndarray,
        correction_traj: np.ndarray,
        current_q: np.ndarray,
        spine3_pos: np.ndarray,
        spine3_aa: np.ndarray,
        body_pos: np.ndarray | None = None,
        goal_pos: np.ndarray | None = None,
    ) -> None:
        """Overlay the cost rollout against the target correction for feedback.

        Renders two full arms on shared equal-square axes across the three
        orthographic views: the trajectory the candidate cost produced
        (``rollout_traj``, red gradient) and the target corrected path the cost
        should match (``correction_traj``, green gradient) — typically the entire
        intended motion (pre-correction history, the correction, and the
        continuation to the goal). The orange current pose and
        gold goal star ground the scene. ``rollout_traj`` is linearly resampled to
        ``correction_traj``'s length so the two arms are frame-comparable. This is
        the per-iteration image fed back to the ``turns`` / ``agent`` cost
        generators so they can see where their motion diverges from the target.

        Args:
            save_path:       Output image path (.png).
            rollout_traj:    ``(R, 3, 3)`` arm trajectory the candidate cost produced.
            correction_traj: ``(T, 3, 3)`` target corrected-path trajectory.
            current_q:       ``(3, 3)`` current arm axis-angle state.
            spine3_pos:      ``(3,)`` spine3 world position.
            spine3_aa:       ``(3,)`` spine3 world axis-angle.
            body_pos:        ``(22, 3)`` reference body; falls back to a translated
                             T-pose when ``None``.
            goal_pos:        ``(3,)`` spine3-relative wrist goal, or ``None``.
        """
        save_path = Path(save_path)
        wrist_chain_idx = 4  # left_wrist in the 5-joint arm chain
        target = np.asarray(correction_traj, dtype=np.float64)
        rollout = _resample_traj(np.asarray(rollout_traj, dtype=np.float64), target.shape[0])
        rollout_positions = self.fk.fk_batch(rollout, spine3_pos, spine3_aa)
        target_positions = self.fk.fk_batch(target, spine3_pos, spine3_aa)
        current_positions = self.fk.fk(current_q, spine3_pos, spine3_aa)
        goal_world = (
            spine3_pos + np.asarray(goal_pos, dtype=np.float64)
            if goal_pos is not None
            else None
        )

        if body_pos is not None:
            ref_body = body_pos
        else:
            ref_body = self.fk.tpose_all_joints + (spine3_pos - self.fk.tpose_spine3_pos)
        cur_full = self.fk.full_body_positions(current_q, spine3_pos, spine3_aa)

        extra_pts = [
            rollout_positions.reshape(-1, 3),
            target_positions.reshape(-1, 3),
        ]
        if goal_world is not None:
            extra_pts.append(goal_world.reshape(1, 3))
        all_pts = np.concatenate([ref_body, current_positions] + extra_pts, axis=0)
        mins = np.min(all_pts, axis=0)
        maxs = np.max(all_pts, axis=0)
        center = (mins + maxs) / 2.0
        radius = max(float(np.max(maxs - mins)) / 2.0, 0.05)
        lims = [(center[i] - radius, center[i] + radius) for i in range(3)]

        def _sampled_arm_frames(ax, traj, cmap, view, alpha):
            n_total = traj.shape[0]
            n_samples = min(12, n_total)
            sample_indices = np.linspace(0, n_total - 1, n_samples).round().astype(int)
            denom = max(1, n_total - 1)
            for frame_idx in sample_indices:
                t = 0.3 + 0.7 * (frame_idx / denom)
                full = self.fk.full_body_positions(traj[frame_idx], spine3_pos, spine3_aa)
                _draw_bones_2d(ax, full, LEFT_ARM_BONE_PAIRS_22, view.hi, view.vi,
                               cmap(t), alpha=alpha, lw=1.2)

        target_cmap = plt.get_cmap("Greens")
        rollout_cmap = plt.get_cmap("Reds")
        target_wrist = target_positions[:, wrist_chain_idx]
        rollout_wrist = rollout_positions[:, wrist_chain_idx]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, view in zip(axes, _ORTHO_VIEWS):
            ax.set_aspect("equal")
            ax.set_title(view.title, fontsize=9)
            ax.set_xlabel(view.hl, fontsize=8)
            ax.set_ylabel(view.vl, fontsize=8)
            ax.set_xlim(*lims[view.hi])
            ax.set_ylim(*lims[view.vi])
            ax.tick_params(labelsize=7)

            _draw_bones_2d(ax, ref_body, ArmVisualizer.BODY_BONES, view.hi, view.vi,
                           ArmVisualizer.BODY_COLOR, alpha=0.45, lw=1.2)

            # Target correction: green full arm + green wrist path.
            _sampled_arm_frames(ax, target, target_cmap, view, alpha=0.45)
            ax.plot(target_wrist[:, view.hi], target_wrist[:, view.vi],
                    color="green", alpha=0.7, linewidth=1.3)

            # Cost rollout: red full arm + red wrist path.
            _sampled_arm_frames(ax, rollout, rollout_cmap, view, alpha=0.5)
            ax.plot(rollout_wrist[:, view.hi], rollout_wrist[:, view.vi],
                    color="firebrick", alpha=0.7, linewidth=1.3)
            ax.scatter(rollout_wrist[-1, view.hi], rollout_wrist[-1, view.vi],
                       marker="x", color="firebrick", s=40, alpha=0.9, zorder=4)

            if goal_world is not None:
                ax.scatter(goal_world[view.hi], goal_world[view.vi],
                           marker="*", color="gold", edgecolors="black",
                           linewidths=0.5, s=180, zorder=6)

            # Current pose arm (orange, shared start)
            _draw_bones_2d(ax, cur_full, LEFT_ARM_BONE_PAIRS_22, view.hi, view.vi,
                           "tab:orange", alpha=1.0, lw=2.2)

        legend_handles = [
            plt.Line2D([0], [0], color="firebrick", linewidth=2, label="cost rollout"),
            plt.Line2D([0], [0], color="green", linewidth=2,
                       label="target corrected path"),
            plt.Line2D([0], [0], color="tab:orange", linewidth=2, label="current"),
        ]
        if goal_world is not None:
            legend_handles.append(
                plt.Line2D([0], [0], marker="*", color="gold", markeredgecolor="black",
                           linestyle="", markersize=11, label="original goal")
            )
        axes[0].legend(handles=legend_handles, fontsize=7, loc="upper left")
        fig.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    def render_joint_angle_comparison(
        self,
        save_path: str | Path,
        *,
        target_series: dict[str, np.ndarray],
        rollout_series: dict[str, np.ndarray],
        reference_series: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Plot joint angles over time: target correction vs. cost rollout.

        One subplot per anatomical joint feature (keyed identically in both
        dicts, e.g. ``elbow_flexion``), each a ``(T,)`` radian series. The target
        corrected path is drawn in green and the candidate cost's rollout in red;
        when ``reference_series`` is given, the initial uncorrected goal-seeking
        path is drawn as a dashed steel-blue curve so the difference between the
        pre-correction and corrected motion is visible. Each rollout and reference
        series is linearly resampled to its target series' length so all curves
        share a frame-index x-axis. Companion to
        :meth:`render_cost_feedback_overlay`: the overlay shows Cartesian shape,
        this shows the temporal shape of each joint angle.

        Args:
            save_path:        Output image path (.png).
            target_series:    ``{feature_name: (T,) radians}`` for the target path.
            rollout_series:   ``{feature_name: (R,) radians}`` for the cost rollout.
            reference_series: ``{feature_name: (S,) radians}`` for the initial
                              uncorrected path, or ``None`` to omit it.
        """
        save_path = Path(save_path)
        names = list(target_series.keys())
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for ax, name in zip(axes.flat, names):
            target = np.asarray(target_series[name], dtype=np.float64)
            rollout = _resample_traj(
                np.asarray(rollout_series[name], dtype=np.float64), target.shape[0]
            )
            frames = np.arange(target.shape[0])
            if reference_series is not None:
                reference = _resample_traj(
                    np.asarray(reference_series[name], dtype=np.float64),
                    target.shape[0],
                )
                ax.plot(frames, reference, color="steelblue", linewidth=1.4,
                        linestyle="--", label="initial uncorrected path")
            ax.plot(frames, target, color="green", linewidth=1.6,
                    label="target corrected path")
            ax.plot(frames, rollout, color="firebrick", linewidth=1.6,
                    label="cost rollout")
            ax.set_title(name.replace("_", " "), fontsize=9)
            ax.set_xlabel("frame", fontsize=8)
            ax.set_ylabel("angle (rad)", fontsize=8)
            ax.tick_params(labelsize=7)
        axes.flat[0].legend(fontsize=7, loc="best")
        fig.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    def finish_live(self, save_path: str, fps: int = 20) -> None:
        """Save the frames recorded during the live session to a video or GIF.

        When :meth:`start_capture` was called before the loop, uses imageio to
        write pre-captured RGB frames directly — no re-rendering, much faster.
        Falls back to ``FuncAnimation.save`` otherwise.

        Args:
            save_path: Output file path.  Use ``.mp4`` or ``.gif``.
            fps:       Frames per second for the saved video (imageio path only).
        """
        assert self._live is not None, "finish_live() called before open_live()"
        live = self._live
        recorded = live.recorded_frames
        if not recorded:
            print("No frames recorded; nothing to save.")
            return

        # Fast path: write pre-captured RGBA frames with imageio
        if self._frame_bufs:
            try:
                import imageio  # pylint: disable=import-outside-toplevel

                imageio.mimsave(save_path, self._frame_bufs, fps=fps)
                print(f"Saved animation to {save_path}")
                return
            except Exception as exc:  # pylint: disable=broad-exception-caught
                print(f"imageio save failed ({exc}), falling back to FuncAnimation")

        n_steps = len(recorded) - 1
        wrist_trace = np.array([f["positions"][_WRIST_IDX] for f in recorded])

        def _update(k: int):
            pos = recorded[k]["positions"]
            arm_pts = pos[LEFT_ARM_JOINT_INDICES_22]
            tr = wrist_trace[: k + 1]
            _update_artists(
                live.artists3d,
                live.artists2d,
                pos,
                arm_pts,
                tr,
                step=k,
                n_steps=n_steps,
                dist=recorded[k]["dist"],
                color=recorded[k].get("color", ArmVisualizer.TARGET_COLOR),
                trace_color=recorded[k].get("trace_color", _TRACE_COLOR),
            )
            all_artists = []
            for a3 in live.artists3d:
                all_artists += [a3["scat"], *a3["lines"], a3["trace"]]
            for a2 in live.artists2d:
                all_artists += [a2["scat"], *a2["lines"], a2["trace"]]
            return all_artists

        # Close the interactive window before rendering.  On Qt/Tk backends,
        # canvas.draw() processes the GUI event queue; if a close event is
        # queued (e.g. from a prior plt.close on another figure), it fires
        # mid-save and truncates the GIF at a random frame.  Closing the
        # window first makes canvas.draw() a no-op while fig.savefig() (used
        # by both Pillow and ffmpeg writers) still renders correctly via its
        # own fresh Agg renderer.
        plt.close(live.fig)
        plt.ioff()
        anim = FuncAnimation(
            live.fig,
            _update,
            frames=len(recorded),
            interval=50,
            blit=False,
        )
        _save(anim, save_path)

    def plot_pose(
        self,
        q: np.ndarray,
        target_q: np.ndarray | None = None,
        spine3_pos: np.ndarray | None = None,
        spine3_aa: np.ndarray | None = None,
        ax: Axes3D | None = None,
    ) -> Axes3D:
        """Plot a single full-body pose with the left arm set by ``q``."""
        if ax is None:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection="3d")

        pos = self._full_body_positions(q, spine3_pos, spine3_aa)
        tpose = self.fk.tpose_all_joints

        ArmVisualizer.draw_bones_3d(
            ax, tpose, ArmVisualizer.BODY_BONES, ArmVisualizer.BODY_COLOR, alpha=0.5, lw=1.5
        )
        ax.scatter(  # type: ignore[misc]
            *tpose[ArmVisualizer.BODY_JOINTS].T,
            color=ArmVisualizer.BODY_COLOR,
            s=20,
            alpha=0.5,
            depthshade=False,
        )
        ArmVisualizer.draw_bones_3d(
            ax,
            pos,
            LEFT_ARM_BONE_PAIRS_22,
            ArmVisualizer.TARGET_COLOR,
            alpha=1.0,
            lw=2,
            label="current",
        )
        ax.scatter(  # type: ignore[misc]
            *pos[LEFT_ARM_JOINT_INDICES_22].T,
            color=ArmVisualizer.TARGET_COLOR,
            s=45,
            depthshade=False,
        )

        if target_q is not None:
            tgt = self._full_body_positions(target_q, spine3_pos, spine3_aa)
            ArmVisualizer.draw_bones_3d(
                ax,
                tgt,
                LEFT_ARM_BONE_PAIRS_22,
                ArmVisualizer.TARGET_COLOR,
                alpha=0.4,
                lw=2,
                linestyle="--",
                label="target",
            )
            ax.scatter(  # type: ignore[misc]
                *tgt[LEFT_ARM_JOINT_INDICES_22].T,
                color=ArmVisualizer.TARGET_COLOR,
                s=35,
                alpha=0.4,
                depthshade=False,
            )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")  # type: ignore[union-attr]
        ax.legend(fontsize=8)
        return ax

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_figure(
        self,
        target_q: np.ndarray,
        lims: list[tuple[float, float]],
        spine3_pos: np.ndarray | None,
        spine3_aa: np.ndarray | None,
        body_pos: np.ndarray | None = None,
        compact: bool = False,
        elbow_height_range: tuple[float, float] | None = None,
        fig: plt.Figure | None = None,
        show_target_arm: bool = True,
    ) -> tuple[plt.Figure, list[dict], list[dict]]:
        """Build the figure with static elements drawn and mutable artists created.

        Args:
            body_pos: ``(22, 3)`` positions for the grey background body.
            compact:  If ``True``, build a single 3-D panel (faster to render).
            fig:      Pre-created figure to use.  When ``None`` (default) a new
                      figure is created via ``plt.figure()``.  Pass a figure
                      with a non-interactive canvas (e.g. ``FigureCanvasAgg``)
                      to avoid touching global pyplot state in background threads.
            show_target_arm: If ``False``, omit the static dashed blue
                      joint-space target arm.

        Returns:
            ``(fig, artists_3d, artists_2d)``
        """
        target_full = self._full_body_positions(target_q, spine3_pos, spine3_aa)
        ref_body = body_pos if body_pos is not None else self.fk.tpose_all_joints

        if compact:
            if fig is None:
                fig = plt.figure(figsize=(8, 8))
            gs = gridspec.GridSpec(1, 1, figure=fig)
            artists_3d = self._build_3d_panels(
                fig,
                gs,
                target_full,
                ref_body,
                lims,
                compact=True,
                elbow_height_range=elbow_height_range,
                show_target_arm=show_target_arm,
            )
            return fig, artists_3d, []

        if fig is None:
            fig = plt.figure(figsize=(20, 9))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
        fig.suptitle("SMPL Left Arm MPC (CEM)", fontsize=13, y=1.01)

        artists_3d = self._build_3d_panels(
            fig,
            gs,
            target_full,
            ref_body,
            lims,
            elbow_height_range=elbow_height_range,
            show_target_arm=show_target_arm,
        )
        artists_2d = self._build_2d_panels(
            fig,
            gs,
            target_full,
            ref_body,
            lims,
            elbow_height_range=elbow_height_range,
            show_target_arm=show_target_arm,
        )
        return fig, artists_3d, artists_2d

    def _build_3d_panels(  # pylint: disable=too-many-locals
        self,
        fig: plt.Figure,
        gs: gridspec.GridSpec,
        target_full: np.ndarray,
        ref_body: np.ndarray,
        lims: list[tuple[float, float]],
        compact: bool = False,
        elbow_height_range: tuple[float, float] | None = None,
        show_target_arm: bool = True,
    ) -> list[dict]:
        views = [_COMPACT_VIEW] if compact else _3D_VIEWS
        artists: list[dict] = []
        for col, (title, elev, azim) in enumerate(views):
            ax: Axes3D = fig.add_subplot(gs[0, col], projection="3d")
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("X", fontsize=7, labelpad=1)
            ax.set_ylabel("Y", fontsize=7, labelpad=1)
            ax.set_zlabel("Z", fontsize=7, labelpad=1)
            ax.tick_params(labelsize=6)
            ax.set_xlim(*lims[0])
            ax.set_ylim(*lims[1])
            ax.set_zlim(*lims[2])
            elbow_planes = _add_elbow_height_planes_3d(ax, lims, elbow_height_range)

            ArmVisualizer.draw_bones_3d(
                ax, ref_body, ArmVisualizer.BODY_BONES, ArmVisualizer.BODY_COLOR, alpha=0.45, lw=1.2
            )
            ax.scatter(
                *ref_body[ArmVisualizer.BODY_JOINTS].T,
                color=ArmVisualizer.BODY_COLOR,
                s=14,
                alpha=0.45,
                depthshade=False,
            )
            if show_target_arm:
                ArmVisualizer.draw_bones_3d(
                    ax,
                    target_full,
                    LEFT_ARM_BONE_PAIRS_22,
                    ArmVisualizer.TARGET_COLOR,
                    alpha=0.4,
                    lw=1.8,
                    linestyle="--",
                    label="target" if col == 0 else None,
                )
                ax.scatter(
                    *target_full[LEFT_ARM_JOINT_INDICES_22].T,
                    color=ArmVisualizer.TARGET_COLOR,
                    s=30,
                    alpha=0.4,
                    depthshade=False,
                )
            if col == 0 and show_target_arm:
                ax.legend(loc="upper left", fontsize=7)

            scat = ax.scatter(
                [],
                [],
                [],
                color=ArmVisualizer.TARGET_COLOR,
                s=40,
                depthshade=False,
                zorder=5,
                label="current" if col == 0 else None,
            )
            lines = [
                ax.plot([], [], [], color=ArmVisualizer.TARGET_COLOR, lw=1.8)[0]
                for _ in LEFT_ARM_BONE_PAIRS_22
            ]
            (trace,) = ax.plot(
                [], [], [], color=_TRACE_COLOR, lw=1, alpha=0.6, linestyle=":"
            )
            mdm_goal_lines = [
                ax.plot([], [], [], color=ArmVisualizer.MDM_COLOR, lw=1.8, linestyle="--")[0]
                for _ in LEFT_ARM_BONE_PAIRS_22
            ]
            mdm_goal_scat = ax.scatter(
                [], [], [], color=ArmVisualizer.MDM_COLOR, s=30, alpha=0.6, depthshade=False
            )
            preview_lines = [
                ax.plot([], [], [], color=ArmVisualizer.MDM_COLOR, lw=1.8, alpha=0.5)[0]
                for _ in LEFT_ARM_BONE_PAIRS_22
            ]
            preview_scat = ax.scatter(
                [], [], [], color=ArmVisualizer.MDM_COLOR, s=30, alpha=0.5, depthshade=False
            )
            cartesian_goal_scat = ax.scatter(
                [],
                [],
                [],
                color="royalblue",
                s=200,
                marker="*",
                depthshade=False,
                zorder=10,
            )
            artists.append(
                {
                    "scat": scat,
                    "lines": lines,
                    "trace": trace,
                    "ax": ax,
                    "is_main": col == 0,
                    "mdm_goal_lines": mdm_goal_lines,
                    "mdm_goal_scat": mdm_goal_scat,
                    "preview_lines": preview_lines,
                    "preview_scat": preview_scat,
                    "cartesian_goal_scat": cartesian_goal_scat,
                    "elbow_planes": elbow_planes,
                }
            )
        return artists

    def _build_2d_panels(  # pylint: disable=too-many-locals
        self,
        fig: plt.Figure,
        gs: gridspec.GridSpec,
        target_full: np.ndarray,
        ref_body: np.ndarray,
        lims: list[tuple[float, float]],
        elbow_height_range: tuple[float, float] | None = None,
        show_target_arm: bool = True,
    ) -> list[dict]:
        artists: list[dict] = []
        for col, view in enumerate(_ORTHO_VIEWS):
            ax = fig.add_subplot(gs[1, col])
            ax.set_aspect("equal")
            ax.set_title(view.title, fontsize=9)
            ax.set_xlabel(view.hl, fontsize=8)
            ax.set_ylabel(view.vl, fontsize=8)
            ax.set_xlim(*lims[view.hi])
            ax.set_ylim(*lims[view.vi])
            ax.tick_params(labelsize=7)
            elbow_height_lines = _add_elbow_height_lines_2d(ax, view, elbow_height_range)

            _draw_bones_2d(
                ax,
                ref_body,
                ArmVisualizer.BODY_BONES,
                view.hi,
                view.vi,
                ArmVisualizer.BODY_COLOR,
                alpha=0.45,
                lw=1.2,
            )
            ax.scatter(
                ref_body[ArmVisualizer.BODY_JOINTS, view.hi],
                ref_body[ArmVisualizer.BODY_JOINTS, view.vi],
                color=ArmVisualizer.BODY_COLOR,
                s=14,
                alpha=0.45,
                zorder=3,
            )
            if show_target_arm:
                _draw_bones_2d(
                    ax,
                    target_full,
                    LEFT_ARM_BONE_PAIRS_22,
                    view.hi,
                    view.vi,
                    ArmVisualizer.TARGET_COLOR,
                    alpha=0.4,
                    lw=1.8,
                    linestyle="--",
                )
                ax.scatter(
                    target_full[LEFT_ARM_JOINT_INDICES_22, view.hi],
                    target_full[LEFT_ARM_JOINT_INDICES_22, view.vi],
                    color=ArmVisualizer.TARGET_COLOR,
                    s=28,
                    alpha=0.4,
                    zorder=4,
                )

            scat = ax.scatter([], [], color=ArmVisualizer.TARGET_COLOR, s=35, zorder=5)
            lines = [
                ax.plot([], [], color=ArmVisualizer.TARGET_COLOR, lw=1.8)[0]
                for _ in LEFT_ARM_BONE_PAIRS_22
            ]
            (trace,) = ax.plot(
                [], [], color=_TRACE_COLOR, lw=1, alpha=0.6, linestyle=":"
            )
            mdm_goal_lines = [
                ax.plot([], [], color=ArmVisualizer.MDM_COLOR, lw=1.8, linestyle="--")[0]
                for _ in LEFT_ARM_BONE_PAIRS_22
            ]
            mdm_goal_scat = ax.scatter(
                [], [], color=ArmVisualizer.MDM_COLOR, s=28, alpha=0.6, zorder=4
            )
            preview_lines = [
                ax.plot([], [], color=ArmVisualizer.MDM_COLOR, lw=1.8, alpha=0.5)[0]
                for _ in LEFT_ARM_BONE_PAIRS_22
            ]
            preview_scat = ax.scatter(
                [], [], color=ArmVisualizer.MDM_COLOR, s=28, alpha=0.5, zorder=4
            )
            cartesian_goal_scat = ax.scatter(
                [],
                [],
                color="royalblue",
                s=200,
                marker="*",
                zorder=10,
            )
            artists.append(
                {
                    "scat": scat,
                    "lines": lines,
                    "trace": trace,
                    "ax": ax,
                    "hi": view.hi,
                    "vi": view.vi,
                    "mdm_goal_lines": mdm_goal_lines,
                    "mdm_goal_scat": mdm_goal_scat,
                    "preview_lines": preview_lines,
                    "preview_scat": preview_scat,
                    "cartesian_goal_scat": cartesian_goal_scat,
                    "elbow_height_lines": elbow_height_lines,
                }
            )
        return artists

    def _run_mpc(
        self,
        mpc: SmplLeftArmMPC,
        initial_q: np.ndarray,
        target_q: np.ndarray,
        n_steps: int,
        spine3_pos: np.ndarray | None,
        spine3_aa: np.ndarray | None,
    ) -> list[dict]:
        mpc.reset_warmstart()
        current_q = np.asarray(initial_q, dtype=np.float64)
        target_q = np.asarray(target_q, dtype=np.float64)

        frames = []
        for step in range(n_steps + 1):
            positions = self._full_body_positions(current_q, spine3_pos, spine3_aa)
            dist = float(np.linalg.norm(current_q - target_q))
            frames.append({"q": current_q.copy(), "positions": positions, "dist": dist})
            if step < n_steps:
                current_q = mpc.step(current_q)

        return frames


# ---------------------------------------------------------------------------
# Module-level drawing helpers (private, used internally by ArmVisualizer)
# ---------------------------------------------------------------------------


def _compute_lims(
    frames: list[dict],
    fk: SmplLeftArmFK,
    target_q: np.ndarray,
    spine3_pos: np.ndarray | None,
    spine3_aa: np.ndarray | None,
    margin: float = 0.05,
) -> list[tuple[float, float]]:
    """Compute per-axis limits spanning all frame positions and the target."""
    target_q = np.asarray(target_q, dtype=np.float64)
    target_full = fk.full_body_positions(target_q, spine3_pos, spine3_aa)
    all_pts = np.vstack([f["positions"] for f in frames] + [target_full])
    return [
        (all_pts[:, i].min() - margin, all_pts[:, i].max() + margin) for i in range(3)
    ]


def _make_frame_updater(
    frames: list[dict],
    artists_3d: list[dict],
    artists_2d: list[dict],
    n_steps: int,
):
    """Return a FuncAnimation update callback that closes over the given data."""
    wrist_trace = np.array([f["positions"][_WRIST_IDX] for f in frames])

    def update(k: int):
        pos = frames[k]["positions"]
        arm_pts = pos[LEFT_ARM_JOINT_INDICES_22]
        dist = frames[k]["dist"]
        tr = wrist_trace[: k + 1]
        _update_artists(
            artists_3d,
            artists_2d,
            pos,
            arm_pts,
            tr,
            step=k,
            n_steps=n_steps,
            dist=dist,
            color=frames[k].get("color", ArmVisualizer.TARGET_COLOR),
            trace_color=frames[k].get("trace_color", _TRACE_COLOR),
        )
        all_artists = []
        for a3 in artists_3d:
            all_artists += [a3["scat"], *a3["lines"], a3["trace"]]
        for a2 in artists_2d:
            all_artists += [a2["scat"], *a2["lines"], a2["trace"]]
        return all_artists

    return update


def _update_artists(  # pylint: disable=too-many-locals
    artists_3d: list[dict],
    artists_2d: list[dict],
    pos: np.ndarray,
    arm_pts: np.ndarray,
    wrist_trace: np.ndarray,
    step: int,
    n_steps: int | None,
    dist: float,
    color: str = ArmVisualizer.TARGET_COLOR,
    trace_color: str = _TRACE_COLOR,
) -> None:
    """Update all mutable artists for a single frame/step."""
    for a3 in artists_3d:
        a3["scat"]._offsets3d = (  # pylint: disable=protected-access
            arm_pts[:, 0],
            arm_pts[:, 1],
            arm_pts[:, 2],
        )
        a3["scat"].set_color([color])
        for line, (pi, ci) in zip(a3["lines"], LEFT_ARM_BONE_PAIRS_22):
            seg = pos[[pi, ci]]
            line.set_data(seg[:, 0], seg[:, 1])
            line.set_3d_properties(seg[:, 2])
            line.set_color(color)
        if len(wrist_trace):
            a3["trace"].set_data(wrist_trace[:, 0], wrist_trace[:, 1])
            a3["trace"].set_3d_properties(wrist_trace[:, 2])
            a3["trace"].set_color(trace_color)
        if a3["is_main"]:
            step_str = f"{step}/{n_steps}" if n_steps is not None else str(step)
            a3["ax"].set_title(
                f"Perspective   step {step_str}   dist={dist:.4f} rad",
                fontsize=8,
            )

    for a2 in artists_2d:
        a2["scat"].set_offsets(arm_pts[:, [a2["hi"], a2["vi"]]])
        a2["scat"].set_facecolor([color])
        for line, (pi, ci) in zip(a2["lines"], LEFT_ARM_BONE_PAIRS_22):
            seg = pos[[pi, ci]]
            line.set_data(seg[:, a2["hi"]], seg[:, a2["vi"]])
            line.set_color(color)
        if len(wrist_trace):
            a2["trace"].set_data(wrist_trace[:, a2["hi"]], wrist_trace[:, a2["vi"]])
            a2["trace"].set_color(trace_color)


def _elbow_plane_vertices(
    lims: list[tuple[float, float]],
    y_value: float,
) -> list[list[tuple[float, float, float]]]:
    x0, x1 = lims[0]
    z0, z1 = lims[2]
    return [[(x0, y_value, z0), (x1, y_value, z0), (x1, y_value, z1), (x0, y_value, z1)]]


def _add_elbow_height_planes_3d(
    ax: Axes3D,
    lims: list[tuple[float, float]],
    elbow_height_range: tuple[float, float] | None,
) -> list[Poly3DCollection]:
    y_values = elbow_height_range if elbow_height_range is not None else (0.0, 0.0)
    visible = elbow_height_range is not None
    planes = []
    for y_value in y_values:
        plane = Poly3DCollection(
            _elbow_plane_vertices(lims, y_value),
            facecolors=_ELBOW_RANGE_COLOR,
            edgecolors=_ELBOW_RANGE_COLOR,
            alpha=0.12,
            linewidths=0.8,
            visible=visible,
        )
        ax.add_collection3d(plane)
        planes.append(plane)
    return planes


def _add_elbow_height_lines_2d(
    ax: plt.Axes,
    view: _OrthoView,
    elbow_height_range: tuple[float, float] | None,
) -> list:
    if view.vi != 1:
        return []
    y_values = elbow_height_range if elbow_height_range is not None else (0.0, 0.0)
    visible = elbow_height_range is not None
    return [
        ax.axhline(
            y_value,
            color=_ELBOW_RANGE_COLOR,
            alpha=0.55,
            linewidth=1.0,
            linestyle="--",
            visible=visible,
        )
        for y_value in y_values
    ]


def _update_elbow_height_artists(
    artists_3d: list[dict],
    artists_2d: list[dict],
    elbow_height_range: tuple[float, float] | None,
) -> None:
    visible = elbow_height_range is not None
    y_values = elbow_height_range if elbow_height_range is not None else (0.0, 0.0)
    for a3 in artists_3d:
        ax = a3["ax"]
        if elbow_height_range is not None:
            y0, y1 = ax.get_ylim3d()
            low_y, high_y = elbow_height_range
            if low_y < y0 or high_y > y1:
                ax.set_ylim3d(min(y0, low_y - 0.05), max(y1, high_y + 0.05))
        lims = [
            tuple(ax.get_xlim3d()),
            tuple(ax.get_ylim3d()),
            tuple(ax.get_zlim3d()),
        ]
        for plane, y_value in zip(a3["elbow_planes"], y_values):
            plane.set_verts(_elbow_plane_vertices(lims, y_value))
            plane.set_visible(visible)
    for a2 in artists_2d:
        if elbow_height_range is not None and a2["vi"] == 1:
            y0, y1 = a2["ax"].get_ylim()
            low_y, high_y = elbow_height_range
            if low_y < y0 or high_y > y1:
                a2["ax"].set_ylim(min(y0, low_y - 0.05), max(y1, high_y + 0.05))
        for line, y_value in zip(a2["elbow_height_lines"], y_values):
            line.set_ydata([y_value, y_value])
            line.set_visible(visible)


def _resample_traj(traj: np.ndarray, n: int) -> np.ndarray:
    """Linearly resample a ``(T, ...)`` trajectory to ``n`` frames along axis 0."""
    t = traj.shape[0]
    if t == n:
        return traj
    src = np.linspace(0.0, 1.0, t)
    dst = np.linspace(0.0, 1.0, n)
    flat = traj.reshape(t, -1)
    out = np.stack(
        [np.interp(dst, src, flat[:, i]) for i in range(flat.shape[1])], axis=1
    )
    return out.reshape((n, *traj.shape[1:]))


def _draw_bones_2d(
    ax: plt.Axes,
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
        ax.plot(
            seg[:, hi],
            seg[:, vi],
            color=color,
            alpha=alpha,
            linewidth=lw,
            linestyle=linestyle,
        )


def _save(anim: FuncAnimation, path: str) -> None:
    if path.endswith(".gif"):
        anim.save(path, writer="pillow", dpi=64)
    elif path.endswith(".mp4"):
        anim.save(path, writer="ffmpeg")
    else:
        anim.save(path)
    print(f"Saved animation to {path}")


if __name__ == "__main__":
    from uncertain_feedback.planners.mpc.arm_mpc import SmplLeftArmMPC

    # Entry point for the animate demo.
    parser = argparse.ArgumentParser(description="Visualise SMPL left arm MPC")
    parser.add_argument("--steps", type=int, default=512, help="Number of MPC steps")
    parser.add_argument("--samples", type=int, default=512, help="CEM sample count")
    parser.add_argument("--horizon", type=int, default=10, help="MPC horizon")
    parser.add_argument(
        "--save", type=str, default=None, help="Save path (e.g. arm.gif)"
    )
    parser.add_argument("--interval", type=int, default=120, help="Frame interval (ms)")
    demo_args = parser.parse_args()

    demo_fk = SmplLeftArmFK()
    demo_mpc = SmplLeftArmMPC(
        horizon=demo_args.horizon, n_mpc_samples=demo_args.samples
    )
    demo_vis = ArmVisualizer(demo_fk)

    demo_initial_q = np.zeros((3, 3))
    demo_target_q = np.array(
        [
            [0.0, -1.45, 0.0],  # left_shoulder
            [0.0, 0.0, 0.4],  # left_elbow
            [0.0, 0.0, 0.0],  # left_wrist
        ]
    )

    demo_fig, demo_anim = demo_vis.animate(
        demo_mpc,
        demo_initial_q,
        demo_target_q,
        n_steps=demo_args.steps,
        interval=demo_args.interval,
        save_path=demo_args.save,
    )
