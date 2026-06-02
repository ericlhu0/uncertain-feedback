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


_ORTHO_VIEWS = [
    _OrthoView("Front (XY)", 0, 1, "X (m)", "Y (m)"),
    _OrthoView("Side (ZY)", 2, 1, "Z (m)", "Y (m)"),
    _OrthoView("Top (XZ)", 0, 2, "X (m)", "Z (m)"),
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
    ) -> tuple[plt.Figure, list[dict], list[dict]]:
        """Build the figure with static elements drawn and mutable artists created.

        Args:
            body_pos: ``(22, 3)`` positions for the grey background body.
            compact:  If ``True``, build a single 3-D panel (faster to render).

        Returns:
            ``(fig, artists_3d, artists_2d)``
        """
        target_full = self._full_body_positions(target_q, spine3_pos, spine3_aa)
        ref_body = body_pos if body_pos is not None else self.fk.tpose_all_joints

        if compact:
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
            )
            return fig, artists_3d, []

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
        )
        artists_2d = self._build_2d_panels(
            fig,
            gs,
            target_full,
            ref_body,
            lims,
            elbow_height_range=elbow_height_range,
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
            if col == 0:
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
