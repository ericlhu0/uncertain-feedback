"""3D stick-figure visualizer for the SMPL left arm MPC.

Draws the full 22-joint SMPL skeleton.  Non-arm joints are fixed at the
T-pose; the left arm joints (collar, shoulder, elbow, wrist) are animated
by :class:`SmplLeftArmMPC`.

Layout (2 rows):
  Row 0 — four 3D panels at different camera angles:
      Perspective  |  Left-front  |  Right-back  |  Top-down
  Row 1 — three orthographic 2D projections:
      Front (XY)   |  Side (ZY)   |  Top (XZ)

Example::

    from uncertain_feedback.planners.mpc import SmplLeftArmMPC, SmplLeftArmFK, ArmVisualizer
    import numpy as np

    fk  = SmplLeftArmFK()
    mpc = SmplLeftArmMPC(horizon=10, n_samples=512)
    vis = ArmVisualizer(fk)

    initial_q = np.zeros((4, 3))
    target_q  = np.array([[0, 0.3, 0], [0, 0.5, 0], [0, 0, 0.4], [0, 0, 0]])

    fig, anim = vis.animate(mpc, initial_q, target_q, n_steps=40)
    plt.show()
    # or: anim.save("arm.gif", writer="pillow")
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

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
        except Exception:
            pass
    else:
        import sys

        print(
            "WARNING: no interactive matplotlib backend found — run: uv add pyqt5",
            file=sys.stderr,
        )
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

from uncertain_feedback.planners.mpc.kinematics import (
    LEFT_ARM_BONE_PAIRS_22,
    LEFT_ARM_JOINT_INDICES_22,
    SMPL_BONE_PAIRS_22,
    SmplLeftArmFK,
)

if TYPE_CHECKING:
    from uncertain_feedback.planners.mpc.arm_mpc import SmplLeftArmMPC

# Visual style
_BODY_COLOR = "#aaaaaa"
_CURRENT_COLOR = "royalblue"
_TARGET_COLOR = "tomato"
_TRACE_COLOR = "cornflowerblue"

_BODY_BONES = [p for p in SMPL_BONE_PAIRS_22 if p not in LEFT_ARM_BONE_PAIRS_22]
_BODY_JOINTS = [j for j in range(22) if j not in LEFT_ARM_JOINT_INDICES_22]

_WRIST_IDX = 20  # left_wrist in the 22-joint array

# 3D camera angles: (title, elev, azim)
_3D_VIEWS = [
    ("Perspective", 45, -60),
    # ("Left-front",   20,   30),
    # ("Right-back",   20,  150),
    # ("Top-down",     70,  -45),
    ("c1", 45, 60),
    ("c1", 45, -120),
]

# 2D orthographic projections: (title, h_axis_idx, v_axis_idx, h_label, v_label)
_ORTHO_VIEWS = [
    ("Front (XY)", 0, 1, "X (m)", "Y (m)"),
    ("Side (ZY)", 2, 1, "Z (m)", "Y (m)"),
    ("Top (XZ)", 0, 2, "X (m)", "Z (m)"),
]


class ArmVisualizer:
    """Animate the full SMPL skeleton with the left arm driven by MPC.

    Args:
        fk:            :class:`SmplLeftArmFK` instance.  If ``None``, one is
                       created with the default SMPL model path.
        smpl_pkl_path: Passed to :class:`SmplLeftArmFK` when ``fk`` is ``None``.
    """

    def __init__(
        self,
        fk: SmplLeftArmFK | None = None,
        smpl_pkl_path: str | None = None,
    ) -> None:
        self.fk = fk if fk is not None else SmplLeftArmFK(smpl_pkl_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

        Row 0: four 3D views at different camera angles.
        Row 1: three orthographic 2D projections (front, side, top).

        Args:
            mpc:        Configured :class:`SmplLeftArmMPC`.
            initial_q:  ``(4, 3)`` starting axis-angle joint angles.
            target_q:   ``(4, 3)`` target axis-angle joint angles.
            n_steps:    Number of MPC steps to simulate.
            spine3_pos: ``(3,)`` world position of spine3.
            spine3_aa:  ``(3,)`` world axis-angle of spine3.
            interval:   Milliseconds between animation frames.
            save_path:  If given, save to this path (e.g. ``"arm.gif"``).

        Returns:
            ``(fig, anim)`` — the matplotlib Figure and FuncAnimation.
        """
        frames = self._run_mpc(mpc, initial_q, target_q, n_steps, spine3_pos, spine3_aa)
        target_full = self.fk.full_body_positions(target_q, spine3_pos, spine3_aa)
        tpose = self.fk.tpose_all_joints

        wrist_trace = np.array([f["positions"][_WRIST_IDX] for f in frames])

        # Global limits across all frames + target
        all_pts = np.vstack([f["positions"] for f in frames] + [target_full])
        mg = 0.05
        lims = [(all_pts[:, i].min() - mg, all_pts[:, i].max() + mg) for i in range(3)]

        # ------------------------------------------------------------------
        # Figure: 2 rows × 4 cols
        #   Row 0: 4 × 3D views
        #   Row 1: 3 × 2D ortho  (last column left empty)
        # ------------------------------------------------------------------
        fig = plt.figure(figsize=(20, 9))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

        fig.suptitle("SMPL Left Arm MPC (CEM)", fontsize=13, y=1.01)

        # ------------------------------------------------------------------
        # Row 0 — 3D panels
        # ------------------------------------------------------------------
        artists_3d: list[dict] = []

        for col, (title, elev, azim) in enumerate(_3D_VIEWS):
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

            # Static body
            _draw_bones_3d(ax, tpose, _BODY_BONES, _BODY_COLOR, alpha=0.45, lw=1.2)
            ax.scatter(
                *tpose[_BODY_JOINTS].T,
                color=_BODY_COLOR,
                s=14,
                alpha=0.45,
                depthshade=False,
            )

            # Static target arm
            _draw_bones_3d(
                ax,
                target_full,
                LEFT_ARM_BONE_PAIRS_22,
                _TARGET_COLOR,
                alpha=0.4,
                lw=1.8,
                linestyle="--",
                label="target" if col == 0 else None,
            )
            ax.scatter(
                *target_full[LEFT_ARM_JOINT_INDICES_22].T,
                color=_TARGET_COLOR,
                s=30,
                alpha=0.4,
                depthshade=False,
            )

            if col == 0:
                ax.legend(loc="upper left", fontsize=7)

            # Mutable current arm
            scat = ax.scatter(
                [],
                [],
                [],
                color=_CURRENT_COLOR,
                s=40,
                depthshade=False,
                zorder=5,
                label="current" if col == 0 else None,
            )
            lines = [
                ax.plot([], [], [], color=_CURRENT_COLOR, lw=1.8)[0]
                for _ in LEFT_ARM_BONE_PAIRS_22
            ]
            (trace,) = ax.plot(
                [], [], [], color=_TRACE_COLOR, lw=1, alpha=0.6, linestyle=":"
            )

            artists_3d.append(
                {
                    "scat": scat,
                    "lines": lines,
                    "trace": trace,
                    "ax": ax,
                    "is_main": col == 0,
                }
            )

        # ------------------------------------------------------------------
        # Row 1 — 2D orthographic panels
        # ------------------------------------------------------------------
        artists_2d: list[dict] = []

        for col, (title, hi, vi, hl, vl) in enumerate(_ORTHO_VIEWS):
            ax2 = fig.add_subplot(gs[1, col])
            ax2.set_aspect("equal")
            ax2.set_title(title, fontsize=9)
            ax2.set_xlabel(hl, fontsize=8)
            ax2.set_ylabel(vl, fontsize=8)
            ax2.set_xlim(*lims[hi])
            ax2.set_ylim(*lims[vi])
            ax2.tick_params(labelsize=7)

            # Static body
            _draw_bones_2d(
                ax2, tpose, _BODY_BONES, hi, vi, _BODY_COLOR, alpha=0.45, lw=1.2
            )
            ax2.scatter(
                tpose[_BODY_JOINTS, hi],
                tpose[_BODY_JOINTS, vi],
                color=_BODY_COLOR,
                s=14,
                alpha=0.45,
                zorder=3,
            )

            # Static target arm
            _draw_bones_2d(
                ax2,
                target_full,
                LEFT_ARM_BONE_PAIRS_22,
                hi,
                vi,
                _TARGET_COLOR,
                alpha=0.4,
                lw=1.8,
                linestyle="--",
            )
            ax2.scatter(
                target_full[LEFT_ARM_JOINT_INDICES_22, hi],
                target_full[LEFT_ARM_JOINT_INDICES_22, vi],
                color=_TARGET_COLOR,
                s=28,
                alpha=0.4,
                zorder=4,
            )

            # Mutable current arm
            scat = ax2.scatter([], [], color=_CURRENT_COLOR, s=35, zorder=5)
            lines = [
                ax2.plot([], [], color=_CURRENT_COLOR, lw=1.8)[0]
                for _ in LEFT_ARM_BONE_PAIRS_22
            ]
            (trace,) = ax2.plot(
                [], [], color=_TRACE_COLOR, lw=1, alpha=0.6, linestyle=":"
            )

            artists_2d.append(
                {"scat": scat, "lines": lines, "trace": trace, "hi": hi, "vi": vi}
            )

        # ------------------------------------------------------------------
        # Animation update
        # ------------------------------------------------------------------
        def update(k: int):
            pos = frames[k]["positions"]  # (22, 3)
            arm_pts = pos[LEFT_ARM_JOINT_INDICES_22]  # (4, 3)
            dist = frames[k]["dist"]
            tr = wrist_trace[: k + 1]

            # Update all 3D panels
            for a3 in artists_3d:
                a3["scat"]._offsets3d = (arm_pts[:, 0], arm_pts[:, 1], arm_pts[:, 2])
                for line, (pi, ci) in zip(a3["lines"], LEFT_ARM_BONE_PAIRS_22):
                    seg = pos[[pi, ci]]
                    line.set_data(seg[:, 0], seg[:, 1])
                    line.set_3d_properties(seg[:, 2])
                a3["trace"].set_data(tr[:, 0], tr[:, 1])
                a3["trace"].set_3d_properties(tr[:, 2])
                if a3["is_main"]:
                    a3["ax"].set_title(
                        f"Perspective   step {k}/{n_steps}   dist={dist:.4f} rad",
                        fontsize=8,
                    )

            # Update all 2D panels
            for a2 in artists_2d:
                hi, vi = a2["hi"], a2["vi"]
                a2["scat"].set_offsets(arm_pts[:, [hi, vi]])
                for line, (pi, ci) in zip(a2["lines"], LEFT_ARM_BONE_PAIRS_22):
                    seg = pos[[pi, ci]]
                    line.set_data(seg[:, hi], seg[:, vi])
                a2["trace"].set_data(tr[:, hi], tr[:, vi])

            all_artists = []
            for a3 in artists_3d:
                all_artists += [a3["scat"], *a3["lines"], a3["trace"]]
            for a2 in artists_2d:
                all_artists += [a2["scat"], *a2["lines"], a2["trace"]]
            return all_artists

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.tight_layout()

        anim = FuncAnimation(
            fig, update, frames=len(frames), interval=interval, blit=False
        )

        if save_path is not None:
            _save(anim, save_path)

        return fig, anim

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

        pos = self.fk.full_body_positions(q, spine3_pos, spine3_aa)
        tpose = self.fk.tpose_all_joints

        _draw_bones_3d(ax, tpose, _BODY_BONES, _BODY_COLOR, alpha=0.5, lw=1.5)
        ax.scatter(
            *tpose[_BODY_JOINTS].T, color=_BODY_COLOR, s=20, alpha=0.5, depthshade=False
        )
        _draw_bones_3d(
            ax,
            pos,
            LEFT_ARM_BONE_PAIRS_22,
            _CURRENT_COLOR,
            alpha=1.0,
            lw=2,
            label="current",
        )
        ax.scatter(
            *pos[LEFT_ARM_JOINT_INDICES_22].T,
            color=_CURRENT_COLOR,
            s=45,
            depthshade=False,
        )

        if target_q is not None:
            tgt = self.fk.full_body_positions(target_q, spine3_pos, spine3_aa)
            _draw_bones_3d(
                ax,
                tgt,
                LEFT_ARM_BONE_PAIRS_22,
                _TARGET_COLOR,
                alpha=0.4,
                lw=2,
                linestyle="--",
                label="target",
            )
            ax.scatter(
                *tgt[LEFT_ARM_JOINT_INDICES_22].T,
                color=_TARGET_COLOR,
                s=35,
                alpha=0.4,
                depthshade=False,
            )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend(fontsize=8)
        return ax

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_mpc(
        self,
        mpc: SmplLeftArmMPC,
        initial_q: np.ndarray,
        target_q: np.ndarray,
        n_steps: int,
        spine3_pos: np.ndarray | None,
        spine3_aa: np.ndarray | None,
    ) -> list[dict]:
        mpc._prev_mean = None
        current_q = np.asarray(initial_q, dtype=np.float64)
        target_q = np.asarray(target_q, dtype=np.float64)

        frames = []
        for step in range(n_steps + 1):
            positions = self.fk.full_body_positions(current_q, spine3_pos, spine3_aa)
            dist = float(np.linalg.norm(current_q - target_q))
            frames.append({"q": current_q.copy(), "positions": positions, "dist": dist})
            if step < n_steps:
                current_q = mpc.step(current_q, target_q)

        return frames


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _draw_bones_3d(
    ax: Axes3D,
    positions: np.ndarray,
    bone_pairs: list[tuple[int, int]],
    color: str,
    alpha: float = 1.0,
    lw: float = 2.0,
    linestyle: str = "-",
    label: str | None = None,
) -> None:
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
        anim.save(path, writer="pillow")
    elif path.endswith(".mp4"):
        anim.save(path, writer="ffmpeg")
    else:
        anim.save(path)
    print(f"Saved animation to {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualise SMPL left arm MPC")
    parser.add_argument("--steps", type=int, default=512, help="Number of MPC steps")
    parser.add_argument("--samples", type=int, default=512, help="CEM sample count")
    parser.add_argument("--horizon", type=int, default=10, help="MPC horizon")
    parser.add_argument(
        "--save", type=str, default=None, help="Save path (e.g. arm.gif)"
    )
    parser.add_argument("--interval", type=int, default=120, help="Frame interval (ms)")
    args = parser.parse_args()

    from uncertain_feedback.planners.mpc.arm_mpc import SmplLeftArmMPC

    fk = SmplLeftArmFK()
    mpc = SmplLeftArmMPC(horizon=args.horizon, n_samples=args.samples)
    vis = ArmVisualizer(fk)

    initial_q = np.zeros((4, 3))
    target_q = np.array(
        [
            [0.3, 0.3, 0.3],  # left_collar
            [0.0, -1.45, 0.0],  # left_shoulder
            [0.0, 0.0, 0.4],  # left_elbow
            [0.0, 0.0, 0.0],  # left_wrist
        ]
    )

    fig, anim = vis.animate(
        mpc,
        initial_q,
        target_q,
        n_steps=args.steps,
        interval=args.interval,
        save_path=args.save,
    )
    plt.show()
