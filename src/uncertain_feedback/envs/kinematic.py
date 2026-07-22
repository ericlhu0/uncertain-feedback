"""Kinematic execution env: the commanded configuration is achieved exactly."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from uncertain_feedback.envs.base import ExecutionEnv


class KinematicEnv(ExecutionEnv):
    """Pass-through env reproducing the original open-loop kinematic rollout."""

    def __init__(self) -> None:
        super().__init__()
        self._q_history: list[np.ndarray] = []

    def execute(self, q_cmd: np.ndarray) -> np.ndarray:
        self._q_history.append(np.asarray(q_cmd, dtype=np.float64).copy())
        return q_cmd

    def hold(self, q: np.ndarray) -> np.ndarray:
        self._q_history.append(np.asarray(q, dtype=np.float64).copy())
        return q

    def visualize(self, path: Path | None = None) -> np.ndarray:
        frame = self._render([self._q_history[-1]])[0]
        if path is not None:
            import imageio  # pylint: disable=import-outside-toplevel

            imageio.imwrite(str(path), frame)
        return frame

    def save_video(self, path: str | Path, fps: int = 20) -> None:
        import imageio  # pylint: disable=import-outside-toplevel

        frames = self._render(self._q_history)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(path), np.stack(frames), fps=fps)

    def _render(self, qs: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Render each ``(7,)`` configuration to an RGB frame."""
        from matplotlib.backends.backend_agg import (  # pylint: disable=import-outside-toplevel
            FigureCanvasAgg,
        )
        from matplotlib.figure import (  # pylint: disable=import-outside-toplevel
            Figure,
        )

        from uncertain_feedback.planners.mpc.kinematics import (  # pylint: disable=import-outside-toplevel
            LEFT_ARM_JOINT_INDICES_22,
            q_to_arm_aa,
        )
        from uncertain_feedback.utils.plot import (  # pylint: disable=import-outside-toplevel
            ArmVisualizer,
        )

        fk = self._fk
        assert fk is not None, "set_pose_context must be called before rendering"
        all_pos = np.stack(
            [
                fk.full_body_positions(
                    q_to_arm_aa(q, fk.elbow_hinge_axis),
                    self._spine3_pos,
                    self._spine3_aa,
                )
                for q in qs
            ]
        )

        fig = Figure(figsize=(6, 6))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111, projection="3d")
        frames: list[np.ndarray] = []
        for pos in all_pos:
            ax.cla()
            ArmVisualizer.draw_smpl_skeleton(
                ax,
                pos,
                highlight_joints=set(LEFT_ARM_JOINT_INDICES_22),
            )
            ArmVisualizer.format_3d_axis(ax, all_pos.reshape(-1, 3)[:, [0, 2, 1]])
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            frames.append(buf.reshape(h, w, 4)[..., :3].copy())
        return frames
