"""MDM-extended MPC for the SMPL left arm.

:class:`LeftArmMPCMDM` inherits all sampling logic from
:class:`~uncertain_feedback.planners.mpc.arm_mpc.SmplLeftArmMPC` and adds:

* :meth:`push_trajectory` — bulk-enqueue an MDM-generated trajectory.
* :meth:`set_mdm_goal` — mark the MDM end-frame in the live visualizer.
* Per-call ``advance_threshold`` override in :meth:`step`.
* MDM-colored arm rendering (darkorange while following an MDM trajectory).
* ``body_pos`` background skeleton in the live visualizer window.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from uncertain_feedback.planners.mpc.arm_mpc import (
    SmplLeftArmMPC,
    _compose_rotvec,
    _VisConfig,
)
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK
from uncertain_feedback.planners.mpc.visualizer import (
    _MDM_COLOR,
    _TARGET_COLOR,
    ArmVisualizer,
)

# ---------------------------------------------------------------------------
# MDM-extended MPC
# ---------------------------------------------------------------------------


class LeftArmMPCMDM(SmplLeftArmMPC):
    """MDM-extended sampling-based MPC for the SMPL left arm.

    Inherits all core MPC logic from :class:`SmplLeftArmMPC`.  Adds MDM
    trajectory integration: bulk-enqueue generated frames, mark the final
    frame as a goal marker in the visualizer, and color the arm orange while
    tracking an MDM trajectory.

    Args:
        horizon:             Number of look-ahead steps.
        n_mpc_samples:       Number of candidate action sequences per step.
        max_angle_delta:     Sampling std dev (radians).
        advance_threshold:   Default L2 distance below which the controller
                             advances to the next queued frame.  Can be
                             overridden per :meth:`step` call.
        trajectory_fraction: Fraction of MDM-generated frames to enqueue
                             (e.g. ``0.75`` enqueues the first 75 % of
                             frames).  Defaults to
                             :attr:`TRAJECTORY_FRACTION`.
        goals:               Initial list of ``(4, 3)`` target configurations.
        goal_threshold:      Threshold passed to the base class (used only
                             when ``advance_threshold`` is not overriding).
        visualize:           If ``True``, open a live matplotlib window.
                             Requires ``fk``.
        fk:                  :class:`SmplLeftArmFK` instance (required when
                             ``visualize=True``).
        spine3_pos:          ``(3,)`` world position of spine3 (optional).
        spine3_aa:           ``(3,)`` world axis-angle of spine3 (optional).
        body_pos:            ``(22, 3)`` world joint positions for the grey
                             background skeleton (e.g. sitting pose).
    """

    TRAJECTORY_FRACTION: float = 0.75
    """Fraction of MDM trajectory frames to enqueue (default 75 %)."""

    def __init__(
        self,
        horizon: int = 10,
        n_mpc_samples: int = 512,
        max_angle_delta: float = 0.0025,
        advance_threshold: float = 0.1,
        trajectory_fraction: float = TRAJECTORY_FRACTION,
        goals: list[np.ndarray] | None = None,
        goal_threshold: float = 0.1,
        visualize: bool = False,
        fk: SmplLeftArmFK | None = None,
        spine3_pos: np.ndarray | None = None,
        spine3_aa: np.ndarray | None = None,
        body_pos: np.ndarray | None = None,
    ) -> None:
        # Base sets up _config, _goals deque, _prev_best, _vis.
        # Pass visualize=False; MDM overrides vis config below.
        super().__init__(
            horizon=horizon,
            n_mpc_samples=n_mpc_samples,
            max_angle_delta=max_angle_delta,
            goals=goals,
            goal_threshold=goal_threshold,
            visualize=False,
            fk=fk,
            spine3_pos=spine3_pos,
            spine3_aa=spine3_aa,
        )
        self.advance_threshold = advance_threshold
        self.trajectory_fraction = trajectory_fraction
        self.visualize = visualize
        if visualize:
            if fk is None:
                raise ValueError("visualize=True requires `fk` to be provided.")
            self._vis_config = _VisConfig(fk, spine3_pos, spine3_aa, body_pos=body_pos)

        # Last frame of the MDM trajectory, shown as a goal marker.
        self._mdm_goal: np.ndarray | None = None
        # Cutoff frame shown as a ghost arm in the live visualiser.
        self._preview_q: np.ndarray | None = None

    # ------------------------------------------------------------------
    # MDM-specific public API
    # ------------------------------------------------------------------

    def push_trajectory(
        self,
        frames: np.ndarray,
    ) -> None:
        """Push an MDM-generated trajectory into the goal queue.

                Each frame of ``frames`` becomes one ``(4, 3)`` target in the goal
                queue.  By default the new trajectory is prepended to the *front* of
                the queue so it executes immediately ahead of any
                previously queued goals.

                Args:
                    frames:   ``(n_frames, 4, 3)`` axis-angle trajectory for
                              ``[left_collar, left_shoulder, left_elbow, left_wrist]``,
                              as returned by
                              :meth:`~uncertain_feedback.motion_generators.mdm.mdm_api\
        .MdmMotionGenerator.generate_left_arm_trajectory`.
        """
        frames = np.asarray(frames, dtype=np.float64)
        # extendleft reverses the iterable, so reverse first to preserve order.
        self._goals.extendleft(frames[::-1])
        # Notify live visualiser of the new preview frame (last enqueued frame).
        preview_q = frames[-1].copy()
        self._preview_q = preview_q
        if self._vis is not None:
            self._vis.update_trajectory_preview(preview_q)

    def set_mdm_goal(self, goal_q: np.ndarray) -> None:
        """Set the MDM end-of-trajectory goal marker.

        Stores ``goal_q`` and immediately updates the live visualizer if it is
        already open.  If the visualizer has not been opened yet (i.e. no
        :meth:`step` has been called), the marker is applied automatically
        when the window first opens.

        Args:
            goal_q: ``(4, 3)`` axis-angle joint angles for the last frame of
                    the MDM-generated trajectory.
        """
        self._mdm_goal = np.asarray(goal_q, dtype=np.float64)
        if self._vis is not None:
            self._vis.update_mdm_goal(self._mdm_goal)

    # ------------------------------------------------------------------
    # step override: advance_threshold + MDM color
    # ------------------------------------------------------------------

    def step(
        self,
        current_q: np.ndarray,
        advance_threshold: float | None = None,
    ) -> np.ndarray:
        """Perform one MPC step.

        Uses the front of the goal queue as the current target.  When the arm
        comes within ``advance_threshold`` of the current target *and* there is
        a subsequent frame in the queue, the front entry is popped and the next
        frame becomes the new target.

        If ``visualize=True`` was set at construction, the live window is
        updated automatically.  The arm is drawn orange while following an MDM
        trajectory (queue length > 1) and blue once only the final goal remains.

        Args:
            current_q:         ``(4, 3)`` current axis-angle joint angles.
            advance_threshold: Distance (L2 norm) below which the MPC advances
                               to the next queued frame.  Defaults to
                               :attr:`advance_threshold`.

        Returns:
            ``(4, 3)`` updated axis-angle joint angles.
        """
        target_q = self._goals[0]
        first_action, _ = self.solve(current_q)
        next_q = _compose_rotvec(np.asarray(current_q, dtype=np.float64), first_action)

        threshold = (
            advance_threshold
            if advance_threshold is not None
            else self.advance_threshold
        )
        dist = float(np.linalg.norm(next_q - target_q))
        if dist < threshold and len(self._goals) > 1:
            self._goals.popleft()
            self.reset_warmstart()
            target_q = self._goals[0]
            dist = float(np.linalg.norm(next_q - target_q))

        if self._vis_config is not None:
            if self._vis is None:
                self._vis = ArmVisualizer(self._vis_config.fk)
                self._vis.open_live(
                    self._goals[-1],
                    self._vis_config.spine_pos,
                    self._vis_config.spine_aa,
                    body_pos=self._vis_config.body_pos,
                    compact=self._vis_config.compact,
                )
                if self._vis_config.capture:
                    self._vis.start_capture()
                if self._mdm_goal is not None:
                    self._vis.update_mdm_goal(self._mdm_goal)
                if self._preview_q is not None:
                    self._vis.update_trajectory_preview(self._preview_q)
            color = _TARGET_COLOR if len(self._goals) <= 1 else _MDM_COLOR
            self._vis.update_step(next_q, dist=dist, color=color)

        return next_q


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SMPL left arm MPC with live visualization"
    )
    parser.add_argument("--steps", type=int, default=750, help="Number of MPC steps")
    parser.add_argument("--samples", type=int, default=512, help="CEM sample count")
    parser.add_argument("--horizon", type=int, default=10, help="MPC horizon")
    parser.add_argument(
        "--no-vis", action="store_true", help="Disable live visualization"
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Save the visualization to this file after the run (e.g. arm.mp4 or arm.gif).",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="",
        help=(
            "Text description for MDM motion generation "
            "(e.g. 'a person waves their left arm'). "
            "When provided, loads the MDM model, starts from the sitting pose, "
            "and tracks the generated trajectory. "
            "Without this flag the demo runs without MDM."
        ),
    )
    parser.add_argument(
        "--text_time",
        type=int,
        default=0,
        help="Time (mpc time step) at which the motion description text is applied",
    )
    parser.add_argument(
        "--save_motion",
        type=str,
        default="",
        help=(
            "If set, save an MP4 of the MDM-generated motion to this path "
            "(e.g. 'motion.mp4').  Only used when --text is provided."
        ),
    )
    parser.add_argument(
        "--start_pose",
        type=str,
        default="sitting_pose.pt",
        help=(
            "Name of the .pt file in MDM_ROOT to use as the initial pose"
            " (default: sitting_pose.pt)"
        ),
    )
    args = parser.parse_args()

    demo_fk = SmplLeftArmFK()

    from uncertain_feedback.consts import (  # pylint: disable=wrong-import-position
        MDM_ROOT,
    )
    from uncertain_feedback.motion_generators.mdm.mdm_api import (  # pylint: disable=wrong-import-position
        MdmMotionGenerator,
    )

    gen = MdmMotionGenerator()
    initial_pose = gen.load_hml_pose(MDM_ROOT / args.start_pose)  # (263,)
    initial_arm_aa, initial_body_positions, initial_spine3_aa = gen.decode_pose(
        initial_pose
    )

    demo_target_q = initial_arm_aa.copy() + np.array(
        [
            [0.0, 0.0, 0.0],  # left_collar
            [0.0, -1.6, 0.8],  # left_shoulder
            [0.0, 0.0, 0.0],  # left_elbow
            [0.0, 0.0, 0.0],  # left_wrist
        ]
    )

    demo_mpc = LeftArmMPCMDM(
        horizon=args.horizon,
        n_mpc_samples=args.samples,
        visualize=not args.no_vis,
        fk=demo_fk,
        goals=[demo_target_q],
        spine3_pos=initial_body_positions[9],
        spine3_aa=initial_spine3_aa,
        body_pos=initial_body_positions,
    )

    demo_q = initial_arm_aa.copy()

    for _ in range(args.text_time):
        demo_q = demo_mpc.step(demo_q)

    # Close the visualizer before generation to avoid it freezing/becoming unresponsive.
    # Keep a reference so the recorded pre-MDM frames survive for the saved output.
    pre_mdm_vis = demo_mpc._vis  # pylint: disable=protected-access
    if (
        pre_mdm_vis is not None
        and pre_mdm_vis._live is not None  # pylint: disable=protected-access
    ):
        plt.close(pre_mdm_vis._live.fig)  # pylint: disable=protected-access
    demo_mpc._vis = None  # pylint: disable=protected-access

    print(
        f"Generating MDM trajectory for: '{args.text}' (starting from current MPC state)"
    )
    current_pose = gen.build_pose_from_arm_aa(initial_pose, demo_q)
    trajectory = gen.generate_left_arm_trajectory(
        args.text,
        start_pose=current_pose,
        save_path=args.save_motion or None,
    )  # (n_frames, 4, 3)
    n_frames = trajectory.shape[0]
    cutoff = max(1, round(n_frames * demo_mpc.trajectory_fraction))
    print(
        f"Generated {n_frames} frames; enqueuing first {cutoff}"
        f" ({demo_mpc.trajectory_fraction:.0%})."
    )

    # If MDM switched the backend to Agg (e.g. for video saving), switch back to interactive
    if plt.get_backend().lower() == "agg":
        for backend in ["Qt5Agg", "TkAgg", "Qt6Agg", "WXAgg", "MacOSX"]:
            try:
                plt.switch_backend(backend)
                break
            except Exception:  # pylint: disable=broad-exception-caught
                continue

    demo_mpc.set_mdm_goal(trajectory[cutoff - 1])
    demo_mpc.push_trajectory(
        trajectory[:cutoff]
    )  # prepends first-fraction MDM frames; demo_target_q stays last

    for _ in range(args.steps - args.text_time):
        demo_q = demo_mpc.step(demo_q)

    vis = demo_mpc._vis  # pylint: disable=protected-access
    if args.save and not args.no_vis and vis is not None:
        # Prepend pre-MDM frames so the saved animation covers the full run.
        if (  # type: ignore[unreachable]  # pylint: disable=protected-access
            pre_mdm_vis is not None
            and pre_mdm_vis._live is not None
            and vis._live is not None
        ):
            vis._live.recorded_frames = (  # pylint: disable=protected-access
                pre_mdm_vis._live.recorded_frames  # pylint: disable=protected-access
                + vis._live.recorded_frames  # pylint: disable=protected-access
            )
        vis.finish_live(args.save)

    plt.ioff()
    plt.show()
