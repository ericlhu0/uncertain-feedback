"""MDM-extended MPC for the SMPL left arm.

:class:`LeftArmMPCMDM` inherits all sampling logic from
:class:`~uncertain_feedback.planners.mpc.arm_mpc.SmplLeftArmMPC` and adds:

* :meth:`validate_trajectory` — check an MDM trajectory against the configured
  range safety costs (advisory: warns but never blocks).
* :meth:`push_trajectory` — validate then queue an MDM-generated trajectory for
  direct frame-by-frame playback (rather than enqueuing it as goals).
* :meth:`set_mdm_goal` — mark the MDM end-frame in the live visualizer.
* MDM-colored arm rendering (darkorange while following an MDM trajectory).
* ``body_pos`` background skeleton in the live visualizer window.

After generation the trajectory is followed directly — one frame per
:meth:`step` — skipping the per-step sampling optimisation (the trajectory was
already validated against the cost functions).  Once it is exhausted, the MPC
resumes sampling toward the final queued goal.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from uncertain_feedback.planners.mpc.arm_mpc import (
    _VisConfig,
    SmplLeftArmMPC,
)
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    LearnablePreferenceCost,
)
from uncertain_feedback.planners.mpc.kinematics import (
    SmplLeftArmFK,
    _compose_rotvec,
    _rate_limited_step,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uncertain_feedback.utils.plot import ArmVisualizer

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
        advance_threshold:   Default L2 distance below which the MPC advances
                             to the next queued goal during the resume phase.
        max_playback_delta:  Maximum per-joint rotation (radians) applied per
                             step while following the MDM trajectory.  Caps the
                             playback angular speed so large frame-to-frame
                             jumps (and the initial jump from the live pose into
                             the trajectory) are traversed smoothly rather than
                             snapped in a single step.
        trajectory_fraction: Fraction of MDM-generated frames to enqueue
                             (e.g. ``0.75`` enqueues the first 75 % of
                             frames).  Defaults to
                             :attr:`TRAJECTORY_FRACTION`.
        goals:               Initial list of ``(3, 3)`` target configurations.
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

    TRAJECTORY_FRACTION: float = 1
    """Fraction of MDM trajectory frames to enqueue (default 75 %)."""

    def __init__(
        self,
        horizon: int = 10,
        n_mpc_samples: int = 512,
        max_angle_delta: float = 0.0025,
        advance_threshold: float = 0.1,
        max_playback_delta: float = 0.1,
        trajectory_fraction: float = 1,
        goals: list[np.ndarray] | None = None,
        goal_threshold: float = 0.1,
        visualize: bool = False,
        fk: SmplLeftArmFK | None = None,
        spine3_pos: np.ndarray | None = None,
        spine3_aa: np.ndarray | None = None,
        body_pos: np.ndarray | None = None,
        extra_costs: CompositeTrajectoryCost | None = None,
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
            body_pos=body_pos,
            extra_costs=extra_costs,
        )
        self.advance_threshold = advance_threshold
        self._max_playback_delta = max_playback_delta
        self.trajectory_fraction = trajectory_fraction
        self.visualize = visualize
        self._mdm_spine3_pos = (
            np.asarray(spine3_pos, dtype=np.float64)
            if spine3_pos is not None
            else None
        )
        self._mdm_spine3_aa = (
            np.asarray(spine3_aa, dtype=np.float64)
            if spine3_aa is not None
            else np.zeros(3, dtype=np.float64)
        )
        self._body_pos = (
            np.asarray(body_pos, dtype=np.float64)
            if body_pos is not None
            else None
        )
        if visualize:
            if fk is None:
                raise ValueError("visualize=True requires `fk` to be provided.")
            self._vis_config = _VisConfig(
                fk, spine3_pos, spine3_aa, body_pos=body_pos
            )

        # Last frame of the MDM trajectory, shown as a goal marker.
        self._mdm_goal: np.ndarray | None = None
        # Cutoff frame shown as a ghost arm in the live visualiser.
        self._preview_q: np.ndarray | None = None

        # Direct-playback buffer: the validated MDM trajectory is followed frame
        # by frame (one frame per step) rather than tracked with the sampling
        # MPC.  Once exhausted, the MPC resumes toward the final queued goal.
        self._playback_frames: np.ndarray | None = None
        self._playback_idx: int = 0

    # ------------------------------------------------------------------
    # MDM-specific public API
    # ------------------------------------------------------------------

    @property
    def mdm_tracking_complete(self) -> bool:
        """True once the MDM trajectory playback has finished (or never started)."""
        return not self._in_playback()

    def _in_playback(self) -> bool:
        """Whether an MDM trajectory is still being followed frame by frame."""
        return (
            self._playback_frames is not None
            and self._playback_idx < len(self._playback_frames)
        )

    def validate_trajectory(
        self,
        frames: np.ndarray,
        tol: float = 1e-6,
    ) -> list[str]:
        """Check a trajectory against the configured range safety costs.

        Evaluates every configured :class:`LearnablePreferenceCost` term (e.g.
        elbow height / flexion / shoulder abduction) on each frame and reports
        any frame whose feature value falls outside the term's ``[min, max]``
        range.  This is advisory only — it never blocks playback — but surfaces
        which safety constraints the generated trajectory would violate.

        Scope: this checks the generated *waypoints* only.  It does not check the
        rate-limited interpolation between frames, nor the ease-in segment
        from the live pose into ``frames[0]``.  For a nonlinear feature such as
        ``elbow_height`` (an FK output), a joint-space geodesic between two
        in-range poses can briefly leave the range, so a clean report here does
        not strictly guarantee the executed motion stays in-band.

        Args:
            frames: ``(n_frames, 3, 3)`` axis-angle trajectory.
            tol:    Violation magnitude below which a frame is treated as
                    in-range (guards against floating-point noise).

        Returns:
            One human-readable warning string per violated cost term (empty when
            the trajectory respects every configured range cost).
        """
        frames = np.asarray(frames, dtype=np.float64)
        warnings: list[str] = []
        n_frames = len(frames)
        for term in self._extra_costs.terms():
            if not isinstance(term, LearnablePreferenceCost):
                continue
            values = term.feature_values(frames)
            violation = np.maximum(term.min_value - values, 0.0) + np.maximum(
                values - term.max_value, 0.0
            )
            bad = np.flatnonzero(violation > tol)
            if bad.size == 0:
                continue
            worst = int(bad[np.argmax(violation[bad])])
            warnings.append(
                f"{term.cost_name}: {bad.size}/{n_frames} frames outside "
                f"[{term.min_value:.3f}, {term.max_value:.3f}] "
                f"(worst frame {worst}: value={values[worst]:.3f})"
            )
        return warnings

    def push_trajectory(
        self,
        frames: np.ndarray,
    ) -> None:
        """Validate an MDM-generated trajectory and queue it for direct playback.

        The trajectory is first checked against the configured range safety
        costs (see :meth:`validate_trajectory`); any violations are warned about
        but do not block playback.  The full-resolution trajectory is then
        stored so that :meth:`step` follows it one frame per step.  Once the
        trajectory is exhausted, the MPC resumes sampling toward the final goal
        already in the queue.

        Args:
            frames:   ``(n_frames, 3, 3)`` axis-angle trajectory for
                      ``[left_shoulder, left_elbow, left_wrist]``,
                      as returned by
                      :meth:`~uncertain_feedback.motion_generators.mdm.mdm_api\
.MdmMotionGenerator.generate_left_arm_trajectory`.
        """
        frames = np.asarray(frames, dtype=np.float64)

        warnings = self.validate_trajectory(frames)
        if warnings:
            print(
                "[validate] generated trajectory violates safety costs "
                "(following anyway):"
            )
            for warning in warnings:
                print(f"[validate]   {warning}")
        else:
            print("[validate] generated trajectory respects all safety costs.")

        self._playback_frames = frames
        self._playback_idx = 0
        self.reset_warmstart()

        # Notify live visualiser of the new preview frame (last MDM frame).
        preview_q = frames[-1].copy()
        self._preview_q = preview_q
        if self._vis is not None:
            self._vis.update_trajectory_preview(preview_q)

    def _advance_playback(self, current_q: np.ndarray) -> np.ndarray:
        """Take one rate-limited step toward the current MDM frame.

        Moves from ``current_q`` toward the active playback frame by at most
        :attr:`_max_playback_delta` radians per joint, advancing the cursor only
        once the frame is reached.  This bounds the angular speed so the initial
        jump from the live pose into the trajectory — and any large
        frame-to-frame jump — is traversed smoothly rather than snapped in a
        single step.  Resets the warm-start once the trajectory is exhausted so
        the MPC resume phase plans from a clean slate.
        """
        assert self._playback_frames is not None
        target_q = self._playback_frames[self._playback_idx]
        next_q, reached = _rate_limited_step(current_q, target_q, self._max_playback_delta)
        if reached:
            self._playback_idx += 1
            if not self._in_playback():
                self.reset_warmstart()
        return next_q

    def set_mdm_goal(self, goal_q: np.ndarray) -> None:
        """Set the MDM end-of-trajectory goal marker.

        Stores ``goal_q`` and immediately updates the live visualizer if it is
        already open.  If the visualizer has not been opened yet (i.e. no
        :meth:`step` has been called), the marker is applied automatically
        when the window first opens.

        Args:
            goal_q: ``(3, 3)`` axis-angle joint angles for the last frame of
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

        While an MDM trajectory is queued for playback, the arm follows it
        directly — without sampling-based re-planning — at a bounded angular
        speed: each step moves toward the current frame by at most
        ``max_playback_delta`` radians per joint (rate limiting), so the
        trajectory was already validated against the safety costs in
        :meth:`push_trajectory`.  Smooth frames are reached in one step; large
        jumps (including the initial jump from the live pose into the
        trajectory) are traversed over several steps.  Once the trajectory is
        exhausted, the MPC resumes sampling toward the final goal in the queue,
        advancing through any remaining goals when within ``advance_threshold``.

        If ``visualize=True`` was set at construction, the live window is
        updated automatically.  The arm is drawn orange while following an MDM
        trajectory and blue once the MPC is reaching the final goal.

        Args:
            current_q:         ``(3, 3)`` current axis-angle joint angles.
            advance_threshold: Distance (L2 norm) below which the MPC advances
                               to the next queued goal during the resume phase.
                               Defaults to :attr:`advance_threshold`.

        Returns:
            ``(3, 3)`` updated axis-angle joint angles.
        """
        playing = self._in_playback()
        if playing:
            next_q = self._advance_playback(np.asarray(current_q, dtype=np.float64))
            dist = float(np.linalg.norm(next_q - self._goals[0])) if self._goals else 0.0
        elif not self._goals:
            # Playback done and no final goal queued (e.g. a headless rollout
            # that only follows the trajectory): hold the last pose.
            next_q = np.asarray(current_q, dtype=np.float64).copy()
            dist = 0.0
        else:
            target_q = self._goals[0]
            first_action, _ = self.solve(current_q)
            next_q = _compose_rotvec(
                np.asarray(current_q, dtype=np.float64), first_action
            )

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
            from uncertain_feedback.utils.plot import ArmVisualizer  # pylint: disable=import-outside-toplevel
            if self._vis is None:
                vis_goal = self._goals[-1] if self._goals else next_q
                self._vis = ArmVisualizer(self._vis_config.fk)
                self._vis.open_live(
                    vis_goal,
                    self._vis_config.spine_pos,
                    self._vis_config.spine_aa,
                    body_pos=self._vis_config.body_pos,
                    compact=self._vis_config.compact,
                    elbow_height_range=self._elbow_height_world_range(),
                )
                if self._vis_config.capture:
                    self._vis.start_capture()
                if self._mdm_goal is not None:
                    self._vis.update_mdm_goal(self._mdm_goal)
                if self._preview_q is not None:
                    self._vis.update_trajectory_preview(self._preview_q)
            color = ArmVisualizer.MDM_COLOR if playing else ArmVisualizer.TARGET_COLOR
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
        "--mdm-frames",
        type=int,
        default=None,
        help="Exact number of MDM frames to generate (1-196). Default is 120.",
    )
    parser.add_argument(
        "--frozen-body",
        action="store_true",
        help="Freeze non-left-arm body features during MDM generation.",
    )
    parser.add_argument(
        "--start_pose",
        type=str,
        default="demo_pose.pt",
        help=(
            "Name of the .pt file in MDM_ROOT to use as the initial pose"
            " (default: demo_pose.pt)"
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
    (
        initial_arm_aa,
        initial_body_positions,
        initial_spine3_aa,
        initial_collar_aa,
    ) = gen.decode_pose_with_collar(initial_pose)
    demo_fk.collar_aa = np.asarray(initial_collar_aa, dtype=np.float64)

    demo_target_q = initial_arm_aa.copy() + np.array(
        [
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
        num_frames=args.mdm_frames,
        frozen_body=args.frozen_body,
    )  # (n_frames, 3, 3)
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
