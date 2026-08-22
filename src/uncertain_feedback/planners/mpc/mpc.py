"""The sampling MPC for the SMPL left arm, composed from pluggable modules.

One planner class, four slots:

* **goal space** (``cartesian=``): the queue of targets the solve loop steers
  toward once feedback playback is done.
* **action space** (``robot_actions=``): what the solve loop samples — human
  arm deltas composed on SO(3) (default) or robot joint deltas projected back
  through the measured grasp.
* **feedback method** (``feedback=``): an MDM playback buffer for
  natural-language corrections, with an optional UQ layer
  (``feedback.uq``) that samples, clusters, and picks.
* **constraints** (``constraints=``): named feasibility constraints that mask
  infeasible rollouts to infinite cost and screen feedback playback.

Every solve call runs the same loop: warm-start the sampling mean, draw and
integrate rollouts through the action space, evaluate the goal space's stage
cost under the constraints, and keep the argmin as the next warm start.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Mapping

import numpy as np

from uncertain_feedback.envs.base import ExecutionEnv
from uncertain_feedback.envs.kinematic import KinematicEnv
from uncertain_feedback.planners.mpc.action_spaces import (
    ActionSpace,
    HumanArmActions,
    RobotActionsConfig,
    RobotJointActions,
    StageCost,
)
from uncertain_feedback.planners.mpc.action_spaces.base import RolloutBatch
from uncertain_feedback.planners.mpc.constraints import (
    CONSTRAINT_BUILDERS,
    FeasibilityConstraint,
)
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    ElbowHeightCost,
    LearnablePreferenceCost,
)
from uncertain_feedback.planners.mpc.feedback import FeedbackConfig, MdmFeedback
from uncertain_feedback.planners.mpc.goal_spaces import (
    CartesianConfig,
    CartesianGoalSpace,
)
from uncertain_feedback.planners.mpc.kinematics import (
    Q_DIM,
    SmplLeftArmFK,
    q_to_arm_aa,
)
from uncertain_feedback.uncertainty.clustering.base import TrajectoryClusterer
from uncertain_feedback.uncertainty.uq_selector import UqClusterResult, UqSelector

if TYPE_CHECKING:
    from uncertain_feedback.motion_generators.base import MotionGenerator
    from uncertain_feedback.motion_generators.steering import SteeringSpec
    from uncertain_feedback.utils.plot import ArmVisualizer


def _as_state_q(value: np.ndarray, name: str) -> np.ndarray:
    """Return a clavicle/shoulder/elbow planner state."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (Q_DIM,):
        raise ValueError(
            f"{name} must have shape ({Q_DIM},) for "
            "[clavicle rotvec, shoulder rotvec, elbow angle], "
            f"got {arr.shape}"
        )
    return arr


@dataclass
class _VisConfig:
    fk: SmplLeftArmFK
    spine_pos: np.ndarray | None
    spine_aa: np.ndarray | None
    body_pos: np.ndarray | None = None
    capture: bool = False
    compact: bool = False


class ArmMPC:
    """Sampling-based MPC for the SMPL left arm with pluggable modules.

    Args:
        horizon:         Number of look-ahead steps.
        n_mpc_samples:   Candidate action sequences sampled per solve call.
        max_angle_delta: Std dev of the human-arm action sampling (radians).
        visualize:       Open a live matplotlib window updated each step.
                         Requires ``fk``.
        fk:              :class:`SmplLeftArmFK` instance.
        spine3_pos:      ``(3,)`` world position of spine3 (default: T-pose).
        spine3_aa:       ``(3,)`` world axis-angle of spine3 (default: zeros).
        body_pos:        ``(22, 3)`` background skeleton joint positions.
        extra_costs:     Composite of extra cost terms scoring every rollout.
        seed:            Seed for planner-local action sampling.
        env:             Execution env realizing each commanded step
                         (default: :class:`KinematicEnv`, exact pass-through).
        initial_q:       ``(7,)`` controlled arm state at run start; ghost-arm
                         anchor in the visualiser. Required with ``cartesian``.
        cartesian:       Goal space: spine3-relative Cartesian wrist goals.
        feedback:        Feedback method: MDM playback (+ optional ``uq``).
        constraints:     Mapping of constraint name to its parsed config
                         (see :data:`CONSTRAINT_BUILDERS`).
        robot_actions:   Switch the action space to robot joint deltas.
        clusterer:       Custom :class:`TrajectoryClusterer` for the UQ layer.
    """

    def __init__(
        self,
        *,
        horizon: int = 10,
        n_mpc_samples: int = 512,
        max_angle_delta: float = 0.0025,
        visualize: bool = False,
        fk: SmplLeftArmFK | None = None,
        spine3_pos: np.ndarray | None = None,
        spine3_aa: np.ndarray | None = None,
        body_pos: np.ndarray | None = None,
        extra_costs: CompositeTrajectoryCost | None = None,
        seed: int | None = None,
        env: ExecutionEnv | None = None,
        initial_q: np.ndarray | None = None,
        cartesian: CartesianConfig | None = None,
        feedback: FeedbackConfig | None = None,
        constraints: Mapping[str, object] | None = None,
        robot_actions: RobotActionsConfig | None = None,
        clusterer: TrajectoryClusterer | None = None,
    ) -> None:
        if cartesian is not None and fk is None:
            raise ValueError("fk is required when cartesian goals are configured.")
        if cartesian is not None and initial_q is None:
            raise ValueError(
                "initial_q is required when cartesian goals are configured."
            )
        if visualize and fk is None:
            raise ValueError("visualize=True requires `fk` to be provided.")
        if constraints and env is None:
            raise ValueError("feasibility constraints require a robot env.")
        if robot_actions is not None and env is None:
            raise ValueError("robot actions require a robot env.")
        if constraints and robot_actions is not None:
            raise ValueError(
                "feasibility constraints compose only with the human action "
                "space; robot rollouts are feasible by construction."
            )

        self._horizon = horizon
        self._n_mpc_samples = n_mpc_samples
        self._max_angle_delta = max_angle_delta
        self._rng = np.random.default_rng(seed) if seed is not None else None
        self.visualize = visualize
        self._extra_costs = extra_costs or CompositeTrajectoryCost()
        self._env: ExecutionEnv = env if env is not None else KinematicEnv()
        self._fk: SmplLeftArmFK = fk if fk is not None else SmplLeftArmFK()
        self._spine3_pos = (
            np.asarray(spine3_pos, dtype=np.float64)
            if spine3_pos is not None
            else self._fk.tpose_spine3_pos
        )
        self._spine3_aa = (
            np.asarray(spine3_aa, dtype=np.float64)
            if spine3_aa is not None
            else np.zeros(3, dtype=np.float64)
        )
        self._body_pos = (
            np.asarray(body_pos, dtype=np.float64) if body_pos is not None else None
        )
        self._initial_q = (
            np.asarray(initial_q, dtype=np.float64) if initial_q is not None else None
        )

        self._constraints: tuple[FeasibilityConstraint, ...] = tuple(
            CONSTRAINT_BUILDERS[name][1](
                cfg,
                env=self._env,
                fk=self._fk,
                spine3_pos=self._spine3_pos,
                spine3_aa=self._spine3_aa,
            )
            for name, cfg in (constraints or {}).items()
        )

        self._human_actions = HumanArmActions(
            self._fk,
            self._rng,
            n_mpc_samples,
            horizon,
            max_angle_delta,
            zero_first_sample=bool(self._constraints),
        )
        self._actions: ActionSpace = (
            RobotJointActions(
                robot_actions,
                self._fk,
                self._rng,
                n_mpc_samples,
                horizon,
                self._spine3_pos,
                self._spine3_aa,
            )
            if robot_actions is not None
            else self._human_actions
        )

        self._goal_space: CartesianGoalSpace | None = (
            CartesianGoalSpace(
                [np.asarray(g, dtype=np.float64) for g in cartesian.goals],
                cartesian.threshold,
                self._fk,
                self._spine3_pos,
                self._spine3_aa,
            )
            if cartesian is not None
            else None
        )

        stall_steps = next(
            (
                c.playback_stall_steps
                for c in self._constraints
                if c.playback_stall_steps is not None
            ),
            None,
        )
        self._feedback: MdmFeedback | None = (
            MdmFeedback(
                max_playback_delta=feedback.max_playback_delta,
                trajectory_fraction=feedback.trajectory_fraction,
                stall_steps=stall_steps,
                anchor_correction=feedback.anchor_correction,
            )
            if feedback is not None
            else None
        )
        self._uq: UqSelector | None = (
            UqSelector(feedback.uq, self._fk, clusterer=clusterer)
            if feedback is not None and feedback.uq is not None
            else None
        )
        self._last_uq_result: UqClusterResult | None = None

        if visualize:
            self._vis_config: _VisConfig | None = _VisConfig(
                self._fk, spine3_pos, spine3_aa, body_pos=body_pos
            )
        else:
            self._vis_config = None
        self._vis: ArmVisualizer | None = None

        # Warm-start: previous best plan shifted forward by one step
        self._prev_best: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Capability flags
    # ------------------------------------------------------------------

    @property
    def has_feedback(self) -> bool:
        """Whether a feedback method is configured."""
        return self._feedback is not None

    @property
    def has_uq(self) -> bool:
        """Whether the feedback method carries a UQ layer."""
        return self._uq is not None

    @property
    def has_goal_space(self) -> bool:
        """Whether a goal space is configured."""
        return self._goal_space is not None

    @property
    def has_constraints(self) -> bool:
        """Whether any feasibility constraint is active."""
        return bool(self._constraints)

    @property
    def uses_robot_actions(self) -> bool:
        """Whether the solve loop samples robot joint deltas."""
        return isinstance(self._actions, RobotJointActions)

    # ------------------------------------------------------------------
    # Goal space
    # ------------------------------------------------------------------

    @property
    def current_cartesian_goal(self) -> np.ndarray | None:
        """The active Cartesian goal, or ``None`` without one."""
        return self._goal_space.current_goal if self._goal_space is not None else None

    def append_cartesian_goal(self, goal: np.ndarray) -> None:
        """Add a Cartesian goal to the back of the queue."""
        assert self._goal_space is not None
        self._goal_space.append(goal)

    def goal_reached(self, q: np.ndarray) -> bool:
        """Whether ``q`` has reached the final goal so the rollout can end.

        :func:`~uncertain_feedback.planners.mpc.rollout.run_planning_loop` polls this
        (together with :attr:`mdm_ready_to_terminate`) to stop early. Always
        ``False`` without a goal space (a pure-feedback rollout runs its full
        step budget).
        """
        if self._goal_space is None:
            return False
        return self._goal_space.reached(q)

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    @property
    def mdm_tracking_complete(self) -> bool:
        """True once feedback playback has finished (or never started)."""
        return self._feedback is None or not self._feedback.in_playback()

    @property
    def mdm_ready_to_terminate(self) -> bool:
        """True only once a correction has been loaded AND finished playing.

        ``mdm_tracking_complete`` is ``True`` *before* any correction is queued
        (no frames yet), so it cannot by itself gate early termination — that
        would let the rollout end before the demonstrated correction ever runs.
        Always ``True`` for planners without a feedback method.
        """
        if self._feedback is None:
            return True
        return self._feedback.started and not self._feedback.in_playback()

    @property
    def trajectory_fraction(self) -> float:
        """Fraction of generated feedback frames enqueued per query."""
        assert self._feedback is not None
        return self._feedback.trajectory_fraction

    @trajectory_fraction.setter
    def trajectory_fraction(self, value: float) -> None:
        assert self._feedback is not None
        self._feedback.trajectory_fraction = value

    @property
    def last_uq_result(self) -> UqClusterResult | None:
        """Return the most recent UQ cluster result, if any."""
        return self._last_uq_result

    def remaining_mdm_trajectory(self, current_q: np.ndarray) -> np.ndarray | None:
        """Return the current pose followed by the unexecuted playback targets."""
        if self._feedback is None:
            return None
        return self._feedback.remaining(_as_state_q(current_q, "current_q"))

    def set_mdm_goal(self, goal_q: np.ndarray) -> None:
        """Set the feedback end-of-trajectory goal marker.

        Stores ``goal_q`` and immediately updates the live visualizer if it is
        already open; otherwise the marker is applied when the window first
        opens.
        """
        assert self._feedback is not None
        self._feedback.mdm_goal = _as_state_q(goal_q, "goal_q")
        if self._vis is not None:
            self._vis.update_mdm_goal(
                q_to_arm_aa(self._feedback.mdm_goal, self._fk.elbow_hinge_axis)
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

        Scope: this checks the generated *waypoints* only.  It does not check
        the rate-limited interpolation between frames, nor the ease-in segment
        from the live pose into ``frames[0]``.  For a nonlinear feature such as
        ``elbow_height`` (an FK output), a joint-space geodesic between two
        in-range poses can briefly leave the range, so a clean report here does
        not strictly guarantee the executed motion stays in-band.

        Args:
            frames: ``(n_frames, 7)`` planner-state trajectory.
            tol:    Violation magnitude below which a frame is treated as
                    in-range (guards against floating-point noise).

        Returns:
            One human-readable warning string per violated cost term (empty
            when the trajectory respects every configured range cost).
        """
        frames = np.asarray(frames, dtype=np.float64)
        if frames.ndim != 2 or frames.shape[1:] != (7,):
            raise ValueError(
                "frames must have shape (n_frames, 7) for "
                "[clavicle rotvec, shoulder rotvec, elbow angle], "
                f"got {frames.shape}"
            )
        aa_frames = q_to_arm_aa(frames, self._fk.elbow_hinge_axis)
        warnings: list[str] = []
        n_frames = len(frames)
        for term in self._extra_costs.terms():
            if not isinstance(term, LearnablePreferenceCost):
                continue
            values = term.feature_values(aa_frames)
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
        current_q: np.ndarray | None = None,
    ) -> None:
        """Validate a generated trajectory and queue it for direct playback.

        The trajectory is checked against the configured range safety costs
        (see :meth:`validate_trajectory`); violations are warned about but do
        not block playback.  With ``current_q`` (the live measured
        configuration) and feasibility constraints active, frames the
        constraints rule out are dropped before playback starts.  The
        surviving frames are stored so that :meth:`step` follows them one
        frame per step; once exhausted, the MPC resumes toward the goal space.

        Args:
            frames: ``(n_frames, 7)`` canonical arm trajectory or
                    ``(n_frames, 3, 3)`` decoded axis-angle trajectory.
            current_q: Live measured configuration at push time; enables the
                    constraint screening.
        """
        feedback = self._feedback
        assert feedback is not None, "push_trajectory requires a feedback method."
        frames = np.asarray(frames, dtype=np.float64)
        q_frames = (
            frames
            if frames.shape[-1:] == (Q_DIM,)
            else self._fk.arm_aa_to_q_batch(frames, self._spine3_aa)
        )
        screened = current_q is not None and bool(self._constraints)
        if screened:
            current = np.asarray(current_q, dtype=np.float64)
            for constraint in self._constraints:
                q_frames = constraint.screen_frames(q_frames, current)
                if len(q_frames) == 0:
                    return
        feedback.reset_stall()

        warnings = self.validate_trajectory(q_frames)
        if warnings:
            print(
                "[validate] generated trajectory violates safety costs "
                "(following anyway):"
            )
            for warning in warnings:
                print(f"[validate]   {warning}")
        else:
            print("[validate] generated trajectory respects all safety costs.")

        feedback.set_frames(q_frames)
        self.reset_warmstart()

        # Notify live visualiser of the new preview frame (last frame).
        if self._vis is not None and feedback.preview_q is not None:
            self._vis.update_trajectory_preview(
                q_to_arm_aa(feedback.preview_q, self._fk.elbow_hinge_axis)
            )
        if screened:
            # The correction path set the goal marker to the unscreened last
            # frame.
            self.set_mdm_goal(q_frames[-1])

    def query_mdm_with_uncertainty(
        self,
        gen: MotionGenerator,
        text: str,
        start_pose: np.ndarray | None = None,
        current_q: np.ndarray | None = None,
        auto_cluster: int | None = None,
        mdm_frames: int | None = None,
        frozen_body: bool = False,
        default_scale: float = 1.0,
        cluster_selector: (
            Callable[[dict[int, np.ndarray]], int | tuple[int, float]] | None
        ) = None,
        steering: SteeringSpec | None = None,
    ) -> np.ndarray:
        """Generate multiple MDM samples, cluster, pick, and queue the mean.

        Runs the UQ layer's sample → cluster → pick pipeline (see
        :meth:`UqSelector.query`), then enqueues the first
        ``trajectory_fraction`` portion of the chosen cluster mean and sets
        the feedback goal marker.
        """
        feedback = self._feedback
        assert feedback is not None and self._uq is not None, (
            "query_mdm_with_uncertainty requires a feedback method with a uq " "layer."
        )
        result = self._uq.query(
            gen,
            text,
            start_pose=start_pose,
            current_q=current_q,
            auto_cluster=auto_cluster,
            mdm_frames=mdm_frames,
            frozen_body=frozen_body,
            default_scale=default_scale,
            cluster_selector=cluster_selector,
            steering=steering,
            trajectory_fraction=feedback.trajectory_fraction,
            spine3_pos=self._spine3_pos,
            spine3_aa=self._spine3_aa,
            body_pos=self._body_pos,
        )
        self._last_uq_result = result
        chosen_mean = result.chosen_mean  # (n_frames, 3, 3)
        if feedback.anchor_correction and current_q is not None:
            chosen_mean = self._fk.anchor_arm_trajectory(
                chosen_mean, current_q, self._spine3_aa
            )

        n_frames = chosen_mean.shape[0]
        cutoff = max(1, round(n_frames * feedback.trajectory_fraction))
        print(
            f"Enqueuing first {cutoff} frames of chosen cluster mean"
            f" ({feedback.trajectory_fraction:.0%})."
        )
        self.set_mdm_goal(
            self._fk.arm_aa_to_q(chosen_mean[cutoff - 1], self._spine3_aa)
        )
        self.push_trajectory(chosen_mean[:cutoff], current_q=current_q)
        return chosen_mean

    # ------------------------------------------------------------------
    # Solve loop
    # ------------------------------------------------------------------

    def _constrained(self, stage_cost: StageCost) -> StageCost:
        """Mask rollouts the feasibility constraints rule out to infinite cost."""
        if not self._constraints:
            return stage_cost
        constraints = self._constraints

        def cost(batch: RolloutBatch) -> np.ndarray:
            feasible = np.ones(batch.aa_trajs.shape[0], dtype=bool)
            for constraint in constraints:
                feasible &= constraint.rollout_feasible(batch)
            return np.where(feasible, stage_cost(batch), np.inf)

        return cost

    def _solve_sampling(
        self,
        current_q: np.ndarray,
        stage_cost: StageCost,
        actions: ActionSpace,
    ) -> tuple[RolloutBatch, int]:
        """One warm-started sample → rollout → cost → argmin pass."""
        # Warm-start: shift previous best plan by one step; fill last with zeros
        if self._prev_best is not None:
            mean = np.concatenate([self._prev_best[1:], np.zeros((1, Q_DIM))], axis=0)
        else:
            mean = np.zeros((self._horizon, Q_DIM), dtype=np.float64)
        batch = actions.rollouts(self._env, current_q, mean)
        costs = actions.shape_costs(batch, stage_cost)
        best = int(np.argmin(costs))
        self._prev_best = batch.actions[best]
        return batch, best

    def solve(self, current_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sample human-arm action sequences and return the best one.

        Args:
            current_q: ``(7,)`` clavicle/shoulder/elbow state.

        Returns:
            Tuple of:

            - ``first_action`` ``(7,)``: best delta to apply at the current step.
            - ``plan`` ``(H, 7)``: full best action sequence.

        Raises:
            RuntimeError: Without a goal space to steer toward.
        """
        if self._goal_space is None:
            raise RuntimeError(
                "Goal queue is empty. Add a goal before calling solve()."
            )
        current_q = np.asarray(current_q, dtype=np.float64)
        stage = self._constrained(self._goal_space.stage_cost(self._extra_costs))
        batch, best = self._solve_sampling(current_q, stage, self._human_actions)
        plan = batch.actions[best]
        return plan[0], plan

    def reset_warmstart(self) -> None:
        """Reset the warm-start plan (call before re-running from a new initial
        pose)."""
        self._prev_best = None

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def step(
        self,
        current_q: np.ndarray,
        advance_threshold: float | None = None,
    ) -> np.ndarray:
        """Perform one MPC step.

        While a feedback trajectory is queued for playback, the arm follows it
        directly — without sampling-based re-planning — at a bounded angular
        speed.  Once the trajectory is exhausted, the MPC resumes sampling
        toward the goal space's queue; with nothing left to do it holds.

        Args:
            current_q: ``(7,)`` current planner state.
            advance_threshold: Unused; kept for API compatibility.

        Returns:
            ``(7,)`` achieved planner state.
        """
        _ = advance_threshold
        current_q = np.asarray(current_q, dtype=np.float64)
        if self._feedback is not None and self._feedback.in_playback():
            return self._playback_step(current_q)
        return self._goal_step(current_q)

    def _playback_step(self, current_q: np.ndarray) -> np.ndarray:
        """Follow the feedback trajectory one rate-limited frame per step."""
        feedback = self._feedback
        assert feedback is not None
        if isinstance(self._actions, RobotJointActions):
            # Each playback frame becomes the target of a one-step robot-action
            # solve instead of a human-q command routed through IK — so
            # playback, too, can only ask for motions the grasp can transmit.
            q_target = feedback.advance(current_q)
            if not feedback.in_playback():
                self.reset_warmstart()
            batch, best = self._solve_sampling(
                current_q, self._actions.tracking_cost(q_target), self._actions
            )
            next_q = self._actions.execute(
                self._env, current_q, self._actions.command(batch, best)
            )
        else:
            q_cmd = feedback.advance(current_q)
            if not feedback.in_playback():
                self.reset_warmstart()
            if any(not c.step_reachable(current_q, q_cmd) for c in self._constraints):
                # Held rather than handed to execution's enumerating fallback.
                q_cmd = current_q
            next_q = self._env.execute(q_cmd)
        dist = (
            float(np.linalg.norm(next_q - feedback.preview_q))
            if feedback.preview_q is not None
            else 0.0
        )
        self._update_vis(next_q, dist, playing=True)
        return next_q

    def _goal_step(self, current_q: np.ndarray) -> np.ndarray:
        """Work through the goal space's queue, holding when it is empty."""
        goal_space = self._goal_space
        if goal_space is None or not goal_space.has_goals:
            return self._actions.hold(self._env, current_q)
        stage = self._constrained(goal_space.stage_cost(self._extra_costs))
        batch, best = self._solve_sampling(current_q, stage, self._actions)
        next_q = self._actions.execute(
            self._env, current_q, self._actions.command(batch, best)
        )
        goal, dist = goal_space.progress(next_q, on_pop=self.reset_warmstart)
        self._update_vis(next_q, dist, cartesian_goal=goal)
        return next_q

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def get_visualizer(self) -> ArmVisualizer | None:
        """Return the active live visualizer, or ``None`` if not yet created."""
        return self._vis

    def close_visualizer(self) -> ArmVisualizer | None:
        """Close the live window and detach the visualizer.

        Returns the detached visualizer so its recorded frames can be reused
        (e.g. to prepend pre-MDM frames after generation).
        """
        vis = self._vis
        if vis is not None:
            vis.close()
        self._vis = None
        return vis

    def set_extra_costs(self, costs: CompositeTrajectoryCost) -> None:
        """Replace the active extra cost terms (e.g. after a preference update)."""
        self._extra_costs = costs
        if self._vis is not None:
            self._vis.update_elbow_height_range(self._elbow_height_world_range())

    def _elbow_height_world_range(self) -> tuple[float, float] | None:
        """Return configured elbow-height bounds as world-space Y coordinates."""
        for term in self._extra_costs._terms:  # pylint: disable=protected-access
            if isinstance(term, ElbowHeightCost):
                spine_y = float(term.context.spine3_pos[1])
                return (
                    spine_y + float(term.min_height),
                    spine_y + float(term.max_height),
                )
        return None

    def set_visualization_mode(self, capture: bool, compact: bool) -> None:
        """Set capture and compact flags on the vis config (call before first
        step)."""
        if self._vis_config is not None:
            self._vis_config.capture = capture
            self._vis_config.compact = compact

    def _update_vis(
        self,
        next_q: np.ndarray,
        dist: float,
        *,
        playing: bool = False,
        cartesian_goal: np.ndarray | None = None,
    ) -> None:
        """Lazily open the live window, then draw this step.

        The arm is drawn orange while following a feedback trajectory and blue
        during the goal phase.
        """
        if self._vis_config is None:
            return
        from uncertain_feedback.utils.plot import (  # pylint: disable=import-outside-toplevel
            ArmVisualizer,
        )

        if self._vis is None:
            if self._goal_space is not None:
                assert self._initial_q is not None
                anchor, show_target_arm = self._initial_q, False
            else:
                anchor, show_target_arm = next_q, True
            self._vis = ArmVisualizer(self._vis_config.fk)
            self._vis.open_live(
                q_to_arm_aa(anchor, self._fk.elbow_hinge_axis),
                self._vis_config.spine_pos,
                self._vis_config.spine_aa,
                body_pos=self._vis_config.body_pos,
                compact=self._vis_config.compact,
                elbow_height_range=self._elbow_height_world_range(),
                show_target_arm=show_target_arm,
            )
            if self._vis_config.capture:
                self._vis.start_capture()
            if self._feedback is not None and self._feedback.mdm_goal is not None:
                self._vis.update_mdm_goal(
                    q_to_arm_aa(self._feedback.mdm_goal, self._fk.elbow_hinge_axis)
                )
            if self._feedback is not None and self._feedback.preview_q is not None:
                self._vis.update_trajectory_preview(
                    q_to_arm_aa(self._feedback.preview_q, self._fk.elbow_hinge_axis)
                )
            if (
                self._goal_space is not None
                and self._goal_space.current_goal is not None
            ):
                self._vis.update_cartesian_target(
                    self._spine3_pos + self._goal_space.current_goal
                )
        if cartesian_goal is not None:
            self._vis.update_cartesian_target(self._spine3_pos + cartesian_goal)
        color = ArmVisualizer.MDM_COLOR if playing else ArmVisualizer.TARGET_COLOR
        self._vis.update_step(
            q_to_arm_aa(next_q, self._fk.elbow_hinge_axis),
            dist=dist,
            color=color,
        )
