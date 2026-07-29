"""Shared interface for execution environments.

An :class:`ExecutionEnv` is the boundary between the planner and the world
that physically moves the human arm. Each MPC step the planner produces a
commanded ``(7,)`` joint configuration; the env realizes it (kinematically,
through a simulated robot, or on real hardware) and returns the configuration
actually achieved, which feeds the next planning step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from uncertain_feedback.envs.grasp import MeasuredGrasp
    from uncertain_feedback.envs.robot_fk import RobotChainFK
    from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK


class ExecutionEnv(ABC):
    """Abstract base class for execution environments."""

    def __init__(self) -> None:
        self._fk: SmplLeftArmFK | None = None
        self._spine3_pos: np.ndarray | None = None
        self._spine3_aa: np.ndarray | None = None
        self._body_pos: np.ndarray | None = None

    def set_pose_context(
        self,
        fk: SmplLeftArmFK,
        spine3_pos: np.ndarray | None,
        spine3_aa: np.ndarray | None,
        body_pos: np.ndarray | None = None,
    ) -> None:
        """Attach the run's kinematics so the env matches the planner's FK.

        ``body_pos`` is the ``(22, 3)`` decoded initial body pose; envs that
        render the whole body use it, defaulting to the SMPL T-pose.
        """
        self._fk = fk
        self._spine3_pos = spine3_pos
        self._spine3_aa = spine3_aa
        self._body_pos = body_pos

    def pose_context(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """The ``(spine3_pos, spine3_aa, body_pos)`` the run should plan against.

        Read *after* :meth:`initial_q`. Default: whatever
        :meth:`set_pose_context` was given. Envs that measure the person place
        the torso where they measured it, so the anchor they end up with is not
        the one the config assumed — and every Cartesian goal is relative to it.
        """
        return self._spine3_pos, self._spine3_aa, self._body_pos

    def initial_q(self, q_nominal: np.ndarray) -> np.ndarray:
        """Return the ``(7,)`` configuration the arm actually starts in.

        Called once before planning, after :meth:`set_pose_context`. Default:
        the nominal configuration from the run config. Envs that *measure* the
        person override this to report where the arm really is, so the planner
        starts from the truth rather than from what the config assumed.
        """
        return q_nominal

    def show_goal(self, q_goal: np.ndarray) -> None:
        """Display the ``(7,)`` configuration the run drives toward.

        Called once the goal is known, which is after :meth:`initial_q` — a
        measured torso anchor moves every spine3-relative goal with it. Default:
        envs with nothing to draw ignore it.
        """

    def preview(
        self,
        plan: Callable[[Callable[[np.ndarray, np.ndarray | None], None]], None],
    ) -> bool:
        """Show the planned trajectory before executing any of it.

        Called once after :meth:`show_goal`. ``plan(on_step)`` rolls the same
        planner and costs forward from :meth:`initial_q` offline, calling
        ``on_step(q, robot_q)`` after each planned step — the ``(7,)`` human
        configuration and, for robot-action planners, the planned ``(7,)``
        robot joints (``None`` otherwise) — so the env can draw the rollout
        live while it is being planned. It is a full MPC rollout, so it is
        passed as a callable and only envs that actually show something pay
        for it. Envs that move a real person override this to let the operator
        watch the plan and approve it; returning ``False`` aborts the run
        before anything moves. Default: nothing to show, so proceed without
        rolling anything out.
        """
        _ = plan
        return True

    @abstractmethod
    def execute(self, q_cmd: np.ndarray) -> np.ndarray:
        """Realize one commanded ``(7,)`` arm configuration.

        Blocks until the step has been executed and returns the ``(7,)``
        configuration actually achieved.
        """

    def hold(self, q: np.ndarray) -> np.ndarray:
        """Handle one planner step that commands no motion.

        Default: send nothing and report ``q`` unchanged. Robot envs may
        override to actively hold the arm at ``q`` (e.g. keep an impedance
        controller engaged).
        """
        return q

    # ------------------------------------------------------------------
    # Robot-action interface (envs with a robot only)
    # ------------------------------------------------------------------
    #
    # A robot-action planner samples robot joint targets instead of human
    # configurations, so it needs the robot's kinematics, state, and grasp —
    # and an execute that takes joint targets directly, bypassing the
    # human-q → grasp FK → IK pipeline of :meth:`execute`.

    def robot_fk(self) -> RobotChainFK:
        """Batched ee-chain FK of this env's robot, in the env's world frame."""
        raise NotImplementedError(f"{type(self).__name__} has no robot")

    def current_robot_q(self) -> np.ndarray:
        """The robot's current ``(7,)`` joint configuration."""
        raise NotImplementedError(f"{type(self).__name__} has no robot")

    def robot_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Padded ``(lower, upper)`` joint boxes the robot must stay in."""
        raise NotImplementedError(f"{type(self).__name__} has no robot")

    def robot_max_joint_delta(self) -> float:
        """Per-step joint-motion cap this env's execution applies."""
        raise NotImplementedError(f"{type(self).__name__} has no robot")

    def solve_robot_ik_exact(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        q_seed: np.ndarray,
    ) -> np.ndarray | None:
        """Solve an exact world-frame gripper pose inside the padded joint box."""
        raise NotImplementedError(f"{type(self).__name__} has no robot IK")

    def solve_robot_ik_exact_batch(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        q_seed: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch form of :meth:`solve_robot_ik_exact`.

        Returns candidate joints and a boolean exact-solution mask. Robot envs
        with a vectorized solver override this scalar default.
        """
        solutions = np.asarray(q_seed, dtype=np.float64).copy()
        feasible = np.zeros(solutions.shape[0], dtype=bool)
        for i in range(solutions.shape[0]):
            solution = self.solve_robot_ik_exact(
                target_pos[i], target_quat[i], solutions[i]
            )
            if solution is not None:
                solutions[i] = solution
                feasible[i] = True
        return solutions, feasible

    def track_robot_ik_batch(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        q_seed: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Continuation-only batch IK: solutions on each seed's own branch.

        What a *feasibility gate* wants, where
        :meth:`solve_robot_ik_exact_batch` is what *execution* wants. Execution
        has to command something, so when continuation cannot place a target it
        falls back to enumerating every analytical branch and taking the
        nearest reachable one. A gate is only asking a yes/no question, and that
        fallback answers it wrongly and expensively: wrongly, because a branch
        change is an exact solution the robot cannot actually get to within a
        step, so the gate would pass a rollout the arm then spends tens of
        steps chasing; expensively, because enumeration is serial and one call
        costs about as much as the whole vectorized continuation over every
        sample, paid on exactly the candidates the gate is about to discard.

        Envs whose solver has no continuation/enumeration split fall back to
        the exact solve, which for them is the same answer.
        """
        return self.solve_robot_ik_exact_batch(target_pos, target_quat, q_seed)

    def current_grasp(self, q: np.ndarray) -> MeasuredGrasp:
        """This step's gripper-on-forearm transform, in the env's world frame.

        ``q`` is the current measured human configuration; envs that have not
        grasped yet use it to establish the grasp first.
        """
        raise NotImplementedError(f"{type(self).__name__} has no robot")

    def execute_robot(self, target: np.ndarray) -> np.ndarray:
        """Realize one commanded ``(7,)`` robot joint configuration.

        Blocks until the step has been executed and returns the ``(7,)``
        *human* arm configuration actually achieved.
        """
        raise NotImplementedError(f"{type(self).__name__} has no robot")

    @abstractmethod
    def visualize(self, path: Path | None = None) -> np.ndarray:
        """Render the last executed configuration as an ``(H, W, 3)`` image.

        Saves the image to ``path`` when given. Requires
        :meth:`set_pose_context` and at least one :meth:`execute`/:meth:`hold`.
        """

    @abstractmethod
    def save_video(self, path: str | Path, fps: int = 20) -> None:
        """Write a video of every configuration executed or held so far.

        Requires :meth:`set_pose_context` and at least one
        :meth:`execute`/:meth:`hold`.
        """
