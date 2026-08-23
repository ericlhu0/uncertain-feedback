"""The paper's grounding system: MDM sampling, clustering, optional steering."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from evaluation.approaches.grounders.base import ClusterSelector, Grounder
from evaluation.approaches.steering import NoSteering, Steering
from evaluation.rig import EvalRig
from evaluation.structs import GroundingResult, InteractionTask
from uncertain_feedback.motion_generators.steering import SteeringSpec
from uncertain_feedback.planners.mpc.kinematics import q_to_arm_aa
from uncertain_feedback.simulated_users import SimulatedUser
from uncertain_feedback.uncertainty import UqConfig, UqSelector, make_clusterer


class MdmGrounder(Grounder):
    """Ground feedback with the text-to-motion model and cluster selection.

    ``diffusion_samples``/``n_clusters`` override the planner config's
    ``feedback.uq`` values when set; the steering module always comes from
    the approach's steering axis, so the planner yaml's steering mode is
    ignored in evaluation (its mechanism knobs still apply).
    """

    requires_generator = True

    def __init__(
        self,
        diffusion_samples: int | None = None,
        n_clusters: int | None = None,
    ) -> None:
        super().__init__()
        self._diffusion_samples = diffusion_samples
        self._n_clusters = n_clusters
        # Written by Approach.__init__ from the steering axis.
        self.steering: Steering = NoSteering()
        self._uq_cfg: UqConfig | None = None
        self._steering_spec: SteeringSpec | None = None

    def reset(
        self,
        rig: EvalRig,
        user: SimulatedUser,
        task: InteractionTask,
        episode_dir: Path,
    ) -> None:
        super().reset(rig, user, task, episode_dir)
        cfg = rig.cfg
        if cfg.feedback is None or cfg.feedback.uq is None:
            raise ValueError("MdmGrounder requires feedback: (with uq:).")
        uq = cfg.feedback.uq
        if self._diffusion_samples is not None:
            uq = replace(uq, diffusion_samples=self._diffusion_samples)
        if self._n_clusters is not None:
            uq = replace(uq, n_clusters=self._n_clusters)
        uq = replace(uq, steering=replace(uq.steering, mode=self.steering.mode))
        self._uq_cfg = uq
        assert rig.gen is not None
        self._steering_spec = self.steering.spec(
            rig.gen, user, uq.steering, seed=task.seed
        )

    def ground(
        self,
        text: str,
        q_feedback: np.ndarray,
        nominal_plan: np.ndarray,
        cluster_selector: ClusterSelector,
        goal: np.ndarray,
    ) -> GroundingResult:
        del nominal_plan, goal
        rig = self.rig
        assert rig.gen is not None and rig.initial_hml_pose is not None
        assert self._uq_cfg is not None
        uq = self._uq_cfg
        clusterer = make_clusterer(uq.clusterer, uq.n_clusters, fk=rig.fk)
        selector = UqSelector(uq, rig.fk, clusterer=clusterer)
        start_pose = rig.gen.build_pose_from_arm_aa(
            rig.initial_hml_pose, q_to_arm_aa(q_feedback, rig.fk.elbow_hinge_axis)
        )
        result = selector.query(
            rig.gen,
            text,
            start_pose=start_pose,
            current_q=q_feedback,
            mdm_frames=rig.cfg.feedback.frames if rig.cfg.feedback else None,
            default_scale=uq.scale,
            cluster_selector=cluster_selector,
            spine3_pos=rig.spine3_pos,
            spine3_aa=rig.spine3_aa,
            body_pos=rig.body_pos,
            steering=self._steering_spec,
        )
        return GroundingResult(
            candidates=result.cluster_means,
            chosen_label=result.chosen_label,
            magnitude=result.scale,
            correction_traj=result.chosen_mean,
        )
