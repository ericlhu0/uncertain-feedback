"""The paper's grounding system: MDM sampling, clustering, optional steering."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from evaluation.approaches.grounders.base import ClusterSelector, Grounder
from evaluation.approaches.steering import NoSteering, Steering
from evaluation.metrics.grounding.structs import GroundingResult
from uncertain_feedback.motion_generators.base import MotionGenerator
from uncertain_feedback.motion_generators.steering import SteeringSpec
from uncertain_feedback.planners.mpc.config import MpcRunConfig
from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.planners.mpc.kinematics import anchor_q_trajectory
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
        cfg: MpcRunConfig,
        gen: MotionGenerator | None,
        user: SimulatedUser,
        seed: int,
        episode_dir: Path,
    ) -> None:
        super().reset(cfg, gen, user, seed, episode_dir)
        if cfg.feedback is None or cfg.feedback.uq is None:
            raise ValueError("MdmGrounder requires feedback: (with uq:).")
        uq = cfg.feedback.uq
        if self._diffusion_samples is not None:
            uq = replace(uq, diffusion_samples=self._diffusion_samples)
        if self._n_clusters is not None:
            uq = replace(uq, n_clusters=self._n_clusters)
        uq = replace(uq, steering=replace(uq.steering, mode=self.steering.mode))
        self._uq_cfg = uq
        self._steering_spec = (  # pylint: disable=assignment-from-none
            self.steering.spec(self.gen, user, uq.steering, seed=seed)
        )

    def ground(
        self,
        text: str,
        human: Human,
        nominal_plan: np.ndarray,
        cluster_selector: ClusterSelector,
    ) -> GroundingResult:
        del nominal_plan
        assert self._uq_cfg is not None
        uq = self._uq_cfg
        feedback = self.cfg.feedback
        assert feedback is not None
        clusterer = make_clusterer(uq.clusterer, uq.n_clusters, fk=human.fk)
        selector = UqSelector(uq, human.fk, clusterer=clusterer)
        result = selector.query(
            self.gen,
            text,
            human,
            prefix=False,
            mdm_frames=feedback.frames,
            default_scale=uq.scale,
            cluster_selector=cluster_selector,
            steering=self._steering_spec,
        )
        candidates = result.cluster_means
        if feedback.anchor_correction:
            # The same re-anchoring production applies (planners/run.py): drop
            # the echoed prefix frame and start the demonstrated shape at the
            # live configuration, so no candidate carries the frame-0 seam.
            candidates = {
                label: human.arm_aa_from_q(
                    anchor_q_trajectory(human.q_from_arm_aa(mean), human.q)
                )
                for label, mean in candidates.items()
            }
        return GroundingResult(
            candidates=candidates,
            chosen_label=result.chosen_label,
            magnitude=result.scale,
            correction_traj=candidates[result.chosen_label],
            samples=result.samples,
            sample_labels=result.labels,
        )
