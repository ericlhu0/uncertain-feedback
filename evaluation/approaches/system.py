"""The paper's grounding system: MDM sampling, clustering, optional steering."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from evaluation.approaches.base import Approach, ClusterSelector
from evaluation.structs import GroundingResult, InteractionTask
from uncertain_feedback.motion_generators.steering import (
    SteeringSpec,
    build_steering_spec,
)
from uncertain_feedback.planners.mpc.kinematics import q_to_arm_aa
from uncertain_feedback.uncertainty import UqConfig, UqSelector, make_clusterer


class SystemApproach(Approach):
    """Ground feedback with the text-to-motion model and cluster selection.

    ``steering_mode``/``diffusion_samples``/``n_clusters`` override the
    planner config's ``feedback.uq`` values when set. Steering is built from
    the persona's hidden bounds (the only steering-cost source wired today),
    so a steered run is the known-preference upper bound; steering from the
    learned cost plugs in here once that wiring exists.
    """

    def __init__(
        self,
        name: str = "full",
        learning: str = "lifelong",
        steering_mode: str | None = None,
        diffusion_samples: int | None = None,
        n_clusters: int | None = None,
        learn_from: str = "chosen",
    ) -> None:
        super().__init__(name=name, learning=learning)
        if learn_from not in ("chosen", "nominal"):
            raise ValueError("learn_from must be 'chosen' or 'nominal'.")
        self.learn_from = learn_from
        self._steering_mode = steering_mode
        self._diffusion_samples = diffusion_samples
        self._n_clusters = n_clusters
        self._uq_cfg: UqConfig | None = None
        self._steering_spec: SteeringSpec | None = None

    def _reset_grounding(self, task: InteractionTask) -> None:
        cfg = self.rig.cfg
        if cfg.feedback is None or cfg.feedback.uq is None:
            raise ValueError("SystemApproach requires feedback: (with uq:).")
        uq = cfg.feedback.uq
        if self._diffusion_samples is not None:
            uq = replace(uq, diffusion_samples=self._diffusion_samples)
        if self._n_clusters is not None:
            uq = replace(uq, n_clusters=self._n_clusters)
        if self._steering_mode is not None:
            uq = replace(uq, steering=replace(uq.steering, mode=self._steering_mode))
        self._uq_cfg = uq
        self._steering_spec = None
        if uq.steering.mode != "off":
            assert self.rig.gen is not None
            self._steering_spec = build_steering_spec(
                self.rig.gen, self.user, uq.steering, seed=task.seed
            )
            if self._steering_spec is None:
                print(
                    f"[evaluation] {self.name}: steering unsupported for persona "
                    f"{self.user.name}; sampling unsteered.",
                    flush=True,
                )

    def ground(
        self,
        text: str,
        q_feedback: np.ndarray,
        nominal_plan: np.ndarray,
        cluster_selector: ClusterSelector,
    ) -> GroundingResult:
        del nominal_plan
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
