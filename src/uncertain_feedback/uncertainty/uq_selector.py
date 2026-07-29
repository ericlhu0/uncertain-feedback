"""Uncertainty-aware MDM trajectory selection.

Instead of blindly following a single diffusion sample, :class:`UqSelector`
draws multiple samples, clusters them with a
:class:`~uncertain_feedback.uncertainty.clustering.base.TrajectoryClusterer`,
presents the clusters (interactively or through a headless selector), and
returns the mean trajectory of the chosen cluster.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK, q_to_arm_aa
from uncertain_feedback.uncertainty.cluster_picker import (
    pick_cluster,
    pick_cluster_positions,
    scale_trajectory,
)
from uncertain_feedback.uncertainty.clustering.base import TrajectoryClusterer
from uncertain_feedback.uncertainty.clustering.xyz_clusterer import (
    XyzPositionClusterer,
)

if TYPE_CHECKING:
    from uncertain_feedback.motion_generators.base import MotionGenerator


@dataclass(frozen=True)
class UqConfig:
    """How many MDM samples to draw, how to cluster them, and how to pick one."""

    diffusion_samples: int = 128
    n_clusters: int = 3
    clusterer: str = "agglo_end_pose"
    auto_cluster: int | None = None
    scale: float = 1.0
    # Delegate cluster selection to the configured simulated user (takes effect
    # only when the user has hidden bounds).
    user_cluster: bool = False


@dataclass(frozen=True)
class UqClusterResult:
    """Clustered MDM trajectory result for downstream experiments."""

    chosen_label: int
    labels: np.ndarray
    cluster_means: dict[int, np.ndarray]
    scale: float = 1.0

    @property
    def chosen_mean(self) -> np.ndarray:
        """Return the selected cluster mean trajectory."""
        return self.cluster_means[self.chosen_label]


class UqSelector:
    """Sample → cluster → pick pipeline over MDM diffusion outputs.

    Args:
        cfg:       Sample/cluster counts (``clusterer`` name is resolved by the
                   demo runner; the default clusterer here is
                   :class:`XyzPositionClusterer`).
        fk:        Forward kinematics for pose decoding and the picker window.
        clusterer: Custom :class:`TrajectoryClusterer` overriding the default.
    """

    def __init__(
        self,
        cfg: UqConfig,
        fk: SmplLeftArmFK,
        clusterer: TrajectoryClusterer | None = None,
    ) -> None:
        self._n_diffusion_samples = cfg.diffusion_samples
        self._fk = fk
        if clusterer is not None:
            self._clusterer = clusterer
        else:
            self._clusterer = XyzPositionClusterer(cfg.n_clusters, fk=fk)

    def query(  # pylint: disable=too-many-locals
        self,
        gen: MotionGenerator,
        text: str,
        *,
        start_pose: np.ndarray | None = None,
        current_q: np.ndarray | None = None,
        auto_cluster: int | None = None,
        mdm_frames: int | None = None,
        frozen_body: bool = False,
        default_scale: float = 1.0,
        cluster_selector: (
            Callable[[dict[int, np.ndarray]], int | tuple[int, float]] | None
        ) = None,
        trajectory_fraction: float = 1.0,
        spine3_pos: np.ndarray | None = None,
        spine3_aa: np.ndarray | None = None,
        body_pos: np.ndarray | None = None,
    ) -> UqClusterResult:
        """Generate multiple MDM samples, cluster them, let the user pick.

        The full pipeline:

        1. Draw ``diffusion_samples`` trajectories from the diffusion model.
        2. Cluster them with the configured :class:`TrajectoryClusterer`.
        3. Show the interactive cluster-picker window (blocks until chosen),
           unless ``cluster_selector`` or ``auto_cluster`` picks headlessly.
        4. Compute (and scale) the mean trajectory of the selected cluster.

        Args:
            gen:        Motion generator (already loaded or lazy).
            text:       Natural-language motion description.
            start_pose: ``(263,)`` HML263 vector conditioning the motion start.
            mdm_frames: Exact number of MDM frames to generate. ``None`` keeps
                        the generator default.
            frozen_body: If ``True``, freeze non-left-arm body features during
                        MDM generation.
            cluster_selector: Optional headless chooser called with the
                        cluster-mean trajectories ``{label: (T, 3, 3)}``;
                        returns the chosen label, or a ``(label, magnitude)``
                        tuple whose magnitude overrides ``default_scale``
                        (used by simulated-user experiments). Takes precedence
                        over ``auto_cluster`` and the interactive picker.
        """
        print(f"Generating {self._n_diffusion_samples} motion samples for: '{text}' …")
        generation_t0 = time.perf_counter()
        use_position_uq = getattr(self._clusterer, "supports_positions", False)
        base_spine_aa = (
            np.asarray(spine3_aa, dtype=np.float64)
            if spine3_aa is not None
            else np.zeros(3, dtype=np.float64)
        )
        if use_position_uq:
            positions = gen.generate_left_arm_position_samples(
                text,
                start_pose=start_pose,
                num_samples=self._n_diffusion_samples,
                num_frames=mdm_frames,
                frozen_body=frozen_body,
            )  # (n_diffusion_samples, n_frames, 22, 3)
            trajectories = None
        else:
            positions = None
            trajectories = gen.generate_left_arm_trajectory(
                text,
                start_pose=start_pose,
                num_samples=self._n_diffusion_samples,
                num_frames=mdm_frames,
                frozen_body=frozen_body,
                spine3_aa=base_spine_aa,
            )  # (n_diffusion_samples, n_frames, 3, 3)
        print(
            f"[timing] MDM generation pipeline: {time.perf_counter() - generation_t0:.3f}s"
        )

        print("Clustering trajectories …")
        cluster_t0 = time.perf_counter()
        if positions is not None:
            labels = self._clusterer.cluster_positions(positions)  # type: ignore[attr-defined]
        else:
            assert trajectories is not None
            labels = self._clusterer.cluster(trajectories)
        print(f"[timing] clustering total: {time.perf_counter() - cluster_t0:.3f}s")
        print(f"labels shape: {labels.shape}")

        cluster_means: dict[int, np.ndarray] = {}
        for label in sorted(int(v) for v in np.unique(labels)):
            if positions is not None:
                selected_positions = positions[labels == label].mean(axis=0)
                cluster_means[label] = gen.smpl_positions_to_left_arm_trajectory(
                    selected_positions,
                    spine3_aa=base_spine_aa,
                )
            else:
                assert trajectories is not None
                cluster_means[label] = trajectories[labels == label].mean(axis=0)

        if cluster_selector is not None:
            selection = cluster_selector(cluster_means)
            if isinstance(selection, tuple):
                chosen_label, scale = int(selection[0]), float(selection[1])
            else:
                chosen_label = int(selection)
                scale = default_scale
            print(
                f"Selector chose cluster {chosen_label} at magnitude "
                f"{scale:.2f} (headless mode)."
            )
        elif auto_cluster is not None:
            chosen_label = int(auto_cluster)
            scale = default_scale
            print(
                f"Auto-selected cluster {chosen_label} at magnitude "
                f"{scale:.2f} (headless mode)."
            )
        else:
            fk = self._fk
            spine_pos = (
                np.asarray(spine3_pos, dtype=np.float64)
                if spine3_pos is not None
                else fk.tpose_spine3_pos
            )
            picker_t0 = time.perf_counter()
            if positions is not None:
                pick_result = pick_cluster_positions(
                    positions,
                    labels,
                    fk=fk,
                    trajectory_fraction=trajectory_fraction,
                    spine_pos=spine_pos,
                    spine_aa=base_spine_aa,
                    body_pos=body_pos,
                    current_arm_aa=(
                        q_to_arm_aa(current_q, self._fk.elbow_hinge_axis)
                        if current_q is not None
                        else None
                    ),
                    init_scale=default_scale,
                    recluster=self._clusterer.cluster_positions,  # type: ignore[attr-defined]
                    n_clusters=self._clusterer.n_clusters,
                )
                refined_positions = positions[pick_result.sample_indices].mean(axis=0)
                cluster_means[pick_result.root_label] = (
                    gen.smpl_positions_to_left_arm_trajectory(
                        refined_positions,
                        spine3_aa=base_spine_aa,
                    )
                )
            else:
                assert trajectories is not None
                pick_result = pick_cluster(
                    trajectories,
                    labels,
                    fk=fk,
                    trajectory_fraction=trajectory_fraction,
                    spine_pos=spine_pos,
                    spine_aa=base_spine_aa,
                    body_pos=body_pos,
                    current_arm_aa=(
                        q_to_arm_aa(current_q, self._fk.elbow_hinge_axis)
                        if current_q is not None
                        else None
                    ),
                    init_scale=default_scale,
                    recluster=self._clusterer.cluster,
                    n_clusters=self._clusterer.n_clusters,
                )
                cluster_means[pick_result.root_label] = trajectories[
                    pick_result.sample_indices
                ].mean(axis=0)
            chosen_label = pick_result.root_label
            scale = pick_result.scale
            print(
                f"[timing] cluster picker total: {time.perf_counter() - picker_t0:.3f}s"
            )
            print(f"User selected cluster {chosen_label} at magnitude {scale:.2f}.")

        # Scale the chosen cluster's motion magnitude (direction preserved).
        # Only the tracked cluster is scaled; other means stay at raw scale.
        if scale != 1.0:
            cluster_means[chosen_label] = scale_trajectory(
                cluster_means[chosen_label], scale
            )
        return UqClusterResult(
            chosen_label=chosen_label,
            labels=np.asarray(labels, dtype=np.intp),
            cluster_means=cluster_means,
            scale=scale,
        )
