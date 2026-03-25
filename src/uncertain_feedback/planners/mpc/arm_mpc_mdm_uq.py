"""MDM-extended MPC with uncertainty quantification for the SMPL left arm.

:class:`LeftArmMPCMDMUQ` inherits from :class:`LeftArmMPCMDM` and adds
uncertainty-aware MDM trajectory selection via:

* :meth:`query_mdm_with_uncertainty` — draws ``n_diffusion_samples`` diffusion
  samples, clusters them with a :class:`TrajectoryClusterer`, shows the
  interactive cluster-picker window, and enqueues the mean of the chosen
  cluster.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from uncertain_feedback.planners.mpc.arm_mpc_mdm import LeftArmMPCMDM
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK
from uncertain_feedback.uncertainty.base import TrajectoryClusterer
from uncertain_feedback.uncertainty.cluster_picker import pick_cluster
from uncertain_feedback.uncertainty.xyz_clusterer import XyzPositionClusterer


class LeftArmMPCMDMUQ(LeftArmMPCMDM):
    """MDM-extended MPC with uncertainty quantification.

    Extends :class:`LeftArmMPCMDM` with uncertainty-aware trajectory
    selection: instead of blindly following a single diffusion sample, this
    class draws multiple samples, clusters them, presents the clusters to the
    user for selection, and follows the mean trajectory of the chosen cluster.

    Args:
        horizon:             Number of look-ahead steps.
        n_mpc_samples:       Number of candidate action sequences per MPC step.
        max_angle_delta:     Sampling std dev (radians).
        advance_threshold:   Default L2 distance for advancing the goal queue.
        trajectory_fraction: Fraction of MDM frames to enqueue (e.g. ``0.75``).
                             Also controls which timestep is shown as a ghost
                             arm in the cluster-picker window.  Defaults to
                             :attr:`~LeftArmMPCMDM.TRAJECTORY_FRACTION`.
        goals:               Initial list of ``(4, 3)`` target configurations.
        goal_threshold:      Threshold passed to the base class.
        visualize:           If ``True``, open a live matplotlib window.
        fk:                  :class:`SmplLeftArmFK` instance (required when
                             ``visualize=True``).
        spine3_pos:          ``(3,)`` world position of spine3.
        spine3_aa:           ``(3,)`` world axis-angle of spine3.
        body_pos:            ``(22, 3)`` world joint positions for the grey
                             background skeleton.
        n_diffusion_samples: Number of independent MDM diffusion samples to draw
                             per :meth:`query_mdm_with_uncertainty` call.
        n_clusters:          Number of KMeans clusters when ``clusterer`` is
                             ``None``.  Ignored if ``clusterer`` is provided.
        clusterer:           Custom :class:`TrajectoryClusterer`.  Defaults to
                             :class:`XyzPositionClusterer` with ``n_clusters``.
    """

    def __init__(
        self,
        horizon: int = 10,
        n_mpc_samples: int = 512,
        max_angle_delta: float = 0.0025,
        advance_threshold: float = 0.1,
        trajectory_fraction: float = LeftArmMPCMDM.TRAJECTORY_FRACTION,
        goals: list[np.ndarray] | None = None,
        goal_threshold: float = 0.1,
        visualize: bool = False,
        fk: SmplLeftArmFK | None = None,
        spine3_pos: np.ndarray | None = None,
        spine3_aa: np.ndarray | None = None,
        body_pos: np.ndarray | None = None,
        n_diffusion_samples: int = 512,
        n_clusters: int = 3,
        clusterer: TrajectoryClusterer | None = None,
    ) -> None:
        super().__init__(
            horizon=horizon,
            n_mpc_samples=n_mpc_samples,
            max_angle_delta=max_angle_delta,
            advance_threshold=advance_threshold,
            trajectory_fraction=trajectory_fraction,
            goals=goals,
            goal_threshold=goal_threshold,
            visualize=visualize,
            fk=fk,
            spine3_pos=spine3_pos,
            spine3_aa=spine3_aa,
            body_pos=body_pos,
        )
        self._n_diffusion_samples = n_diffusion_samples
        if clusterer is not None:
            self._clusterer = clusterer
        else:
            _fk = fk if fk is not None else SmplLeftArmFK()
            self._clusterer = XyzPositionClusterer(n_clusters, fk=_fk)

    # ------------------------------------------------------------------
    # UQ pipeline
    # ------------------------------------------------------------------

    def query_mdm_with_uncertainty(
        self,
        gen: "MdmMotionGenerator",  # type: ignore[name-defined]  # noqa: F821
        text: str,
        start_pose: np.ndarray | None = None,
        current_arm_aa: np.ndarray | None = None,
    ) -> None:
        """Generate multiple MDM samples, cluster them, let the user pick.

        The full pipeline:

        1. Draw ``num_samples`` trajectories from the diffusion model.
        2. Cluster them with the configured :class:`TrajectoryClusterer`.
        3. Show the interactive cluster-picker window (blocks until chosen).
           Each cluster panel shows the arm at the ``trajectory_fraction``
           timestep as a semi-transparent ghost arm.
        4. Compute the mean trajectory of the selected cluster.
        5. Enqueue the first ``trajectory_fraction`` portion of the mean
           trajectory and set the MDM goal.

        Args:
            gen:        :class:`~uncertain_feedback.motion_generators.mdm\
.mdm_api.MdmMotionGenerator` instance (already loaded or lazy).
            text:       Natural-language motion description.
            start_pose: ``(263,)`` HML263 vector to condition the start of
                        the generated motion.  Pass the output of
                        :meth:`~uncertain_feedback.motion_generators.mdm\
.mdm_api.MdmMotionGenerator.build_pose_from_arm_aa`.
        """
        print(
            f"Generating {self._n_diffusion_samples} MDM samples for: '{text}' …"
        )
        trajectories = gen.generate_left_arm_trajectory(
            text,
            start_pose=start_pose,
            num_samples=self._n_diffusion_samples,
        )  # (n_diffusion_samples, n_frames, 4, 3)

        print("Clustering trajectories …")
        labels = self._clusterer.cluster(trajectories)  # (num_samples,)
        print(f"labels shape: {labels.shape}")

        fk = (
            self._vis_config.fk
            if self._vis_config is not None
            else SmplLeftArmFK()
        )
        spine_pos = self._vis_config.spine_pos if self._vis_config is not None else None
        spine_aa = self._vis_config.spine_aa if self._vis_config is not None else None
        body_pos = self._vis_config.body_pos if self._vis_config is not None else None
        chosen_label = pick_cluster(
            trajectories, labels, fk=fk,
            trajectory_fraction=self.trajectory_fraction,
            spine_pos=spine_pos,
            spine_aa=spine_aa,
            body_pos=body_pos,
            current_arm_aa=current_arm_aa,
        )
        print(f"User selected cluster {chosen_label}.")

        chosen_mean = trajectories[labels == chosen_label].mean(axis=0)
        # (n_frames, 4, 3)

        n_frames = chosen_mean.shape[0]
        cutoff = max(1, round(n_frames * self.trajectory_fraction))
        print(
            f"Enqueuing first {cutoff} frames of chosen cluster mean"
            f" ({self.trajectory_fraction:.0%})."
        )
        self.set_mdm_goal(chosen_mean[cutoff - 1])
        self.push_trajectory(chosen_mean[:cutoff])


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SMPL left arm MPC with uncertainty-quantified MDM"
    )
    parser.add_argument("--steps", type=int, default=750, help="Number of MPC steps")
    parser.add_argument("--samples", type=int, default=512, help="MPC # samples")
    parser.add_argument("--horizon", type=int, default=10, help="MPC horizon")
    parser.add_argument(
        "--no-vis", action="store_true", help="Disable live visualization"
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Save the visualization to this file (e.g. arm.mp4 or arm.gif).",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="move my arm up",
        help="Text description for MDM motion generation.",
    )
    parser.add_argument(
        "--text_time",
        type=int,
        default=0,
        help="MPC step at which MDM generation is triggered.",
    )
    parser.add_argument(
        "--save_motion",
        type=str,
        default="",
        help="Save MDM motion video to this path (e.g. motion.mp4).",
    )
    parser.add_argument(
        "--start_pose",
        type=str,
        default="sitting_pose.pt",
        help="Name of the .pt file in MDM_ROOT to use as the initial pose.",
    )
    parser.add_argument(
        "--diffusion-samples",
        type=int,
        default=128,
        help="Number of MDM diffusion samples to draw for uncertainty quantification.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=3,
        help="Number of clusters for trajectory grouping.",
    )
    parser.add_argument(
        "--trajectory-fraction",
        type=float,
        default=LeftArmMPCMDM.TRAJECTORY_FRACTION,
        help=(
            "Fraction of the MDM trajectory to enqueue (default: "
            f"{LeftArmMPCMDM.TRAJECTORY_FRACTION:.0%}). "
            "Also sets the ghost-arm timestep in the cluster picker."
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
            [0.0, 0.0, 0.0],   # left_collar
            [0.0, -1.6, 0.8],  # left_shoulder
            [0.0, 0.0, 0.0],   # left_elbow
            [0.0, 0.0, 0.0],   # left_wrist
        ]
    )

    demo_mpc = LeftArmMPCMDMUQ(
        horizon=args.horizon,
        n_mpc_samples=args.samples,
        visualize=not args.no_vis,
        fk=demo_fk,
        goals=[demo_target_q],
        spine3_pos=initial_body_positions[9],
        spine3_aa=initial_spine3_aa,
        body_pos=initial_body_positions,
        n_diffusion_samples=args.diffusion_samples,
        n_clusters=args.n_clusters,
        trajectory_fraction=args.trajectory_fraction,
    )

    demo_q = initial_arm_aa.copy()

    for _ in range(args.text_time):
        demo_q = demo_mpc.step(demo_q)

    # Close the visualizer before generation to avoid it freezing.
    pre_mdm_vis = demo_mpc._vis  # pylint: disable=protected-access
    if (
        pre_mdm_vis is not None
        and pre_mdm_vis._live is not None  # pylint: disable=protected-access
    ):
        plt.close(pre_mdm_vis._live.fig)  # pylint: disable=protected-access
    demo_mpc._vis = None  # pylint: disable=protected-access

    current_pose = gen.build_pose_from_arm_aa(initial_pose, demo_q)
    demo_mpc.query_mdm_with_uncertainty(gen, args.text, start_pose=current_pose,
                                        current_arm_aa=demo_q)

    # If MDM switched the backend to Agg, switch back to interactive.
    if plt.get_backend().lower() == "agg":
        for backend in ["Qt5Agg", "TkAgg", "Qt6Agg", "WXAgg", "MacOSX"]:
            try:
                plt.switch_backend(backend)
                break
            except Exception:  # pylint: disable=broad-exception-caught
                continue

    for _ in range(args.steps - args.text_time):
        demo_q = demo_mpc.step(demo_q)

    vis = demo_mpc._vis  # pylint: disable=protected-access
    if args.save and not args.no_vis and vis is not None:
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
