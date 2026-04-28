"""Unified entry point for running arm MPC motion planning.

Usage examples::

    # UQ planner with live GUI (default pose + default text prompt)
    python -m uncertain_feedback.planners.run --live

    # Save a compact video without watching
    python -m uncertain_feedback.planners.run --planner arm_mpc_mdm --text "wave left arm" --save out.mp4

    # Custom starting pose and arm override
    python -m uncertain_feedback.planners.run --pose my_pose.pt --arm my_arm.npy --live

    # Plain MPC (no MDM)
    python -m uncertain_feedback.planners.run --planner arm_mpc --live --steps 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from uncertain_feedback.consts import MDM_ROOT
from uncertain_feedback.planners.mpc import (
    ArmVisualizer,
    LeftArmMPCMDM,
    LeftArmMPCMDMUQ,
    SmplLeftArmFK,
    SmplLeftArmMPC,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run arm MPC motion planning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--planner",
        choices=["arm_mpc", "arm_mpc_mdm", "arm_mpc_mdm_uq"],
        default="arm_mpc_mdm_uq",
        help="Planner to run",
    )

    # --- Model ---
    p.add_argument(
        "--model-path",
        type=Path,
        default=None,
        dest="model_path",
        help="Path to MDM weights .pt file. Defaults to the base humanml model.",
    )

    # --- Pose input ---
    p.add_argument(
        "--pose",
        type=Path,
        default=None,
        help=(
            "Path to body pose .pt file (HML263 format). "
            f"Defaults to {MDM_ROOT}/sitting_pose.pt. "
            "Ignored for --planner arm_mpc."
        ),
    )
    p.add_argument(
        "--arm",
        type=Path,
        default=None,
        help="Optional .npy file with (4, 3) arm axis-angles to override the pose's arm.",
    )

    # --- MPC params ---
    p.add_argument("--steps", type=int, default=750, help="MPC steps to run")
    p.add_argument("--samples", type=int, default=512, help="CEM sample count")
    p.add_argument("--horizon", type=int, default=10, help="MPC horizon length")

    # --- Visualization ---
    p.add_argument(
        "--live",
        action="store_true",
        help="Show interactive matplotlib window while running",
    )
    p.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save video to this path (.mp4 or .gif). Uses compact 1-panel layout.",
    )
    p.add_argument("--fps", type=int, default=20, help="FPS for saved video")

    # --- MDM args (arm_mpc_mdm, arm_mpc_mdm_uq) ---
    p.add_argument(
        "--text",
        type=str,
        default="move my arm up",
        help="Natural-language MDM motion description (mdm/uq planners only)",
    )
    p.add_argument(
        "--text-time",
        type=int,
        default=0,
        dest="text_time",
        help="MPC step at which MDM generation is triggered",
    )
    p.add_argument(
        "--save-motion",
        type=Path,
        default=None,
        dest="save_motion",
        help="Save the raw MDM full-body motion video to this path (arm_mpc_mdm only)",
    )
    p.add_argument(
        "--trajectory-fraction",
        type=float,
        default=LeftArmMPCMDM.TRAJECTORY_FRACTION,
        dest="trajectory_fraction",
        help="Fraction of MDM trajectory to enqueue",
    )

    # --- UQ args (arm_mpc_mdm_uq) ---
    p.add_argument(
        "--diffusion-samples",
        type=int,
        default=128,
        dest="diffusion_samples",
        help="Number of MDM diffusion samples to draw for UQ (uq planner only)",
    )
    p.add_argument(
        "--n-clusters",
        type=int,
        default=3,
        dest="n_clusters",
        help="Number of trajectory clusters (uq planner only)",
    )
    p.add_argument(
        "--auto-cluster",
        type=int,
        default=None,
        dest="auto_cluster",
        help=(
            "Skip the interactive cluster picker and use this cluster index (0-based). "
            "Required when running headlessly (no display)."
        ),
    )

    return p


def _restore_interactive_backend() -> None:
    if plt.get_backend().lower() == "agg":
        for backend in ("Qt5Agg", "TkAgg", "Qt6Agg", "WXAgg", "MacOSX"):
            try:
                plt.switch_backend(backend)
                break
            except Exception:  # pylint: disable=broad-exception-caught
                continue


def _get_vis(mpc: SmplLeftArmMPC) -> ArmVisualizer | None:
    return mpc.get_visualizer()


def main() -> None:
    args = build_parser().parse_args()

    uses_mdm = args.planner in ("arm_mpc_mdm", "arm_mpc_mdm_uq")
    visualize = args.live or (args.save is not None)
    # Compact (1-panel) rendering when saving without live view — faster to render and encode
    compact = (args.save is not None) and not args.live

    # --- Load pose ---
    gen = None
    initial_pose = None
    arm_aa: np.ndarray
    body_pos: np.ndarray | None = None
    spine3_pos: np.ndarray | None = None
    spine3_aa: np.ndarray | None = None

    if uses_mdm:
        from uncertain_feedback.motion_generators.mdm.mdm_api import (  # pylint: disable=import-outside-toplevel
            MdmMotionGenerator,
        )

        gen = MdmMotionGenerator(model_path=args.model_path)
        pose_path = args.pose if args.pose is not None else MDM_ROOT / "sitting_pose.pt"
        initial_pose = gen.load_hml_pose(pose_path)
        arm_aa, body_pos, spine3_aa = gen.decode_pose(initial_pose)
        spine3_pos = body_pos[9]
    else:
        arm_aa = np.zeros((4, 3))

    if args.arm is not None:
        arm_aa = np.load(args.arm)

    fk = SmplLeftArmFK()

    # Default goal: arm raised from the initial pose
    default_goal = arm_aa.copy() + np.array(
        [[0.0, 0.0, 0.0], [0.0, -1.6, 0.8], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )

    # --- Build planner ---
    common: dict = dict(
        horizon=args.horizon,
        n_mpc_samples=args.samples,
        visualize=visualize,
        fk=fk,
    )

    mpc: SmplLeftArmMPC
    if args.planner == "arm_mpc":
        mpc = SmplLeftArmMPC(**common, goals=[default_goal])
    elif args.planner == "arm_mpc_mdm":
        mpc = LeftArmMPCMDM(
            **common,
            goals=[default_goal],
            spine3_pos=spine3_pos,
            spine3_aa=spine3_aa,
            body_pos=body_pos,
            trajectory_fraction=args.trajectory_fraction,
        )
    else:
        mpc = LeftArmMPCMDMUQ(
            **common,
            goals=[default_goal],
            spine3_pos=spine3_pos,
            spine3_aa=spine3_aa,
            body_pos=body_pos,
            trajectory_fraction=args.trajectory_fraction,
            n_diffusion_samples=args.diffusion_samples,
            n_clusters=args.n_clusters,
        )

    # Propagate capture / compact flags into the vis config
    if mpc._vis_config is not None:  # pylint: disable=protected-access
        mpc._vis_config.capture = args.save is not None  # pylint: disable=protected-access
        mpc._vis_config.compact = compact  # pylint: disable=protected-access

    # --- MPC loop ---
    from tqdm import tqdm  # pylint: disable=import-outside-toplevel

    q = arm_aa.copy()
    mdm_triggered = False
    pre_mdm_vis: ArmVisualizer | None = None

    for step in tqdm(range(args.steps), desc="MPC", unit="step"):
        # Trigger MDM generation at the configured step
        if uses_mdm and args.text and step == args.text_time and not mdm_triggered:
            mdm_triggered = True
            assert gen is not None and initial_pose is not None

            # Close vis before the blocking MDM computation to avoid GUI freeze
            pre_mdm_vis = mpc.close_visualizer()

            current_pose = gen.build_pose_from_arm_aa(initial_pose, q)

            if args.planner == "arm_mpc_mdm":
                traj = gen.generate_left_arm_trajectory(
                    args.text,
                    start_pose=current_pose,
                    save_path=str(args.save_motion) if args.save_motion else None,
                )
                n_frames = traj.shape[0]
                cutoff = max(1, round(n_frames * mpc.trajectory_fraction))
                mpc.set_mdm_goal(traj[cutoff - 1])
                mpc.push_trajectory(traj[:cutoff])
            else:
                mpc.query_mdm_with_uncertainty(
                    gen,
                    args.text,
                    start_pose=current_pose,
                    current_arm_aa=q,
                    auto_cluster=args.auto_cluster,
                )

            # MDM generation can switch matplotlib to the Agg backend; restore it
            _restore_interactive_backend()

        q = mpc.step(q)

    # --- Save video ---
    if args.save and visualize:
        vis = _get_vis(mpc)
        if vis is not None:
            if pre_mdm_vis is not None and pre_mdm_vis._frame_bufs:  # pylint: disable=protected-access
                vis.prepend_frames(pre_mdm_vis._frame_bufs)  # pylint: disable=protected-access
            vis.finish_live(str(args.save), fps=args.fps)

    if args.live:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
