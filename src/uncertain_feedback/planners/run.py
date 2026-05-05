"""Unified entry point for running arm MPC motion planning.

Usage examples::

    # UQ planner with live GUI (default pose + default text prompt)
    python -m uncertain_feedback.planners.run --mpc-config mpc.yaml --live

    # Save a compact video without watching
    python -m uncertain_feedback.planners.run --mpc-config mpc.yaml --text "wave left arm" --save out.mp4

    # Custom starting pose and arm override
    python -m uncertain_feedback.planners.run --mpc-config mpc.yaml --pose my_pose.pt --arm my_arm.npy --live

    # Plain MPC (no MDM)
    python -m uncertain_feedback.planners.run --mpc-config plain_mpc.yaml --live
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np

from uncertain_feedback.consts import MDM_ROOT
from uncertain_feedback.planners.mpc import (
    ArmMPCCartesianNoMDM,
    ArmVisualizer,
    LeftArmMPCCartesian,
    LeftArmMPCMDM,
    LeftArmMPCMDMUQ,
    SmplLeftArmFK,
    SmplLeftArmMPC,
)
from uncertain_feedback.planners.mpc.config import load_mpc_config
from uncertain_feedback.planners.mpc.costs import MpcCostContext, build_extra_costs

_MDM_PLANNERS = {"arm_mpc_mdm", "arm_mpc_mdm_uq", "arm_mpc_cartesian"}


@dataclass
class _InitialPoseState:
    """Initial controlled-arm state plus the whole-body context it came from."""

    arm_aa: np.ndarray
    fixed_collar_aa: np.ndarray
    body_pos: np.ndarray | None = None
    spine3_pos: np.ndarray | None = None
    spine3_aa: np.ndarray | None = None
    hml_pose: np.ndarray | None = None

    @classmethod
    def tpose(cls) -> "_InitialPoseState":
        return cls(arm_aa=np.zeros((3, 3)), fixed_collar_aa=np.zeros(3))


def _make_motion_generator(model_path: Path | None) -> Any:
    from uncertain_feedback.motion_generators.mdm.mdm_api import (  # pylint: disable=import-outside-toplevel
        MdmMotionGenerator,
    )

    return MdmMotionGenerator(model_path=model_path)


def _load_initial_pose_state(
    args: argparse.Namespace,
    uses_mdm: bool,
    config_pose: Path | None = None,
    motion_generator_factory: Callable[[Path | None], Any] | None = None,
) -> tuple[Any | None, _InitialPoseState]:
    """Load the optional HML pose used to initialize all planner variants.

    MDM-backed planners keep the historical default sitting pose. Non-MDM
    planners keep their T-pose default unless the user explicitly passes
    ``--pose`` or sets ``pose`` in the YAML config.
    """
    pose_path = args.pose if args.pose is not None else config_pose
    if uses_mdm and pose_path is None:
        pose_path = MDM_ROOT / "demo_pose.pt"
    if pose_path is None:
        return None, _InitialPoseState.tpose()

    factory = motion_generator_factory or _make_motion_generator
    gen = factory(args.model_path)
    hml_pose = gen.load_hml_pose(pose_path)
    arm_aa, body_pos, spine3_aa, fixed_collar_aa = gen.decode_pose_with_collar(
        hml_pose
    )
    return gen, _InitialPoseState(
        arm_aa=np.asarray(arm_aa, dtype=np.float64),
        fixed_collar_aa=np.asarray(fixed_collar_aa, dtype=np.float64),
        body_pos=np.asarray(body_pos, dtype=np.float64),
        spine3_pos=np.asarray(body_pos[9], dtype=np.float64),
        spine3_aa=np.asarray(spine3_aa, dtype=np.float64),
        hml_pose=np.asarray(hml_pose, dtype=np.float64),
    )


def _apply_arm_override(state: _InitialPoseState, arm_path: Path | None) -> None:
    if arm_path is None:
        return

    arm_override = np.load(arm_path)
    if arm_override.shape == (4, 3):
        state.fixed_collar_aa = np.asarray(arm_override[0], dtype=np.float64)
        state.arm_aa = np.asarray(arm_override[1:], dtype=np.float64)
    else:
        state.arm_aa = np.asarray(arm_override, dtype=np.float64)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run arm MPC motion planning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--mpc-config",
        type=Path,
        required=True,
        dest="mpc_config",
        help="Required YAML file with MPC planner, controller settings, and costs.",
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
            "Override the YAML pose path with a body pose .pt file (HML263 format). "
            f"MDM-backed planners default to {MDM_ROOT}/demo_pose.pt; "
            "non-MDM planners use T-pose unless YAML pose or this override is set."
        ),
    )
    p.add_argument(
        "--arm",
        type=Path,
        default=None,
        help=(
            "Optional .npy file with (3, 3) shoulder/elbow/wrist axis-angles. "
            "A legacy (4, 3) file is accepted and its first row fixes the collar."
        ),
    )

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
        "--mdm-frames",
        type=int,
        default=None,
        dest="mdm_frames",
        help="Exact number of MDM frames to generate (1-196). Default is 120.",
    )
    p.add_argument(
        "--frozen-body",
        action="store_true",
        dest="frozen_body",
        help="Freeze non-left-arm body features during MDM generation.",
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
    cfg = load_mpc_config(args.mpc_config)

    uses_mdm = cfg.planner in _MDM_PLANNERS
    visualize = args.live or (args.save is not None)
    # Compact (1-panel) rendering when saving without live view — faster to render and encode
    compact = (args.save is not None) and not args.live

    # --- Load pose ---
    gen, initial_state = _load_initial_pose_state(args, uses_mdm, cfg.pose)
    _apply_arm_override(initial_state, args.arm)
    initial_pose = initial_state.hml_pose
    arm_aa = initial_state.arm_aa
    fixed_collar_aa = initial_state.fixed_collar_aa
    body_pos = initial_state.body_pos
    spine3_pos = initial_state.spine3_pos
    spine3_aa = initial_state.spine3_aa

    fk = SmplLeftArmFK()
    cost_context = MpcCostContext(
        fk=fk,
        spine3_pos=np.asarray(
            spine3_pos if spine3_pos is not None else fk.tpose_spine3_pos,
            dtype=np.float64,
        ),
        spine3_aa=np.asarray(
            spine3_aa if spine3_aa is not None else np.zeros(3), dtype=np.float64
        ),
        fixed_collar_aa=np.asarray(fixed_collar_aa, dtype=np.float64),
    )
    extra_costs = build_extra_costs(cfg.costs, cost_context)

    # Default goal: arm raised from the initial pose
    default_goal = arm_aa.copy() + np.array(
        [[0.0, 0.7, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )

    # --- Build planner ---
    common: dict = dict(
        horizon=cfg.horizon,
        n_mpc_samples=cfg.n_mpc_samples,
        max_angle_delta=cfg.max_angle_delta,
        goal_threshold=cfg.goal_threshold,
        visualize=visualize,
        fk=fk,
        spine3_pos=spine3_pos,
        spine3_aa=spine3_aa,
        fixed_collar_aa=fixed_collar_aa,
        body_pos=body_pos,
        extra_costs=extra_costs,
    )

    mpc: SmplLeftArmMPC
    if cfg.planner == "arm_mpc":
        mpc = SmplLeftArmMPC(
            **common,
            goals=[default_goal],
        )
    elif cfg.planner == "arm_mpc_mdm":
        mpc = LeftArmMPCMDM(
            **common,
            goals=[default_goal],
            advance_threshold=cfg.advance_threshold,
            trajectory_fraction=cfg.trajectory_fraction,
        )
    elif cfg.planner == "arm_mpc_cartesian_no_mdm":
        if not cfg.cartesian.goals:
            raise ValueError(
                "cartesian.goals is required when planner is arm_mpc_cartesian_no_mdm."
            )
        _spine3_ref = spine3_pos if spine3_pos is not None else fk.tpose_spine3_pos
        init_wrist_rel = (
            fk.fk_controlled(arm_aa, fixed_collar_aa, spine3_pos, spine3_aa)[-1]
            - _spine3_ref
        )
        print(f"Initial wrist position (spine3-relative): {init_wrist_rel}")
        mpc = ArmMPCCartesianNoMDM(
            cartesian_goals=[np.array(g) for g in cfg.cartesian.goals],
            initial_arm_aa=arm_aa,
            cartesian_threshold=cfg.cartesian.threshold,
            **common,
        )
    elif cfg.planner == "arm_mpc_cartesian":
        if not cfg.cartesian.goals:
            raise ValueError(
                "cartesian.goals is required when planner is arm_mpc_cartesian."
            )
        _spine3_ref = spine3_pos if spine3_pos is not None else fk.tpose_spine3_pos
        init_wrist_rel = (
            fk.fk_controlled(arm_aa, fixed_collar_aa, spine3_pos, spine3_aa)[-1]
            - _spine3_ref
        )
        print(f"Initial wrist position (spine3-relative): {init_wrist_rel}")
        mpc = LeftArmMPCCartesian(
            cartesian_goals=[np.array(g) for g in cfg.cartesian.goals],
            initial_arm_aa=arm_aa,
            cartesian_threshold=cfg.cartesian.threshold,
            **common,
            advance_threshold=cfg.advance_threshold,
            trajectory_fraction=cfg.trajectory_fraction,
            n_diffusion_samples=cfg.uq.diffusion_samples,
            n_clusters=cfg.uq.n_clusters,
        )
    else:
        mpc = LeftArmMPCMDMUQ(
            **common,
            goals=[default_goal],
            advance_threshold=cfg.advance_threshold,
            trajectory_fraction=cfg.trajectory_fraction,
            n_diffusion_samples=cfg.uq.diffusion_samples,
            n_clusters=cfg.uq.n_clusters,
        )

    # Propagate capture / compact flags into the vis config
    if mpc._vis_config is not None:  # pylint: disable=protected-access
        mpc._vis_config.capture = (
            args.save is not None
        )  # pylint: disable=protected-access
        mpc._vis_config.compact = compact  # pylint: disable=protected-access

    # --- MPC loop ---
    from tqdm import tqdm  # pylint: disable=import-outside-toplevel

    q = arm_aa.copy()
    mdm_triggered = False
    pre_mdm_vis: ArmVisualizer | None = None

    for step in tqdm(range(cfg.steps), desc="MPC", unit="step"):
        # Trigger MDM generation at the configured step
        if uses_mdm and args.text and step == args.text_time and not mdm_triggered:
            mdm_triggered = True
            assert gen is not None and initial_pose is not None

            # Close vis before the blocking MDM computation to avoid GUI freeze
            pre_mdm_vis = mpc.close_visualizer()

            current_pose = gen.build_pose_from_arm_aa(initial_pose, q)

            if cfg.planner == "arm_mpc_mdm":
                traj = gen.generate_left_arm_trajectory(
                    args.text,
                    start_pose=current_pose,
                    save_path=str(args.save_motion) if args.save_motion else None,
                    num_frames=args.mdm_frames,
                    frozen_body=args.frozen_body,
                    spine3_aa=spine3_aa,
                    fixed_collar_aa=fixed_collar_aa,
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
                    auto_cluster=cfg.uq.auto_cluster,
                    mdm_frames=args.mdm_frames,
                    frozen_body=args.frozen_body,
                )

            # MDM generation can switch matplotlib to the Agg backend; restore it
            _restore_interactive_backend()

        q = mpc.step(q)

    # --- Save video ---
    if args.save and visualize:
        vis = _get_vis(mpc)
        if vis is not None:
            if (
                pre_mdm_vis is not None and pre_mdm_vis._frame_bufs
            ):  # pylint: disable=protected-access
                vis.prepend_frames(
                    pre_mdm_vis._frame_bufs
                )  # pylint: disable=protected-access
            vis.finish_live(str(args.save), fps=args.fps)

    if args.live:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
