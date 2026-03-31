"""MDM API for generating left arm motion trajectories from text descriptions.

Provides :class:`MdmMotionGenerator`, which lazily loads the MDM model and
HumanML3D dataset and exposes a simple one-call interface for generating left
arm trajectories.  The caller supplies a starting pose (loaded via
:meth:`MdmMotionGenerator.load_hml_pose`) and the generator constrains all
body joints except the left arm to that pose via inpainting — the same
approach used in ``sample_leftarm.py``.

Typical usage::

    gen = MdmMotionGenerator()

    # Load the starting pose and decode it.
    start_pose = gen.load_hml_pose("path/to/pose.pt")
    initial_q, body_positions, spine3_aa = gen.decode_pose(start_pose)

    # Build a start pose that reflects the current arm configuration.
    current_pose = gen.build_pose_from_arm_aa(start_pose, current_arm_aa)

    # Generate a trajectory from a text prompt.
    trajectory = gen.generate_left_arm_trajectory(
        "a person raises their left arm above their head",
        start_pose=current_pose,
    )  # (n_frames, 4, 3)

    # Enqueue trajectory into the MPC.
    mpc.push_trajectory(trajectory)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# sys.path / chdir setup — mirror sample_leftarm.py
# ---------------------------------------------------------------------------

_SRC_ROOT = Path(__file__).resolve().parents[3]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

# pylint: disable=wrong-import-position
from uncertain_feedback.consts import MDM_MODEL_WEIGHTS_PATH, MDM_ROOT
from uncertain_feedback.motion_generators.mdm.hml_smpl_conversion import (
    HmlArmFeatureInfo,
    hml263_batch_to_smpl_body_pose,
    hml263_to_smpl_body_pose,
    smpl_arm_aa_to_hml263_frame,
    smpl_body_pose_to_arm_aa,
    smpl_body_pose_to_positions,
)
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK

_MDM_SUBDIR = MDM_ROOT / "motion-diffusion-model"

_MAX_FRAMES = 196  # HumanML3D hard limit
_FPS = 20  # HumanML3D frame rate


class MdmMotionGenerator:  # pylint: disable=too-many-instance-attributes
    """Lazy-loading wrapper for the MDM model.

    The MDM model and HumanML3D dataset are loaded on the first call to
    :meth:`generate_left_arm_trajectory` or :meth:`decode_pose`.
    Subsequent calls reuse the already-loaded resources.

    Args:
        model_path: Path to the MDM weights ``.pt`` file.  Defaults to
                    :data:`uncertain_feedback.consts.MDM_MODEL_WEIGHTS_PATH`.
        seed:       Random seed passed to ``fixseed`` for reproducibility.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        seed: int = 10,
    ) -> None:
        self._model_path = (
            Path(model_path) if model_path is not None else MDM_MODEL_WEIGHTS_PATH
        )
        self._seed = seed

        # Populated lazily by _ensure_loaded().
        self._model: Any = None
        self._diffusion: Any = None
        self._data: Any = None
        self._args: Any = None
        self._dist_util: Any = None
        self._fk: SmplLeftArmFK = SmplLeftArmFK()
        self._not_l_arm_mask: np.ndarray | None = None  # (263,) bool

        # Populated in _ensure_loaded() for start_arm_aa support.
        self._arm_info: HmlArmFeatureInfo | None = None  # HML263 arm feature offsets
        self._hml_mean: np.ndarray | None = None  # (263,) normalization mean
        self._hml_std: np.ndarray | None = None  # (263,) normalization std

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_loaded(  # pylint: disable=too-many-locals,too-many-statements
        self,
    ) -> None:
        """Load the MDM model and dataset if not already done."""
        if self._model is not None:
            return

        if str(_MDM_SUBDIR) not in sys.path:
            sys.path.insert(0, str(_MDM_SUBDIR))
        os.chdir(_MDM_SUBDIR)

        # pylint: disable=import-outside-toplevel,import-error
        from data_loaders import humanml_utils
        from data_loaders.get_data import get_dataset_loader
        from utils import dist_util
        from utils.fixseed import fixseed
        from utils.model_util import create_model_and_diffusion, load_saved_model
        from utils.sampler_util import ClassifierFreeSampleModel

        # Start from model's saved args so all model/diffusion/dataset fields are present.
        _args_json = self._model_path.parent / "args.json"
        _model_args = {}
        if _args_json.exists():
            with open(_args_json, encoding="utf-8") as _f:
                _model_args = json.load(_f)
        args = argparse.Namespace(**_model_args)

        # Overlay inference config from YAML (inference/sampling/edit settings).
        _config_path = Path(__file__).parent / "mdm_configs" / "mdm_config.yaml"
        with open(_config_path, encoding="utf-8") as _f:
            _cfg = yaml.safe_load(_f)
        _cfg["model_path"] = str(self._model_path)
        for _k, _v in _cfg.items():
            setattr(args, _k, _v)

        if args.pred_len == 0:
            args.pred_len = args.context_len

        fixseed(self._seed)
        dist_util.setup_dist(args.device)

        print("Loading MDM dataset…")
        data = get_dataset_loader(
            name="humanml",
            batch_size=1,
            num_frames=_MAX_FRAMES,
            split="test",
            hml_mode="text_only",
            fixed_len=0,
            pred_len=0,
            device=dist_util.dev(),
        )

        print("Loading MDM model…")
        model, diffusion = create_model_and_diffusion(args, data)
        load_saved_model(model, str(self._model_path), use_avg=args.use_ema)
        model = ClassifierFreeSampleModel(model)
        model.to(dist_util.dev())
        model.eval()

        self._model = model
        self._diffusion = diffusion
        self._data = data
        self._args = args
        self._dist_util = dist_util

        # --- Build left-arm inpainting mask (same logic as sample_leftarm.py) --
        hml_joint_names = humanml_utils.HML_JOINT_NAMES
        n_hml_joints = humanml_utils.NUM_HML_JOINTS
        l_arm_joints = [
            hml_joint_names.index(name)
            for name in ["left_shoulder", "left_elbow", "left_wrist"]
        ]
        l_arm_binary = np.array([i not in l_arm_joints for i in range(n_hml_joints)])
        not_l_arm_mask = np.concatenate(
            (
                [True] * (1 + 2 + 1),
                l_arm_binary[1:].repeat(3),
                l_arm_binary[1:].repeat(6),
                l_arm_binary.repeat(3),
                [True] * 4,
            )
        )
        not_l_arm_mask = not_l_arm_mask | humanml_utils.HML_ROOT_MASK
        self._not_l_arm_mask = not_l_arm_mask  # (263,) bool

        # Precompute HML263 feature offsets for left arm joints.
        # HML263 block layout:
        #   [4 root] [21×3 positions] [21×6 rotations] [22×3 velocities] [4 contacts]
        _rot_block_offset = 4 + 63  # 6D rotation block starts at 67
        _vel_block_offset = (
            4 + 63 + 126
        )  # 193; velocity block uses all 22 joints (not j-1)
        self._arm_info = HmlArmFeatureInfo(
            l_arm_joints=l_arm_joints,
            arm_6d_offsets=[_rot_block_offset + (j - 1) * 6 for j in l_arm_joints],
            arm_vel_offsets=[_vel_block_offset + j * 3 for j in l_arm_joints],
        )

        # Store normalization stats for building custom start frames.
        t2m_ds = data.dataset.t2m_dataset
        self._hml_mean = np.asarray(t2m_ds.mean).flatten()[:263]
        self._hml_std = np.asarray(t2m_ds.std).flatten()[:263]

        print("MdmMotionGenerator ready.")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_hml_pose(self, path: str | Path) -> np.ndarray:
        """Load a saved HML263 pose file and return as a ``(263,)`` numpy
        array.

        Args:
            path: Path to the ``.pt`` file.

        Returns:
            ``(263,)`` HML263 feature vector suitable for use as ``start_pose``
            in :meth:`generate_left_arm_trajectory` or :meth:`decode_pose`.
        """
        import torch  # pylint: disable=import-outside-toplevel

        return (
            torch.load(Path(path), map_location="cpu", weights_only=True)
            .squeeze()
            .numpy()
        )  # (263,)

    def decode_pose(
        self, pose: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode a ``(263,)`` HML263 pose into arm angles and body positions.

        Args:
            pose: ``(263,)`` HML263 feature vector (e.g. as returned by
                  :meth:`load_hml_pose`).

        Returns:
            arm_aa:         ``(4, 3)`` left arm axis-angles for
                            ``[left_collar, left_shoulder, left_elbow,
                            left_wrist]``.
            body_positions: ``(22, 3)`` world joint positions for all SMPL
                            joints.
            spine3_aa:      ``(3,)`` world axis-angle of spine3 (joint 9).
                            Pass this to :meth:`SmplLeftArmFK.fk` as
                            ``spine3_aa``, together with
                            ``spine3_pos=body_positions[9]``, so that arm
                            joint positions computed from ``arm_aa`` match
                            those in ``body_positions``.
        """
        from scipy.spatial.transform import (  # pylint: disable=import-outside-toplevel
            Rotation,
        )

        self._ensure_loaded()
        import torch  # pylint: disable=import-outside-toplevel

        pose_t = torch.tensor(
            pose, dtype=torch.float32, device=self._dist_util.dev()
        ).unsqueeze(
            0
        )  # (1, 263)
        body_pose = hml263_to_smpl_body_pose(
            pose_t, self._data, self._model, self._fk.tpose_all_joints
        )  # (1, 21, 3)
        arm_aa = smpl_body_pose_to_arm_aa(body_pose[0])  # (4, 3)
        body_positions = smpl_body_pose_to_positions(
            body_pose[0], self._fk.tpose_all_joints
        )  # (22, 3)

        # Spine3 world rotation: accumulate local rotations along the spine
        # chain (root→spine1(j=3)→spine2(j=6)→spine3(j=9)).
        # body_pose[j-1] is the local rotation for SMPL joint j.
        spine3_world_rot = (
            Rotation.from_rotvec(body_pose[0][2])  # spine1 (j=3)
            * Rotation.from_rotvec(body_pose[0][5])  # spine2 (j=6)
            * Rotation.from_rotvec(body_pose[0][8])  # spine3 (j=9)
        )
        spine3_aa = spine3_world_rot.as_rotvec()  # (3,)

        return arm_aa, body_positions, spine3_aa

    def build_pose_from_arm_aa(
        self,
        base_pose: np.ndarray,
        arm_aa: np.ndarray,
    ) -> np.ndarray:
        """Patch arm joint features into a base HML263 pose.

        Converts ``arm_aa`` to HML263 6D rotation features and replaces the
        corresponding entries in ``base_pose``.  Use this to construct a
        ``start_pose`` that reflects the MPC's current arm configuration
        before calling :meth:`generate_left_arm_trajectory`.

        Args:
            base_pose: ``(263,)`` HML263 feature vector (e.g. sitting pose
                       from :meth:`load_hml_pose`).
            arm_aa:    ``(4, 3)`` axis-angle for
                       ``[left_collar, left_shoulder, left_elbow,
                       left_wrist]``.

        Returns:
            ``(263,)`` HML263 feature vector with arm joints patched to
            ``arm_aa``.
        """
        self._ensure_loaded()
        assert self._arm_info is not None
        assert self._hml_mean is not None
        assert self._hml_std is not None
        return smpl_arm_aa_to_hml263_frame(
            base_norm=np.asarray(base_pose, dtype=np.float64),
            arm_aa=np.asarray(arm_aa, dtype=np.float64),
            arm_info=self._arm_info,
            hml_mean=self._hml_mean,
            hml_std=self._hml_std,
            fk=self._fk,
        )

    def generate_left_arm_trajectory(  # pylint: disable=too-many-locals
        self,
        text: str,
        motion_length_seconds: float = 6.0,
        start_pose: np.ndarray | None = None,
        save_path: str | Path | None = None,
        num_samples: int = 1,
    ) -> np.ndarray:
        """Generate a left arm motion trajectory from a text description.

        All body joints except the left arm (left_shoulder, left_elbow,
        left_wrist) are inpainted to ``start_pose`` throughout the motion.
        The first 10 frames are locked to the arm configuration encoded in
        ``start_pose``.  To start from the MPC's current arm state, pass a
        ``start_pose`` built with :meth:`build_pose_from_arm_aa`.

        Args:
            text:                  Natural-language description of the desired
                                   motion (e.g. ``"a person waves their left
                                   arm"``).
            motion_length_seconds: Length of the generated motion in seconds.
                                   Capped at 9.8 s (HumanML3D maximum).
            start_pose:            ``(263,)`` HML263 feature vector used as
                                   the inpainting base for all joints
                                   throughout the motion.  Pass the output of
                                   :meth:`load_hml_pose` or
                                   :meth:`build_pose_from_arm_aa`.
            save_path:             If provided, save a full-body visualization
                                   of the generated motion to this path as an
                                   MP4 (e.g. ``"motion.mp4"``).  Uses the same
                                   ``plot_3d_motion`` pipeline as
                                   ``sample_leftarm.py``.  Requires ``ffmpeg``
                                   and ``moviepy``.  Defaults to ``None``
                                   (no video saved).  When ``num_samples > 1``,
                                   only the first sample is visualized.
            num_samples:           Number of independent diffusion samples to
                                   draw in a single forward pass.  Defaults to
                                   ``1`` (backward-compatible).

        Returns:
            ``(n_frames, 4, 3)`` axis-angle trajectory when ``num_samples==1``.
            ``(num_samples, n_frames, 4, 3)`` when ``num_samples > 1``.
        """
        self._ensure_loaded()
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        assert self._arm_info is not None
        assert self._hml_mean is not None
        assert self._hml_std is not None

        # pylint: disable=import-outside-toplevel,import-error
        import torch
        from data_loaders.tensors import collate

        dist_util = self._dist_util
        model = self._model
        diffusion = self._diffusion
        data = self._data
        args = self._args

        n_frames = min(_MAX_FRAMES, int(motion_length_seconds * _FPS))

        # --- Build model_kwargs via collate (mirrors sample_leftarm.py) ------
        collate_args = [
            {"inp": torch.zeros(n_frames), "tokens": None, "lengths": n_frames}
        ]
        collate_args = [{**arg, "text": text} for arg in collate_args]  # type: ignore[dict-item]
        _, model_kwargs = collate(collate_args)

        # Move mask/lengths tensors to device.
        for key in ("mask", "lengths"):
            if key in model_kwargs["y"] and hasattr(model_kwargs["y"][key], "to"):
                model_kwargs["y"][key] = model_kwargs["y"][key].to(dist_util.dev())

        # --- Inpainting: start pose + left-arm-only mask ----------------------
        start_frame = torch.tensor(
            start_pose, dtype=torch.float32, device=dist_util.dev()
        ).unsqueeze(
            -1
        )  # (263, 1)

        # input_motions: (num_samples, 263, 1, n_frames)
        input_motions = (
            start_frame.unsqueeze(0).unsqueeze(-1).repeat(num_samples, 1, 1, n_frames)
        )
        model_kwargs["y"]["inpainted_motion"] = input_motions

        mask_tensor = torch.tensor(
            self._not_l_arm_mask, dtype=torch.bool, device=input_motions.device
        )
        # Expand: (263,) → (num_samples, 263, 1, n_frames)
        model_kwargs["y"]["inpainting_mask"] = (
            mask_tensor.unsqueeze(0)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .repeat(num_samples, 1, 1, n_frames)
        )
        # Lock the first 10 frames to the start frame.
        model_kwargs["y"]["inpainting_mask"][..., :10] = True

        # --- Classifier-free guidance scale ----------------------------------
        model_kwargs["y"]["scale"] = (
            torch.ones(num_samples, device=dist_util.dev()) * args.guidance_param
        )

        # --- Run diffusion sampling ------------------------------------------
        print(f"Generating motion for: '{text}' ({n_frames} frames)…")
        sample = diffusion.p_sample_loop(
            model,
            (num_samples, model.njoints, model.nfeats, n_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
        # sample: (1, 263, 1, n_frames) — normalized HumanML3D

        # --- Optionally save a full-body visualization MP4 -------------------
        if save_path is not None:
            # pylint: disable=import-outside-toplevel,import-error
            from data_loaders.humanml.scripts.motion_process import recover_from_ric
            from data_loaders.humanml.utils import paramUtil
            from data_loaders.humanml.utils.plot_script import plot_3d_motion

            n_joints = 22
            vis = data.dataset.t2m_dataset.inv_transform(
                sample.cpu().permute(0, 2, 3, 1)
            ).float()
            vis = recover_from_ric(vis, n_joints)
            vis = vis.view(-1, *vis.shape[2:]).permute(0, 2, 3, 1)
            vis = model.rot2xyz(
                x=vis,
                mask=None,
                pose_rep="xyz",
                glob=True,
                translation=True,
                jointstype="smpl",
                vertstrans=True,
                betas=None,
                beta=0,
                glob_rot=None,
                get_rotations_back=False,
            )
            motion = (
                vis[0].cpu().numpy().transpose(2, 0, 1)[:n_frames]
            )  # (n_frames, 22, 3)
            clip = plot_3d_motion(
                str(save_path),
                paramUtil.t2m_kinematic_chain,
                motion,
                title=text,
                dataset="humanml",
                fps=_FPS,
            )
            clip.set_duration(float(n_frames) / _FPS).write_videofile(
                str(save_path), fps=_FPS, codec="libx264", audio=False, logger=None
            )
            print(f"Saved motion video to {save_path}")

        # --- Convert normalized HML → SMPL body_pose → arm axis-angles ------
        if num_samples == 1:
            # Single-sample fast path: no ThreadPoolExecutor overhead.
            hml_vec = sample[0, :, 0, :].T  # (n_frames, 263)
            body_pose = hml263_to_smpl_body_pose(
                hml_vec, data, model, self._fk.tpose_all_joints
            )  # (n_frames, 21, 3)
            return smpl_body_pose_to_arm_aa(body_pose)  # (n_frames, 4, 3)

        # Batch path: one rot2xyz call + parallel IK across all samples.
        # sample: (num_samples, 263, 1, n_frames) → (num_samples, n_frames, 263)
        hml_vecs = sample[:, :, 0, :].permute(0, 2, 1)
        body_pose_batch = hml263_batch_to_smpl_body_pose(
            hml_vecs, data, model, self._fk.tpose_all_joints
        )  # (num_samples, n_frames, 21, 3)
        return smpl_body_pose_to_arm_aa(
            body_pose_batch
        )  # (num_samples, n_frames, 4, 3)
