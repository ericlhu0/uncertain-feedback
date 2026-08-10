"""MDM API for generating left arm motion trajectories from text descriptions.

Provides :class:`MdmMotionGenerator`, which lazily loads the MDM model and
HumanML3D dataset and exposes a simple one-call interface for generating left
arm trajectories.  The caller supplies a starting pose (loaded via
:meth:`MdmMotionGenerator.load_pose`) and the generator constrains all
body joints except the left arm to that pose via inpainting — the same
approach used in ``sample_leftarm.py``.

Typical usage::

    gen = MdmMotionGenerator()

    # Load the starting pose and decode it.
    start_pose = gen.load_pose("path/to/pose.pt")
    initial_q, body_positions, spine3_aa, collar_aa = gen.decode_pose(start_pose)

    # Build a start pose that reflects the current arm configuration.
    current_pose = gen.build_pose_from_arm_aa(start_pose, current_arm_aa)

    # Generate a trajectory from a text prompt.
    trajectory = gen.generate_left_arm_trajectory(
        "a person raises their left arm above their head",
        start_pose=current_pose,
    )  # (n_frames, 3, 3)

    # Validate against the safety costs and queue for direct playback.
    mpc.push_trajectory(trajectory)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

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
from uncertain_feedback.motion_generators.base import MotionGenerator
from uncertain_feedback.motion_generators.mdm.hml_smpl_conversion import (
    hml263_batch_to_smpl_body_pose,
    hml263_batch_to_smpl_positions,
    hml263_to_smpl_body_pose,
    smpl_arm_aa_seq_to_hml263_frames,
    smpl_arm_aa_to_hml263_frame,
    smpl_body_pose_to_arm_aa,
    smpl_body_pose_to_collar_aa,
    smpl_body_pose_to_positions,
    smpl_body_pose_to_spine3_aa,
)
from uncertain_feedback.motion_generators.steering import (
    SteeringEvent,
    SteeringSpec,
    conflict_warning,
    make_cond_fn,
    resample_indices,
)

if TYPE_CHECKING:
    from uncertain_feedback.simulated_users.base import SimulatedUser

_MDM_SUBDIR = MDM_ROOT / "motion-diffusion-model"

_MAX_FRAMES = 196  # HumanML3D hard limit
_FPS = 20  # HumanML3D frame rate
# Leading frames pinned to the start pose during inpainting.  Measured: 8 cuts
# the frame-0 seam teleport roughly in half and a static prefix is as good as a
# real arm history — see CODEBASE_MAP.md, "How many frames to pin".  It also
# sets train_leftarm.py's --n_prefix default and the steering-cost slice in
# torch_features.py, which both read this constant.  The prefix is
# generation-time conditioning only: it is additive to the requested frame
# count and stripped from the returned motion apart from its last frame.
N_PREFIX_FRAMES = 8


def build_inpainting_tensors(  # pylint: disable=too-many-arguments
    torch_module: Any,
    prefix: Any,
    n_prefix: int,
    n_frames: int,
    num_samples: int,
    body_mask: np.ndarray,
) -> tuple[Any, Any]:
    """Build the ``inpainted_motion`` / ``inpainting_mask`` pair for sampling.

    Frames past the prefix hold its **last** frame, so ``body_mask`` freezes the
    non-arm channels against the configuration the generated motion continues
    from.  The first ``n_prefix`` frames are locked on every channel.

    Args:
        torch_module: The ``torch`` module (imported lazily by callers).
        prefix:       ``(K, 263)`` float32 tensor of pinned frames.
        n_prefix:     Number of leading frames to lock.
        n_frames:     Total frames in the generated motion.
        num_samples:  Diffusion batch size.
        body_mask:    ``(263,)`` bool — channels frozen for the whole clip.

    Returns:
        ``(inpainted_motion, inpainting_mask)``, each
        ``(num_samples, 263, 1, n_frames)``.
    """
    frames = prefix[-1].unsqueeze(-1).repeat(1, n_frames)  # (263, n_frames)
    frames[:, : prefix.shape[0]] = prefix.T
    input_motions = frames.unsqueeze(1).unsqueeze(0).repeat(num_samples, 1, 1, 1)

    mask = torch_module.tensor(
        body_mask, dtype=torch_module.bool, device=prefix.device
    )  # (263,)
    mask = (
        mask.unsqueeze(0)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .repeat(num_samples, 1, 1, n_frames)
    )
    mask[..., :n_prefix] = True
    return input_motions, mask


def build_prefix_tensor(torch_module: Any, start_pose: np.ndarray, device: Any) -> Any:
    """Return the ``(N_PREFIX_FRAMES, 263)`` pinned prefix for ``start_pose``.

    A ``(263,)`` pose is expanded into a static prefix — measured to be as good
    as a real history.  A 2-D prefix must already hold exactly
    ``N_PREFIX_FRAMES`` frames: the steering cost in ``torch_features.py``
    drops that many frames, so a shorter or longer prefix would mis-score.
    """
    pose = np.asarray(start_pose)
    if pose.ndim == 1:
        pose = np.repeat(pose[np.newaxis, :], N_PREFIX_FRAMES, axis=0)
    if pose.shape[0] != N_PREFIX_FRAMES:
        raise ValueError(
            f"start_pose prefix must have {N_PREFIX_FRAMES} frames, "
            f"got {pose.shape[0]}"
        )
    return torch_module.tensor(pose, dtype=torch_module.float32, device=device)


def _sync_cuda_if_needed(torch_module: Any) -> None:
    """Synchronize CUDA before wall-clock timings when a GPU is active."""
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def _resolve_num_frames(
    motion_length_seconds: float,
    num_frames: int | None,
) -> int:
    """Resolve and validate the MDM sample length in frames."""
    resolved = (
        int(motion_length_seconds * _FPS) if num_frames is None else int(num_frames)
    )
    if resolved < 1:
        raise ValueError(f"num_frames must be >= 1, got {resolved}")
    if resolved > _MAX_FRAMES:
        raise ValueError(f"num_frames must be <= {_MAX_FRAMES}, got {resolved}")
    return resolved


def _resolve_total_frames(
    motion_length_seconds: float,
    num_frames: int | None,
) -> int:
    """Frames to sample so the requested count survives prefix stripping.

    The pinned prefix is additive: a request for ``n`` frames samples
    ``n + N_PREFIX_FRAMES - 1`` so that ``n`` frames remain once the prefix is
    dropped (its last frame is kept as the output's frame 0).
    """
    resolved = _resolve_num_frames(motion_length_seconds, num_frames)
    n_total = resolved + N_PREFIX_FRAMES - 1
    if n_total > _MAX_FRAMES:
        raise ValueError(
            f"num_frames plus the {N_PREFIX_FRAMES - 1} extra pinned frames must "
            f"be <= {_MAX_FRAMES}, got {resolved} + {N_PREFIX_FRAMES - 1} = {n_total}"
        )
    return n_total


class MdmMotionGenerator(
    MotionGenerator
):  # pylint: disable=too-many-instance-attributes
    """Lazy-loading wrapper for the MDM model.

    The MDM model and HumanML3D dataset are loaded on the first call to
    :meth:`generate_left_arm_trajectory` or :meth:`decode_pose`.
    Subsequent calls reuse the already-loaded resources.

    Args:
        model_path: Path to the MDM weights ``.pt`` file.  Defaults to
                    :data:`uncertain_feedback.consts.MDM_MODEL_WEIGHTS_PATH`.
        seed:       Random seed passed to ``fixseed`` for reproducibility.
        lock_seed:  Reset the seed before every generation so repeated calls
                    with identical inputs produce identical samples.
    """

    # Class-level so it is readable on instances built without __init__.
    _last_steering_events: tuple[SteeringEvent, ...] = ()

    def __init__(
        self,
        model_path: str | Path | None = None,
        seed: int = 10,
        lock_seed: bool = False,
    ) -> None:
        super().__init__()
        self._model_path = (
            Path(model_path) if model_path is not None else MDM_MODEL_WEIGHTS_PATH
        ).resolve()
        self._seed = seed
        self._lock_seed = lock_seed
        self._fixseed: Callable[[int], None] | None = None

        # Populated lazily by _ensure_loaded().
        self._model: Any = None
        self._diffusion: Any = None
        self._data: Any = None
        self._args: Any = None
        self._dist_util: Any = None
        self._not_l_arm_mask: np.ndarray | None = None  # (263,) bool

        # Populated in _ensure_loaded() for start_arm_aa support.
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
        # Fall back to the base model's args.json if the checkpoint has none (e.g. fine-tuned
        # runs that were saved without writing args.json).
        _args_json = self._model_path.parent / "args.json"
        if not _args_json.exists():
            _args_json = MDM_MODEL_WEIGHTS_PATH.parent / "args.json"
            print(
                f"No args.json in {self._model_path.parent.name}/, "
                f"falling back to base model args: {_args_json}"
            )
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

        if getattr(args, "pred_len", 0) == 0:
            args.pred_len = getattr(args, "context_len", 0)

        # Fill in fields that older checkpoints may not have saved in args.json.
        _arg_defaults = {
            "unconstrained": False,
            "pred_len": 0,
            "context_len": 0,
            "use_ema": False,
            "dataset": "humanml",
        }
        for _k, _dv in _arg_defaults.items():
            if not hasattr(args, _k):
                setattr(args, _k, _dv)

        fixseed(self._seed)
        self._fixseed = fixseed
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

        # Store normalization stats for building custom start frames.
        t2m_ds = data.dataset.t2m_dataset
        self._hml_mean = np.asarray(t2m_ds.mean).flatten()[:263]
        self._hml_std = np.asarray(t2m_ds.std).flatten()[:263]

        print("MdmMotionGenerator ready.")

    def _reset_seed_if_locked(self) -> None:
        if self._lock_seed:
            assert self._fixseed is not None
            self._fixseed(self._seed)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_seed(self, seed: int, lock_seed: bool = False) -> None:
        """Re-seed the generator without reloading the model.

        Applies the seed immediately if the model is already loaded;
        otherwise it takes effect when :meth:`_ensure_loaded` runs.
        """
        self._seed = seed
        self._lock_seed = lock_seed
        if self._fixseed is not None:
            self._fixseed(seed)

    def load_pose(self, path: str | Path) -> np.ndarray:
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Decode a ``(263,)`` HML263 pose into arm angles, body positions, spine, and collar.

        Args:
            pose: ``(263,)`` HML263 feature vector (e.g. as returned by
                  :meth:`load_pose`).

        Returns:
            arm_aa:         ``(3, 3)`` left arm axis-angles for
                            ``[left_shoulder, left_elbow, left_wrist]``.
            body_positions: ``(22, 3)`` world joint positions for all SMPL
                            joints.
            spine3_aa:      ``(3,)`` world axis-angle of spine3 (joint 9).
                            Pass this to :meth:`SmplLeftArmFK.fk` as
                            ``spine3_aa``, together with
                            ``spine3_pos=body_positions[9]``, so that arm
                            joint positions computed from ``arm_aa`` match
                            those in ``body_positions``.
            collar_aa:      ``(3,)`` local left-collar axis-angle from the
                            decoded start pose.
        """
        self._ensure_loaded()
        import torch  # pylint: disable=import-outside-toplevel

        pose_t = torch.tensor(
            pose, dtype=torch.float32, device=self._dist_util.dev()
        ).unsqueeze(
            0
        )  # (1, 263)
        body_pose = hml263_to_smpl_body_pose(
            pose_t, self._data, self._fk.tpose_all_joints
        )  # (1, 21, 3)
        arm_aa = smpl_body_pose_to_arm_aa(body_pose[0])  # (3, 3)
        collar_aa = smpl_body_pose_to_collar_aa(body_pose[0])  # (3,)
        body_positions = smpl_body_pose_to_positions(
            body_pose[0], self._fk.tpose_all_joints
        )  # (22, 3)
        spine3_aa = smpl_body_pose_to_spine3_aa(body_pose[0])  # (3,)

        return arm_aa, body_positions, spine3_aa, collar_aa

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
                       from :meth:`load_pose`).
            arm_aa:    ``(3, 3)`` axis-angle for
                       ``[left_shoulder, left_elbow, left_wrist]``.

        Returns:
            ``(263,)`` HML263 feature vector with arm joints patched to
            ``arm_aa``.
        """
        self._ensure_loaded()
        assert self._hml_mean is not None
        assert self._hml_std is not None
        return smpl_arm_aa_to_hml263_frame(
            base_norm=np.asarray(base_pose, dtype=np.float64),
            arm_aa=np.asarray(arm_aa, dtype=np.float64),
            hml_mean=self._hml_mean,
            hml_std=self._hml_std,
            fk=self._fk,
        )

    @property
    def prefix_frames(self) -> int:
        """Number of leading frames pinned as generation-time conditioning."""
        return N_PREFIX_FRAMES

    def build_prefix_from_arm_history(
        self,
        base_pose: np.ndarray,
        arm_aa_seq: np.ndarray,
    ) -> np.ndarray:
        """Patch a sequence of arm configurations into a ``(K, 263)`` prefix.

        The multi-frame counterpart of :meth:`build_pose_from_arm_aa`: pass the
        MPC's recent arm history (oldest → newest, ending at the current
        configuration) and the resulting prefix carries real velocity features
        rather than the ~0 of a single pinned frame.  Feed the result to
        :meth:`generate_left_arm_trajectory` as ``start_pose``, which requires
        ``K == prefix_frames``.

        Args:
            base_pose:  ``(263,)`` HML263 feature vector (e.g. sitting pose
                        from :meth:`load_pose`).
            arm_aa_seq: ``(K, 3, 3)`` axis-angles for
                        ``[left_shoulder, left_elbow, left_wrist]`` per frame,
                        oldest → newest.

        Returns:
            ``(K, 263)`` HML263 prefix frames.
        """
        self._ensure_loaded()
        assert self._hml_mean is not None
        assert self._hml_std is not None
        return smpl_arm_aa_seq_to_hml263_frames(
            base_norm=np.asarray(base_pose, dtype=np.float64),
            arm_aa_seq=np.asarray(arm_aa_seq, dtype=np.float64),
            hml_mean=self._hml_mean,
            hml_std=self._hml_std,
            fk=self._fk,
        )

    @property
    def last_steering_events(self) -> tuple[SteeringEvent, ...]:
        """Per-step steering diagnostics recorded by the most recent generation."""
        return self._last_steering_events

    def build_user_steering_cost(
        self, user: SimulatedUser
    ) -> Callable[[Any], Any] | None:
        """Compile a persona's hidden bounds into a steering cost over ``x̂0``.

        Returns ``None`` when none of the persona's bounds can be scored from
        positions alone (see ``torch_features.supported_bounds``).
        """
        self._ensure_loaded()
        assert self._hml_mean is not None
        assert self._hml_std is not None
        # pylint: disable=import-outside-toplevel
        import torch

        from uncertain_feedback.motion_generators.mdm.torch_features import (
            build_user_bound_cost,
        )

        device = self._dist_util.dev()
        # float32 to match x̂0: float64 stats would upcast the autograd graph.
        mean = torch.tensor(self._hml_mean, dtype=torch.float32, device=device)
        std = torch.tensor(self._hml_std, dtype=torch.float32, device=device)
        return build_user_bound_cost(user, mean, std)

    def _sample_hml(  # pylint: disable=too-many-locals,too-many-statements
        self,
        text: str,
        start_pose: np.ndarray | None,
        num_samples: int,
        n_frames: int,
        frozen_body: bool,
        steering: SteeringSpec | None,
        speed: float | None = None,
    ) -> Any:
        """Draw ``num_samples`` normalized HML263 samples for ``text``.

        Returns ``(num_samples, 263, 1, n_frames)``.  The denoising loop is
        written out here rather than delegated to ``p_sample_loop`` so steering
        can inspect and act on every step's ``x̂0``; with ``steering=None`` it is
        equivalent to ``p_sample_loop``.
        """
        self._ensure_loaded()
        self._reset_seed_if_locked()
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        assert self._hml_mean is not None
        assert self._hml_std is not None
        assert self._not_l_arm_mask is not None
        assert start_pose is not None, "sampling requires a start pose to pin"
        self._align_fk_collar_to_pose(start_pose)

        # pylint: disable=import-outside-toplevel,import-error
        import torch
        from data_loaders.tensors import collate
        from tqdm.auto import tqdm

        dist_util = self._dist_util
        model = self._model
        diffusion = self._diffusion
        args = self._args

        # --- Build model_kwargs via collate (mirrors sample_leftarm.py) ------
        collate_args = [
            {"inp": torch.zeros(n_frames), "tokens": None, "lengths": n_frames}
        ]
        collate_args = [{**arg, "text": text} for arg in collate_args]  # type: ignore[dict-item]
        _, model_kwargs = collate(collate_args)

        # Move mask/lengths tensors to device and expand to num_samples batch.
        for key in ("mask", "lengths"):
            if key in model_kwargs["y"] and hasattr(model_kwargs["y"][key], "to"):
                t = model_kwargs["y"][key].to(dist_util.dev())
                if num_samples > 1 and t.shape[0] == 1:
                    t = t.repeat(num_samples, *([1] * (t.dim() - 1)))
                model_kwargs["y"][key] = t

        # --- Inpainting: start pose + optional frozen body --------------------
        # start_pose is a (263,) pose, expanded into a static prefix, or an
        # (N_PREFIX_FRAMES, 263) prefix from build_prefix_from_arm_history.
        prefix = build_prefix_tensor(torch, start_pose, dist_util.dev())
        body_mask = (
            self._not_l_arm_mask
            if frozen_body
            else np.zeros_like(self._not_l_arm_mask, dtype=bool)
        )
        input_motions, inpainting_mask = build_inpainting_tensors(
            torch, prefix, N_PREFIX_FRAMES, n_frames, num_samples, body_mask
        )
        model_kwargs["y"]["inpainted_motion"] = input_motions
        model_kwargs["y"]["inpainting_mask"] = inpainting_mask

        # --- Classifier-free guidance scale ----------------------------------
        model_kwargs["y"]["scale"] = (
            torch.ones(num_samples, device=dist_util.dev()) * args.guidance_param
        )

        # --- Optional scalar speed channel (--speed_cond checkpoints only) ----
        if getattr(model, "speed_cond", False):
            # pylint: disable=import-outside-toplevel,import-error
            from model.mdm import (  # type: ignore[import-not-found]
                SPEED_NEUTRAL,
                SPEED_SCALE,
            )

            model_kwargs["y"]["speed"] = torch.full(
                (num_samples, 1),
                (SPEED_NEUTRAL if speed is None else speed) / SPEED_SCALE,
                device=dist_util.dev(),
            )

        # --- Run diffusion sampling ------------------------------------------
        print(f"Generating motion for: '{text}' ({n_frames} frames)…")
        _sync_cuda_if_needed(torch)
        diffusion_t0 = time.perf_counter()

        mode = "off" if steering is None else steering.config.mode
        cond_fn: Callable[..., Any] | None = None
        resample_steps: tuple[int, ...] = ()
        guide_from = 0
        if steering is not None:
            resample_steps = steering.config.resample_steps
            if mode == "cg":
                cond_fn = make_cond_fn(steering.cost, steering.config.guidance_weight)
                guide_from = steering.config.guide_from

        # Encoding the text once — as p_sample_loop does — is both faster and
        # necessary: under no_grad the cached embedding stays a leaf tensor, so
        # ClassifierFreeSampleModel's deepcopy still works in cg mode.
        if "text" in model_kwargs["y"]:
            with torch.no_grad():
                model_kwargs["y"]["text_embed"] = model.encode_text(
                    model_kwargs["y"]["text"]
                )

        img = torch.randn(
            num_samples, model.njoints, model.nfeats, n_frames, device=dist_util.dev()
        )
        events: list[SteeringEvent] = []
        for step_idx, i in enumerate(tqdm(list(range(diffusion.num_timesteps))[::-1])):
            t = torch.tensor([i] * num_samples, device=dist_util.dev())
            if cond_fn is not None and step_idx >= guide_from:
                out = diffusion.p_sample_with_grad(
                    model,
                    img,
                    t,
                    clip_denoised=False,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
            else:
                with torch.no_grad():
                    out = diffusion.p_sample(
                        model,
                        img,
                        t,
                        clip_denoised=False,
                        model_kwargs=model_kwargs,
                    )
            img = out["sample"]
            if step_idx in resample_steps:
                assert steering is not None
                with torch.no_grad():
                    costs = steering.cost(out["pred_xstart"]).cpu().numpy()
                # Off the torch RNG stream on purpose, so a steered run and an
                # unsteered one at the same seed share their initial noise.
                indices, ess = resample_indices(
                    costs,
                    steering.config.temperature,
                    np.random.default_rng((steering.seed, step_idx)),
                )
                unique_ancestors = None
                if mode == "resample":
                    unique_ancestors = int(np.unique(indices).size)
                    # Only the latent is reindexed: every per-sample entry of
                    # model_kwargs is an identical broadcast of the same pose.
                    img = img[
                        torch.as_tensor(
                            np.ascontiguousarray(indices), device=dist_util.dev()
                        )
                    ]
                events.append(
                    SteeringEvent(
                        step=step_idx,
                        cost_mean=float(costs.mean()),
                        frac_violating=float((costs > 0).mean()),
                        ess=ess,
                        unique_ancestors=unique_ancestors,
                    )
                )
                if len(events) == 1:
                    warning = conflict_warning(events[0], num_samples)
                    if warning is not None:
                        print(warning)

        self._last_steering_events = tuple(events)
        _sync_cuda_if_needed(torch)
        print(f"[timing] diffusion sampling: {time.perf_counter() - diffusion_t0:.3f}s")
        return img.detach()

    def generate_left_arm_trajectory(  # pylint: disable=too-many-locals
        self,
        text: str,
        motion_length_seconds: float = 6.0,
        start_pose: np.ndarray | None = None,
        save_path: str | Path | None = None,
        num_samples: int = 1,
        num_frames: int | None = None,
        frozen_body: bool = False,
        spine3_aa: np.ndarray | None = None,
        *,
        steering: SteeringSpec | None = None,
        speed: float | None = None,
    ) -> np.ndarray:
        """Generate a left arm motion trajectory from a text description.

        All body joints except the left arm (left_shoulder, left_elbow,
        left_wrist) are inpainted to ``start_pose`` throughout the motion.
        The first ``N_PREFIX_FRAMES`` frames are locked to the arm
        configuration(s) encoded in ``start_pose``.  To start from the MPC's
        current arm state, pass a ``(263,)`` ``start_pose`` built with
        :meth:`build_pose_from_arm_aa` — it is expanded into a static prefix —
        or an ``(N_PREFIX_FRAMES, 263)`` prefix from
        :meth:`build_prefix_from_arm_history` to condition on the arm's recent
        history instead.

        The prefix is **generation-time conditioning only**: it is additive to
        the requested length and stripped from the result, so the returned
        trajectory has exactly the requested number of frames and its frame 0
        is the last pinned frame — the configuration the arm is in right now.
        The ``save_path`` video is a diagnostic and still shows the full clip
        including the prefix.

        Args:
            text:                  Natural-language description of the desired
                                   motion (e.g. ``"a person waves their left
                                   arm"``).
            motion_length_seconds: Length of the generated motion in seconds.
                                   Capped at 9.8 s (HumanML3D maximum).
            start_pose:            ``(263,)`` HML263 feature vector used as
                                   the inpainting base for all joints
                                   throughout the motion, or an
                                   ``(N_PREFIX_FRAMES, 263)`` prefix.  Pass the
                                   output of :meth:`load_pose`,
                                   :meth:`build_pose_from_arm_aa` or
                                   :meth:`build_prefix_from_arm_history`.
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
            num_frames:            Exact number of MDM frames to return.  If
                                   ``None``, derived from
                                   ``motion_length_seconds`` at 20 FPS.
            frozen_body:           If ``True``, inpaint/freeze all non-left-arm
                                   body features for the full motion.  If
                                   ``False``, only the first ``N_PREFIX_FRAMES`` frames are
                                   locked to ``start_pose``.
            spine3_aa:             Optional fixed MPC spine3 world axis-angle.
                                   When provided, the returned goals are
                                   projected into the fixed MPC base instead
                                   of using MDM's body frame.
            steering:              Optional cost-steering spec biasing the
                                   diffusion samples toward a user cost model.
            speed:                 Requested mean left-arm speed in metres per
                                   MDM frame (20 FPS).  Only used by
                                   ``--speed_cond`` checkpoints; ``None`` means
                                   the dataset-neutral value.

        Returns:
            ``(n_frames, 3, 3)`` axis-angle trajectory when ``num_samples==1``.
            ``(num_samples, n_frames, 3, 3)`` when ``num_samples > 1``.
        """
        n_total = _resolve_total_frames(motion_length_seconds, num_frames)
        sample = self._sample_hml(
            text, start_pose, num_samples, n_total, frozen_body, steering, speed
        )
        model = self._model
        data = self._data

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
                vis[0].cpu().numpy().transpose(2, 0, 1)[:n_total]
            )  # (n_total, 22, 3)
            clip = plot_3d_motion(
                str(save_path),
                paramUtil.t2m_kinematic_chain,
                motion,
                title=text,
                dataset="humanml",
                fps=_FPS,
            )
            clip.set_duration(float(n_total) / _FPS).write_videofile(
                str(save_path), fps=_FPS, codec="libx264", audio=False, logger=None
            )
            print(f"Saved motion video to {save_path}")

        # --- Convert normalized HML → SMPL body_pose/XYZ → arm axis-angles ---
        # The prefix is dropped only after decoding: HML263 root features are
        # frame-to-frame velocities that recover_from_ric integrates from frame
        # 0, so slicing raw frames would move every decoded frame.
        use_fixed_base = spine3_aa is not None
        if use_fixed_base:
            hml_vecs = sample[:, :, 0, :].permute(0, 2, 1)
            convert_t0 = time.perf_counter()
            positions = hml263_batch_to_smpl_positions(hml_vecs, data, model)
            arm_aa = self.smpl_positions_to_left_arm_trajectory(
                positions,
                spine3_aa=spine3_aa,
            )
            print(
                "[timing] HML-to-fixed-base arm conversion total: "
                f"{time.perf_counter() - convert_t0:.3f}s"
            )
            arm_aa = arm_aa[:, N_PREFIX_FRAMES - 1 :]
            return arm_aa[0] if num_samples == 1 else arm_aa

        if num_samples == 1:
            # Single-sample fast path: no ThreadPoolExecutor overhead.
            hml_vec = sample[0, :, 0, :].T  # (n_total, 263)
            convert_t0 = time.perf_counter()
            body_pose = hml263_to_smpl_body_pose(
                hml_vec, data, self._fk.tpose_all_joints
            )  # (n_total, 21, 3)
            print(
                "[timing] HML-to-arm conversion: "
                f"{time.perf_counter() - convert_t0:.3f}s"
            )
            arm_aa = smpl_body_pose_to_arm_aa(body_pose)
            return arm_aa[N_PREFIX_FRAMES - 1 :]  # (n_frames, 3, 3)

        # Batch path: recover positions for all samples at once, then IK.
        # sample: (num_samples, 263, 1, n_total) → (num_samples, n_total, 263)
        hml_vecs = sample[:, :, 0, :].permute(0, 2, 1)
        convert_t0 = time.perf_counter()
        body_pose_batch = hml263_batch_to_smpl_body_pose(
            hml_vecs, data, self._fk.tpose_all_joints
        )  # (num_samples, n_total, 21, 3)
        print(
            "[timing] HML-to-arm conversion total: "
            f"{time.perf_counter() - convert_t0:.3f}s"
        )
        arm_aa = smpl_body_pose_to_arm_aa(body_pose_batch)
        return arm_aa[:, N_PREFIX_FRAMES - 1 :]  # (num_samples, n_frames, 3, 3)

    def generate_left_arm_position_samples(  # pylint: disable=too-many-locals
        self,
        text: str,
        motion_length_seconds: float = 6.0,
        start_pose: np.ndarray | None = None,
        num_samples: int = 1,
        num_frames: int | None = None,
        frozen_body: bool = False,
        *,
        steering: SteeringSpec | None = None,
        speed: float | None = None,
    ) -> np.ndarray:
        """Generate MDM samples and return SMPL XYZ positions without IK.

        This is intended for UQ clustering and preview rendering.  It runs the
        same batched diffusion process as :meth:`generate_left_arm_trajectory`,
        then stops after HML recovery / ``rot2xyz`` instead of converting every
        frame to local arm axis-angles.

        As in :meth:`generate_left_arm_trajectory`, the ``N_PREFIX_FRAMES``
        pinned frames are conditioning only: they are additive to
        ``num_frames`` and stripped after decoding, so frame 0 of the result is
        the last pinned frame (the current configuration).  ``start_pose`` is a
        ``(263,)`` pose — expanded into a static prefix — or an
        ``(N_PREFIX_FRAMES, 263)`` prefix.

        Returns:
            ``(num_samples, n_frames, 22, 3)`` global SMPL joint positions,
            where ``n_frames`` is the requested count.
        """
        n_total = _resolve_total_frames(motion_length_seconds, num_frames)
        sample = self._sample_hml(
            text, start_pose, num_samples, n_total, frozen_body, steering, speed
        )

        convert_t0 = time.perf_counter()
        hml_vecs = sample[:, :, 0, :].permute(0, 2, 1)
        positions = hml263_batch_to_smpl_positions(hml_vecs, self._data, self._model)
        print(
            "[timing] HML-to-position conversion total: "
            f"{time.perf_counter() - convert_t0:.3f}s"
        )
        return positions[:, N_PREFIX_FRAMES - 1 :]
