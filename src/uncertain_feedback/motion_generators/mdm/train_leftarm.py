"""Left-arm fine-tuning entry point.

Wraps ``train.train_mdm`` and monkey-patches the generate-during-training
call so that sanity-check samples are produced with a fixed starting pose
and configurable body-freezing behaviour.

Usage (from motion-diffusion-model/ — same cwd as plain training):

    uv run python ../train_leftarm.py \\
        --save_dir ./save/my_finetuned_v1 \\
        --start_pose sitting_pose.pt \\
        --body_mode freeze \\
        --dataset humanml \\
        --resume_checkpoint ./save/humanml_enc_512_50steps/model000750000.pt \\
        --diffusion_steps 50 --mask_frames --use_ema \\
        --batch_size 1 --num_steps 1000 --save_interval 100 \\
        --lr 1e-4 --gen_during_training

Extra flags (``--start_pose``, ``--n_prefix``, ``--body_mode``, and the replay-mix
flags below) are consumed here and stripped from sys.argv before forwarding to
``train.train_mdm``; they are written into the run's ``args.json`` beside MDM's.
Omitting ``--gen_during_training`` means the patched path is never reached —
training is identical to plain train_mdm.

Replay-mix flags (datasets whose ids name their source, ``C…`` correction clips,
``R…`` retimed copies, ``H…`` HumanML3D): ``--distill_weight`` /
``--distill_without_dropout`` (distillation of the correction clips toward the
starting checkpoint), ``--correction_fraction`` (batch composition),
``--clean_prefix`` / ``--prefix_noise`` (the executed-history prefix of ``C``/``R``
clips kept clean in the training input, see :mod:`clean_prefix`), and
``--humanml_root`` (train on a mix directly instead of swapping
``dataset/HumanML3D``). ``--unfreeze_clip_layers`` / ``--clip_lr_scale`` train the top
of CLIP's text tower and ship it in the checkpoint (the PC's speed recipe).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — mirrors sample_leftarm.py
# ---------------------------------------------------------------------------

_SRC_ROOT = Path(__file__).resolve().parents[3]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from uncertain_feedback.consts import MDM_ROOT  # noqa: E402
from uncertain_feedback.motion_generators.mdm.clean_prefix import (  # noqa: E402
    CropFromStart,
    loss_mask_without_prefix,
    noised_prefix,
    splice_prefix,
)
from uncertain_feedback.motion_generators.mdm.mdm_api import (  # noqa: E402
    N_PREFIX_FRAMES,
)

_MDM_SUBMODULE = MDM_ROOT / "motion-diffusion-model"
if str(_MDM_SUBMODULE) not in sys.path:
    sys.path.insert(0, str(_MDM_SUBMODULE))

os.chdir(_MDM_SUBMODULE)

# ---------------------------------------------------------------------------
# Parse our extra args and strip them from sys.argv so train_mdm's parser
# doesn't see unknown flags.
# ---------------------------------------------------------------------------

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument(
    "--start_pose",
    type=str,
    default="sitting_pose.pt",
    help=(
        "Filename (in MDM_ROOT) of the .pt HML263 pose tensor (263, 1) "
        "used as the starting pose during gen-during-training."
    ),
)
_parser.add_argument(
    "--n_prefix",
    type=int,
    default=N_PREFIX_FRAMES,
    help=(
        "Number of leading frames to pin entirely to the starting pose "
        "(default matches inference: mdm_api.N_PREFIX_FRAMES). "
        "Set to 0 for no fixed prefix."
    ),
)
_parser.add_argument(
    "--body_mode",
    choices=["freeze", "free", "both"],
    default="freeze",
    help=(
        "'freeze': freeze all non-left-arm body features every frame (default). "
        "'free': only the prefix frames are constrained; body moves freely. "
        "'both': run each mode separately, saving to frozen/ and free/ subdirs; "
        "produces num_samples x 2*num_repetitions trajectories total."
    ),
)
_parser.add_argument(
    "--correction_fraction",
    type=float,
    default=None,
    help=(
        "For a dataset_replay_mix dataset (ids C... / H...): draw this fraction of every "
        "batch from the correction clips and the rest from HumanML3D, by weighted "
        "sampling with replacement. Default: plain shuffling of the dataset as built."
    ),
)
_parser.add_argument(
    "--distill_weight",
    type=float,
    default=0.0,
    help=(
        "For a dataset_replay_mix dataset (ids C... / H...): add this weight times the "
        "masked L2 between the model's output and a frozen copy of the checkpoint training "
        "starts from, on the correction clips only, the copy given the same text and the "
        "same guidance dropout. Default 0: no distillation."
    ),
)
_parser.add_argument(
    "--distill_without_dropout",
    action="store_true",
    help=(
        "With --distill_weight: train with every dropout of the model off, so its outputs "
        "and the frozen copy's (eval mode) come from the same network function and the "
        "distillation term has no dropout floor to train away."
    ),
)
_parser.add_argument(
    "--humanml_root",
    type=Path,
    default=None,
    help=(
        "Train on this built dataset or replay mix directly (a path relative to the MDM "
        "submodule root or absolute), with its own loader cache under <dir>/mdm_cache, "
        "instead of whatever dataset/HumanML3D holds (the swap finetune_standing.sh "
        "performs), so runs can go in parallel. Its Mean/Std must equal dataset/HumanML3D's."
    ),
)
_parser.add_argument(
    "--humanml_cache",
    type=Path,
    default=None,
    help=(
        "With --humanml_root: where its loader cache goes (default <root>/mdm_cache); the "
        "cache of a 47k-motion mix is ~5 GB, so a filesystem with room."
    ),
)
_parser.add_argument(
    "--unfreeze_clip_layers",
    type=int,
    default=0,
    help=(
        "Fine-tune CLIP's top N text blocks, ln_final and text_projection with the rest "
        "of the model, in float32; every checkpoint then ships them, and a run resumed "
        "from such a checkpoint carries them on. The PC's speed recipe of 2026-09-30 "
        "used 2. Default 0: CLIP frozen, as MDM has it."
    ),
)
_parser.add_argument(
    "--clip_lr_scale",
    type=float,
    default=1.0,
    help=(
        "With --unfreeze_clip_layers: the learning rate of the CLIP weights as a multiple "
        "of --lr (the PC's recipe: 0.1)."
    ),
)
_parser.add_argument(
    "--clean_prefix",
    type=int,
    default=0,
    help=(
        "Keep the first K frames of every clip that carries an executed-history prefix "
        "(ids C... / R...) clean in the training input and out of the loss, and start their "
        "crops at frame 0, so the model learns to continue from the history it is pinned to "
        "at sampling (clean_prefix.py). Default 0: off, the prefix is noised like every "
        "other frame, as the deployed and K-ndsh checkpoints were trained."
    ),
)
_parser.add_argument(
    "--prefix_noise",
    type=int,
    default=None,
    help=(
        "With --clean_prefix: noise the prefix frames to diffusion step min(t, N) instead "
        "of leaving them exactly clean (the PC sweep's final setting: 25 of 50)."
    ),
)
_our_args, _remaining_argv = _parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining_argv

_START_POSE_PATH: Path = MDM_ROOT / _our_args.start_pose
_N_PREFIX: int = _our_args.n_prefix
_BODY_MODE: str = _our_args.body_mode

# ---------------------------------------------------------------------------
# Shared inpainting helper (defined in sample_leftarm)
# ---------------------------------------------------------------------------

from uncertain_feedback.motion_generators.mdm.sample_leftarm import (  # noqa: E402
    apply_leftarm_inpainting,
)

# ---------------------------------------------------------------------------
# Custom generate function (replaces sample.generate.main during training)
# ---------------------------------------------------------------------------


def _generate_leftarm_impl(gen_args):
    """Drop-in replacement for ``sample.generate.main`` used during training.

    Dispatches to ``_run_one`` (defined below) according to ``_BODY_MODE``:

    * ``"freeze"`` / ``"free"`` → one run, ``num_samples × num_reps`` trajectories.
    * ``"both"`` → two runs (``frozen/`` and ``free/`` subdirs),
      ``num_samples × 2 × num_reps`` trajectories total.
    """
    import shutil

    import numpy as np
    import torch
    from data_loaders.humanml.scripts.motion_process import recover_from_ric
    from data_loaders.humanml.utils import paramUtil
    from data_loaders.humanml.utils.plot_script import plot_3d_motion
    from sample.generate import (
        construct_template_variables,
        load_dataset,
        save_multiple_samples,
    )
    from utils import dist_util
    from utils.fixseed import fixseed
    from utils.model_util import create_model_and_diffusion, load_saved_model
    from utils.sampler_util import ClassifierFreeSampleModel

    args = gen_args
    args.batch_size = (
        args.num_samples
    )  # mirrors generate.py — one batch of exactly num_samples
    fixseed(args.seed)
    max_frames = 196 if args.dataset in ["kit", "humanml"] else 60
    fps = 12.5 if args.dataset == "kit" else 20
    n_frames = min(max_frames, int(args.motion_length * fps))
    total_num_samples = args.num_samples * args.num_repetitions

    data = load_dataset(args, max_frames, n_frames)

    model, diffusion = create_model_and_diffusion(args, data)
    # Never the EMA copy: with avg_model_beta 0.9999 it is still ~61% base model
    # after 5000 steps, so EMA samples would show no fine-tuning progress. This
    # also matches inference, which overlays use_ema: false (mdm_config.yaml).
    load_saved_model(model, args.model_path, use_avg=False)
    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)
    model.to(dist_util.dev())
    model.eval()

    motion_shape = (args.batch_size, model.njoints, model.nfeats, n_frames)
    sample_fn = diffusion.p_sample_loop

    # Pull one batch from the test split for text prompts and lengths.
    _, model_kwargs = next(iter(data))
    model_kwargs["y"] = {
        k: v.to(dist_util.dev()) if torch.is_tensor(v) else v
        for k, v in model_kwargs["y"].items()
    }

    start_pose = torch.load(_START_POSE_PATH, map_location=dist_util.dev())

    _, _, _, sample_file_template, _, all_file_template = construct_template_variables(
        False
    )
    skeleton = paramUtil.t2m_kinematic_chain

    def _run_one(fix_body: bool, out_path: str) -> str:
        """Run one full generation pass and save results to out_path."""
        # Overwrite inpainting tensors for this pass (apply_leftarm_inpainting
        # always creates new tensors, so calling it twice is safe).
        apply_leftarm_inpainting(
            model_kwargs,
            start_pose=start_pose,
            n_prefix=_N_PREFIX,
            batch_size=args.batch_size,
            n_frames=n_frames,
            fix_body=fix_body,
        )

        if args.guidance_param != 1:
            model_kwargs["y"]["scale"] = (
                torch.ones(args.batch_size, device=dist_util.dev())
                * args.guidance_param
            )
        if "text" in model_kwargs["y"]:
            model_kwargs["y"]["text_embed"] = model.encode_text(
                model_kwargs["y"]["text"]
            )

        all_motions: list = []
        all_lengths: list = []
        all_text: list = []

        for rep_i in range(args.num_repetitions):
            print(f"### Sampling [repetitions #{rep_i}]")
            sample = sample_fn(
                model,
                motion_shape,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )

            if model.data_rep == "hml_vec":
                n_joints = 22 if sample.shape[1] == 263 else 21
                sample = data.dataset.t2m_dataset.inv_transform(
                    sample.cpu().permute(0, 2, 3, 1)
                ).float()
                sample = recover_from_ric(sample, n_joints)
                sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

            rot2xyz_pose_rep = (
                "xyz" if model.data_rep in ["xyz", "hml_vec"] else model.data_rep
            )
            rot2xyz_mask = (
                None
                if rot2xyz_pose_rep == "xyz"
                else model_kwargs["y"]["mask"].reshape(args.batch_size, n_frames).bool()
            )
            sample = model.rot2xyz(
                x=sample,
                mask=rot2xyz_mask,
                pose_rep=rot2xyz_pose_rep,
                glob=True,
                translation=True,
                jointstype="smpl",
                vertstrans=True,
                betas=None,
                beta=0,
                glob_rot=None,
                get_rotations_back=False,
            )

            text_key = "text" if "text" in model_kwargs["y"] else "action_text"
            all_text += model_kwargs["y"][text_key]
            all_motions.append(sample.cpu().numpy())
            all_lengths.append(model_kwargs["y"]["lengths"].cpu().numpy())
            print(f"created {len(all_motions) * args.batch_size} samples")

        all_motions_np = np.concatenate(all_motions, axis=0)[:total_num_samples]
        all_text = all_text[:total_num_samples]
        all_lengths_np = np.concatenate(all_lengths, axis=0)[:total_num_samples]

        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        os.makedirs(out_path)

        npy_path = os.path.join(out_path, "results.npy")
        np.save(
            npy_path,
            np.array(
                {
                    "motion": all_motions_np,
                    "text": all_text,
                    "lengths": all_lengths_np,
                    "num_samples": args.num_samples,
                    "num_repetitions": args.num_repetitions,
                },
                dtype=object,
            ),
        )
        with open(npy_path.replace(".npy", ".txt"), "w", encoding="utf-8") as fw:
            fw.write("\n".join(all_text))
        with open(npy_path.replace(".npy", "_len.txt"), "w", encoding="utf-8") as fw:
            fw.write("\n".join([str(l) for l in all_lengths_np]))

        print(f"saving visualizations to [{out_path}]...")
        animations = np.empty(
            shape=(args.num_samples, args.num_repetitions), dtype=object
        )
        max_length = int(max(all_lengths_np))

        for sample_i in range(args.num_samples):
            for rep_i in range(args.num_repetitions):
                caption = all_text[rep_i * args.batch_size + sample_i]
                length = int(all_lengths_np[rep_i * args.batch_size + sample_i])
                motion = all_motions_np[rep_i * args.batch_size + sample_i].transpose(
                    2, 0, 1
                )[:max_length]
                if motion.shape[0] > length:
                    motion[length:-1] = motion[length - 1]
                animation_save_path = os.path.join(
                    out_path, sample_file_template.format(sample_i, rep_i)
                )
                animations[sample_i, rep_i] = plot_3d_motion(
                    animation_save_path,
                    skeleton,
                    motion,
                    dataset=args.dataset,
                    title=caption,
                    fps=fps,
                    gt_frames=[],
                )

        save_multiple_samples(
            out_path,
            {"all": all_file_template},
            animations,
            fps,
            max(list(all_lengths_np) + [n_frames]),
        )

        abs_path = os.path.abspath(out_path)
        print(f"[Done] Results are at [{abs_path}]")
        return out_path

    # Dispatch
    if _BODY_MODE == "both":
        _run_one(fix_body=True, out_path=os.path.join(args.output_dir, "frozen"))
        _run_one(fix_body=False, out_path=os.path.join(args.output_dir, "free"))
        return args.output_dir
    if _BODY_MODE == "freeze":
        return _run_one(fix_body=True, out_path=args.output_dir)
    return _run_one(fix_body=False, out_path=args.output_dir)  # "free"


def _generate_leftarm(gen_args):
    """Generate samples without perturbing the training RNG.

    The generation path calls ``fixseed(args.seed)``, which would reset the
    global RNGs every save interval and make training replay the same data
    permutation and noise tape between checkpoints. Snapshotting and restoring
    the RNG state keeps the training RNG stream unperturbed by generation.
    """
    import random

    import numpy as np
    import torch

    states = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )
    try:
        return _generate_leftarm_impl(gen_args)
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])
        if states[3] is not None:
            torch.cuda.set_rng_state_all(states[3])


# ---------------------------------------------------------------------------
# Monkey-patch and launch training
# ---------------------------------------------------------------------------

import train.training_loop as _training_loop  # noqa: E402  # pylint: disable=wrong-import-order

_training_loop.generate = _generate_leftarm

# Resuming loads opt<step>.pt when it exists next to the resume checkpoint, and
# the restored param groups carry the ORIGINAL run's lr (1e-4 for the stock
# humanml_enc_512_50steps opt state), silently overriding --lr. Every fine-tune
# resumed from that checkpoint before 2026-08-09 — including customv3_fixed —
# therefore trained at 1e-4 regardless of its args.json.
_orig_load_optimizer_state = _training_loop.TrainLoop._load_optimizer_state


def _load_optimizer_state_keep_lr(self: Any) -> None:
    checkpoint = self.find_resume_checkpoint() or self.resume_checkpoint
    opt_path = Path(checkpoint).parent / f"opt{self.resume_step:09}.pt"
    if not opt_path.exists():
        # The original skips silently, e.g. resuming the deployed run, which kept no opt file.
        print(
            f"[train_leftarm] no {opt_path.name} beside the checkpoint; fresh optimizer"
        )
        return
    # The groups' lrs as built from --lr and any per-group multiplier (--clip_lr_scale);
    # resetting every group to --lr would erase it.
    lrs = [group["lr"] for group in self.opt.param_groups]
    try:
        _orig_load_optimizer_state(self)
    except ValueError as exc:
        # New conditioning heads (e.g. --speed_cond) or parameter groups
        # (--unfreeze_clip_layers) change the parameter count or grouping, so the
        # checkpoint's Adam state no longer matches; start it fresh.
        print(f"[train_leftarm] optimizer state not loaded ({exc}); starting fresh")
        return
    for group, lr in zip(self.opt.param_groups, lrs):
        group["lr"] = lr
    print(f"[train_leftarm] optimizer state loaded; lrs kept at {lrs}")


_training_loop.TrainLoop._load_optimizer_state = _load_optimizer_state_keep_lr


def _auto_increment_save_dir() -> None:
    """If --save_dir already exists (and --overwrite not set), bump to the next free _N suffix."""
    if "--overwrite" in sys.argv:
        return
    for i, arg in enumerate(sys.argv):
        if arg == "--save_dir" and i + 1 < len(sys.argv):
            base, key = sys.argv[i + 1], i + 1
        elif arg.startswith("--save_dir="):
            base, key = arg.split("=", 1)[1], None
        else:
            continue
        if not os.path.exists(base):
            return
        n = 2
        while os.path.exists(f"{base}_{n}"):
            n += 1
        candidate = f"{base}_{n}"
        print(f"[train_leftarm] '{base}' already exists — using '{candidate}'")
        if key is not None:
            sys.argv[key] = candidate
        else:
            sys.argv[i] = f"--save_dir={candidate}"
        return


def _source_letter(name: str) -> str:
    """The source of a ``dataset_replay_mix`` clip id: ``C`` (correction clip, ``C<id>_r<k>``),
    ``R`` (retimed copy) or ``H`` (HumanML3D). The humanml loader prefixes sub-caption
    segments with ``<letter>_``, which is stripped before the check.
    """
    clip = name[2:] if name[1:2] == "_" else name
    if clip[:1] not in ("C", "H", "R"):
        raise ValueError(
            f"{name!r} is not a dataset_replay_mix id (C... / R... / H...)"
        )
    return clip[:1]


def _is_correction(name: str) -> bool:
    """Whether a replay-mix id is a correction clip, the distilled source."""
    return _source_letter(name) == "C"


def _has_history_prefix(name: str) -> bool:
    """Whether a replay-mix id's clip starts with the arm's executed history (``C``/``R``)."""
    return _source_letter(name) in ("C", "R")


def _noised_input(
    diffusion: Any, x_start: Any, t: Any, noise: Any, y: dict[str, Any]
) -> Any:
    """The training input at step ``t``; the clean-prefix patch replaces this."""
    del y
    return diffusion.q_sample(x_start, t, noise=noise)


def _loss_mask(y: dict[str, Any], mask: Any) -> Any:
    """The frames a loss term covers; the clean-prefix patch replaces this."""
    del y
    return mask


def _key_batches() -> None:
    """Hand every batch its clip ids as ``y["db_key"]`` (MDM's t2m_collate forwards an item's
    eighth element), so the loss can tell the sources apart. Idempotent."""
    from data_loaders.humanml.data.dataset import (  # type: ignore[import-not-found]
        Text2MotionDatasetV2,
    )

    if getattr(Text2MotionDatasetV2, "_keyed", False):
        return
    getitem = Text2MotionDatasetV2.__getitem__

    def keyed_getitem(self: Any, item: int) -> tuple:
        return (*getitem(self, item), self.name_list[self.pointer + item])

    Text2MotionDatasetV2.__getitem__ = keyed_getitem
    Text2MotionDatasetV2._keyed = True


def _clean_prefix(n_prefix: int, noise_cap: int | None) -> None:
    """Patch MDM's loss and loader so clips with a history prefix train with it clean.

    For ``C``/``R`` clips the first ``n_prefix`` frames of ``x_t`` are the clip's own
    frames noised to step ``min(t, noise_cap)`` (clean when the cap is ``None``), those
    frames are left out of every loss term through the loss mask only (the model's
    attention mask is unchanged), and the loader's crop starts at frame 0. HumanML3D
    clips are untouched. Installed before the distillation patch, which reads
    :func:`_noised_input` and :func:`_loss_mask` so its teacher sees the same input
    and frames as the model. See :mod:`clean_prefix`.
    """
    import torch
    from data_loaders.humanml.data import dataset as dataset_module  # type: ignore
    from diffusion.gaussian_diffusion import (  # type: ignore[import-not-found]
        GaussianDiffusion,
    )

    global _noised_input, _loss_mask  # pylint: disable=global-statement
    _key_batches()
    training_losses = GaussianDiffusion.training_losses
    getitem = dataset_module.Text2MotionDatasetV2.__getitem__
    real_random = dataset_module.random
    crop_from_start = CropFromStart(real_random)

    def start_cropped_getitem(self: Any, item: int) -> tuple:
        prefixed = _has_history_prefix(self.name_list[self.pointer + item])
        dataset_module.random = crop_from_start if prefixed else real_random
        try:
            return getitem(self, item)
        finally:
            dataset_module.random = real_random

    def rows(y: dict[str, Any], device: Any) -> Any:
        return torch.tensor(
            [_has_history_prefix(k) for k in y["db_key"]], device=device
        )

    def noised_input(
        diffusion: Any, x_start: Any, t: Any, noise: Any, y: dict[str, Any]
    ) -> Any:
        q_sample = GaussianDiffusion.q_sample  # the class's, not a patched instance's
        x_t = q_sample(diffusion, x_start, t, noise=noise)
        prefix = noised_prefix(
            lambda x, tt, noise=None: q_sample(diffusion, x, tt, noise=noise),
            x_start,
            t,
            noise,
            noise_cap,
        )
        return splice_prefix(x_t, prefix, n_prefix, rows(y, x_start.device))

    def loss_mask(y: dict[str, Any], mask: Any) -> Any:
        return loss_mask_without_prefix(mask, n_prefix, rows(y, mask.device))

    def prefixed_losses(
        self: Any,
        model: Any,
        x_start: Any,
        t: Any,
        model_kwargs: Any = None,
        noise: Any = None,
        dataset: Any = None,
    ) -> Any:
        if noise is None:
            noise = torch.randn_like(x_start)
        y = model_kwargs["y"]
        x_t, mask = noised_input(self, x_start, t, noise, y), loss_mask(y, y["mask"])
        masked_l2 = self.masked_l2

        def spliced_q_sample(x: Any, tt: Any, noise: Any = None) -> Any:
            # MDM's training_losses draws its input exactly once, from these arguments.
            assert x is x_start and tt is t and noise is not None
            return x_t

        def prefixed_masked_l2(
            a: Any, b: Any, _mask: Any, *args: Any, **kw: Any
        ) -> Any:
            return masked_l2(a, b, mask, *args, **kw)

        self.q_sample, self.masked_l2 = spliced_q_sample, prefixed_masked_l2
        try:
            return training_losses(
                self, model, x_start, t, model_kwargs, noise=noise, dataset=dataset
            )
        finally:
            del self.q_sample
            self.masked_l2 = masked_l2

    dataset_module.Text2MotionDatasetV2.__getitem__ = start_cropped_getitem
    GaussianDiffusion.training_losses = prefixed_losses
    _noised_input, _loss_mask = noised_input, loss_mask
    print(
        f"[train_leftarm] clean prefix: the first {n_prefix} frames of C.../R... clips are "
        + ("clean" if noise_cap is None else f"noised to step min(t, {noise_cap})")
        + " in the training input, out of the loss, and cropped from frame 0"
    )


def _point_loader_at(root: Path, cache: Path | None = None) -> None:
    """Point MDM's humanml loader at ``root`` with a cache of its own.

    MDM resolves the humanml data root from a hardcoded string in ``get_opt`` and keys
    its motion cache (``dataset/t2m_<split>.npy``) on the split name alone, which is why
    ``finetune_standing.sh`` swaps ``dataset/HumanML3D`` and moves the caches aside.
    This rewrites the resolved root and gives the run its own ``cache`` directory
    (``<root>/mdm_cache`` by default; delete it after rebuilding the dataset), so
    training no longer touches the shared directory.
    """
    import numpy as np
    from data_loaders.humanml.data import dataset as dataset_module  # type: ignore

    live = Path("dataset/HumanML3D")
    for stat in ("Mean.npy", "Std.npy"):
        if not np.allclose(np.load(root / stat), np.load(live / stat)):
            raise ValueError(
                f"{root}/{stat} differs from {live}'s, which inference reads"
            )
    cache = (root / "mdm_cache" if cache is None else cache).resolve()
    (cache / "dataset").mkdir(parents=True, exist_ok=True)
    if not (cache / "glove").exists():
        (cache / "glove").symlink_to(Path("glove").resolve())
    get_opt, init = dataset_module.get_opt, dataset_module.HumanML3D.__init__

    def rooted_get_opt(opt_path: str, device: Any) -> Any:
        opt = get_opt(opt_path, device)
        opt.data_root = str(root)
        opt.motion_dir = str(root / "new_joint_vecs")
        opt.text_dir = str(root / "texts")
        return opt

    def cached_init(self: Any, *args: Any, **kwargs: Any) -> None:
        init(self, *args, **{**kwargs, "cache_path": str(cache)})

    dataset_module.get_opt = rooted_get_opt
    dataset_module.HumanML3D.__init__ = cached_init
    print(f"[train_leftarm] humanml loader rooted at {root} (cache {cache})")


def _record_args(extra: dict[str, Any]) -> None:
    """Write train_leftarm's own flags into the run's ``args.json`` beside MDM's, so a
    checkpoint says how it was trained (``mdm_api`` reads ``clean_prefix`` /
    ``prefix_noise`` from there)."""
    import json

    from train import train_mdm  # type: ignore[import-not-found]

    train_loop = train_mdm.TrainLoop

    def recording_train_loop(args: Any, *rest: Any) -> Any:
        path = Path(args.save_dir) / "args.json"
        recorded = json.loads(path.read_text(encoding="utf-8"))
        recorded.update(extra)
        path.write_text(
            json.dumps(recorded, indent=4, sort_keys=True), encoding="utf-8"
        )
        return train_loop(args, *rest)

    train_mdm.TrainLoop = recording_train_loop


def _mixed_loader(loader: Any, fraction: float) -> Any:
    """Re-draw a replay mix so ``fraction`` of every batch comes from the correction clips."""
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    t2m = loader.dataset.t2m_dataset
    names = t2m.name_list[t2m.pointer : t2m.pointer + len(loader.dataset)]
    is_corr = torch.tensor([_is_correction(n) for n in names])
    n_corr = int(is_corr.sum())
    weights = torch.where(
        is_corr, fraction / n_corr, (1.0 - fraction) / (len(names) - n_corr)
    ).tolist()
    print(
        f"[train_leftarm] mixed sampling: {n_corr} correction / {len(names) - n_corr} "
        f"HumanML3D entries, {fraction:.0%} of draws from corrections"
    )
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        sampler=WeightedRandomSampler(weights, len(names), replacement=True),
        num_workers=loader.num_workers,
        drop_last=True,
        collate_fn=loader.collate_fn,
    )


def _distill_toward_start(weight: float, dropout: bool = True) -> None:
    """Patch MDM's training loss so the correction clips of a replay mix (told apart by
    :func:`_is_correction` from the clip ids, which every batch now carries) also pay
    ``weight`` times the masked L2 to a frozen copy of the model as training starts — the
    resumed checkpoint, copied at the first step. The copy runs in eval mode on the same
    noised input and timestep, unconditioned where MDM's caption dropout dropped the clip's
    text and conditioned on it otherwise, so the term pulls the denoiser back toward the
    starting model's own predictions on the corrections while the HumanML3D clips are free to
    move. Without ``dropout`` the trained model's dropout is switched off at that first step,
    so both sides of the term are the same function.
    """
    import copy

    import torch
    from diffusion.gaussian_diffusion import (  # type: ignore[import-not-found]
        GaussianDiffusion,
    )
    from model.mdm import MDM  # type: ignore[import-not-found]

    _key_batches()
    mask_cond = MDM.mask_cond
    training_losses = GaussianDiffusion.training_losses
    frozen: dict[str, Any] = {}

    def recording_mask_cond(self: Any, cond: Any, force_mask: bool = False) -> Any:
        # MDM's own caption dropout, keeping which clips it dropped.
        if force_mask or not self.training or self.cond_mask_prob <= 0:
            return mask_cond(self, cond, force_mask)
        bs = cond.shape[-2]
        self.dropped_text = torch.bernoulli(
            torch.ones(bs, device=cond.device) * self.cond_mask_prob
        )
        return cond * (1.0 - self.dropped_text).view(1, bs, 1)

    def distilled_losses(
        self: Any,
        model: Any,
        x_start: Any,
        t: Any,
        model_kwargs: Any = None,
        noise: Any = None,
        dataset: Any = None,
    ) -> Any:
        # MDM's SpacedDiffusion hands in its timestep-remapping wrapper around the MDM;
        # the frozen copy is wrapped the same way.
        mdm = getattr(model, "model", model)
        if "start" not in frozen:
            start = copy.deepcopy(mdm)
            start.eval()  # MDM.train returns None, so no chaining
            start.requires_grad_(False)
            frozen["start"] = getattr(self, "_wrap_model", lambda m: m)(start)
            if not dropout:
                for module in mdm.modules():
                    if isinstance(module, torch.nn.Dropout):
                        module.p = 0.0
                    elif isinstance(module, torch.nn.MultiheadAttention):
                        module.dropout = 0.0
        if noise is None:
            noise = torch.randn_like(x_start)
        seen: dict[str, Any] = {}

        def recording_forward(x: Any, timesteps: Any, y: Any = None) -> Any:
            mdm.dropped_text = None
            seen["out"] = type(mdm).forward(mdm, x, timesteps, y)
            seen["dropped"] = mdm.dropped_text
            return seen["out"]

        mdm.forward = recording_forward
        try:
            terms = training_losses(
                self, model, x_start, t, model_kwargs, noise=noise, dataset=dataset
            )
        finally:
            del mdm.forward
        y = model_kwargs["y"]
        correction = torch.tensor(
            [_is_correction(key) for key in y["db_key"]], device=x_start.device
        )
        x_t = _noised_input(self, x_start, t, noise, y)
        timesteps = self._scale_timesteps(t)
        with torch.no_grad():
            start = frozen["start"]
            target = torch.where(
                seen["dropped"].bool().view(-1, 1, 1, 1),
                start(x_t, timesteps, y={**y, "uncond": True}),
                start(x_t, timesteps, y={**y}),
            )
        terms["distill"] = (
            self.masked_l2(target, seen["out"], _loss_mask(y, y["mask"])) * correction
        )
        terms["loss"] = terms["loss"] + weight * terms["distill"]
        return terms

    MDM.mask_cond = recording_mask_cond
    GaussianDiffusion.training_losses = distilled_losses
    print(
        f"[train_leftarm] correction clips (ids C...) also pay {weight} x the L2 to the "
        "starting checkpoint's predictions"
    )


def _clip_in_checkpoint(n_layers: int, lr_scale: float) -> None:
    """Patch MDM so CLIP text weights can train and travel with the checkpoint.

    With ``n_layers`` > 0 the top ``n_layers`` text blocks of CLIP, its ``ln_final`` and
    ``text_projection`` train with the rest of the model, in float32 (MDM keeps CLIP in
    half precision, which Adam cannot train), in their own AdamW group at ``lr_scale``
    times the learning rate; the PC's speed recipe of 2026-09-30 used 2 and 0.1. MDM's
    ``save`` drops every ``clip_model.`` key, so the tuned keys are written back into each
    checkpoint. A run resumed from a checkpoint that ships CLIP keys loads them in float32
    (MDM would round them into its half-precision copy) and keeps shipping them, frozen
    unless ``n_layers`` says otherwise, so a body trained against tuned text never
    travels without it. Without either, nothing changes but one read of the resumed
    checkpoint.
    """
    import torch
    from train import train_mdm  # type: ignore[import-not-found]

    create = train_mdm.create_model_and_diffusion
    adamw = _training_loop.AdamW
    save = _training_loop.TrainLoop.save
    shipped: set[str] = set()  # the clip_model. keys every checkpoint carries
    tuned: set[int] = set()  # ids of the CLIP parameters that train

    def shipped_keys(path: str | None) -> set[str]:
        if not path:
            return set()
        state = torch.load(path, map_location="cpu")
        return {k for k in state.get("model", state) if k.startswith("clip_model.")}

    def creating(args: Any, data: Any) -> Any:
        model, diffusion = create(args, data)
        shipped.update(shipped_keys(args.resume_checkpoint))
        clip = model.clip_model
        if shipped or n_layers:
            clip.float()
        if n_layers:
            for module in (*clip.transformer.resblocks[-n_layers:], clip.ln_final):
                module.requires_grad_(True)
            clip.text_projection.requires_grad_(True)
            tuned.update(id(p) for p in clip.parameters() if p.requires_grad)
            shipped.update(
                f"clip_model.{name}"
                for name, p in clip.named_parameters()
                if p.requires_grad
            )
        if shipped:
            print(
                f"[train_leftarm] {len(shipped)} CLIP weights ship with each checkpoint"
                + (
                    f", {len(tuned)} of them training at {lr_scale} x lr"
                    if tuned
                    else " (carried from the resumed checkpoint, frozen)"
                )
            )
        return model, diffusion

    def grouped_adamw(params: Any, lr: float, **kwargs: Any) -> Any:
        params = list(params)
        if lr_scale == 1.0 or not tuned:
            return adamw(params, lr=lr, **kwargs)
        return adamw(
            [
                {"params": [p for p in params if id(p) not in tuned]},
                {"params": [p for p in params if id(p) in tuned], "lr": lr * lr_scale},
            ],
            lr=lr,
            **kwargs,
        )

    def saving(self: Any) -> None:
        save(self)
        if not shipped:
            return
        path = os.path.join(self.save_dir, self.ckpt_file_name())
        checkpoint = torch.load(path, map_location="cpu")
        targets = (
            [
                (checkpoint[key], module)
                for key, module in (
                    ("model", self.model),
                    ("model_avg", self.model_avg),
                )
                if key in checkpoint
            ]
            if "model" in checkpoint
            else [(checkpoint, self.model)]
        )
        for target, module in targets:
            target.update(
                {n: v for n, v in module.state_dict().items() if n in shipped}
            )
        torch.save(checkpoint, path)

    train_mdm.create_model_and_diffusion = creating
    _training_loop.AdamW = grouped_adamw
    _training_loop.TrainLoop.save = saving


if __name__ == "__main__":
    import train.train_mdm  # noqa: E402

    if _our_args.distill_without_dropout and not _our_args.distill_weight:
        _parser.error("--distill_without_dropout needs --distill_weight")
    if _our_args.prefix_noise is not None and not _our_args.clean_prefix:
        _parser.error("--prefix_noise needs --clean_prefix")
    if _our_args.humanml_cache is not None and _our_args.humanml_root is None:
        _parser.error("--humanml_cache needs --humanml_root")
    if _our_args.clip_lr_scale != 1.0 and not _our_args.unfreeze_clip_layers:
        _parser.error("--clip_lr_scale needs --unfreeze_clip_layers")
    if _our_args.humanml_root is not None:
        _point_loader_at(_our_args.humanml_root.resolve(), _our_args.humanml_cache)
    # The clean-prefix patch goes first: the distillation patch wraps the loss it finds.
    if _our_args.clean_prefix:
        _clean_prefix(_our_args.clean_prefix, _our_args.prefix_noise)
    if _our_args.distill_weight:
        _distill_toward_start(
            _our_args.distill_weight, dropout=not _our_args.distill_without_dropout
        )
    _clip_in_checkpoint(_our_args.unfreeze_clip_layers, _our_args.clip_lr_scale)
    if _our_args.correction_fraction is not None:
        _get_loader = train.train_mdm.get_dataset_loader
        train.train_mdm.get_dataset_loader = lambda **kw: _mixed_loader(
            _get_loader(**kw), _our_args.correction_fraction
        )
    _auto_increment_save_dir()
    _record_args(
        {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(_our_args).items()
        }
    )
    train.train_mdm.main()
