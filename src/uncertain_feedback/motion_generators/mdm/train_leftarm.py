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

Extra flags (``--start_pose``, ``--n_prefix``, ``--body_mode``) are consumed
here and stripped from sys.argv before forwarding to ``train.train_mdm``.
Omitting ``--gen_during_training`` means the patched path is never reached —
training is identical to plain train_mdm.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — mirrors sample_leftarm.py
# ---------------------------------------------------------------------------

_SRC_ROOT = Path(__file__).resolve().parents[3]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from uncertain_feedback.consts import MDM_ROOT  # noqa: E402
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
    try:
        _orig_load_optimizer_state(self)
    except ValueError as exc:
        # New conditioning heads (e.g. --speed_cond) change the parameter count,
        # so the checkpoint's Adam state no longer matches; start it fresh.
        print(f"[train_leftarm] optimizer state not loaded ({exc}); starting fresh")
        return
    for group in self.opt.param_groups:
        group["lr"] = self.lr
    print(f"[train_leftarm] optimizer state loaded; lr reset to {self.lr:g}")


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


if __name__ == "__main__":
    import train.train_mdm  # noqa: E402

    _auto_increment_save_dir()
    train.train_mdm.main()
