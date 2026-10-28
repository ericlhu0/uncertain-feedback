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
from pathlib import Path
from typing import Any

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
    "--uncond_tag",
    default=None,
    help=(
        "Captions starting with this tag (the replay clips of a tagged mix) are never "
        "dropped for classifier-free guidance, so the unconditional model is learned "
        "from the untagged (correction) clips alone. Default: MDM's dropout for every clip."
    ),
)
_parser.add_argument(
    "--tag_null",
    action="store_true",
    help=(
        "With --uncond_tag: a tagged caption is replaced by the bare tag at MDM's dropout "
        "rate instead of never being dropped, so the tag alone is the replay clips' own "
        "unconditional prompt (sample them with run_prior.py --null-prompt) and the empty "
        "condition stays the correction clips'."
    ),
)
_parser.add_argument(
    "--strip_tag",
    action="store_true",
    help=(
        "With --uncond_tag: the tag is removed from a kept tagged caption before it is "
        "encoded, so it only marks the clip's source for the dropout rule and the model "
        "sees the replay captions as written."
    ),
)
_parser.add_argument(
    "--distill_weight",
    type=float,
    default=0.0,
    help=(
        "With --uncond_tag: add this weight times the masked L2 between the model's output "
        "and a frozen copy of the checkpoint training starts from, on the untagged "
        "(correction) clips only, the copy given the same text and the same guidance "
        "dropout. Default 0: no distillation."
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


def _mixed_loader(loader: Any, fraction: float) -> Any:
    """Re-draw a replay mix so ``fraction`` of every batch comes from the correction clips.

    Ids follow ``dataset_replay_mix``: ``C<id>_r<k>`` corrections, ``H<id>`` HumanML3D; the
    humanml loader prefixes sub-caption segments with ``<letter>_``, which is stripped
    before the check.
    """
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    t2m = loader.dataset.t2m_dataset
    names = t2m.name_list[t2m.pointer : t2m.pointer + len(loader.dataset)]
    is_corr = torch.tensor(
        [(n[2:] if n[1:2] == "_" else n).startswith("C") for n in names]
    )
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


def _keep_tagged_conditions(
    tag: str, tag_null: bool = False, strip_tag: bool = False
) -> None:
    """Patch MDM so training never drops the text of captions starting with ``tag``; with
    ``tag_null`` such a caption is instead replaced by the bare tag at MDM's dropout rate,
    and with ``strip_tag`` a kept one is encoded without the tag.
    """
    import torch
    from model.mdm import MDM  # type: ignore[import-not-found]

    forward, mask_cond = MDM.forward, MDM.mask_cond

    def tagged_forward(self: Any, x: Any, timesteps: Any, y: Any = None) -> Any:
        if self.training and y is not None and "text" in y:
            tagged = [text.startswith(tag) for text in y["text"]]
            if tag_null:
                null = (torch.rand(len(tagged)) < self.cond_mask_prob).tolist()
                y = {
                    **y,
                    "text": [
                        tag.strip() if is_tagged and drop else text
                        for text, is_tagged, drop in zip(y["text"], tagged, null)
                    ],
                }
            if strip_tag:
                y = {
                    **y,
                    "text": [
                        text[len(tag) :] if text.startswith(tag) else text
                        for text in y["text"]
                    ],
                }
            self.keep_cond = torch.tensor(tagged, device=x.device)
        try:
            return forward(self, x, timesteps, y)
        finally:
            self.keep_cond = None

    def tagged_mask_cond(self: Any, cond: Any, force_mask: bool = False) -> Any:
        keep = getattr(self, "keep_cond", None)
        if keep is None or force_mask or not self.training or self.cond_mask_prob <= 0:
            return mask_cond(self, cond, force_mask)
        drop = torch.bernoulli(
            torch.full(keep.shape, self.cond_mask_prob, device=cond.device)
        )
        self.dropped_text = drop * (~keep).float()  # read by _distill_toward_start
        return cond * (1.0 - self.dropped_text).view(1, -1, 1)

    MDM.forward, MDM.mask_cond = tagged_forward, tagged_mask_cond
    print(
        f"[train_leftarm] text never dropped for captions starting with {tag!r}"
        + (f"; replaced by {tag.strip()!r} at the dropout rate" if tag_null else "")
        + ("; the tag is stripped before encoding" if strip_tag else "")
    )


def _distill_toward_start(weight: float, tag: str) -> None:
    """Patch MDM's training loss so the clips whose caption does not start with ``tag`` (the
    corrections of a tagged mix) also pay ``weight`` times the masked L2 to a frozen copy of
    the model as training starts — the resumed checkpoint, copied at the first step. The copy
    runs in eval mode on the same noised input and timestep, unconditioned where
    :func:`_keep_tagged_conditions` dropped the clip's text and conditioned on it otherwise,
    so the term pulls the denoiser back toward the starting model's own predictions on the
    corrections while the tagged clips are free to move.
    """
    import copy

    import torch
    from diffusion.gaussian_diffusion import (  # type: ignore[import-not-found]
        GaussianDiffusion,
    )

    training_losses = GaussianDiffusion.training_losses
    frozen: dict[str, Any] = {}

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
            [not text.startswith(tag) for text in y["text"]], device=x_start.device
        )
        x_t = self.q_sample(x_start, t, noise=noise)
        timesteps = self._scale_timesteps(t)
        with torch.no_grad():
            start = frozen["start"]
            target = torch.where(
                seen["dropped"].bool().view(-1, 1, 1, 1),
                start(x_t, timesteps, y={**y, "uncond": True}),
                start(x_t, timesteps, y={**y}),
            )
        terms["distill"] = self.masked_l2(target, seen["out"], y["mask"]) * correction
        terms["loss"] = terms["loss"] + weight * terms["distill"]
        return terms

    GaussianDiffusion.training_losses = distilled_losses
    print(
        f"[train_leftarm] captions not starting with {tag!r} also pay {weight} x the L2 "
        "to the starting checkpoint's predictions"
    )


if __name__ == "__main__":
    import train.train_mdm  # noqa: E402

    if (
        _our_args.tag_null or _our_args.strip_tag or _our_args.distill_weight
    ) and _our_args.uncond_tag is None:
        _parser.error("--tag_null, --strip_tag and --distill_weight need --uncond_tag")
    if _our_args.uncond_tag is not None:
        _keep_tagged_conditions(
            _our_args.uncond_tag, _our_args.tag_null, _our_args.strip_tag
        )
    if _our_args.distill_weight:
        _distill_toward_start(_our_args.distill_weight, _our_args.uncond_tag)
    if _our_args.correction_fraction is not None:
        _get_loader = train.train_mdm.get_dataset_loader
        train.train_mdm.get_dataset_loader = lambda **kw: _mixed_loader(
            _get_loader(**kw), _our_args.correction_fraction
        )
    _auto_increment_save_dir()
    train.train_mdm.main()
