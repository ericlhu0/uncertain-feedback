# This code is based on https://github.com/openai/guided-diffusion
"""Generate a large batch of image samples from a model and save them as a
large numpy array.

This can be used to produce samples for FID evaluation.
"""

import sys
from pathlib import Path

# Ensure local package imports work when this file is executed directly.
_SRC_ROOT = Path(__file__).resolve().parents[3]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from uncertain_feedback.consts import MDM_ROOT

# Treat the external MDM repo as source-only dependency, not a Python package.
# This allows running this script from anywhere in the project tree.
if str(MDM_ROOT / "motion-diffusion-model") not in sys.path:
    sys.path.insert(0, str(MDM_ROOT / "motion-diffusion-model"))

import os
import shutil

import data_loaders.humanml.utils.paramUtil as paramUtil
import numpy as np
import torch
from data_loaders import humanml_utils
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from data_loaders.tensors import collate
from sample.generate import construct_template_variables, save_multiple_samples
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils.sampler_util import ClassifierFreeSampleModel

from uncertain_feedback.motion_generators.mdm.mdm_parser_util import edit_args


def main():
    os.chdir(MDM_ROOT / "motion-diffusion-model")
    args = edit_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace("model", "").replace(".pt", "")
    max_frames = 196 if args.dataset in ["kit", "humanml"] else 60
    fps = 12.5 if args.dataset == "kit" else 20
    n_frames = min(max_frames, int(args.motion_length * fps))

    dist_util.setup_dist(args.device)
    if out_path == "":
        out_path = os.path.join(
            os.path.dirname(args.model_path),
            "edit_{}_{}_{}_seed{}".format(name, niter, args.edit_mode, args.seed),
        )
        if args.text_condition != "":
            out_path += "_" + args.text_condition.replace(" ", "_").replace(".", "")

    print("Loading dataset...")
    assert (
        args.num_samples <= args.batch_size
    ), f"Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})"
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = (
        args.num_samples
    )  # Sampling a single batch from the testset, with exactly args.num_samples
    # data = get_dataset_loader(name=args.dataset,
    #                           batch_size=args.batch_size,
    #                           num_frames=max_frames,
    #                           split='test',
    #                           hml_mode='train')  # in train mode, you get both text and motion.
    data = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=max_frames,
        split="test",
        hml_mode=(
            "train" if args.pred_len > 0 else "text_only"
        ),  # We need to sample a prefix from the dataset
        fixed_len=args.pred_len + args.context_len,
        pred_len=args.pred_len,
        device=dist_util.dev(),
    )
    data.fixed_length = n_frames
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    load_saved_model(model, args.model_path, use_avg=args.use_ema)

    model = ClassifierFreeSampleModel(
        model
    )  # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    # iterator = iter(data)
    # input_motions, model_kwargs = next(iterator)
    # input_motions = input_motions.to(dist_util.dev())
    # texts = [args.text_condition] * args.num_samples
    # model_kwargs['y']['text'] = texts
    # if args.text_condition == '':
    #     args.guidance_param = 0.  # Force unconditioned generation

    ### from is_using_data = False branch of generate.py
    ### adapted to expect args.text_condition
    collate_args = [
        {"inp": torch.zeros(n_frames), "tokens": None, "lengths": n_frames}
    ] * args.num_samples
    texts = [args.text_condition] * args.num_samples
    collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
    _, model_kwargs = collate(collate_args)
    ###

    print(model_kwargs["y"].keys())
    for k, v in model_kwargs["y"].items():
        print(k)
        print(v)
        try:
            print(v.shape)
        except:
            print("not array")

    # add custom inpaint (probably zeros for now)
    # assert max_frames == input_motions.shape[-1]
    # gt_frames_per_sample = {}
    # model_kwargs['y']['inpainted_motion'] = input_motions

    input_motions = torch.zeros(args.num_samples, 263, 1, n_frames, device="cuda")
    # TODO: start the guy in a sitting position
    # TODO: generate the starting position by calling the model!!
    sitting_pose_path = MDM_ROOT / "demo_pose.pt"
    sitting_pose = torch.load(sitting_pose_path, map_location="cuda")  # (263, 1)
    input_motions = (
        sitting_pose.unsqueeze(0).unsqueeze(-1).repeat(args.num_samples, 1, 1, n_frames)
    )
    model_kwargs["y"]["inpainted_motion"] = input_motions
    gt_frames_per_sample = {}

    #### NEW STUFF !!

    HML_L_ARM_JOINTS = [
        humanml_utils.HML_JOINT_NAMES.index(name)
        for name in ["left_shoulder", "left_elbow", "left_wrist"]
    ]
    HML_L_ARM_JOINTS_BINARY = np.array(
        [i not in HML_L_ARM_JOINTS for i in range(humanml_utils.NUM_HML_JOINTS)]
    )

    NOT_L_ARM_MASK = np.concatenate(
        (
            [True] * (1 + 2 + 1),
            HML_L_ARM_JOINTS_BINARY[1:].repeat(3),
            HML_L_ARM_JOINTS_BINARY[1:].repeat(6),
            HML_L_ARM_JOINTS_BINARY.repeat(3),
            [True] * 4,
        )
    )
    NOT_L_ARM_MASK = NOT_L_ARM_MASK | humanml_utils.HML_ROOT_MASK

    # Inpainting strategy:
    #   - Frame 0: fix ALL features to the demo pose (seeds the initial state).
    #   - All other frames: only arm features are free; body+root are inpainted.
    #   Set FIX_BODY_ALL_FRAMES = False to let the body move freely after frame 0.
    FIX_BODY_ALL_FRAMES = False

    if FIX_BODY_ALL_FRAMES:
        body_mask = torch.tensor(
            NOT_L_ARM_MASK, dtype=torch.bool, device=input_motions.device
        )
        model_kwargs["y"]["inpainting_mask"] = (
            body_mask
            .unsqueeze(0)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .repeat(
                input_motions.shape[0],
                1,
                input_motions.shape[2],
                input_motions.shape[3],
            )
        )
    else:
        # All frames free — model generates everything.
        model_kwargs["y"]["inpainting_mask"] = torch.zeros(
            input_motions.shape[0], 263, 1, n_frames,
            dtype=torch.bool, device=input_motions.device,
        )

    # Pin the first N_PREFIX frames to the demo pose.
    # With only 1 frame the model immediately diverges to its prior (standing).
    # A longer prefix forces the model to stay near the initial pose before transitioning.
    N_PREFIX = 10
    model_kwargs["y"]["inpainting_mask"][..., :N_PREFIX] = True

    ####

    all_motions = []
    all_lengths = []
    all_text = []

    samples = []

    for rep_i in range(args.num_repetitions):
        print(f"### Start sampling [repetitions #{rep_i}]")

        # add CFG scale to batch
        model_kwargs["y"]["scale"] = (
            torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
        )

        sample_fn = diffusion.p_sample_loop

        sample = sample_fn(
            model,
            (args.batch_size, model.njoints, model.nfeats, n_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        samples.append(sample)

        # Recover XYZ *positions* from HumanML3D vector representation.
        # For hml_vec, recover_from_ric already returns world-space xyz, so
        # rot2xyz is skipped — it would apply root translation a second time
        # (vertstrans=True) and produce drifting/exploding joint positions.
        if model.data_rep == "hml_vec":
            n_joints = 22 if sample.shape[1] == 263 else 21
            sample = data.dataset.t2m_dataset.inv_transform(
                sample.cpu().permute(0, 2, 3, 1)
            ).float()
            sample = recover_from_ric(sample, n_joints)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)
        else:
            rot2xyz_pose_rep = model.data_rep
            rot2xyz_mask = model_kwargs["y"]["mask"].reshape(
                args.batch_size, n_frames
            ).bool()
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

        all_text += model_kwargs["y"]["text"]
        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs["y"]["lengths"].cpu().numpy())

        print(f"created {len(all_motions) * args.batch_size} samples")

    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, "results.npy")
    print(samples)
    print(torch.cat(samples).shape)
    print(f"saving results file to [{npy_path}]")
    np.save(
        npy_path,
        {
            "motion": all_motions,
            "text": all_text,
            "lengths": all_lengths,
            "samples": (torch.cat(samples)).cpu().numpy(),
            "num_samples": args.num_samples,
            "num_repetitions": args.num_repetitions,
        },
    )
    with open(npy_path.replace(".npy", ".txt"), "w") as fw:
        fw.write("\n".join(all_text))
    with open(npy_path.replace(".npy", "_len.txt"), "w") as fw:
        fw.write("\n".join([str(l) for l in all_lengths]))

    if args.no_video:
        exit()

    print(f"saving visualizations to [{out_path}]...")
    skeleton = (
        paramUtil.kit_kinematic_chain
        if args.dataset == "kit"
        else paramUtil.t2m_kinematic_chain
    )

    # Recover XYZ *positions* from HumanML3D vector representation
    if model.data_rep == "hml_vec":
        input_motions = data.dataset.t2m_dataset.inv_transform(
            input_motions.cpu().permute(0, 2, 3, 1)
        ).float()
        input_motions = recover_from_ric(input_motions, n_joints)
        input_motions = (
            input_motions.view(-1, *input_motions.shape[2:])
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )

    (
        sample_print_template,
        row_print_template,
        all_print_template,
        sample_file_template,
        row_file_template,
        all_file_template,
    ) = construct_template_variables(args.unconstrained)
    max_vis_samples = 6
    num_vis_samples = min(args.num_samples, max_vis_samples)
    animations = np.empty(shape=(args.num_samples, args.num_repetitions), dtype=object)
    max_length = max(all_lengths)

    for sample_i in range(args.num_samples):
        caption = "Input Motion"
        length = model_kwargs["y"]["lengths"][sample_i]
        motion = input_motions[sample_i].transpose(2, 0, 1)[:length]
        save_file = "input_motion{:02d}.mp4".format(sample_i)
        animation_save_path = os.path.join(out_path, save_file)
        rep_files = [animation_save_path]
        # FIXME - fix and bring back the following:
        print(f'[({sample_i}) "{caption}" | -> {save_file}]')
        input_clip = plot_3d_motion(
            animation_save_path,
            skeleton,
            motion,
            title=caption,
            dataset=args.dataset,
            fps=fps,
            vis_mode="gt",
            gt_frames=gt_frames_per_sample.get(sample_i, []),
        )
        input_clip = input_clip.set_duration(float(length) / fps)
        input_clip.write_videofile(
            animation_save_path, fps=fps, codec="libx264", audio=False, logger=None
        )
        for rep_i in range(args.num_repetitions):
            caption = all_text[rep_i * args.batch_size + sample_i]
            if caption == "":
                caption = "Edit [{}] unconditioned".format(args.edit_mode)
            else:
                caption = "Edit [{}]: {}".format(args.edit_mode, caption)
            length = all_lengths[rep_i * args.batch_size + sample_i]
            motion = all_motions[rep_i * args.batch_size + sample_i].transpose(2, 0, 1)[
                :length
            ]
            save_file = "sample{:02d}_rep{:02d}.mp4".format(sample_i, rep_i)
            animation_save_path = os.path.join(out_path, save_file)
            rep_files.append(animation_save_path)
            gt_frames = gt_frames_per_sample.get(sample_i, [])
            print(f'[({sample_i}) "{caption}" | Rep #{rep_i} | -> {save_file}]')
            animations[sample_i, rep_i] = plot_3d_motion(
                animation_save_path,
                skeleton,
                motion,
                dataset=args.dataset,
                title=caption,
                fps=fps,
                gt_frames=gt_frames,
            )
            animations[sample_i, rep_i] = animations[sample_i, rep_i].set_duration(
                float(length) / fps
            )
            animations[sample_i, rep_i].write_videofile(
                animation_save_path, fps=fps, codec="libx264", audio=False, logger=None
            )
            # Credit for visualization: https://github.com/EricGuo5513/text-to-motion

        all_rep_save_file = os.path.join(out_path, "sample{:02d}.mp4".format(sample_i))
        ffmpeg_rep_files = [f" -i {f} " for f in rep_files]
        hstack_args = f" -filter_complex hstack=inputs={args.num_repetitions+1}"
        ffmpeg_rep_cmd = (
            f"ffmpeg -y -loglevel warning "
            + "".join(ffmpeg_rep_files)
            + f"{hstack_args} {all_rep_save_file}"
        )
        os.system(ffmpeg_rep_cmd)
        print(f'[({sample_i}) "{caption}" | all repetitions | -> {all_rep_save_file}]')

    save_multiple_samples(
        out_path,
        {"all": all_file_template},
        animations,
        fps,
        max(list(all_lengths) + [n_frames]),
    )

    abs_path = os.path.abspath(out_path)
    print(f"[Done] Results are at [{abs_path}]")


if __name__ == "__main__":
    main()
