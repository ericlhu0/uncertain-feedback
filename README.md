# Confidence-Aware Language Grounding

## Getting Started
Clone https://github.com/GuyTevet/motion-diffusion-model as `src/uncertain_feedback/motion_generators/mdm/motion-diffusion-model` and download the [required weights](https://github.com/GuyTevet/motion-diffusion-model?tab=readme-ov-file#mdm-is-now-40x-faster--04-secsample), [data](https://github.com/GuyTevet/motion-diffusion-model?tab=readme-ov-file#2-get-data) and [SMPL model](https://github.com/GuyTevet/motion-diffusion-model/blob/main/prepare/download_smpl_files.sh)

## Running (Custom) Motion Generation
```
uv run python src/uncertain_feedback/motion_generators/mdm/sample_leftarm.py \
--model_path save/humanml_enc_512_50steps/model000750000.pt \
--text_condition "a person barely raises their left hand." \
--num_samples 1 \
--num_repetitions 1 \
--motion_length 5.0 \
```

## Get HML263 from sequence of images of human
On first install, build detectron2 for your GPU architecture (replace `8.9` with your GPU's compute capability, e.g. `8.0` for A100):
```
TORCH_CUDA_ARCH_LIST="8.9" uv sync --reinstall-package detectron2
```

Step 1 — inference (produces data/demo/smpl_out.npz)
``` 
uv run python src/uncertain_feedback/data_collection/_mhr_inference_worker.py \
--image_folder src/uncertain_feedback/data_collection/data/demo/images \
--output_path src/uncertain_feedback/data_collection/data/demo/smpl_out.npz
```

Step 2 — HML263 conversion + visualization (produces demo/comparison.png)
```
uv run python src/uncertain_feedback/data_collection/data/demo/run_demo.py
```

## Label data and make dataset
0. If manually generating synthetic trajectories, use this to produce a formatted HumanML3D dataset
```
uv run python -m uncertain_feedback.data_collection.trajectory_editor.server \
--hml_stats_dir src/uncertain_feedback/motion_generators/mdm/motion-diffusion-model/dataset/HumanML3D
```

1. Turn videos into images
```
uv run python src/uncertain_feedback/data_collection/extract_all_frames.py \
--videos_dir src/uncertain_feedback/data_collection/data/demo/videos/ \
--frames_dir src/uncertain_feedback/data_collection/data/demo/video_frames/
```

2. Label segments with text descriptions in browser
```
uv run python src/uncertain_feedback/data_collection/labeler.py \
--frames_dir src/uncertain_feedback/data_collection/data/demo/video_frames/
```
                                                                                       
3. Build MDM dataset
```
uv run python src/uncertain_feedback/data_collection/build_mdm_dataset.py \
--frames_dir src/uncertain_feedback/data_collection/data/demo/video_frames/ \
--labels_json src/uncertain_feedback/data_collection/data/demo/video_frames/labels.json \
--output_dir src/uncertain_feedback/motion_generators/mdm/motion-diffusion-model/dataset/HumanML3Dnew \
--fix_body \
--n_augment 49 \
--noise_std 0.05
```

4. Fine-tune motion-diffusion-model
First rename the original `src/uncertain_feedback/motion_generators/mdm/motion-diffusion-model/dataset/HumanML3D` to something else, and rename your new generated `.../HumanML3Dnew` (or whatever was created with the trajectory editor web ui) to `.../HumanML3D`

**Always clear the dataset cache before retraining** (stale cache silently uses the old dataset):
```
rm -f src/uncertain_feedback/motion_generators/mdm/motion-diffusion-model/dataset/t2m_train.npy \
src/uncertain_feedback/motion_generators/mdm/motion-diffusion-model/dataset/t2m_val.npy \
src/uncertain_feedback/motion_generators/mdm/motion-diffusion-model/dataset/t2m_test.npy
```

Then run from the MDM submodule directory:
```
cd src/uncertain_feedback/motion_generators/mdm/motion-diffusion-model/

uv run python -m train.train_mdm \
    --save_dir ./save/my_finetuned_v1 \       # output dir; must not already exist (add --overwrite to reuse)
    --dataset humanml \
    --resume_checkpoint ./save/humanml_enc_512_50steps/model000750000.pt \
    --diffusion_steps 50 \                    # must match the checkpoint (50 for humanml_enc_512_50steps)
    --mask_frames \                           # mask non-target frames during training
    --use_ema \                               # EMA model averaging (recommended)
    --batch_size 1 \                          # use 1 for small datasets; increase if GPU memory allows
    --num_steps 1000 \                        # total gradient steps; 500–2000 is typical for fine-tuning
    --save_interval 100 \                     # checkpoint every N steps; ~10% of num_steps gives ~10 checkpoints
    --lr 1e-4 \                               # learning rate; try 5e-5 for more conservative fine-tuning
    --gen_during_training                     # generate sample motions at each save_interval for sanity checks
```

To generate sanity-check samples from a fixed starting pose, use `train_leftarm.py` instead. It accepts the same flags plus three extras:
```
uv run python ../train_leftarm.py \
    --save_dir ./save/my_finetuned_v1 \
    --start_pose demo_pose.pt \
    --n_prefix 1 \
    --body_mode both \
    --dataset humanml \
    --resume_checkpoint ./save/humanml_enc_512_50steps/model000750000.pt \
    --diffusion_steps 50 \
    --mask_frames \
    --use_ema \
    --batch_size 8 \
    --num_steps 1000 \
    --save_interval 100 \
    --lr 1e-4 \
    --gen_during_training \
    --gen_num_samples 3 \
    --gen_num_repetitions 3
```
`--body_mode` controls body freezing in the generated samples:
- `freeze` (default): non-left-arm body features fixed every frame; produces `num_samples × num_reps` trajectorie0s
- `free`: only prefix frames constrained, body moves freely; produces `num_samples × num_reps` trajectories
- `both`: runs both modes, saves to `frozen/` and `free/` subdirs; produces `num_samples × 2×num_reps` trajectories total

Key hyperparameter guidance:
- `--num_steps`: 500–2000 for small datasets; watch the loss curve for overfitting
- `--lr`: default 1e-4; lower to 5e-5 for more stable fine-tuning on small datasets
- `--save_interval`: set to ~10% of `--num_steps` to get ~10 checkpoints to pick from
- `--diffusion_steps`: must match the pre-trained checkpoint (50 for `humanml_enc_512_50steps`)
- `--overwrite`: add this flag to continue training into an already-existing `--save_dir`

5. Run motion generation with the new model

From the same directory as training (`motion-diffusion-model/`):
```
# still inside motion-diffusion-model/
uv run python ../sample_leftarm.py \
    --model_path save/my_finetuned_v1/model000001000.pt \
    --text_condition "raise my left arm" \
    --num_samples 3 \
    --num_repetitions 5 \
    --motion_length 5.0
```
`--model_path` is always relative to `motion-diffusion-model/` regardless of cwd (the script does an internal `os.chdir`). Output videos are saved under `save/my_finetuned_v1/edit_*/`. (1s = 20 frames)


## Running Experiments

Three planner variants are available in `src/uncertain_feedback/planners/mpc/`. All commands run from the **repo root**. The `--start_pose` argument is a filename resolved against `MDM_ROOT` (`src/uncertain_feedback/motion_generators/mdm/`) internally — it does not depend on cwd.

### Base MPC (baseline, no MDM)
```
uv run python -m uncertain_feedback.planners.mpc.arm_mpc \
  --steps 500 \
  --samples 256 \
  --horizon 10
```
No model weights required. Moves the left arm to hardcoded goals with live matplotlib visualization.

### MDM + MPC (no uncertainty quantification)
```
uv run python -m uncertain_feedback.planners.mpc.arm_mpc_mdm \
  --text "raise my left arm" \
  --text_time 0 \
  --steps 750 \
  --start_pose sitting_pose.pt \
  --save arm_mdm.mp4
```
Key args: `--text` (motion description), `--text_time` (MPC step at which MDM is triggered, default 0), `--start_pose` (`.pt` filename in `MDM_ROOT`), `--save` (output mp4/gif), `--save_motion` (save the raw MDM motion video separately).

### MDM + MPC + Uncertainty Quantification (main experiment)
```
uv run python -m uncertain_feedback.planners.mpc.arm_mpc_mdm_uq \
  --text "raise my left arm" \
  --text_time 0 \
  --steps 750 \
  --diffusion-samples 128 \
  --n-clusters 3 \
  --start_pose sitting_pose.pt \
  --save arm_uq.mp4
```
Generates `--diffusion-samples` MDM trajectories, clusters them into `--n-clusters` groups via KMeans, opens an interactive matplotlib cluster-picker window (blocks until you click a cluster panel), then tracks the mean of the chosen cluster with MPC.

Additional args vs. the plain MDM+MPC variant:
- `--diffusion-samples`: number of MDM samples to draw (default 128; more = better coverage, slower)
- `--n-clusters`: number of trajectory clusters shown in the picker (default 3)
- `--trajectory-fraction`: fraction of MDM frames to enqueue as MPC waypoints (default 0.75)

## Thanks
This repository is based on [python-starter](https://github.com/tomsilver/python-starter), which is a general starter repository (not limited to research project code).