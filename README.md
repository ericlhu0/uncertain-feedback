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
    --save_dir ./save/my_finetuned \
    --start_pose demo_pose.pt \
    --n_prefix 1 \
    --body_mode both \
    --dataset humanml \
    --resume_checkpoint ./save/humanml_enc_512_50steps/model000750000.pt \
    --diffusion_steps 100 \
    --mask_frames \
    --use_ema \
    --batch_size 8 \
    --num_steps 700 \
    --save_interval 100 \
    --lr 3e-5 \
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


## Running MPC Experiments

Use the unified runner from the repo root:
```
uv run python -m uncertain_feedback.planners.run --mpc-config path/to/mpc.yaml
```

Controller settings now live in the required YAML file passed with `--mpc-config`.
The initial whole-body HML pose can be set with `pose:` in the YAML. Runtime
inputs still stay on the command line: `--model-path`, `--arm`, `--text`,
`--text-time`, `--save`, `--live`, `--mdm-frames`, and `--frozen-body`.
`--pose` is still accepted as an override for the YAML pose.

Supported YAML `planner` values:
- `arm_mpc`: joint-space MPC only, no MDM.
- `arm_mpc_mdm`: one MDM trajectory, then MPC tracks it.
- `arm_mpc_mdm_uq`: multiple MDM samples, clustering/picker, then MPC tracks the selected mean.
- `arm_mpc_cartesian`: MDM/UQ first, then Cartesian wrist-goal MPC.
- `arm_mpc_cartesian_no_mdm`: Cartesian wrist-goal MPC only, no MDM and no UQ.

### Minimal Joint-Space MPC Config
Save as `src/uncertain_feedback/planners/mpc/configs/mpc_plain.yaml`:
```yaml
planner: arm_mpc
steps: 500
horizon: 10
n_mpc_samples: 256
max_angle_delta: 0.001
goal_threshold: 0.01
```

Run:
```bash
uv run python -m uncertain_feedback.planners.run \
  --mpc-config src/uncertain_feedback/planners/mpc/configs/mpc_plain.yaml \
  --live
```

### MDM + UQ Config
Save as `src/uncertain_feedback/planners/mpc/configs/mpc_mdm_uq.yaml`:
```yaml
planner: arm_mpc_mdm_uq
steps: 750
horizon: 10
n_mpc_samples: 512
max_angle_delta: 0.0025
pose: "src/uncertain_feedback/motion_generators/mdm/demo_pose.pt"
goal_threshold: 0.1
advance_threshold: 0.1
trajectory_fraction: 1.0

uq:
  diffusion_samples: 128
  n_clusters: 3
  auto_cluster: null
```

Run with an interactive cluster picker:
```bash
uv run python -m uncertain_feedback.planners.run \
  --mpc-config src/uncertain_feedback/planners/mpc/configs/mpc_mdm_uq.yaml \
  --model-path "src/uncertain_feedback/motion_generators/mdm/motion-diffusion-model/save/my_finetuned_final/model000750500.pt" \
  --text "raise my left arm" \
  --mdm-frames 100 \
  --save out.mp4 \
  --live
```

For headless runs, set `uq.auto_cluster` in the YAML:
```yaml
uq:
  diffusion_samples: 128
  n_clusters: 3
  auto_cluster: 0
```

### Cartesian MPC With MDM/UQ
This planner first follows the selected/generated MDM arm trajectory, then switches
to Cartesian wrist goals.

Save as `src/uncertain_feedback/planners/mpc/configs/arm_mpc_cartesian_mdm.yaml`:
```yaml
planner: arm_mpc_cartesian
steps: 750
horizon: 10
n_mpc_samples: 512
max_angle_delta: 0.0025
pose: "src/uncertain_feedback/motion_generators/mdm/demo_pose.pt"
goal_threshold: 0.1
advance_threshold: 0.1
trajectory_fraction: 1.0

uq:
  diffusion_samples: 200
  n_clusters: 5
  auto_cluster: null

cartesian:
  goals:
    - [0.3, 0.5, 0.0]
  threshold: 0.05
```

Run:
```bash
uv run python -m uncertain_feedback.planners.run \
  --mpc-config src/uncertain_feedback/planners/mpc/configs/arm_mpc_cartesian_mdm.yaml \
  --model-path "src/uncertain_feedback/motion_generators/mdm/motion-diffusion-model/save/my_finetuned_final/model000750500.pt" \
  --text "raise my left arm" \
  --mdm-frames 100 \
  --save out.mp4 \
  --live
```

### Cartesian MPC Without MDM or UQ
Use `arm_mpc_cartesian_no_mdm` when you want direct Cartesian wrist-goal MPC only.
This path does not generate motion or run clustering. If you set an HML `pose:`,
the runner decodes that pose once to initialize the arm, collar, spine, and
background body.

Save as `src/uncertain_feedback/planners/mpc/configs/arm_mpc_cartesian_no_mdm.yaml`:
```yaml
planner: arm_mpc_cartesian_no_mdm
steps: 750
horizon: 10
n_mpc_samples: 512
max_angle_delta: 0.0025
pose: "src/uncertain_feedback/motion_generators/mdm/demo_pose.pt"
goal_threshold: 0.1

cartesian:
  goals:
    - [0.3, 0.5, 0.1]
  threshold: 0.05
```

Run:
```bash
uv run python -m uncertain_feedback.planners.run \
  --mpc-config src/uncertain_feedback/planners/mpc/configs/arm_mpc_cartesian_no_mdm.yaml \
  --live
```

### Optional Elbow-Height Cost
Any MPC YAML can include an `elbow_height` cost. Heights are spine3-relative Y
coordinates in metres. The cost is zero inside `[min, max]`, penalizes violations
across the predicted rollout, and adds a progress penalty if the elbow starts
outside the range and moves farther out.

```yaml
costs:
  elbow_height:
    min: 0.10
    max: 0.45
    weight: 100.0
    progress_weight: 100.0  # optional; defaults to weight
```

When MDM is enabled, set `preference_learning: false` to keep the configured
elbow-height bounds fixed after generated trajectories.

`--arm` can override the starting arm state with a `.npy` file. The preferred
shape is `(3, 3)` for `[left_shoulder, left_elbow, left_wrist]`. Legacy `(4, 3)`
files are accepted; the first row fixes the left collar and the remaining rows
control shoulder, elbow, and wrist.


## Thanks
This repository is based on [python-starter](https://github.com/tomsilver/python-starter), which is a general starter repository (not limited to research project code).
