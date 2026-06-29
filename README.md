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
First initialize the SAM 3D Body and MHR submodules:
```
git submodule update --init --recursive \
  src/uncertain_feedback/data_collection/sam-3d-body \
  src/uncertain_feedback/data_collection/MHR
```

Create the Conda environment used by the SAM 3D Body inference worker:
```
conda create -n sam_3d_body python=3.11 -y

conda run -n sam_3d_body python -m pip install --upgrade pip setuptools wheel

# CUDA build for the RTX 4080 / driver CUDA 12.2 setup on this machine.
conda run -n sam_3d_body python -m pip install \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

conda run -n sam_3d_body python -m pip install \
  pytorch-lightning pyrender opencv-python yacs scikit-image einops timm dill \
  pandas rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils \
  webdataset chump networkx==3.2.1 roma joblib seaborn wandb appdirs appnope \
  ffmpeg cython jsonlines pytest xtcocotools loguru optree fvcore black \
  pycocotools tensorboard huggingface_hub ninja

# Build detectron2 for your GPU architecture. RTX 4080 uses compute capability 8.9.
TORCH_CUDA_ARCH_LIST="8.9" FORCE_CUDA=1 MAX_JOBS=8 \
  conda run -n sam_3d_body python -m pip install \
  git+https://github.com/facebookresearch/detectron2.git@a1ce2f9 \
  --no-build-isolation --no-deps

# Optional, but needed for the upstream SAM 3D Body demo's default --fov_name moge2.
conda run -n sam_3d_body python -m pip install git+https://github.com/microsoft/MoGe.git

# Match detectron2's declared dependency range.
conda run -n sam_3d_body python -m pip install iopath==0.1.9
```

Verify the environment:
```
conda run -n sam_3d_body python -m pip check
conda run -n sam_3d_body python -c "import sys; sys.path.insert(0, 'src/uncertain_feedback/data_collection/sam-3d-body'); import torch, detectron2, sam_3d_body; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(detectron2.__version__)"
```

Download the gated SAM 3D Body checkpoint after Hugging Face access is approved:
```
conda run -n sam_3d_body hf download facebook/sam-3d-body-dinov3 \
  --local-dir src/uncertain_feedback/data_collection/sam-3d-body/checkpoints/sam-3d-body-dinov3
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
uv run python src/uncertain_feedback/data_collection/extract_all_frames.py
```

2. Label segments with text descriptions in browser
```
uv run python src/uncertain_feedback/data_collection/labeler.py
```
                                                                                       
3. Build MDM dataset
```
uv run python src/uncertain_feedback/data_collection/build_mdm_dataset.py --output_dir ./my_mdm_dataset/
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
(run from repo root. argument paths should be relative to `MDM_ROOT`)
```
uv run python src/uncertain_feedback/motion_generators/mdm/train_leftarm.py \
    --save_dir ./save/customv2 \
    --start_pose demo_pose.pt \
    --n_prefix 1 \
    --body_mode both \
    --dataset humanml \
    --resume_checkpoint ./save/humanml_enc_512_50steps/model000750000.pt \
    --diffusion_steps 100 \
    --mask_frames \
    --use_ema \
    --batch_size 8 \
    --num_steps 5000 \
    --save_interval 250 \
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


## Kimodo backend setup

[Kimodo](https://github.com/nv-tlabs/kimodo) (NVIDIA) is an optional second
text-to-motion backend. It pins `pydantic>=2` and `transformers==5.1.0`, which conflict
with the main environment, so it lives in its own conda env and is called via a subprocess
worker (`motion_generators/kimodo/_kimodo_inference_worker.py`).

**1. Hugging Face / Llama-3 access** (kimodo's text encoder uses gated
`meta-llama/Meta-Llama-3-8B-Instruct`):
- Accept the license at https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
- Create a read token at https://huggingface.co/settings/tokens
- `hf auth login` (or write the token to `~/.cache/huggingface/token`)

**2. Create the isolated env and install kimodo** (env name must match
`KIMODO_CONDA_ENV`, default `kimodo`):
```bash
conda create -n kimodo python=3.10 -y
conda install -n kimodo -y -c conda-forge cmake cxx-compiler   # kimodo C++ extension
# Install a torch build matching your GPU (cu128 for Blackwell sm_120):
conda run -n kimodo pip install torch --index-url https://download.pytorch.org/whl/cu128
conda run -n kimodo pip install "git+https://github.com/nv-tlabs/kimodo.git"
```
Model weights download automatically on first use. Set `TEXT_ENCODER_DEVICE=cpu` to cut
VRAM from ~17 GB to <3 GB.

**3. Run** any MDM-backed planner with `motion_generator: kimodo` in its YAML, e.g.:
```bash
uv run python src/uncertain_feedback/planners/run.py \
  --mpc-config src/uncertain_feedback/planners/mpc/configs/arm_mpc_cartesian_kimodo.yaml
```
The kimodo start pose is a SMPL `body_pose (21,3)` `.npy` (`motion_generators/kimodo/start_pose.npy`). The wrapper converts it through the same FK used by the visualizer; the worker retargets that pose onto Kimodo's skeleton and applies the resulting positions and global rotations as the frame-0 Kimodo constraint. `--frozen-body` is not supported with `motion_generator: kimodo`.

To generate only a kimodo motion and render it to video, without MPC:
```bash
TEXT_ENCODER_DEVICE=cpu uv run python src/uncertain_feedback/motion_generators/kimodo/generate_motion.py \
  --text "raise my left arm" \
  --num-frames 100 \
  --output-npz kimodo_motion.npz \
  --output-video kimodo_motion.mp4
```

## Running a Single MPC Run

`run.py` performs one end-to-end run: plan with sampling MPC, optionally inject a
language/LLM-generated cost correction at `text_time`, and finish the trajectory
with MPC. To compare multiple LLM costs across UQ clusters, use the experiment
runner instead (see [Running Cluster Experiments](#running-cluster-experiments)).

Use the runner from the repo root:
```
uv run python src/uncertain_feedback/planners/run.py --mpc-config path/to/mpc.yaml
```

Controller settings live in the required YAML file passed with `--mpc-config`.
The initial whole-body HML pose can be set with `pose:` in the YAML. Runtime
inputs still stay on the command line: `--model-path`, `--arm`, `--text`, `--save`, `--live`, and `--frozen-body`.
`--pose` is still accepted as an override for the YAML pose.

Supported YAML `planner` values:
- `arm_mpc`: joint-space MPC only, no MDM.
- `arm_mpc_mdm`: one MDM trajectory, then MPC tracks it.
- `arm_mpc_mdm_uq`: multiple MDM samples, clustering/picker, then MPC tracks the selected mean.
- `arm_mpc_cartesian`: MDM/UQ first, then Cartesian wrist-goal MPC.
- `arm_mpc_cartesian_no_mdm`: Cartesian wrist-goal MPC only, no MDM and no UQ.

### Motion-generation backend (`motion_generator`)

The text-to-motion backend is selected by the optional YAML key `motion_generator`:
- `mdm` (default): the in-process Motion Diffusion Model (see Getting Started).
- `kimodo`: NVIDIA's [kimodo](https://github.com/nv-tlabs/kimodo) SMPL-X model, run in an
  isolated conda env via a subprocess worker (see [Kimodo backend setup](#kimodo-backend-setup)).

Both backends expose the same interface, so any MDM-backed planner
(`arm_mpc_mdm`, `arm_mpc_mdm_uq`, `arm_mpc_cartesian`) works with either by setting
`motion_generator:` in its config.

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
uv run python src/uncertain_feedback/planners/run.py \
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
mdm_frames: 50

uq:
  diffusion_samples: 128
  n_clusters: 3
  auto_cluster: null
```

Run with an interactive cluster picker:
```bash
uv run python src/uncertain_feedback/planners/run.py \
  --mpc-config src/uncertain_feedback/planners/mpc/configs/mpc_mdm_uq.yaml \
  --model-path "src/uncertain_feedback/motion_generators/mdm/motion-diffusion-model/save/my_finetuned_final/model000750500.pt" \
  --text "raise my left arm" \
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
uv run python src/uncertain_feedback/planners/run.py \
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
uv run python src/uncertain_feedback/planners/run.py \
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

  elbow_flexion_angle:
    min: 0.40
    max: 1.80
    weight: 50.0

  shoulder_abduction_angle:
    min: 0.10
    max: 1.20
    weight: 50.0
```

When MDM is enabled, set `preference_learning: false` to keep the configured
preference bounds fixed after generated trajectories. Angle costs are in
radians; `elbow_flexion_angle` uses the controlled elbow rotation magnitude,
and `shoulder_abduction_angle` uses the upper-arm angle away from torso-down in
the spine3 frame.

`--arm` can override the starting arm state with a `.npy` file. The preferred
shape is `(3, 3)` for `[left_shoulder, left_elbow, left_wrist]`. Legacy `(4, 3)`
files are accepted; the first row fixes the left collar and the remaining rows
control shoulder, elbow, and wrist.


## Running Cluster Experiments

Experiments live separately from a single run, under
`src/uncertain_feedback/experiments/`. The experiment runner drives a UQ planner
(`arm_mpc_mdm_uq` or `arm_mpc_cartesian`, with `llm_cost.enabled: true`) up to
`text_time` to obtain the cluster set, then generates one LLM cost per cluster,
rolls each one out headlessly, and writes per-cluster metrics plus a
`comparison_summary.json`:

```bash
uv run python src/uncertain_feedback/experiments/run_experiment.py \
  --mpc-config src/uncertain_feedback/planners/mpc/configs/arm_mpc_cartesian_mdm_llm.yaml \
  --text "raise my left arm"
```

Add `--rollout-steps N` to cap the per-cluster rollout length (defaults to
`steps - text_time`), and `--save-video` to render each rollout to an MP4. The
saved video uses the same `ArmVisualizer` layout as a live run, so the only
difference between watching and saving is the flag.

### Comparing cost-generation backends

The backend experiment is the orthogonal axis: it holds the correction fixed (the
**chosen** UQ cluster) and generates a cost with each backend (`llm` / `turns` /
`agent`), then scores them all on the same rollout-vs-MDM L2 metric and writes a
`backend_comparison.json` ranking. It requires `planner: arm_mpc_cartesian` (the
scorer needs a persistent Cartesian goal) and `llm_cost.enabled: true`:

```bash
uv run python src/uncertain_feedback/experiments/run_backend_experiment.py \
  --mpc-config src/uncertain_feedback/planners/mpc/configs/arm_mpc_cartesian_mdm_llm.yaml \
  --text "raise my left arm" \
  --backends turns agent \
  --save-video
```

Pass the neutral base config (`arm_mpc_cartesian_mdm_llm.yaml`), not a
backend-specific one — the experiment sets `llm_cost.backend` itself for each
backend. All other `llm_cost` settings (`model`, `max_turns`, `prompt`,
`use_images`, and `codex_cmd` for the `agent` backend) come from that config, so
make sure its `codex_cmd` works on this host. Use `--backends llm turns` to
compare a subset, `--rollout-steps N` and
`--save-video` to render an MP4 per backend. With image feedback enabled,
`--save-video` also saves the rollout videos for every intermediate `turns` and
`agent` candidate cost. A backend that fails to produce a cost (e.g. `codex`
unavailable) is recorded as failed and the rest still rank.

#### Visual cost feedback (turns / agent)

When `llm_cost.use_images: true`, the iterating backends refine the cost against a
**rendered comparison** — a rollout-vs-correction overlay (red "cost rollout" vs
green "target correction") — not just the scalar L2 score, which is still kept for
selection and ranking:

- `turns`: each turn renders `turn_<i>/comparison.png` and feeds it (plus the
  score) back to the model via the multi-turn conversation. With `--save-video`,
  each turn also saves `turn_<i>/rollout.npy` and `turn_<i>/rollout.mp4`.
- `agent`: codex receives the initial context overlay image paths as text in
  `TASK.md`, is instructed to load those local files itself, and writes
  `ITERATION_LOG.md` describing what it saw in each image, why each cost
  revision was made, and whether it stopped because the movement matched well
  enough or because it determined the available cost API could not make it match. It also gets a pickled `state.pkl` and a render script it
  runs itself to inspect its rollout and iterate. The wrapper appends
  `ITERATION_LOG.md` into `codex.log` when the run finishes. The script can
  also be run standalone to re-render any candidate:

  ```bash
  uv run python src/uncertain_feedback/experiments/render_cost_comparison.py \
    --state <run_dir>/agent/state.pkl \
    --response <run_dir>/agent/response.json \
    --out comparison.png \
    --archive-dir candidates \
    --save-video
  ```

  It loads the pickled `EvalState`, rolls the goal-seeking MPC with the candidate
  cost, prints the L2 score, and writes the overlay PNG. With `--archive-dir`,
  each invocation creates `candidate_<i>/` containing `response.json`, `cost.py`,
  `score.json`, `comparison.png`, and, with `--save-video`, `rollout.npy` plus
  `rollout.mp4`.

With `use_images: false` both backends fall back to score-only text feedback.


## Thanks
This repository is based on [python-starter](https://github.com/tomsilver/python-starter), which is a general starter repository (not limited to research project code).
