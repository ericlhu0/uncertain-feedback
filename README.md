# Confidence-Aware Language Grounding

## Getting Started
Clone https://github.com/GuyTevet/motion-diffusion-model as `src/uncertain_feedback/motion_generators/mdm/motion-diffusion-model` and download the [required weights](https://github.com/GuyTevet/motion-diffusion-model?tab=readme-ov-file#mdm-is-now-40x-faster--04-secsample), [data](https://github.com/GuyTevet/motion-diffusion-model?tab=readme-ov-file#2-get-data) and [SMPL model](https://github.com/GuyTevet/motion-diffusion-model/blob/main/prepare/download_smpl_files.sh)

## Running (Custom) Motion Generation
```
uv run python src/uncertain_feedback/motion_generators/mdm/sample_leftarm.py \
--model_path src/uncertain_feedback/motion_generators/mdm/motion-diffusion-model/save/humanml_enc_512_50steps/model000750000.pt \
--text_condition "a person barely raises their left hand." \
--num_samples 1 \
--num_repetitions 1 \
--motion_length 5.0 \
--initial_pose_path src/uncertain_feedback/motion_generators/mdm/demo_pose.pt
--fix_body
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
Pose-estimation results are cached as `(N, 22, 3)` position arrays in `<frames_dir>/../mdm_cache/`
with a `_v<N>` version tag in the filename; bumping `_CACHE_VERSION` in `build_mdm_dataset.py`
(done whenever the pose-estimation/conversion changes) invalidates old entries automatically.
Cache files without a version tag predate the 2026-06-29 chirality fix and are mirrored; delete them.

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
    --use_ema \                               # EMA model averaging (see note below — inference loads the raw weights)
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
    --save_dir ./save/customv3 \
    --start_pose demo_pose.pt \
    --n_prefix 1 \
    --body_mode both \
    --dataset humanml \
    --resume_checkpoint ./save/humanml_enc_512_50steps/model000750000.pt \
    --diffusion_steps 50 \
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
- `--use_ema`: keeps an extra `model_avg` (EMA, beta 0.9999) copy in each checkpoint. **The EMA
  weights are ~95% the base model after 500 fine-tune steps** (still ~60% after 5000), so they must
  never be used for inference on short fine-tunes — the fine-tuned behavior is drowned out and
  generations revert to base-MDM whole-body motions. Inference always loads the raw `model` weights:
  `mdm_configs/mdm_config.yaml` sets `use_ema: false`, overriding the training flag saved in the
  checkpoint's `args.json`. Do not remove that override.

5. Run motion generation with the new model

From the repo root:
```
uv run python src/uncertain_feedback/motion_generators/mdm/sample_leftarm.py \
    --model_path src/uncertain_feedback/motion_generators/mdm/motion-diffusion-model/save/my_finetuned_v1/model000001000.pt \
    --text_condition "raise my left arm" \
    --num_samples 3 \
    --num_repetitions 5 \
    --motion_length 5.0
```
All paths are relative to wherever you invoke the script. Output videos are saved under `save/my_finetuned_v1/edit_*/` inside `motion-diffusion-model/`. (1s = 20 frames)


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

**Generation speed.** Kimodo's text encoder is an 8B LLM2Vec model. With no encoder
server reachable on `127.0.0.1:9550` it loads locally, and `TEXT_ENCODER_DEVICE=cpu`
(auto-set when VRAM < 18 GB) runs it on CPU — slow per prompt. The worker encodes the
(shared) prompt once per call regardless of `uq.diffusion_samples`, so generation time is
dominated by the one-time ~85 s model load plus diffusion, not the sample count. To speed
diffusion, lower `num_denoising_steps` (top-level YAML key, kimodo only; default 100,
try 30-50 for a quality/speed trade).

To generate only a kimodo motion and render it to video, without MPC:
```bash
TEXT_ENCODER_DEVICE=cpu uv run python src/uncertain_feedback/motion_generators/kimodo/generate_motion.py \
  --text "raise my left arm" \
  --num-frames 100 \
  --output-npz kimodo_motion.npz \
  --output-video kimodo_motion.mp4 \
  --start-pose src/uncertain_feedback/motion_generators/kimodo/start_pose_kimodo.npy
```

## Running a Single MPC Run

`run.py` performs one end-to-end run: plan with sampling MPC, inject the first
language/LLM-generated correction at `text_time` (or earlier when a configured
simulated user becomes uncomfortable), and finish the trajectory with MPC. A
restricted simulated user can interrupt again whenever the executed motion
crosses its discomfort threshold after first returning to the comfortable side.
Each interruption replaces the unfinished MDM playback suffix and starts a new
correction from the arm's actual current pose.

Add the repeated-feedback threshold to an MDM config with:

```yaml
corrections:
  trigger_threshold: 0.02
```

`transfer.trigger_threshold` is still accepted as a legacy fallback. The global
`steps` value bounds the run; there is no separate correction-count limit. When
LLM costs are enabled, one cost is generated and stacked per correction. At the
end of the trajectory, the saved feedback rounds are passed through the same
multi-round combinator used across goals, and a successful unified cost replaces
the stacked generated terms. Configured hand-authored costs are always preserved.
Artifacts are grouped under
`<llm_cost.artifact_dir>/<timestamp>/trajectory_00/`, with `round_<N>/`,
`history.json`, `executed_trajectory.npy`, and the optional
`combine_after_trajectory_00/` directory.

To compare multiple LLM costs across UQ clusters, use the experiment runner
instead (see [Running Cluster Experiments](#running-cluster-experiments)).

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
- `arm_mpc_mdm`: MDM correction playback followed by joint-goal MPC, with repeated corrections for restricted users.
- `arm_mpc_mdm_uq`: UQ clustering/picker followed by joint-goal MPC, with repeated corrections for restricted users.
- `arm_mpc_cartesian`: MDM/UQ correction playback followed by Cartesian wrist-goal MPC, with repeated corrections for restricted users.
- `arm_mpc_cartesian_no_mdm`: Cartesian wrist-goal MPC only, no MDM and no UQ.

Set the optional top-level `seed` key to control MPC action sampling. It defaults
to `0`; use another nonnegative integer to reproduce a different sampling
sequence.

### Motion-generation backend (`motion_generator`)

The text-to-motion backend is selected by the optional YAML key `motion_generator`:
- `mdm` (default): the in-process Motion Diffusion Model (see Getting Started).
- `kimodo`: NVIDIA's [kimodo](https://github.com/nv-tlabs/kimodo) SMPL-X model, run in an
  isolated conda env via a subprocess worker (see [Kimodo backend setup](#kimodo-backend-setup)).

Both backends expose the same interface, so any MDM-backed planner
(`arm_mpc_mdm`, `arm_mpc_mdm_uq`, `arm_mpc_cartesian`) works with either by setting
`motion_generator:` in its config.

### Simulated user (`user:`)

Every run loads a simulated care recipient alongside the pose, selected by the
optional YAML key `user:` (default `unrestricted` — no movement restrictions).
Restricted personas (`adhesive_capsulitis`, `elbow_contracture`, `painful_arc`,
`stroke_flexor_synergy`, `triceps_long_head_contracture`,
`biceps_long_head_contracture`, `brachial_plexus_mechanosensitivity`,
`out_of_synergy_reach_preference`, `cross_body_pain`;
see `src/uncertain_feedback/simulated_users/personas.py`)
carry hidden joint-limit bounds and a fixed feedback line. When the configured
user has bounds:

Hidden feature bounds are soft comfort/evaluation penalties, not simulator hard
stops. They may encode either a physical or symptom-limited range or a documented
movement preference; each persona description states which interpretation applies.
`out_of_synergy_reach_preference` models the rehabilitation goal of combining
shoulder flexion/elevation with elbow extension during outward reaching, a movement
outside the post-stroke flexor synergy ([Hadjiosif et al., 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11403926/);
[Ellis et al., 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC5825888/)).

- `--text` defaults to the user's feedback line (an explicit `--text` still wins;
  with the unrestricted user the default stays `"move my arm up"`).
- `uq.user_cluster: true` delegates UQ cluster selection to the user (it picks
  the most comfortable cluster mean), taking precedence over `uq.auto_cluster`
  and the interactive picker.

The same hidden bounds are the evaluation ground truth for the
[transfer experiment](#simulated-user-transfer-experiment).

### Minimal Joint-Space MPC Config
Save as `src/uncertain_feedback/planners/mpc/configs/mpc_plain.yaml`:
```yaml
planner: arm_mpc
steps: 500
horizon: 10
n_mpc_samples: 256
seed: 0  # reproducible MPC action sampling
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
  clusterer: agglo_end_pose  # kmeans_end_pose | agglo_end_pose | agglo_path_pca | agglo_t2m
  auto_cluster: null
  scale: 1.0  # default motion-magnitude scale for the chosen cluster
```

`uq.clusterer` selects the clustering method (used by the demo runner; the MPC
planners keep their injected clusterer):

- `kmeans_end_pose` — KMeans on spine3-relative arm-chain positions at a
  single late frame.
- `agglo_end_pose` — same end-pose features, average-linkage agglomerative
  (default).
- `agglo_path_pca` — full-trajectory features (arm-chain path resampled to 15
  equidistant-arclength waypoints, PCA to 95% variance), agglomerative.
- `agglo_t2m` — 512-dim T2M motion-encoder embeddings (the FID/R-precision
  evaluator), agglomerative. Requires a one-time weights download:

  ```bash
  cd src/uncertain_feedback/motion_generators/mdm/motion-diffusion-model
  bash prepare/download_t2m_evaluators.sh  # needs gdown
  ```

In the interactive picker, each cluster panel has a **magnitude** slider
(range 0.0–2.0) that scales that trajectory's motion up or down while
preserving the direction of motion at every timestep (`scale` in joint-angle
space about the start pose: `1.0` = unchanged, `0.0` = hold start). `uq.scale`
sets the slider's initial value and is used directly as the scale in headless
runs. Select a panel and click **Refine selected** to cluster only that
panel's raw trajectories into `uq.n_clusters` child options; refinement can be
repeated recursively. When a selected option has fewer trajectories than
`uq.n_clusters`, each trajectory becomes its own child option. **Back** restores
the parent options and selection, and **Confirm** accepts the selected mean at
the current depth.

Run with an interactive cluster picker:
```bash
uv run python src/uncertain_feedback/planners/run.py \
  --mpc-config src/uncertain_feedback/planners/mpc/configs/mpc_mdm_uq.yaml \
  --model-path "src/uncertain_feedback/motion_generators/mdm/motion-diffusion-model/save/my_finetuned_final/model000750500.pt" \
  --text "raise my left arm" \
  --save out.mp4 \
  --live
```

For headless runs, set `uq.auto_cluster` in the YAML (and optionally `uq.scale`
to apply a fixed magnitude without the GUI):
```yaml
uq:
  diffusion_samples: 128
  n_clusters: 3
  auto_cluster: 0
  scale: 1.0
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


## Running Simulated-User Experiments

Experiments live separately from a single run, under
`src/uncertain_feedback/experiments/`. The default experiment runs one simulated
persona with one cost-generation backend on the original goal only: initial
rollout, hidden-cost trigger, MDM/UQ candidates, oracle cluster selection, one
generated cost, and original-goal evaluation (`base`, `tracking`, `generated`,
`oracle`). It requires `planner: arm_mpc_cartesian`, `llm_cost.enabled: true`,
and a persona with hidden bounds:

```bash
uv run python src/uncertain_feedback/experiments/run_experiment.py \
  --mpc-config src/uncertain_feedback/planners/mpc/configs/arm_mpc_cartesian_mdm_llm_transfer.yaml \
  --persona adhesive_capsulitis \
  --backend agent \
  --save-video
```

`--persona` defaults to the config's `user:` key, and `--backend` defaults to
`llm_cost.backend`. Artifacts go to `experiment_artifacts/<timestamp>/`,
including `experiment_summary.json`.

The older per-cluster comparison remains available as an explicit cluster
experiment. It drives a UQ planner to the feedback point, extracts every cluster,
generates one cost per cluster with the selected backend, rolls each one out
headlessly, and writes `comparison_summary.json`. For Cartesian experiments,
each cluster entry also contains a `hidden_cost_evaluation` comparing `base`,
`oracle`, and that cluster's `generated` cost on the original goal:

```bash
uv run python src/uncertain_feedback/experiments/run_cluster_experiment.py \
  --mpc-config src/uncertain_feedback/planners/mpc/configs/arm_mpc_cartesian_mdm_llm.yaml \
  --text "raise my left arm" \
  --backend llm
```

Add `--rollout-steps N` to cap the per-cluster rollout length (defaults to
`steps - text_time`), and `--save-video` to render each rollout to an MP4.

### Comparing cost-generation backends

`llm_cost.backend` selects how the cost is generated:

Unless overridden by `llm_cost.model` or `OPENAI_MODEL`, LLM cost generation uses
`gpt-5.6-luna` with `xhigh` reasoning effort. Reasoning effort follows the model
(`gpt-5.6-luna` → `xhigh`, `gpt-5.6-sol` → `low`); any other model is sent without
one. The demo-runner config (`arm_mpc_cartesian_mdm_llm_transfer.yaml`)
pins `gpt-5.6-sol`.

- `llm` — three focused LLM calls, run once: **interpret** (instruction + contrast
  images + compact summary, including rollout-labeled chosen-vs-marked-wrong terminal
  joint-feature comparisons when the person explicitly rejects UQ alternatives →
  plain-language preference),
  **ground** (preference + full
  numeric summaries → concrete features and bounds — a bound is a constant threshold by
  default, or pose-dependent, its threshold sliding with a second feature via anchor
  points, only when the feedback ties one joint's range to another's position), **author**
  (spec + runtime API /
  output contract → cost JSON). Each stage's prompt and raw response are saved as
  `<stage>_prompt.txt` / `<stage>_response.txt`, with a readable aggregate in
  `stage_log.md`. Every successful backend also writes `rationale.json`, which links
  the instruction, interpretation and modality-specific evidence, grounded numeric
  terms and their sources, final explanations, and the winning cost's empirical
  chosen/original/marked-wrong trajectory ranking when comparison trajectories exist.
- `turns` — a multi-turn conversation that rolls out each candidate, scores it, and feeds
  the score plus rendered comparisons back to refine the grounding and authoring while
  keeping the initial interpretation fixed.
- `agent` — delegates the same staged method and optional rollout iteration to the
  external `codex` CLI. The agent is required to write `stage_log.md` with its Stage 1,
  Stage 2, and Stage 3 responses; that log is also appended to `codex.log`. Each run is
  wrapped in a fail-closed Bubblewrap filesystem namespace. It can read only its staged
  task inputs, a minimal rollout runtime, and the project virtual environment; the real
  repository, simulator personas, and previous artifact runs are not mounted. Linux
  hosts using this backend must provide the `bwrap` executable. Staged `state.pkl`
  files contain only the operational MPC fields needed to reproduce a rollout
  (planner, goal, sampling/step settings, thresholds, and seed), never the full run
  config or its `user`/`persona_goals` metadata. Legacy round states are sanitized
  while being staged for combination.

The backend experiment is the orthogonal axis: it holds the correction fixed (the
**chosen** UQ cluster) and generates a cost with each backend (`llm` / `turns` /
`agent`), then scores them all on the same rollout-vs-MDM L2 metric and writes
a `backend_comparison.json` ranking. Each backend entry also includes a
`hidden_cost_evaluation` comparing `base`, `oracle`, and `generated` using the
simulated user's hidden cost and original Cartesian goal. It requires
`planner: arm_mpc_cartesian` (the
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
backend. All other `llm_cost` settings (`model`, `max_turns`, `use_images`, and
`codex_cmd` for the `agent` backend) come from that config, so
make sure its `codex_cmd` works on this host and that Bubblewrap is installed. The
configured Codex sandbox may be `danger-full-access` because Codex itself runs inside
the outer restricted namespace. Use `--backends llm turns` to
compare a subset, `--rollout-steps N` and
`--save-video` to render an MP4 per backend. With image feedback enabled,
`--save-video` also saves the rollout videos for every intermediate `turns` and
`agent` candidate cost. A backend that fails to produce a cost (e.g. `codex`
unavailable) is recorded as failed and the rest still rank.

Every generated LLM-cost artifact directory also includes
`reference_with_correction.mp4`, a video of the target reference trajectory that
contains the correction (`full_correction_traj` when available, otherwise the
MDM correction segment).

#### Visual cost feedback (turns / agent)

When `llm_cost.use_images: true`, the iterating backends refine the cost against
**two rendered comparisons** — a spatial rollout-vs-correction overlay (red "cost
rollout" vs green "target correction") and a joint-angle-over-time graph plotting
the same two trajectories' arm joint angles (green target vs red rollout) so the
model sees the shape of the movement, not just endpoints — not just the scalar L2
score, which is still kept for selection and ranking:

- `turns`: each turn renders `turn_<i>/comparison.png` and `turn_<i>/angles.png`
  and feeds both (plus the score) back to the model via the multi-turn
  conversation. With `--save-video`, each turn also saves `turn_<i>/rollout.npy`
  and `turn_<i>/rollout.mp4`.
- `agent`: codex receives the initial context overlay image paths as text in
  `TASK.md`, is instructed to load those local files itself, and writes
  `ITERATION_LOG.md` describing what it saw in each image, why each cost
  revision was made, and whether it stopped because the movement matched well
  enough or because it determined the available cost API could not make it match. It also gets a pickled `state.pkl` and a render script it
  runs itself (writing both `comparison.png` and `angles.png`) to inspect its
  rollout and iterate. The wrapper appends
  `ITERATION_LOG.md` into `codex.log` when the run finishes. The script can
  also be run standalone to re-render any candidate:

  ```bash
  uv run python src/uncertain_feedback/experiments/render_cost_comparison.py \
    --state <run_dir>/agent/state.pkl \
    --response <run_dir>/agent/response.json \
    --out comparison.png \
    --angles-out angles.png \
    --archive-dir candidates \
    --save-video
  ```

  It loads the pickled `EvalState`, rolls the goal-seeking MPC with the candidate
  cost, prints the L2 score, and writes the overlay PNG (plus the joint-angle
  graph when `--angles-out` is given). With `--archive-dir`, each invocation
  creates `candidate_<i>/` containing `response.json`, `cost.py`, `score.json`,
  `comparison.png`, `angles.png`, and, with `--save-video`, `rollout.npy` plus
  `rollout.mp4`.

With `use_images: false` both backends fall back to score-only text feedback.

### Simulated-user transfer experiment

The transfer experiment closes the evaluation loop with a **hidden ground
truth**: a simulated care recipient (`src/uncertain_feedback/simulated_users/`)
holds a clinically motivated ROM restriction the cost generator never sees. The
persona decides when feedback is given (the first step the initial plan violates
the hidden cost), what is said (its fixed feedback line — `--text` is ignored),
and which UQ cluster it picks: transfer experiments score each scaled raw
cluster mean with the hidden oracle cost and choose the lowest-scoring option.
The generated cost is then evaluated by rolling out to the **original goal and
each held-out `transfer.goals` entry** and measuring hidden-cost violation plus goal
completion — so a cost only wins by generalizing beyond the correction it was
generated from.

```bash
uv run python src/uncertain_feedback/experiments/run_transfer_experiment.py \
  --mpc-config src/uncertain_feedback/planners/mpc/configs/arm_mpc_cartesian_mdm_llm_transfer.yaml \
  --save-video
```

The persona comes from the config's `user:` key (the example config sets
`adhesive_capsulitis`); `--persona` takes one or more persona names to run, and
`--all-personas` runs every persona with hidden bounds. Each persona gets its
own timestamped artifact dir, reusing one loaded MDM setup. With `--save-video`,
iterating cost backends also save candidate rollout artifacts under
`cost_generation/`.
Personas: `adhesive_capsulitis`, `elbow_contracture`, `painful_arc`,
`stroke_flexor_synergy` (active-effort pose proxy),
`out_of_synergy_reach_preference` (soft preference for progressively more elbow extension with elevation),
`triceps_long_head_contracture` (maximum elbow flexion falls with elevation),
`biceps_long_head_contracture` (minimum elbow flexion rises with shoulder extension),
`brachial_plexus_mechanosensitivity` (minimum elbow flexion rises with abduction),
and `cross_body_pain`
(pose-dependent bound: tolerable elevation drops linearly as the upper arm
adducts past the midline) — the unrestricted default is rejected. Requires `planner: arm_mpc_cartesian`, `llm_cost.enabled: true`,
`cartesian.goals`, and a `transfer:` block:

```yaml
transfer:
  goals:                     # held-out spine3-relative wrist targets
    - [-0.25, 0.0, -0.05]
corrections:
  trigger_threshold: 0.02    # hidden-cost violation (rad) at which the user interrupts
```

Because the experiment runs one persona at a time, each restriction needs its
own goal geometry to make the default plan visibly require a correction. An
optional `persona_goals:` block overrides the correction goal and transfer goals
for the active persona (falling back to the top-level `cartesian.goals` /
`transfer.goals` for personas without an entry). Goals sit inside the
constraint-compliant reach envelope: the constraint-respecting solution reaches
the goal, while the default plan reaches the same target by violating the
restriction (frozen shoulder raises the upper arm; flexor synergy straightens
the elbow), so the correction is "reach it a different way," not "give up":

```yaml
persona_goals:
  adhesive_capsulitis:       # frozen shoulder: base raises the upper arm; compliant keeps it low
    cartesian: [[0.42, 0.30, 0.12]]                                    # low-hand lateral (hand not overhead)
    transfer:  [[0.50, 0.20, 0.10], [0.11, 0.44, 0.26], [-0.04, 0.42, 0.18]]
  stroke_flexor_synergy:     # flexor synergy: base straightens the elbow; compliant keeps it bent
    cartesian: [[0.42, 0.48, 0.12]]
    transfer:  [[0.18, 0.52, 0.14], [0.13, 0.50, 0.28], [-0.04, 0.48, 0.18]]
```

Artifacts go to `transfer_artifacts/<timestamp>/`: `initial_rollout.npy`,
`cluster_options.png` (the UQ cluster candidates the simulated user chose among,
chosen cluster highlighted), the cost-generation directory (same layout as a
live run), per-condition rollouts
(`base/`, `tracking/`, `generated/`, `oracle/`, with MP4s under `--save-video`),
and `transfer_summary.json` with per-condition per-goal metrics
(`mean_violation`, `max_violation`, `frac_frames_violated`, `goal_reach`) plus
`cluster_selection_method` and `cluster_oracle_scores` for the UQ options.
`tracking` (following the correction trajectory directly) is only defined for
the original goal; on transfer goals it is identical to `base` — that contrast
is the argument for persisting a cost function rather than a trajectory.

### Multi-round pose-dependent experiment

The multi-round experiment treats every entry in `cartesian.goals` as the next
goal for the same care recipient. Each triggered round still runs the normal
single-round generator against the hand-authored base comfort costs. From the
second triggered round onward, a Codex combinator replays every round's feedback,
trajectories, summaries, images, and generated cost to author one unified cost.
The unified cost replaces earlier generated costs; generated terms are never
stacked.

The example config includes two base-vs-oracle-screened pose-dependent scenarios
using `demo_pose_v3.pt` (initial spine3-relative wrist position approximately
`[0.356, -0.091, 0.316]`). Coordinates below are spine3-relative wrist goals:

| Persona | Round 1 | Round 2 | Round 3 purpose |
|---|---|---|---|
| `triceps_long_head_contracture` | `[0.25, 0.32, 0.15]` | `[-0.05, 0.40, 0.15]` (across body) | `[0.10, 0.10, 0.25]`: low close reach that rejects an overly global elbow-extension rule |
| `cross_body_pain` | `[-0.04, 0.45, 0.15]` | `[-0.12, 0.25, 0.25]` | `[0.42, 0.48, 0.12]`: high lateral reach that rejects an overly global elevation cap |

The first two goals deliberately sample different parts of each hidden diagonal
boundary. The third is a comfortable generalization goal on the allowed side of
the boundary, chosen to expose a simpler but overrestrictive rule. A sufficiently
good generator may infer the coupled preference after the
first correction; if it overfits that observation, the second violation supplies
the contrasting evidence needed by the multi-round combinator.

```bash
uv run python src/uncertain_feedback/experiments/run_multi_round_experiment.py \
  --mpc-config src/uncertain_feedback/planners/mpc/configs/arm_mpc_cartesian_mdm_llm_multiround.yaml \
  --persona stroke_flexor_synergy \
  --save-video
```

Artifacts go to `multi_round_artifacts/<timestamp>/`. Each `round_<k>/` contains
the initial rollout, correction, pickled evaluation state, cluster overlay, and
unchanged per-round `cost_generation/` artifacts. `history.json` is the durable
full-context round history. Each `combine_round_<k>/` contains the unified cost,
iteration log, per-round comparison images, and `scores.json`. Final
base/generated/oracle rollouts are evaluated for every goal, and
`hidden_bounds_goal_<k>.png` plots those trajectories against the persona's true
forbidden region; for coupled personas this is the direct visual check that the
learned pose-dependent anchors follow the hidden diagonal boundary.

`trajectory_corpus/` holds one entry per goal — the goal's full executed
rollout (`traj_<i>.npy`) plus a per-frame joint-feature `traj_<i>_features.csv`
— recorded in `manifest.json`. Each manifest entry carries `goal`, `n_frames`,
`trigger_step`/`trigger_violation` (`null` when the goal never triggered a
correction), `feedback_text`, and `comfortable_until` (the trigger step, or
`n_frames` if none): frames `[0, comfortable_until)` were executed without a
discomfort report. Every generated-cost backend uses this corpus to keep previously
accepted poses unpenalized, so goals that never trigger a correction still contribute
negative evidence. The `llm` and `turns` backends receive numeric per-entry feature
ranges; after authoring, all backends are checked directly on every accepted pose and
the cost is rejected if any receives positive cost. Before an agent run, the relevant
trajectory and feature files are copied into its task workspace and the oracle-derived
`feedback_text` and `trigger_violation` fields are removed from the staged manifest.
The original manifest remains unchanged for the demo UI and session history; only
explicit round instructions enter the agent prompt.


## Demo runner web tool

Browser tool for designing and presenting simulated-user demo scenarios
interactively — the staged pipeline above (base rollout → MDM/UQ correction →
cost generation), but with every knob tweakable and every trajectory
inspectable before committing to an experiment config, plus a **demo/dev mode
toggle** and full **session replay**. Run it from the repo root (the artifact
root is CWD-relative):

```bash
uv run python src/uncertain_feedback/demo_runner/server.py \
  [--mpc-config src/uncertain_feedback/planners/mpc/configs/arm_mpc_cartesian_mdm_llm_transfer.yaml] \
  [--personas-file demo_runner_personas.json] \
  [--trajectory-configs-file demo_runner_trajectory_configs.json] \
  [--host 127.0.0.1] [--port 6781]
```

Then open `http://127.0.0.1:6781`. The config supplies the pose, MPC settings,
UQ defaults, `mdm_frames`, per-persona goal presets, and the `llm_cost` backend
used by the cost-generation stage. With the default config, cost generation
starts on `llm (single-pass)`; the backend dropdown can still select `turns` or
`agent`. When `motion_generator: mdm`, the top-level `seed` is reset before each
MDM generation request, so identical demo inputs reproduce the same samples.

**Demo vs dev mode.** The header button toggles between them and the choice
persists in `localStorage` (demo is the default). Both modes use the same
guided left control bar: Scenario is a persistent setup/status panel above the
four stages Trajectory decision, Language correction, Cost generation, and
Apply feedback. Its configuration section is collapsible and closes
automatically when a trajectory starts, while Start/Exit and the live
trajectory status remain visible. Starting or continuing a trajectory stops at Trajectory
decision, where the operator either enters a correction or ignores the comfort
violation and continues. Entering a correction unlocks Language correction.
Selecting a cluster keeps that stage open for inspection and enables its
bottom *Next* button; pressing *Next* advances to Cost generation. Generating a
cost enables the Cost stage's bottom *Next* button; pressing it advances to
Apply feedback. Applying feedback at another trigger returns
to Trajectory decision. The current and completed stages remain clickable for
review, while future stages stay disabled. While cost generation runs, its user-visible progress appears
directly in the Cost stage in both modes; OpenAI-backed generation includes its
opt-in reasoning-summary stream there. Raw private reasoning is not exposed.
The right column begins with a persistent **Cost functions** section. Each
pending or active cost is shown by its natural-language summary, which expands
to reveal the generated Python. Individual round costs are listed separately
until combination replaces them with one unified cost.

*Dev* mode exposes all setup and debugging controls inside that organization.
*Demo* mode keeps the Scenario panel's saved start-pose and goal selectors,
but hides everything used to author those choices: the persona selector/editor,
initial-pose sliders, numeric goal fields, and save controls. It also hides
on-graph bound dragging, the sample/count/clusterer fields, the cost backend
picker and inline generated Python in the workflow panel, the trajectory corpus,
and the server console. The expandable right-column cost section remains visible.
Cluster refinement and explicit *Exclude* controls remain available in both
modes. Hidden controls are hidden, not removed, so they still
supply their configured values to the pipeline; in demo mode the graphs render
but their bound handles are inert. Cluster cards reserve oracle/full-path
violation diagnostics for dev mode and omit negative goal-status labels.

**Sessions.** All work happens inside a *session* — one simulated user plus the
context that accumulates while correcting them: an on-disk trajectory corpus,
committed correction rounds, and one unified cost. `POST /api/session/start`
with `{"persona": <name>}` begins one (creating
`demo_runner_artifacts/<timestamp>_session_<persona>/`
with `trajectory_corpus/` and `session.json`); trajectories are then spawned
from the session and carry its learned costs installed from frame 0, so
corrections compound across trajectories within the same session. The session's
cost generators read its trajectory corpus (every executed segment, comfortable or
not — see *Grounding cost generation in past comfort* above).
`POST /api/manual_trajectory/start` now accepts only `arm_aa` and `goal`; the
persona is fixed by the active session.
Sessions persist to `session.json` after every mutation and survive a server
restart: `GET /api/sessions` lists them and `POST /api/session/resume` with
`{"dir": <session dir>}` reloads one, recompiling every round cost and the
unified cost from stored code + params + the pickled eval state. A resumed session starts
with no live trajectory (MPC stepping state is not persisted) but full context.
`DELETE /api/corpus/<index>` drops one corpus entry. (There is no
`/api/base_rollout`; base rollouts happen only as the first step of starting a
trajectory.)

In the browser header, click the session summary to open its dropdown. The
dropdown starts a new session for a selected persona and lists saved sessions
with *Resume* and *Delete* actions (`DELETE /api/sessions/<name>`). While no
session is active, the persona dropdown remains enabled and *Start trajectory*
is disabled. An active session locks that persona and shows its start time plus
trajectory, corpus, round, and unified-cost counts. Resuming restores its
persisted rounds, unified cost, and corpus through `POST /api/session/resume`.
Starting or resuming replaces the active session, while deleting the active
session clears the displayed trajectory and accumulated context.

Stages (each stage's controls unlock once the previous one ran):

1. **Scenario** — edit the start arm pose (per-joint axis-angle sliders limited
   where the selected persona has a joint box, with a live SMPL body preview), the
   spine3-relative Cartesian goal, and the simulated user. Initial poses and
   goals can each be named, saved, and selected independently; they persist in
   `demo_runner_trajectory_configs.json` by default, and saving an existing
   name updates it. *Start trajectory*
   starts one stateful `arm_mpc_cartesian` execution and animates it live: the
   browser requests one MPC step at a time (`POST /api/live_trajectory/start`,
   then `POST /api/live_trajectory/step`), follows the newest frame in the body
   views and scrubber, and stops requesting steps as soon as the selected
   persona crosses its discomfort threshold and the trajectory pauses (or it
   completes). `text_time` is ignored. An
   oracle-cost rollout from the configured initial pose is generated and
   displayed when the page loads, and starting an edited scenario regenerates
   it for that pose and goal. While the trajectory is paused, *Oracle from MDM trigger*
   obtains a comparison rollout from the current trigger pose and prepends the
   already executed path. Run the normal language-correction stage,
   manually inspect/re-cluster/scale/select an MDM cluster, and run the normal
   cost-generation stage. *Apply feedback + continue trajectory* then installs exactly
   that selected correction and cost and resumes the same animated stepping
   (`POST /api/live_trajectory/apply_round`) to the next
   trigger. This pause → choose → generate cost → continue cycle can repeat
   until the goal is reached or `steps` is exhausted; no cluster is selected
   automatically. *Ignore comfort violation + continue* dismisses only the
   current discomfort event without adding feedback; the trajectory can pause
   again after it returns to comfort and crosses a bound later. The accumulated
   executed trajectory is saved under
   `demo_runner_artifacts/<timestamp>_session_<persona>/<timestamp>_trajectory/`.
   The live planner (its MPC stepping state) is held in server memory and is lost
   when the server restarts, but the session's rounds, unified cost, and corpus
   persist and can be resumed. *Exit trajectory* abandons the live planner,
   retains the already-written trajectory artifacts and the session's rounds/costs,
   and unlocks the scenario controls so another trajectory can be started in the
   same session (with the learned costs still installed).
2. **Language correction** — edit the MDM prompt (prefilled with the persona's
   feedback line), sample count, and cluster count; *Generate* draws diffusion
   samples from the feedback-trigger pose (shown as a purple ghost body in
   the SMPL body views; falls back to the start pose when the base never
   violates) and clusters them (*Re-cluster* reuses cached samples at the
   currently displayed refinement level). The *Clusterer* dropdown selects the
   clustering method (see `uq.clusterer` above; initial value from the config)
   and applies on the next Generate/Re-cluster/Refine. Each cluster is
   represented by its medoid — an actual MDM sample — rather than the
   elementwise mean. Every
   cluster option is automatically integrated into the full corrected
   trajectory — executed history → scaled correction → comfort-only goal
   continuation — so cards show what actually happens if that option is taken:
   sample count, oracle score, full-path violation, and goal reach. Click a
   card to select it. Use *Mark wrong* on any explicitly undesirable alternatives;
   only those marked cards are used as negative contrast for cost generation, while
   unmarked non-chosen cards carry no preference signal. Selection and exclusion
   are independent: an excluded card can remain selected for inspection, but the
   workflow cannot advance until it is restored or another included card is selected.
   *Refine selected* clusters only that card's samples and
   can be used recursively; if the card has fewer samples than the requested
   cluster count, each sample becomes its own child cluster. *Back* restores the
   prior options and selection. The breadcrumb reports the current depth and
   subset size. Drag *Magnitude* to rescale (re-clusters the current subset and
   re-assembles at the new scale). Outside this tool there is no mark-wrong interaction:
   automated and simulated-user paths pass no negative clusters, so generated
   costs contrast the chosen correction only with the original reference plan.
3. **Cost generation** — runs the selected cost-generation backend (`llm`,
   `turns`, or `agent`) on the selected correction, rolls the MPC out with the
   generated cost installed from the original edited start pose, and shows the
   resulting trajectory, metrics, and cost code. Its grounded feature limits are
   overlaid on the corresponding feature graphs in blue. A *why this cost* disclosure
   shows the interpretation evidence, grounded sources, explanation, and ranking table;
   the same disclosure is retained on each committed round card. Artifacts go to
   `demo_runner_artifacts/<timestamp>_session_<persona>/<timestamp>_<backend>/`.
   The complete browser payload
   is also saved as `demo_runner_payload.json`; refreshing the browser restores
   the generated trajectories, overlays, code, and pending Apply action without
   rerunning cost generation. This refresh recovery uses the live server session;
   restarting the server still clears the pending action.

4. **Multi-round feedback** — interactive version of the multi-round
   experiment above. *Commit round* records the last generated cost as a
   feedback round (goal, feedback text, trigger pose, cost, pickled eval
   state). With ≥2 committed rounds, *Combine rounds (codex)* runs the
   `CombineCostGenerator` over all rounds and installs the resulting unified
   cost, which **replaces** the per-round costs. Each committed card has a
   *Remove feedback* action; removal immediately excludes that round from future
   planning/combinations and invalidates a previously unified result. It does
   not rewind frames already executed under that feedback. Clicking *Combine rounds*
   automatically rolls the unified cost out from the initial pose. Combining also
   works between trajectories (no active trajectory): there is no pose or goal to
   demonstrate on, so the rollout, metrics, and goal status are omitted and the
   unified cost is installed when the next trajectory starts. The unified cost's
   rollout (dark cyan), penalty field, code, and per-round scores are shown
   in the panel and survive base rollouts; *Reset* clears all rounds. To
   probe pose dependence, give feedback at several goals that span the
   conditioning feature and check the unified penalty field against the
   diagonal oracle limit in the phase plot. Combine artifacts go to
   `demo_runner_artifacts/<timestamp>_session_<persona>/<timestamp>_combine/`;
   round state is written to the session's `session.json` and is restored when
   the session is resumed after a restart.
5. **Trajectory corpus** — lists every session manifest entry with its kind,
   round, goal, comfortable frame range, and optional trigger/violation. *Delete
   entry* calls `DELETE /api/corpus/<index>` and immediately re-renders the
   renumbered manifest. This edits only corpus evidence; committed round and
   unified-cost controls remain in the preceding session-context panel.

The center panel shows front/side/top SMPL body views with base or accumulated
multi-turn execution (red), optional
oracle-cost (brown), correction (green), full corrected path (teal), and
generated-cost (blue) overlays. Only the selected cluster's body and trace are shown in these views;
all cluster trajectories remain plotted on the joint-angle graphs. Visible
trajectories are translucent neutral SMPL meshes at the scrubbed frame; wrist traces turn bright red on frames that violate the hidden
bounds — with a scrubber/play control (arrow keys step, space plays) and
per-trajectory violation strips aligned under the scrubber. A console panel at
the bottom (dev mode only) streams the server's stdout live, so long stages (MPC rollouts, MDM
sampling, cost generation) show their progress. OpenAI-backed cost generation
also streams the model's opt-in reasoning summary here, and the multi-round
Codex combiner tees its user-visible agent log here while retaining the same
output in `codex.log`; raw private reasoning is not exposed. The right column plots each
joint feature over time with the persona's oracle limits shaded in red and
the generated-cost limits shaded in blue. Both limit overlays have
independent, default-on toggles. When the persona has a `coupled`
(pose-dependent) bound, a feature-vs-feature phase graph is added at the top of
the column: the conditioning feature on the x-axis, the bounded feature on the
y-axis, the pose-dependent limit as a diagonal line with its violating side
shaded, and each trajectory traced through the plane with a dot at the scrubbed
frame — so you can see the limit itself sliding as the two features co-vary.
When the generator emits grounded `terms`, the generated cost's declared limit is
drawn exactly in blue: constant bounds add a dashed threshold to the time graph,
and pose-dependent bounds add a piecewise-linear boundary with held endpoint values
and a shaded violating side to the phase graph. The parser reads the canonical
`rationale.json` grounding first and falls back to the backend's Stage 2 artifact.
The compiled cost is also sampled over a cloud of plausible poses
(executed/candidate frames plus small joint-angle perturbations), with each penalized
pose shaded by penalty magnitude. This sampled **penalty field** remains the check of
what the executable cost actually does and works for cost shapes that cannot be
expressed as a declared bound. When the compiled cost reads exactly one named joint
feature, its sampled penalty region is also shaded across that feature's time graph.

Personas are selectable before starting a session, then locked for that
session. The edit/new/delete controls remain available (the bounds editor
supports `hidden` and `coupled` bounds over the shared joint features), and
editing the active persona immediately re-detects its live trajectory trigger.
Persona bounds are also editable **directly on the graphs**: drag the red
bound lines on the per-feature time graphs (handles on the right edge) or the
two handles on a phase plot's diagonal limit line; each graph has compact
controls to add/retype/delete bounds for that feature, and the phase-plot
column has a row to add a new coupled bound for any feature pair. Violations
(red trace segments, strips) recompute live while dragging, edits save
automatically after ~0.6 s, and if a base rollout is loaded the server
re-detects the feedback-trigger frame under the new bounds without re-running
the MPC. Custom personas persist to `--personas-file`; edits to built-in
personas last until the server restarts.

**Session replay.** Every session records a beat stream as it runs (the
recorder lives in `session.py`):

```
demo_runner_artifacts/<timestamp>_session_<persona>/replay/
  index.json              # {persona, started, beats: [{kind, time, file}, ...]}
  0000_trajectory.json    # {kind, time, file, persona: <snapshot>, data: <the UI payload>}
  0001_clusters.json
  ...
```

Each beat stores the exact payload the browser received, so replay re-feeds it
through the normal render path — **no MDM, LLM, or MPC calls**. Cluster choices
and generated costs remain provisional while they are explored. When a round is
committed or applied, replay records only its accepted root-to-leaf cluster
selection path and final generated cost; discarded picks, re-clusters,
backtracking, and cost previews are omitted. Beat kinds are `trajectory`
(rollout + trigger), `oracle`, `clusters`, `pick`, `cost`, and `round`. Open the
session dropdown and press **Replay** on a saved session, then step with the
prev/next buttons in the header; **Exit replay** reloads the page. Replay is
orthogonal to the mode — demo+replay is the presentation, dev+replay reads the
same beats with the graphs and code visible — and it never touches the live
session, so one can stay parked while you replay another.

**Fork &amp; continue live.** While replaying, press **Fork &amp; continue live**
in the header to branch off. This copies the recorded session into a fresh
`<timestamp>_session_<persona>` dir, resumes the copy as the active live session
(full context — corpus, committed rounds, unified cost), and hands control back
to the live UI so you can run new manual trajectories and corrections from that
accumulated state. The original recording is left untouched, so its replay still
reads cleanly. Like resume, a fork restores context but not the ephemeral MPC
trajectory, so you continue by starting a new rollout.

Two consequences worth knowing. Only sessions recorded after this feature exists
can be replayed: MDM samples are stochastic and were never persisted, so the
clusters you rejected cannot be regenerated after the fact — they survive only
because the beat recorded them. And each beat carries its own persona snapshot,
so a replay renders under the bounds that were in force when it was recorded;
editing the persona later will not silently redraw a past demo.


## Thanks
This repository is based on [python-starter](https://github.com/tomsilver/python-starter), which is a general starter repository (not limited to research project code).
