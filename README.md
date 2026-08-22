# Confidence-Aware Language Grounding

## Getting Started
Clone https://github.com/GuyTevet/motion-diffusion-model as `src/uncertain_feedback/motion_generators/mdm/motion-diffusion-model` and download the [required weights](https://github.com/GuyTevet/motion-diffusion-model?tab=readme-ov-file#mdm-is-now-40x-faster--04-secsample), [data](https://github.com/GuyTevet/motion-diffusion-model?tab=readme-ov-file#2-get-data) and [SMPL model](https://github.com/GuyTevet/motion-diffusion-model/blob/main/prepare/download_smpl_files.sh)

### Kinova Gen3 URDF

`env: real` and `sim_mannequin` with `robot: kinova_gen3` load
`~/kortex_description/robots/gen3_7dof_no_vision_robotiq_2f_85.urdf`, as does the
controller server's `config_tuned.yaml`. Generate it from
[ros_kortex](https://github.com/Kinovarobotics/ros_kortex) (`noetic-devel`), then
copy the `kortex_description` package to `~/kortex_description` — the URDF's
`package://` mesh URIs resolve against the home directory:

```
xacro kortex_description/robots/gen3_robotiq_2f_85.xacro \
  dof:=7 vision:=false gripper:=robotiq_2f_85 sim:=false \
  -o gen3_7dof_no_vision_robotiq_2f_85.urdf
```

Then set the six Robotiq finger joints (`finger_joint`,
`{left,right}_inner_knuckle_joint`, `{left,right}_inner_finger_joint`,
`right_outer_knuckle_joint`) to `type="fixed"`. Stock xacro emits them as
revolute, but `envs/real.py` treats *every* non-fixed joint as an arm DOF and
raises on the 13-vs-7 mismatch; pinocchio likewise needs `nv == 7` server-side.
Sanity check: at the `kinova_gen3` home configuration `tool_frame` sits at
`(0.576, 0.002, 0.434)` in the base frame.

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
Pass `--hml_stats_dir .../dataset/custom1` unless you are sure of what `dataset/HumanML3D`
currently points at — the default is that path, and `finetune_standing.sh` symlinks it to
whichever dataset is training. Every `custom1*` dataset carries the same stock `Mean.npy`/`Std.npy`.

Pose-estimation results are cached as `(N, 22, 3)` position arrays in `<frames_dir>/../mdm_cache/`
with a `_v<N>` version tag in the filename; bumping `_CACHE_VERSION` in `build_mdm_dataset.py`
(done whenever the pose-estimation/conversion changes) invalidates old entries automatically.
Untagged cache files predate the 2026-07-07 encoding switch and hold `(N, 263)` *features*
rather than positions — they are never read and can be deleted. Anything under
`data/mdm_cache.mirrored_bak_*` predates the 2026-06-29 chirality fix and is mirrored.

3b. (Optional) Build a speed-variant dataset instead — retimes each cached segment into
fast / normal / slow variants plus a conditional variant that slows while the wrist is
above the shoulder, with matching speed-language captions. Reads the `mdm_cache`
positions directly, so steps 1–3 must have run (and cached) at least once:
```
uv run python src/uncertain_feedback/data_collection/build_speed_dataset.py --output_dir ./speed_mdm_dataset/
```
`--caption_style` picks the speed language: `adverb` (default), `vivid`, or `none`. `none`
gives every variant the same speed-free caption, for runs where speed must come from MDM's
experimental `--speed_cond` scalar channel (`MdmMotionGenerator.generate_*(speed=...)`,
metres per frame) instead of the text.

3c. (Optional) Build a **correction-clip** dataset around one base trajectory — for a
checkpoint that is multimodal about a single scenario rather than a broad motion corpus.
Stage (a) rolls one naive MPC trajectory to a Cartesian goal, then for each run samples a
hidden comfort bound *anchored on that rollout* (so violation is guaranteed), replans from
the induced trigger step under the oracle cost, and writes an 8-frame naive history prefix
spliced onto a slice of the corrected continuation:
```
uv run python evaluation/generate_correction_clips.py --out_dir outputs/correction_clips --n_runs 0
```
`--n_runs 0` writes only the base artifacts — the naive rollout, the body geometry and an
empty manifest — and the labeling UI then plans each correction on demand as you press Next
(one oracle rollout, 2–13 s, prefetched while you type). Pass `--n_runs 32` instead to batch
them up front; both paths produce identical runs, since each is seeded on `(seed, index)`
rather than on a streaming generator.

Stage (a) **refuses to write into a directory that already holds a clip set** — it rewrites
`manifest.json` wholesale, so doing so would blank that set's captions and orphan the
labeling sessions underneath it. Pass a new `--out_dir` instead.

Only this step needs the GPU/MDM env — it loads the generator once for the start-pose
geometry and caches it to `geometry.npz`. Knobs:
`--seed`, `--trigger_window LOW HIGH` (accepted range for the naive rollout's first
violation), `--margin_range LOW HIGH` (radians the bound sits past the naive feature
value), `--correction_frames LOW HIGH` (corrected frames kept per clip, before the
prefix), and `--max_angle_delta`.

The default scenario is `evaluation/conf/mpc_demo_low1.yaml`: the `mpc_demo_base1` body and
goal, but the arm **starts resting low** (wrist ~0.29 m below spine3, elbow bent 0.6 rad)
rather than already held near shoulder height, so the reach the clips branch off is a full
lift. It is a separate file because `mpc_demo_base1.yaml` is the shared demo/comparison
scenario that `outputs/comparison_demo` and `outputs/base1` were generated against. The low
start is also simply better for this purpose: the longer reach evens out which anatomical
feature the sampled bound lands on — base1's short reach skewed hard toward two features.
Pass `--config evaluation/conf/mpc_demo_base1.yaml` for the old start (and scale
`--trigger_window` down to suit its shorter reach).

`--max_angle_delta` (default `0.00125`) overrides the config's action-sampling spread and
is **the one knob for how big and how fast a clip's motion is**. It is a std dev, not a
per-step cap, so it sets distance travelled per frame; because a clip is a fixed frame
budget, halving it halves both the speed and the ground covered. Measured on the low1 start:

| `--max_angle_delta` | reach | wrist path per clip | speed | clip frames |
|---|---|---|---|---|
| 0.01 (config's own) | 28 fr | — | — | too short to clip |
| 0.0025 | 85 fr (4.2 s) | 0.351 m | 0.0079 m/fr | 50–64 |
| **0.00125** (default) | 165 fr (8.2 s) | **0.186 m** | **0.0042 m/fr** | 50–64 |

Clip length and padding are unaffected, so lowering it buys smaller, gentler corrections
outright; the cost is generation time (~4–13 s per run instead of ~3–7 s, hidden by the
prefetch). `--margin_range` looks like a size knob but is **not** — it sets how far past the
naive value the bound sits, i.e. which way and how insistently the correction deviates, not
how far the arm travels in the window. Halving it moved wrist path by under a centimetre
while making corrections less distinct from the naive path, so it is left wide.

`--trigger_window` counts naive frames, so it must scale with `--max_angle_delta`:
`(12, 100)` suits the 165-frame reach, `(6, 50)` the 85-frame one. Pushing its top end near
the end of the reach starts producing continuations too short to fill the window, which get
padded by holding the last frame (recorded as `pad_frames`).

Clips are paced more slowly than the `0.01` demo the finetuned checkpoint is queried in;
MDM output is tracked as a *path* (playback advances on proximity, not on a clock), so the
pacing costs nothing downstream, but the pinned prefix a clip carries is slower than the one
inference pins.

Then hand-label. The browser UI serves every clip next to the bound that produced it and
writes typed captions straight back into the manifest (saves on blur), so nothing needs
editing by hand:
```
uv run python evaluation/label_correction_clips.py
# then: ssh -L 6768:localhost:6768 user@host  →  http://localhost:6768
```
**Every launch labels into its own session directory.** `--clips_dir` defaults to
`outputs/correction_clips`, so labeling normally needs no flags at all. The UI forks
`outputs/correction_clips/session_<timestamp>/` off the clip set — the base artifacts copied
in, an empty manifest, and the timestamp as the session's seed — and writes its runs and
captions there, so it never touches the set it came from or an earlier session. Runs are
seeded on `(seed, index)`, so the fresh seed is what keeps two sessions off the same sampled
bounds. A session directory is self-contained: stage (b) reads it exactly like a clip set,
and several of them combine into one dataset. Pass `--resume` with a session directory to
carry on captioning that one instead of starting another:
```
uv run python evaluation/label_correction_clips.py \
    --clips_dir outputs/correction_clips/session_20260817_2321 --resume
```

The UI shows **one run at a time**: three orthogonal projections (Front XY / Side ZY / Top
XZ), a scrubber, a caption box, and Prev / Next. Next saves the caption and advances,
planning that run if it does not exist yet; the run after it is prefetched in the background
while you type, so Next normally lands instantly (~0.1 s) instead of waiting on the rollout.
Generation is serialized, so a click that races the prefetch waits for the same result
rather than duplicating the work, and saving a caption never blocks on a rollout.

Playback covers the **whole** story in three colour-coded phases:

| colour | phase | frames |
|---|---|---|
| gray | naive approach | frame 0 → trigger |
| **blue** | **the training clip — this is what you caption** | `correction_frames` after the clip anchor (pale head = pinned prefix) |
| orange | rest of the oracle rollout, context only | everything after the trigger outside the clip |

So the branch is visible in context rather than starting 8 frames before it, and — since
clips end mid-reach by design — you can see where the correction was still heading instead
of guessing. The whole wrist path is drawn at all times, faint ahead of the playhead and
solid behind it, and a faint ghost of the trigger pose stays on screen so the departure
point is always in view.

**The clip window is draggable.** The sampled cut anchors at the trigger, but the
interesting motion is often further along the rollout, so the blue bar under the scrubber
can be grabbed by either end to resize or in the middle to slide anywhere along the motion.
Colours and the frame counter update live while dragging; on release the clip is re-cut on
disk and `window_violation` is recomputed from that run's own recorded bound. **No replan is
needed** — the naive rollout and oracle continuation are already saved, so a clip is just a
different cut of them, which is why dragging is instant while generating a new run is not.
The bar's pale head is the `n_prefix` pinned prefix (conditioning, not the described
motion); the solid blue is what your caption covers.

Anchoring later means the pinned prefix is recent *corrected* history rather than naive
history. That is still a valid clip: inference pins whatever the arm just did, so a prefix
taken from mid-correction matches the query distribution just as well.

The clip is clamped to `MIN_WINDOW`–`MAX_WINDOW` (42–180 corrected frames, so 50–188 with
the prefix) — above the t2m loader's minimum and below MDM's 196-frame cap. Dragging to the
very end holds the anchor back rather than padding, so a dragged clip is always real frames
(`pad_frames` 0).

Stage (a) writes **no video** — rendering was 3.5 MB of a 3.9 MB 32-clip set, against 436 KB
without it. The viewer reads only `manifest.json`, `naive.npy`, `geometry.npz` and the
per-run `clip.npy` / `continuation.npy`, reconstructing each preview through `motion_frames` at ~9–40 KB of JSON per
run, so it needs neither the MDM environment nor a GPU — including for the corrections it
plans on demand, since `clip_source_from_dir` rebuilds the rollout context from those same
files (`geometry.npz` carries the generator-decoded body for exactly this reason). A
bootstrapped set is 32 KB and each run adds ~20–32 KB (clip, features, and the full
continuation the preview's context phase needs). Runs whose continuation never reached
the goal are marked in the header; blank runs are skipped by stage (b).

Editing `outputs/correction_clips/manifest.json` directly works too. Each run records the sampled
`feature`/`bound_type`/`bound_value`, `trigger_step`, `clip_anchor` (where the cut sits —
equal to `trigger_step` until you drag it), `window_violation` (how much of the described
window still violates the bound) and `continuation_reach`. Skip a run by leaving
its caption empty.

`continuation_reach` describes the **whole** continuation, not the clip: a clip keeps only
`correction_frames` of it — a median 46% on the 32-clip demo set — so 30 of 32 clips end
mid-reach (median clip-final wrist distance 0.11 m against a 0.05 threshold) even though
their continuations reached. That is the intended shape, not a defect: at inference MDM
emits a `feedback.frames`-length correction that the MPC tracks and then finishes the reach
under its own cost, so a clip should teach a steering nudge, not a completed reach. On the
demo goal,
every `elbow_flexion upper_bound` run fails to reach — capping elbow flexion puts the goal
out of geometric range, so the MPC burns its full step budget — yet those clips carry as
much arm motion as the reaching ones and are still good training content. Judge a clip by
its video and `window_violation`; treat `continuation_reach: false` as a prompt to look,
not as a rejection.

Stage (b) encodes every captioned clip into HumanML3D format. `--clips_dir` takes **several
directories**, which is how separate labeling sessions become one training set — ids are
handed out across the whole run, so nothing collides:
```
uv run python src/uncertain_feedback/data_collection/build_correction_dataset.py \
    --clips_dir outputs/correction_clips outputs/correction_clips/session_* \
    --output_dir src/uncertain_feedback/motion_generators/mdm/motion-diffusion-model/dataset/correction_demo1 \
    --transplants 4
```
`--transplants N` adds N augmented copies of every captioned clip, each replaying that
clip's joint-space correction from a **randomly drawn** frame of the naive rollout: the
prefix becomes the arm's real history at that frame, and the described window is the same
deltas applied from there. This is what puts a behaviour labeled high up the reach in front
of the model with the arm low, which matters because the prefix is inpainted at inference —
the caption is the only other handle the model has. Draws are rejected and redrawn if they
leave the anatomical joint box or inflate the bounded feature's excursion past 3x (the
features are `arccos`/`arcsin` of rotated axes, so the same joint delta lands differently
from a different base pose). Measured on 31 captioned clips at `--transplants 4`: 0
rejections, and clips whose prefix sits in the top height band (+0.15 to +0.35 m) go from 2
to 57. The correction's *shape* transfers exactly; the absolute feature values it was
generated under do not, so caption a behaviour ("raise my arm a bit"), never a limit.
Transplants of one clip can land in different splits, which is one more reason not to read
anything into val loss at this dataset size.

Clips are encoded with `smpl_arm_aa_seq_to_hml263_frames` — the *same* function inference
uses to build its pinned prefix — so training clips and query-time prefixes share a body
and an encoding by construction. `--hml_stats_dir` defaults to `dataset/custom1_seatedcanon`,
**not** `dataset/HumanML3D` (that path is the fine-tune swap slot and holds whichever
dataset trained last).

Fine-tune on it from the **stock** checkpoint. `finetune_standing.sh` already resumes from
`./save/humanml_enc_512_50steps/model000750000.pt`, so pass no `--resume_checkpoint`
(MDM's argparse is last-flag-wins, so passing one would override it). Note the runner
hardcodes `--num_steps 5000`; a longer run needs an explicit override in the trailing args:
```
# ran 2026-08-18: matches the deployed lr/steps recipe — MODE COLLAPSED, see below
bash src/uncertain_feedback/motion_generators/mdm/finetune_standing.sh \
    correction_demo1 correction_demo1_lr1e7_10k 1e-7 \
    --num_steps 10000 --gen_during_training --start_pose mdm_sit_pose.pt --n_prefix 8

# ran 2026-08-18: the usable checkpoint
bash src/uncertain_feedback/motion_generators/mdm/finetune_standing.sh \
    correction_demo1 correction_demo1_lr1e5_5k 1e-5 \
    --gen_during_training --start_pose mdm_sit_pose.pt --n_prefix 8
```

**Pick the recipe by diversity, not by loss** (val splits contain transplants of training
clips, so val loss is meaningless here). Measured on left-wrist trajectories across the 6
`--gen_during_training` samples at every saved checkpoint:

| recipe | wrist path / frame | cross-sample spread | verdict |
|---|---|---|---|
| training clips (target) | 0.0039 m | — | — |
| `lr 1e-5 / 5k` | 0.0045 m | 0.205 m, no trend over the run | usable |
| `lr 1e-7 / 10k` | 0.0021 m | 0.316 → 0.023 m, monotone | collapsed — all samples identical |

10k steps x batch 8 over 123 examples is ~650 epochs on a set where every clip carries the
same caption, so even lr 1e-7 accumulates into drift toward the dataset mean. The
deployed-checkpoint recipe does **not** transfer to this dataset.

After training, flip `use_ema` to false in `save/<run>/args.json` if you will sample from
`sample_leftarm.py` (`mdm_api` is safe either way — `mdm_configs/mdm_config.yaml` overlays
it). Then point `consts.MDM_MODEL_WEIGHTS_PATH` at the checkpoint.

**Try it in the demo runner, on the scenario the clips were trained on.** Launch the demo
runner (see *Demo runner web tool*) and in the Scenario panel pick start pose
**`low1 (clip training start)`** and goal **`hits limit 1`** — that pair is exactly
`evaluation/conf/mpc_demo_low1.yaml`'s `arm:` and `cartesian.goals[0]` `[0.25, 0.3, 0.18]`,
verified against `outputs/correction_clips/naive.npy` frame 0. Type the correction
**`raise my arm a bit`** verbatim: every clip in the set carries that one caption, so it is
the only prompt the checkpoint knows. Two expected mismatches: the demo runs at
`max_angle_delta 0.01` while the clips were generated at `0.00125`, so the pinned prefix is
paced ~8x faster than any the model trained on; and the checkpoint's training body sits
0.060 m from `mdm_sit_pose.pt` (vs 0.190 m for `custom1_seatedcanon`, so this part is
*better* matched than the deployed checkpoint).

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
    --n_prefix 8 \
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

**Building the standing dataset.** Every clip in `dataset/custom1` has a seated body, so
fine-tuning on it teaches the checkpoint to snap standing bodies into sitting.
`graft_standing_dataset.py` rebuilds the same 44 clips on one static standing body sampled
from the base checkpoint, writing `dataset/custom1_standing` (49 frames per clip; `texts/`,
the splits and the stock `Mean.npy`/`Std.npy` are copied from `custom1` unchanged):
```
uv run python src/uncertain_feedback/motion_generators/mdm/graft_standing_dataset.py \
    --verify-dir /tmp/graft_verify
```
`--verify-dir` is optional; when given it writes a skeleton contact sheet plus
seated-vs-grafted comparison strips there. The script always prints the round-trip
deviations (left-arm joint positions, root height, foot contacts).

**Official-encoding dataset (use this one).** `custom1` was built 2026-06-29 with the old
homegrown HML263 encoding, so `custom1_standing` changes both the posture *and* the skeleton
(`uniform_skeleton` retargets onto the canonical t2m bone lengths).
`canonicalize_seated_dataset.py` writes `dataset/custom1_seatedcanon`: the same seated
motions, unchanged, pushed through the current `process_file` encoding so only the skeleton
differs. It doubles as the drop-in **official-encoding** training set — query-time pinned
frames are always official-encoding, so a checkpoint fine-tuned on `custom1_seatedcanon`
reads the arm rotations it is conditioned on (rot6d round-trip 0.018 normalized RMS) while
one fine-tuned on `custom1` does not (1.45 RMS). See `CODEBASE_MAP.md` §9 "Dataset encoding
provenance".
```
uv run python src/uncertain_feedback/motion_generators/mdm/canonicalize_seated_dataset.py \
    --verify-dir /tmp/graft_verify
```
It prints the spine3-relative left-wrist deviation and the pelvis heights before/after, and
with `--verify-dir` writes `04_seatedcanon_*.png` original-vs-re-encoded skeleton strips.

**Automated swap-and-train runner.** `finetune_standing.sh` does the whole dance above
(dataset swap, cache handling, training, restore) for one dataset with the `customv3_fixed`
recipe — batch 8, lr 3e-5, 5000 steps, save_interval 250, `--mask_frames --use_ema`, seed 10,
resumed from `humanml_enc_512_50steps/model000750000.pt`:
```
bash src/uncertain_feedback/motion_generators/mdm/finetune_standing.sh \
    [dataset_name] [save_name] [lr] [extra train_leftarm.py args...]
```
Defaults are `custom1_standing`, `custom_standing_v1` and lr `3e-5`; anything after the 3rd
argument is forwarded verbatim to `train_leftarm.py`. To save a sample video at every
checkpoint (one `model*.pt.samples/samples_00_to_02.mp4` per save interval, generated from the
deployed query pose so the samples reflect inference conditions):
```
bash src/uncertain_feedback/motion_generators/mdm/finetune_standing.sh \
    custom1_seatedcanon custom_seatedcanon_lowlr_v1 1e-5 \
    --gen_during_training --gen_num_samples 3 --gen_num_repetitions 2 \
    --start_pose demo_pose.pt --body_mode freeze
```
Generation adds ~15 s per checkpoint (50 diffusion steps) and reads the *test* split of the
swapped dataset for its prompts; it loads the raw (non-EMA) weights, matching inference. It
symlinks
`dataset/HumanML3D -> dataset/<dataset_name>`, sets the `t2m_{train,val,test}.npy` caches aside,
and restores everything via an `EXIT` trap (discarding any cache written during the swap so the
next run cannot inherit it). Preflight refuses to run if the dataset is missing, if its
`Mean.npy`/`Std.npy` differ from the live `HumanML3D` stats, if the backup name is taken, or if
`--save_dir` already exists (`train_leftarm.py` would otherwise silently train into `<name>_2`).
Expect ~10–15 min: ~7 min compute plus ~6 GB of checkpoint I/O.

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
Planner-state artifacts use the explicit 7-DOF layout
`[clavicle_rotvec(3), shoulder_rotvec(3), elbow_flexion(1)]`:
`executed_trajectory.npy`, `correction.npy`, and `interrupted_reference.npy`
therefore have shape `(T, 7)`.

All anatomical arm features use this same representation through
`planners/mpc/arm_features.py`. This is the single implementation used by MPC
feature costs, generated-cost helpers and summaries, simulated-user bounds,
trajectory-corpus CSVs, and the demo runner graphs/penalty field. In particular,
`shoulder_internal_external_rotation` is the signed twist of the anatomical
shoulder block `q[3:6]`; clavicle rotation `q[0:3]` does not contribute.
`shoulder_elevation` remains one of the five shared features.

Use the runner from the repo root:
```
uv run python src/uncertain_feedback/planners/run.py --mpc-config path/to/mpc.yaml
```

Controller settings live in the required YAML file passed with `--mpc-config`.
The initial whole-body HML pose defaults to `consts.MDM_START_POSE_PATH`
(`mdm_sit_pose.pt`) for every config; override it per-config with `pose:` in
the YAML only to deviate from that shared default, and the
initial left arm can be overridden with an inline 3×3 `arm:` list of
`[shoulder, elbow, wrist]` axis-angles. Runtime inputs still stay on the
command line: `--model-path`, `--arm`, `--text`, `--save`, `--live`,
`--frozen-body`, and `--interactive` (take each correction from the operator
mid-run instead of `--text` at `text_time` — see
[Interactive corrections with a live person](#interactive-corrections-with-a-live-person---interactive)).
`--pose` and `--arm` are still accepted as overrides for the YAML values.

There is one planner class (`ArmMPC`); its capabilities are selected by the
presence of top-level YAML sections, one per module slot:

| Section | Module | Absent means |
|---|---|---|
| `cartesian:` | Cartesian wrist-goal space (`goals`, `threshold`) | no goal phase (hold after feedback) |
| `feedback:` | MDM correction playback (`max_playback_delta`, `trajectory_fraction`, `frames`, `text_time`, `anchor_correction`), with an optional nested `uq:` layer (`diffusion_samples`, `n_clusters`, `clusterer`, `auto_cluster`, `scale`, `user_cluster`, `steering`) | no correction phase |
| `constraints:` | named feasibility constraints; `robot_ik:` (`max_residual`, `grasp_residual_frames`, `playback_stall_steps`) discards rollouts and playback frames the robot cannot track by continuation IK | unconstrained |
| `robot_actions:` | sample robot joint deltas instead of human-arm deltas (`max_joint_delta`, `joint_delta_std`, `infeasibility_weight`, `max_grasp_residual`, `grasp_residual_frames`) | human-arm sampling |

So the full method is `cartesian:` + `feedback:` (with `uq:`); the plain
goal-seeking baseline is `cartesian:` alone. `constraints:` and
`robot_actions:` are mutually exclusive (robot rollouts are feasible by
construction). The old `planner:` names are retired and rejected by the
loader.

Set the optional top-level `seed` key to control MPC action sampling. It defaults
to `0`; use another nonnegative integer to reproduce a different sampling
sequence.

### Motion-generation backend (`motion_generator`)

The text-to-motion backend is selected by the optional YAML key `motion_generator`:
- `mdm` (default, currently the only backend): the in-process Motion Diffusion
  Model (see Getting Started).

Backends expose the shared `MotionGenerator` interface, so any config with a
`feedback:` section works with any backend registered in
`MOTION_GENERATOR_BUILDERS`.

### Execution environment (`env`)

The execution environment realizing each commanded MPC step is selected by the
optional YAML key `env`:
- `kinematic` (default): the commanded arm configuration is achieved exactly
  (open-loop kinematic rollout, the original behavior).
- `sim_robot_visual`: same kinematic pass-through for the human arm, plus a
  visualization-only PyBullet scene (no physics) in which a Franka Panda is
  posed via IK each step so its end-effector holds a grasp point on the human
  forearm just distal of the elbow. The human renders as a posed SMPL body
  mesh fitted to the run's decoded initial body pose (the same fit the demo
  runner displays — sitting for the demo poses), with the left arm re-posed
  per step. The robot model is vendored from
  [limb-manipulation](https://github.com/empriselab/limb-manipulation) under
  `src/uncertain_feedback/envs/assets/panda/`.
- `sim_mannequin`: physics-based real-world proxy. The robot (a Franka Panda
  by default, or a Kinova Gen3 7-DOF with a Robotiq 2F-85 gripper via
  `env_params: {robot: kinova_gen3}`, loaded from
  `~/kortex_description`) starts grasping
  the forearm of the passive 4-DOF articulated mannequin left arm from
  [limb-manipulation](https://github.com/empriselab/limb-manipulation)
  (vendored under `src/uncertain_feedback/envs/assets/human/`, along with the
  torso/head and remaining limbs rendered as static visual context) via a
  fixed PyBullet constraint, and tracks the same forearm grasp-pose trajectory as
  `sim_robot_visual` under rate-limited position control with
  `stepSimulation` and gravity. The achieved arm configuration is measured
  back from the mannequin's link positions (direction-based retargeting onto
  the SMPL skeleton), so commanded and achieved configs diverge; the rendered
  frames overlay the commanded arm pose as a green ghost skeleton
  (capsules along shoulder→elbow→wrist plus elbow/wrist markers) so the
  tracking error is visible. The
  mannequin keeps the real hardware's link lengths, and its shoulder is
  fixed, so clavicle commands are not realized — runs converge slower than
  the kinematic envs (use more `steps`, see
  `plain_mannequin.yaml`). Tunable via the YAML
  `env_params:` mapping (forwarded to the env constructor):
  `robot` (`panda` or `kinova_gen3`),
  `robot_max_joint_delta` (per-step cap on each robot joint's travel, rad),
  `robot_base_offset` (robot base position relative to spine3, pybullet
  frame), `robot_joint_limit_padding` (rad, default 0; shrinks the robot's
  joint limits so commanded targets stay clear of them — set it slightly
  above the real controller's `joint_limit_padding_deg` when mirroring),
  and — with `robot: kinova_gen3` only — `real_mirror_host` /
  `real_mirror_confirm_start` (see below).

  **Real-arm mirroring** (`real_mirror_host`): when set, the sim robot's
  achieved joint configuration is forwarded each MPC step to a real Kinova
  Gen3 through an [emprise-gen3-controller](https://github.com/empriselab)
  ZMQ server, which shadows the sim in 1 kHz joint position mode (the
  server-side OTG interpolates between the sparse targets). The sim remains
  the source of truth — physics, mannequin read-back, and planning all stay
  simulated. With mirroring on, the sim robot plans against the controller's
  enforced joint-limit table (narrower than the kortex URDF's) so every
  mirrored command passes the controller's safety checks. The controller's
  client library is not a project dependency (the private repo only exists on
  rig hosts): check it out beside this repo and install it into the venv with
  `uv pip install -e ../../emprise-gen3-controller` after `uv sync`. Start the
  server first from the controller repo on the machine that reaches the robot:

  ```
  cd ~/emprise-gen3-controller
  .venv/bin/python scripts/launch_server.py --config config_tuned.yaml
  ```

  then run any `sim_mannequin` config with
  `env_params: {robot: kinova_gen3, real_mirror_host: "127.0.0.1"}` (see
  `plain_mannequin_kinova_mirror.yaml`). At the first
  MPC step the real arm closes its gripper, zeros its joints (upright
  reference pose), moves to the sim's grasp configuration, and opens the
  gripper there; the two moves are streamed slowly through position mode
  (0.15 rad/s peak joint speed, `_MOVE_SPEED_RAD_S` in
  `envs/real_mirror.py`) — with `real_mirror_confirm_start: true` (default)
  the operator must confirm each move and the gripper opening at the
  terminal; set it to `false` for headless/mock runs. On exit — including Ctrl+C, which is caught to halt the arm before
  the process dies — the arm is returned to high-level mode, holding its
  last position. The arm can also be zeroed on its own:

  ```
  uv run python src/uncertain_feedback/envs/zero_kinova.py [--host HOST] [--yes]
  ```

- `real`: the real world. The human arm state is *measured* rather than
  simulated: OptiTrack rigid bodies on the shoulder, elbow, and wrist are
  streamed in over NatNet (multicast) and converted to the planner's `(7,)`
  configuration each step, so the MPC closes the loop on the actual person.
  Three more rigid bodies supply the registration: one on the **robot base**,
  whose position fixes where the robot stands (replacing `sim_mannequin`'s
  hardcoded `robot_base_offset`); one on the **collar**, which fixes where the
  person is; and one on the **right collar** (read at calibration only), which
  closes the last degree of freedom — the person's facing (see below).
  IK is analytical, from `ssik`'s prebuilt Kinova Gen3 solver
  (`ssik.prebuilt.gen3_ik`); PyBullet runs in `DIRECT` for scene geometry only
  (no physics, no cameras). Each step continues the branch the arm is already
  on, and enumerates every analytical branch only when that continuation leaves
  the controller's joint limits — see **Inverse kinematics** below. The robot is
  commanded through the same `real_mirror_host` ZMQ path as above. Unlike the
  `sim_mannequin` mirror, the first MPC step moves nothing and touches no
  gripper: **the grasp must already be established** before the run, and it is
  *measured* rather than assumed (see below). `env_params:`
  `mocap_host` (the OptiTrack PC),
  `mocap_rigid_bodies` (Motive streaming ids for `robot_base`, `collar`,
  `collar_right`, `shoulder`, `elbow`, `wrist`),
  `mocap_hold_timeout` (s, default 0.5 — how long a tracking dropout is
  covered by holding the last valid pose before the run raises and halts),
  `recording` (replay a saved snapshot instead of the live streams — see
  **Recording the real environment** below; set this *or* `mocap_host`, not
  both),
  `live_view` / `live_view_fps` (live mesh window — see below),
  plus `robot`, `robot_max_joint_delta`, `robot_joint_limit_padding`, and
  `real_mirror_host` as above; `real_mirror_confirm_start` here only prompts
  once before the arm starts tracking (there is no move to confirm);
  `control_mode` (default `position_joint` — the emprise controller mode the
  arm tracks targets in. Set `compliant_joint` for joint-space impedance: the
  arm yields to the person instead of tracking stiffly, taking the same
  sparse joint targets. Stiffness gains are the server's, from its
  `config_tuned.yaml`). See `plain_real.yaml`.

  **The grasp is measured, not assumed — and re-measured every step.** Put the
  gripper on the person's forearm and close it *before* starting the run — jog the
  arm there however you normally would. At each MPC step the env reads the real
  arm's joint configuration, computes the gripper pose from it, and stores the
  rigid transform between that pose and the measured forearm frame
  (`MeasuredGrasp` in `envs/grasp.py`); the commanded arm configuration is then
  mapped through *that* transform. So the MPC tracks where the gripper actually is
  on the limb, instead of the nominal 15%-along-the-forearm, top-down grasp the
  simulated envs place their robot at. The reference frame is the FK's forearm
  bone rotation, which carries its own roll and cannot flip when the forearm
  passes through vertical — so the gripper takes the forearm's full rotation each
  step rather than only its direction change.

  Re-measuring rather than capturing once matters because the physical grasp is
  not rigid over a trajectory: the forearm turns and slides a little inside the
  fingers, and sleeve and skin move over the bone. The transform is the lever arm
  the gripper is swung on, so a stale one converts the commanded forearm motion
  into the wrong gripper motion. Two consequences:

  - The command can be an **absolute** gripper pose. A transform captured once
    could not afford that — SMPL FK and the tracked limb disagree on segment
    lengths, so an absolute target re-anchors on that read-back bias every step
    and integrates it into drift, which is why `sim_mannequin` (and this env,
    before) commands a target relative to the read-back. Re-measuring absorbs the
    bias into the transform instead, which makes the pose implied by the measured
    configuration *exactly* the current end-effector pose — the relative form
    reduces to the absolute one identically.
  - The **on-forearm check below runs every step**, so it doubles as a slip
    guard: a grasp that creeps along the forearm or off it walks the offset out of
    tolerance and halts the run.

  With `real_mirror_host: null` there is no real gripper to read, so the dry run
  puts the IK robot on the nominal grasp and calibrates from that; the plumbing
  is otherwise identical.

  The preview reports the **grasp error** over the plan before prompting —
  position and attitude between the gripper pose each commanded configuration
  asks for and the one the robot actually reached, as mean, max, and the step
  the max falls on:

  ```
  [real] grasp error over the plan: position mean 35.1 mm, max 68.6 mm at step 39; attitude mean 5.1 deg, max 8.4 deg at step 38
  ```

  Zeros mean the arm tracks the plan exactly. Non-zero has two causes, both
  worth knowing before approving: a stretch of the plan outside the padded joint
  box, where the IK deliberately spends attitude to hold position (so the
  gripper stays on the forearm), and a plan that moves faster than
  `robot_max_joint_delta` permits, which shows up as the arm lagging.

  **Inverse kinematics.** Solved analytically by `ssik`'s prebuilt Gen3 artifact
  rather than numerically. It works in `end_effector_link` relative to
  `base_link`, so a target leaves both frames it arrives in — the PyBullet world
  the scene is built in (the robot stands at its *measured* base pose) and the
  `tool_frame` between the gripper's fingers where the grasp is measured, 12 cm
  further along. Two layers:

  - **Continuation** (the normal case, ~0.3 ms). `ssik` Newton-continues from
    the arm's current configuration and rejects its own step if it would land on
    a different branch. This comes first because the Gen3 is redundant: a pose
    has a whole self-motion manifold of exact solutions, so *which* one a solver
    returns is not pinned down by the pose, and taking whichever scores best
    walks the gripper across the person for nothing — measured over one plan it
    moved the solution up to 1.5 rad in a step against a
    `robot_max_joint_delta` of 0.01.
  - **Enumeration**, when continuation leaves the controller's padded joint box.
    `ssik` returns *every* analytical branch ranked against the current
    configuration, so the nearest reachable one is known outright instead of
    searched for.

  Only when no branch fits the box does anything numerical run: a bounded
  least-squares that keeps position and spends orientation
  (`_IK_ORIENTATION_WEIGHT_M_PER_RAD`), matching what the physical grasp does —
  the gripper stays clamped on the forearm and the wrist carries the miss. Its
  residual uses `gen3_ik.fk`, not PyBullet's: PyBullet holds link state in
  single precision, so differencing it over the optimiser's ~1e-8 step returns a
  Jacobian that is mostly rounding error.

  `ssik` has no released wheel for Python 3.10 (every published release is
  cp311+), so `pyproject.toml` pins the tested commit from `main`, where the
  3.10 floor landed. It asks for `numpy>=1.26`, overridden to the MDM pin
  (`numpy<1.24`) — its extensions are built against the numpy 2 headers, whose
  ABI runs back to 1.19.

  **Live view** (`live_view: true`, the default in the real config). Opens a
  PyBullet GUI window showing the scene the MPC is solving in: the measured
  person as a posed SMPL **mesh** (same fit the demo runner shows) and the
  **robot's own URDF meshes**, both in the registered frame. It comes up right
  after registration — before any command is sent — so a bad registration (person
  or robot in the wrong place, arm on the wrong side) is visible while nothing is
  moving. Requires a display; set `live_view: false` when running headless over
  ssh. Closing the window mid-run does not end the process: the env reconnects
  its pybullet backend headless, restores the robot's joint state, and the run
  continues on the real robot without the drawing.

  **The planner's skeleton** is drawn inside the person in orange: the five joints
  its FK returns and its costs read — spine3, collar, shoulder, elbow, wrist — as
  bones and joint balls (`ArmSkeletonBody`). The body around it is a shaped,
  skinned surface *posed from* that chain; this is the chain, so the two agreeing
  is the check that the mesh is not telling you a story. It is real geometry, not
  debug lines, so it also survives into `getCameraImage` screenshots and videos,
  and it is cheap enough to refresh every step while the mesh is rate-limited —
  during a run the skeleton is live even when the body around it lags. The person
  is therefore drawn translucent (`BODY_XRAY_COLOR`); an opaque body would simply
  hide the thing being checked. Lower that alpha for a fainter body, or raise it to
  1.0 to go back to a solid person and give up seeing the chain.

  **Goal ghost.** The configuration the run drives toward is drawn in the same
  window as a **translucent green arm** (`env.show_goal`, wired up in
  `build_run`), so the gap between the person's own mesh and the ghost *is* the
  remaining error. Only the arm is drawn: it is all the MPC controls, so a
  whole-body ghost would coincide with the person's mesh everywhere else and
  z-fight into speckle. A Cartesian goal fixes the wrist alone, so the posture
  shown is the nearest configuration reaching it (`q_reaching_wrist` in
  `kinematics.py`, a wrist-position least-squares fit pulled toward the arm's
  current configuration); joint-space configs pass their goal `q` straight
  through. It is set once, from the first goal, and does not follow later queued
  goals or MDM corrections. Transparency only renders in the GUI window — the
  offscreen TinyRenderer used by `getCameraImage` ignores alpha.

  PyBullet cannot update a mesh's vertices, so each human pose replaces the whole
  6890-vertex body — ~140 ms in the GUI, during which the MPC step is blocked and
  the mocap receive thread is delayed toward `mocap_hold_timeout`. The mesh is
  therefore rate-limited by `live_view_fps` (default 5); the robot is re-posed
  every step regardless, since `resetJointState` costs nothing. Raising
  `live_view_fps` buys smoother human motion at the cost of a slower control loop
  — and above ~10 it starts tripping `MocapStaleError`.

  **Why the right-collar body matters.** The person's orientation relative to
  the robot is one unmeasured degree of freedom, and getting it wrong rotates
  every measured bone direction (with `sim_mannequin`'s `166 deg` base yaw
  carried over, the *left* arm landed on the person's right). The left→right
  collar line is the torso's mediolateral axis — requiring its measured
  direction to match the start pose's *solves* the registration yaw instead of
  assuming it, with no assumption about the arm. The robot is then loaded into
  the scene at that solved yaw, so the scene and the measured directions share
  one frame. Only the yaw is fitted: mocap, the Kinova base, and PyBullet are
  all Z-up, and any leftover angle is measurement noise in the collar bodies'
  heights, which must not tilt the world. `mocap/monitor.py` prints the solved
  yaw, and a near-vertical collar axis (which would leave the yaw undetermined)
  is rejected outright. Mount the two collar bodies so their pivots straddle
  the sternum symmetrically: the line between them *is* the measured facing,
  so a right body sitting forward or back of the left biases the yaw by that
  offset angle.

  **Where the person is, is measured too.** The registered PyBullet frame *is*
  the mocap world turned by the solved yaw, so the person's torso is anchored on
  their **measured collar** and the robot on its measured base. Neither is a
  constant of the config: sit somewhere else next run and the whole body moves
  there, while the bolted-down robot stays put. The config's start pose supplies
  the torso's shape and orientation only — its position is discarded. Cartesian
  goals are spine3-relative, so they follow the person, and the anchor is frozen
  for the run, which is what `MpcCostContext` requires. A run must therefore read
  the anchor back from `env.pose_context()` after `initial_q` (`planners/run.py`
  does); planning against the config's `spine3_pos` would put the goals on a
  torso that is not where the person is.

  Within a run, only bone *directions* come from mocap: they are re-anchored at
  the frozen collar and rescaled to SMPL bone lengths, so mid-run torso
  translation is deliberately not tracked — the registration yaw is likewise
  solved once, so a person who rotates in their seat drifts out of registration
  silently.

  **The run starts from the measured arm configuration.** `RealEnv.initial_q`
  registers against the person before planning and hands the planner the pose
  they are actually in — it prints
  `[real] waiting for mocap rigid bodies [...]` first, so a run stalled on
  untracked markers is visible instead of looking like a hang. The arm they
  start with need not match the config's
  `arm:` — every slot, clavicle included, is overwritten by mocap.
  Registration moves nothing; the robot still takes its grasp on the first MPC
  step, at the forearm's measured position.

  **Motive setup.** In Motive's Streaming pane set **Local Interface** to the
  OptiTrack PC's LAN address (on loopback the stream never leaves that
  machine), transmission `Multicast` on `239.255.42.99`, command port `1510`,
  data port `1511`, **Z-up**, and enable **Rigid Bodies**. Keep the up axis Z:
  mocap, the Kinova base, and PyBullet then share one vertical, which is what
  lets the registration be a single yaw; a Y-up stream would only add a fixed
  conversion and another place for a sign to be wrong.

  The **robot base body's orientation must be aligned to the Kinova base frame**
  (`+x` forward, `+z` up — `+x` is the Gen3's reach direction: at the repo's
  home configuration the gripper sits at `(0.576, 0.002, 0.434)` in the base
  frame). It is used *only* for the robot's facing in the scene, never for the
  measured bone directions, so a misaligned plate leaves the person correctly
  placed while the robot is loaded rotated away from reality — IK then solves
  against the wrong pose, with nothing catching it, so check it in the dry run.
  Create the six rigid
  bodies and put their streaming ids in `mocap_rigid_bodies`. NatNet 3 or
  newer is required (verified against Motive 3.0.3.1 / NatNet 4.0).

  **Verify before anything moves**, in this order:

  ```
  # 1. mocap only, no robot: validity, frame rate, derived q, rendered arm
  uv run python src/uncertain_feedback/mocap/monitor.py --host 192.168.2.243 \
      --base-id 1 --collar-id 2 --collar-right-id 6 --shoulder-id 3 \
      --elbow-id 4 --wrist-id 5 --video mocap_check.mp4

  # 2. full loop on the live person with real_mirror_host: null — IK is solved
  #    but no command reaches the arm
  uv run python src/uncertain_feedback/planners/run.py \
      --mpc-config src/uncertain_feedback/planners/mpc/configs/plain_real.yaml \
      --env-video real_rollout.mp4

  # 3. the same with real_mirror_host set, after taking the grasp by hand and
  #    launching the controller server
  ```

  `--env-video` renders the *measured* arm trajectory through
  `ArmVisualizer.render_rollout_video()`.

  **Recording the real environment.** `env: real` senses the world through
  exactly two channels — the OptiTrack stream and the Gen3's measured joint
  state — so both can be captured to a file and replayed later without the lab.
  Reading the arm is passive: nothing is commanded, no mode switched, the
  gripper untouched.

  ```
  uv run python src/uncertain_feedback/envs/record_real.py \
      --host 192.168.2.243 --controller-host 127.0.0.1 \
      --require-bodies 1 2 3 4 5 6 --seconds 30 \
      --out real_recordings/lab.npz
  ```

  It stores every streamed rigid body (not only the configured six) plus the
  full robot state per frame, and blocks until `--require-bodies` are all
  tracked so the window does not open on a dropout. Point `env_params.recording`
  at the file to replay it — see
  `ik_gated_replay.yaml`, which needs no hardware, no
  display, and no network:

  ```
  uv run python src/uncertain_feedback/planners/run.py \
      --mpc-config src/uncertain_feedback/planners/mpc/configs/ik_gated_replay.yaml
  ```

  Everything measured is the real thing: the registration and its yaw, the
  person's segment lengths, the robot's base pose, the grasp read off the
  recorded joint configuration, and therefore the analytical IK and the
  feasibility gate. A relative `recording` path is resolved from the repo root,
  because loading MDM chdirs the process into its submodule.

  **What replay does not reproduce is the loop closing on the person.** The
  mocap side is playback: `execute` returns the *recorded* configuration, so the
  person does not respond to the robot and the human arm does not advance no
  matter what is commanded (the robot side does move — `ReplayMirror` is an
  ideal tracker seeded at the recorded joints). A full `run.py` replay therefore
  exercises every solve but runs out its step budget in place. For a rollout
  that actually progresses, use the planner's offline stand-in — the same
  `RobotPlanPreviewEnv` rollout the plan preview runs
  (`_rollout_gated_reference_trajectory` in `planners/run.py`), which drives the
  arm kinematically from the measured snapshot and reaches the Cartesian goal.

  **Rendering that rollout with meshes.** `RealEnv.save_scene_video(human_q,
  robot_q, path)` renders the live view's scene offscreen — the person as an
  SMPL mesh shaped to their *measured* arm lengths, the Gen3 at its measured
  base, the green goal ghost — so a headless replay still shows whether the
  robot is where the person is, which a stick figure of the joint angles cannot.
  Pass the planner's own robot joints where it has them (the gated and
  robot-action rollouts report them per step); pass `None` for a human-space
  rollout and the robot is chased through the same IK `execute` uses. Needs no
  display: pybullet renders it in `DIRECT`. The arm chain is in the scene but
  the offscreen renderer draws the body opaque, so use `save_video` to see the
  chain itself.

  **The plan is previewed before anything moves** (`preview_plan: true`, the
  default; needs `live_view`). After registration the run rolls the same planner
  — same action space, same feasibility gate, against the same robot IK and
  padded joint box — with the same goals and costs, kinematically from the
  *measured* start configuration, drawing each step in the live view the moment
  its MPC solve finishes — the
  person's mesh at the registered anchor, posed by the same FK the planner costs
  against, the Gen3 at its measured base, the gripper carried by the grasp
  measured off the real arm — then asks:

  ```
  [real] planning the preview — each step is drawn in the live view as it is solved (robot + arm chain every step, mesh at live_view_fps)
  [real] preview done. Enter = run on the robot, n = abort:
  ```

  Answering `n` ends the run before the first command. Drawing goes through the
  same rate limit as the run's live view — the robot and the planner's arm chain
  move every step, the person's mesh (~140 ms per re-pose) refreshes at
  `live_view_fps` — so the preview runs at roughly planning speed, and the
  animation *is* the planning progress: a plan going somewhere unacceptable is
  visible without waiting for the rest of the rollout. The preview is a
  kinematic rollout, so it assumes the arm tracks each command exactly; the real
  loop closes on mocap and will drift from it.

  The drawn arm is the planner's arm: `_start_live_view` hands the calibrated
  segment lengths to `SmplMeshCache`, which fits `betas` to them, and every mesh
  pose is shifted so its shoulder lands on the FK's — measured at under a
  millimetre from collar through wrist on subjects 5–12% off SMPL neutral (it was
  ~4 cm at the wrist before the shape fit, against a 0.05 m
  `cartesian.threshold`). Its *torso*, legs, and head are still the config
  `pose:`: nothing measures them, the MPC does not use them, and the shape fit
  leaves them about a centimetre off the anchor the goals hang from.

  Which rollout previews the run is `_select_preview_rollout` in
  `planners/run.py`: configs with `constraints:` preview constrained, configs
  with `robot_actions:` preview the robot-action solve, plain Cartesian
  configs preview the human-action rollout — the preview always carries the
  same feasibility constraints and action space as the run. The robot-action
  and constrained rollouts run against `RobotPlanPreviewEnv`, an offline
  double of the env frozen at the measured state that delegates exact IK back
  to the env itself; they report their own robot joints per step, so the
  window shows the planned robot rather than a second IK chasing the arm. A
  feedback correction does not exist before the run — the user has not spoken
  yet — so only the goal-seeking phase is previewed, and configs without a
  `cartesian:` section skip the preview.

  The grasp is re-measured every step from the real ee pose and the measured
  forearm frame, and the gripper pose it implies rides the forearm rigidly —
  rotation included. The IK tracks that rigid pose exactly while it is
  reachable (the joint limits planned against are the emprise controller's
  table, which now carries the kortex URDF hard stops minus
  `robot_joint_limit_padding`); if a stretch of the plan leaves the reachable
  set, position wins — the gripper stays on the forearm and the wrist attitude
  carries the miss, rather than the robot drifting off the arm. The
  measured grasp itself is used as given — nothing checks that it landed on the
  forearm. A wrong registration yaw, a base plate misaligned in Motive, or a
  mirror arm that was never actually placed on the person (e.g. a fresh sim
  controller) bakes that offset into a long lever arm the commands then swing
  through, so verify the registration in the dry run (`real_mirror_host: null`)
  — which stages the robot at the nominal grasp — before commanding the arm.

Every planner takes the env at construction (defaulting to `kinematic`), and
`step` returns the configuration the env actually achieved. Envs also expose
`visualize()` (image of the current state) and `save_video()` (video of the
whole executed trajectory); `sim_robot_visual` frames stack a front view over
a top-down view. Pass `--env-video out.mp4` to `run.py` to write the env's
video at the end of a run:

```
uv run python src/uncertain_feedback/planners/run.py \
    --mpc-config src/uncertain_feedback/planners/mpc/configs/plain_sim.yaml \
    --env-video env_rollout.mp4
```

(`plain_sim.yaml` is the Cartesian no-MDM config with
`env: sim_robot_visual`; `plain_mannequin.yaml` is the
same run with `env: sim_mannequin`;
`plain_mannequin_kinova.yaml` is the mannequin run
with the Kinova Gen3;
`plain_mannequin_kinova_mirror.yaml` additionally
mirrors the sim robot onto the real Gen3;
`plain_real.yaml` is the mocap-closed-loop real run;
`mdm_llm_real.yaml` is the *full method* on that rig (see
below); the `--env-video` flag works with any env).

### The `robot_ik` feasibility constraint

A `constraints: robot_ik:` section keeps sampling human-arm deltas, but gates
each rollout's first `grasp_residual_frames` frames with the robot environment's
own IK. On `env: real`, this is the same analytical Gen3 branch continuation
used for execution — one vectorized Newton batch over every sample, seeded
sequentially through each rollout and filtered against the controller's padded
joint-limit box. A rollout is discarded when its remaining gripper pose error
exceeds `constraints.robot_ik.max_residual` (metres + radians). Each sample set
includes a zero-motion hold, so an all-infeasible draw holds instead of
selecting the least-infeasible motion.

**The gate continues; it does not enumerate.** Execution's
`solve_robot_ik_exact_batch` falls back to enumerating every analytical branch
for candidates continuation cannot place, because execution has to command
something and the nearest reachable branch is the best available. The gate uses
`track_robot_ik_batch` instead, which stops at continuation, for two reasons. A
branch change is an *exact* solution, so the residual check cannot fault it, yet
it sits far enough away that the arm spends tens of steps at
`robot_max_joint_delta` reaching it with the grasp wrong the whole way — a
measured case needed 0.78 rad, or 78 steps at the default cap. And enumeration
is serial: one call costs about as much as the entire vectorized continuation
over all 1000 samples, and it is paid on exactly the candidates about to be
discarded. Gating on the enumerating solve made the cost scale with
*infeasibility*, so a plan pressed against a limit went from 16 ms to tens of
seconds per frame and stayed there, since the resulting hold keeps the arm in
the region that caused it:

| samples continuation rejects | gate cost (continuation) | gate cost (with enumeration) |
| --- | --- | --- |
| 0 / 1000 | 16 ms | 16 ms |
| 62 / 1000 | 21 ms | 1.6 s |
| 814 / 1000 | 32 ms | 26 s |
| 983 / 1000 | 78 ms | 41 s |

The gate does not apply `robot_max_joint_delta`; that remains an execution-rate
cap rather than an IK joint limit. The preview *does* apply it when carrying its
robot along, as execution will, so a joint-limit-feasible plan can still show
transient robot lag there if it asks for faster motion than that cap permits.

The preview runs this constrained planner (`preview_plan`, below). It has to:
previewed unconstrained it drew the run walking into exactly the poses the
constraint discards, ending with the gripper visibly off the forearm — a
trajectory the run would never take.

```
uv run python src/uncertain_feedback/planners/run.py \
    --mpc-config src/uncertain_feedback/planners/mpc/configs/ik_gated_real.yaml
```

This config sets `control_mode: compliant_joint`, so the arm tracks the plan
with joint-space impedance — it yields when the person resists rather than
wrestling the grasp. Expect the measured joints to lag the commanded targets
more than under `position_joint`; `_drive` re-solves from the measured state
each step, so the loop self-corrects, but the per-step grasp-error report reads
slightly optimistic.

### Full method with the IK screens

Adding a `feedback:` section (with `uq:`) to a constrained config gives the
full method with MDM/UQ/LLM corrections on top of the constraint. The goal
phase is the constrained sampling above, unchanged. MDM playback stays
direct — rate-limited frames, no sampling — but MDM knows nothing about the
robot, so an unreachable frame would otherwise land in execution's IK
fallbacks: the branch enumeration (an exact solution up to ~1.5 rad away that
the arm then chases for tens of steps with the grasp wrong throughout) or the
position-priority miss (attitude twisted against the forearm). Playback is
therefore screened with the same continuation IK at three points, all against
the live measured grasp and robot state:

- **Push**: when a correction is queued, its frames are walked sequentially
  through continuation IK from the current robot joints; unreachable frames
  are dropped with a `[push] dropped n/m ...` report, and playback follows the
  reachable ones. (The walk advances only through kept frames — the ones
  playback will actually visit.)
- **Step**: each rate-limited playback step is checked before it is commanded
  (~0.3 ms) and held when continuation cannot place it, so `_drive`'s own solve
  is guaranteed to succeed by continuation — neither fallback ever fires
  during playback.
- **Stall**: the cursor normally advances only once the measured arm reaches
  the frame, so a frame the arm cannot reach — or settle on, under compliant
  control — would hold the run forever. Closest approach to the frame is
  tracked instead (monotone, so mocap jitter cannot reset the counter), and
  after `playback_stall_steps` steps without progress the frame is skipped
  with a `[playback] frame i stalled ...` log. A person actively resisting
  reads as a stall too — the arm yields and moves on rather than pressing.

The executed motion is the trajectory's *reachable shadow*: exact tracking
wherever the correction is feasible, decelerate-hold-skip across the stretches
it is not. `mdm_ik_gated_real.yaml` is
`mdm_llm_real.yaml` with the `constraints: robot_ik:` keys and
`control_mode: compliant_joint` (any residual model mismatch becomes a bounded
gentle pull, not a wrench — verify the server-side gains in
`emprise-gen3-controller`'s `config_tuned.yaml` with the mannequin held before
a live run):

```
uv run python src/uncertain_feedback/planners/run.py \
    --mpc-config src/uncertain_feedback/planners/mpc/configs/mdm_ik_gated_real.yaml \
    --interactive --env-video real_gated_full_method.mp4
```

Stage it like every real run: mocap monitor, then the whole loop with
`real_mirror_host: null` (type a correction and check the `[push]` and
`[playback]` reports with nothing commanding the arm), then live — a
trivially-reachable correction first, then a deliberately over-extended one to
watch drop → hold → skip. Two rig-tuning notes: under `compliant_joint` the
measured arm settles short of commands, so if steady-state deflection under
load exceeds `max_playback_delta` (0.002) every frame finishes by stall-skip —
loosen it if the skip logs say so; and a fully-unreachable correction costs at
worst `feedback.frames × playback_stall_steps` steps of the session's `steps`
budget.

### The robot action space (`robot_actions:`)

A `robot_actions:` section switches the same Cartesian planner to sampling
**robot joint deltas** instead of human-arm deltas:
every rollout is robot-feasible by construction, each rolled-out ee pose is
mapped through the rigid measured grasp back to a human arm configuration for
the (unchanged) human-arm costs, and the best joint target is sent to the robot
directly — no grasp FK, no IK, no per-joint delta clipping. It requires
`env: sim_mannequin` or `env: real`, and takes five keys:
`max_joint_delta` (per-step inf-norm cap on sampled joint deltas, rad),
`joint_delta_std` (sampling-noise std around the warm-started previous
plan, rad; keep it well below the cap or the uniform rescale drowns the warm
start in noise — `null` defaults to a third of the cap),
`infeasibility_weight` (soft penalty on motions the grasp cannot
transmit), and `max_grasp_residual` / `grasp_residual_frames` — a hard gate
that discards any rollout whose first `grasp_residual_frames` frames exceed
`max_grasp_residual` (metres of elbow displacement + radians of untransmitted
roll per frame). Only the leading frames are gated — they are what actually
gets executed; the tail is re-solved every step and stays soft-penalized. The
residual floor scales with `robot_actions.max_joint_delta`, so tighten or loosen the
gate together with the sampling scale; a tighter gate preserves the grasp more
strictly at the cost of slower goal progress (a very large value disables it).

```
# sim rehearsal (Kinova mannequin, physics)
uv run python src/uncertain_feedback/planners/run.py \
    --mpc-config src/uncertain_feedback/planners/mpc/configs/robot_mannequin_kinova.yaml \
    --env-video robot_action_sim.mp4

# real rig, same dry-run-first protocol as plain_real.yaml
uv run python src/uncertain_feedback/planners/run.py \
    --mpc-config src/uncertain_feedback/planners/mpc/configs/robot_real.yaml \
    --env-video robot_action_real.mp4

# full method (MDM/UQ/LLM corrections) with the robot-action sampler
uv run python src/uncertain_feedback/planners/run.py \
    --mpc-config src/uncertain_feedback/planners/mpc/configs/mdm_robot_real.yaml \
    --interactive --env-video robot_action_full_method.mp4
```

MDM playback with `robot_actions:` keeps its rate-limited cursor, but
each playback frame is tracked with the robot-action solve rather than commanded
through IK, so its `feedback.max_playback_delta` is looser (0.02) than the
human-action config's 0.002 — the robot-space tracking has a noise floor the
cursor must be able to cross.

The plan preview on `env: real` rolls out the *actual* robot-action solve
against a kinematic stand-in of the measured rig (robot chain, measured grasp,
joint state — see `envs/robot_preview.py`), so the animation poses the robot at
its planned joints and the robot and arm stay consistent by construction. The
human-action configs keep the previous preview, whose robot chases the
human-space plan through IK and can visibly lag it.

### Interactive corrections with a live person (`--interactive`)

`env` is orthogonal to the planner modules, so the full method — MDM
correction, UQ clustering, one LLM cost per round — runs on the real rig by
config alone: `mdm_llm_real.yaml` is
`plain_real.yaml`'s env with a `feedback:` section (uq
nested) and the LLM-cost keys.
Prefer `mdm_ik_gated_real.yaml` (previous section) when the
robot is in the loop — same method, but every robot-facing step is IK-screened
so unreachable MDM frames are dropped, held, or skipped instead of handed to
execution's IK fallbacks.

What a config cannot supply is *when* a real person wants a correction and *what
they say*: a scripted run injects one preset `--text` at `text_time`. Pass
`--interactive` and both come from the operator instead:

```
uv run python src/uncertain_feedback/planners/run.py \
    --mpc-config src/uncertain_feedback/planners/mpc/configs/mdm_llm_real.yaml \
    --interactive --env-video real_full_method.mp4
```

Press **enter** at any point to pause at the next MPC step, then type what the
person asked for (typing the correction directly and pressing enter does both at
once). The run then does exactly what it does in a scripted round — MDM samples
from the arm's *current* pose, the cluster picker opens, an LLM cost is generated
and stacked — and resumes tracking the chosen correction. Repeat as often as the
person speaks; `steps` bounds the whole session. `text_time` is ignored, and
`--text` with it.

While paused nothing is commanded, so the arm stays where the controller last put
it — the person is held, not released. Enter pressed during the pause queues the
*next* pause rather than being swallowed, so an accidental double-press costs one
extra round. `--interactive` requires an MDM-backed planner and is rejected
before the env is built, i.e. before anything touches the hardware.

Two knobs matter more here than in sim:

- **`feedback.uq.auto_cluster` must stay unset** for the person to choose among the
  clusters — that is the point of sampling several. With it set, the round takes
  that cluster silently.
- **`max_playback_delta`** (0.002 in the real config, against `max_angle_delta`
  0.001) bounds how far the commanded configuration may lead the *measured* arm
  while the correction plays back. Playback advances a frame only once the arm
  reaches it, and the env's `robot_max_joint_delta` clips every robot joint
  besides, so raising it makes corrections finish sooner rather than making the
  robot lunge — but it is the first thing to re-tune on the rig, together with
  `steps` (3000, since every correction is traversed at those caps).

Stage it exactly like the no-MDM real run: mocap monitor first, then the whole
loop with `real_mirror_host: null` so MDM, the picker, and cost generation all
run against the live person while no command reaches the arm, then set the host.
The generation stalls are worth knowing before someone is in the gripper: 500
diffusion samples plus an LLM round pause the loop for tens of seconds with the
arm held. Lower `feedback.uq.diffusion_samples` or set `llm_cost.enabled: false` to
shorten them.

### Simulated user (`user:`)

Every run loads a simulated care recipient alongside the pose, selected by the
optional YAML key `user:` (default `unrestricted` — no movement restrictions).
Restricted personas (`adhesive_capsulitis`, `elbow_contracture`, `painful_arc`,
`stroke_flexor_synergy`, `triceps_long_head_contracture`,
`biceps_long_head_contracture`, `brachial_plexus_mechanosensitivity`,
`out_of_synergy_reach_preference`, `cross_body_pain`,
`morning_shoulder_stiffness`, `spastic_elbow_flexors`;
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
- `feedback.uq.user_cluster: true` delegates UQ cluster selection to the user (it picks
  the most comfortable cluster mean), taking precedence over `feedback.uq.auto_cluster`
  and the interactive picker.

Beyond the five anatomical position features, hidden bounds may reference each
feature's velocity (`<feature>_velocity`, rad/s; `spastic_elbow_flexors` caps
elbow extension speed as a function of elbow flexion) and the session clock
(`time_of_day`, hours; `morning_shoulder_stiffness` limits shoulder elevation
before 11:00). Time-conditioned personas need the optional YAML key
`simulated_user.time_of_day` (hours in `[0, 24)`, default unset = untimed
session), which both `planners/run.py` and the demo runner pass into the cost
context.

The same hidden bounds are the evaluation ground truth for the method-level
experiments that will live in the repo-root `evaluation/` directory.

### MDM + UQ Config
Save as `src/uncertain_feedback/planners/mpc/configs/mpc_mdm_uq.yaml`:
```yaml
steps: 750
horizon: 10
n_mpc_samples: 512
seed: 0  # reproducible MPC action sampling
max_angle_delta: 0.0025
pose: "src/uncertain_feedback/motion_generators/mdm/demo_pose.pt"

feedback:
  trajectory_fraction: 1.0
  frames: 50
  uq:
    diffusion_samples: 128
    n_clusters: 3
    clusterer: agglo_end_pose  # kmeans_end_pose | agglo_end_pose | agglo_path_pca | agglo_t2m
    auto_cluster: null
    scale: 1.0  # default motion-magnitude scale for the chosen cluster
    steering:
      mode: "cg"            # cg (default) | resample | off
      resample_steps: [15, 25, 35, 45]
      temperature: 0.5        # resample only
      guide_from: 10          # cg only
      guidance_weight: 1.0e5  # cg only

cartesian:
  goals:
    - [0.3, 0.5, 0.0]
  threshold: 0.05
```

`feedback.uq.clusterer` selects the clustering method (used by the demo runner;
the MPC planner keeps its injected clusterer):

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

#### Steering diffusion sampling toward the user's cost (`feedback.uq.steering`)

`mode: "cg"` (the default, and what every checked-in config ships) biases
sampling toward the configured simulated user's hidden bounds, so the candidates
that reach clustering already respect the user's constraints; `mode: "off"`
samples MDM unsteered. The cost is compiled from the persona's
`elbow_flexion` / `shoulder_elevation` bounds and read off each denoising step's
`x̂0` prediction; `JointBoxLimit` and bounds on other features are skipped (named
in a log line). Steering is implemented by the MDM backend; with any other
backend the run logs a skip and samples unsteered, as it does when the persona
has no supported bounds (e.g. `user: unrestricted`, the default).

The key applies to both frontends: `run.py` steers every UQ correction round
with the YAML mode, and the demo runner's stage-2 **Steering** dropdown starts
at the YAML mode and can override it per generation (the other knobs stay
YAML-only).

- **`cg`** (classifier guidance, the default): from `guide_from` on, nudge the
  reverse diffusion mean by `-guidance_weight · ∇ₓ cost(x̂0)`. Roughly doubles
  sampling time (≈5 s → ≈9.5 s at N=500 × 50 frames), needs a differentiable
  cost, and `guidance_weight` needs per-cost calibration — the useful band is
  1e4–1e5 — but it also handles the case `resample` cannot: a prompt that
  genuinely contradicts the cost.
- **`resample`** (alternative): at each `resample_steps` index, score every
  chain and resample the population with weights
  `softmax(-z(cost)/temperature)`. Cost-agnostic (the cost need not be
  differentiable), effectively free at N=500, and it can only reweight motions
  MDM was already willing to produce — so samples stay on-manifold. In the
  triceps stress scenario this took oracle violations from 28% of samples to 0%
  with cluster diversity preserved (endpoint spread slightly *up*).

Each scoring step is logged (`cost`, fraction violating, ESS, surviving
ancestors in `resample`). When the first event shows ~every sample violating
*and* an ESS collapse, the log carries an explicit conflict warning: the prompt
and the cost disagree, so mere reweighting (`mode: resample`) cannot help — the
remaining levers are guidance (`mode: cg`, the default) and rewording the
correction. The warning is a diagnostic only; nothing switches mode
automatically.

In the interactive picker, each cluster panel has a **magnitude** slider
(range 0.0–2.0) that scales that trajectory's motion up or down while
preserving the direction of motion at every timestep (`scale` in joint-angle
space about the start pose: `1.0` = unchanged, `0.0` = hold start). `feedback.uq.scale`
sets the slider's initial value and is used directly as the scale in headless
runs. Select a panel and click **Refine selected** to cluster only that
panel's raw trajectories into `feedback.uq.n_clusters` child options; refinement can be
repeated recursively. When a selected option has fewer trajectories than
`feedback.uq.n_clusters`, each trajectory becomes its own child option. **Back** restores
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

For headless runs, set `feedback.uq.auto_cluster` in the YAML (and optionally
`feedback.uq.scale` to apply a fixed magnitude without the GUI):
```yaml
feedback:
  uq:
    diffusion_samples: 128
    n_clusters: 3
    auto_cluster: 0
    scale: 1.0
```

### Cartesian MPC With MDM/UQ
With both `feedback:` and `cartesian:` sections the planner first follows the
selected/generated MDM arm trajectory, then switches to Cartesian wrist goals.

Save as `src/uncertain_feedback/planners/mpc/configs/mdm.yaml`:
```yaml
steps: 750
horizon: 10
n_mpc_samples: 512
max_angle_delta: 0.0025
pose: "src/uncertain_feedback/motion_generators/mdm/demo_pose.pt"

feedback:
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
  --mpc-config src/uncertain_feedback/planners/mpc/configs/mdm.yaml \
  --model-path "src/uncertain_feedback/motion_generators/mdm/motion-diffusion-model/save/my_finetuned_final/model000750500.pt" \
  --text "raise my left arm" \
  --mdm-frames 100 \
  --save out.mp4 \
  --live
```

### Cartesian MPC Without MDM or UQ
A config whose only module section is `cartesian:` is direct Cartesian
wrist-goal MPC. This path does not generate motion or run clustering. If you
set an HML `pose:`, the runner decodes that pose once to initialize the arm,
collar, spine, and background body.

Save as `src/uncertain_feedback/planners/mpc/configs/plain.yaml`:
```yaml
steps: 750
horizon: 10
n_mpc_samples: 512
max_angle_delta: 0.0025
pose: "src/uncertain_feedback/motion_generators/mdm/demo_pose.pt"

cartesian:
  goals:
    - [0.3, 0.5, 0.1]
  threshold: 0.05
```

Run:
```bash
uv run python src/uncertain_feedback/planners/run.py \
  --mpc-config src/uncertain_feedback/planners/mpc/configs/plain.yaml \
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
radians; `elbow_flexion_angle` uses the scalar anatomical elbow hinge angle,
and `shoulder_abduction_angle` uses the upper-arm angle away from torso-down in
the spine3 frame.

`--arm` can override the starting arm state with a `.npy` file. The preferred
shape is `(3, 3)` for `[left_shoulder, left_elbow, left_wrist]`. Legacy `(4, 3)`
files are accepted; the first row fixes the left collar and the remaining rows
control shoulder, elbow, and wrist. This input is converted once to the internal
7-DOF planner state; arbitrary off-hinge elbow rotation is anatomically decoded
into shoulder rotation plus a scalar elbow-flexion angle while preserving arm
joint positions.



### `feedback.anchor_correction` — the frame-0 seam

A generator returns the pinned prefix's last frame as frame 0 and its first *free*
frame as frame 1, so any pull the model has toward its training prior's start pose
lands as a one-frame teleport between them — measured at 0.114 m on the deployed
`correction_demo1` checkpoint, against a ~0.005 m normal step. `anchor_correction`
(default **true**) drops frame 0 and shifts the rest so the correction begins at the
arm's current configuration (`SmplLeftArmFK.anchor_arm_trajectory`, applied in
planner `q` space so the elbow hinge stays parameterised). Per-frame displacements —
the demonstrated shape — are untouched; the seam cannot exist, since frame 0 *is*
the current configuration. The trajectory is one frame shorter. Set it false to
track the raw sample, which is what you want when measuring how big the seam is.

This also cleans up `feedback.uq.scale`: `scale_trajectory` anchors at frame 0 and
scales every offset from it, so with the raw sample the Magnitude slider scales the
teleport along with the motion (`scale: 0.5` halves the seam as a side effect).
Anchored, it scales only real motion.

Applied in all three correction paths: `Mpc.query_mdm_with_uncertainty` (the planner's UQ
route), `run.py`'s single-sample route, and the demo runner, which builds and pushes its own
corrections — `demo_runner/session.py::_activate_cluster_level` anchors each cluster mean, so
the browser's feature plots, the oracle cluster scores, the Magnitude slider and the tracked
trajectory all see the anchored version.

## Cost-generation backends

`llm_cost.backend` selects how the cost is generated:

Unless overridden by `llm_cost.model` or `OPENAI_MODEL`, LLM cost generation uses
`gpt-5.6-luna` with `xhigh` reasoning effort. Reasoning effort follows the model
(`gpt-5.6-luna` → `xhigh`, `gpt-5.6-sol` → `low`); any other model is sent without
one. The demo-runner config (`mdm_llm_transfer.yaml`)
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

Every generated LLM-cost artifact directory also includes
`reference_with_correction.mp4`, a video of the target reference trajectory that
contains the correction (`full_correction_traj` when available, otherwise the
MDM correction segment).

### Visual cost feedback (turns / agent)

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
  uv run python src/uncertain_feedback/evaluation_mechanism/render_cost_comparison.py \
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

### Accepted-pose evidence from the executed-trajectory corpus

Every backend also grounds itself in an **executed-trajectory corpus**. A
session's `trajectory_corpus/` holds one entry per goal — the goal's full executed
rollout as canonical `(T, 7)` q states (`traj_<i>.npy`) plus a per-frame
joint-feature `traj_<i>_features.csv`
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

## Method-level evaluation

Scripts that evaluate the *whole* pipeline end to end live in the repo-root
`evaluation/` directory, outside `src/`, structured as **benchmarks × approaches ×
metrics** with hydra configs (see `evaluation/README.md` for the full guide and the
paper-experiment → config mapping). Built on the per-stage façades
(`motion_generators`, `uncertainty`, `cost_generation`, `evaluation_mechanism`,
`planners.mpc.rollout`, `simulated_users`).

CPU-only smoke run (no MDM, no LLM):

```
uv run python evaluation/run_single_experiment.py approach=edit_baseline \
    approach.learning=none benchmark=smoke mpc_config=evaluation/conf/mpc_smoke.yaml
```

Single experiment / sweep / aggregation:

```
uv run python evaluation/run_single_experiment.py approach=full benchmark=personas_core
uv run python evaluation/run_single_experiment.py -m seed=0,1,2 \
    approach=full,no_steering benchmark=abstraction_sweep
uv run python evaluation/analyze_results.py multirun/ --out evaluation_analysis/
```

Approaches: `full`, `no_steering`, `immediate_only`, `no_learning` (SystemApproach
ablations over `learning` and `steering_mode`) and `edit_baseline` (predefined
parameterized edits, no text-to-motion model). Benchmarks: `smoke`,
`personas_core`, `abstraction_sweep` (verbalizer sweep), `lifelong` (per-persona
goal sequences; pair with `mpc_config=...mdm_llm_transfer.yaml`). Outputs land in
hydra's `outputs/`/`multirun/` dirs as `results.csv` (per feedback round) and
`episodes.csv` (per episode), plus per-round `.npy`/cost artifacts.

Note the in-`src` package `evaluation_mechanism/` is a different thing: it is how the
method scores its *own* generated cost functions, and the `agent` backend's sandbox
imports it at runtime.

## Demo runner web tool

Browser tool for designing and presenting simulated-user demo scenarios
interactively — the staged pipeline above (base rollout → MDM/UQ correction →
cost generation), but with every knob tweakable and every trajectory
inspectable before committing to an experiment config, plus a **demo/dev mode
toggle** and full **session replay**. Run it from the repo root (the artifact
root is CWD-relative):

```bash
uv run python src/uncertain_feedback/demo_runner/server.py \
  [--mpc-config src/uncertain_feedback/planners/mpc/configs/mdm_llm_transfer.yaml] \
  [--personas-file demo_runner_personas.json] \
  [--trajectory-configs-file demo_runner_trajectory_configs.json] \
  [--host 127.0.0.1] [--port 6781]
```

Then open `http://127.0.0.1:6781`. The config supplies the pose, MPC settings,
UQ defaults, `feedback.frames`, per-persona goal presets, and the `llm_cost` backend
used by the cost-generation stage. With the default config, cost generation
starts on `llm (single-pass)`; the backend dropdown can still select `turns` or
`agent`. The top-level `seed` is reset before each MDM generation request, so
identical demo inputs reproduce the same samples.

**Demo vs dev mode.** The header button toggles between them and the choice
persists in `localStorage` (demo is the default). Both modes use the same
guided left control bar: Scenario is a persistent setup/status panel above the
four stages Trajectory decision, Language correction, Cost generation, and
Apply feedback. Its configuration section is collapsible and closes
automatically when a trajectory starts, while Start/Exit and the live
trajectory status remain visible. Starting or continuing a trajectory stops at Trajectory
decision, where the operator either enters a correction or ignores the comfort
violation and continues. A correction does not have to wait for that stop:
*Correct from frame N* beside the scrubber takes one at any frame, comfortable
or not (see stage 1). Entering a correction unlocks Language correction.
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
   again after it returns to comfort and crosses a bound later.

   **Corrections at any frame.** The discomfort trigger is not the only way in.
   *Correct from frame N*, beside the frame scrubber, pauses for a correction at
   the frame the scrubber is on (`POST /api/live_trajectory/request_correction`
   with `{"step": <frame>}`, or `{"step": null}` for wherever the rollout stands)
   and opens Language correction, whether or not the simulated user is anywhere
   near a bound; the pause is reported with reason `operator`. Pressed while
   frames are still streaming, it stops the stepping at the current frame
   instead of racing it (the scrubber follows the newest frame during a live
   rollout, so scrub *after* stopping, not during). Scrubbing back to an
   **already executed frame** first makes the correction retroactive: the frames after it are discarded, the
   planner is rebuilt at that pose, and the correction stage conditions on the
   history up to it — so *Apply feedback + continue trajectory* re-rolls the
   rest of the trajectory from that frame under the new cost instead of from the
   end. `executed_trajectory.npy` is rewritten to the truncated path. Committed
   rounds are **not** rewound: rewinding undoes execution, not what the session
   has learned. *Ignore comfort violation + continue* also resumes from an
   operator pause without adding feedback, so a rewind can be abandoned. The accumulated
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
   clustering method (see `feedback.uq.clusterer` above; initial value from the config)
   and applies on the next Generate/Re-cluster/Refine. The *Steering* dropdown
   (cg / resample / off) steers the next *Generate* toward the persona's hidden
   bounds (see `feedback.uq.steering` above; initial value from the config's
   mode, other steering knobs stay YAML-only); scoring events and any
   prompt/cost conflict warning appear in the log panel. Each cluster is
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

4. **Multi-round feedback** — *Commit round* records the last generated cost as a
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
