# uncertain-feedback Codebase Map

**Last updated:** 2026-06-27  
**Branch:** mpc

> **Maintenance rule:** Update this file whenever a new module, planner, cost term, or major data-pipeline step is added.

---

## 1. Top-Level Purpose

Confidence-aware language-grounded arm motion planning.

The system takes a **natural-language prompt** (e.g. "raise my left arm") and produces a **physically-tracked arm trajectory** on a simulated SMPL humanoid, using:

1. **MDM** (Motion Diffusion Model) — generates candidate arm trajectories from text
2. **UQ / clustering** — draws multiple diffusion samples, clusters them, lets the user (or auto-selector) pick a behaviorally-distinct cluster
3. **Sampling-based MPC** — tracks the selected trajectory, respecting configurable kinematic cost terms
4. **LLM-generated costs** — optionally asks an LLM (OpenAI) to write Python cost functions that are compiled and injected at runtime

---

## 2. Repository Layout

```
uncertain-feedback/
├── src/uncertain_feedback/
│   ├── consts.py                     # Project-wide paths (MDM_ROOT, weights)
│   ├── planners/
│   │   ├── run.py                    # Single-run CLI: plan → language correction → finish
│   │   └── mpc/
│   │       ├── __init__.py           # Public exports
│   │       ├── config.py             # YAML → MpcRunConfig dataclass
│   │       ├── kinematics.py         # SmplLeftArmFK, SMPL topology constants
│   │       ├── costs/                # Cost package (public surface: mpc.costs)
│   │       │   ├── __init__.py       # Re-exports the public cost API
│   │       │   ├── base.py           # Cost terms + registry + preference learning
│   │       │   ├── llm_costs.py      # LLM-generated Python cost pipeline
│   │       │   └── prompts/          # Prompt templates loaded from .txt files
│   │       │       ├── __init__.py   # PROMPTS registry + build_llm_cost_prompt
│   │       │       ├── runtime_api.txt       # Shared technical contract
│   │       │       ├── output_contract.txt   # Shared output rules
│   │       │       └── templates/    # One .txt per prompt (stem = registry key)
│   │       │           ├── default.txt
│   │       │           ├── caregiver.txt
│   │       │           └── goal_safe.txt
│   │       ├── arm_mpc.py            # SmplLeftArmMPC (base sampling MPC)
│   │       ├── arm_mpc_mdm.py        # LeftArmMPCMDM (+ MDM trajectory tracking)
│   │       ├── arm_mpc_mdm_uq.py     # LeftArmMPCMDMUQ (+ UQ clustering)
│   │       ├── arm_mpc_cartesian_base.py  # _CartesianGoalsMixin (shared Cartesian logic)
│   │       ├── arm_mpc_cartesian.py  # LeftArmMPCCartesian (MDM then Cartesian)
│   │       ├── arm_mpc_cartesian_no_mdm.py  # ArmMPCCartesianNoMDM (pure Cartesian)
│   │       └── configs/              # Example YAML config files
│   │           ├── arm_mpc_cartesian_mdm.yaml
│   │           ├── arm_mpc_cartesian_mdm_learn.yaml
│   │           ├── arm_mpc_cartesian_mdm_llm.yaml
│   │           ├── arm_mpc_cartesian_mdm_llm_turns.yaml  # backend: turns (multi-turn scored selection)
│   │           ├── arm_mpc_cartesian_mdm_llm_agent.yaml  # backend: agent (codex CLI)
│   │           └── arm_mpc_cartesian_no_mdm.yaml
│   ├── experiments/                  # Multi-run experiment machinery (separate from a single run)
│   │   ├── cluster_comparison.py     # Generate + roll out one LLM cost per UQ cluster
│   │   ├── run_experiment.py         # CLI entry point for cluster comparison experiments
│   │   ├── backend_comparison.py     # Generate one cost per backend (llm/turns/agent), score uniformly
│   │   └── run_backend_experiment.py # CLI entry point for per-backend comparison experiments
│   ├── motion_generators/
│   │   └── mdm/
│   │       ├── mdm_api.py            # MdmMotionGenerator: text → arm trajectory
│   │       ├── hml_smpl_conversion.py  # HML263 ↔ SMPL pose conversions
│   │       ├── mdm_parser_util.py    # CLI arg parser helpers for MDM scripts
│   │       ├── sample_leftarm.py     # Standalone left-arm generation script
│   │       ├── train_leftarm.py      # Fine-tuning script for left arm
│   │       ├── create_tpose.py       # T-pose demo pose generator
│   │       ├── make_initial_pose.py  # Build an HML pose from a body config
│   │       ├── visualize_sitting_pose.py
│   │       └── motion-diffusion-model/   # Git submodule (GuyTevet/MDM)
│   ├── uncertainty/
│   │   ├── clustering/               # Trajectory clustering methods (subclass to add)
│   │   │   ├── base.py               # TrajectoryClusterer (template: _to_features + _fit_predict)
│   │   │   └── xyz_clusterer.py      # XyzPositionClusterer (KMeans on FK positions)
│   │   └── cluster_picker.py         # Interactive matplotlib cluster picker UI (stays here)
│   ├── data_collection/
│   │   ├── build_mdm_dataset.py      # Build HumanML3D dataset from video/labels
│   │   ├── extract_all_frames.py     # Video → frames
│   │   ├── labeler.py                # Browser-based text labeling UI (Flask)
│   │   ├── mhr_pose_estimator.py     # MHR human pose estimation wrapper
│   │   ├── mhr_to_hml263_pipeline.py # MHR → HML263 feature pipeline
│   │   ├── smpl_to_hml263.py         # SMPL body_pose → HML263
│   │   ├── _mhr_inference_worker.py  # SMPL inference worker (detectron2 / SAM)
│   │   ├── _mhr_inference_worker_conversion.py
│   │   ├── run_mhr_demo.py           # End-to-end demo runner
│   │   ├── show_conversion.py        # Visualization for SMPL→HML conversion
│   │   ├── viz.py                    # Data collection visualization helpers
│   │   └── trajectory_editor/
│   │       ├── server.py             # Flask web UI for hand-authoring trajectories
│   │       └── hml_decode.py         # HML decode utilities for the editor
│   ├── llm/
│   │   ├── base_model.py             # BaseModel ABC (get_full_output)
│   │   └── openai_model.py           # OpenAI wrapper implementing BaseModel (Chat + Responses APIs)
│   └── utils/
│       └── plot.py                   # ArmVisualizer (live MPC window + static drawing)
├── README.md                         # Full setup + run instructions
├── CODEBASE_MAP.md                   # This file
├── CLAUDE.md                         # Instructions for AI assistants
├── .claude/
│   └── POSE_REPRESENTATION_AUDIT.md  # Canonical reference for all pose formats
└── pyproject.toml                    # Python 3.10, uv-managed dependencies
```

---

## 3. Planner Hierarchy

```
SmplLeftArmMPC (arm_mpc.py)
│  Core sampling MPC: sample N action sequences, roll out, pick min-cost.
│  State: (3, 3) axis-angle for [left_shoulder, left_elbow, left_wrist].
│  Warm-start: shifts previous best plan by one step each iteration.
│
├── LeftArmMPCMDM (arm_mpc_mdm.py)
│     Adds: validate_trajectory() (advisory safety-cost check), push_trajectory()
│     (validate + queue for rate-limited playback), set_mdm_goal(),
│     MDM-colored arm rendering, body_pos background skeleton. The MDM trajectory
│     is played back directly (no per-step sampling) at a bounded angular speed
│     (max_playback_delta per joint per step), so jumps are eased not snapped;
│     the MPC then resumes sampling toward the final goal.
│
│   └── LeftArmMPCMDMUQ (arm_mpc_mdm_uq.py)
│         Adds: query_mdm_with_uncertainty() — draws N diffusion samples,
│         clusters with XyzPositionClusterer, shows cluster picker UI,
│         enqueues mean of chosen cluster.
│
└── _CartesianGoalsMixin (arm_mpc_cartesian_base.py)
      Adds: Cartesian wrist-goal queue; cost switches from joint-space to
      world-space wrist position.

    ├── LeftArmMPCCartesian (arm_mpc_cartesian.py)
    │     Inherits LeftArmMPCMDMUQ + _CartesianGoalsMixin.
    │     First tracks MDM/UQ trajectory, then switches to Cartesian wrist goals.
    │
    └── ArmMPCCartesianNoMDM (arm_mpc_cartesian_no_mdm.py)
          Cartesian MPC only — no MDM, no UQ.
          Decodes optional HML pose to initialize body but generates nothing.
```

**YAML `planner` → class mapping:**

| `planner` value              | Class                    |
|------------------------------|--------------------------|
| `arm_mpc`                    | `SmplLeftArmMPC`         |
| `arm_mpc_mdm`                | `LeftArmMPCMDM`          |
| `arm_mpc_mdm_uq`             | `LeftArmMPCMDMUQ`        |
| `arm_mpc_cartesian`          | `LeftArmMPCCartesian`    |
| `arm_mpc_cartesian_no_mdm`   | `ArmMPCCartesianNoMDM`   |

---

## 4. Main Data Flow

```
Text prompt
    │
    ▼
MdmMotionGenerator.generate_left_arm_trajectory()   [mdm_api.py]
    │   Loads HML263 start pose, inpaints non-arm joints, runs diffusion
    │
    ▼  (n_frames, 3, 3)  arm axis-angle trajectory
    │
    │  [if UQ]
    ├── Run N times → (N, n_frames, 3, 3)
    │       │
    │       ▼
    │   XyzPositionClusterer.cluster()                [clustering/xyz_clusterer.py]
    │       KMeans on FK positions at frame ~100
    │       │
    │       ▼
    │   pick_cluster() / pick_cluster_positions()     [cluster_picker.py]
    │       Interactive matplotlib window
    │       Returns chosen label → cluster mean trajectory
    │
    ▼  chosen (n_frames, 3, 3) trajectory
    │
SmplLeftArmMPC / subclass
    │   MDM trajectory validated against safety costs, then played back at a
    │   bounded angular speed (rate-limited, push_trajectory); after
    │   playback the MPC samples toward the final goal.
    │   Each MPC step: sample N×H action sequences, rollout, compute cost, take best
    │
    ├── Joint-space cost: L2 to current goal in (3,3) axis-angle space
    └── Extra costs: CompositeTrajectoryCost terms [costs/base.py]
            ElbowHeightCost
            ElbowFlexionAngleCost
            ShoulderAbductionAngleCost
            GeneratedPythonCost [costs/llm_costs.py]
    │
    ▼  next_q (3, 3) each step
    │
ArmVisualizer.update_step()                          [utils/plot.py]
    Live matplotlib 3D window + optional video capture
```

---

## 5. Kinematics (`kinematics.py`)

- **`SmplLeftArmFK`** — loads SMPL neutral PKL once; stores T-pose bone offsets for the arm chain
- Joint chain: `spine3 (9) → left_collar (13) → left_shoulder (16) → left_elbow (18) → left_wrist (20)`
- MPC controls 3 joints: shoulder, elbow, wrist — collar is fixed from initial pose decode
- Key methods:
  - `fk(arm_aa, spine3_pos, spine3_aa) → (5, 3)` world positions
  - `fk_batch(arm_aa, ...) → (N, 5, 3)` batched
  - `arm_aa_from_positions(positions, spine3_aa) → (3, 3)` inverse: XYZ → local axis-angles
  - `full_body_positions(arm_aa, ...) → (22, 3)` for visualization

---

## 6. Cost System (`costs/base.py`)

### Cost term protocol

All cost terms implement `TrajectoryCost`: callable `(q_trajs: np.ndarray) → np.ndarray` where input is `(N, H+1, 3, 3)` rollout states and output is `(N,)` per-rollout scalar costs.

### Learnable preference costs (`LearnablePreferenceCost` Protocol)

Expose `min_value`, `max_value`, `feature_values()`, `with_range()` so the runner can automatically update bounds from MDM demonstrations after trajectory generation.

### Registered cost terms (YAML keys → classes)

| YAML key                  | Class                       | Feature                                      |
|---------------------------|-----------------------------|----------------------------------------------|
| `elbow_height`            | `ElbowHeightCost`           | Spine3-relative elbow Y position (meters)    |
| `elbow_flexion_angle`     | `ElbowFlexionAngleCost`     | Elbow rotation magnitude (radians)           |
| `shoulder_abduction_angle`| `ShoulderAbductionAngleCost`| Upper-arm angle from torso-down (radians)    |

### Cost structure

- `_range_rollout_cost()` — shared helper: penalizes out-of-range feature values across predicted rollout, plus a progress penalty when already violating and moving farther out
- `CompositeTrajectoryCost` — sums a list of `TrajectoryCost` terms
- `replace_cost_in_composite()` — swap one term type in a composite
- `update_preference_cost()` — snaps one bound (min or max) from MDM vs. MPC trajectory statistics

**Adding a new cost term:**
1. Write a frozen dataclass implementing `TrajectoryCost` (and optionally `LearnablePreferenceCost`)
2. Add a `_build_<name>()` factory
3. Register in `COST_BUILDERS` dict
4. Update this map

---

## 7. LLM Cost Pipeline (`costs/llm_costs.py`)

When `llm_cost.enabled: true` in the YAML:

1. `build_motion_summaries()` — text summaries of recent MPC steps and MDM trajectory
2. `render_prompt_images()` — delegates to `ArmVisualizer.render_trajectory_overlay()` for the 3-view overlay (optional)
3. `build_llm_cost_prompt()` (in `costs/prompts/__init__.py`) — assembles the prompt from a named template (a `costs/prompts/templates/<name>.txt` head + shared `runtime_api.txt`/`output_contract.txt`), selected by `llm_cost.prompt`
4. LLM (OpenAI, configurable model) returns JSON: `{description, code, params, explanation, recipient_explanation}`
5. `parse_llm_cost_response()` → `LlmCostResponse`
6. `GeneratedPythonCost.__post_init__()` → `compile_generated_cost()` compiles the code snippet
7. `GeneratedCostContext` provides the runtime sandbox: `fk`, `spine3_pos/aa`, `current_q`, `mdm_traj`, `recent_q`, and FK helper methods
8. Artifacts (prompt JSON, images, cost.py) saved to `llm_cost_artifacts/<timestamp>/`

**LLM cost cluster experiment** (`llm_cost.cluster_experiment.enabled`): runs the LLM cost on each cluster's mean trajectory for `rollout_steps` steps and uses costs to rank / auto-select clusters.

---

## 8. Configuration System (`config.py`)

`MpcRunConfig` (frozen dataclass) loaded from YAML by `load_mpc_config(path)`.

**Top-level YAML keys:**

| Key                    | Type     | Purpose                                               |
|------------------------|----------|-------------------------------------------------------|
| `planner`              | str      | One of the 5 planner choices                         |
| `steps`                | int      | Total MPC steps to run                               |
| `horizon`              | int      | MPC look-ahead steps                                 |
| `n_mpc_samples`        | int      | Candidate action sequences per step                  |
| `max_angle_delta`      | float    | Sampling std dev (radians)                           |
| `pose`                 | path?    | HML pose `.pt` file for initial body state           |
| `goal_threshold`       | float    | L2 dist threshold to pop goal (default 0.01)         |
| `advance_threshold`    | float    | Goal-advance threshold for the MPC resume phase (default 0.1). MDM frames are now played back directly, so this no longer governs MDM frame advancement. |
| `max_playback_delta`   | float    | Max per-joint rotation (radians) per step while following the MDM trajectory (default 0.1). Rate limit: caps playback angular speed so the initial jump into the trajectory and any large frame-to-frame jump are eased rather than snapped. |
| `trajectory_fraction`  | float    | Fraction of MDM frames to enqueue (default 1.0)      |
| `preference_learning`  | bool     | Auto-update cost bounds from MDM (default true)      |
| `preference_alpha`     | float    | Blend weight for preference update (default 0.5)     |
| `preference_window`    | int      | MPC step history for preference update (default 50)  |
| `uq.*`                 | UqConfig | `diffusion_samples`, `n_clusters`, `auto_cluster`    |
| `cartesian.*`          | CartesianConfig | `goals` (list of [x,y,z]), `threshold`        |
| `costs.*`              | dict     | Named cost terms with their params                   |
| `llm_cost.*`           | LlmCostConfig | `enabled`, `model`, `strict`, `artifact_dir`, `use_images`, `cluster_experiment` |

---

## 9. Data Collection Pipeline (`data_collection/`)

Full pipeline from raw video to fine-tuned MDM model:

```
Step 0 (optional): Hand-author synthetic trajectories
    └── trajectory_editor/server.py  (Flask browser UI)
        └── trajectory_editor/hml_decode.py

Step 1: Extract frames from videos
    └── extract_all_frames.py → video_frames/

Step 2: Label video segments with text
    └── labeler.py (Flask browser UI) → labels.json

Step 3: Build MDM dataset
    └── build_mdm_dataset.py → HumanML3D dataset dir
        Uses: smpl_to_hml263.py, mhr_to_hml263_pipeline.py

Step 4: Fine-tune MDM
    └── motion-diffusion-model/train/train_mdm.py
        (or train_leftarm.py for left-arm-specific training)

Step 5: Generate with fine-tuned model
    └── sample_leftarm.py / mdm_api.py
```

**From real video (alternative Step 1–3):**
```
Videos → _mhr_inference_worker.py (detectron2 + SAM → SMPL .npz)
       → mhr_to_hml263_pipeline.py → HML263 features
       → build_mdm_dataset.py → dataset
```

---

## 10. Pose Representations

See `.claude/POSE_REPRESENTATION_AUDIT.md` for full reference. Key formats:

| Format                  | Shape      | Where used                                    |
|-------------------------|------------|-----------------------------------------------|
| Controlled arm aa       | `(3, 3)`   | MPC state/action (shoulder, elbow, wrist)     |
| HML263 feature vector   | `(263,)`   | MDM input/output                              |
| SMPL body_pose          | `(21, 3)`  | SMPL model; intermediate conversion format   |
| XYZ joint positions     | `(22, 3)`  | FK output, visualization, clustering features|
| Arm chain positions     | `(5, 3)`   | FK arm output (spine3 through wrist)          |
| Spine3 anchor           | `(3,)`×2   | Position + axis-angle; fixed reference frame  |
| Cartesian wrist goal    | `(3,)`     | Spine3-relative target for Cartesian MPC      |
| MPC rollout batch       | `(N,H+1,3,3)` | N samples × H+1 steps × 3 joints × 3 dim |

---

## 11. Key Entry Points

| Command / Script                                      | Purpose                              |
|-------------------------------------------------------|--------------------------------------|
| `uv run python src/.../planners/run.py --mpc-config <yaml>` | Single MPC run (plan → language correction → finish) |
| `uv run python src/.../experiments/run_experiment.py --mpc-config <yaml>` | Per-cluster LLM-cost comparison experiment |
| `uv run python src/.../experiments/run_backend_experiment.py --mpc-config <yaml>` | Per-backend (llm/turns/agent) cost comparison experiment |
| `uv run python src/.../sample_leftarm.py`             | Standalone MDM generation            |
| `uv run python src/.../data_collection/labeler.py`    | Browser labeling UI                  |
| `uv run python src/.../trajectory_editor/server.py`   | Synthetic trajectory editor          |
| `uv run python src/.../data_collection/build_mdm_dataset.py` | Build HumanML3D dataset        |
| `uv run python -m train.train_mdm` (in MDM subdir)    | Fine-tune MDM model                  |

---

## 12. External Dependencies / Submodules

- **motion-diffusion-model** — git submodule at `motion_generators/mdm/motion-diffusion-model/` (GuyTevet/MDM)
  - HumanML3D dataset expected at `.../dataset/HumanML3D/`
  - Pretrained weights at `.../save/humanml_enc_512_50steps/model000750000.pt`
  - SMPL neutral model at `.../body_models/smpl/SMPL_NEUTRAL.pkl`
- **OpenAI** — used for LLM cost generation (`llm/openai_model.py`)
- **sklearn** — KMeans clustering in `clustering/base.py`
- **smplx** — SMPL model loading
- **detectron2** — human pose estimation in data collection worker
