# uncertain-feedback Codebase Map

**Last updated:** 2026-07-09  
**Branch:** simulated-users-standard

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
│   │       │   ├── generated.py      # Runtime context, cost compile/exec, summaries, image render
│   │       │   ├── cost_generator.py # CostGenerator base + create_cost_generator factory + scoring
│   │       │   ├── llm_costs.py      # backend: llm (interpret→ground→author, single pass) + re-exports
│   │       │   ├── turns_costs.py    # backend: turns (fixed interpret, ground+author refinement)
│   │       │   ├── agent_costs.py    # backend: agent (codex CLI)
│   │       │   ├── cost_feedback.py  # EvalState (picklable rollout state for agent backend)
│   │       │   └── prompts/          # Staged prompt text files
│   │       │       ├── __init__.py   # staged prompt builders + image placeholder substitution
│   │       │       ├── runtime_api.txt       # Shared technical contract
│   │       │       ├── output_contract.txt   # Shared output rules
│   │       │       └── stages/       # interpret.txt, ground.txt, author.txt, refine.txt
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
│   │           ├── arm_mpc_cartesian_mdm_llm_transfer.yaml  # simulated-user transfer experiment
│   │           └── arm_mpc_cartesian_no_mdm.yaml
│   ├── experiments/                  # Multi-run experiment machinery (separate from a single run)
│   │   ├── experiment_pipeline.py    # Staged simulated-user experiment core (trigger, UQ, cost, eval)
│   │   ├── run_experiment.py         # CLI: one persona + one backend on the original goal
│   │   ├── cluster_comparison.py     # Generate + roll out one cost per UQ cluster
│   │   ├── run_cluster_experiment.py # CLI entry point for per-cluster comparison experiments
│   │   ├── backend_comparison.py     # Generate one cost per backend (llm/turns/agent), score uniformly
│   │   ├── run_backend_experiment.py # CLI entry point for per-backend comparison experiments
│   │   ├── transfer_experiment.py    # Adds held-out transfer-goal eval around experiment_pipeline
│   │   ├── run_transfer_experiment.py # CLI entry point for simulated-user transfer experiments
│   │   └── render_cost_comparison.py # CLI the agent backend runs to render rollout-vs-correction overlay
│   ├── motion_generators/
│   │   ├── __init__.py               # MOTION_GENERATOR_BUILDERS registry + make_motion_generator
│   │   ├── base.py                   # MotionGenerator ABC (shared backend interface)
│   │   ├── kimodo/                   # kimodo (NVIDIA) backend, isolated conda env
│   │   │   ├── kimodo_api.py         # KimodoMotionGenerator (subprocess bridge)
│   │   │   ├── _kimodo_inference_worker.py  # standalone worker (runs in kimodo env)
│   │   │   ├── generate_motion.py    # Standalone kimodo text-to-motion + video CLI
│   │   │   └── start_pose.npy        # SMPL body_pose (21,3) default start pose
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
│   ├── simulated_users/
│   │   ├── base.py                   # SimulatedUser, HiddenBound/CoupledBound, violations, cluster choice, oracle cost term
│   │   ├── personas.py               # Clinically motivated personas (PERSONAS registry)
│   │   └── viz.py                    # render_hidden_bounds: shaded forbidden regions + trajectories
│   ├── data_collection/
│   │   ├── build_mdm_dataset.py      # Build HumanML3D dataset from video/labels
│   │   ├── extract_all_frames.py     # Video → frames
│   │   ├── labeler.py                # Browser-based text labeling UI (Flask)
│   │   ├── mhr_pose_estimator.py     # MHR human pose estimation wrapper
│   │   ├── mhr_to_hml263_pipeline.py # MHR → HML263 feature pipeline
│   │   ├── smpl_to_hml263.py         # positions → HML263 via official HumanML3D process_file
│   │   ├── t2m_example_frame.npy     # (22,3) t2m target-skeleton frame for uniform_skeleton
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
│         clusters with XyzPositionClusterer, shows cluster picker UI (each
│         cluster has a magnitude slider that scales the chosen motion about
│         its start pose via scale_trajectory), enqueues the (scaled) mean of
│         the chosen cluster. Headless selection: `auto_cluster` (fixed label)
│         or `cluster_selector` (callable on the cluster means; used by
│         simulated-user experiments; takes precedence). Transfer experiments
│         use a selector that scores each scaled raw cluster mean with the
│         hidden oracle cost and chooses the lowest score.
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
| `elbow_flexion_angle`     | `ElbowFlexionAngleCost`     | Upper-arm/forearm bend angle (radians; 0 = straight) |
| `shoulder_abduction_angle`| `ShoulderAbductionAngleCost`| Upper-arm angle from torso-down (radians)    |

### Always-on cost terms (not YAML-registered)

- `JointLimitCost` — linear (L1) penalty on rollout joint rotations that leave a per-slot axis-angle box. Not a YAML key: `build_run()` builds it from the active persona's `joint_limits` (`SimulatedUser.limit_cost()`) and appends it to the composite, so the planner refuses to *generate* motions outside the same anatomical box the persona uses to *score* them. Linear (not squared) so the gradient stays firm at the boundary.

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
3. Staged prompt builders (in `costs/prompts/__init__.py`) assemble `interpret` (instruction + images + compact summary), `ground` (interpretation + full summaries), and `author` (numeric spec + `runtime_api.txt`/`output_contract.txt`) prompts from `costs/prompts/stages/*.txt`
4. LLM (OpenAI, configurable model) returns final author JSON: `{description, code, params, explanation, recipient_explanation}`
5. `parse_llm_cost_response()` → `LlmCostResponse`
6. `GeneratedPythonCost.__post_init__()` → `compile_generated_cost()` compiles the code snippet
7. `GeneratedCostContext` provides the runtime sandbox: `fk`, `spine3_pos/aa`, `current_q`, `mdm_traj`, `recent_q`, and FK helper methods
8. Artifacts (stage prompts/responses, `stage_log.md`, images, cost.py, `reference_with_correction.mp4`) saved to `llm_cost_artifacts/<timestamp>/`

**LLM cost cluster experiment** (`llm_cost.cluster_experiment.enabled`): runs the LLM cost on each cluster's mean trajectory for `rollout_steps` steps and uses costs to rank / auto-select clusters.

### Cost-generation backends (`costs/cost_generator.py`)

`create_cost_generator()` selects one of three iteration mechanisms via `llm_cost.backend`; all share the staged prompting strategy and `CostGenerator` (stage helpers, compile/validate, save, install):

- `llm` (`llm_costs.py`) — single-pass staged calls: interpret → ground → author; only the author output is parsed/compiled. Stage prompt/response pairs are aggregated in `stage_log.md`.
- `turns` (`turns_costs.py`) — interprets once, then runs a stateful ground+author conversation; keeps the best cost by ranking consistency (`rank_candidate_cost`, falling back to the L2 rollout score when the context has no comparison trajectories). `stage_log.md` includes the interpretation plus each refine turn's prompt snapshot and response.
- `agent` (`agent_costs.py`) — delegates the staged method to the `codex` CLI, which emits the same `response.json` and must write `stage_log.md` with Stage 1 / Stage 2 / Stage 3 responses.

### Cost evaluation & visual feedback

- `rank_candidate_cost()` (`cost_generator.py`) evaluates a candidate cost by **ranking consistency**: the cost is applied directly to the trajectories whose preference order the user revealed — the chosen correction (`mdm_traj`) must cost less than the original plan (`reference_traj`, strict) and no more than the rejected UQ cluster means (`context.rejected_trajs`, weak). All trajectories are first resampled to equidistant joint-space-arclength points (`resample_equidistant`) so only the path matters, not timing (MDM output is systematically slower than a fresh rollout — timing is a pipeline artifact, not intent), then compared after z-normalization. Returns a `CostRanking` (rank accuracy + normalized margin + inert flag; `sort_key` orders candidates), or `None` when the context has no comparison trajectories — then `turns` falls back to the L2 rollout score. `rejected_trajs` is collected in `run.py` from `last_uq_result.cluster_means` (non-chosen labels) and threaded through `build_generated_cost_context` / `EvalState`.
- `evaluate_candidate_cost()` rolls the goal-seeking MPC with a candidate cost installed (`_make_cost_eval_rollout` in `run.py`) and returns the mean FK-position L2 distance to the MDM correction (`_score_rollout`; both trajectories resampled equidistant-in-arclength, so the score is path-only). Lower is better; `inf` when the planner has no Cartesian goal. Drives the `backend_comparison.json` ranking and the `turns` fallback.
- **Goal-reachability check** (`goal_reach_report()`): because ranking/`_score_rollout` grade only against the correction preferences, a cost can score well while stopping the arm short of the goal. `goal_reach_report(context, rollout)` reproduces the MPC's own `ArmMPCCartesian.goal_reached` criterion (FK the final rollout frame, spine3-relative wrist vs `context.cartesian_goal` within `context.cartesian_threshold`), returning `{reached, distance, threshold}` (or `None` for non-Cartesian planners). The goal + threshold are threaded onto `GeneratedCostContext` via `build_generated_cost_context` / `EvalState`. This is gated on `goal_conflict` (`parse_goal_conflict()` reads the stage-1 flag): only when stage one judged the goal reachable does missing it count against a candidate. In the `turns` backend selection is keyed on `(reach_rank, *ranking.sort_key)` so a goal-reaching candidate beats a non-reaching one and ranking consistency orders candidates within that, and each turn's feedback tells the model whether the goal was reached; in the `agent` backend `render_cost_comparison.py` prints a `goal reached:` line and `TASK.md` instructs codex to keep revising until the goal is reached (unless it conflicts).
- `CostGenerator.begin()` writes `reference_with_correction.mp4` into every generated-cost artifact directory using `context.full_correction_traj` when present, falling back to `context.mdm_traj`.
- `evaluate_and_render()` does the **same single rollout** and additionally renders `ArmVisualizer.render_cost_feedback_overlay()` — a rollout (red) vs target-corrected-path (green) overlay — returning `(score, image_path)`. When an `angle_path` is passed it also renders `ArmVisualizer.render_joint_angle_comparison()` — a joint-angle-over-time graph (one subplot per anatomical joint feature; green target vs red rollout, per-frame series from `build_joint_angle_series()`) — so the model sees the temporal shape of the motion, not just endpoints. The green target is `context.full_correction_traj` when present (the **entire** intended path: pre-correction history → MDM correction → comfort-only continuation to the goal, assembled once in `run.py:_assemble_full_correction_traj` and threaded through `build_generated_cost_context` / `EvalState`), falling back to the MDM correction segment (`mdm_traj`) alone. The score (`_score_rollout`) still measures distance to the MDM correction segment only. It can also persist the exact rollout array and MP4 used for that feedback image. When `use_images` is on:
  - `turns` feeds `turn_<i>/comparison.png` + `turn_<i>/angles.png` + score back through the conversation each turn; backend experiments with `--save-video` also write `turn_<i>/rollout.npy` and `turn_<i>/rollout.mp4`.
  - `agent` self-iterates: `EvalState` (`costs/cost_feedback.py`, picklable bundle that rebuilds the rollout + context off-process) is saved to `state.pkl`, the initial overlay paths are listed as text in `TASK.md`, and codex is instructed to load those local image files and append image observations, revision rationale, and the final stop reason to `ITERATION_LOG.md`. Codex runs `experiments/render_cost_comparison.py` (writing both `comparison.png` and `angles.png`) to render and inspect its own rollout before finalizing `response.json`; the wrapper appends `ITERATION_LOG.md` into `codex.log`. With `--archive-dir` and `--save-video`, each self-check is archived under `candidate_<i>/` with JSON, score, rollout, and MP4 artifacts.

---

## 8. Configuration System (`config.py`)

`MpcRunConfig` (frozen dataclass) loaded from YAML by `load_mpc_config(path)`.

**Top-level YAML keys:**

| Key                    | Type     | Purpose                                               |
|------------------------|----------|-------------------------------------------------------|
| `planner`              | str      | One of the 5 planner choices                         |
| `motion_generator`     | str      | Text-to-motion backend: `mdm` (default) or `kimodo`  |
| `steps`                | int      | Total MPC steps to run                               |
| `horizon`              | int      | MPC look-ahead steps                                 |
| `n_mpc_samples`        | int      | Candidate action sequences per step                  |
| `max_angle_delta`      | float    | Sampling std dev (radians)                           |
| `pose`                 | path?    | HML pose `.pt` file for initial body state           |
| `goal_threshold`       | float    | L2 dist threshold to pop goal (default 0.01)         |
| `advance_threshold`    | float    | Goal-advance threshold for the MPC resume phase (default 0.1). MDM frames are now played back directly, so this no longer governs MDM frame advancement. |
| `max_playback_delta`   | float    | Max per-joint rotation (radians) per step while following the MDM trajectory (default 0.1). Rate limit: caps playback angular speed so the initial jump into the trajectory and any large frame-to-frame jump are eased rather than snapped. |
| `trajectory_fraction`  | float    | Fraction of MDM frames to enqueue (default 1.0)      |
| `num_denoising_steps`  | int?     | Kimodo DDIM steps (kimodo backend only; None = backend default 100) |
| `preference_learning`  | bool     | Auto-update cost bounds from MDM (default true)      |
| `preference_alpha`     | float    | Blend weight for preference update (default 0.5)     |
| `preference_window`    | int      | MPC step history for preference update (default 50)  |
| `user`                 | str      | Simulated-user persona name (default `unrestricted`); loaded by `build_run` into `RunSetup.user` for every run |
| `uq.*`                 | UqConfig | `diffusion_samples`, `n_clusters`, `auto_cluster`, `scale` (default motion-magnitude scale for the chosen cluster; slider initial value in the GUI, applied directly when headless), `user_cluster` (delegate cluster choice to the configured user when it has bounds; precedence over `auto_cluster`/GUI) |
| `cartesian.*`          | CartesianConfig | `goals` (list of [x,y,z]), `threshold`        |
| `costs.*`              | dict     | Named cost terms with their params                   |
| `llm_cost.*`           | LlmCostConfig | `enabled`, `model`, `strict`, `artifact_dir`, `use_images`, `backend`, `max_turns`, `codex_cmd` |
| `transfer.*`           | TransferConfig | `goals` (held-out spine3-relative wrist targets), `trigger_threshold` (hidden-cost violation in rad that triggers simulated feedback) |
| `persona_goals.*`      | dict[str, PersonaGoals] | Per-persona override of `cartesian`/`transfer` goals for simulated-user experiments (applied by `experiment_pipeline.apply_persona_goals`; falls back to top-level goals when absent). Each restriction needs its own goal geometry to make the default plan visibly require a correction. |

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

> **Camera→world handedness:** `_mhr_inference_worker.py` converts SAM-3D-Body's
> right-handed OpenCV camera keypoints (Y-down) to world (Y-up) by negating
> **Y and Z** (180° about X). Negating Y alone mirrors body chirality (fixed
> 2026-06-29). Any `data/mdm_cache/*` or `new_joint_vecs/*` collected before that
> fix is mirrored and must be regenerated. See
> `.claude/POSE_REPRESENTATION_AUDIT.md` §6.

> **HML263 encoding (2026-07-07):** `positions_to_hml263` now delegates to the
> **official** HumanML3D `process_file` (MDM submodule) — `uniform_skeleton`
> retargeting onto the t2m skeleton, quaternion IK for 6D rotations,
> heading-local forward-difference velocities, squared-velocity foot contacts.
> It returns **N-1** feature frames for N position frames. The custom feature
> assembly (min-rotation identity-root IK, world-frame backward-difference
> velocities) is gone; `smpl_params_to_hml263` / `smpl_params_to_positions`
> were deleted (dead). `build_mdm_dataset.py` now caches `(N, 22, 3)` positions
> (not features) in version-tagged files (`_CACHE_VERSION`), clamps sequence
> length at the positions level, and applies augmentation noise to arm features
> only. Start poses (`demo_pose.pt`) created with the old encoder should be
> regenerated before the next fine-tune.

> **HML263 decoding (2026-07-09):** the official encoding above stores 6D
> rotations relative to the **t2m reference skeleton** (arms hanging down),
> not this repo's SMPL T-pose — so the 6D block can no longer be read as
> repo-convention `body_pose`. `hml263_to_smpl_body_pose` /
> `hml263_batch_to_smpl_body_pose` now decode via `recover_from_ric`
> positions + minimum-rotation IK (the direct 6D read produced ~90° arm
> errors on officially-encoded poses), and `smpl_arm_aa_to_hml263_frame`
> re-encodes the patched frame with the official `positions_to_hml263`
> instead of writing repo-convention 6D features in place. Note the official
> encoder re-faces the body to Z+ from hips+shoulders, so re-encoding a frame
> whose arm moved shifts the canonical heading slightly.

---

## 10. Pose Representations

See `.claude/POSE_REPRESENTATION_AUDIT.md` for full reference. Key formats:

| Format                  | Shape      | Where used                                    |
|-------------------------|------------|-----------------------------------------------|
| Controlled arm aa       | `(3, 3)`   | MPC state/action (shoulder, elbow, wrist)     |
| HML263 feature vector   | `(263,)`   | MDM input/output                              |
| SMPL body_pose          | `(21, 3)`  | SMPL model; intermediate conversion format   |
| XYZ joint positions     | `(22, 3)`  | FK output, visualization, clustering features|
| Global joint rotations  | `(22, 3, 3)` | Derived in Kimodo worker for frame-0 constraint |
| Arm chain positions     | `(5, 3)`   | FK arm output (spine3 through wrist)          |
| Spine3 anchor           | `(3,)`×2   | Position + axis-angle; fixed reference frame  |
| Cartesian wrist goal    | `(3,)`     | Spine3-relative target for Cartesian MPC      |
| MPC rollout batch       | `(N,H+1,3,3)` | N samples × H+1 steps × 3 joints × 3 dim |

---

## 11. Key Entry Points

| Command / Script                                      | Purpose                              |
|-------------------------------------------------------|--------------------------------------|
| `uv run python src/.../planners/run.py --mpc-config <yaml>` | Single MPC run (plan → language correction → finish) |
| `uv run python src/.../experiments/run_experiment.py --mpc-config <yaml> [--persona <name>] [--backend agent]` | One simulated-user persona/backend experiment on the original goal |
| `uv run python src/.../experiments/run_cluster_experiment.py --mpc-config <yaml>` | Per-cluster cost comparison experiment |
| `uv run python src/.../experiments/run_backend_experiment.py --mpc-config <yaml> [--persona <name>]` | Per-backend (llm/turns/agent) cost comparison experiment |
| `uv run python src/.../experiments/run_transfer_experiment.py --mpc-config <yaml> [--persona <name>]` | Simulated-user transfer experiment (hidden-cost evaluation on held-out goals); persona defaults to the config's `user:` |
| `uv run python src/.../experiments/render_cost_comparison.py --state state.pkl --response response.json --out cmp.png [--angles-out angles.png] [--archive-dir candidates --save-video]` | Render/archive a candidate cost rollout vs the correction — spatial overlay plus optional joint-angle-over-time graph (agent backend self-service tool) |
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
- **kimodo** (NVIDIA, `github.com/nv-tlabs/kimodo`) — second text-to-motion backend.
  Installed in an **isolated conda env** (`KIMODO_CONDA_ENV`, default `kimodo`) because it
  pins `pydantic>=2` / `transformers==5.1.0`, conflicting with the main env. Invoked via
  `conda run` subprocess (`kimodo/_kimodo_inference_worker.py`), mirroring the SAM/MHR
  worker pattern. Needs gated HF access to `meta-llama/Meta-Llama-3-8B-Instruct`.
- **OpenAI** — used for LLM cost generation (`llm/openai_model.py`)
- **sklearn** — KMeans clustering in `clustering/base.py`
- **smplx** — SMPL model loading
- **detectron2** — human pose estimation in data collection worker

---

## 13. Simulated Users (`simulated_users/`)

Simulated care recipients with **hidden** comfort costs. A user is a standard
part of every run: `build_run` resolves the `user:` config key via
`get_persona()` into `RunSetup.user`, so all entry points carry a
`SimulatedUser` with joint limits alongside the loaded pose (the default
`unrestricted` persona has no bounds). Restricted users default the MDM
instruction to their `feedback_text` (`resolve_feedback_text` in
`planners/run.py`) and, with `uq.user_cluster: true`, pick the UQ cluster.
The hidden bounds are the evaluation ground truth in headless experiments
(never shown to the cost generator).

- `HiddenBound` — one restriction over a shared joint feature (radians):
  `upper_bound` / `lower_bound` / `avoid_band` (painful range), optionally gated
  by a `FeatureCondition` on another feature.
- `CoupledBound` — **pose-dependent limit**: the threshold on one feature moves
  linearly with another (`threshold = intercept + slope * cond_value`), e.g.
  stroke flexor synergy (required elbow bend grows with shoulder elevation).
- `JointBoxLimit` — anatomical per-axis box on one controlled slot's raw
  axis-angle (not a feature bound). `DEFAULT_ARM_JOINT_LIMITS` in `personas.py`
  is shared by **every** persona (including `unrestricted`): a tight box on the
  `left_shoulder` slot (which drives the clavicle under the repo FK convention)
  around the seated `demo_pose.pt` neutral, plus generous boxes on the
  upper-arm and forearm slots. Summed into `compute_violations` and
  `HiddenCostTerm` via `SimulatedUser.limit_violation_series`; the
  feedback-gating checks still test `user.bounds` only, so `unrestricted`
  remains a no-feedback persona.
- `render_hidden_bounds()` (`viz.py`) — one panel per bound for debugging:
  pose-dependent bounds are drawn in the conditioning-vs-bounded feature plane
  with the forbidden region shaded (computed by evaluating `bound.violation` on
  a grid, so the picture cannot drift from the code) and trajectories traced
  through it; simple bounds as feature-over-time with the forbidden range shaded.
- `SimulatedUser` — persona: `name`, clinical `description`, `feedback_text`
  (what the user says when the robot violates a bound), `bounds`.
- Behaviors: `first_violation_step()` (feedback trigger), `choose_cluster()`
  (picks the most comfortable UQ cluster mean), `violation_metrics()`
  (mean/max/frac violated — evaluation metric), `HiddenCostTerm` (oracle
  planner cost adapter implementing `TrajectoryCost`; also used by transfer
  experiments to score scaled raw UQ cluster options before cost generation).
- Features come from the same `GeneratedCostContext` joint-feature helpers the
  cost generator uses (via `feature_series()`), so hidden bounds and generated
  bounds are directly comparable.
- Personas (`personas.py`): `unrestricted` (default; no bounds),
  `adhesive_capsulitis`, `elbow_contracture`, `painful_arc` (elevation avoid band),
  `stroke_flexor_synergy` (coupled bound).
  Registry: `PERSONAS` / `get_persona(name)`.

> **Elbow-flexion feature fix (2026-07-06):**
> `GeneratedCostContext.elbow_flexion_angles` (generator-side) and
> `compute_elbow_flexion_angles` in `costs/base.py` (the `elbow_flexion_angle`
> YAML cost + preference learning) both now measure the angle between the upper
> arm and forearm (0 = straight), computed from the **wrist-slot joint
> rotation** applied to the T-pose bone axes (equivalent to the FK-position
> angle, no FK needed). The old elbow-slot rotvec norm could not see an
> anatomical elbow bend at all — under the repo FK convention (audit §3) the
> bend is encoded in the wrist slot — and instead tracked upper-arm
> re-orientation. Any old `elbow_flexion_angle` min/max values from before
> this fix are on the wrong scale.

> **Shoulder internal/external-rotation feature fix (2026-07-07):**
> `GeneratedCostContext.shoulder_internal_external_rotation_angles` now
> twist-decomposes the **composed collar∘shoulder∘elbow** rotation about the
> T-pose upper-arm axis. Under the same FK convention that composition orients
> the upper-arm bone, so the old shoulder-slot-only twist gave different
> readings for physically identical poses (and read 0 for twist encoded in the
> elbow slot, as position-derived MDM trajectories may do).
> Abduction/adduction and flexion/extension were audited and are correct.
> All joint features (and the cost-side `compute_elbow_flexion_angles` /
> `compute_shoulder_abduction_angles`) are now closed-form joint-angle
> computations — composed slot rotations applied to T-pose bone axes — with
> outputs identical to the previous FK-position geometry (~5× faster on MPC
> rollout batches since no FK positions are computed).

> **Shoulder-elevation feature (2026-07-07):**
> `GeneratedCostContext.shoulder_elevation_angles` — upper-arm angle from
> straight down (0 = arm at the side, pi/2 = horizontal, pi = overhead),
> plane-agnostic. Added because the abduction (lateral component) and flexion
> (depth component) proxies both read ~0 for a vertical upper arm, so **no
> combination of them can bound elevation**: poses with the arm overhead
> satisfied any abd/flexion bound, letting planners satisfy "keep the arm low"
> restrictions while visibly lifting the arm. It is part of
> `build_joint_angle_series` / `_joint_feature_summary` (LLM prompts + plots)
> and of the simulated-user `FEATURE_NAMES`. `adhesive_capsulitis` now caps
> elevation (1.25 rad), `painful_arc` now avoids the 1.05-2.1 rad elevation
> band, and `stroke_flexor_synergy` couples required elbow bend to elevation;
> all previously used evadable abduction bounds.
> `HiddenCostTerm.weight` default rose 1.0 → 10.0 so the oracle condition is
> compliance-first rather than trading small violations for goal progress.

## 14. Motion Generator Backends (`motion_generators/`)

All backends implement the `MotionGenerator` ABC (`motion_generators/base.py`) and are
selected by the `motion_generator` YAML key via the `MOTION_GENERATOR_BUILDERS` registry
in `motion_generators/__init__.py` (mirrors the `COST_BUILDERS` pattern). Planners/experiments
hold a `MotionGenerator` and call its methods; the per-backend "pose" array is treated as
opaque.

**Interface** (abstract unless noted): `load_pose`, `decode_pose`, `build_pose_from_arm_aa`, `generate_left_arm_trajectory`, `generate_left_arm_position_samples`,
and the shared concrete `smpl_positions_to_left_arm_trajectory` (SMPL XYZ → arm axis-angles
via `SmplLeftArmFK`, implemented once on the base).

| Backend  | Class                   | Pose repr            | Generation                                  |
|----------|-------------------------|----------------------|---------------------------------------------|
| `mdm`    | `MdmMotionGenerator`    | HML263 `(263,)`      | in-process diffusion (MDM submodule)        |
| `kimodo` | `KimodoMotionGenerator` | SMPL body_pose `(21,3)` | subprocess to isolated conda env (worker) |

The kimodo backend reuses `hml_smpl_conversion`'s SMPL-side helpers
(`smpl_body_pose_to_arm_aa/_collar_aa/_positions/_spine3_aa`) since kimodo's SMPL-X
`get_amass_parameters()` `pose_body (T,63)` maps directly to SMPL `body_pose (T,21,3)`.
For start poses, `KimodoMotionGenerator` converts SMPL `body_pose (21,3)` through the
same FK used by the visualizer and sends Kimodo visualizer-FK joint positions; the worker retargets them onto Kimodo's
skeleton and builds a frame-0 `FullBodyConstraintSet` from the resulting positions and
global rotations.

**Adding a new backend:** subclass `MotionGenerator`, implement the 5 abstract methods, and
register a builder in `MOTION_GENERATOR_BUILDERS`.
