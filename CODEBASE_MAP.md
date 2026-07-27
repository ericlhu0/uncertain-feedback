# uncertain-feedback Codebase Map

**Last updated:** 2026-07-27
**Branch:** real-env

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
│   │   ├── run.py                    # Single-run CLI + repeated-correction callbacks/artifacts
│   │   ├── correction_session.py     # Edge-triggered repeated correction session state machine
│   │   └── mpc/
│   │       ├── __init__.py           # Public exports
│   │       ├── config.py             # YAML → MpcRunConfig dataclass
│   │       ├── kinematics.py         # SmplLeftArmFK, SMPL topology constants
│   │       ├── arm_features.py        # Canonical q conversion + shared anatomical arm features
│   │       ├── costs/                # Cost package (public surface: mpc.costs)
│   │       │   ├── __init__.py       # Re-exports the public cost API
│   │       │   ├── base.py           # Cost terms + registry + preference learning
│   │       │   ├── generated.py      # Runtime context, cost compile/exec, summaries, image render
│   │       │   ├── cost_generator.py # CostGenerator base + create_cost_generator factory + scoring
│   │       │   ├── llm_costs.py      # backend: llm (interpret→ground→author, single pass) + re-exports
│   │       │   ├── turns_costs.py    # backend: turns (fixed interpret, ground+author refinement)
│   │       │   ├── agent_costs.py    # backend: agent (codex CLI)
│   │       │   ├── combine_costs.py  # Multi-round history + unified replacement-cost agent
│   │       │   ├── cost_feedback.py  # EvalState (picklable rollout state for agent backend)
│   │       │   └── prompts/          # Staged prompt text files
│   │       │       ├── __init__.py   # staged prompt builders + image placeholder substitution
│   │       │       ├── runtime_api.txt       # Shared technical contract
│   │       │       ├── output_contract.txt   # Shared output rules
│   │       │       └── stages/       # interpret.txt, ground.txt, author.txt, refine.txt, combine.txt
│   │       ├── arm_mpc.py            # SmplLeftArmMPC (base sampling MPC)
│   │       ├── arm_mpc_mdm.py        # LeftArmMPCMDM (+ MDM trajectory tracking)
│   │       ├── arm_mpc_mdm_uq.py     # LeftArmMPCMDMUQ (+ UQ clustering)
│   │       ├── arm_mpc_cartesian_base.py  # _CartesianGoalsMixin (shared Cartesian logic)
│   │       ├── arm_mpc_cartesian.py  # LeftArmMPCCartesian (MDM then Cartesian)
│   │       ├── arm_mpc_cartesian_no_mdm.py  # ArmMPCCartesianNoMDM (pure Cartesian)
│   │       ├── arm_mpc_ik_gated.py   # ArmMPCCartesianNoMDMIKGated (human-action MPC gated by the env's robot IK)
│   │       ├── arm_mpc_robot.py      # _RobotActionsMixin + ArmMPCCartesianNoMDMRobot (sample robot joint deltas, cost the human arm)
│   │       ├── arm_mpc_cartesian_robot.py  # LeftArmMPCCartesianRobot (MDM+UQ in robot joint space)
│   │       └── configs/              # Example YAML config files
│   │           ├── arm_mpc_cartesian_mdm.yaml
│   │           ├── arm_mpc_cartesian_mdm_learn.yaml
│   │           ├── arm_mpc_cartesian_mdm_llm.yaml
│   │           ├── arm_mpc_cartesian_mdm_llm_turns.yaml  # backend: turns (multi-turn scored selection)
│   │           ├── arm_mpc_cartesian_mdm_llm_agent.yaml  # backend: agent (codex CLI)
│   │           ├── arm_mpc_cartesian_mdm_llm_transfer.yaml  # simulated-user transfer experiment
│   │           ├── arm_mpc_cartesian_mdm_llm_multiround.yaml # pose-dependent multi-round experiment
│   │           ├── arm_mpc_cartesian_no_mdm.yaml
│   │           ├── arm_mpc_cartesian_no_mdm_sim.yaml  # same, with env: sim_robot_visual
│   │           ├── arm_mpc_cartesian_no_mdm_sim_mannequin.yaml  # same, with env: sim_mannequin (physics; more steps)
│   │           ├── arm_mpc_cartesian_no_mdm_sim_mannequin_kinova.yaml  # same, with robot: kinova_gen3
│   │           ├── arm_mpc_cartesian_no_mdm_sim_mannequin_kinova_real.yaml  # same, mirroring the sim robot onto the real Gen3 (real_mirror_host)
│   │           ├── arm_mpc_cartesian_no_mdm_real.yaml  # env: real (OptiTrack-measured human arm + real Gen3)
│   │           ├── arm_mpc_cartesian_no_mdm_robot_real.yaml  # robot-action sampler on env: real
│   │           ├── arm_mpc_cartesian_no_mdm_robot_sim_mannequin_kinova.yaml  # robot-action sampler rehearsal on sim_mannequin
│   │           └── arm_mpc_cartesian_robot_mdm_llm_real.yaml  # full method with the robot-action sampler on env: real
│   ├── experiments/                  # Multi-run experiment machinery (separate from a single run)
│   │   ├── experiment_pipeline.py    # Staged simulated-user experiment core (trigger, UQ, cost, eval)
│   │   ├── run_experiment.py         # CLI: one persona + one backend on the original goal
│   │   ├── cluster_comparison.py     # Per-cluster rollout + hidden-cost condition evaluation
│   │   ├── run_cluster_experiment.py # CLI entry point for per-cluster comparison experiments
│   │   ├── backend_comparison.py     # Per-backend proxy score + hidden-cost condition evaluation
│   │   ├── run_backend_experiment.py # CLI entry point for per-backend comparison experiments
│   │   ├── transfer_experiment.py    # Adds held-out transfer-goal eval around experiment_pipeline
│   │   ├── run_transfer_experiment.py # CLI entry point for simulated-user transfer experiments
│   │   ├── trajectory_corpus.py      # TrajectoryCorpus: per-session on-disk log of executed trajectories (npy + per-frame feature csv + manifest.json)
│   │   ├── multi_round_experiment.py # Multi-goal feedback history + cost combination loop (logs every goal's executed rollout to trajectory_corpus/ and threads corpus_dir into both codex generators)
│   │   ├── run_multi_round_experiment.py # CLI entry point for multi-round experiments
│   │   ├── episode_loop.py           # Automated simulated-user episode: oracle path → trigger → attribute → verbalize → choose → cost gen → re-trigger loop
│   │   ├── run_episode_experiment.py # CLI entry point for automated simulated-user episodes
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
│   ├── envs/
│   │   ├── __init__.py               # ENV_BUILDERS registry + make_env
│   │   ├── base.py                   # ExecutionEnv ABC (execute/hold; show_goal displays the goal config for envs that can draw it; preview(plan) rolls the plan out before any of it executes — plan(on_step) streams each step through the env's on_step callback as its solve finishes so the env draws the rollout live while it is planned — and returns False to abort the run, the plan passed as a lazy callable so only envs that show it pay for the rollout; set_pose_context attaches the run's FK; initial_q reports the arm's real start config and pose_context the torso anchor to plan against — defaults: the config's, both overridden by envs that measure the person; abstract visualize/save_video; robot-action interface for the robot-action planners — robot_fk/current_robot_q/robot_joint_limits/current_grasp/execute_robot, NotImplementedError by default, implemented by sim_mannequin and real: execute_robot takes a (7,) robot joint target directly (no grasp FK, no IK; delta capped by uniform scaling so a saturating joint slows the motion instead of bending its direction) and returns the achieved *human* q)
│   │   ├── kinematic.py              # KinematicEnv (pass-through; matplotlib visualize/save_video)
│   │   ├── grasp.py                  # grasp_pose_fk: human q → *nominal* gripper grasp pose on the forearm (sim envs); forearm_frame_fk + MeasuredGrasp: gripper-on-forearm transform measured from an ee pose + forearm pose, rigid over one step and re-measured each step by the real env
│   │   ├── human_mesh.py             # HumanMeshBody: posed SMPL body mesh as a pybullet visual body (remove+recreate per pose — no vertex-update API); shared by sim_robot_visual (offscreen) and real (live GUI); `arm_only` draws just the left arm, for the translucent green goal ghost drawn over the person. `ArmSkeletonBody`: the planner's own 5-joint arm chain as bones + joint balls (real geometry, not debug lines, so it survives into `getCameraImage` screenshots and videos; bones built once at their rigid lengths, then only repositioned, so it can refresh every step while the mesh cannot). `BODY_XRAY_COLOR` is the translucent person the real env draws when the chain is inside them
│   │   ├── sim_robot_visual.py       # SimRobotVisualEnv (no physics: kinematic human rendered as posed SMPL mesh + Panda posed via PyBullet IK)
│   │   ├── sim_mannequin.py          # SimMannequinEnv (physics proxy: robot drags passive 4-DOF mannequin arm via fixed constraint; achieved q read back from link positions; robot: panda (vendored) or kinova_gen3 (URDF from /home/emprise/kortex_description); optional real_mirror_host forwards achieved robot q to the real Gen3)
│   │   ├── real_mirror.py            # RealArmMirror: streams sim robot joint configs to the real Kinova Gen3 over ZMQ (emprise-gen3-controller); start(): close gripper → zero → slow streamed ramp to grasp config → confirmed gripper open → 1 kHz joint position mode; start_from_grasp(): position mode only, nothing moved (real env, grasp already taken)
│   │   ├── real.py                   # RealEnv (env: real — human arm measured from OptiTrack rigid bodies via mocap/, including the run's start configuration (initial_q), the torso anchor, placed on the measured collar and read back via pose_context, and the skeleton's arm segment lengths (fk.scale_arm_lengths from the calibration frame's marker distances, so the shared FK plans on the person's proportions); grasp *measured* from the real ee pose + forearm frame (MeasuredGrasp, rigid) and re-measured every step since the real grasp shifts, never taken or assumed — which absorbs the FK-vs-real bias and so lets `_drive` command an absolute gripper pose; the measured transform is trusted as-is (no on-forearm sanity check, so a bad registration, a slipped grasp, or a mirror arm that never grasped will not halt the run); IK (`_solve_ik`) is analytical, via `ssik`'s prebuilt Gen3 artifact (`ssik.prebuilt.gen3_ik`, baked geometry verified identical to the kortex URDF; solves `end_effector_link` in `base_link`, so targets are moved out of the pybullet world and off `tool_frame` by `_ee_target_in_base`): it first asks for the Newton continuation from the arm's current configuration (`_ik_solutions(track=True)` → `solve(q_seed=..., max_solutions=1)`, ~0.3 ms, and it rejects its own step if it lands on a different branch), and only if that misses the controller's padded box does it enumerate every analytical branch and take the nearest one that fits. Continuation comes first because the arm is redundant — a pose has a whole self-motion manifold of exact solutions, and picking whichever scores best moved the solution up to 1.5 rad in a step against `robot_max_joint_delta`. Only when *no* branch fits the box does anything numerical run (`_nearest_infeasible_ik`): a bounded least-squares with position-priority weighting (`_IK_ORIENTATION_WEIGHT_M_PER_RAD`), so the gripper stays on the forearm and the wrist attitude carries the miss. Its residual uses `gen3_ik.fk`, not pybullet's — pybullet holds link state in single precision, so differencing it over the optimiser's ~1e-8 step yields a Jacobian that is mostly rounding; `preview(plan)` draws each planned step in the live view as its solve finishes, against the measured geometry, and aborts the run unless the operator approves, before any command reaches the arm — and prints the grasp error over the plan (`_grasp_error` per step, summarised by `_print_grasp_error`: position/attitude between the commanded grasp pose and the pose actually reached, mean/max/worst step), which catches both an unreachable stretch and a plan that outruns `robot_max_joint_delta`; robot commanded through RealArmMirror; PyBullet DIRECT for IK only, no physics — or GUI with `live_view`, drawing the measured person as a translucent SMPL mesh (rate-limited by `live_view_fps`, shaped to the measured arm via `SmplMeshCache(arm_lengths=...)`) with the planner's own 5-joint arm chain (`ArmSkeletonBody`, refreshed every step) inside it, beside the robot's URDF meshes, plus a translucent green goal-pose arm via `show_goal`; closing the window mid-run does not end the process — `_ensure_backend` (called at each step entry point) detects the dead client, reconnects DIRECT, reloads the robot at the registered base, restores its joint state from the `_set_joints` shadow copy, and the run continues headless)
│   │   ├── robot_fk.py               # RobotChainFK: batched analytic ee-chain FK, extracted once from a loaded pybullet body (base→ee ancestry, vectorized numpy), so a robot-action MPC can roll out thousands of joint samples per step without pybullet
│   │   ├── robot_preview.py          # RobotPlanPreviewEnv: kinematic stand-in snapshotting the real env's robot chain/grasp/joints/limits so preview_planned_trajectory can roll out the actual robot-action planner offline; records the planned robot joint trajectory for the preview animation
│   │   ├── zero_kinova.py            # CLI: zero the real Gen3's joints via the ZMQ server (RealArmMirror.zero)
│   │   ├── assets/panda/             # Franka Panda URDF + meshes (vendored from empriselab/limb-manipulation)
│   │   └── assets/human/             # Mannequin URDFs + meshes: articulated 4-DOF left arm, plus torso/head, right arm, legs as visual context (vendored from empriselab/limb-manipulation)
│   ├── mocap/
│   │   ├── natnet.py                 # Minimal NatNet 3+/4 client: multicast receiver thread + rigid-body frame decoder (no PyPI dep); MocapStaleError, require_fresh
│   │   ├── registration.py           # ArmRegistration: mocap→pybullet rotation solved from the measured left→right collar axis, the torso's mediolateral direction (yaw-only fit, no arm assumption), so pybullet = the mocap world turned by that yaw — torso anchored on the *measured* collar (`spine3_smpl`/`translation_smpl`, frozen for the run, moves between runs) and robot on its measured base; bone directions → planner q; arm_keypoints
│   │   └── monitor.py                # CLI: verify the live stream (validity, frame rate, derived q, rollout video) with no robot involved
│   ├── uncertainty/
│   │   ├── clustering/               # Trajectory clustering methods (CLUSTERER_BUILDERS registry + make_clusterer)
│   │   │   ├── base.py               # TrajectoryClusterer (template: _positions_to_features/_to_features + _fit_predict), medoid_indices, agglomerative_labels
│   │   │   ├── xyz_clusterer.py      # XyzPositionClusterer (KMeans on end-pose features), AggloEndPoseClusterer
│   │   │   ├── path_pca_clusterer.py # PathPcaClusterer (arclength-resampled arm path + PCA, agglomerative)
│   │   │   └── t2m_clusterer.py      # T2mEmbeddingClusterer (T2M motion-encoder embeddings, agglomerative; needs downloaded evaluator weights)
│   │   └── cluster_picker.py         # Recursive matplotlib cluster picker with refine/back navigation
│   ├── simulated_users/
│   │   ├── base.py                   # SimulatedUser, HiddenBound/CoupledBound, violations, cluster choice, oracle cost term
│   │   ├── attribution.py            # attribute_correction: nominal-vs-oracle-window contrast → CorrectionIntent
│   │   ├── verbalizers.py            # vague/everyday/joint_resolved verbalizers + VERBALIZERS registry
│   │   ├── visual.py                 # VisualVerbalizer: VLM speaks from rendered pose images, disk-cached
│   │   ├── chooser.py                # choose_correction: oracle-path lexicographic cluster+magnitude chooser
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
│   ├── demo_runner/                  # Sole browser demo tool: guided pipeline (scenario → language correction → cost generation → apply feedback), demo/dev modes, session replay. Port 6781
│   │   ├── core.py                   # DemoRig: process-lifetime state (config, personas+CRUD, named start/goal configs, motion gen, pose/FK/mesh/context) + begin/list/resume session
│   │   ├── session.py                # Session (one persona: corpus + rounds + unified cost, session.json persistence/resume; rounds record root-to-leaf cluster_labels) and Trajectory (live MPC stepping + per-trajectory correction scratch, including per-level explicit undesirable-cluster marks). Replay defers provisional cluster/cost events and records only the final accepted selection path
│   │   ├── server.py                 # create_app(static_dir) factory holding every pipeline route + boot() (stdout tee + rig) + read-only /api/artifact/<path> rooted at demo artifacts, plus runner-only one-step live MPC routes and /api/replay/<name>[/<i>] (re-mints dead mesh ids from recorded arm_positions)
│   │   └── static/                   # Demo/dev mode toggle + replay UI: session lifecycle bar/resume picker/corpus controls, Three.js body views, persona/feature graphs, generated-cost rationale disclosures; cluster selection is independent of exclusion, with Next gated on an included selection
│   ├── llm/
│   │   ├── base_model.py             # BaseModel ABC (get_full_output)
│   │   └── openai_model.py           # OpenAI wrapper implementing BaseModel (Chat + Responses APIs)
│   └── utils/
│       ├── plot.py                   # ArmVisualizer (live MPC window + static drawing)
│       └── smpl_mesh.py              # SmplMeshCache: coherent SMPL mesh fitted to decoded body joints, arm re-posed per frame (demo runner UI + sim_robot_visual + real envs); optional `arm_lengths` fits `betas` to measured clavicle/upper-arm/forearm lengths (only bone *directions* survive into a mesh pose, so a neutral body draws a neutral-length arm — cm-scale error at the wrist once the planner's FK is calibrated to a person); every pose is shifted to land the mesh's shoulder on the FK's, so the drawn arm sits on the chain the MPC costs instead of on the torso fit's residual
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
│  State: (7,) [clavicle rotvec, shoulder rotvec, elbow flexion angle].
│  Actions only use the shoulder + elbow DOFs — the clavicle slots are zeroed
│  in _sample_actions (a robot holding the forearm cannot actuate the
│  shoulder girdle), so the clavicle stays at its (measured) current value.
│  Warm-start: shifts previous best plan by one step each iteration.
│
├── LeftArmMPCMDM (arm_mpc_mdm.py)
│     Adds: validate_trajectory() (advisory safety-cost check), push_trajectory()
│     (validate + queue for rate-limited playback), set_mdm_goal(),
│     MDM-colored arm rendering, body_pos background skeleton. The MDM trajectory
│     is played back directly (no per-step sampling) at a bounded angular speed
│     (max_playback_delta per joint per step), so jumps are eased not snapped;
│     the MPC then resumes sampling toward the final goal. A new push replaces
│     any unfinished playback suffix; remaining_mdm_trajectory() snapshots it.
│
│   └── LeftArmMPCMDMUQ (arm_mpc_mdm_uq.py)
│         Adds: query_mdm_with_uncertainty() — draws N diffusion samples,
│         clusters with XyzPositionClusterer, shows cluster picker UI (each
│         cluster has a magnitude slider that scales the chosen motion about
│         its start pose via scale_trajectory); the user can recursively
│         re-cluster a selected cluster's raw samples and back up through the
│         hierarchy before enqueuing the final subset mean. Headless selection:
│         `auto_cluster` (fixed label)
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
    │   └── LeftArmMPCCartesianRobot (arm_mpc_cartesian_robot.py)
    │         + _RobotActionsMixin. Same corrections/UQ/LLM machinery, but both
    │         phases sample robot joint deltas and command the robot directly
    │         (env.execute_robot). Playback keeps its rate-limited cursor; each
    │         frame becomes the target of a robot-space tracking solve (geodesic
    │         shoulder error — measured rotvecs sit near the ±pi boundary).
    │
    └── ArmMPCCartesianNoMDM (arm_mpc_cartesian_no_mdm.py)
          Cartesian MPC only — no MDM, no UQ.
          Decodes optional HML pose to initialize body but generates nothing.

        ├── ArmMPCCartesianNoMDMIKGated (arm_mpc_ik_gated.py)
        │     Samples human-arm deltas, but checks each rollout's leading
        │     gripper poses with the environment's execution IK. RealEnv uses
        │     the same analytical Gen3 branch continuation/enumeration and
        │     padded controller joint box as execution, seeded sequentially
        │     through the rollout. Continuation across all MPC samples is a
        │     vectorized Gen3 FK/Jacobian solve; failures fall back individually
        │     to analytical enumeration. Residuals above max_grasp_ik_residual
        │     are gated; an explicit zero-motion sample makes hold the safe
        │     fallback. The execution-rate cap is deliberately not part of IK.
        │
        └── ArmMPCCartesianNoMDMRobot (arm_mpc_robot.py)
              + _RobotActionsMixin (arm_mpc_robot.py): samples robot joint
              deltas instead of human-arm deltas — every rollout is
              robot-feasible by construction, and each sampled action is capped
              at max_robot_joint_delta (inf-norm, uniformly scaled), the same
              cap execute_robot enforces. Rollouts go robot FK (RobotChainFK)
              → inverse of the rigid MeasuredGrasp → grasp-position-anchored
              projection (project_forearm_frames, kinematics.py) onto the 4-DOF
              arm manifold, so every cost term still receives (N, H+1, 3, 3)
              human-arm axis-angles unchanged. Rollouts whose leading frames
              (grasp_residual_frames) exceed max_grasp_residual are discarded
              outright — they would break the grasp; the squared whole-horizon
              residual (elbow displacement + untransmittable forearm roll) is
              also soft-penalized (robot_infeasibility_weight). Requires
              env: real or sim_mannequin.
```

**YAML `planner` → class mapping:**

| `planner` value                    | Class                       |
|------------------------------------|-----------------------------|
| `arm_mpc`                          | `SmplLeftArmMPC`            |
| `arm_mpc_mdm`                      | `LeftArmMPCMDM`             |
| `arm_mpc_mdm_uq`                   | `LeftArmMPCMDMUQ`           |
| `arm_mpc_cartesian`                | `LeftArmMPCCartesian`       |
| `arm_mpc_cartesian_no_mdm`         | `ArmMPCCartesianNoMDM`      |
| `arm_mpc_cartesian_no_mdm_ik_gated` | `ArmMPCCartesianNoMDMIKGated` |
| `arm_mpc_cartesian_robot`          | `LeftArmMPCCartesianRobot`  |
| `arm_mpc_cartesian_no_mdm_robot`   | `ArmMPCCartesianNoMDMRobot` |

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
    │       (Demo Runner instead builds the clusterer via make_clusterer
    │        from uq.clusterer / its UI dropdown, and represents each cluster
    │        by its medoid sample rather than the mean)
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
    │   CorrectionSession monitors each executed pose. The first text-time or
    │   discomfort event starts MDM; later discomfort threshold crossings replace
    │   the active suffix and start MDM again from the actual pose. Per-correction
    │   generated costs stack during execution and are unified at trajectory end.
    │   Each MPC step: sample N×H action sequences, rollout, compute cost, take best
    │
    ├── Joint-space cost: L2 to current goal in 7-DOF q-space
    └── Extra costs: CompositeTrajectoryCost terms [costs/base.py]
            ElbowHeightCost
            ElbowFlexionAngleCost
            ShoulderAbductionAngleCost
            GeneratedPythonCost [costs/llm_costs.py]
    │
    ▼  next_q (7,) each step; converted to (3,3) arm aa for visualization
    │
ArmVisualizer.update_step()                          [utils/plot.py]
    Live matplotlib 3D window + optional video capture
```

---

## 5. Kinematics (`kinematics.py`)

- **`SmplLeftArmFK`** — loads SMPL neutral PKL once; stores T-pose bone offsets for the arm chain. `scale_arm_lengths(clavicle, upper_arm, forearm)` rescales the arm bones to measured segment lengths (directions and hinge axis unchanged, idempotent) — `RealEnv._register` calls it with the calibration frame's marker distances, so real runs plan on the person's proportions
- Joint chain: `spine3 (9) → left_collar (13) → left_shoulder (16) → left_elbow (18) → left_wrist (20)`
- MPC controls 7 DOFs: clavicle rotvec (3), shoulder rotvec (3), and elbow flexion (1) — but samples actions only in the shoulder + elbow slots (see Section 3)
- Key methods:
  - `fk(arm_aa, spine3_pos, spine3_aa) → (5, 3)` world positions
  - `fk_batch(arm_aa, ...) → (N, 5, 3)` batched
  - `q_reaching_wrist(fk, wrist_target, q_seed, ...) → (7,)` inverse of a Cartesian goal: nearest configuration whose wrist hits a world point (least squares, pulled toward `q_seed` since 3 constraints leave the posture free) — used to show a Cartesian goal as a *pose*
  - `bone_world_rotations(arm_aa, spine3_aa) → [Rotation] * 4` the rotations `fk` applies to each bone ([spine3→collar, clavicle, upper arm, forearm]); the forearm entry is the frame a measured grasp rides (`envs/grasp.py`), flip-free unlike a positions-plus-world-up frame
  - `arm_aa_from_positions(positions, spine3_aa) → (3, 3)` inverse: XYZ → local axis-angles
  - `arm_aa_to_q(arm_aa, spine3_aa) → (7,)` boundary conversion; off-hinge elbow rotation is anatomically decoded
  - `q_to_arm_aa(q, elbow_hinge_axis) → (..., 3, 3)` FK/visualization/cost boundary
  - `full_body_positions(arm_aa, ...) → (22, 3)` for visualization
- **`anatomical_elbow_wrist_slots(...)`** (module function) — anatomically-constrained
  elbow + wrist slot rotations for the left arm, shared by the three reconstruction sites
  (`arm_aa_from_positions`, `hml_smpl_conversion.positions_to_smpl_body_pose`,
  `utils/smpl_mesh._generate`). The elbow slot carries the recovered shoulder
  internal/external rotation (twist about the upper arm onto the observed flexion-plane
  normal); the wrist slot is a pure forearm hinge with locked (neutral) pronation,
  referenced to the stable elbow frame. Positions are preserved exactly. See audit §3a.
- **`project_forearm_frames(fk, ee_pos, forearm_rot, grasp_offset, q_ref, ...)`** (module
  function) — batched projection of grasp-implied forearm frames onto the 4-DOF arm
  manifold, **anchored on the gripper position** (the component a grasp couples firmly,
  while orientation slips — which is also what the physics does): shoulder fixed from
  `q_ref`'s clavicle, elbow at the nearest point of the two-sphere intersection circle
  (upper-arm length ∩ rigid elbow-to-gripper distance), forearm rotation minimally
  re-anchored so the gripper stays at the commanded ee position, roll dropped. Returns
  `(..., 3, 3)` arm axis-angles (vectorized `anatomical_elbow_wrist_slots` conventions),
  projected wrist positions, and a residual (elbow projection metres + untransmitted
  roll radians). The robot-action planners' rollout→cost boundary (~45 ms per 1000×11
  batch). The earlier direction-preserving (non-anchored) projection made the sampler
  *diverge* on sim_mannequin — the optimizer exploited motion the projection discarded.

---

## 6. Cost System (`costs/base.py`)

### Cost term protocol

All cost terms implement `TrajectoryCost`: callable `(q_trajs: np.ndarray) → np.ndarray` with `(N,)` per-rollout output. The MPC currently decodes its `(N, H+1, 7)` state batch to the `(N, H+1,3,3)` FK/cost boundary once before invoking the cost system; every anatomical feature immediately canonicalizes through `arm_features.py`, so q and boundary inputs have identical feature semantics. Generated costs receive the decoded boundary for source compatibility, while their context trajectories are canonical q.

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

1. `build_motion_summaries()` — text summaries of recent MPC steps and the MDM trajectory, plus rollout-labeled chosen-vs-marked-wrong terminal joint-feature comparisons when explicitly rejected UQ candidates are available
2. `render_prompt_images()` — delegates to `ArmVisualizer.render_trajectory_overlay()` for the 3-view overlay (optional)
3. Staged prompt builders (in `costs/prompts/__init__.py`) assemble `interpret` (instruction + images + compact summary), `ground` (interpretation + full summaries), and `author` (numeric spec + `runtime_api.txt`/`output_contract.txt`) prompts from `costs/prompts/stages/*.txt`
4. LLM (OpenAI, configurable model) streams its opt-in reasoning summary to stdout, then returns final author JSON: `{description, code, params, explanation, recipient_explanation}`
5. `parse_llm_cost_response()` → `LlmCostResponse`
6. `GeneratedPythonCost.__post_init__()` → `compile_generated_cost()` compiles the code snippet
7. `GeneratedCostContext` provides the runtime sandbox: `fk`, `spine3_pos/aa`, `current_q`, `mdm_traj`, `recent_q`, and FK helper methods
8. Artifacts (stage prompts/responses, `stage_log.md`, images, cost.py, `reference_with_correction.mp4`) saved to `llm_cost_artifacts/<timestamp>/`; `CostGenerator.save_rationale()` also writes `rationale.json`, chaining the instruction, self-reported modality evidence, grounded terms and per-number sources, final explanations, and the winning `CostRanking` table (or `null` when unavailable)

**LLM cost cluster experiment** (`llm_cost.cluster_experiment.enabled`): runs the LLM cost on each cluster's mean trajectory for `rollout_steps` steps and uses costs to rank / auto-select clusters.

### Cost-generation backends (`costs/cost_generator.py`)

`create_cost_generator()` selects one of three iteration mechanisms via `llm_cost.backend`; all share the staged prompting strategy and `CostGenerator` (stage helpers, compile/validate, save, install):

- `llm` (`llm_costs.py`) — single-pass staged calls: interpret → ground → author; only the author output is parsed/compiled. Stage prompt/response pairs are aggregated in `stage_log.md`; the authored cost is ranked for `rationale.json`.
- `turns` (`turns_costs.py`) — interprets once, then runs a stateful ground+author conversation; keeps the best cost by ranking consistency (`rank_candidate_cost`, falling back to the L2 rollout score when the context has no comparison trajectories). `stage_log.md` includes the interpretation plus each refine turn's prompt snapshot and response, and the winning turn's ranking is copied into `rationale.json`.
- `agent` (`agent_costs.py`) — delegates the staged method to the `codex` CLI, which emits the same `response.json` and must write `stage_log.md` with Stage 1 / Stage 2 / Stage 3 responses; those sections are parsed leniently for `rationale.json` and the final cost is ranked locally.
- `CombineCostGenerator` (`combine_costs.py`) is constructed directly by the multi-round experiment, not selected as a backend. It replays all successful `CostRound` contexts and replaces every prior `GeneratedPythonCost` with one unified constant or pose-dependent cost. Its user-visible Codex agent output is teed live to stdout (and therefore the Demo Runner console) while remaining in `codex.log`. Its `scores.json` evaluates that same cost independently against every round's pickled `EvalState`. Each `CostRound` also carries the round's root-to-leaf `cluster_labels` path and generation evidence chain — `description`, `explanation`, `interpretation` (stage-1 response), `grounding` (stage-2 response), all defaulted to `""` — populated from the round's `rationale.json` (via `CostGenerationResult`/`_rationale_fields` in `experiment_pipeline.py`) and rendered under a "Why this cost was generated:" block in the combine prompt (`build_combine_task_body`, empty fields skipped).
- Every generator accepts an optional `corpus_dir: Path | None` threaded from `generate_cost_for_cluster` through `create_cost_generator`. `llm` and `turns` receive per-entry accepted-pose feature ranges in their grounding prompt; agent/combine retain the staged per-frame corpus workflow. Shared `CostGenerator` validation evaluates each authored cost on stationary two-frame rollouts of every pose before `comfortable_until` and rejects the cost if any accepted pose receives positive cost. `None` leaves non-corpus callers unchanged.

### Cost evaluation & visual feedback

- `rank_candidate_cost()` (`cost_generator.py`) evaluates a candidate cost by **ranking consistency**: the cost is applied directly to the trajectories whose preference order the user revealed — the chosen correction (`mdm_traj`) must cost strictly less than the original plan (`reference_traj`) and every UQ cluster the Demo Runner user explicitly marked wrong (`context.rejected_trajs`). Unmarked non-chosen clusters are dropped from summaries, images, and ranking; automated/simulated-user paths pass no negatives and rank only against the reference plan. All comparison trajectories are first resampled to equidistant joint-space-arclength points (`resample_equidistant`) so only the path matters, not timing (MDM output is systematically slower than a fresh rollout — timing is a pipeline artifact, not intent), then compared after z-normalization. Returns a `CostRanking` (rank accuracy + normalized margin + inert flag + explicit original-plan-improvement flag; `sort_key` orders candidates), or `None` when the context has no comparison trajectories — then `turns` falls back to the L2 rollout score. Every generator backend rejects rather than installs a cost whose chosen-correction score is not strictly lower than its original-plan score. The Demo Runner threads its per-level `undesirable_labels` into `generate_cost_for_cluster`; `build_motion_summaries` exposes rollout-labeled `candidate_comparison` entries containing chosen, current, original-plan, and per-marked-wrong-cluster terminal values and deltas.
- `evaluate_candidate_cost()` rolls the goal-seeking MPC with a candidate cost installed (`_make_cost_eval_rollout` in `run.py`) and returns the mean FK-position L2 distance to the MDM correction (`_score_rollout`; both trajectories resampled equidistant-in-arclength, so the score is path-only). Lower is better; `inf` when the planner has no Cartesian goal. Drives the `backend_comparison.json` ranking and the `turns` fallback.
- **Goal-reachability check** (`goal_reach_report()`): because ranking/`_score_rollout` grade only against the correction preferences, a cost can score well while stopping the arm short of the goal. `goal_reach_report(context, rollout)` reproduces the MPC's own `ArmMPCCartesian.goal_reached` criterion (FK the final rollout frame, spine3-relative wrist vs `context.cartesian_goal` within `context.cartesian_threshold`), returning `{reached, distance, threshold}` (or `None` for non-Cartesian planners). The goal + threshold are threaded onto `GeneratedCostContext` via `build_generated_cost_context` / `EvalState`. This is gated on `goal_conflict` (`parse_goal_conflict()` reads the stage-1 flag): only when stage one judged the goal reachable does missing it count against a candidate. In the `turns` backend selection is keyed on `(reach_rank, *ranking.sort_key)` so a goal-reaching candidate beats a non-reaching one and ranking consistency orders candidates within that, and each turn's feedback tells the model whether the goal was reached; in the `agent` backend `render_cost_comparison.py` prints a `goal reached:` line and `TASK.md` instructs codex to keep revising until the goal is reached (unless it conflicts).
- `CostGenerator.begin()` writes `reference_with_correction.mp4` into every generated-cost artifact directory using `context.full_correction_traj` when present, falling back to `context.mdm_traj`.
- `evaluate_and_render()` does the **same single rollout** and additionally renders `ArmVisualizer.render_cost_feedback_overlay()` — a rollout (red) vs target-corrected-path (green) overlay — returning `(score, image_path)`. When an `angle_path` is passed it also renders `ArmVisualizer.render_joint_angle_comparison()` — a joint-angle-over-time graph (one subplot per anatomical joint feature; green target vs red rollout, per-frame series from `build_joint_angle_series()`) — so the model sees the temporal shape of the motion, not just endpoints. The green target is `context.full_correction_traj` when present (the **entire** intended path: pre-correction history → MDM correction → comfort-only continuation to the goal, assembled once in `run.py:_assemble_full_correction_traj` and threaded through `build_generated_cost_context` / `EvalState`), falling back to the MDM correction segment (`mdm_traj`) alone. The score (`_score_rollout`) still measures distance to the MDM correction segment only. It can also persist the exact rollout array and MP4 used for that feedback image. When `use_images` is on:
  - `turns` feeds `turn_<i>/comparison.png` + `turn_<i>/angles.png` + score back through the conversation each turn; backend experiments with `--save-video` also write `turn_<i>/rollout.npy` and `turn_<i>/rollout.mp4`.
  - `agent` self-iterates: `EvalState` (`costs/cost_feedback.py`, picklable bundle that rebuilds the rollout + context off-process) is saved to `state.pkl`, the initial overlay paths are listed as text in `TASK.md`, and codex is instructed to load those local image files and append image observations, revision rationale, and the final stop reason to `ITERATION_LOG.md`. Before serialization, the full `MpcRunConfig` is reduced to frozen `EvalMpcConfig`, which retains only operational rollout fields (planner, Cartesian goals/threshold, step/sampling parameters, goal threshold, and seed); `user`, `persona_goals`, and all other config metadata are absent. `EvalState.load()` converts legacy full-config pickles, and combine staging always load/re-saves each round state so old sessions are sanitized too. Codex runs `experiments/render_cost_comparison.py` (writing both `comparison.png` and `angles.png`) to render and inspect its own rollout before finalizing `response.json`; the wrapper appends `ITERATION_LOG.md` into `codex.log`. With `--archive-dir` and `--save-video`, each self-check is archived under `candidate_<i>/` with JSON, score, rollout, and MP4 artifacts.

---

## 8. Configuration System (`config.py`)

`MpcRunConfig` (frozen dataclass) loaded from YAML by `load_mpc_config(path)`.

**Top-level YAML keys:**

| Key                    | Type     | Purpose                                               |
|------------------------|----------|-------------------------------------------------------|
| `planner`              | str      | One of the 8 planner choices                         |
| `motion_generator`     | str      | Text-to-motion backend: `mdm` (default) or `kimodo`  |
| `env`                  | str      | Execution environment realizing each MPC step: `kinematic` (default), `sim_robot_visual` (no-physics PyBullet scene with a Panda IK'd to a forearm grasp point), `sim_mannequin` (physics proxy: Panda drags the passive limb-manipulation mannequin arm; achieved q measured back from link positions), or `real` (real world: human arm measured from OptiTrack rigid bodies, real Gen3 commanded over ZMQ). Passed to the planner constructor (`build_run`, demo runner) after `set_pose_context`; every planner's `step` returns the env-achieved q. Envs render via `visualize()`/`save_video()` (`run.py --env-video`). |
| `env_params`           | mapping  | Keyword args forwarded to the env constructor by `make_env` (empty default). `sim_mannequin` accepts `robot` (`panda` default, or `kinova_gen3` — Gen3 7-DOF + Robotiq 2F-85 URDF loaded from `/home/emprise/kortex_description`), `robot_max_joint_delta` (per-step robot joint travel cap, rad), `robot_base_offset` (robot base position relative to spine3, pybullet frame), `robot_joint_limit_padding` (rad; shrinks robot joint limits so commands clear the real controller's soft limits), and — kinova_gen3 only — `real_mirror_host`/`real_mirror_confirm_start` (mirror the sim robot's achieved joint trajectory onto the real Gen3 via an emprise-gen3-controller ZMQ server, planning against the controller's enforced joint-limit table; see `envs/real_mirror.py`). `real` accepts `mocap_host` (OptiTrack PC address), `mocap_rigid_bodies` (Motive streaming ids for `robot_base`/`collar`/`collar_right`/`shoulder`/`elbow`/`wrist`; the right-collar body — read at calibration only — makes the registration yaw measurable: the left→right collar axis is the torso's facing, so its measured direction solves the yaw with no arm assumption, and the robot is loaded at that solved yaw), `mocap_hold_timeout` (s, default 0.5 — dropout hold before raising `MocapStaleError`), plus `robot`, `robot_max_joint_delta`, `robot_joint_limit_padding`, `real_mirror_host` (null = mocap-only dry run, no command reaches the arm), `real_mirror_confirm_start` (one prompt before tracking starts — the real env moves nothing at startup, since the grasp must already be taken and is measured from the real ee pose), `live_view`/`live_view_fps` (PyBullet GUI window with the measured person as an SMPL mesh next to the robot's meshes; needs a display, mesh refresh rate-limited because each refresh replaces the whole mesh), and `preview_plan` (default true; draw each planned step in the live view as it is solved and wait for operator approval before the first command — needs `live_view`). |
| `steps`                | int      | Total MPC steps to run                               |
| `horizon`              | int      | MPC look-ahead steps                                 |
| `n_mpc_samples`        | int      | Candidate action sequences per step                  |
| `seed`                 | int      | MPC action-sampling seed (default 0); also locks each MDM generation request in Demo Runner |
| `max_angle_delta`      | float    | Sampling std dev (radians)                           |
| `max_robot_joint_delta` | float   | Robot-action planners only: per-step inf-norm cap on sampled robot joint deltas (rad, default 0.005), the same cap `execute_robot` enforces. The env's `robot_max_joint_delta` remains the hardware backstop, applied in `execute_robot` with uniform scaling (whole delta slowed) rather than the IK path's per-joint clip. |
| `robot_joint_delta_std` | float \| null | Robot-action planners only: std of the joint-delta sampling noise around the warm-started previous plan (rad, default null = a third of the cap). Keep it well below `max_robot_joint_delta` — at std == cap nearly every sample saturates the inf-norm cap and the uniform rescale drowns the warm-started mean in noise. |
| `robot_infeasibility_weight` | float | Robot-action planners only: weight on the squared grasp-transmission residual from `project_forearm_frames` (default 1.0) — penalizes sampled robot motions the arm cannot follow (elbow off the upper-arm sphere, forearm roll a parallel-jaw grasp cannot transmit). |
| `max_grasp_residual` / `grasp_residual_frames` | float / int | Robot-action planners only: hard gate — rollouts whose first `grasp_residual_frames` frames (default 3) exceed `max_grasp_residual` per frame (default 0.02; m + rad) are discarded before the argmin (fallback: least-violating sample if none pass). Only the leading frames are gated since only they get executed; the tail stays soft-penalized. The residual floor scales with `max_robot_joint_delta`; tighter = stricter grasp preservation, slower progress. |
| `max_grasp_ik_residual` | float | IK-gated human-action planner only: maximum gripper pose error (metres + radians, default 0.001) over the leading `grasp_residual_frames`. The real env evaluates it with the same analytical IK and padded joint-limit box as execution. |
| `pose`                 | path?    | HML pose `.pt` file for initial body state           |
| `arm`                  | 3×3 list? | Initial left-arm `[shoulder, elbow, wrist]` axis-angle override on top of `pose` (same semantics as `--arm`, which wins when both are given) |
| `goal_threshold`       | float    | L2 dist threshold to pop goal (default 0.01)         |
| `advance_threshold`    | float    | Goal-advance threshold for the MPC resume phase (default 0.1). MDM frames are now played back directly, so this no longer governs MDM frame advancement. |
| `max_playback_delta`   | float    | Max per-joint rotation (radians) per step while following the MDM trajectory (default 0.1). Rate limit: caps playback angular speed so the initial jump into the trajectory and any large frame-to-frame jump are eased rather than snapped. |
| `trajectory_fraction`  | float    | Fraction of MDM frames to enqueue (default 1.0)      |
| `num_denoising_steps`  | int?     | Kimodo DDIM steps (kimodo backend only; None = backend default 100) |
| `preference_learning`  | bool     | Auto-update cost bounds from MDM (default true)      |
| `preference_alpha`     | float    | Blend weight for preference update (default 0.5)     |
| `preference_window`    | int      | MPC step history for preference update (default 50)  |
| `user`                 | str      | Simulated-user persona name (default `unrestricted`); loaded by `build_run` into `RunSetup.user` for every run |
| `corrections.*`        | CorrectionConfig | `trigger_threshold` (default 0.02 rad). Restricted users trigger on a new above-threshold episode after returning to comfort; legacy `transfer.trigger_threshold` is accepted as a fallback. |
| `uq.*`                 | UqConfig | `diffusion_samples`, `n_clusters`, `clusterer` (kmeans_end_pose \| agglo_end_pose [default] \| agglo_path_pca \| agglo_t2m; Demo Runner dropdown default), `auto_cluster`, `scale` (default motion-magnitude scale for the chosen cluster; slider initial value in the GUI, applied directly when headless), `user_cluster` (delegate cluster choice to the configured user when it has bounds; precedence over `auto_cluster`/GUI) |
| `cartesian.*`          | CartesianConfig | `goals` (list of [x,y,z]), `threshold`        |
| `costs.*`              | dict     | Named cost terms with their params                   |
| `llm_cost.*`           | LlmCostConfig | `enabled`, `model` (default `gpt-5.6-luna`; reasoning effort follows the model: `gpt-5.6-luna` → `xhigh`, `gpt-5.6-sol` → `low`), `strict`, `artifact_dir`, `use_images`, `backend`, `max_turns`, `codex_cmd` |
| `transfer.*`           | TransferConfig | `goals` (held-out spine3-relative wrist targets); legacy configs may still provide `trigger_threshold` as a fallback for `corrections.trigger_threshold` |
| `persona_goals.*`      | dict[str, PersonaGoals] | Per-persona override of `cartesian`/`transfer` goals for simulated-user experiments (applied by `experiment_pipeline.apply_persona_goals`; falls back to top-level goals when absent). Each restriction needs its own goal geometry to make the default plan visibly require a correction. |
| `simulated_user.*`     | SimulatedUserConfig | Automated episode settings (`run_episode_experiment.py`): `verbalizer` (`vague` \| `everyday` [default] \| `joint_resolved` \| `visual`), `seed` (everyday sampling rng), `max_rounds` (default 3; capped episodes log as failures), `magnitudes` (chooser grid, default `[0.5, 0.75, 1.0, 1.25, 1.5]`), `nominal_steps` (base-MPC continuation length for attribution, default 20) |

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
| Planner arm state       | `(7,)`     | Clavicle (3), shoulder (3), elbow flexion (1)|
| Controlled arm aa       | `(3, 3)`   | FK, visualization, costs, motion-gen boundary|
| HML263 feature vector   | `(263,)`   | MDM input/output                              |
| SMPL body_pose          | `(21, 3)`  | SMPL model; intermediate conversion format   |
| XYZ joint positions     | `(22, 3)`  | FK output, visualization, clustering features|
| Global joint rotations  | `(22, 3, 3)` | Derived in Kimodo worker for frame-0 constraint |
| Arm chain positions     | `(5, 3)`   | FK arm output (spine3 through wrist)          |
| Spine3 anchor           | `(3,)`×2   | Position + axis-angle; fixed reference frame  |
| Cartesian wrist goal    | `(3,)`     | Spine3-relative target for Cartesian MPC      |
| MPC rollout batch       | `(N,H+1,7)` | N samples × H+1 steps × 7 arm DOFs          |

---

## 11. Key Entry Points

| Command / Script                                      | Purpose                              |
|-------------------------------------------------------|--------------------------------------|
| `uv run python src/.../planners/run.py --mpc-config <yaml>` | Single MPC run (plan → language correction → finish) |
| `uv run python src/.../experiments/run_experiment.py --mpc-config <yaml> [--persona <name>] [--backend agent]` | One simulated-user persona/backend experiment on the original goal |
| `uv run python src/.../experiments/run_cluster_experiment.py --mpc-config <yaml>` | Per-cluster cost comparison with base/oracle/generated hidden-cost evaluation for Cartesian goals |
| `uv run python src/.../experiments/run_backend_experiment.py --mpc-config <yaml> [--persona <name>]` | Per-backend (llm/turns/agent) cost comparison with base/oracle/generated hidden-cost evaluation |
| `uv run python src/.../experiments/run_transfer_experiment.py --mpc-config <yaml> [--persona <name>]` | Simulated-user transfer experiment (hidden-cost evaluation on held-out goals); persona defaults to the config's `user:` |
| `uv run python src/.../experiments/run_multi_round_experiment.py --mpc-config <yaml> [--persona <name>]` | Multi-round cost experiment; `cartesian.goals` is the ordered round sequence and successful feedback contexts are unified into one replacement cost |
| `uv run python src/.../experiments/run_episode_experiment.py --mpc-config <yaml> [--persona <name>] [--save-video]` | Automated simulated-user episode: reactive verbalized feedback + oracle-path cluster choice + re-trigger loop until the goal is reached cleanly |
| `uv run python src/.../experiments/render_cost_comparison.py --state state.pkl --response response.json --out cmp.png [--angles-out angles.png] [--archive-dir candidates --save-video]` | Render/archive a candidate cost rollout vs the correction — spatial overlay plus optional joint-angle-over-time graph (agent backend self-service tool) |
| `uv run python src/.../demo_runner/server.py [--mpc-config <yaml>] [--trajectory-configs-file <json>] [--port 6781]` | Browser tool (run from the repo root; artifact root is CWD-relative). Named initial-pose and goal libraries persist through `/api/trajectory-configs/<kind>`; clicking the header summary opens a dropdown to start/resume/delete sessions (one locked persona), the corpus panel browses/deletes executed evidence, and session-owned rounds/unified costs carry into successive trajectories; sessions persist to `session.json` and are resumable after restart (`/api/session/start`, `/api/sessions`, `/api/session/resume`, `DELETE /api/sessions/<name>`, `DELETE /api/corpus/<i>`); cluster selection exposes `/api/pick_cluster` and the explicit-negative `/api/mark_cluster`; pending-cost and committed-round payloads include `rationale`, rendered as *why this cost*, while `/api/artifact/<path>` serves their `rationale.json`/`stage_log.md` files from the demo artifact root (see README). Both **demo** and **dev** use a collapsible Scenario configuration above the guided Trajectory decision → Language correction → Cost generation → Apply feedback stages; it auto-collapses on trajectory start while Start/Exit and live status remain visible. The right column begins with persistent pending/active cost summaries whose disclosures show the generated Python; a unified cost replaces individual round entries after combination. Runner-only `/api/live_trajectory/start`, `/step`, and `/apply_round` routes animate initial and post-cost MPC execution one frame at a time until discomfort or completion; tab 3 streams cost-generation progress and the opt-in OpenAI reasoning summary through `/api/logs` character cursors. Scenario always exposes saved start-pose/goal selection, the decision stage chooses correction or ignore/continue, cluster selection remains in Language correction until its explicit Next action, and completed cost generation similarly waits for an explicit Next action before Apply feedback; refinement and Exclude controls remain available in both modes, cluster oracle/violation diagnostics are dev-only, completed stages are reviewable, and repeated feedback returns to the decision. Dev exposes advanced controls inside this organization; demo hides scenario authoring, bound dragging, sampling knobs, cost backend/code in the workflow panel, corpus, and console. The Cost functions panel's Combine rounds (codex) action is visible in both modes. The choice persists in `localStorage`. Every session records a beat stream to `<session>/replay/` (`index.json` + one `NNNN_<kind>.json` per beat, each with the payload the UI received and a per-beat persona snapshot); `GET /api/replay/<name>` and `/api/replay/<name>/<i>` serve it, and Replay steps through it with no MDM/LLM/MPC calls while synchronizing the stage navigator |
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
- **sklearn** — KMeans + AgglomerativeClustering in `clustering/base.py`, PCA in `clustering/path_pca_clusterer.py`
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
  is shared by **every** persona (including `unrestricted`): generous boxes on
  the upper-arm and forearm slots; the clavicle-driving `left_shoulder` slot is
  currently unrestricted. Summed into `compute_violations` and
  `HiddenCostTerm` via `SimulatedUser.limit_violation_series`; the
  feedback-gating checks still test `user.bounds` only, so `unrestricted`
  remains a no-feedback persona.
- `render_hidden_bounds()` (`viz.py`) — one panel per bound for debugging:
  pose-dependent bounds are drawn in the conditioning-vs-bounded feature plane
  with the forbidden region shaded (computed by evaluating `bound.violation` on
  a grid, so the picture cannot drift from the code) and trajectories traced
  through it; demo bounds as feature-over-time with the forbidden range shaded.
- `SimulatedUser` — persona: `name`, clinical `description`, `feedback_text`
  (what the user says when the robot violates a bound), `bounds`.
- Hidden feature bounds are soft oracle penalties. Persona descriptions distinguish
  physical/symptom-limited ranges from documented movement preferences; they are
  not enforced as hard simulator stops.
- Behaviors: `first_violation_step()` (feedback trigger), `choose_cluster()`
  (picks the most comfortable UQ cluster mean), `violation_metrics()`
  (mean/max/frac violated — evaluation metric), `HiddenCostTerm` (oracle
  planner cost adapter implementing `TrajectoryCost`; also used by transfer
  experiments to score scaled raw UQ cluster options before cost generation).
- `attribute_correction()` (`attribution.py`) — deterministic feedback
  attribution for the automated episode loop: joins the trigger pose to the
  nearest oracle-path waypoint (`j ≥ min_join`), contrasts the robot's nominal
  continuation against the oracle window as signed feature deltas
  (nominal − oracle, four features) plus world-frame wrist/elbow offsets, and
  returns a frozen `CorrectionIntent`. No thresholds inside;
  `has_feedback_content()` (any |feature delta| > 0.15 rad) is the
  level-invariant termination check.
- Verbalizers (`verbalizers.py`, `visual.py`) — one function per specificity
  level, all phrasing the same `CorrectionIntent` and returning an
  `Utterance(text, form)` or `None` exactly when `has_feedback_content` is
  false: `verbalize_vague` (fixed complaint), `verbalize_joint_resolved` (top
  two above-dead-band features through an 8-entry phrase table),
  `verbalize_everyday` (seeded categorical over arm/elbow/joint-resolved
  phrases, weight ∝ contrast magnitude × form prior), and `VisualVerbalizer`
  (VLM sees rendered trigger + oracle-window-end poses, responses disk-cached
  per episode/round). `VERBALIZERS` registry maps config names to callables.
  Intent deltas are nominal − oracle, so phrases point the opposite way.
- `choose_correction()` (`chooser.py`) — episode-loop replacement for
  `choose_cluster` (which stays for legacy call sites): candidates are every
  cluster mean × magnitude (default grid 0.5–1.5, scaled via
  `scale_trajectory` in the means' native axis-angle representation),
  lexicographic — pain filter (max violation ≤ 0.02 rad, the trigger
  threshold) then score `min_j(‖end − oracle[j]‖ + remaining_arc(j))` over
  `j ≥ min_join`; candidates and the oracle path are canonicalized to 7-DOF q
  (`arm_features.canonical_arm_q`) before scoring. All-painful → lowest mean
  violation with `no_acceptable_cluster=True`. Returns frozen `ChoiceResult`
  (label, magnitude, per-cluster acceptability + scores).
- Features come from the shared `arm_features.arm_feature_series()` implementation
  used by `GeneratedCostContext`, so hidden bounds, generated costs, corpus CSVs,
  and demo graphs are directly comparable. Demo Runner parses validated constant and
  pose-dependent grounded terms from `rationale.json` (falling back to the Stage 2
  artifact) and draws their exact threshold or piecewise interpolation boundary.
  It also records which named helpers each compiled generated cost reads: two-feature
  costs render as sampled penalty fields on phase plots, while costs reading exactly
  one named feature also render their sampled penalty region on that feature's time
  graph even without a declared bound.
- Personas (`personas.py`): `unrestricted` (default; no bounds),
  `adhesive_capsulitis`, `elbow_contracture`, `painful_arc` (elevation avoid band),
  `stroke_flexor_synergy` (active-effort coupled proxy),
  `out_of_synergy_reach_preference` (soft preference for more elbow extension as
  shoulder elevation increases),
  `triceps_long_head_contracture` (maximum elbow flexion falls with elevation),
  `biceps_long_head_contracture` (minimum elbow flexion rises with shoulder extension),
  `brachial_plexus_mechanosensitivity` (minimum elbow flexion rises with abduction),
  `cross_body_pain` (coupled bound:
  tolerable elevation = 2.2 + 4.5·abduction, i.e. the limit slides down the
  farther the upper arm adducts past the midline — blocks the direct route to
  across-body goals so the compliant correction is a visibly different path;
  smooth by design so escaping the painful region has a gradient, unlike a
  hard conditional zone).
  Registry: `PERSONAS` / `get_persona(name)`.

> **Canonical arm-feature layer (2026-07-20):**
> `planners/mpc/arm_features.py` is the single implementation for all five
> anatomical features. It accepts canonical q arrays ending in `(7,)` and
> decoded FK/cost-boundary arrays ending in `(3,3)`; boundary input is converted
> to q once before feature evaluation. Hand-authored MPC feature costs,
> `GeneratedCostContext`, simulated-user bounds, corpus CSV generation, demo
> trajectory graphs, and the sampled generated-cost penalty field all route
> through it. `shoulder_internal_external_rotation` is now the signed twist of
> the anatomical shoulder block `q[3:6]` about the T-pose upper-arm axis;
> clavicle `q[0:3]` does not contribute. The other four features, including
> plane-independent `shoulder_elevation`, remain available. Generated Python
> source still receives decoded `(3,3)` states for compatibility, but named
> feature helpers and all stored context/corpus trajectories are q-native.

> **Anatomical left-arm reparameterization (2026-07-14):**
> The three left-arm reconstruction sites now route the elbow/wrist slots through
> `kinematics.anatomical_elbow_wrist_slots` instead of per-slot shortest-arc
> `align_vectors`. The elbow slot carries the **recovered shoulder
> internal/external rotation** (twist about the upper arm onto the observed
> flexion-plane normal `u × f`); the wrist slot is a **pure forearm hinge** with
> pronation locked to the elbow-relative neutral, moving the reconstruction's
> singularity off the T-pose antiparallel (which real clips hit) and onto the
> unreachable 180° elbow flexion. Positions are preserved exactly; direction
> features (`elbow_flexion`, shoulder flexion/abduction/elevation) are unchanged.
> On the reference clip this dropped the displayed hand-frame drift from 136°→~43°
> (matching the true 41° forearm-direction change), wrist rate-limiter caps 6→0,
> and `left_wrist` `JointBoxLimit` violations 2→0. The recovered flexion-plane
> twist is stored in the anatomical shoulder block of q. Full details in audit §3a.

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

`MdmMotionGenerator` builds its inference args from the checkpoint's `args.json` overlaid with
`mdm/mdm_configs/mdm_config.yaml` (the YAML wins). The YAML pins `use_ema: false`: fine-tune
checkpoints save the training flag `use_ema: true`, but their EMA (`model_avg`, beta 0.9999)
weights stay ≈95% base model after 500 steps, so loading them silently reverts generation to
base-MDM behavior. Inference must always load the raw `model` weights.

The kimodo backend reuses `hml_smpl_conversion`'s SMPL-side helpers
(`smpl_body_pose_to_arm_aa/_collar_aa/_positions/_spine3_aa`) since kimodo's SMPL-X
`get_amass_parameters()` `pose_body (T,63)` maps directly to SMPL `body_pose (T,21,3)`.
For start poses, `KimodoMotionGenerator` converts SMPL `body_pose (21,3)` through the
same FK used by the visualizer and sends Kimodo visualizer-FK joint positions; the worker retargets them onto Kimodo's
skeleton and builds a frame-0 `FullBodyConstraintSet` from the resulting positions and
global rotations.

**Adding a new backend:** subclass `MotionGenerator`, implement the 5 abstract methods, and
register a builder in `MOTION_GENERATOR_BUILDERS`.
