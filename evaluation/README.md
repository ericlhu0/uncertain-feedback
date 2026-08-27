# Method-level evaluation

This directory evaluates the whole pipeline end to end — the experiments a
researcher runs to measure how well the method works, as opposed to anything
the method itself executes at runtime (that is `src/uncertain_feedback/
evaluation_mechanism/`, which the codex sandbox imports).

It follows the **benchmarks × approaches × metrics** structure of
[python-research-starter](https://github.com/tomsilver/python-research-starter),
built on the per-stage façades:

| Stage | Façade |
| --- | --- |
| text → candidate motions | `uncertain_feedback.motion_generators` |
| cluster + select | `uncertain_feedback.uncertainty` |
| correction → cost code | `uncertain_feedback.cost_generation` |
| score a generated cost | `uncertain_feedback.evaluation_mechanism` |
| execution | `uncertain_feedback.planners.mpc.rollout` |
| simulated care recipients | `uncertain_feedback.simulated_users` |

## Layout

- `benchmarks/` — `InteractionBenchmark`: personas × verbalizer abstraction
  levels × goal sequences, scored against the personas' hidden bounds.
- `approaches/` — an `Approach` is a named composition of three modules:
  a **grounder** (language → candidate motions), a **cost_gen** setting
  (language → persistent planner costs), and a **steering** method. Each
  axis is a package with one file per method (`approaches/grounders/`,
  `approaches/cost_gen/`, `approaches/steering/`) and a matching hydra config
  group, so any grid cell composes from the CLI: `approach=full
  approach/grounder=edit approach/cost_gen=immediate approach/steering=none`.
  - **Grounders** (`approaches/grounders/`): `MdmGrounder` (`grounder=mdm`,
    MDM sampling + clustering), `NominalGrounder` (`grounder=none`, the
    nominal plan is the only candidate — language is left to cost_gen),
    `LlmTrajectoryGrounder` (the pure-agent family, see below),
    `ParameterizedEditGrounder` (`grounder=edit`, single-joint axis-offset
    ramps), `BridgePotentialFieldGrounder` (`grounder=bridge`, BRIDGE-style
    attract/repel potential fields at body landmarks, Wang et al. HRI '26,
    oracle selection standing in for their LLM interpreter),
    `BridgeInterpreterGrounder` (`grounder=bridge_llm`, language-faithful: an
    LLM maps the utterance to landmark/polarity/strength; the user only tunes
    playback magnitude), and `KeypointGrounder` (`grounder=keypoint`, one 3D
    workspace keypoint for the elbow or wrist). Non-MDM grounders require
    `mpc_config=evaluation/conf/mpc_edit_baseline.yaml`.
  - **cost_gen** (`approaches/cost_gen/`): `NoCostGen` (`cost_gen=none`),
    `ImmediateCostGen` (`cost_gen=immediate`, stack every per-round cost), or
    `ConsolidateCostGen` (`cost_gen=consolidate`, one unified replacement
    cost via combination; formerly "lifelong"). Each takes `source`: `chosen`
    (anchor on the selected correction) or `nominal` (language-only prompt —
    required with `grounder=none`, which selects no correction;
    `cost_gen=language_only` is immediate with a nominal source).
  - **steering** (`approaches/steering/`): `NoSteering` (`steering=none`) or
    `ClassifierGuidanceSteering` (`steering=cg`); `cg` requires the mdm
    grounder and errors otherwise.
- **Pure-agent grounder** (`grounders/llm_trajectory.py`) — the same feedback
  grounded by an LLM with no motion prior.
  One interpretation call per round returns the 4 most likely *behaviourally
  distinct* readings of the utterance, given the full trajectory context in
  anatomical space (the nominal plan as a per-frame feature table, the current
  pose, arm/body landmark positions, and the goal); each reading becomes one
  candidate motion, and the simulated user selects among the 4 exactly as it
  selects among the system's 4 clusters.
  `LlmTrajectoryGrounder` writes the motion itself —
  `approach.grounder.output_space` is `positions` (elbow + wrist XYZ) or
  `anatomical` (the five joint features), and `approach.grounder.n_waypoints`
  is `0` (dense: every frame), `1` (`agent_waypoint`: the goal of the
  correction, after which the episode loop's continuation resumes toward the
  original goal), or more (`agent_sparse_waypoints`: linearly interpolated,
  not MPC-tracked). The LLM output is only *converted* — no displacement
  caps, no repair — so implausible motions show up honestly as candidates the
  chooser rejects. Every round's interpretations are saved to
  `interpretations_XX.json` in the episode dir. (The former `agent_cost`
  approach — each interpretation authored as a throwaway MPC cost and rolled
  out — was removed when cost generation became its own axis; `cost_only`
  covers language-to-cost grounding, and a multi-candidate variant would be a
  `CostGen.n_interpretations` knob.)
- `episode.py` — the interaction loop: oracle path → discomfort trigger →
  attribute → verbalize → ground → simulated-user choice → learn → continue,
  re-triggering until resolution or the round cap. Learned costs persist
  across a task's goal sequence.
- `metrics.py` / `structs.py` — per-round records and shared dataclasses.
- `run_single_experiment.py` — hydra entry point; writes `results.csv`
  (per-round) and `episodes.csv` (per-episode) plus per-round artifacts.
- `run_comparison.py` / `comparison.py` — run one scenario through several
  approaches with the utterance **given as text**, and write a side-by-side
  video plus a per-approach table.
- `analyze_results.py` — aggregates one or more runs into tables and plots.
- `conf/` — hydra configs. `approach/` holds one yaml per named composition
  (`full`, `no_steering`, `immediate_only`, `no_learning`,
  `language_only_learning`, `cost_only`, `edit_baseline`, `bridge_baseline`,
  `bridge_llm`, `llm_keypoint`, `agent_*`) built from the config groups
  `approach/grounder/`, `approach/cost_gen/`, and `approach/steering/`;
  `benchmark/` as before.
  Also `mpc_smoke.yaml` (a
  CPU-only planner config for harness smoke runs), `mpc_edit_baseline.yaml`
  (`mdm_llm.yaml` with the MDM pose file replaced by its decoded `arm:` angles;
  required by `approach=edit_baseline`, which skips the motion-generator load),
  and `mpc_demo_base1.yaml` (the demo runner's `base1` initial pose and
  `hits limit 1` goal, keeping the MDM `pose:` so the system and the baselines
  can share one rig).

The planning rig itself lives outside this package: `build_rig` /
`PlanningRig` in `uncertain_feedback.planners.rig` load the `MpcRunConfig` and
derive the planning context, skipping the MDM load for approaches that do not
need it.

**Comparing baselines against the system arms.** `build_rig` applies `arm:` over
the pose file's arm when the generator is loaded (the precedence
`planners/run.py` uses), so one config can drive both. The baselines still
default to skipping the motion-generator load, which leaves them on T-pose
torso geometry; pass `+approach.grounder.use_generator_rig=true` (accepted by
the keypoint and llm-trajectory grounders) to load it anyway so spine3/body
match the system exactly and the nominal rollout is byte-identical across
arms. Match the candidate counts too — `feedback.uq.n_clusters` for the system
against `approach.grounder.n_interpretations` for the baselines.

## Running

Smoke run (no GPU, no LLM — edit baseline, learning disabled):

```
uv run python evaluation/run_single_experiment.py approach=edit_baseline \
    approach/cost_gen=none benchmark=smoke mpc_config=evaluation/conf/mpc_smoke.yaml
```

Full system on one benchmark (needs MDM weights + GPU; cost generation needs
`OPENAI_API_KEY`, consolidation needs the `codex` CLI):

```
uv run python evaluation/run_single_experiment.py approach=full benchmark=personas_core
```

Pure-agent baselines (need `OPENAI_API_KEY`; no GPU, no MDM):

```
uv run python evaluation/run_single_experiment.py approach=agent_waypoint \
    benchmark=smoke mpc_config=evaluation/conf/mpc_edit_baseline.yaml
uv run python evaluation/run_single_experiment.py approach=agent_dense_positions \
    approach.grounder.output_space=anatomical benchmark=smoke \
    mpc_config=evaluation/conf/mpc_edit_baseline.yaml
```

Compare grounding methods on one scenario with your own sentence — writes
`comparison.mp4` (nominal + one panel per approach, shared axes) and
`comparison.csv`:

```
uv run python evaluation/run_comparison.py \
    --feedback "keep my arm closer to my body and keep my arm lower" \
    --persona triceps_long_head_contracture --goal 0.25 0.3 0.18 \
    --mpc-config evaluation/conf/mpc_demo_base1.yaml --out outputs/comparison
```

`--approaches` picks the arms (any YAML name in `conf/approach/`; the default
set is the four llm-trajectory arms plus `cost_only` and `full`), `--learning`
(`none`/`immediate`/`consolidate`) forces the cost-gen mode on every arm so
they differ only in grounding, and `--no-video` skips rendering.
The `scripted` verbalizer replays the sentence every round; the persona still
scores candidates against its own hidden intent, so only the words the
grounding methods read are yours.

Sweeps use hydra multirun:

```
uv run python evaluation/run_single_experiment.py -m seed=0,1,2 \
    approach=full,no_steering,immediate_only benchmark=abstraction_sweep
uv run python evaluation/analyze_results.py multirun/ --out evaluation_analysis/
```

## Simulated-user chooser

The simulated user selects among candidate corrections per
`simulated_user.chooser` in the planner config (CLI override: `sim_chooser=`):
`intent_aligned` (default — the comfortable candidate whose motion best aligns
with the private `CorrectionIntent`; per-round `correction_alignment` /
`best_alignment` land in `results.csv`), `progress` (legacy oracle-path
progress; superhuman path knowledge, kept for ablation), and `random`
(uniform over comfortable candidates). A candidate is acceptable when its
mean playback violation stays under the trigger threshold. Results produced
before 2026-08-06 used the `progress` chooser with a max-violation
acceptability bar. `approach=language_only_learning` is the companion
ablation: costs are authored from the utterance + nominal plan only
(`cost_gen.source: nominal`), so the selected motion never reaches the cost
generator.

## Paper experiments → configs

| Claim (paper §IV) | Command sketch |
| --- | --- |
| A: high-level feedback → low-level preferences | `-m seed=0,1,2 approach=immediate_only,no_learning benchmark=abstraction_sweep` |
| B: remember + combine across interactions | `-m seed=0,1,2 approach=full,immediate_only benchmark=lifelong mpc_config=src/uncertain_feedback/planners/mpc/configs/mdm_llm_transfer.yaml` |
| C: grounding improves with feedback | `-m seed=0,1,2 approach=full,no_steering benchmark=lifelong ...` (compare per-event curves in `analyze_results.py`) |
| Richness vs. predefined edits | `-m seed=0,1,2 approach=full benchmark=abstraction_sweep`, plus `-m seed=0,1,2 approach=bridge_llm,llm_keypoint benchmark=abstraction_sweep mpc_config=evaluation/conf/mpc_edit_baseline.yaml` (add `edit_baseline,bridge_baseline` for the weak-reference and oracle-upper-bound arms) |
| Motion prior vs. a pure agent | `-m seed=0,1,2 approach=agent_waypoint,agent_sparse_waypoints,agent_dense_positions,agent_dense_anatomical benchmark=abstraction_sweep mpc_config=evaluation/conf/mpc_edit_baseline.yaml` (candidate-set-matched against `approach=full`: 4 candidates, same chooser) |

Per-round metrics in `results.csv` cover candidate coverage
(`any_acceptable`, `n_acceptable`), candidate hidden-cost (`candidate_hidden_*`,
the Exp-C curve), learning outcomes (`cost_accepted`, `unified_installed`),
and continuation violations/resolution; `episodes.csv` has episode-level
resolution, reach, and executed-trajectory violations.

## Known seams

- **Steering source**: `ClassifierGuidanceSteering` steers diffusion from the
  persona's hidden bounds (`build_steering_spec`), the only steering-cost
  source wired today — i.e. a known-preference upper bound. Steering from the
  *learned* cost would be a new `Steering` subclass in `approaches/steering/`
  once that wiring exists. The approach's steering module always sets the
  mode in evaluation; the planner yaml's `feedback.uq.steering.mode` only
  matters for the demo path, though its mechanism knobs (`guidance_weight`,
  `guide_from`, …) still apply.
- **Consolidation** (`cost_gen: consolidate`) shells out to the `codex`
  CLI (see `llm_cost.codex_cmd` in the planner config).
- **Pure-agent baselines get no feasibility help by design** ("no tooling"):
  written trajectories are converted, never repaired, and infeasible geometry is
  only re-imposed by FK bone lengths. Their quality also hinges on the
  interpretation call producing genuinely distinct hypotheses — four paraphrases
  collapse best-of-4 to best-of-1, which the saved `interpretations_XX.json`
  makes checkable.
- **Speed-shaped feedback is inexpressible** for these baselines and for the
  current system alike (the pipeline is timing-blind); the benchmarks'
  feedback is position/path-shaped.
- The human study (paper §IV-D) is out of scope here; this harness covers the
  simulation experiments.
