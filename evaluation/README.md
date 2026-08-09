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
- `approaches/` — the system variants: `SystemApproach` (MDM grounding with
  `learning`/`steering_mode` knobs) and two predefined-edit baselines without a
  text-to-motion model: `ParameterizedEditApproach` (single-joint axis-offset
  ramps) and `BridgePotentialFieldApproach` (BRIDGE-style attract/repel
  potential fields at body landmarks, Wang et al. HRI '26, with oracle
  selection standing in for their LLM interpreter). `BridgeInterpreterApproach`
  (`approach=bridge_llm`) is the language-faithful variant: an LLM maps the
  utterance to (landmark, polarity, strength) with trajectory + conversation
  context; the simulated user only tunes playback magnitude.
  `KeypointApproach` (`approach=llm_keypoint`) is the strongest
  language-dependent baseline: an LLM emits one 3D workspace keypoint for the
  elbow or wrist to follow, given numeric scene context; again the user only
  tunes magnitude of the single proposed motion. The serious richness
  comparisons are `bridge_llm` and `llm_keypoint`; `edit_baseline` remains the
  weak reference point. All baselines require
  `mpc_config=evaluation/conf/mpc_edit_baseline.yaml`.
- `episode.py` — the interaction loop: oracle path → discomfort trigger →
  attribute → verbalize → ground → simulated-user choice → learn → continue,
  re-triggering until resolution or the round cap. Learned costs persist
  across a task's goal sequence.
- `rig.py` — loads the `MpcRunConfig` and derives the planning context;
  skips the MDM load for approaches that do not need it.
- `metrics.py` / `structs.py` — per-round records and shared dataclasses.
- `run_single_experiment.py` — hydra entry point; writes `results.csv`
  (per-round) and `episodes.csv` (per-episode) plus per-round artifacts.
- `analyze_results.py` — aggregates one or more runs into tables and plots.
- `conf/` — hydra configs (`approach/`, `benchmark/`), `mpc_smoke.yaml` (a
  CPU-only planner config for harness smoke runs), and `mpc_edit_baseline.yaml`
  (`mdm_llm.yaml` with the MDM pose file replaced by its decoded `arm:` angles;
  required by `approach=edit_baseline`, which skips the motion-generator load).

## Running

Smoke run (no GPU, no LLM — edit baseline, learning disabled):

```
uv run python evaluation/run_single_experiment.py approach=edit_baseline \
    approach.learning=none benchmark=smoke mpc_config=evaluation/conf/mpc_smoke.yaml
```

Full system on one benchmark (needs MDM weights + GPU; learning needs
`OPENAI_API_KEY`, lifelong combination needs the `codex` CLI):

```
uv run python evaluation/run_single_experiment.py approach=full benchmark=personas_core
```

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
(`learn_from: nominal`), so the selected motion never reaches the cost
generator.

## Paper experiments → configs

| Claim (paper §IV) | Command sketch |
| --- | --- |
| A: high-level feedback → low-level preferences | `-m seed=0,1,2 approach=immediate_only,no_learning benchmark=abstraction_sweep` |
| B: remember + combine across interactions | `-m seed=0,1,2 approach=full,immediate_only benchmark=lifelong mpc_config=src/uncertain_feedback/planners/mpc/configs/mdm_llm_transfer.yaml` |
| C: grounding improves with feedback | `-m seed=0,1,2 approach=full,no_steering benchmark=lifelong ...` (compare per-event curves in `analyze_results.py`) |
| Richness vs. predefined edits | `-m seed=0,1,2 approach=full benchmark=abstraction_sweep`, plus `-m seed=0,1,2 approach=bridge_llm,llm_keypoint benchmark=abstraction_sweep mpc_config=evaluation/conf/mpc_edit_baseline.yaml` (add `edit_baseline,bridge_baseline` for the weak-reference and oracle-upper-bound arms) |

Per-round metrics in `results.csv` cover candidate coverage
(`any_acceptable`, `n_acceptable`), candidate hidden-cost (`candidate_hidden_*`,
the Exp-C curve), learning outcomes (`cost_accepted`, `unified_installed`),
and continuation violations/resolution; `episodes.csv` has episode-level
resolution, reach, and executed-trajectory violations.

## Known seams

- **Steering source**: `SystemApproach` steers diffusion from the persona's
  hidden bounds (`build_steering_spec`), the only steering-cost source wired
  today — i.e. a known-preference upper bound. Steering from the *learned*
  cost plugs into `SystemApproach._reset_grounding` once that wiring exists.
- **Lifelong combination** (`learning: lifelong`) shells out to the `codex`
  CLI (see `llm_cost.codex_cmd` in the planner config).
- The human study (paper §IV-D) is out of scope here; this harness covers the
  simulation experiments.
