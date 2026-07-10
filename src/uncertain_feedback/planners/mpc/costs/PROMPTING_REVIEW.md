# LLM / Agent Cost-Generation Prompting — Full Review

_A complete map of every prompt, contract, image, and summary fed to the cost
generator, plus observed tensions. Covers `costs/prompts/**`, the three backends
(`llm_costs.py`, `turns_costs.py`, `agent_costs.py`), the shared plumbing in
`cost_generator.py` / `generated.py` / `cost_feedback.py`, and the rendering in
`utils/plot.py`._

---

## 1. What this subsystem does

When the user corrects a robot arm motion in natural language, MDM turns that into a
plausible arm trajectory. Optionally, an LLM (or coding agent) then writes a **Python
cost function** that is compiled and injected into the MPC optimizer so the
goal-seeking controller reproduces the *preference* behind the correction — not the
exact path.

The cost function the LLM must produce has a fixed signature:

```python
def cost(q_trajs, context, params):
    # q_trajs: (n_rollouts, horizon+1, 3, 3) axis-angle for
    #          [left_shoulder, left_elbow, left_wrist]
    # returns: finite np.ndarray of shape (n_rollouts,)
```

Everything below exists to steer the LLM toward writing a *good* such function:
minimal, joint-space, preference-shaped (not path-copying), and goal-safe.

---

## 2. How a prompt is assembled

Source: `prompts/__init__.py`.

Every backend uses the same staged prompting strategy:

```
interpret.txt + instruction + contrast images + compact summaries
    -> plain-language preference JSON
ground.txt + interpretation + full numeric summaries
    -> numeric feature/bound specification JSON
author.txt + specification + runtime_api.txt + output_contract.txt
    -> final cost JSON
```

Assembly details:

- `build_interpret_prompt`, `build_ground_prompt`, `build_author_prompt`, and
  `build_refine_prompt` load `prompts/stages/*.txt`.
- `compact_summaries()` strips raw joint-angle arrays from stage one so interpretation
  does not do stage two's numeric work.
- Substitution is via `str.replace` (not `str.format`), so literal braces in the
  guideline prose are left untouched. Only `{instruction}`, `{summaries}`, and the
  image placeholders are replaced.
- `turns` uses `build_refine_prompt`: it fixes the interpretation once, then lets the
  conversation revise both the numeric grounding and the code against rollout feedback.
- `agent` uses `build_staged_task_body`: the same stage heads are inlined into `TASK.md`
  so codex follows the same decomposition.

### Image placeholders

`IMAGE_PLACEHOLDERS` (in `prompts/__init__.py`) maps a placeholder token to a
self-describing sentence that replaces it *only when that image was actually
rendered*. If the image is missing, the placeholder is deleted (no dangling text) and
nothing is attached. Images attach in the order their placeholders appear in the head.

| Placeholder | Replacement guidance (abbreviated) | Rendered when |
|---|---|---|
| `{current_cluster_traj_img}` | "ONLY the chosen path's full arm motion — read the chosen posture clearly." | always |
| `{other_clusters_traj_img}` | "chosen arm alongside the OTHER candidate arms (grey) — see which dimension separates the chosen path." | >1 cluster |
| `{reference_traj_img}` | "chosen arm alongside the ORIGINAL-GOAL reference arm (green, dashed wrist path) + gold star — where your cost must NOT block the goal." | reference traj exists |

Note the guidance text lives **in the placeholder map**, not the templates, precisely
so a placeholder can be dropped whole when its image is unavailable.

---

## 3. Stage heads (`prompts/stages/*.txt`)

There are four stage heads: `interpret.txt`, `ground.txt`, `author.txt`, and
`refine.txt`. `interpret.txt` requests **all three** contrast images (`current` +
`others` + `reference`) when available and is written for the **UQ multi-candidate,
iterative** case.

Persona: *the caregiver of a person with limited arm mobility.* The robot was already
driving the arm to a goal when the user interrupted with one spoken instruction; the
robot produced several candidate motions; the user picked **one**. The head's job is to
put into words what made the chosen motion right — what the other candidates got wrong —
and express only that as a joint-space preference that adjusts **how** the arm moves
without preventing it from still reaching the original goal. `ground.txt` turns that
preference into feature/bound numbers; `author.txt` implements those numbers without
changing them.

Core instructions in the head:

- **Resolve the distinguishing dimension to a supported joint feature.** Height/reach
  language must be attributed to elbow flexion, shoulder flexion/extension, shoulder
  abduction/adduction, or shoulder internal/external rotation; there is no separate
  hand-height feature in the authoring API.
- **Infer the preferred *set*, not the demonstrated point.** A feature value on the
  chosen trajectory is one example inside a broader comfortable set — convert to a band /
  one-sided bound / tolerance, never an exact target.
- **Say one plain caregiver sentence first**, then translate only that into the fewest
  cost terms — usually one destination term, plus at most one posture term. Params ≤ 3.
- **Capture what distinguishes the CHOSEN path.** The instruction is identical across
  candidates, so it can't be the distinguisher — *where the arm ends up and how it's
  held* is. A cost that would be unchanged had the user picked a different candidate is
  wrong. Explicitly bans the "wrist stays above its start height" floor (every candidate
  rises from the same pose, so it encodes nothing).
- **Prefer named joint-angle helpers** for the differentiator ("most 'where the hand
  goes' differences are downstream of a joint being held differently — check joint
  angles first").
- **Do NOT block the original goal** (dedicated section — this is why the reference image
  is shown): the cost is *added* to the goal-reaching objective, so it must be near-zero
  all along the reference path *and* at its final joint posture, and the preferred set
  must be broad enough to contain both the chosen endpoint and the goal. No walls,
  pinpoint targets, or competing attractors; keep magnitude modest.
- **Human-explainable test.** A term is valid only if you could explain it to the person
  in everyday words; "minimizes squared deviation / Gaussian centered at / weighted
  blend" is a rejection signal.
- **No path-copying** (no Gaussian time-peaks, arc/sweep matching, mid-trajectory shape)
  and **no smoothness / velocity / acceleration terms.**
- Ships **How-to-think / Contrast / Examples** blocks with good vs. reject phrasings,
  and describes the two per-iteration feedback images (see §6b) for the refining
  backends.

---

## 4. Shared runtime API contract (`runtime_api.txt`)

Appended verbatim to every prompt. Tells the model what it may call inside `cost`:

- **Signature & shapes:** `q_trajs` is `(n_rollouts, horizon+1, 3, 3)` for
  `[left_shoulder, left_elbow, left_wrist]` axis-angle.
- **Preferred joint-feature helpers** (return radians, accept any array ending in
  `(3,3)`):
  - `context.elbow_flexion_angles(q_trajs)`
  - `context.shoulder_flexion_extension_angles(q_trajs)` — signed depth (z) component
  - `context.shoulder_abduction_adduction_angles(q_trajs)` — signed lateral (x),
    positive = away from torso (left arm)
  - `context.shoulder_internal_external_rotation_angles(q_trajs)` — signed twist about
    T-pose upper-arm axis
  - These are documented as **preference-friendly component-angle approximations in the
    spine3 frame**, explicitly "not a clinical joint angle decomposition."
- **Raw joint indexing** `q_trajs[:, :, j, :]` (0=shoulder, 1=elbow, 2=wrist) allowed
  for advanced cases, with a warning not to pick one arbitrary rotvec component and
  call it "flexion." Wrist terms only when a wrist-rotation preference is clearly
  implied (the arm model has no hand joint beyond the wrist).
- **Cartesian/FK helpers are for UNDERSTANDING only:** `fk_rollouts`, `mdm_positions`,
  `current_positions`, `joint_index`, `mdm_traj`, `recent_q/recent_positions`. Positions
  are `(x, y, z)` with **y = up** (a repeated warning: don't use z for height). NumPy
  indexing only — dict-style `mdm_positions['left_wrist']` raises `TypeError`.
- Stats like max wrist height should be **read from the Summaries** and hardcoded, not
  recomputed.
- `np` available; no imports.
- Return finite `(n_rollouts,)`.

**Key line:** _"The returned cost itself must stay joint-space … do not use these
position outputs to compute it."_

---

## 5. Shared output contract (`output_contract.txt`)

Appended verbatim to every prompt. The "Hard requirements":

1. **Joint-space ONLY.** Build the cost exclusively from joint angles (named helpers,
   plus raw axis-angle joints only for clearly-implied wrist rotation). All terms in
   radians.
2. **No Cartesian/world positions in the cost.** Do not call `fk_rollouts`/`fk_batch`,
   do not read any `*_positions`, do not use `joint_index`. Positions & summaries are
   for understanding only — translate "reach up / lower / across body" into joint-angle
   features, **not** wrist/elbow x/y/z heights.
3. Return **only JSON** with keys: `description, code, params, explanation,
   recipient_explanation`.
4. Prefer costs over **future timesteps** `q_trajs[:, 1:]`, not just the initial state.
5. **No smoothness / velocity / acceleration.**
6. **Avoid exact pose matching** — prefer one-sided bounds / tolerance bands; exact
   target only when clearly called for.
7. **Fewest terms; params ≤ 3.**
8. Field semantics:
   - `description` — the caregiver preference in plain human language (not code
     structure).
   - `explanation` — 3–6 sentences for developers: what preference & why, which terms &
     what each penalizes, which numbers came from where in the trajectory data.
   - `recipient_explanation` — 1–3 plain sentences said directly to the care recipient,
     no jargon / "cost" / "penalty" / measurements.

---

## 6. The images (what the model actually sees)

Two families of images, all rendered by `ArmVisualizer` in `utils/plot.py`. All use
three orthographic views side-by-side, a translucent static reference body, the
**orange** current pose (shared start), and a **gold star** goal.

### 6a. Prompt-grounding overlays — `render_cluster_contrast_overlay`

Produced by `render_prompt_images(...)` (in `generated.py`), wired in `run.py`. Up to
three separately-readable PNGs, each keyed to a placeholder (§2):

- **`current.png`** (`current_cluster_traj_img`): chosen cluster only — **blue-gradient
  full arm** (up to 12 sampled frames, light→dark = early→late) + steelblue wrist path.
  Start/end markers intentionally omitted.
- **`others.png`** (`other_clusters_traj_img`): chosen (blue) + every **other candidate
  cluster** as a grey-gradient full arm + darkgrey wrist path + `x` end marker. Only
  rendered when >1 cluster.
- **`reference.png`** (`reference_traj_img`): chosen (blue) + **original-goal reference**
  as green-gradient full arm + green **dashed** wrist path. Only when a reference
  trajectory exists.

Each image computes its **own** equal-square axis limits from only the layers it draws,
so it's individually legible (the chosen arm may sit at a slightly different scale
across the three images). The chosen cluster defaults to `context.mdm_traj` when there
are no clusters.

### 6b. Iteration feedback images (turns / agent backends only)

Rendered per candidate by `evaluate_and_render(...)`:

- **`comparison.png`** — `render_cost_feedback_overlay`: the motion the candidate cost
  *actually produced* (**red** gradient full arm + firebrick wrist path + `x` endpoint)
  overlaid on the **target corrected path** (**green**). The rollout is resampled to the
  target length so the two arms are frame-comparable.
- **`angles.png`** — `render_joint_angle_comparison`: a 2×2 grid, one subplot per
  anatomical feature (elbow flexion; shoulder flexion/extension; abduction/adduction;
  internal/external rotation), plotting **green target vs red rollout** angle-over-frame.
  This is the "read the curve *shapes*, not just endpoints" signal the templates
  reference.

The "target corrected path" (green) is `context.full_correction_traj` when available —
the **whole** intended motion (pre-correction history → the correction → continuation to
goal) — falling back to `context.mdm_traj`.

---

## 7. The Summaries JSON (`build_motion_summaries`)

Attached as text under `Summaries:` in every prompt. All positions are
**spine3-relative** `(x, y, z)`, y = up. Structure:

- **`current`** — single-frame: raw joint angles, per-joint position, shoulder/elbow
  axis-angle value+norm, and `joint_features` (the four named angles at this frame).
- **`mdm_traj`** — the correction: `joint_angles` array stats (shape/min/max/mean/
  start/end), per-joint `positions` series stats (x/y/z min/median/max/start/end +
  delta), shoulder/elbow start/end values+norms, and `joint_features` series stats
  (min/median/max/start/end per feature).
- **`recent`** — same shape as `mdm_traj` over the recent executed window
  (`preference_window`, default 50); `{}` if empty.
- **`reference`** _(when a reference exists)_ — same shape, the original-goal rollout
  (path + endpoint joint posture the cost must not block).
- **`cartesian_goal`** _(when a Cartesian goal exists)_ — the spine3-relative wrist
  target as a 3-vector.

The joint-feature stats are the numeric anchors the templates tell the model to convert
into bands/bounds (e.g. "the elbow-flexion bound of 1.1 rad came from the median
elbow_flexion of the MDM range").

---

## 8. The three backends

All constructed identically via `create_cost_generator(cfg, ...)`; only the factory
branches on `cfg.backend`. Shared base `CostGenerator` supplies stage helpers, LLM
construction, JSON parse/compile/smoke-test, artifact saving, and installation.

- **`llm`** (`LlmCostGenerator`, default): single-pass staged flow. Runs
  interpret → ground → author once, parses/compiles only author output, and saves
  `interpret_*`, `ground_*`, `author_*`, `stage_log.md`, `cost.py`, and `params.json`.
- **`turns`** (`TurnsCostGenerator`, `max_turns` default 6): a real stateful
  conversation. It interprets once, then iterates grounding+authoring. Each turn:
  parse → roll out → score (FK L2, lower=better) → render `comparison.png` +
  `angles.png` → feed the score, both images, and a JSON joint-feature comparison back
  as the next user message. Keeps the best-scoring cost; stops early after
  `_NO_IMPROVE_PATIENCE = 2` non-improving turns. `stage_log.md` records the fixed
  interpretation and every refine-turn prompt snapshot/response.
- **`agent`** (`AgentCostGenerator`, `codex_cmd`): writes the same prompt into `TASK.md`
  as an inlined staged task and delegates to the external `codex` CLI, which authors
  `response.json` itself. It must also write `stage_log.md` with Stage 1, Stage 2, and
  Stage 3 responses; that file is appended to `codex.log`. When an `EvalState` is
  available it also drops `state.pkl` + points codex at
  `experiments/render_cost_comparison.py` so codex can roll out, render
  `comparison.png`/`angles.png`, and iterate — logging each visual comparison in a
  required `ITERATION_LOG.md`.

Model / config knobs (`LlmCostConfig`, `config.py`): `enabled` (default false),
`model` (`OPENAI_MODEL` env → `gpt-5.6-luna`, reasoning effort `xhigh`), `strict`, `artifact_dir`, `use_images`
(default true), `prompt` (default `"1"`), `backend` (default `"llm"`), `max_turns` (6),
`codex_cmd`. System prompt + `temperature=0.2`, `max_tokens=16000` set in
`_make_llm_model`.

---

## 9. Observations & tensions (review findings)

These are inconsistencies worth deciding on — not necessarily bugs, but places where
the prompt stack currently pulls in two directions.

1. **The joint-space-only mandate is structurally under-expressive for the task's
   central preference — height — so the template conflicts with it.** This is the
   headline finding, and it survives the consolidation to a single template.

   `output_contract.txt` is an *absolute* hard requirement: **joint-space only, no
   Cartesian / position terms in the cost**, with no carve-out. But there is **no
   joint-angle feature for wrist height (y)**: the four helpers are elbow flexion,
   shoulder flexion/extension (z / depth), abduction/adduction (x / lateral), and
   internal/external rotation (twist) — none is height. `runtime_api.txt` itself tells
   the model to read height from a **position**: `mdm_positions[:, 4, 1]`. So "raise my
   arm higher / lower it / reach further" — precisely the corrections this system exists
   to serve — **cannot be written cleanly in the four helpers**; they inherently need a
   Cartesian read the contract bans.

   The remaining template (`1`) still points both ways: its recommended **primary term is
   a destination region** ("a comfort band/ball around where the hand arrives — a height
   band, a distance-from-torso range") and it explicitly *permits* Cartesian ("reach for
   a Cartesian position region or height band when the differentiator can't be captured
   by a joint angle") — both of which the shared contract forbids outright. It does now
   lead with "prefer joint-angle helpers first," which narrows but does not remove the
   conflict.

   **Recommendation — this is a design decision to make, not a doc fix:** either (a)
   relax the hard requirement to allow bounded height / position-region terms (matching
   what the template and the height-less helper set actually require), or (b) accept
   that height corrections get lossily re-encoded into shoulder flexion/abduction angles,
   and rewrite the template's residual region/height-band language accordingly. Leaving
   the contract and template as-is asks the model to satisfy two mutually exclusive
   instructions. (Quick proof: try writing "keep the wrist above height X" using only the
   four helpers with no position read — you can't.)

2. **"Target" means three different things in the feedback loop.** For a candidate:
   - the **scalar score** (`_score_rollout`) compares the rollout to `context.mdm_traj`
     (the MDM correction segment only);
   - the **overlay image** green arm and the **joint-comparison JSON**
     (`build_rollout_joint_comparison`) use `context.full_correction_traj` (history +
     correction + goal continuation).
   So the number the model is told to minimize and the picture it's told to match are
   measured against **different reference trajectories**. Usually close, but they can
   diverge (the score ignores the goal-continuation tail the image shows). Consider
   scoring against `full_correction_traj` too, for consistency.

3. **Feedback-loop language leaks into single-shot use.** The template says "each attempt
   returns two feedback images / as you refine…". With the **default backend `llm`**
   (single shot) there is no refinement loop, so that paragraph is misleading. Backend
   and prompt are independent config knobs, so `backend=llm` with the iterative-framed
   template is the default pairing.

4. **Joint-feature helpers are approximations, surfaced as if anatomical.** The helpers
   are explicitly "coarse SMPL-space proxies, not a clinical decomposition"
   (`generated.py` docstrings), yet the template presents them as clean anatomical DOFs
   (elbow flexion, abduction/adduction, etc.). For preference-shaping this is probably
   fine, but any numeric bound the model derives ("1.1 rad") inherits that
   approximation — worth keeping in mind when reading generated `explanation` fields.

5. **Two orthogonal "distinguisher" instructions could conflict.** The template tells the
   model both (a) "capture the destination or your cost collapses across candidates" and
   (b) "this rule OVERRIDES the other paths: keep your cost near zero at THIS endpoint,
   never correct toward the majority." These are reconcilable (center a broad region on
   the chosen endpoint), but the model is asked to be simultaneously *distinguishing* and
   *endpoint-faithful*, and both framings get emphatic "OVERRIDES/wrong" language. Watch
   generated costs for over-tight endpoint balls that trade goal-safety for
   distinctiveness.

_Resolved by the consolidation to a single template:_ the earlier findings that template
`2` (the old default) was the most flagrant Cartesian offender, and that goal-safety
guidance / the reference image were unaligned across templates `2`–`5`, no longer apply —
`1` is the former `5`, which carries the full goal-safety section and all three images.
The stale `defaults to "default"` docstring comment is also now fixed (reads `"1"`).

---

## 10. File index

| File | Role |
|---|---|
| `prompts/__init__.py` | staged prompt assembly, placeholder→image mapping |
| `prompts/runtime_api.txt` | shared API contract (§4) |
| `prompts/output_contract.txt` | shared hard requirements + JSON schema (§5) |
| `prompts/stages/*.txt` | interpret / ground / author / refine heads (§3) |
| `cost_generator.py` | base class, factory, scoring, `evaluate_and_render` |
| `llm_costs.py` | single-pass staged backend |
| `turns_costs.py` | multi-turn conversational backend (fixed interpret, ground+author refine) |
| `agent_costs.py` | codex-CLI agent backend, `TASK.md` authoring |
| `generated.py` | runtime context, cost compile/exec, summaries, `render_prompt_images` |
| `cost_feedback.py` | `EvalState` — picklable rollout state for the off-process agent |
| `utils/plot.py` | `render_cluster_contrast_overlay`, `render_cost_feedback_overlay`, `render_joint_angle_comparison` |
| `planners/run.py` (~763–839) | wiring: builds context/summaries/images, selects backend |
| `planners/mpc/config.py` (`LlmCostConfig`) | config knobs & defaults |
