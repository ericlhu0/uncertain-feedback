"""Prompt templates for the LLM trajectory-cost generator.

The guideline body lives here as named templates so prompts can be swapped without
touching the generation code. Register a new variant by adding an entry to
``PROMPTS``; select one via ``llm_cost.prompt`` in the MPC config (defaults to
``"default"``).

A template is a ``str`` containing the literal placeholders ``{instruction}``,
``{image_section}``, and ``{summaries}``, which :func:`build_llm_cost_prompt`
substitutes (via ``str.replace``, so other braces in the guideline text are left
untouched).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_DEFAULT_PROMPT = """
You are a robot controller assisting a person with mobility limitations with arm movements. Your task is to generate a compact Python cost function that encodes the smallest set of user or caregiver preferences that realistically explains this correction. Keep in mind that the language instruction can be ambiguous and omit certain preferences that are implied by the motion. Think in terms of what the person likely cared about: what region the arm should reach, how high or far it should move, what joint-angle range should be allowed, or whether one clearly uncomfortable/support-related posture should be avoided. Do not encode every visible difference between the original and corrected trajectory. A feature value at the corrected trajectory is not the only preferable feature value; it is just one example inside a broader preferred set. Your goal is to infer that preferred set and add only the minimum cost terms needed to express it. Also keep in mind that joint limits are not independent from each other: for example, a user can be comfortable with extending their elbow in front of them but not to the side.

Instruction:
{instruction}

{image_section}

Guidelines:
- DO: first infer the most realistic human preference behind the correction in plain language. Then implement the fewest cost terms that capture that preference.
- DO: default to one cost term. Add a second or third term only when it captures a genuinely separate preference that is clearly implied by the instruction and trajectory.
- DO: use the MDM trajectory summaries to extract the *intent* behind the motion — e.g. how high the person wants to reach (wrist z range or lower bound), roughly where they want the arm to end (a preferred region, usually not an exact point), how much lateral reach is involved, joint angle limits, etc. Use these numbers to define preference regions, ranges, thresholds, or margins.
- DO: ground each preference term numerically in the generated trajectory data. The MDM trajectory tells you what this person's version of the instruction means quantitatively — treat those numbers as examples inside a preferred set, not as exact targets to imitate. You may draw targets from any part of the trajectory — start, end, or mid-trajectory — if that best represents the preference, but convert point values into a range, one-sided bound, or tolerance region whenever possible (e.g. "elbow stay above height X mid-motion").
- DO: include multiple cost terms only when the instruction and trajectory imply multiple distinct preferences, and keep the set small and natural. A typical cost should have one preference term; two or three should be uncommon.
- DO: keep params minimal and interpretable. Prefer trajectory-derived bounds, tolerances, region centers with radii, or thresholds. Avoid inventing separate tuning-weight params for each term unless there is a clear reason.
- DO: write costs grounded in what a caregiver would consider: reaching a goal, keeping the arm supported, or avoiding uncomfortable positions.
- DO: consider relative preferences only when implied by the instruction or trajectory, such as keeping the wrist/elbow near the torso (spine3) or near another arm joint. Express these as broad distance limits or preferred distance ranges, not exact pairwise distances, unless exact contact is clearly intended.
- DO: use named joint-angle features when they are the natural way to express the preference, such as elbow bend/flexion, shoulder flexion/extension, shoulder abduction/adduction, shoulder internal/external rotation, or avoiding an uncomfortable joint configuration. Prefer the helper methods in the Runtime API and broad ranges or one-sided bounds over exact full-arm pose matching.
- DO NOT: imitate the trajectory's shape or timing — no Gaussian time peaks tied to specific arc positions, no sweep patterns that encode the MDM trajectory geometry, no costs that enforce a specific mid-trajectory path.
- DO NOT: write costs that only make sense as "follow this exact path." A caregiver cares about whether the arm reaches a useful region and whether it's comfortable throughout, not about replicating the specific arc the MDM generated.
- DO: prefer penalties for being outside a box/ball/height band, below a minimum, above a maximum, outside a directional/lateral region, or outside a relative-distance range. Exact Cartesian point penalties are acceptable when the instruction clearly specifies a precise target, contact point, or placement.
- DO NOT: add generic smoothness, velocity, or acceleration penalties.
- DO NOT: add endpoint, wrist height, elbow support, progress, and posture terms all at once. Choose the smallest subset that explains the correction.

Framing examples:
- Caregiver-style (good): "Help the arm reach a comfortable high region and keep the elbow from dropping below the shoulder."
- Region-style (good): "Penalize the wrist only when it is below the preferred height band or outside the broad forward-reaching region."
- Relative-style (good when implied): "Keep the wrist from drifting too far away from the torso while reaching."
- Exact-point style (only when precise target is explicit): "Place the wrist exactly at a specified button or contact point."
- Exact-point style (avoid by default): "Force the wrist to end exactly at the MDM final wrist xyz coordinate."
- Trajectory-imitation (avoid): "Apply a Gaussian reward peaking at mid-horizon to encourage the upward arc, then enforce a leftward descending sweep to match the MDM path."

Runtime API:
- Define exactly: def cost(q_trajs, context, params):
- q_trajs has shape (n_rollouts, horizon + 1, 3, 3) for left_shoulder,
  left_elbow, left_wrist axis-angle states.
- Prefer these named joint-feature helpers when writing joint-space costs:
  context.elbow_flexion_angles(q_trajs)
  context.shoulder_flexion_extension_angles(q_trajs)
  context.shoulder_abduction_adduction_angles(q_trajs)
  context.shoulder_internal_external_rotation_angles(q_trajs)
  Each accepts any array ending in (3, 3), including q_trajs[:, 1:] or
  context.mdm_traj, and returns the matching leading shape. Values are radians.
  The shoulder helpers are preference-friendly component-angle approximations
  in the spine3 frame: flexion/extension is signed upper-arm movement along
  the z axis, abduction/adduction is signed lateral upper-arm movement along
  the x axis (positive is away from the torso for the left arm), and
  internal/external rotation is signed twist around the T-pose upper-arm axis.
- Raw joint indexing is also allowed for advanced cases: q_trajs[:, :, 0, :]
  is left_shoulder, q_trajs[:, :, 1, :] is left_elbow, and q_trajs[:, :, 2, :]
  is left_wrist. The final dimension is a 3D local axis-angle rotation vector,
  not separate named anatomical DOFs. Avoid selecting one arbitrary rotvec
  component and calling it flexion/abduction/rotation. The left_wrist row exists
  in the SMPL/HML state, but this simplified arm model has no hand joint beyond
  the wrist, so use wrist joint-angle terms only when a wrist-rotation
  preference is clearly implied.
- context.fk_rollouts(q_trajs) returns positions with shape
  (n_rollouts, horizon + 1, 5, 3) for spine3, left_collar, left_shoulder,
  left_elbow, left_wrist.
- context.mdm_positions has shape (T, 5, 3) — a numpy array. Use numpy
  indexing only: context.mdm_positions[:, 4, 2] gives wrist z over the MDM
  trajectory; context.mdm_positions[-1, 4, :] gives the final wrist position.
  Do NOT use dict-style access like context.mdm_positions['left_wrist'] — it
  will raise a TypeError at runtime.
- context.current_positions has shape (5, 3) — a numpy array. Use numpy
  indexing: context.current_positions[4] gives the current wrist position.
  Do NOT use dict-style access.
- context.joint_index(name) accepts spine3, collar, shoulder, elbow, wrist and
  left_* aliases and returns an integer index (0–4).
- If you need statistics such as maximum wrist z during the MDM trajectory,
  read those values from the Summaries section below and encode them as
  hardcoded floats or params in your cost function — do not try to compute
  them from context.mdm_positions using dict-style access.
- context.mdm_traj and context.recent_q/recent_positions are also available
  as numpy arrays.
- np is available. Do not import anything.
- Return a finite numpy array with shape (n_rollouts,).

Hard requirements:
- Return only JSON with keys: description, code, params, explanation, recipient_explanation.
- Prefer costs over future timesteps q_trajs[:, 1:], not only the initial state.
- Do not include smoothness, velocity, or acceleration costs.
- Avoid exact Cartesian point matching for wrists, elbows, or other body parts by default. If a Cartesian location is useful, prefer a region with a tolerance/radius or one-sided bounds, and penalize only violations outside that region. Use an exact point only when the instruction clearly calls for a precise target or contact point.
- Use the least number of cost terms possible. If one term explains the realistic human preference, use one term.
- Params should usually contain no more than three values; if more are needed, simplify the cost.
- The description field must state the caregiver preference the cost encodes in plain human language — how a caregiver would explain what they are trying to achieve. Do not describe the code structure or mathematical approach.
- The explanation field must describe in 3–6 sentences: (1) what preferences the cost captures and why they are implied by the instruction and trajectory, (2) which cost terms were chosen and what each one penalizes, and (3) which specific numbers were drawn from the trajectory data and where they came from (e.g. "the wrist max-z of 0.42 m came from the 95th percentile of the MDM wrist-z range"). This field is for developer understanding — be specific and concrete.
- The recipient_explanation field must be 1–3 brief, plain-language sentences you would say directly to the care recipient. Avoid technical terms, measurements, code, "cost", "penalty", and robot-control jargon. Explain how the new cost functions will guide their arm movement in layman terms; they should be understand what motions their arms will and won't be able to make with the cost functions.

Summaries:
{summaries}
""".strip()


# Registry of available prompt templates, keyed by name. Add a variant by adding
# an entry; select it via ``llm_cost.prompt`` in the MPC config.
PROMPTS: dict[str, str] = {
    "default": _DEFAULT_PROMPT,
}


def build_llm_cost_prompt(
    instruction: str,
    summaries: dict[str, Any],
    image_paths: list[Path],
    prompt: str = "default",
) -> str:
    """Build the text prompt for the cost-generator LLM.

    Args:
        instruction: The user/caregiver correction text.
        summaries: JSON-serializable trajectory summaries.
        image_paths: Rendered overlay images attached to the request, if any.
        prompt: Name of the registered template in :data:`PROMPTS`.
    """
    try:
        template = PROMPTS[prompt]
    except KeyError as exc:
        raise ValueError(
            f"Unknown llm_cost prompt {prompt!r}; available: {sorted(PROMPTS)}"
        ) from exc
    image_section = "An image of the trajectory is attached." if image_paths else ""
    return (
        template.replace("{instruction}", instruction)
        .replace("{image_section}", image_section)
        .replace("{summaries}", json.dumps(summaries, indent=2, sort_keys=True))
    )
