"""Drafting a correction clip's captions with a VLM.

Shared by both captioning front-ends: ``label.py``'s *Draft caption* button, which
offers the drafts to a human to edit, and ``autolabel.py``, which writes them
straight into the manifest. Both render the *same* single image of the described
window — start pose, end pose, and the wrist and elbow traces between them — and
ask for ``n`` phrasings in one completion, so a hand-labeled set and an
auto-labeled one are drawn from the same distribution.

Reads only the artifacts stage (a) wrote, so captioning needs neither the MDM
environment nor a GPU — just ``OPENAI_API_KEY`` and the planner YAML's
``llm_cost.model``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from uncertain_feedback.data_collection.dataset_auto_correction.clips import (
    clip_source_from_dir,
    motion_frames,
)
from uncertain_feedback.llm.openai_model import OpenAIModel
from uncertain_feedback.planners.mpc.arm_features import arm_aa_from_state
from uncertain_feedback.planners.mpc.costs import MpcCostContext
from uncertain_feedback.utils.plot import ArmVisualizer
from uncertain_feedback.utils.smpl_mesh import SmplMeshCache

_LOG = "[autolabel]"

# The stock Draft caption prompt. Not the visual verbalizer's
# :data:`~uncertain_feedback.simulated_users.visual.PROMPT`, which describes the
# before/after image pair that verbalizer renders at evaluation time; this one
# describes the single window image :func:`render_window` draws.
DRAFT_PROMPT = "You are role-playing a care recipient whose arm a caregiver robot is moving. The image shows you from three angles. Your left arm in orange is where it is now, in blue where you want it to end up, and the two traces are the paths your wrist and your elbow take between them, with arrows pointing the way they travel. Say one short sentence to the caregiver to tell them how to move your arm. Sometimes it's easier to talk about how parts of the arm are moving, and sometimes it's easier to talk about joints, and some verbs already imply a direction. Use casual phrases that a real care recipient would actually say, and they don't have to be full sentences. Keep the description simple and pretty much one short phrase, so there shouldn't be commas; don't give a bunch of directions. Do not say things like please."

# Appended to the prompt when one draft call asks for several captions. One
# completion rather than N, so the model sees the phrasings it has already written
# and can make them differ; independent samples drift only as far as temperature
# takes them, which on this task is mostly synonyms.
DRAFT_N_INSTRUCTION = (
    "Give exactly {n} different ways of saying it, one per line, and nothing "
    "else — no numbering, bullets or quotes. Every line asks for the same change, "
    "but vary the expression and the level of abstraction: some naming the body "
    "part and the direction concretely, others asking loosely for the outcome or "
    "for how it should feel."
)

MAX_DRAFTS = 5

_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def draft_lines(text: str, n_drafts: int) -> list[str]:
    """Split one completion into at most ``n_drafts`` caption lines."""
    lines = [
        _LIST_MARKER.sub("", line).strip().strip("\"'") for line in text.splitlines()
    ]
    return [line for line in lines if line][:n_drafts]


def caption_prompt_from(clips_dir: Path) -> str:
    """A clip set's or session's saved Draft caption prompt, else the stock one."""
    manifest = json.loads((clips_dir / "manifest.json").read_text(encoding="utf-8"))
    return str(manifest.get("caption_prompt", DRAFT_PROMPT))


def caption_model(model: str) -> OpenAIModel:
    """The VLM behind Draft caption; needs ``OPENAI_API_KEY``."""
    return OpenAIModel(
        model=model,
        system_prompt=(
            "You answer with short spoken sentences, one per line, and nothing else."
        ),
    )


def render_window(
    clips_dir: Path, row: dict[str, Any], context: MpcCostContext, mesh: SmplMeshCache
) -> Path:
    """Render the run's *currently cut* window as the one image the VLM is shown.

    The window is read off the manifest row rather than passed in, so a draft
    always describes the clip as it stands on disk — which is what ``/clip``
    rewrites on every drag in the labeling UI.
    """
    naive = np.load(clips_dir / row["naive_file"])
    continuation = np.load(clips_dir / row["continuation_file"])
    motion, _ = motion_frames(naive, continuation, row["trigger_step"])
    anchor = row["clip_anchor"]
    end = min(anchor + row["correction_frames"], len(motion) - 1)
    window_aa = np.stack(
        [arm_aa_from_state(state, context) for state in motion[anchor : end + 1]]
    )
    image_path = clips_dir / row["run_id"] / "suggest.png"
    ArmVisualizer(context.fk).render_correction_summary(
        image_path,
        arm_traj=window_aa,
        spine3_pos=context.spine3_pos,
        spine3_aa=context.spine3_aa,
        mesh=mesh,
    )
    return image_path


def draft_captions(
    clips_dir: Path,
    row: dict[str, Any],
    context: MpcCostContext,
    mesh: SmplMeshCache,
    model: OpenAIModel,
    prompt: str,
    n_drafts: int,
) -> list[str]:
    """Ask the VLM for ``n_drafts`` captions of one run's window, in one completion."""
    if n_drafts > 1:
        prompt = f"{prompt}\n\n{DRAFT_N_INSTRUCTION.format(n=n_drafts)}"
    image_path = render_window(clips_dir, row, context, mesh)
    text = model.get_full_output(prompt, image_input=[str(image_path)])
    return draft_lines(text, n_drafts)


def autolabel_clip_set(clips_dir: Path, n_captions: int, prompt: str) -> None:
    """Caption every uncaptioned run in a clip set, saving after each one.

    The body mesh, the VLM client and the clip source are per-set, so they are
    built once and reused; fitting the mesh alone is a few seconds of
    optimization. The manifest is rewritten after every run rather than at the
    end, so an interrupted job keeps what it paid for and re-running skips it.
    """
    manifest_path = clips_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = clip_source_from_dir(clips_dir)
    model_name = source.run_cfg.llm_cost.model
    if model_name is None:
        raise ValueError("Auto-labeling needs llm_cost.model in the planner YAML.")
    mesh = SmplMeshCache(np.load(clips_dir / manifest["geometry_file"])["body_pos"])
    model = caption_model(model_name)
    manifest["caption_prompt"] = prompt
    for row in manifest["runs"]:
        if row.get("captions"):
            continue
        row["captions"] = draft_captions(
            clips_dir, row, source.context, mesh, model, prompt, n_captions
        )
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"{_LOG} {row['run_id']}: {row['captions']}", flush=True)
    done = sum(1 for row in manifest["runs"] if row.get("captions"))
    print(f"{_LOG} {done}/{len(manifest['runs'])} runs captioned in {clips_dir}")
