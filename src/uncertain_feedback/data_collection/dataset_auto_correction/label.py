"""Flask web app for captioning a generated correction-clip set.

The labeling step between stage (a) and stage (b) of the correction-clip
pipeline. ``generate.py`` writes a manifest whose ``captions`` lists are empty;
this serves each run's motion next to the bound that produced it and writes typed
captions straight back into that manifest, so ``build_dataset.py`` can read it
with no further hand-editing::

    uv run python \\
        src/uncertain_feedback/data_collection/dataset_auto_correction/generate.py

    uv run python \\
        src/uncertain_feedback/data_collection/dataset_auto_correction/label.py \\
        [--clips_dir <clip set>] [--port 6768]

Then open http://localhost:6768 (SSH-tunnel:
``ssh -L 6768:localhost:6768 user@host``). Captions save on blur; runs left blank
are skipped by stage (b).

A run holds *several* captions — ``+ caption`` adds a row, ``✕`` drops one — and
stage (b) writes them as the lines of that clip's text file, which the humanml
loader samples one of per epoch. Several phrasings of one correction therefore
train as one motion, not as duplicates of it.

Each launch forks its own session directory off the clip set
(``<clips_dir>/session_<timestamp>/``, see :func:`new_session_dir`) and labels
there, so a session never writes into the set it came from or into an earlier
session. Pass ``--resume`` with a session directory to carry on captioning it,
and pass several directories to stage (b) to train on them together.

Each run animates in three orthogonal projections drawn in the browser from the
trajectories on disk — stage (a) writes no video, so a clip set stays a few hundred
KB. Playback covers the *whole* story, walking that run's naive rollout from frame 0
to the trigger and then into the corrected window, with the transition marked on the
scrubber and in the wrist trail. Reads only ``manifest.json``, ``geometry.npz`` and
the per-run ``naive.npy`` / ``clip.npy``, so it needs neither the MDM environment
nor a GPU.

**Draft caption** hands the first sentence to a VLM: the server renders *one* image
of the selected window — the arm where it starts, the arm where it ends, and the
paths the wrist and elbow sweep between them — shows it to ``llm_cost.model``, and
appends the reply as a new caption row for the
labeler to edit or drop. The number beside the button is how many drafts one click
asks for: they come back from *one* completion, one per line, so the model can make
them differ on purpose rather than by sampling noise
(:data:`~uncertain_feedback.data_collection.dataset_auto_correction.captioning.DRAFT_N_INSTRUCTION`
asks it to vary both wording and level of abstraction). The prompt behind them is
editable in the *Draft caption prompt* panel, which starts at
:data:`~uncertain_feedback.data_collection.dataset_auto_correction.captioning.DRAFT_PROMPT`
and is stored per session as the manifest's ``caption_prompt``. It is the only part of this tool that reaches the network, and
it needs ``OPENAI_API_KEY`` in the environment.

``autolabel.py`` runs that same draft over a whole clip set with no human in the
loop; both call :mod:`.captioning`, so the two label sets are drawn the same way.
"""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import flask
import numpy as np
from flask import Flask, jsonify, render_template_string, request
from flask.typing import ResponseReturnValue

from uncertain_feedback.data_collection.common.paths import DEFAULT_CLIP_SET
from uncertain_feedback.data_collection.dataset_auto_correction.captioning import (
    DRAFT_PROMPT,
    MAX_DRAFTS,
    caption_model,
    caption_prompt_from,
    draft_captions,
)
from uncertain_feedback.data_collection.dataset_auto_correction.clips import (
    MAX_WINDOW,
    MIN_WINDOW,
    ClipSource,
    arm_positions,
    clip_source_from_dir,
    motion_frames,
    new_session_dir,
)
from uncertain_feedback.planners.mpc.kinematics import (
    LEFT_ARM_BONE_PAIRS_22,
    SMPL_BONE_PAIRS_22,
    SmplLeftArmFK,
)
from uncertain_feedback.utils.smpl_mesh import SmplMeshCache

if TYPE_CHECKING:
    from uncertain_feedback.llm.openai_model import OpenAIModel

app = Flask(__name__)

DEFAULT_CLIPS_DIR = DEFAULT_CLIP_SET

# The 5-joint arm chain fk() returns is sequential:
# [spine3, left_collar, left_shoulder, left_elbow, left_wrist].
_ARM_BONES = [[i, i + 1] for i in range(4)]
_WRIST = 4
_BODY_BONES = [list(p) for p in SMPL_BONE_PAIRS_22 if p not in LEFT_ARM_BONE_PAIRS_22]


# ---------------------------------------------------------------------------
# Manifest and geometry access
# ---------------------------------------------------------------------------


def _clips_dir() -> Path:
    return Path(flask.current_app.config["CLIPS_DIR"])


def _manifest_path() -> Path:
    return _clips_dir() / "manifest.json"


def _load_manifest() -> dict[str, Any]:
    return json.loads(_manifest_path().read_text(encoding="utf-8"))


def _save_manifest(manifest: dict[str, Any]) -> None:
    _manifest_path().write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# On-demand run generation
#
# A run costs one oracle MPC continuation — a few seconds — so runs are created
# as they are needed rather than batched up front. _GEN_LOCK serializes the
# rollouts (a click on Next that races the prefetch waits for the same result
# instead of duplicating it); _FILE_LOCK is held only for the read-modify-write
# of manifest.json, so saving a caption never waits on a rollout.
# ---------------------------------------------------------------------------

_GEN_LOCK = threading.Lock()
_FILE_LOCK = threading.Lock()
# Draft caption renders through matplotlib and then waits on the network, so its
# own lock keeps concurrent clicks from interleaving renders or piling up calls.
_SUGGEST_LOCK = threading.Lock()
_SOURCE: list[ClipSource] = []
_MODEL: list[OpenAIModel] = []
_MESH: list[SmplMeshCache] = []


def _source() -> ClipSource:
    """The lazily-built clip source for this directory."""
    if not _SOURCE:
        _SOURCE.append(clip_source_from_dir(_clips_dir()))
    return _SOURCE[0]


def _row_captions(row: dict[str, Any]) -> list[str]:
    """A run's captions; rows captioned before the list existed hold one string."""
    if "captions" in row:
        return [str(c) for c in row["captions"]]
    return [c for c in [str(row.get("caption", "")).strip()] if c]


def _mesh() -> SmplMeshCache:
    """The clip set's body as a posable SMPL mesh, fitted on first use.

    Fitting the torso is a few seconds of optimization and the body never
    changes within a clip set, so the first Draft caption pays for it and the
    rest reuse it.
    """
    if not _MESH:
        _MESH.append(SmplMeshCache(_geometry()["body_pos"]))
    return _MESH[0]


def _caption_prompt() -> str:
    """This session's Draft caption prompt: the labeler's, else the stock one."""
    return caption_prompt_from(_clips_dir())


def _caption_model() -> OpenAIModel:
    """The lazily-built VLM behind Draft caption; needs ``OPENAI_API_KEY``.

    Built on the first click rather than at startup, so a labeling session that
    never drafts a caption never constructs the OpenAI client.
    """
    if not _MODEL:
        model = _source().run_cfg.llm_cost.model
        if model is None:
            raise ValueError("Draft caption needs llm_cost.model in the planner YAML.")
        _MODEL.append(caption_model(model))
    return _MODEL[0]


def _ensure_run(index: int) -> dict[str, Any] | None:
    """Return run ``index``, generating it if it does not exist yet.

    ``None`` when ``index`` is beyond the next run to create, so a stale client
    cannot open a gap in the manifest.
    """
    with _FILE_LOCK:
        runs = _load_manifest()["runs"]
        if index < len(runs):
            return runs[index]
        if index > len(runs):
            return None

    with _GEN_LOCK:
        with _FILE_LOCK:
            runs = _load_manifest()["runs"]
            if index < len(runs):
                return runs[index]
        row = _source().generate(index)
        with _FILE_LOCK:
            manifest = _load_manifest()
            if index < len(manifest["runs"]):
                return manifest["runs"][index]
            manifest["runs"].append(row)
            _save_manifest(manifest)
        return row


def _prefetch(index: int) -> None:
    """Warm run ``index`` in the background so Next lands instantly."""

    def work() -> None:
        with app.app_context():
            _ensure_run(index)

    threading.Thread(target=work, daemon=True).start()


def _geometry() -> dict[str, Any]:
    """The decoded body geometry plus an FK bound to its collar rotation."""
    geo = np.load(_clips_dir() / _load_manifest()["geometry_file"])
    fk = SmplLeftArmFK()
    fk.collar_aa = geo["collar_aa"]
    return {
        "fk": fk,
        "spine3_pos": geo["spine3_pos"],
        "spine3_aa": geo["spine3_aa"],
        "body_pos": geo["body_pos"],
    }


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_PAGE = """
<!doctype html>
<title>Correction clips — labeling</title>
<style>
  :root { --ink:#1a1a19; --muted:#6b6a63; --line:#d8d7d0; --bg:#faf9f5;
          --accent:#2a78d6; --warn:#eda100; --after:#eb6834; }
  body { margin:0; padding:24px 28px 40px; background:var(--bg); color:var(--ink);
         font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
         display:flex; justify-content:center; }
  .wrap { width:100%; max-width:1080px; }
  h1 { font-size:18px; margin:0 0 2px; }
  .sub { color:var(--muted); font-size:12.5px; }
  .panel { background:#fff; border:1px solid var(--line); border-radius:10px;
           padding:16px 18px; margin-top:16px; }
  .head { display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; }
  .rid { font-weight:600; font-size:15px; }
  .rid.flagged { color:var(--warn); }
  .progress { margin-left:auto; color:var(--muted); font-size:12.5px;
              font-variant-numeric:tabular-nums; }
  .meta { color:var(--muted); font-size:12.5px; margin:8px 0 12px;
          font-variant-numeric:tabular-nums; }
  .meta b { color:var(--ink); font-weight:600; }
  .views { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; }
  .view { position:relative; }
  .view canvas { width:100%; aspect-ratio:1; display:block; background:#fbfbf9;
                 border:1px solid var(--line); border-radius:6px; }
  .view span { position:absolute; top:5px; left:7px; font-size:10.5px;
               color:var(--muted); pointer-events:none; }
  .transport { display:flex; align-items:center; gap:10px; margin-top:10px; }
  button { font:13px inherit; padding:5px 13px; border:1px solid var(--line);
           background:#fff; border-radius:6px; cursor:pointer; color:var(--ink); }
  button:hover:not(:disabled) { border-color:var(--accent); }
  button:disabled { opacity:.45; cursor:default; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff;
                   font-weight:600; }
  .track { position:relative; flex:1; height:20px; }
  input[type=range] { width:100%; margin:0; position:relative; z-index:2; }
  .tick { position:absolute; top:0; width:2px; height:9px;
          background:var(--accent); z-index:1; }
  .tick-end { background:var(--after); }
  .cliprow { display:flex; align-items:center; gap:10px; margin-top:6px; }
  .cliplbl { font-size:11.5px; color:var(--muted); width:52px; text-align:right; }
  .cliptrack { position:relative; flex:1; height:20px; border-radius:5px;
               background:#efeee8; border:1px solid var(--line);
               box-sizing:border-box; }
  .clipspan { position:absolute; top:-1px; bottom:-1px; background:var(--accent);
              border-radius:5px; cursor:grab; touch-action:none; }
  .clipspan.dragging { cursor:grabbing; }
  .clipspan .pre { position:absolute; top:0; bottom:0; left:0;
                   background:rgba(255,255,255,.45); border-right:1px solid #fff;
                   border-radius:5px 0 0 5px; pointer-events:none; }
  .grip { position:absolute; top:-3px; bottom:-3px; width:11px; cursor:ew-resize;
          background:var(--ink); border-radius:3px; touch-action:none;
          border:2px solid #fff; box-sizing:border-box; }
  .grip-l { left:-6px; }
  .grip-r { right:-6px; }
  .key { display:flex; gap:14px; margin-top:8px; font-size:11.5px;
         color:var(--muted); flex-wrap:wrap; }
  .key i { display:inline-block; width:16px; height:2.5px; vertical-align:middle;
           margin-right:5px; border-radius:2px; }
  .fnum { font-size:12px; color:var(--muted); font-variant-numeric:tabular-nums;
          min-width:118px; text-align:right; }
  textarea { width:100%; box-sizing:border-box; margin-top:12px; padding:8px 9px;
             font:14px/1.5 inherit; border:1px solid var(--line);
             border-radius:6px; resize:vertical; min-height:58px; color:var(--ink); }
  textarea:focus { outline:2px solid var(--accent); outline-offset:-1px;
                   border-color:transparent; }
  .caprow { display:flex; align-items:flex-start; gap:8px; }
  .caprow textarea { min-height:44px; }
  .drop { margin-top:12px; color:var(--muted); padding:5px 9px; }
  .drop:hover:not(:disabled) { border-color:var(--after); color:var(--after); }
  .hint { color:var(--muted); font-size:12px; }
  #ndraft { width:48px; font:13px inherit; padding:5px 6px; color:var(--ink);
            border:1px solid var(--line); border-radius:6px; }
  details.prompt { margin-top:14px; }
  details.prompt summary { color:var(--muted); font-size:12.5px; cursor:pointer; }
  details.prompt textarea { min-height:104px; font-size:13px; }
  .nav { display:flex; align-items:center; gap:10px; margin-top:12px; }
  .status { color:var(--muted); font-size:12px; }
  .status.saved { color:var(--accent); }
  #busy { display:none; color:var(--accent); font-size:12.5px; }
  #busy.on { display:inline; }
</style>

<div class="wrap">
  <h1>Correction clips — labeling</h1>
  <div class="sub">
    One run at a time. <b>Describe the blue part</b> — that is what becomes the training
    clip — as the person would ask for it, then press Next; the following run is planned
    on demand while you type. Orange is the rest of the oracle rollout — <b>drag the blue
    clip bar</b> onto any stretch of it that looks more interesting, by either end to
    resize or in the middle to slide. The pale head of the bar is the pinned prefix
    (conditioning, not described); everything solid blue is what your caption covers.
  </div>

  <div class="panel">
    <div class="head">
      <span class="rid" id="rid">…</span>
      <span class="progress" id="progress"></span>
    </div>
    <div class="meta" id="meta"></div>
    <div class="views">
      {% for label in view_labels %}
        <div class="view"><canvas></canvas><span>{{ label }}</span></div>
      {% endfor %}
    </div>
    <div class="transport">
      <button id="play">▶</button>
      <div class="track"><div class="tick" id="tick"></div>
        <div class="tick tick-end" id="tick2"></div>
        <input type="range" id="scrub" min="0" value="0"></div>
      <div class="fnum" id="fnum"></div>
    </div>
    <div class="cliprow">
      <span class="cliplbl">clip</span>
      <div class="cliptrack" id="cliptrack">
        <div class="clipspan" id="clipspan">
          <div class="grip grip-l" id="gripL"></div>
          <div class="pre" id="clippre"></div>
          <div class="grip grip-r" id="gripR"></div>
        </div>
      </div>
      <div class="fnum" id="clipnum"></div>
    </div>
    <div class="key">
      <span><i style="background:#9b9a92"></i>naive approach</span>
      <span><i style="background:#2a78d6"></i>the clip — caption this</span>
      <span><i style="background:#eb6834"></i>rest of the rollout (context)</span>
      <span><i style="background:#e34948;height:8px;width:8px;border-radius:50%"></i>goal</span>
    </div>
    <div id="captions"></div>
    <div class="nav">
      <button id="addcap">+ caption</button>
      <span class="hint">each caption becomes one line of this clip's text file</span>
    </div>
    <div class="nav">
      <button id="prev">← Prev</button>
      <button id="next" class="primary">Next →</button>
      <button id="draft">Draft caption</button>
      <input type="number" id="ndraft" min="1" max="5" value="1"
             title="drafts to add per click">
      <span id="busy">planning the next correction…</span>
      <span class="status" id="status"></span>
    </div>
    <details class="prompt">
      <summary>Draft caption prompt</summary>
      <textarea id="prompt">{{ caption_prompt }}</textarea>
      <div class="nav"><button id="resetprompt">restore default</button>
        <span class="hint">sent with the two rendered poses; saved with this session</span>
      </div>
    </details>
  </div>
  <div class="sub" style="margin-top:10px">writing to {{ manifest_path }}</div>
</div>

<script>
const VIEWS = [[0, 1], [2, 1], [0, 2]];
// Each panel puts the first axis rightward and the second up, which places the
// camera on their cross product: +Z for Front, but -X for Side and -Y for Top —
// the person's right, and below the floor. Mirroring those two horizontally puts
// the viewer on the person's left and above instead, so the left arm is the near
// one in every panel and does not read as the right arm.
const VIEW_FLIP = [false, true, true];
// Blue = the frames that become the training clip; orange = the rest of the oracle
// rollout, where the correction was still heading. Adjacent slots in the project's
// validated categorical order, so the pair stays separable under CVD.
const GRAY = '#9b9a92', BLUE = '#2a78d6', AFTER = '#eb6834',
      BODY = '#dcdbd4', INK = '#1a1a19';
// The clip spans [anchor - n_prefix + 1 .. clip_end]; frames up to `anchor` are the
// pinned prefix (conditioning) and the rest is what the caption describes. Outside
// the clip: gray before the bound trips, orange after.
const inClip = (s, m) => s > m.anchor && s <= m.clip_end;
const inPrefix = (s, m) => s > m.anchor - m.n_prefix && s <= m.anchor;
const phaseColor = (s, m) =>
  inClip(s, m) ? BLUE : (inPrefix(s, m) ? BLUE : (s <= m.transition ? GRAY : AFTER));
const phaseAlpha = (s, m) => (inPrefix(s, m) && !inClip(s, m) ? .45 : 1);
const phaseName = (s, m) => inClip(s, m) ? 'clip'
  : (inPrefix(s, m) ? 'clip prefix (pinned)'
  : (s <= m.transition ? 'naive approach' : 'rollout, outside clip'));
const DEFAULT_PROMPT = {{ default_prompt|tojson }};
const canvases = [...document.querySelectorAll('canvas')];
const el = id => document.getElementById(id);
let cur = null, idx = 0, timer = null;

function project(pts, ia, ib, box, pad) {
  return pts.map(p => [pad + (p[ia] - box.lo[ia]) * box.k,
                       box.h - pad - (p[ib] - box.lo[ib]) * box.k]);
}

function drawView(cv, ia, ib, m, frame, flip) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr)) { cv.width = w * dpr; cv.height = h * dpr; }
  const c = cv.getContext('2d');
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.clearRect(0, 0, w, h);
  if (flip) { c.translate(w, 0); c.scale(-1, 1); }

  // One shared square scale per view so the arm never rescales mid-playback.
  const pad = 14, span = Math.max(m.span[ia], m.span[ib]) || 0.1;
  const box = {lo: m.lo, k: (Math.min(w, h) - 2 * pad) / span, h: h};
  const P = pts => project(pts, ia, ib, box, pad);

  c.lineWidth = 1.5; c.strokeStyle = BODY;
  const body = P(m.body);
  for (const [i, j] of m.body_bones) {
    c.beginPath(); c.moveTo(...body[i]); c.lineTo(...body[j]); c.stroke();
  }

  // Whole wrist path is always drawn, so the clip is seen against where the
  // correction was heading; the part not yet played is faint.
  const trail = P(m.frames.map(f => f[{{ wrist }}]));
  for (let s = 1; s < trail.length; s++) {
    c.strokeStyle = phaseColor(s, m);
    c.globalAlpha = (s <= frame ? 1 : .25) * phaseAlpha(s, m);
    c.lineWidth = inClip(s, m) ? 2.4 : 1.3;
    c.beginPath(); c.moveTo(...trail[s - 1]); c.lineTo(...trail[s]); c.stroke();
  }
  c.globalAlpha = 1;

  if (frame > m.transition) {   // ghost of the pose the correction departs from
    const at = P(m.frames[m.transition]);
    c.strokeStyle = INK; c.globalAlpha = .22; c.lineWidth = 1.6;
    for (const [i, j] of m.arm_bones) {
      c.beginPath(); c.moveTo(...at[i]); c.lineTo(...at[j]); c.stroke();
    }
    c.globalAlpha = 1;
  }

  const arm = P(m.frames[frame]);
  const col = phaseColor(frame, m);
  c.globalAlpha = phaseAlpha(frame, m);
  c.strokeStyle = col; c.lineWidth = 2.8;
  for (const [i, j] of m.arm_bones) {
    c.beginPath(); c.moveTo(...arm[i]); c.lineTo(...arm[j]); c.stroke();
  }
  c.fillStyle = col;
  for (const p of arm) { c.beginPath(); c.arc(p[0], p[1], 2.8, 0, 7); c.fill(); }

  const [gx, gy] = P([m.goal])[0];
  c.fillStyle = '#e34948';
  c.beginPath(); c.arc(gx, gy, 4.4, 0, 7); c.fill();
  c.strokeStyle = '#fff'; c.lineWidth = 1; c.stroke();
}

function render() {
  if (!cur) return;
  const m = cur.motion, f = +el('scrub').value, last = m.frames.length - 1;
  canvases.forEach((cv, i) =>
    drawView(cv, VIEWS[i][0], VIEWS[i][1], m, f, VIEW_FLIP[i]));
  el('fnum').textContent = `${f} / ${last} · ${phaseName(f, m)}`;

  // Clip bar mirrors the same [anchor - n_prefix + 1 .. clip_end] span.
  const from = Math.max(0, m.anchor - m.n_prefix + 1);
  const pct = i => (i / last) * 100;
  el('clipspan').style.left = pct(from) + '%';
  el('clipspan').style.width = (pct(m.clip_end) - pct(from)) + '%';
  const preFrac = (m.anchor - from + 1) / (m.clip_end - from + 1);
  el('clippre').style.width = (preFrac * 100) + '%';
  el('tick').style.left = `calc(${pct(m.transition)}% - 1px)`;
  el('tick2').style.left = `calc(${pct(m.clip_end)}% - 1px)`;
  el('tick2').style.display = m.clip_end < last ? 'block' : 'none';
  el('clipnum').textContent =
    `${m.n_prefix}+${m.clip_end - m.anchor} = ${m.n_prefix + m.clip_end - m.anchor} fr`;
}

// Drag the clip span (whole thing, or either end) along the motion. Colours and
// the frame counter update live; the re-cut is written once on release.
function initDrag() {
  const track = el('cliptrack');
  let mode = null, grabbed = 0;

  const frameAt = ev => {
    const r = track.getBoundingClientRect();
    const t = Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width));
    return Math.round(t * (cur.motion.frames.length - 1));
  };

  const start = (m_, ev) => {
    if (!cur) return;
    mode = m_; grabbed = frameAt(ev) - cur.motion.anchor;
    el('clipspan').classList.add('dragging');
    ev.target.setPointerCapture(ev.pointerId);
    ev.preventDefault();
  };

  el('gripL').addEventListener('pointerdown', ev => start('left', ev));
  el('gripR').addEventListener('pointerdown', ev => start('right', ev));
  el('clipspan').addEventListener('pointerdown', ev => {
    if (ev.target.classList.contains('grip')) return;
    start('move', ev);
  });

  const onMove = ev => {
    if (!mode || !cur) return;
    const m = cur.motion, last = m.frames.length - 1, f = frameAt(ev);
    let anchor = m.anchor, win = m.clip_end - m.anchor;
    if (mode === 'left') { anchor = f; win = m.clip_end - anchor; }
    else if (mode === 'right') { win = f - m.anchor; }
    else { anchor = f - grabbed; }
    win = Math.min(m.max_window, Math.max(m.min_window, win));
    anchor = Math.min(last - 1, Math.max(0, anchor));
    win = Math.min(win, last - anchor);
    if (win < m.min_window) { anchor = Math.max(0, last - m.min_window); win = m.min_window; }
    m.anchor = anchor; m.clip_end = anchor + win;
    render();
  };
  window.addEventListener('pointermove', onMove);

  window.addEventListener('pointerup', async () => {
    if (!mode) return;
    mode = null;
    el('clipspan').classList.remove('dragging');
    const m = cur.motion;
    const res = await fetch('/clip', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({run_id: cur.run.run_id, anchor: m.anchor,
                            window: m.clip_end - m.anchor})
    });
    const data = await res.json();
    cur.run = data.run;                       // clamped values + fresh violation
    m.anchor = data.run.clip_anchor;
    m.clip_end = Math.min(m.anchor + data.run.correction_frames, m.frames.length - 1);
    showMeta();
    render();
    el('status').className = 'status saved';
    el('status').textContent = 'clip re-cut';
  });
}

function stop() {
  if (timer) { clearInterval(timer); timer = null; el('play').textContent = '▶'; }
}

function showMeta() {
  const r = cur.run, flagged = !r.continuation_reach.reached;
  el('rid').textContent = r.run_id + (flagged ? ' · did not reach goal' : '');
  el('rid').className = 'rid' + (flagged ? ' flagged' : '');
  el('meta').innerHTML =
    `<b>${r.feature.replace(/_/g, ' ')}</b> · ${r.bound_type.replace(/_/g, ' ')} ` +
    `at ${r.bound_value.toFixed(3)} rad<br>trigger step ${r.trigger_step} · ` +
    `clip ${r.clip_frames} frames from anchor ${r.clip_anchor}` +
    (r.pad_frames ? ` (${r.pad_frames} held)` : '') +
    ` · residual violation ${r.window_violation.max_violation.toFixed(3)} rad`;
}

// A run carries a list of captions. The rows live in the DOM and every save
// sends all of them, so an emptied box is a deletion and the server never has to
// merge partial edits.
const capRows = () => [...document.querySelectorAll('#captions textarea')];

function addCap(text, focus) {
  const row = document.createElement('div');
  row.className = 'caprow';
  const box = document.createElement('textarea');
  box.placeholder = 'e.g. raise my arm without twisting my shoulder inward';
  box.value = text || '';
  box.addEventListener('blur', saveCaption);
  const drop = document.createElement('button');
  drop.className = 'drop';
  drop.textContent = '✕';
  drop.title = 'remove this caption';
  drop.addEventListener('click', async () => {
    row.remove();
    if (!capRows().length) addCap('', false);
    await saveCaption();
  });
  row.append(box, drop);
  el('captions').append(row);
  if (focus) box.focus();
  return box;
}

function showCaptions(list) {
  el('captions').innerHTML = '';
  (list && list.length ? list : ['']).forEach(c => addCap(c, false));
}

async function saveCaption() {
  if (!cur) return;
  await fetch('/caption', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({run_id: cur.run.run_id, captions: capRows().map(b => b.value)})
  }).then(r => r.json()).then(d => { cur.done = d.done; cur.run.captions = d.captions; });
  el('status').className = 'status saved';
  el('status').textContent = 'saved';
}

// Per session, not per run: the manifest keeps it, so a reopened session drafts
// with the wording the labeler settled on.
async function savePrompt() {
  await fetch('/prompt', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({prompt: el('prompt').value})
  });
  el('status').className = 'status saved';
  el('status').textContent = 'prompt saved';
}

async function show(i) {
  stop();
  el('busy').classList.add('on');
  el('next').disabled = el('prev').disabled = true;
  const res = await fetch('/run/' + i);
  el('busy').classList.remove('on');
  if (!res.ok) { el('next').disabled = false; el('prev').disabled = i === 0; return; }
  cur = await res.json();
  idx = cur.index;

  const r = cur.run, m = cur.motion;
  showMeta();
  el('progress').textContent =
    `run ${idx + 1} · ${cur.generated} generated · ${cur.done} captioned`;
  showCaptions(r.captions);
  el('scrub').max = m.frames.length - 1;
  el('scrub').value = 0;
  el('prev').disabled = idx === 0;
  el('next').disabled = false;
  el('status').textContent = '';
  render();
  capRows()[0].focus();
}

el('scrub').addEventListener('input', () => { stop(); render(); });
window.addEventListener('resize', render);

el('play').addEventListener('click', () => {
  if (timer) { stop(); return; }
  el('play').textContent = '❚❚';
  timer = setInterval(() => {
    const s = el('scrub');
    s.value = (+s.value >= +s.max) ? 0 : +s.value + 1;
    render();
  }, 50);   // 20 fps, the project's MOTION_FPS
});

el('next').addEventListener('click', async () => {
  await saveCaption();
  await show(idx + 1);
});

el('addcap').addEventListener('click', () => addCap('', true));

el('prev').addEventListener('click', async () => {
  await saveCaption();
  if (idx > 0) await show(idx - 1);
});

// Ask the VLM about the window as it is currently cut, under whatever prompt the
// panel holds. Drafts arrive as new caption rows and are saved by the same blur
// handler as typed ones, so nothing is written without the labeler seeing it.
el('draft').addEventListener('click', async () => {
  if (!cur) return;
  const asked = cur.run.run_id;
  el('draft').disabled = true;
  el('status').className = 'status';
  el('status').textContent = 'drafting…';
  const res = await fetch('/suggest', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({run_id: asked, n: +el('ndraft').value,
                          prompt: el('prompt').value})
  });
  el('draft').disabled = false;
  if (!res.ok) { el('status').textContent = 'draft failed — see the server log'; return; }
  const data = await res.json();
  if (!cur || cur.run.run_id !== asked) return;   // labeler moved on while it ran
  capRows().forEach(b => { if (!b.value.trim()) b.parentElement.remove(); });
  const added = data.captions.map(c => addCap(c, false));
  el('status').textContent = `drafted ${added.length} — edit, then Next`;
  if (!added.length) addCap('', true); else added[added.length - 1].focus();
});

el('prompt').addEventListener('blur', savePrompt);

el('resetprompt').addEventListener('click', async () => {
  el('prompt').value = DEFAULT_PROMPT;
  await savePrompt();
});

initDrag();
show(0);
</script>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def page() -> str:
    """Render the single-run labeling view."""
    return render_template_string(
        _PAGE,
        manifest_path=_manifest_path(),
        view_labels=[
            "Front (XY) — from the front",
            "Side (ZY) — from the person's left",
            "Top (XZ) — from above",
        ],
        wrist=_WRIST,
        caption_prompt=_caption_prompt(),
        default_prompt=DRAFT_PROMPT,
    )


@app.route("/run/<int:index>")
def run(index: int) -> ResponseReturnValue:
    """Run ``index`` — generated on demand — with its motion and progress counts."""
    row = _ensure_run(index)
    if row is None:
        return jsonify({"error": "index beyond the next run"}), 404
    manifest = _load_manifest()
    _prefetch(index + 1)
    return jsonify(
        {
            "run": {**row, "captions": _row_captions(row)},
            "motion": _motion(manifest, row),
            "index": index,
            "generated": len(manifest["runs"]),
            "done": sum(1 for r in manifest["runs"] if _row_captions(r)),
        }
    )


def _motion(manifest: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """One run's whole motion as arm-chain positions, plus the transition index."""
    clips_dir = _clips_dir()
    geo = _geometry()

    naive = np.load(clips_dir / row["naive_file"])
    continuation = np.load(clips_dir / row["continuation_file"])
    motion, transition = motion_frames(naive, continuation, row["trigger_step"])
    anchor = row["clip_anchor"]
    clip_end = min(anchor + row["correction_frames"], len(motion) - 1)
    positions = arm_positions(motion, geo["fk"], geo["spine3_pos"], geo["spine3_aa"])

    body = np.asarray(geo["body_pos"], dtype=np.float64)
    goal = np.asarray(row["goal"], dtype=np.float64) + geo["spine3_pos"]
    extent = np.concatenate([positions.reshape(-1, 3), body, goal[None]])
    lo, hi = extent.min(axis=0), extent.max(axis=0)
    return {
        "frames": np.round(positions, 5).tolist(),
        "body": np.round(body, 5).tolist(),
        "goal": np.round(goal, 5).tolist(),
        "transition": transition,
        "anchor": anchor,
        "clip_end": clip_end,
        "n_prefix": manifest["n_prefix_frames"],
        "min_window": MIN_WINDOW,
        "max_window": MAX_WINDOW,
        "arm_bones": _ARM_BONES,
        "body_bones": _BODY_BONES,
        "lo": lo.tolist(),
        "span": (hi - lo).tolist(),
    }


@app.route("/clip", methods=["POST"])
def clip() -> ResponseReturnValue:
    """Re-cut one run's clip at a dragged ``(anchor, window)``.

    Only the clip files and the derived manifest fields change — the sampled
    bound, the naive rollout and the oracle continuation are untouched, so moving
    the window never needs a replan.
    """
    payload: Any = request.get_json(silent=True) or {}
    clips_dir = _clips_dir()
    with _FILE_LOCK:
        manifest = _load_manifest()
        row = next(r for r in manifest["runs"] if r["run_id"] == payload.get("run_id"))
        naive = np.load(clips_dir / row["naive_file"])
        continuation = np.load(clips_dir / row["continuation_file"])
        motion, _ = motion_frames(naive, continuation, row["trigger_step"])
        row.update(
            _source().cut(
                row,
                motion,
                anchor=int(payload["anchor"]),
                window=int(payload["window"]),
            )
        )
        _save_manifest(manifest)
    return jsonify({"ok": True, "run": row})


@app.route("/suggest", methods=["POST"])
def suggest() -> ResponseReturnValue:
    """Draft a caption for the run's *currently selected* window with the VLM.

    Renders the described window as one image — start pose, end pose and the wrist
    and elbow traces between them (``/clip`` rewrites the window on every drag, so
    the draft always describes what the labeler is looking at) — and asks the model
    for ``n`` sentences under the panel's prompt, in one completion so the phrasings
    differ deliberately. The manifest is untouched: drafts land in caption rows and
    save on blur like typed ones.
    """
    payload: Any = request.get_json(silent=True) or {}
    with _FILE_LOCK:
        row = next(
            r for r in _load_manifest()["runs"] if r["run_id"] == payload.get("run_id")
        )
    prompt = str(payload.get("prompt", "")).strip() or _caption_prompt()
    n_drafts = min(max(int(payload.get("n", 1)), 1), MAX_DRAFTS)
    with _SUGGEST_LOCK:
        captions = draft_captions(
            _clips_dir(),
            row,
            _source().context,
            _mesh(),
            _caption_model(),
            prompt,
            n_drafts,
        )
    return jsonify({"captions": captions})


@app.route("/caption", methods=["POST"])
def caption() -> ResponseReturnValue:
    """Write one run's captions into the manifest.

    The client sends every row it holds, so an emptied box is a deletion; blanks
    are dropped here rather than stored.
    """
    payload: Any = request.get_json(silent=True) or {}
    captions = [str(c).strip() for c in payload.get("captions", []) if str(c).strip()]
    with _FILE_LOCK:
        manifest = _load_manifest()
        for row in manifest["runs"]:
            if row["run_id"] == payload.get("run_id"):
                row["captions"] = captions
        _save_manifest(manifest)
        done = sum(1 for r in manifest["runs"] if _row_captions(r))
    return jsonify({"ok": True, "done": done, "captions": captions})


@app.route("/prompt", methods=["POST"])
def caption_prompt() -> ResponseReturnValue:
    """Store this session's Draft caption prompt; blank restores the stock one."""
    payload: Any = request.get_json(silent=True) or {}
    with _FILE_LOCK:
        manifest = _load_manifest()
        manifest["caption_prompt"] = (
            str(payload.get("prompt", "")).strip() or DRAFT_PROMPT
        )
        _save_manifest(manifest)
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and start the captioning server."""
    parser = argparse.ArgumentParser(
        description="Web interface for captioning a generated correction-clip set."
    )
    parser.add_argument(
        "--clips_dir",
        default=DEFAULT_CLIPS_DIR,
        help=(
            "Clip set from generate.py (default: "
            f"{DEFAULT_CLIPS_DIR}). A new session directory is forked off it, so "
            "the set itself is never written to."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Label into --clips_dir itself instead of forking a new session — "
            "point it at a session directory to carry on captioning that one."
        ),
    )
    parser.add_argument(
        "--port", type=int, default=6768, help="Port to serve on (default: 6768)."
    )
    args = parser.parse_args()

    base_dir = Path(args.clips_dir).expanduser().resolve()
    clips_dir = base_dir if args.resume else new_session_dir(base_dir)
    app.config["CLIPS_DIR"] = str(clips_dir)
    print(f"Captioning clips in : {clips_dir}")
    print(f"Writing captions to : {clips_dir / 'manifest.json'}")
    print(f"Open                : http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
