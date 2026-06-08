"""Flask web app for labeling pre-extracted motion frame sequences.

Run ``extract_all_frames.py`` first to produce a directory of frame folders,
then start this server against that directory::

    uv run python src/uncertain_feedback/data_collection/extract_all_frames.py \\
        --videos_dir ./recordings/ \\
        --frames_dir ./frames/

    uv run python src/uncertain_feedback/data_collection/labeler.py \\
        --frames_dir ./frames/ \\
        [--port 6767]

Then open http://localhost:6767 (SSH-tunnel: ``ssh -L 6767:localhost:6767 user@host``).
Labels are persisted to ``<frames_dir>/labels.json``.

labels.json format::

    {
      "clip01": [
        {"start_frame": 15, "end_frame": 96,
         "start_sec": 1.5, "end_sec": 9.6, "caption": "a person sits down"}
      ]
    }

Clip names are the subdirectory names under *frames_dir* (i.e. the video stems).
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
from pathlib import Path
from typing import Any

import flask
from flask import Flask, Response, jsonify, render_template_string, request

app = Flask(__name__)

# ---------------------------------------------------------------------------
# HTML templates
# ---------------------------------------------------------------------------

_INDEX_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Motion Labeler</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"
        rel="stylesheet">
</head>
<body class="bg-light">
<div class="container py-4" style="max-width:700px">
  <div class="d-flex justify-content-between align-items-center mb-3">
    <h4 class="mb-0">Motion Labeler</h4>
    <span class="text-muted small">{{ labeled }}/{{ total }} clips have segments</span>
  </div>
  <div class="progress mb-3" style="height:6px">
    <div class="progress-bar bg-success"
         style="width:{{ (labeled/total*100)|int if total else 0 }}%"></div>
  </div>
  <ul class="list-group">
  {% for clip in clips %}
    <li class="list-group-item d-flex justify-content-between align-items-center">
      <a href="/label/{{ clip }}" class="text-decoration-none text-dark">{{ clip }}</a>
      {% set segs = labels.get(clip, []) %}
      {% if segs %}
        <span class="badge bg-success rounded-pill">{{ segs|length }} segment(s)</span>
      {% else %}
        <span class="badge bg-secondary rounded-pill">unlabeled</span>
      {% endif %}
    </li>
  {% endfor %}
  </ul>
</div>
</body>
</html>"""

_LABEL_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Label: {{ clip_name }}</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"
        rel="stylesheet">
  <style>
    .frame-box { background:#111; border-radius:6px; overflow:hidden; }
    .frame-box img { width:100%; display:block; object-fit:contain; aspect-ratio:16/9; }
    .seg-item { font-size:.85rem; }

    /* Dual-handle range scrubber */
    .range-wrap {
      position: relative;
      height: 36px;
      display: flex;
      align-items: center;
    }
    .range-track {
      position: absolute;
      left: 0; right: 0;
      height: 6px;
      background: #dee2e6;
      border-radius: 3px;
      pointer-events: none;
    }
    .range-fill {
      position: absolute;
      height: 100%;
      background: #0d6efd;
      border-radius: 3px;
    }
    .range-wrap input[type=range] {
      position: absolute;
      width: 100%;
      margin: 0;
      background: transparent;
      -webkit-appearance: none;
      pointer-events: none;
    }
    .range-wrap input[type=range]::-webkit-slider-thumb {
      -webkit-appearance: none;
      pointer-events: all;
      width: 18px; height: 18px;
      border-radius: 50%;
      border: 2px solid #fff;
      box-shadow: 0 1px 4px rgba(0,0,0,.3);
      cursor: pointer;
    }
    .range-wrap input[type=range]::-moz-range-thumb {
      pointer-events: all;
      width: 18px; height: 18px;
      border-radius: 50%;
      border: 2px solid #fff;
      box-shadow: 0 1px 4px rgba(0,0,0,.3);
      cursor: pointer;
    }
    #start-slider::-webkit-slider-thumb { background: #0d6efd; }
    #start-slider::-moz-range-thumb     { background: #0d6efd; }
    #end-slider::-webkit-slider-thumb   { background: #198754; }
    #end-slider::-moz-range-thumb       { background: #198754; }
    #end-slider { z-index: 2; }
    #start-slider { z-index: 3; }
    .caption-edit {
      outline: none; cursor: text; display: inline-block; min-width: 40px;
      border-bottom: 1px dashed #adb5bd; border-radius: 2px;
    }
    .caption-edit:focus { border-bottom-color: #0d6efd; background: rgba(13,110,253,.05); }
    .review-thumb { height: 48px; border-radius: 4px; background: #111; object-fit: cover; }
  </style>
</head>
<body class="bg-light">
<div class="container-fluid py-3 px-4" style="max-width:1100px">

  <!-- Header -->
  <div class="d-flex justify-content-between align-items-center mb-3">
    <div>
      <h5 class="mb-0">{{ clip_name }}</h5>
      <small class="text-muted">{{ idx }}/{{ total }}
        &nbsp;·&nbsp; {{ frame_count }} frames @ {{ frame_fps|round(1) }} fps
      </small>
    </div>
    <div class="btn-group btn-group-sm">
      {% if prev %}<a href="/label/{{ prev }}" class="btn btn-outline-secondary">&#8592;</a>{% endif %}
      <a href="/" class="btn btn-outline-secondary">List</a>
      {% if next %}<a href="/label/{{ next }}" class="btn btn-outline-secondary">&#8594;</a>{% endif %}
    </div>
  </div>

  <!-- Frame previews -->
  <div class="row g-2 mb-2">
    <div class="col-6">
      <div class="frame-box">
        <img id="start-img" src="/frame/{{ clip_name }}/0" alt="start frame">
      </div>
      <div class="d-flex justify-content-between mt-1 px-1">
        <span class="small fw-semibold text-primary">Start</span>
        <span class="small text-muted" id="start-info">frame 0 · 0.00 s</span>
      </div>
    </div>
    <div class="col-6">
      <div class="frame-box">
        <img id="end-img"
             src="/frame/{{ clip_name }}/{{ [frame_count-1,0]|max }}"
             alt="end frame">
      </div>
      <div class="d-flex justify-content-between mt-1 px-1">
        <span class="small fw-semibold text-success">End</span>
        <span class="small text-muted" id="end-info">
          frame {{ [frame_count-1,0]|max }} · {{ ((frame_count-1)/frame_fps)|round(2) if frame_fps else 0 }} s
        </span>
      </div>
    </div>
  </div>

  <!-- Playback -->
  <div class="card p-2 mb-3">
    <div class="frame-box mb-2">
      <canvas id="play-canvas" width="640" height="360"
              style="width:100%; height:auto; display:block;"></canvas>
    </div>
    <div class="d-flex align-items-center gap-2 mt-1">
      <button id="play-btn" onclick="togglePlay()" class="btn btn-sm btn-outline-primary">&#9654; Play</button>
      <input type="range" id="play-scrubber" class="form-range flex-grow-1"
             min="0" max="0" step="1" value="0">
      <span class="small text-muted" id="play-info" style="white-space:nowrap">frame 0</span>
    </div>
  </div>

  <!-- Dual-handle scrubber -->
  <div class="card p-3 mb-3">
    <div class="range-wrap">
      <div class="range-track">
        <div class="range-fill" id="range-fill"></div>
      </div>
      <input type="range" id="start-slider"
             min="0" max="{{ [frame_count-1, 0]|max }}" step="1" value="0">
      <input type="range" id="end-slider"
             min="0" max="{{ [frame_count-1, 0]|max }}" step="1"
             value="{{ [frame_count-1,0]|max }}">
    </div>
    <div id="segment-bar" style="position:relative;height:8px;background:#dee2e6;border-radius:3px;margin-top:6px;"></div>
    <div class="d-flex align-items-center gap-2 mt-2 flex-wrap">
      <button onclick="randomSnippet()" class="btn btn-outline-secondary btn-sm">&#127922; Random snippet</button>
      <label class="small text-muted mb-0">N =</label>
      <input type="number" id="snippet-n" class="form-control form-control-sm"
             value="20" min="1" max="{{ [frame_count-1,0]|max }}" style="width:80px">
      <span class="small text-muted" id="snippet-duration"></span>
    </div>
  </div>

  <!-- AI review panel -->
  <div class="card p-3 mb-3" id="review-card" style="display:none">
    <div class="d-flex justify-content-between align-items-center mb-2">
      <span class="fw-semibold">Review AI Labels <span id="review-count" class="badge bg-secondary rounded-pill ms-1">0</span></span>
      <button onclick="saveReview()" class="btn btn-success btn-sm">Save all &#10003;</button>
    </div>
    <div id="review-list"></div>
  </div>

  <!-- Caption -->
  <div class="card p-3 mb-3">
    <textarea id="caption-input" class="form-control mb-2" rows="2"
              placeholder="a person sits down"></textarea>
    <div class="d-flex gap-2 align-items-center flex-wrap">
      <button onclick="addSegment()" class="btn btn-primary btn-sm">+ Add segment</button>
      <button id="ai-btn" onclick="aiCaption()" class="btn btn-outline-secondary btn-sm">&#129302; AI labels</button>
      <input type="number" id="ai-variants" class="form-control form-control-sm"
             value="5" min="1" max="10" style="width:60px" title="Number of caption variants">
      {% if next %}
      <button onclick="saveAndNext()" class="btn btn-success btn-sm">Save &amp; Next &#8594;</button>
      {% endif %}
      <span id="add-status" class="text-muted small"></span>
    </div>
  </div>

  <!-- Segment list -->
  <div class="card p-3">
    <div class="d-flex justify-content-between align-items-center mb-2">
      <span class="fw-semibold">Segments</span>
      <span id="seg-count" class="badge bg-secondary rounded-pill">0</span>
    </div>
    <div id="seg-list"></div>
  </div>

</div>

<script>
const CLIP_NAME   = {{ clip_name | tojson }};
const FRAME_COUNT = {{ frame_count }};
const FRAME_FPS   = {{ frame_fps }};
let segments = {{ segments | tojson }};

const startSlider = document.getElementById('start-slider');
const endSlider   = document.getElementById('end-slider');
const startImg    = document.getElementById('start-img');
const endImg      = document.getElementById('end-img');
const startInfo   = document.getElementById('start-info');
const endInfo     = document.getElementById('end-info');
const rangeFill   = document.getElementById('range-fill');

function fmt(idx) {
  const sec = FRAME_FPS > 0 ? (idx / FRAME_FPS).toFixed(2) : '—';
  return `frame ${idx} · ${sec} s`;
}

function updateFill() {
  const max = parseInt(startSlider.max) || 1;
  const s = parseInt(startSlider.value);
  const e = parseInt(endSlider.value);
  rangeFill.style.left  = (s / max * 100) + '%';
  rangeFill.style.width = ((e - s) / max * 100) + '%';
  // keep start thumb on top when handles are close so it stays grabbable
  startSlider.style.zIndex = s > max - 10 ? 4 : 3;
}

function updateStart() {
  const idx = parseInt(startSlider.value);
  startImg.src = `/frame/${CLIP_NAME}/${idx}`;
  startInfo.textContent = fmt(idx);
  if (idx > parseInt(endSlider.value)) {
    endSlider.value = idx;
    endImg.src = `/frame/${CLIP_NAME}/${idx}`;
    endInfo.textContent = fmt(idx);
  }
  updateFill();
}
function updateEnd() {
  const idx = parseInt(endSlider.value);
  endImg.src = `/frame/${CLIP_NAME}/${idx}`;
  endInfo.textContent = fmt(idx);
  if (idx < parseInt(startSlider.value)) {
    startSlider.value = idx;
    startImg.src = `/frame/${CLIP_NAME}/${idx}`;
    startInfo.textContent = fmt(idx);
  }
  updateFill();
}

startSlider.addEventListener('input', updateStart);
endSlider.addEventListener('input', updateEnd);

// ── Random snippet ─────────────────────────────────────────────────────────
const snippetN     = document.getElementById('snippet-n');
const snippetDur   = document.getElementById('snippet-duration');

function updateSnippetDuration() {
  const n = parseInt(snippetN.value) || 0;
  snippetDur.textContent = FRAME_FPS > 0 ? `(${(n / FRAME_FPS).toFixed(1)} s)` : '';
}

function randomSnippet() {
  const n   = Math.max(1, parseInt(snippetN.value) || 20);
  const max = parseInt(startSlider.max);
  if (n > max) return;
  const start = Math.floor(Math.random() * (max - n + 1));
  const end   = start + n;
  startSlider.value = start;
  endSlider.value   = end;
  updateStart();
  updateEnd();
  syncScrubberRange();
  setPlayFrame(start);
}

snippetN.addEventListener('input', updateSnippetDuration);
updateSnippetDuration();

// ── Playback ───────────────────────────────────────────────────────────────
const playCanvas   = document.getElementById('play-canvas');
const playCtx      = playCanvas.getContext('2d');
const playBtn      = document.getElementById('play-btn');
const playInfo     = document.getElementById('play-info');
const playScrubber = document.getElementById('play-scrubber');
let playTimer      = null;
let playIdx        = 0;
const frameCache   = {};  // index → Image object

function drawFrame(img) {
  playCtx.fillStyle = '#111';
  playCtx.fillRect(0, 0, playCanvas.width, playCanvas.height);
  if (img && img.naturalWidth) playCtx.drawImage(img, 0, 0, playCanvas.width, playCanvas.height);
}

function syncScrubberRange() {
  playScrubber.min   = startSlider.value;
  playScrubber.max   = endSlider.value;
}

function setPlayFrame(idx) {
  playIdx = idx;
  playInfo.textContent  = fmt(idx);
  playScrubber.value    = idx;
  const img = frameCache[idx];
  if (img && img.naturalWidth) { drawFrame(img); return; }
  // not cached yet — fetch via a temp Image and draw when ready
  const tmp = new Image();
  tmp.onload = () => { frameCache[idx] = tmp; if (playIdx === idx) drawFrame(tmp); };
  tmp.src = `/frame/${CLIP_NAME}/${idx}`;
}

function stopPlay() {
  if (playTimer) { clearInterval(playTimer); playTimer = null; }
  playBtn.innerHTML = '&#9654; Play';
  playBtn.disabled  = false;
}

function preloadRange(start, end, onReady) {
  let pending = end - start + 1;
  if (pending <= 0) { onReady(); return; }
  for (let i = start; i <= end; i++) {
    if (frameCache[i] && frameCache[i].naturalWidth) { if (--pending === 0) onReady(); continue; }
    const img = new Image();
    img.onload  = () => { frameCache[i] = img; if (--pending === 0) onReady(); };
    img.onerror = () => {                       if (--pending === 0) onReady(); };
    img.src = `/frame/${CLIP_NAME}/${i}`;
    frameCache[i] = img;
  }
}

function startPlay() {
  stopPlay();
  const start = parseInt(startSlider.value);
  const end   = parseInt(endSlider.value);
  playBtn.innerHTML = '&#8987; Loading…';
  playBtn.disabled  = true;
  preloadRange(start, end, () => {
    playIdx = start;
    const fps = FRAME_FPS > 0 ? FRAME_FPS : 20;
    playTimer = setInterval(() => {
      drawFrame(frameCache[playIdx]);
      playInfo.textContent = fmt(playIdx);
      playScrubber.value   = playIdx;
      const endNow = parseInt(endSlider.value);
      playIdx = playIdx + 1 > endNow ? parseInt(startSlider.value) : playIdx + 1;
    }, 1000 / fps);
    playBtn.innerHTML = '&#9646;&#9646; Pause';
    playBtn.disabled  = false;
  });
}

function togglePlay() {
  if (playTimer) stopPlay();
  else startPlay();
}

// Seeking via play scrubber — pauses playback
playScrubber.addEventListener('input', () => {
  stopPlay();
  setPlayFrame(parseInt(playScrubber.value));
});

// Keep scrubber range and preview in sync when snippet changes
startSlider.addEventListener('input', () => { syncScrubberRange(); if (!playTimer) setPlayFrame(parseInt(startSlider.value)); });
endSlider.addEventListener('input',   () => { syncScrubberRange(); if (!playTimer) setPlayFrame(parseInt(startSlider.value)); });

// ── AI captioning ──────────────────────────────────────────────────────────
async function aiCaption() {
  const btn      = document.getElementById('ai-btn');
  const status   = document.getElementById('add-status');
  const variants = Math.max(1, parseInt(document.getElementById('ai-variants').value) || 5);
  btn.disabled = true;
  btn.textContent = `…${variants}`;
  status.textContent = '';
  try {
    const res = await fetch(`/autolabel/${CLIP_NAME}`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        start_frame: parseInt(startSlider.value),
        end_frame:   parseInt(endSlider.value),
        variants,
      }),
    });
    const data = await res.json();
    if (data.captions) {
      review.push({
        start_frame: parseInt(startSlider.value),
        end_frame:   parseInt(endSlider.value),
        captions: data.captions,
      });
      renderReview();
    } else {
      status.textContent = data.error || 'AI caption failed.';
    }
  } catch (e) {
    status.textContent = 'Request failed.';
  } finally {
    btn.disabled = false;
    btn.innerHTML = '&#129302; AI labels';
  }
}

// ── AI review ─────────────────────────────────────────────────────────────
let review = [];

function renderReview() {
  const card = document.getElementById('review-card');
  const list = document.getElementById('review-list');
  const cnt  = document.getElementById('review-count');
  const total = review.reduce((s, r) => s + (r.captions || []).length, 0);
  cnt.textContent = `${review.length} snippets · ${total} captions`;
  if (review.length === 0) { card.style.display = 'none'; return; }
  card.style.display = '';
  list.innerHTML = review.map((item, i) => {
    const captions = item.captions || [];
    const capRows = captions.map((cap, j) => `
      <div class="d-flex align-items-center gap-2 mb-1">
        <input class="form-control form-control-sm flex-grow-1"
               value="${escHtml(cap)}"
               oninput="review[${i}].captions[${j}] = this.value">
        <button class="btn btn-outline-danger btn-sm py-0 px-1"
                onclick="discardCaption(${i},${j})">✕</button>
      </div>
    `).join('');
    const errorRow = item.error
      ? `<div class="small text-danger mt-1">${escHtml(item.error)}</div>` : '';
    return `
      <div class="border rounded p-2 mb-2">
        <div class="d-flex align-items-center gap-2 mb-2">
          <img src="/frame/${CLIP_NAME}/${item.start_frame}" class="review-thumb">
          <img src="/frame/${CLIP_NAME}/${item.end_frame}" class="review-thumb">
          <span class="small text-muted" style="white-space:nowrap">f${item.start_frame}–f${item.end_frame}</span>
          <button class="btn btn-outline-danger btn-sm py-0 px-2 ms-auto"
                  onclick="discardReview(${i})">✕ all</button>
        </div>
        ${capRows}${errorRow}
        <button class="btn btn-outline-secondary btn-sm mt-1 py-0"
                onclick="addReviewCaption(${i})">+ Add label</button>
      </div>
    `;
  }).join('');
}

function addReviewCaption(i) {
  review[i].captions.push('');
  renderReview();
  // focus the new input
  const inputs = document.querySelectorAll(`#review-list .border`)[i]?.querySelectorAll('input');
  if (inputs) inputs[inputs.length - 1]?.focus();
}

function discardCaption(i, j) {
  review[i].captions.splice(j, 1);
  if (review[i].captions.length === 0) review.splice(i, 1);
  renderReview();
}

function discardReview(i) {
  review.splice(i, 1);
  renderReview();
}

function saveReview() {
  review.forEach(r => {
    (r.captions || []).filter(c => c && c.trim()).forEach(cap => {
      segments.push({
        start_frame: r.start_frame, end_frame: r.end_frame,
        start_sec: FRAME_FPS > 0 ? r.start_frame / FRAME_FPS : 0,
        end_sec:   FRAME_FPS > 0 ? r.end_frame   / FRAME_FPS : 0,
        caption: cap.trim(),
      });
    });
  });
  review = [];
  renderReview();
  renderSegments();
  persist(false);
}

// ── Segment management ─────────────────────────────────────────────────────
function renderSegmentBar() {
  const bar = document.getElementById('segment-bar');
  const max = Math.max(FRAME_COUNT - 1, 1);
  bar.innerHTML = segments.map(s => {
    const left  = (s.start_frame / max * 100).toFixed(2);
    const width = Math.max(0.3, (s.end_frame - s.start_frame) / max * 100).toFixed(2);
    return `<div title="${escHtml(s.caption)}" style="position:absolute;left:${left}%;width:${width}%;top:0;bottom:0;background:rgba(25,135,84,0.55);border-radius:2px;"></div>`;
  }).join('');
}

function renderSegments() {
  const el = document.getElementById('seg-list');
  const cnt = document.getElementById('seg-count');
  cnt.textContent = segments.length;
  renderSegmentBar();
  if (segments.length === 0) {
    el.innerHTML = '<p class="text-muted small mb-0">No segments yet.</p>';
    return;
  }
  el.innerHTML = segments.map((s, i) => `
    <div class="seg-item d-flex justify-content-between align-items-start border rounded p-2 mb-1">
      <div class="flex-grow-1 me-2">
        <span class="badge bg-light text-dark border me-1">
          f${s.start_frame}–f${s.end_frame}
          &nbsp;(${s.start_sec.toFixed(2)}s–${s.end_sec.toFixed(2)}s)
        </span>
        <span class="caption-edit" contenteditable="true"
              onblur="updateCaption(${i}, this.textContent)"
              >${escHtml(s.caption)}</span>
      </div>
      <button class="btn btn-outline-danger btn-sm py-0 px-1 ms-2"
              onclick="deleteSeg(${i})">✕</button>
    </div>
  `).join('');
}
function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function updateCaption(i, text) {
  if (segments[i]) { segments[i].caption = text.trim(); persist(false); }
}

function addSegment() {
  const caption = document.getElementById('caption-input').value.trim();
  const status  = document.getElementById('add-status');
  if (!caption) { status.textContent = 'Enter a caption.'; return; }
  const sf = parseInt(startSlider.value);
  const ef = parseInt(endSlider.value);
  if (ef <= sf) { status.textContent = 'End must be after start.'; return; }
  segments.push({
    start_frame: sf, end_frame: ef,
    start_sec: FRAME_FPS > 0 ? sf / FRAME_FPS : 0,
    end_sec:   FRAME_FPS > 0 ? ef / FRAME_FPS : 0,
    caption,
  });
  document.getElementById('caption-input').value = '';
  status.textContent = '';
  renderSegments();
  persist(false);
}

function deleteSeg(i) {
  segments.splice(i, 1);
  renderSegments();
  persist(false);
}

function persist(andNext) {
  fetch('/label/' + CLIP_NAME, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({segments}),
  })
  .then(r => r.json())
  .then(d => { if (d.ok && andNext) window.location.href = '/label/{{ next }}'; });
}

function saveAndNext() { persist(true); }

// ── Boot ───────────────────────────────────────────────────────────────────
updateFill();
syncScrubberRange();
setPlayFrame(parseInt(startSlider.value));
renderSegments();
document.getElementById('caption-input')?.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); addSegment(); }
});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Clip helpers
# ---------------------------------------------------------------------------


def _list_clips(frames_dir: Path) -> list[str]:
    """Return sorted list of clip names (subdirectories containing frame images)."""
    return sorted(
        name
        for name in os.listdir(frames_dir)
        if (frames_dir / name).is_dir() and not name.startswith(".")
    )


def _clip_meta(frames_dir: Path, clip_name: str) -> dict[str, Any]:
    """Return ``{count, fps}`` from the clip's meta.json, or best-effort defaults."""
    meta_path = frames_dir / clip_name / "meta.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
            return data
    # Fall back: count frames by listing the directory
    clip_dir = frames_dir / clip_name
    count = len([p for p in clip_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"}])
    return {"count": count, "fps": 0.0}


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------


def _labels_path() -> Path:
    return Path(flask.current_app.config["FRAMES_DIR"]) / "labels.json"


def _load_labels() -> dict[str, list[dict[str, Any]]]:
    p = _labels_path()
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                data: dict[str, list[dict[str, Any]]] = json.load(f)
                return data
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def _save_labels_to_disk(data: dict[str, list[dict[str, Any]]]) -> None:
    with open(_labels_path(), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index() -> str:
    """Render the clip list with segment counts."""
    frames_dir = Path(flask.current_app.config["FRAMES_DIR"])
    clips = _list_clips(frames_dir)
    labels = _load_labels()
    labeled = sum(1 for c in clips if labels.get(c))
    return render_template_string(
        _INDEX_TEMPLATE,
        clips=clips,
        labels=labels,
        labeled=labeled,
        total=len(clips),
    )


@app.route("/frame/<path:clip_name>/<int:idx>")
def serve_frame(clip_name: str, idx: int) -> Response:
    """Serve the extracted frame at index *idx* (0-based URL, 1-based on disk)."""
    frames_dir = Path(flask.current_app.config["FRAMES_DIR"])
    clip_dir = frames_dir / clip_name
    frame_file = f"frame_{idx + 1:06d}.jpg"
    return flask.send_from_directory(  # type: ignore[return-value]
        str(clip_dir), frame_file
    )


@app.route("/label/<path:clip_name>", methods=["GET"])
def label_page(clip_name: str) -> str:
    """Render the frame-scrubber labeling page for a single clip."""
    frames_dir = Path(flask.current_app.config["FRAMES_DIR"])
    clips = _list_clips(frames_dir)
    labels = _load_labels()
    idx = clips.index(clip_name) if clip_name in clips else 0
    meta = _clip_meta(frames_dir, clip_name)
    return render_template_string(
        _LABEL_TEMPLATE,
        clip_name=clip_name,
        frame_count=meta["count"],
        frame_fps=meta["fps"],
        segments=labels.get(clip_name, []),
        prev=clips[idx - 1] if idx > 0 else None,
        next=clips[idx + 1] if idx < len(clips) - 1 else None,
        idx=idx + 1,
        total=len(clips),
    )


@app.route("/label/<path:clip_name>", methods=["POST"])
def save_labels(clip_name: str) -> Response:
    """Persist segments for a clip to labels.json."""
    payload: Any = request.get_json(silent=True)
    new_segments: list[dict[str, Any]] = (
        payload.get("segments", []) if isinstance(payload, dict) else []
    )
    all_labels = _load_labels()
    all_labels[clip_name] = new_segments
    _save_labels_to_disk(all_labels)
    return jsonify({"ok": True})


def _make_autolabel_prompt(variants: int) -> str:
    shared = (
        "You are helping label motion data for a caregiving robot. "
        "Given two images showing the start and end of an arm motion, "
        "identify which arm is moving (left or right) from the perspective of the person whose arm is moving and what the motion is. "
        "Write phrases in the first person, as if a care recipient is asking their robot caregiver to perform that motion. "
        "Use 'my left arm' or 'my right arm' as appropriate from the perspective of the person whose arm is moving. "
        "Also don't be afraid to change the body part to a more relevant/specific body part that would be natural for the instruction. "
        "For example, elbow, shoulder, hand, forearm, upper arm, or whatever else."
        "Vary the level of specificity: some phrases should be general (e.g. 'lift my right arm'), "
        "others more specific (e.g. 'bring my left arm up a little bit to shoulder height'). "
        "Each phrase must of a natural length for this scenario."
    )
    if variants == 1:
        return shared + " Reply with only one phrase, no punctuation, no explanation."
    return (
        shared
        + f" Write {variants} distinct phrases, one per line, "
        "no numbering, no punctuation, no extra text."
    )


def _autolabel_many(
    start_path: Path, end_path: Path, model: str, variants: int = 1
) -> list[str]:
    """Call OpenAI vision API and return a list of caption strings."""
    import openai  # imported lazily so missing dep only errors on use

    def _b64(p: Path) -> str:
        return base64.b64encode(p.read_bytes()).decode()

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _make_autolabel_prompt(variants)},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(start_path)}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(end_path)}"}},
                ],
            },
        ],
        max_completion_tokens=max(64, 32 * variants),
    )
    text = response.choices[0].message.content.strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines if lines else [text]


@app.route("/autolabel/<path:clip_name>", methods=["POST"])
def autolabel(clip_name: str) -> Response:
    """Generate an AI caption for a snippet using OpenAI vision."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return jsonify({"error": "OPENAI_API_KEY not set"}), 503

    payload: Any = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "invalid payload"}), 400
    start_frame: int = int(payload.get("start_frame", 0))
    end_frame: int = int(payload.get("end_frame", 0))
    variants: int = max(1, int(payload.get("variants", 1)))

    frames_dir = Path(flask.current_app.config["FRAMES_DIR"])
    clip_dir = frames_dir / clip_name
    start_path = clip_dir / f"frame_{start_frame + 1:06d}.jpg"
    end_path   = clip_dir / f"frame_{end_frame + 1:06d}.jpg"

    if not start_path.exists() or not end_path.exists():
        return jsonify({"error": "frame file not found"}), 404

    model = flask.current_app.config.get("OPENAI_MODEL", "gpt-5.4")
    try:
        captions = _autolabel_many(start_path, end_path, model, variants=variants)
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"error": str(exc)}), 500

    return jsonify({"captions": captions})


@app.route("/autolabel-batch/<path:clip_name>", methods=["POST"])
def autolabel_batch(clip_name: str) -> Response:
    """Generate AI captions for multiple snippets in parallel."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return jsonify({"error": "OPENAI_API_KEY not set"}), 503

    payload: Any = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "expected {snippets, variants}"}), 400

    snippets: list[dict[str, Any]] = payload.get("snippets", [])
    variants: int = max(1, int(payload.get("variants", 5)))

    frames_dir = Path(flask.current_app.config["FRAMES_DIR"])
    clip_dir = frames_dir / clip_name
    model = flask.current_app.config.get("OPENAI_MODEL", "gpt-5.4")

    def process(item: dict[str, Any]) -> dict[str, Any]:
        sf = int(item.get("start_frame", 0))
        ef = int(item.get("end_frame", 0))
        start_path = clip_dir / f"frame_{sf + 1:06d}.jpg"
        end_path   = clip_dir / f"frame_{ef + 1:06d}.jpg"
        if not start_path.exists() or not end_path.exists():
            return {"start_frame": sf, "end_frame": ef, "captions": [], "error": "frame not found"}
        try:
            captions = _autolabel_many(start_path, end_path, model, variants=variants)
            return {"start_frame": sf, "end_frame": ef, "captions": captions}
        except Exception as exc:  # pylint: disable=broad-except
            return {"start_frame": sf, "end_frame": ef, "captions": [], "error": str(exc)}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process, snippets))

    return jsonify(results)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and start the labeling server."""
    parser = argparse.ArgumentParser(
        description="Web interface for labeling pre-extracted motion frame sequences."
    )
    parser.add_argument(
        "--frames_dir",
        default=str(Path(__file__).parent / "data" / "frames"),
        help="Directory of per-clip frame subdirectories (default: data_collection/frames/).",
    )
    parser.add_argument(
        "--port", type=int, default=6767, help="Port to serve on (default: 6767)."
    )
    parser.add_argument(
        "--openai-model", default="gpt-5.4", help="OpenAI model for AI captioning (default: gpt-5.4)."
    )
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir).expanduser().resolve()
    frames_dir.mkdir(parents=True, exist_ok=True)

    app.config["FRAMES_DIR"] = str(frames_dir)
    app.config["OPENAI_MODEL"] = args.openai_model
    print(f"Serving frames from : {frames_dir}")
    print(f"Labels saved to     : {frames_dir / 'labels.json'}")
    print(f"Open                : http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
