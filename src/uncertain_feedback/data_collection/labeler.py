"""Flask web app for labeling pre-extracted motion frame sequences.

Run ``extract_all_frames.py`` first to produce a directory of frame folders,
then start this server against that directory::

    uv run python src/uncertain_feedback/data_collection/extract_all_frames.py \\
        --videos_dir ./recordings/ \\
        --frames_dir ./frames/

    uv run python src/uncertain_feedback/data_collection/labeler.py \\
        --frames_dir ./frames/ \\
        [--port 5000]

Then open http://localhost:5000 (SSH-tunnel: ``ssh -L 5000:localhost:5000 user@host``).
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
    .scrubber { width:100%; cursor:pointer; accent-color: #0d6efd; }
    #end-slider { accent-color: #198754; }
    .seg-item { font-size:.85rem; }
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

  <div class="row g-3">

    <!-- Start scrubber -->
    <div class="col-md-5">
      <div class="card p-2">
        <div class="d-flex justify-content-between mb-1">
          <span class="fw-semibold text-primary small">Start frame</span>
          <span class="text-muted small" id="start-info">frame 0 · 0.00 s</span>
        </div>
        <div class="frame-box mb-2">
          <img id="start-img" src="/frame/{{ clip_name }}/0" alt="start frame">
        </div>
        <input type="range" class="scrubber" id="start-slider"
               min="0" max="{{ [frame_count-1, 0]|max }}" step="1" value="0">
      </div>
    </div>

    <!-- End scrubber -->
    <div class="col-md-5">
      <div class="card p-2">
        <div class="d-flex justify-content-between mb-1">
          <span class="fw-semibold text-success small">End frame</span>
          <span class="text-muted small" id="end-info">
            frame {{ [frame_count-1,0]|max }} · {{ ((frame_count-1)/frame_fps)|round(2) if frame_fps else 0 }} s
          </span>
        </div>
        <div class="frame-box mb-2">
          <img id="end-img"
               src="/frame/{{ clip_name }}/{{ [frame_count-1,0]|max }}"
               alt="end frame">
        </div>
        <input type="range" class="scrubber" id="end-slider"
               min="0" max="{{ [frame_count-1,0]|max }}" step="1"
               value="{{ [frame_count-1,0]|max }}">
      </div>
    </div>

    <!-- Caption + segment list -->
    <div class="col-md-2 d-flex flex-column gap-2">
      <div class="card p-2 flex-grow-0">
        <label class="form-label fw-semibold small mb-1">Caption</label>
        <textarea id="caption-input" class="form-control form-control-sm" rows="3"
                  placeholder="a person sits down"></textarea>
        <button onclick="addSegment()" class="btn btn-primary btn-sm mt-2 w-100">
          + Add
        </button>
        <div id="add-status" class="text-muted small mt-1"></div>
      </div>

      {% if next %}
      <button onclick="saveAndNext()" class="btn btn-success btn-sm w-100">
        Save &amp; Next &#8594;
      </button>
      {% endif %}
    </div>
  </div>

  <!-- Segment list -->
  <div class="card mt-3 p-3">
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

// ── Scrubber logic ─────────────────────────────────────────────────────────
const startSlider = document.getElementById('start-slider');
const endSlider   = document.getElementById('end-slider');
const startImg    = document.getElementById('start-img');
const endImg      = document.getElementById('end-img');
const startInfo   = document.getElementById('start-info');
const endInfo     = document.getElementById('end-info');

function fmt(idx) {
  const sec = FRAME_FPS > 0 ? (idx / FRAME_FPS).toFixed(2) : '—';
  return `frame ${idx} · ${sec} s`;
}

function updateStart() {
  const idx = parseInt(startSlider.value);
  startImg.src = `/frame/${CLIP_NAME}/${idx}`;
  startInfo.textContent = fmt(idx);
  // clamp end to start if needed and keep end image in sync
  if (idx > parseInt(endSlider.value)) {
    endSlider.value = idx;
    endImg.src = `/frame/${CLIP_NAME}/${idx}`;
    endInfo.textContent = fmt(idx);
  }
}
function updateEnd() {
  const idx = parseInt(endSlider.value);
  endImg.src = `/frame/${CLIP_NAME}/${idx}`;
  endInfo.textContent = fmt(idx);
  // clamp start to end if needed and keep start image in sync
  if (idx < parseInt(startSlider.value)) {
    startSlider.value = idx;
    startImg.src = `/frame/${CLIP_NAME}/${idx}`;
    startInfo.textContent = fmt(idx);
  }
}

startSlider.addEventListener('input', updateStart);
endSlider.addEventListener('input', updateEnd);

// ── Segment management ─────────────────────────────────────────────────────
function renderSegments() {
  const el = document.getElementById('seg-list');
  const cnt = document.getElementById('seg-count');
  cnt.textContent = segments.length;
  if (segments.length === 0) {
    el.innerHTML = '<p class="text-muted small mb-0">No segments yet.</p>';
    return;
  }
  el.innerHTML = segments.map((s, i) => `
    <div class="seg-item d-flex justify-content-between align-items-start border rounded p-2 mb-1">
      <div>
        <span class="badge bg-light text-dark border me-1">
          f${s.start_frame}–f${s.end_frame}
          &nbsp;(${s.start_sec.toFixed(2)}s–${s.end_sec.toFixed(2)}s)
        </span>
        ${escHtml(s.caption)}
      </div>
      <button class="btn btn-outline-danger btn-sm py-0 px-1 ms-2"
              onclick="deleteSeg(${i})">✕</button>
    </div>
  `).join('');
}
function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
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
        required=True,
        help="Directory of per-clip frame subdirectories (output of extract_all_frames.py).",
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Port to serve on (default: 5000)."
    )
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir).expanduser().resolve()
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    app.config["FRAMES_DIR"] = str(frames_dir)
    print(f"Serving frames from : {frames_dir}")
    print(f"Labels saved to     : {frames_dir / 'labels.json'}")
    print(f"Open                : http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
