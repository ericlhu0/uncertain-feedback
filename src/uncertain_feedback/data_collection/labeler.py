"""Flask web app for labeling motion videos with text descriptions.

Run::

    uv run python src/uncertain_feedback/data_collection/labeler.py \\
        --videos_dir ./recordings/ \\
        [--port 5000]

Then open http://localhost:5000 in a browser (SSH-tunnel from a remote server with
``ssh -L 5000:localhost:5000 user@host``).  Labels are persisted to
``<videos_dir>/labels.json``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import flask
from flask import Flask, Response, jsonify, render_template_string, request

_VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv", ".webm", ".m4v"}

app = Flask(__name__)

# ---------------------------------------------------------------------------
# HTML templates
# ---------------------------------------------------------------------------

_INDEX_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Motion Labeler</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
<div class="container py-4" style="max-width:700px">
  <div class="d-flex justify-content-between align-items-center mb-3">
    <h4 class="mb-0">Motion Labeler</h4>
    <span class="text-muted small">{{ labeled }}/{{ total }} labeled</span>
  </div>
  <div class="progress mb-3" style="height:6px">
    <div class="progress-bar bg-success" style="width:{{ (labeled/total*100)|int if total else 0 }}%"></div>
  </div>
  <ul class="list-group">
  {% for video in videos %}
    <li class="list-group-item d-flex justify-content-between align-items-center">
      <a href="/label/{{ video }}" class="text-decoration-none text-dark">{{ video }}</a>
      {% if labels.get(video) %}
        <span class="badge bg-success rounded-pill">{{ labels[video]|length }} label(s)</span>
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
  <title>Label: {{ filename }}</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
<div class="container py-4" style="max-width:800px">
  <div class="d-flex justify-content-between align-items-center mb-3">
    <div>
      <h5 class="mb-0">{{ filename }}</h5>
      <small class="text-muted">{{ idx }} / {{ total }}</small>
    </div>
    <div class="btn-group btn-group-sm">
      {% if prev %}<a href="/label/{{ prev }}" class="btn btn-outline-secondary">&#8592; Prev</a>{% endif %}
      <a href="/" class="btn btn-outline-secondary">List</a>
      {% if next %}<a href="/label/{{ next }}" class="btn btn-outline-secondary">Next &#8594;</a>{% endif %}
    </div>
  </div>

  <video controls class="w-100 rounded mb-3" style="max-height:480px;background:#000"
         src="/video/{{ filename }}"></video>

  <div class="card">
    <div class="card-body">
      <label class="form-label fw-semibold">
        Labels <span class="text-muted fw-normal small">(one per line)</span>
      </label>
      <textarea id="labels-input" class="form-control font-monospace" rows="4"
                placeholder="a person sits down slowly">{{ existing_labels }}</textarea>
      <div class="d-flex align-items-center mt-2 gap-2">
        <button onclick="save(false)" class="btn btn-primary btn-sm">Save</button>
        {% if next %}
        <button onclick="save(true)" class="btn btn-success btn-sm">Save &amp; Next &#8594;</button>
        {% endif %}
        <span id="status" class="text-muted small ms-1"></span>
      </div>
    </div>
  </div>
</div>
<script>
function save(goNext) {
  const raw = document.getElementById('labels-input').value;
  const labels = raw.split('\\n').map(s => s.trim()).filter(s => s.length > 0);
  const status = document.getElementById('status');
  status.textContent = 'Saving…';
  fetch('/label/{{ filename }}', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({labels})
  })
  .then(r => r.json())
  .then(d => {
    if (d.ok) {
      status.textContent = 'Saved ✓';
      {% if next %}if (goNext) { window.location.href = '/label/{{ next }}'; }{% endif %}
    } else {
      status.textContent = 'Error saving';
    }
  })
  .catch(() => { status.textContent = 'Error saving'; });
}
document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { save(false); }
});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _labels_path() -> Path:
    return Path(flask.current_app.config["VIDEOS_DIR"]) / "labels.json"


def _load_labels() -> dict[str, list[str]]:
    p = _labels_path()
    if p.exists():
        with open(p, encoding="utf-8") as f:
            data: dict[str, list[str]] = json.load(f)
            return data
    return {}


def _save_labels_to_disk(data: dict[str, list[str]]) -> None:
    with open(_labels_path(), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _list_videos(videos_dir: Path) -> list[str]:
    return sorted(
        name
        for name in os.listdir(videos_dir)
        if Path(name).suffix.lower() in _VIDEO_EXTENSIONS
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index() -> str:
    """Render the video list with labeled/unlabeled status."""
    videos_dir = Path(flask.current_app.config["VIDEOS_DIR"])
    videos = _list_videos(videos_dir)
    labels = _load_labels()
    labeled = sum(1 for v in videos if labels.get(v))
    return render_template_string(
        _INDEX_TEMPLATE,
        videos=videos,
        labels=labels,
        labeled=labeled,
        total=len(videos),
    )


@app.route("/video/<path:filename>")
def serve_video(filename: str) -> Response:
    """Serve a video file with support for HTTP Range requests (seek support)."""
    videos_dir: str = flask.current_app.config["VIDEOS_DIR"]
    return flask.send_from_directory(  # type: ignore[return-value]
        videos_dir, filename, conditional=True
    )


@app.route("/label/<path:filename>", methods=["GET"])
def label_page(filename: str) -> str:
    """Render the labeling page for a single video."""
    videos_dir = Path(flask.current_app.config["VIDEOS_DIR"])
    videos = _list_videos(videos_dir)
    labels = _load_labels()
    idx = videos.index(filename) if filename in videos else 0
    return render_template_string(
        _LABEL_TEMPLATE,
        filename=filename,
        existing_labels="\n".join(labels.get(filename, [])),
        prev=videos[idx - 1] if idx > 0 else None,
        next=videos[idx + 1] if idx < len(videos) - 1 else None,
        idx=idx + 1,
        total=len(videos),
    )


@app.route("/label/<path:filename>", methods=["POST"])
def save_labels(filename: str) -> Response:
    """Persist labels for a video to labels.json."""
    payload: Any = request.get_json(silent=True)
    new_labels: list[str] = (
        payload.get("labels", []) if isinstance(payload, dict) else []
    )
    all_labels = _load_labels()
    all_labels[filename] = new_labels
    _save_labels_to_disk(all_labels)
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and start the labeling server."""
    parser = argparse.ArgumentParser(
        description="Web interface for labeling motion videos."
    )
    parser.add_argument(
        "--videos_dir", required=True, help="Directory containing video files."
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Port to serve on (default: 5000)."
    )
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir).expanduser().resolve()
    if not videos_dir.is_dir():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")

    app.config["VIDEOS_DIR"] = str(videos_dir)
    print(f"Serving videos from : {videos_dir}")
    print(f"Labels saved to     : {videos_dir / 'labels.json'}")
    print(f"Open                : http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
