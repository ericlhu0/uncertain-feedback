"""Demo-designer web server.

Usage::

    uv run python src/uncertain_feedback/demo_designer/server.py \\
        [--mpc-config src/uncertain_feedback/planners/mpc/configs/arm_mpc_cartesian_mdm_llm_transfer.yaml] \\
        [--personas-file demo_designer_personas.json] \\
        [--host 127.0.0.1] [--port 6780]
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

from uncertain_feedback.demo_designer.core import DemoSession

_STATIC_DIR = Path(__file__).parent / "static"
_ARTIFACT_ROOT = Path("demo_designer_artifacts").resolve()
_DEFAULT_CONFIG = (
    Path(__file__).parents[1]
    / "planners/mpc/configs/arm_mpc_cartesian_mdm_llm_transfer.yaml"
)

app = Flask(__name__, static_folder=str(_STATIC_DIR))
# Dev tool: never serve stale app.js against a newer server API.
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

session: DemoSession | None = None

# Serializes the heavy pipeline endpoints so the log endpoint stays responsive
# while an MPC rollout / MDM generation / cost generation runs in another
# request thread.
_heavy_lock = threading.Lock()


class _TeeBuffer:
    """stdout tee that keeps completed lines for the browser console."""

    def __init__(self, stream) -> None:
        self._stream = stream
        self._partial = ""
        self.lines: list[str] = []

    def write(self, text: str) -> None:
        self._stream.write(text)
        self._partial += text
        while "\n" in self._partial:
            line, self._partial = self._partial.split("\n", 1)
            if line.strip():
                self.lines.append(line)

    def flush(self) -> None:
        self._stream.flush()


_log_buffer = _TeeBuffer(sys.stdout)


def _run(fn):
    try:
        return jsonify(fn())
    except Exception as exc:  # surfaced to the browser as the error banner
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(exc)}), 400


def _run_heavy(fn):
    with _heavy_lock:
        return _run(fn)


@app.route("/")
def index():
    return send_from_directory(_STATIC_DIR, "index.html")


@app.route("/static/<path:filename>")
def static_files(filename: str):
    return send_from_directory(_STATIC_DIR, filename)


@app.route("/api/artifact/<path:relpath>")
def artifact_file(relpath: str):
    return send_from_directory(_ARTIFACT_ROOT, relpath)


@app.route("/api/init")
def init():
    return _run_heavy(session.init_payload)


@app.route("/api/logs")
def logs():
    since = int(request.args.get("since", 0))
    lines = _log_buffer.lines
    if since > len(lines):
        since = 0
    return jsonify({"lines": lines[since:], "next": len(lines)})


@app.route("/api/personas", methods=["POST"])
def upsert_persona():
    data = request.get_json(force=True)

    def do():
        result = session.upsert_persona(data)
        return {"personas": session.personas_payload(), **result}

    return _run(do)


@app.route("/api/personas/<name>", methods=["DELETE"])
def delete_persona(name: str):
    def do():
        session.delete_persona(name)
        return {"personas": session.personas_payload()}

    return _run(do)


@app.route("/api/preview_pose", methods=["POST"])
def preview_pose():
    data = request.get_json(force=True)
    return _run(lambda: session.preview_pose(data["arm_aa"]))


@app.route("/api/mesh/<mesh_id>")
def mesh(mesh_id: str):
    try:
        vertices = session.mesh_vertices(mesh_id)
    except KeyError as exc:
        return jsonify({"error": exc.args[0]}), 404
    response = Response(vertices.tobytes(), mimetype="application/octet-stream")
    response.headers["X-Mesh-Frames"] = str(vertices.shape[0])
    response.headers["X-Mesh-Vertices"] = str(vertices.shape[1])
    response.headers["X-Mesh-Dtype"] = "float32-le"
    return response


@app.route("/api/base_rollout", methods=["POST"])
def base_rollout():
    data = request.get_json(force=True)
    return _run_heavy(
        lambda: session.run_base(
            data["arm_aa"],
            data["goal"],
            data["persona"],
            bool(data.get("show_oracle", False)),
        )
    )


@app.route("/api/manual_trajectory/start", methods=["POST"])
def start_manual_trajectory():
    data = request.get_json(force=True)
    return _run_heavy(
        lambda: session.start_manual_trajectory(
            data["arm_aa"],
            data["goal"],
            data["persona"],
        )
    )


@app.route("/api/manual_trajectory/exit", methods=["POST"])
def exit_manual_trajectory():
    return _run_heavy(session.exit_manual_trajectory)


@app.route("/api/oracle_rollout", methods=["POST"])
def oracle_rollout():
    return _run_heavy(lambda: session.run_oracle(from_trigger=True))


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True)
    return _run_heavy(
        lambda: session.generate(
            data["prompt"],
            int(data["n_samples"]),
            int(data["n_clusters"]),
            float(data["scale"]),
        )
    )


@app.route("/api/recluster", methods=["POST"])
def recluster():
    data = request.get_json(force=True)
    return _run_heavy(
        lambda: session.recluster(int(data["n_clusters"]), float(data["scale"]))
    )


@app.route("/api/pick_cluster", methods=["POST"])
def pick_cluster():
    data = request.get_json(force=True)
    return _run(lambda: session.pick_cluster(int(data["label"])))


@app.route("/api/refine_cluster", methods=["POST"])
def refine_cluster():
    data = request.get_json(force=True)
    return _run_heavy(
        lambda: session.refine_cluster(
            int(data["label"]),
            int(data["n_clusters"]),
            float(data["scale"]),
        )
    )


@app.route("/api/back_cluster", methods=["POST"])
def back_cluster():
    return _run_heavy(session.back_cluster)


@app.route("/api/generate_cost", methods=["POST"])
def generate_cost():
    data = request.get_json(force=True)
    return _run_heavy(lambda: session.generate_cost(data["backend"]))


@app.route("/api/commit_round", methods=["POST"])
def commit_round():
    return _run(session.commit_round)


@app.route("/api/apply_round", methods=["POST"])
def apply_round():
    return _run_heavy(session.apply_round_and_continue)


@app.route("/api/manual_trajectory/ignore_violation", methods=["POST"])
def ignore_comfort_violation():
    return _run_heavy(session.ignore_comfort_violation)


@app.route("/api/rounds/<int:index>", methods=["DELETE"])
def remove_round(index: int):
    return _run(lambda: session.remove_round(index))


@app.route("/api/combine_rounds", methods=["POST"])
def combine_rounds():
    return _run_heavy(session.combine_rounds)


@app.route("/api/reset_rounds", methods=["POST"])
def reset_rounds():
    return _run(session.reset_rounds)


def main() -> None:
    global session
    parser = argparse.ArgumentParser(description="Demo-designer web server")
    parser.add_argument("--mpc-config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument(
        "--personas-file", type=Path, default=Path("demo_designer_personas.json")
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6780)
    args = parser.parse_args()

    sys.stdout = _log_buffer
    session = DemoSession(args.mpc_config, args.personas_file)
    print(f"Serving demo designer at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
