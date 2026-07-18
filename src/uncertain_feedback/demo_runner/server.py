"""Demo-runner web server.

Run from the repo root -- the artifact root is resolved against the CWD.

Usage::

    uv run python src/uncertain_feedback/demo_runner/server.py \\
        [--mpc-config src/uncertain_feedback/planners/mpc/configs/arm_mpc_cartesian_mdm_llm_transfer.yaml] \\
        [--personas-file demo_runner_personas.json] \\
        [--trajectory-configs-file demo_runner_trajectory_configs.json] \\
        [--host 127.0.0.1] [--port 6781]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory

from uncertain_feedback.demo_runner.core import DemoRig, persona_from_json

if TYPE_CHECKING:
    from uncertain_feedback.demo_runner.session import Session, Trajectory

_STATIC_DIR = Path(__file__).parent / "static"
_ARTIFACT_ROOT = Path("demo_runner_artifacts").resolve()
_DEFAULT_CONFIG = (
    Path(__file__).parents[1]
    / "planners/mpc/configs/arm_mpc_cartesian_mdm_llm_transfer.yaml"
)

rig: DemoRig | None = None

# Serializes the heavy pipeline endpoints so the log endpoint stays responsive
# while an MPC rollout / MDM generation / cost generation runs in another
# request thread.
_heavy_lock = threading.Lock()


def _require_session() -> "Session":
    if rig.session is None:
        raise ValueError("No active session. Start or resume a session first.")
    return rig.session


def _require_trajectory() -> tuple["Session", "Trajectory"]:
    session = _require_session()
    trajectory = session.trajectory
    if trajectory is None:
        raise ValueError("No active trajectory. Start a trajectory first.")
    return session, trajectory


class _TeeBuffer:
    """Stdout tee that keeps completed lines and streamable raw text."""

    def __init__(self, stream) -> None:
        self._stream = stream
        self._partial = ""
        self.text = ""
        self.lines: list[str] = []

    def write(self, text: str) -> None:
        self._stream.write(text)
        self.text += text
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


def create_app(static_dir: Path) -> Flask:
    """Build the Flask app: pipeline routes plus runner-only live/replay routes."""
    app = Flask(__name__, static_folder=str(static_dir))
    # Dev tool: never serve stale app.js against a newer server API.
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    @app.route("/")
    def index():
        return send_from_directory(static_dir, "index.html")

    @app.route("/static/<path:filename>")
    def static_files(filename: str):
        return send_from_directory(static_dir, filename)

    @app.route("/api/artifact/<path:relpath>")
    def artifact_file(relpath: str):
        return send_from_directory(_ARTIFACT_ROOT, relpath)

    @app.route("/api/init")
    def init():
        return _run_heavy(rig.init_payload)

    @app.route("/api/logs")
    def logs():
        since = int(request.args.get("since", 0))
        lines = _log_buffer.lines
        if since > len(lines):
            since = 0
        payload = {"lines": lines[since:], "next": len(lines)}
        if "char_since" in request.args:
            char_since = int(request.args["char_since"])
            if char_since > len(_log_buffer.text):
                char_since = 0
            payload.update(
                text=_log_buffer.text[char_since:],
                next_char=len(_log_buffer.text),
            )
        return jsonify(payload)

    @app.route("/api/personas", methods=["POST"])
    def upsert_persona():
        data = request.get_json(force=True)

        def do():
            result = rig.upsert_persona(data)
            return {"personas": rig.personas_payload(), **result}

        return _run(do)

    @app.route("/api/personas/<name>", methods=["DELETE"])
    def delete_persona(name: str):
        def do():
            rig.delete_persona(name)
            return {"personas": rig.personas_payload()}

        return _run(do)

    @app.route("/api/preview_pose", methods=["POST"])
    def preview_pose():
        data = request.get_json(force=True)
        return _run(lambda: rig.preview_pose(data["arm_aa"]))

    @app.route("/api/trajectory-configs/<kind>", methods=["POST"])
    def upsert_trajectory_config(kind: str):
        data = request.get_json(force=True)
        return _run(
            lambda: {"trajectory_configs": rig.upsert_trajectory_config(kind, data)}
        )

    @app.route("/api/mesh/<mesh_id>")
    def mesh(mesh_id: str):
        try:
            vertices = rig.mesh_vertices(mesh_id, request.args.get("frame", type=int))
        except KeyError as exc:
            return jsonify({"error": exc.args[0]}), 404
        response = Response(vertices.tobytes(), mimetype="application/octet-stream")
        response.headers["X-Mesh-Frames"] = str(vertices.shape[0])
        response.headers["X-Mesh-Vertices"] = str(vertices.shape[1])
        response.headers["X-Mesh-Dtype"] = "float32-le"
        return response

    @app.route("/api/session/start", methods=["POST"])
    def start_session():
        data = request.get_json(force=True)
        return _run_heavy(lambda: rig.begin_session(data["persona"]).payload())

    @app.route("/api/sessions")
    def list_sessions():
        return _run(rig.list_sessions)

    @app.route("/api/sessions/<name>", methods=["DELETE"])
    def delete_session(name: str):
        return _run(lambda: rig.delete_session(name))

    @app.route("/api/session/resume", methods=["POST"])
    def resume_session():
        data = request.get_json(force=True)
        return _run_heavy(lambda: rig.resume_session(data["dir"]).payload())

    @app.route("/api/manual_trajectory/start", methods=["POST"])
    def start_manual_trajectory():
        data = request.get_json(force=True)

        def do():
            session = _require_session()
            session.start_trajectory(data["arm_aa"], data["goal"])
            return session._trajectory_payload()

        return _run_heavy(do)

    @app.route("/api/manual_trajectory/exit", methods=["POST"])
    def exit_manual_trajectory():
        def do():
            session, _ = _require_trajectory()
            return session.exit_trajectory()

        return _run_heavy(do)

    @app.route("/api/oracle_rollout", methods=["POST"])
    def oracle_rollout():
        def do():
            session, _ = _require_trajectory()
            return session.run_oracle(from_trigger=True)

        return _run_heavy(do)

    @app.route("/api/generate", methods=["POST"])
    def generate():
        data = request.get_json(force=True)

        def do():
            session, _ = _require_trajectory()
            return session.generate(
                data["prompt"],
                int(data["n_samples"]),
                int(data["n_clusters"]),
                float(data["scale"]),
                str(data["clusterer"]),
            )

        return _run_heavy(do)

    @app.route("/api/recluster", methods=["POST"])
    def recluster():
        data = request.get_json(force=True)

        def do():
            session, _ = _require_trajectory()
            return session.recluster(
                int(data["n_clusters"]), float(data["scale"]), str(data["clusterer"])
            )

        return _run_heavy(do)

    @app.route("/api/rescale", methods=["POST"])
    def rescale():
        data = request.get_json(force=True)

        def do():
            session, _ = _require_trajectory()
            return session.rescale(float(data["scale"]))

        return _run_heavy(do)

    @app.route("/api/pick_cluster", methods=["POST"])
    def pick_cluster():
        data = request.get_json(force=True)

        def do():
            session, _ = _require_trajectory()
            return session.pick_cluster(int(data["label"]))

        return _run(do)

    @app.route("/api/mark_cluster", methods=["POST"])
    def mark_cluster():
        data = request.get_json(force=True)

        def do():
            session, _ = _require_trajectory()
            return session.mark_cluster(int(data["label"]), bool(data["undesirable"]))

        return _run(do)

    @app.route("/api/refine_cluster", methods=["POST"])
    def refine_cluster():
        data = request.get_json(force=True)

        def do():
            session, _ = _require_trajectory()
            return session.refine_cluster(
                int(data["label"]),
                int(data["n_clusters"]),
                float(data["scale"]),
                str(data["clusterer"]),
            )

        return _run_heavy(do)

    @app.route("/api/back_cluster", methods=["POST"])
    def back_cluster():
        def do():
            session, _ = _require_trajectory()
            return session.back_cluster()

        return _run_heavy(do)

    @app.route("/api/generate_cost", methods=["POST"])
    def generate_cost():
        data = request.get_json(force=True)

        def do():
            session, _ = _require_trajectory()
            return session.generate_cost(data["backend"])

        return _run_heavy(do)

    @app.route("/api/commit_round", methods=["POST"])
    def commit_round():
        def do():
            session, _ = _require_trajectory()
            return session.commit_round()

        return _run(do)

    @app.route("/api/apply_round", methods=["POST"])
    def apply_round():
        def do():
            session, _ = _require_trajectory()
            return session.apply_round_and_continue()

        return _run_heavy(do)

    @app.route("/api/manual_trajectory/ignore_violation", methods=["POST"])
    def ignore_comfort_violation():
        def do():
            session, _ = _require_trajectory()
            return session.ignore_comfort_violation()

        return _run_heavy(do)

    @app.route("/api/rounds/<int:index>", methods=["DELETE"])
    def remove_round(index: int):
        return _run(lambda: _require_session().remove_round(index))

    # Codex-driven combination runs for minutes; a synchronous request outlives
    # the browser's connection ("Failed to fetch") even though the server finishes.
    # Run it in the background and let the frontend poll for the result.
    combine_job: dict[str, Any] = {"thread": None, "result": None, "error": None}

    @app.route("/api/combine_rounds", methods=["POST"])
    def combine_rounds():
        thread = combine_job["thread"]
        if thread is not None and thread.is_alive():
            return jsonify({"status": "running"})
        session = _require_session()
        combine_job.update(result=None, error=None)

        def work() -> None:
            try:
                with _heavy_lock:
                    combine_job["result"] = session.combine_rounds()
            except Exception as exc:  # surfaced to the browser via /status
                import traceback

                traceback.print_exc()
                combine_job["error"] = str(exc)

        thread = threading.Thread(target=work, daemon=True)
        combine_job["thread"] = thread
        thread.start()
        return jsonify({"status": "started"})

    @app.route("/api/combine_rounds/status")
    def combine_rounds_status():
        thread = combine_job["thread"]
        if thread is not None and thread.is_alive():
            return jsonify({"status": "running"})
        if combine_job["error"] is not None:
            return jsonify({"status": "error", "error": combine_job["error"]})
        if combine_job["result"] is not None:
            return jsonify({"status": "done", "result": combine_job["result"]})
        return jsonify({"status": "idle"})

    @app.route("/api/reset_rounds", methods=["POST"])
    def reset_rounds():
        return _run(lambda: _require_session().reset_rounds())

    @app.route("/api/corpus/<int:index>", methods=["DELETE"])
    def remove_corpus_entry(index: int):
        return _run(lambda: _require_session().remove_corpus_entry(index))

    @app.route("/api/live_trajectory/start", methods=["POST"])
    def start_live_trajectory() -> Any:
        data = request.get_json(force=True)

        def do() -> dict[str, Any]:
            session = _require_session()
            session.start_trajectory(data["arm_aa"], data["goal"], advance=False)
            return session._trajectory_payload(current_mesh_only=True)

        return _run_heavy(do)

    @app.route("/api/live_trajectory/step", methods=["POST"])
    def step_live_trajectory() -> Any:
        return _run_heavy(
            lambda: _require_session().advance_trajectory(current_mesh_only=True)
        )

    @app.route("/api/live_trajectory/apply_round", methods=["POST"])
    def apply_round_live_trajectory() -> Any:
        return _run_heavy(
            lambda: _require_session().apply_round_and_continue(
                advance=False, current_mesh_only=True
            )
        )

    @app.route("/api/replay/<name>/fork", methods=["POST"])
    def fork_replay(name: str):
        return _run_heavy(lambda: rig.fork_session(name).payload())

    @app.route("/api/replay/<name>")
    def replay_index(name: str):
        return _run(
            lambda: json.loads((_replay_dir(name) / "index.json").read_text("utf-8"))
        )

    @app.route("/api/replay/<name>/<int:index>")
    def replay_beat(name: str, index: int):
        def do() -> dict[str, Any]:
            beats = json.loads(
                (_replay_dir(name) / "index.json").read_text("utf-8")
            )["beats"]
            beat = json.loads(
                (_replay_dir(name) / beats[index]["file"]).read_text("utf-8")
            )
            _remint_meshes(beat["data"])
            beat["persona"]["feature_box_ranges"] = rig._feature_box_ranges(
                persona_from_json(beat["persona"])
            )
            return beat

        return _run(do)

    return app


def _remint_meshes(node: Any) -> None:
    """Re-register recorded poses so a replayed beat gets live mesh ids.

    Recorded ``mesh_id``s are dead: the cache is per-process and only 64 entries
    deep. Registration is lazy (vertices are generated on fetch), and one beat
    mints far fewer ids than the cache holds, so a beat never self-evicts.
    """
    if isinstance(node, dict):
        if "mesh_id" in node and "arm_positions" in node:
            node["mesh_id"] = rig.meshes.register(
                np.asarray(node["arm_positions"], dtype=np.float64)
            )
        for value in node.values():
            _remint_meshes(value)
    elif isinstance(node, list):
        for value in node:
            _remint_meshes(value)


def _replay_dir(name: str) -> Path:
    return _ARTIFACT_ROOT / name / "replay"


def boot(
    mpc_config: Path, personas_file: Path, trajectory_configs_file: Path
) -> None:
    """Install the stdout tee and construct the process-wide rig."""
    global rig
    sys.stdout = _log_buffer
    rig = DemoRig(mpc_config, personas_file, trajectory_configs_file)


app = create_app(_STATIC_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo-runner web server")
    parser.add_argument("--mpc-config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument(
        "--personas-file", type=Path, default=Path("demo_runner_personas.json")
    )
    parser.add_argument(
        "--trajectory-configs-file",
        type=Path,
        default=Path("demo_runner_trajectory_configs.json"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6781)
    args = parser.parse_args()

    boot(args.mpc_config, args.personas_file, args.trajectory_configs_file)
    print(f"Serving demo runner at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
