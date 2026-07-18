"""Demo-runner web server.

Serves a presentation-oriented frontend against the demo-designer backend: the
same routes (via :func:`create_app`) plus session replay. Run from the repo root
-- the artifact root is resolved against the CWD.

Usage::

    uv run python src/uncertain_feedback/demo_runner/server.py \\
        [--mpc-config src/uncertain_feedback/planners/mpc/configs/arm_mpc_cartesian_mdm_llm_transfer.yaml] \\
        [--personas-file demo_designer_personas.json] \\
        [--trajectory-configs-file demo_designer_trajectory_configs.json] \\
        [--host 127.0.0.1] [--port 6781]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from flask import request

from uncertain_feedback.demo_designer import server as designer_server
from uncertain_feedback.demo_designer.core import persona_from_json
from uncertain_feedback.demo_designer.server import (
    _ARTIFACT_ROOT,
    _run,
    boot,
    create_app,
)

_STATIC_DIR = Path(__file__).parent / "static"
_DEFAULT_CONFIG = (
    Path(__file__).parents[1]
    / "planners/mpc/configs/arm_mpc_cartesian_mdm_llm_transfer.yaml"
)

app = create_app(_STATIC_DIR)


@app.route("/api/live_trajectory/start", methods=["POST"])
def start_live_trajectory() -> Any:
    data = request.get_json(force=True)

    def do() -> dict[str, Any]:
        session = designer_server._require_session()
        session.start_trajectory(data["arm_aa"], data["goal"], advance=False)
        return session._trajectory_payload(current_mesh_only=True)

    return designer_server._run_heavy(do)


@app.route("/api/live_trajectory/step", methods=["POST"])
def step_live_trajectory() -> Any:
    return designer_server._run_heavy(
        lambda: designer_server._require_session().advance_trajectory(
            current_mesh_only=True
        )
    )


@app.route("/api/live_trajectory/apply_round", methods=["POST"])
def apply_round_live_trajectory() -> Any:
    return designer_server._run_heavy(
        lambda: designer_server._require_session().apply_round_and_continue(
            advance=False, current_mesh_only=True
        )
    )


def _remint_meshes(node: Any) -> None:
    """Re-register recorded poses so a replayed beat gets live mesh ids.

    Recorded ``mesh_id``s are dead: the cache is per-process and only 64 entries
    deep. Registration is lazy (vertices are generated on fetch), and one beat
    mints far fewer ids than the cache holds, so a beat never self-evicts.
    """
    rig = designer_server.rig
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


@app.route("/api/replay/<name>/fork", methods=["POST"])
def fork_replay(name: str):
    return designer_server._run_heavy(
        lambda: designer_server.rig.fork_session(name).payload()
    )


@app.route("/api/replay/<name>")
def replay_index(name: str):
    return _run(
        lambda: json.loads((_replay_dir(name) / "index.json").read_text("utf-8"))
    )


@app.route("/api/replay/<name>/<int:index>")
def replay_beat(name: str, index: int):
    def do() -> dict[str, Any]:
        beats = json.loads((_replay_dir(name) / "index.json").read_text("utf-8"))[
            "beats"
        ]
        beat = json.loads((_replay_dir(name) / beats[index]["file"]).read_text("utf-8"))
        _remint_meshes(beat["data"])
        beat["persona"]["feature_box_ranges"] = designer_server.rig._feature_box_ranges(
            persona_from_json(beat["persona"])
        )
        return beat

    return _run(do)


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo-runner web server")
    parser.add_argument("--mpc-config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument(
        "--personas-file", type=Path, default=Path("demo_designer_personas.json")
    )
    parser.add_argument(
        "--trajectory-configs-file",
        type=Path,
        default=Path("demo_designer_trajectory_configs.json"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6781)
    args = parser.parse_args()

    boot(args.mpc_config, args.personas_file, args.trajectory_configs_file)
    print(f"Serving demo runner at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
