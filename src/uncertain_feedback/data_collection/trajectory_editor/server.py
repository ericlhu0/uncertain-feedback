"""Trajectory editor web server.

Usage::

    uv run python -m uncertain_feedback.data_collection.trajectory_editor.server \\
        --hml_stats_dir /path/to/HumanML3D \\
        [--host 127.0.0.1] [--port 6769]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import spacy
from flask import Flask, jsonify, request, send_from_directory
from scipy.interpolate import CubicSpline

from uncertain_feedback.data_collection.build_mdm_dataset import (
    _pos_tag,
    _write_text_file,
)
from uncertain_feedback.data_collection.smpl_to_hml263 import (
    load_hml_stats,
    positions_to_hml263,
)
from uncertain_feedback.data_collection.trajectory_editor.hml_decode import (
    EDITABLE_JOINTS,
    FIXED_JOINTS,
    demo_pt_to_positions,
    get_tpose_bone_lengths,
)
from uncertain_feedback.planners.mpc.kinematics import SMPL_BONE_PAIRS_22, SMPL_PARENTS_22

_MDM_MIN_FRAMES = 40
_MDM_MAX_FRAMES = 196

_STATIC_DIR = Path(__file__).parent / "static"

app = Flask(__name__, static_folder=str(_STATIC_DIR))

# Populated by main() from CLI args
_defaults: dict = {}


@app.route("/api/config")
def config():
    return jsonify(_defaults)


@app.route("/")
def index():
    return send_from_directory(_STATIC_DIR, "index.html")


@app.route("/static/<path:filename>")
def static_files(filename: str):
    return send_from_directory(_STATIC_DIR, filename)


@app.route("/api/base-pose")
def base_pose():
    pt_path = request.args.get("pt_path", "").strip()
    hml_stats_dir = request.args.get("hml_stats_dir", "").strip()

    if not pt_path:
        return jsonify({"error": "pt_path is required"}), 400
    if not hml_stats_dir:
        return jsonify({"error": "hml_stats_dir is required"}), 400

    pt_path = Path(pt_path)
    hml_stats_dir = Path(hml_stats_dir)

    if not pt_path.exists():
        return jsonify({"error": f"demo.pt not found: {pt_path}"}), 404
    if not hml_stats_dir.exists():
        return jsonify({"error": f"hml_stats_dir not found: {hml_stats_dir}"}), 404

    try:
        positions = demo_pt_to_positions(pt_path, hml_stats_dir)
        bone_lengths = get_tpose_bone_lengths()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({
        "joints": positions.tolist(),
        "editable": EDITABLE_JOINTS,
        "fixed": FIXED_JOINTS,
        "bone_pairs": [[int(p), int(c)] for p, c in SMPL_BONE_PAIRS_22],
        "parents": SMPL_PARENTS_22,
        "bone_lengths": bone_lengths,
    })


@app.route("/api/load-trajectory")
def load_trajectory():
    npy_path = request.args.get("npy_path", "").strip()
    if not npy_path:
        return jsonify({"error": "npy_path is required"}), 400

    npy_path = Path(npy_path)
    if not npy_path.exists():
        return jsonify({"error": f"File not found: {npy_path}"}), 404

    try:
        hml263 = np.load(str(npy_path)).astype(np.float64)
    except Exception as exc:
        return jsonify({"error": f"Failed to load .npy: {exc}"}), 500

    if hml263.ndim != 2 or hml263.shape[1] != 263:
        return jsonify({"error": f"Expected shape (N, 263), got {list(hml263.shape)}"}), 400

    N = hml263.shape[0]
    positions = np.zeros((N, 22, 3), dtype=np.float64)
    for t in range(N):
        raw = hml263[t]
        positions[t, 0] = [0.0, float(raw[3]), 0.0]
        positions[t, 1:] = raw[4:67].reshape(21, 3)

    return jsonify({"positions": positions.tolist(), "num_frames": N})


@app.route("/api/export", methods=["POST"])
def export_trajectory():
    data = request.get_json(force=True)

    # ── Validate required fields ──────────────────────────────────────────────
    required = ["keyframes", "caption", "output_dir", "hml_stats_dir", "total_frames", "pt_path"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    keyframes: list[dict] = data["keyframes"]
    caption: str = data["caption"].strip()
    output_dir = Path(data["output_dir"])
    hml_stats_dir = Path(data["hml_stats_dir"])
    total_frames: int = int(data["total_frames"])
    pt_path = Path(data["pt_path"])
    n_augment: int = int(data.get("n_augment", 0))
    noise_std: float = float(data.get("noise_std", 0.05))

    if not caption:
        return jsonify({"error": "Caption must not be empty"}), 400
    if total_frames < _MDM_MIN_FRAMES or total_frames > _MDM_MAX_FRAMES:
        return jsonify({"error": f"total_frames must be {_MDM_MIN_FRAMES}–{_MDM_MAX_FRAMES}"}), 400
    if len(keyframes) < 1:
        return jsonify({"error": "Need at least 1 keyframe"}), 400
    if n_augment < 0:
        return jsonify({"error": "n_augment must be >= 0"}), 400
    if noise_std < 0:
        return jsonify({"error": "noise_std must be >= 0"}), 400

    # Sort keyframes by frame number
    keyframes = sorted(keyframes, key=lambda k: k["frame"])
    kf_frames = np.array([k["frame"] for k in keyframes], dtype=float)
    kf_positions = np.array([k["positions"] for k in keyframes], dtype=np.float64)  # (K, 22, 3)

    # ── Load stats and base pose ──────────────────────────────────────────────
    try:
        mean, std = load_hml_stats(hml_stats_dir)
        base_positions = demo_pt_to_positions(pt_path, hml_stats_dir)  # (22, 3)
    except Exception as exc:
        return jsonify({"error": f"Failed to load stats/base pose: {exc}"}), 500

    # ── Interpolate keyframes → full trajectory ────────────────────────────────
    all_frames = np.arange(total_frames, dtype=float)
    trajectory = np.zeros((total_frames, 22, 3), dtype=np.float64)

    if len(keyframes) == 1:
        # Single keyframe: tile it
        trajectory[:] = kf_positions[0]
    else:
        for j in range(22):
            for d in range(3):
                cs = CubicSpline(kf_frames, kf_positions[:, j, d])
                trajectory[:, j, d] = cs(all_frames)

    # Override fixed joints with the base pose values across all frames
    for j in FIXED_JOINTS:
        trajectory[:, j, :] = base_positions[j].astype(np.float64)

    # ── Convert to HML263 ─────────────────────────────────────────────────────
    try:
        hml263 = positions_to_hml263(
            trajectory.astype(np.float64),
            mean=mean,
            std=std,
            normalize=False,
        )  # (N, 263) unnormalized float32
    except Exception as exc:
        return jsonify({"error": f"HML263 conversion failed: {exc}"}), 500

    # ── Write MDM dataset layout ──────────────────────────────────────────────
    vecs_dir = output_dir / "new_joint_vecs"
    texts_dir = output_dir / "texts"
    vecs_dir.mkdir(parents=True, exist_ok=True)
    texts_dir.mkdir(parents=True, exist_ok=True)

    # Find next available ID
    existing = sorted(vecs_dir.glob("*.npy"))
    next_int = len(existing) + 1
    base_motion_id = f"{next_int:06d}"

    np.save(vecs_dir / f"{base_motion_id}.npy", hml263)

    nlp = spacy.load("en_core_web_sm")
    _write_text_file(texts_dir / f"{base_motion_id}.txt", [caption], nlp)

    all_ids = [base_motion_id]

    if n_augment > 0:
        aug_rng = np.random.default_rng(next_int)
        for _ in range(n_augment):
            next_int += 1
            aug_id = f"{next_int:06d}"
            hml263_aug = hml263.copy()
            noise_norm = aug_rng.standard_normal(hml263.shape).astype(np.float32)
            hml263_aug += noise_norm * noise_std * std
            np.save(vecs_dir / f"{aug_id}.npy", hml263_aug)
            _write_text_file(texts_dir / f"{aug_id}.txt", [caption], nlp)
            all_ids.append(aug_id)

    # Update split files (append to train)
    for split_file in ["train.txt", "val.txt", "test.txt"]:
        split_path = output_dir / split_file
        if not split_path.exists():
            split_path.write_text("", encoding="utf-8")

    with open(output_dir / "train.txt", "a", encoding="utf-8") as f:
        for mid in all_ids:
            f.write(f"{mid}\n")

    # Copy Mean.npy and Std.npy if not already present
    for stat_file in ["Mean.npy", "Std.npy"]:
        dst = output_dir / stat_file
        src = hml_stats_dir / stat_file
        if not dst.exists() and src.exists():
            shutil.copy2(src, dst)

    return jsonify({
        "ok": True,
        "motion_id": base_motion_id,
        "motion_ids": all_ids,
        "shape": list(hml263.shape),
        "output_dir": str(output_dir),
        "n_augment": n_augment,
    })


def main():
    parser = argparse.ArgumentParser(description="Trajectory editor web server")
    parser.add_argument("--hml_stats_dir", type=Path, required=True,
                        help="Directory containing Mean.npy and Std.npy")
    parser.add_argument("--pt_path", type=Path, default=None,
                        help="Default demo.pt path to pre-fill in the editor")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Default export output directory to pre-fill in the editor")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6769)
    args = parser.parse_args()

    if not args.hml_stats_dir.exists():
        raise SystemExit(f"hml_stats_dir not found: {args.hml_stats_dir}")

    _defaults["hml_stats_dir"] = str(args.hml_stats_dir)
    _defaults["pt_path"] = str(args.pt_path) if args.pt_path else ""
    _defaults["output_dir"] = str(args.output_dir) if args.output_dir else ""

    print(f"Serving editor at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
