"""Standalone kimodo inference worker — runs inside the isolated ``kimodo``
conda env (pydantic>=2, transformers==5.1.0), invoked by
:class:`KimodoMotionGenerator` via ``conda run``.

Generates motion from text, optionally constrains frame 0 to a start pose,
and writes SMPL ``body_pose`` axis-angles to an ``.npz``.

NOTE — integration points to confirm on the first real run (the kimodo Python
API for these is not fully documented):
  (1) the model ``__call__`` keyword names / how ``num_samples`` batches output;
  (2) the output key holding local joint rotations and its array shape;
  (3) ``get_amass_parameters`` signature + how its ``pose_body`` maps to the 21
      SMPL body joints (expected ``(T, 63)`` → ``(T, 21, 3)``);
"""

from __future__ import annotations

import argparse
from functools import partial

import numpy as np
from tqdm.auto import tqdm


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="kimodo inference worker")
    p.add_argument("--text", required=True)
    p.add_argument("--num_frames", type=int, required=True)
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--model_name", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--start_positions_path", default=None)
    p.add_argument("--num_denoising_steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=10)
    return p.parse_args()


_N_SMPL_BODY = 21  # SMPL body joints (1..21); first 63 cols of kimodo pose_body


class _DedupTextEncoder:
    """Encode identical prompts once.

    Every UQ sample shares one prompt, but kimodo's LLM2Vec encoder runs an 8B
    LLM forward pass per list entry (internal ``batch_size=1``), so encoding
    ``num_samples`` copies costs ``num_samples`` serial passes — on CPU when the
    GPU is too small to hold the encoder, this dominates runtime. Encoding each
    unique string once and scattering the rows back is bit-identical (the
    per-item pass is unchanged) while making encode cost independent of
    ``num_samples``.
    """

    def __init__(self, encoder):  # type: ignore[no-untyped-def]
        self._encoder = encoder

    def __call__(self, texts):  # type: ignore[no-untyped-def]
        if isinstance(texts, str):
            return self._encoder(texts)
        unique = list(dict.fromkeys(texts))
        feat, lengths = self._encoder(unique)
        row_of = {text: i for i, text in enumerate(unique)}
        rows = [row_of[text] for text in texts]
        return feat[rows], [lengths[i] for i in rows]

    def __getattr__(self, name):  # type: ignore[no-untyped-def]
        return getattr(self._encoder, name)

def _positions_to_global_rots(model, positions):  # type: ignore[no-untyped-def]
    """Per-joint global rotations reproducing ``positions`` under kimodo's
    skeleton FK (standard SMPL convention): joint ``j``'s rotation orients its
    *outgoing* bones (``j`` → children), so it is recovered by least-squares
    aligning the neutral outgoing bones to the target outgoing bones. Leaf
    joints have no outgoing bone and inherit their parent's rotation (they do
    not affect any joint position).
    """
    from scipy.spatial.transform import Rotation

    skeleton = model.skeleton
    neutral_joints = skeleton.neutral_joints.detach().cpu().numpy()
    joint_parents = skeleton.joint_parents
    parents = (
        joint_parents.detach().cpu().numpy()
        if hasattr(joint_parents, "detach")
        else np.asarray(joint_parents)
    )
    nbjoints = skeleton.nbjoints

    positions_np = np.asarray(positions, dtype=np.float64).reshape(nbjoints, 3)
    children: list[list[int]] = [[] for _ in range(nbjoints)]
    for j in range(nbjoints):
        parent = int(parents[j])
        if parent >= 0:
            children[parent].append(j)

    world_rots = [Rotation.identity() for _ in range(nbjoints)]
    global_rots = np.empty((nbjoints, 3, 3), dtype=np.float32)
    # parent(j) < j in SMPL topology, so a leaf's parent is already resolved.
    for j in range(nbjoints):
        kids = children[j]
        if not kids:
            parent = int(parents[j])
            world_rots[j] = (
                world_rots[parent] if parent >= 0 else Rotation.identity()
            )
        else:
            neutral_bones = [neutral_joints[c] - neutral_joints[j] for c in kids]
            target_bones = [positions_np[c] - positions_np[j] for c in kids]
            world_rots[j], _ = Rotation.align_vectors(target_bones, neutral_bones)
        global_rots[j] = world_rots[j].as_matrix().astype(np.float32)

    return global_rots


def _build_constraints(model, start_positions_path):  # type: ignore[no-untyped-def]
    if start_positions_path is None:
        return []

    import torch

    from kimodo.constraints import FullBodyConstraintSet

    device = model.skeleton.device
    target_positions_np = np.load(start_positions_path)
    # Match kimodo's standardization: joint positions are global (not
    # canonicalized) with Y the absolute height, and the first frame is placed
    # with the feet on the floor (Y=0) and the root over the origin. The
    # MDM/SMPL start pose instead centers the root near the origin with the
    # feet ~1 m below, so without this the constrained body starts a meter
    # underground and the model yanks it up over the first frames, crumpling.
    root_idx = model.skeleton.root_idx
    target_positions_np[:, [0, 2]] -= target_positions_np[root_idx, [0, 2]]
    global_rots_np = _positions_to_global_rots(model, target_positions_np)
    global_rots = torch.as_tensor(
        global_rots_np, dtype=torch.float32, device=device
    ).reshape(1, model.skeleton.nbjoints, 3, 3)
    root_position = torch.as_tensor(
        target_positions_np[root_idx], dtype=torch.float32, device=device
    ).reshape(1, 3)
    local_rots = model.skeleton.global_rots_to_local_rots(global_rots)
    _, positions, _ = model.skeleton.fk(local_rots, root_position)
    # Ground the feet in kimodo-FK space (its bone lengths differ from SMPL's),
    # so the lowest joint sits exactly on the floor at Y=0 rather than ~0.15 m
    # under it — otherwise the model lifts the feet over the first frames.
    positions[..., 1] -= positions[..., 1].min()
    frame_indices = torch.tensor([0], dtype=torch.long, device=device)
    return [
        FullBodyConstraintSet(model.skeleton, frame_indices, positions, global_rots)
    ]


def main() -> None:
    args = _parse_args()

    import os

    import torch

    from kimodo.exports import get_amass_parameters
    from kimodo.model.load_model import load_model
    from kimodo.tools import seed_everything

    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _vram_gb = (
        torch.cuda.get_device_properties(0).total_memory / 1024**3
        if torch.cuda.is_available()
        else 0
    )
    os.environ.setdefault("TEXT_ENCODER_DEVICE", "cpu" if _vram_gb < 18 else device)

    model, resolved = load_model(
        args.model_name, device=device, return_resolved_name=True
    )
    model.text_encoder = _DedupTextEncoder(model.text_encoder)
    print(f"[kimodo worker] loaded model: {resolved} on {device}")

    # Condition generation on the start pose's heading. Without this the model
    # uses its default first_heading_angle=0 (facing +Z) and reorients the body
    # to it regardless of the constraint, spinning the start pose around.
    first_heading_angle = None
    if args.start_positions_path is not None:
        from kimodo.motion_rep.feature_utils import compute_heading_angle

        start_positions = np.load(args.start_positions_path)
        start_positions_t = torch.as_tensor(
            start_positions, dtype=torch.float32, device=device
        ).reshape(1, 1, -1, 3)
        angle = compute_heading_angle(start_positions_t, model.skeleton).reshape(-1)[0]
        first_heading_angle = angle.repeat(args.num_samples)

    # A list of prompts drives num_samples: pass one entry per sample of the
    # same text (kimodo sets num_samples = len(prompts)).
    prompts = [args.text] * args.num_samples
    output = model(
        prompts,
        args.num_frames,
        args.num_denoising_steps,
        constraint_lst=_build_constraints(model, args.start_positions_path),
        first_heading_angle=first_heading_angle,
        return_numpy=True,
        post_processing=True,
        progress_bar=partial(tqdm, desc="kimodo motion generation"),
    )

    local_rot_mats = np.asarray(output["local_rot_mats"])  # (B, T, J, 3, 3)
    root_positions = np.asarray(output["root_positions"])  # (B, T, 3)
    n_samples, n_frames, n_joints = local_rot_mats.shape[:3]
    local_rot_t = torch.as_tensor(local_rot_mats, device=device)
    root_pos_t = torch.as_tensor(root_positions, device=device)
    _, positions_t, _ = model.skeleton.fk(
        local_rot_t.reshape(-1, n_joints, 3, 3),
        root_pos_t.reshape(-1, 3),
    )
    positions = (
        positions_t.detach().cpu().numpy().reshape(n_samples, n_frames, n_joints, 3)
    )

    # get_amass_parameters is @ensure_batched and returns
    # (trans (B,T,3), root_orient (B,T,3), pose_body (B,T,(J-1)*3)).
    _trans, _root_orient, pose_body = get_amass_parameters(
        local_rot_mats, root_positions, model.skeleton, z_up=True
    )
    pose_body = np.asarray(pose_body)

    # First 21 SMPL body joints → SMPL body_pose (B, T, 21, 3).
    body_pose = pose_body[..., : _N_SMPL_BODY * 3].reshape(
        pose_body.shape[0], pose_body.shape[1], _N_SMPL_BODY, 3
    )
    np.savez(args.output_path, body_pose=body_pose, positions=positions)
    print(
        f"[kimodo worker] wrote {body_pose.shape} body_pose and "
        f"{positions.shape} positions to {args.output_path}"
    )


if __name__ == "__main__":
    main()
