"""Standalone worker: SAM 3D Body inference + MHR-70 → SMPL-22 direct mapping.

Runs inside the sam-3d-body conda env.  Invoked by
:class:`~uncertain_feedback.data_collection.mhr_pose_estimator.MhrPoseEstimator`
via ``conda run``.

Usage::

    conda run -n sam_3d_body python _mhr_inference_worker.py \\
        --image_folder ./frames \\
        --output_path /tmp/smpl_out.npz \\
        --sam_checkpoint_path ./checkpoints/model.ckpt \\
        [--sam_repo_path ./sam-3d-body] \\
        [--mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt] \\
        [--bbox_thresh 0.8] \\
        [--detector_name vitdet] \\
        [--fov_name moge2]

No MHR repo or SMPL model required — joint positions are computed by mapping
``pred_keypoints_3d`` (MHR-70) directly to approximate SMPL-22 positions via a
fixed index mapping and simple geometric interpolations for the 7 missing joints.

Output .npz keys
----------------
``smpl_positions``   (N, 22, 3) – world-space SMPL-22 joint positions.
``frame_paths``      (N,)       – absolute paths of the processed image files.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

# Add sam-3d-body submodule to sys.path before importing its packages.
# Pass --sam_repo_path at runtime to override the default submodule location.
_SAM_REPO_DEFAULT = Path(__file__).parent / "sam-3d-body"
if str(_SAM_REPO_DEFAULT) not in sys.path:
    sys.path.insert(0, str(_SAM_REPO_DEFAULT))

# sam_3d_body and tools.* are only importable after the sys.path setup above.
# pylint: disable=wrong-import-position,import-error,no-name-in-module
from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body  # type: ignore[import]
from tools.build_detector import HumanDetector  # type: ignore[import]

# pylint: enable=wrong-import-position,import-error,no-name-in-module


def _natural_sort_key(s: str) -> list:
    """Sort key that orders strings containing numbers naturally."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


def _bbox_area(bbox: np.ndarray) -> float:
    """Return area of an [x1, y1, x2, y2] bounding box."""
    return float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))


def _gather_images(image_folder: str) -> list[str]:
    """Return image paths in *image_folder* sorted by natural filename order."""
    extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
    paths = [
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if os.path.splitext(f)[1].lower() in extensions
    ]
    return sorted(paths, key=lambda p: _natural_sort_key(os.path.basename(p)))


def _mhr70_to_smpl22(kps70: np.ndarray) -> np.ndarray:
    """Map MHR-70 camera-space keypoints to approximate SMPL-22 joint positions.

    Direct mappings cover 15 of 22 SMPL joints (both legs, neck, both arms).
    The remaining 7 joints (pelvis, spine1/2/3, l_collar, r_collar, head) are
    estimated geometrically.

    MHR-70 index reference (from mhr70.py)::

        0=nose  3=l_ear  4=r_ear  5=l_shoulder  6=r_shoulder
        7=l_elbow  8=r_elbow  9=l_hip  10=r_hip  11=l_knee  12=r_knee
        13=l_ankle  14=r_ankle  15=l_big_toe_tip  17=l_heel
        18=r_big_toe_tip  20=r_heel  41=r_wrist  62=l_wrist  69=neck

    Args:
        kps70: ``(70, 3)`` — MHR keypoints already translated by ``pred_cam_t``.

    Returns:
        ``(22, 3)`` — SMPL-22 joint positions in the same coordinate space.
    """
    out = np.zeros((22, 3), dtype=np.float64)

    # ── Direct mappings ───────────────────────────────────────────────────────
    out[1] = kps70[9]  # l_hip
    out[2] = kps70[10]  # r_hip
    out[4] = kps70[11]  # l_knee
    out[5] = kps70[12]  # r_knee
    out[7] = kps70[13]  # l_ankle
    out[8] = kps70[14]  # r_ankle
    out[10] = 0.5 * (kps70[15] + kps70[17])  # l_foot ≈ avg(l_big_toe, l_heel)
    out[11] = 0.5 * (kps70[18] + kps70[20])  # r_foot ≈ avg(r_big_toe, r_heel)
    out[12] = kps70[69]  # neck
    out[16] = kps70[5]  # l_shoulder
    out[17] = kps70[6]  # r_shoulder
    out[18] = kps70[7]  # l_elbow
    out[19] = kps70[8]  # r_elbow
    out[20] = kps70[62]  # l_wrist
    out[21] = kps70[41]  # r_wrist

    # ── Geometric approximations ──────────────────────────────────────────────
    out[0] = 0.5 * (kps70[9] + kps70[10])  # pelvis = mid(l_hip, r_hip)

    pelvis = out[0]
    neck = out[12]
    out[3] = pelvis + 0.25 * (neck - pelvis)  # spine1
    out[6] = pelvis + 0.50 * (neck - pelvis)  # spine2
    out[9] = pelvis + 0.75 * (neck - pelvis)  # spine3

    out[13] = 0.5 * (kps70[69] + kps70[5])  # l_collar = mid(neck, l_shoulder)
    out[14] = 0.5 * (kps70[69] + kps70[6])  # r_collar = mid(neck, r_shoulder)
    out[15] = 0.5 * (kps70[3] + kps70[4])  # head     = mid(l_ear, r_ear)

    return out


def main() -> None:  # pylint: disable=too-many-locals,too-many-statements
    """Run SAM 3D Body inference and map MHR-70 keypoints to SMPL-22 positions."""
    parser = argparse.ArgumentParser(
        description="SAM 3D Body inference + MHR-70 → SMPL-22 direct keypoint mapping"
    )
    parser.add_argument("--image_folder", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument(
        "--sam_checkpoint_path",
        default=str(
            Path(__file__).parent
            / "sam-3d-body"
            / "checkpoints"
            / "sam-3d-body-dinov3"
            / "model.ckpt"
        ),
    )
    parser.add_argument(
        "--sam_repo_path",
        default=str(Path(__file__).parent / "sam-3d-body"),
    )
    parser.add_argument(
        "--mhr_path",
        default=str(
            Path(__file__).parent
            / "sam-3d-body"
            / "checkpoints"
            / "sam-3d-body-dinov3"
            / "assets"
            / "mhr_model.pt"
        ),
    )
    parser.add_argument("--bbox_thresh", type=float, default=0.8)
    parser.add_argument("--detector_name", default="vitdet")
    parser.add_argument("--fov_name", default="")
    args = parser.parse_args()

    # If a non-default sam_repo_path was passed, insert it at front of sys.path.
    sam_repo_path = os.path.expanduser(args.sam_repo_path)
    if sam_repo_path not in sys.path:
        sys.path.insert(0, sam_repo_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # ── Load SAM 3D Body ─────────────────────────────────────────────────────
    print("[worker] Loading SAM 3D Body model …")
    model, model_cfg = load_sam_3d_body(
        args.sam_checkpoint_path, device=device, mhr_path=args.mhr_path
    )

    human_detector = HumanDetector(name=args.detector_name, device=device, path="")

    fov_estimator = None
    if args.fov_name:
        from tools.build_fov_estimator import (  # type: ignore[import]  # pylint: disable=import-outside-toplevel,import-error,no-name-in-module
            FOVEstimator,
        )

        fov_estimator = FOVEstimator(name=args.fov_name, device=device, path="")

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=None,
        fov_estimator=fov_estimator,
    )

    # ── Process images ───────────────────────────────────────────────────────
    image_paths = _gather_images(args.image_folder)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.image_folder}")
    print(f"[worker] Found {len(image_paths)} image(s).")

    sam_outputs: list[dict] = []
    frame_paths: list[str] = []

    for img_path in image_paths:
        outputs = estimator.process_one_image(img_path, bbox_thr=args.bbox_thresh)
        if not outputs:
            warnings.warn(f"No person detected in {img_path!r}; skipping frame.")
            continue
        best = max(outputs, key=lambda o: _bbox_area(o["bbox"]))
        sam_outputs.append(best)
        frame_paths.append(img_path)

    if not sam_outputs:
        raise RuntimeError("No person detected in any frame.")
    print(f"[worker] Kept {len(sam_outputs)} frame(s) with detections.")

    # ── MHR-70 → SMPL-22 direct keypoint mapping ────────────────────────────
    print("[worker] Mapping MHR-70 keypoints → SMPL-22 positions …")
    all_positions: list[np.ndarray] = []
    for out in sam_outputs:
        kps70 = np.asarray(out["pred_keypoints_3d"], dtype=np.float64)  # (70, 3)
        cam_t = np.asarray(out["pred_cam_t"], dtype=np.float64)  # (3,)
        # Translate to camera-space world position, then flip Y from image-down to world-up
        kps70_world = kps70 + cam_t
        kps70_world[:, 1] *= -1
        smpl22 = _mhr70_to_smpl22(kps70_world)  # (22, 3)
        all_positions.append(smpl22)

    positions_arr = np.stack(all_positions, axis=0)  # (N, 22, 3)
    print(f"[worker] Positions shape: {positions_arr.shape}")

    # ── Save results ─────────────────────────────────────────────────────────
    np.savez(
        args.output_path,
        smpl_positions=positions_arr,
        frame_paths=np.array(frame_paths),
    )
    print(f"[worker] Saved SMPL-22 positions to {args.output_path}")


if __name__ == "__main__":
    main()
