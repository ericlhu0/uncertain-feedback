"""Standalone worker: SAM 3D Body inference + MHR→SMPL conversion (via Conversion class).

NOTE: This version requires ``pymomentum`` (Meta-internal) and the MHR repo to be
installed in the conda env.  It is kept as a reference for when the full Conversion
class is available.  For the pymomentum-free path, use ``_mhr_inference_worker.py``
instead.

Runs inside the sam-3d-body conda env.  Invoked by
:class:`~uncertain_feedback.data_collection.mhr_pose_estimator.MhrPoseEstimator`
via ``conda run``.

Usage::

    conda run -n sam-3d-body python _mhr_inference_worker_conversion.py \\
        --image_folder ./frames \\
        --output_path /tmp/smpl_out.npz \\
        --sam_checkpoint_path ./checkpoints/model.ckpt \\
        --smpl_model_path ~/MHR/tools/mhr_smpl_conversion/data/SMPL_NEUTRAL.pkl \\
        [--mhr_repo_path ~/MHR] \\
        [--sam_repo_path ./sam-3d-body] \\
        [--mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt] \\
        [--bbox_thresh 0.8] \\
        [--detector_name vitdet] \\
        [--fov_name moge2]

Output .npz keys
----------------
``smpl_body_pose``    (N, 69)  – 23 joints × 3 axis-angle; first 21 joints are
                                  the 22-joint SMPL body pose used by this project.
``smpl_global_orient`` (N, 3)  – root orientation axis-angle.
``smpl_betas``        (N, 10)  – shape coefficients.
``smpl_transl``       (N, 3)   – root translation.
``result_errors``     (N,)     – per-frame conversion error (avg vertex dist in cm).
``frame_paths``       (N,)     – absolute paths of the processed image files.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import smplx  # type: ignore[import]
import torch

# Add sam-3d-body submodule and MHR repo to sys.path before importing their packages.
# Pass --sam_repo_path / --mhr_repo_path at runtime to override the defaults.
_SAM_REPO_DEFAULT = Path(__file__).parent / "sam-3d-body"
if str(_SAM_REPO_DEFAULT) not in sys.path:
    sys.path.insert(0, str(_SAM_REPO_DEFAULT))

_MHR_REPO_DEFAULT = Path(__file__).parent / "MHR"
_MHR_CONVERSION_DEFAULT = _MHR_REPO_DEFAULT / "tools" / "mhr_smpl_conversion"
for _p in (_MHR_CONVERSION_DEFAULT, _MHR_REPO_DEFAULT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# conversion, mhr, sam_3d_body and tools.* are only importable after sys.path setup above.
# pylint: disable=wrong-import-position,import-error,no-name-in-module
from conversion import Conversion  # type: ignore[import]
from mhr.mhr import MHR  # type: ignore[import]
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


def main() -> None:  # pylint: disable=too-many-locals,too-many-statements
    """Run SAM 3D Body inference and convert MHR output to SMPL parameters."""
    parser = argparse.ArgumentParser(
        description="SAM 3D Body inference + MHR→SMPL conversion worker"
    )
    parser.add_argument("--image_folder", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--sam_checkpoint_path", required=True)
    parser.add_argument("--smpl_model_path", required=True)
    parser.add_argument(
        "--mhr_repo_path",
        default=str(Path(__file__).parent / "MHR"),
    )
    parser.add_argument(
        "--sam_repo_path",
        default=str(Path(__file__).parent / "sam-3d-body"),
    )
    parser.add_argument(
        "--mhr_path",
        default=str(Path(__file__).parent / "MHR" / "assets" / "mhr_model.pt"),
    )
    parser.add_argument("--bbox_thresh", type=float, default=0.8)
    parser.add_argument("--detector_name", default="vitdet")
    # parser.add_argument("--segmentor_name", default="sam2")
    parser.add_argument("--fov_name", default="moge2")
    args = parser.parse_args()

    # Insert any non-default repo paths at front of sys.path.
    for repo_path in (args.sam_repo_path, args.mhr_repo_path):
        p = os.path.expanduser(repo_path)
        if p not in sys.path:
            sys.path.insert(0, p)
    conversion_tool_path = os.path.join(
        os.path.expanduser(args.mhr_repo_path), "tools", "mhr_smpl_conversion"
    )
    if conversion_tool_path not in sys.path:
        sys.path.insert(0, conversion_tool_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        if torch.cuda.get_device_properties(0).major >= 8:
            # tf32 is safe for dense matmuls; skip bfloat16 autocast — the MHR
            # TorchScript model uses sparse ops that don't support BFloat16.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # ── Load SAM 3D Body ─────────────────────────────────────────────────────
    print("[worker] Loading SAM 3D Body model …")
    model, model_cfg = load_sam_3d_body(
        args.sam_checkpoint_path, device=device, mhr_path=args.mhr_path
    )

    human_detector = HumanDetector(name=args.detector_name, device=device, path="")

    human_segmentor = None
    # if args.segmentor_name:
    #     from tools.build_sam import HumanSegmentor  # type: ignore[import]

    #     human_segmentor = HumanSegmentor(
    #         name=args.segmentor_name, device=device, path=""
    #     )

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
        human_segmentor=human_segmentor,
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
        # Pick the person with the largest bounding box.
        best = max(outputs, key=lambda o: _bbox_area(o["bbox"]))
        sam_outputs.append(best)
        frame_paths.append(img_path)

    if not sam_outputs:
        raise RuntimeError("No person detected in any frame.")
    print(f"[worker] Kept {len(sam_outputs)} frame(s) with detections.")

    # ── Load MHR and SMPL models ─────────────────────────────────────────────
    print("[worker] Loading MHR model …")
    mhr_model = MHR.from_files(lod=1, device=device)

    print("[worker] Loading SMPL model …")
    smpl_model = smplx.SMPL(model_path=args.smpl_model_path)

    # ── MHR → SMPL conversion ────────────────────────────────────────────────
    print("[worker] Running MHR → SMPL conversion …")
    converter = Conversion(mhr_model=mhr_model, smpl_model=smpl_model, method="pytorch")
    result = converter.convert_sam3d_output_to_smpl(
        sam3d_outputs=sam_outputs,
        return_smpl_parameters=True,
        return_fitting_errors=True,
    )

    if result.result_parameters is None:
        raise RuntimeError("Conversion returned no SMPL parameters.")

    params = result.result_parameters  # dict[str, torch.Tensor]
    body_pose = params["body_pose"].detach().cpu().numpy()  # (N, 69)
    global_orient = params["global_orient"].detach().cpu().numpy()  # (N, 3)
    betas = params["betas"].detach().cpu().numpy()  # (N, 10)
    transl = params["transl"].detach().cpu().numpy()  # (N, 3)
    errors = (
        result.result_errors
        if result.result_errors is not None
        else np.zeros(len(sam_outputs))
    )

    print(f"[worker] Conversion errors (avg vertex dist): {errors}")

    # ── Save results ─────────────────────────────────────────────────────────
    np.savez(
        args.output_path,
        smpl_body_pose=body_pose,
        smpl_global_orient=global_orient,
        smpl_betas=betas,
        smpl_transl=transl,
        result_errors=errors,
        frame_paths=np.array(frame_paths),
    )
    print(f"[worker] Saved SMPL params to {args.output_path}")


if __name__ == "__main__":
    main()
