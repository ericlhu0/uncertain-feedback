"""Subprocess bridge: invoke the MHR inference worker inside the sam-3d-body conda env.

The worker script ``_mhr_inference_worker.py`` lives in the same package directory
and runs as a standalone Python script inside the ``sam-3d-body`` conda environment,
which has SAM 3D Body installed.  No MHR repo or SMPL model is required — joint
positions are obtained by mapping ``pred_keypoints_3d`` (MHR-70) directly to
approximate SMPL-22 positions.

Example::

    from uncertain_feedback.data_collection.mhr_pose_estimator import (
        MhrEstimatorConfig, MhrPoseEstimator,
    )

    config = MhrEstimatorConfig(
        sam_checkpoint_path=Path(
            "data_collection/sam-3d-body/checkpoints/sam-3d-body-dinov3/model.ckpt"
        ),
    )
    estimator = MhrPoseEstimator(config)
    result = estimator.run(Path("./video_frames/"))
    # result["smpl_positions"].shape == (N, 22, 3)
"""

from __future__ import annotations

import subprocess
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class MhrEstimatorConfig:  # pylint: disable=too-few-public-methods
    """Configuration for :class:`MhrPoseEstimator`.

    Attributes:
        sam_repo_path: Root of the ``sam-3d-body`` git submodule.
        conda_env: Name of the conda environment containing SAM 3D Body.
        sam_checkpoint_path: Path to the SAM 3D Body model checkpoint.
        mhr_path: Path to ``mhr_model.pt`` passed to ``load_sam_3d_body``.
            Defaults to the assets directory inside the submodule.
        bbox_thresh: Bounding-box detection confidence threshold.
        detector_name: Human detector name (passed to SAM estimator).
        fov_name: FOV estimator name (passed to SAM estimator).
    """

    sam_repo_path: Path = field(
        default_factory=lambda: Path(__file__).parent / "sam-3d-body"
    )
    conda_env: str = "sam_3d_body"
    sam_checkpoint_path: Optional[Path] = None
    mhr_path: str = ""
    bbox_thresh: float = 0.8
    detector_name: str = "vitdet"
    fov_name: str = "moge2"


class MhrPoseEstimator:  # pylint: disable=too-few-public-methods
    """Run SAM 3D Body + MHR→SMPL conversion on a folder of images.

    The heavy inference is done inside the ``sam-3d-body`` conda environment via
    ``conda run``, so this class works even when the current Python environment
    does not have SAM or the MHR conversion package installed.

    Args:
        config: Estimator configuration.
    """

    def __init__(self, config: MhrEstimatorConfig) -> None:
        self._config = config
        self._worker_path = Path(__file__).parent / "_mhr_inference_worker.py"
        if not self._worker_path.exists():
            raise FileNotFoundError(f"Worker script not found: {self._worker_path}")

    def run(self, image_folder: Path) -> dict[str, np.ndarray]:
        """Run pose estimation on all images in a folder.

        Images are treated as an ordered video sequence (natural sort by filename).
        Frames in which no person is detected are silently dropped — the output
        sequence length may therefore be shorter than the number of input images.

        Args:
            image_folder: Directory containing input images (jpg/png/etc.).

        Returns:
            Dictionary with keys:

            * ``"smpl_positions"`` – ``(N, 22, 3)`` world-space SMPL-22 joint positions
            * ``"frame_paths"``    – ``(N,)`` str (absolute image paths)

        Raises:
            RuntimeError: If the worker subprocess exits with a non-zero code.
            FileNotFoundError: If *image_folder* does not exist.
        """
        image_folder = Path(image_folder).expanduser().resolve()
        if not image_folder.is_dir():
            raise FileNotFoundError(f"Image folder not found: {image_folder}")

        cfg = self._config
        if cfg.sam_checkpoint_path is None:
            raise ValueError("MhrEstimatorConfig.sam_checkpoint_path must be set.")

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            output_path = tmp.name

        cmd = [
            "conda",
            "run",
            "-n",
            cfg.conda_env,
            "python",
            str(self._worker_path),
            "--image_folder",
            str(image_folder),
            "--output_path",
            output_path,
            "--sam_checkpoint_path",
            str(cfg.sam_checkpoint_path.expanduser()),
            "--sam_repo_path",
            str(cfg.sam_repo_path),
            "--mhr_path",
            cfg.mhr_path,
            "--bbox_thresh",
            str(cfg.bbox_thresh),
            "--detector_name",
            cfg.detector_name,
            "--fov_name",
            cfg.fov_name,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.stdout:
            print(result.stdout, end="")
        if result.returncode != 0:
            raise RuntimeError(
                f"MHR inference worker failed (exit {result.returncode}):\n"
                f"{result.stderr}"
            )
        if result.stderr:
            warnings.warn(f"Worker stderr:\n{result.stderr}")

        data = np.load(output_path, allow_pickle=True)
        return {k: data[k] for k in data.files}
