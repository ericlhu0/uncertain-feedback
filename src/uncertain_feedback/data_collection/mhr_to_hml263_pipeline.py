"""Top-level pipeline: image folder → HML263 feature array.

Orchestrates the full data-collection pipeline:

1. :class:`~uncertain_feedback.data_collection.mhr_pose_estimator.MhrPoseEstimator`
   runs SAM 3D Body inference + MHR→SMPL conversion (inside the ``sam-3d-body``
   conda env via subprocess).
2. :func:`~uncertain_feedback.data_collection.smpl_to_hml263.positions_to_hml263`
   converts world-space joint positions to the 263-dim HumanML3D feature vector
   via the official HumanML3D ``process_file``.

Example::

    from pathlib import Path
    from uncertain_feedback.data_collection.mhr_to_hml263_pipeline import (
        MhrToHml263Config, MhrToHml263Pipeline,
    )
    from uncertain_feedback.data_collection.mhr_pose_estimator import MhrEstimatorConfig

    config = MhrToHml263Config(
        mhr_estimator_config=MhrEstimatorConfig(
            sam_checkpoint_path=Path("~/sam-3d-body/checkpoints/model.ckpt"),
            smpl_model_path=Path("~/MHR/tools/mhr_smpl_conversion/data/SMPL_NEUTRAL.pkl"),
        ),
        hml_stats_dir=Path("path/to/HumanML3D/Mean_Std"),
    )
    pipeline = MhrToHml263Pipeline(config)
    hml263 = pipeline.run(Path("./video_frames/"))
    # hml263.shape == (N-1, 263)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

_HML_TARGET_FPS: float = 20.0


def resample_positions(positions: np.ndarray, n_out: int) -> np.ndarray:
    """Linearly resample ``(N, 22, 3)`` joint positions to ``n_out`` frames."""
    n_in = len(positions)
    if n_in == n_out:
        return positions
    old_t = np.linspace(0.0, 1.0, n_in)
    new_t = np.linspace(0.0, 1.0, n_out)
    flat = positions.reshape(n_in, -1)  # (N, 66)
    resampled = np.stack(
        [np.interp(new_t, old_t, flat[:, i]) for i in range(flat.shape[1])],
        axis=1,
    ).astype(np.float32)
    return resampled.reshape(n_out, 22, 3)


def _resample_smpl_positions(
    positions: np.ndarray,
    source_fps: float,
    target_fps: float = _HML_TARGET_FPS,
) -> np.ndarray:
    """Resample (N, 22, 3) SMPL positions from *source_fps* to *target_fps*."""
    if abs(source_fps - target_fps) < 1e-6:
        return positions
    n_out = max(1, round(len(positions) * target_fps / source_fps))
    return resample_positions(positions, n_out)


from uncertain_feedback.data_collection.mhr_pose_estimator import (
    MhrEstimatorConfig,
    MhrPoseEstimator,
)
from uncertain_feedback.data_collection.smpl_to_hml263 import (
    load_hml_stats,
    positions_to_hml263,
)


@dataclass
class MhrToHml263Config:
    """Configuration for :class:`MhrToHml263Pipeline`.

    Attributes:
        mhr_estimator_config: Config for the SAM + MHR→SMPL estimator.
        hml_stats_dir: Directory containing ``Mean.npy`` and ``Std.npy`` for
            HumanML3D normalization.
        output_normalized: If True (default), output z-normalized HML263
            vectors. If False, output raw HML263 vectors.
    """

    mhr_estimator_config: MhrEstimatorConfig = field(default_factory=MhrEstimatorConfig)
    hml_stats_dir: Optional[Path] = None
    output_normalized: bool = True


class MhrToHml263Pipeline:
    """Convert an image folder to an HML263 motion array.

    Args:
        config: Pipeline configuration.
    """

    def __init__(self, config: MhrToHml263Config) -> None:
        self._config = config
        self._estimator = MhrPoseEstimator(config.mhr_estimator_config)

    def run(self, image_folder: Path, source_fps: float = _HML_TARGET_FPS) -> np.ndarray:
        """Run the full pipeline: images → HML263.

        Args:
            image_folder: Directory of image frames (treated as a video
                sequence; ordered by natural sort on filenames).
            source_fps: Frame rate at which the images were extracted.
                SMPL positions are resampled to 20 FPS before HML263
                conversion.  Defaults to 20 (no resampling).

        Returns:
            ``(N-1, 263)`` HML263 feature array, ``float32``, where ``N`` is
            the number of position frames after resampling.
            Normalized vs. raw is controlled by
            :attr:`MhrToHml263Config.output_normalized`.

        Raises:
            ValueError: If ``hml_stats_dir`` is not configured.
            RuntimeError: If SAM + MHR→SMPL worker fails.
        """
        if self._config.hml_stats_dir is None:
            raise ValueError("MhrToHml263Config.hml_stats_dir must be set.")

        positions = self.run_to_smpl_positions(image_folder, source_fps)
        mean, std = load_hml_stats(self._config.hml_stats_dir)

        return positions_to_hml263(
            positions=positions,
            mean=mean,
            std=std,
            normalize=self._config.output_normalized,
        )

    def run_to_smpl_positions(
        self, image_folder: Path, source_fps: float = _HML_TARGET_FPS
    ) -> np.ndarray:
        """Return world-space 22-joint positions resampled to 20 FPS.

        Args:
            image_folder: Directory of image frames.
            source_fps: Frame rate at which the images were extracted.

        Returns:
            ``(N, 22, 3)`` world-space joint positions in SMPL units at 20 FPS.
        """
        result = self._estimator.run(image_folder)
        return _resample_smpl_positions(result["smpl_positions"], source_fps)
