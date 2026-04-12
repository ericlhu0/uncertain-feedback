"""Top-level pipeline: image folder → HML263 feature array.

Orchestrates the full data-collection pipeline:

1. :class:`~uncertain_feedback.data_collection.mhr_pose_estimator.MhrPoseEstimator`
   runs SAM 3D Body inference + MHR→SMPL conversion (inside the ``sam-3d-body``
   conda env via subprocess).
2. :func:`~uncertain_feedback.data_collection.smpl_to_hml263.smpl_params_to_hml263`
   converts SMPL body_pose to the 263-dim HumanML3D feature vector.

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
    # hml263.shape == (N, 263)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from uncertain_feedback.data_collection.mhr_pose_estimator import (
    MhrEstimatorConfig,
    MhrPoseEstimator,
)
from uncertain_feedback.data_collection.smpl_to_hml263 import (
    load_hml_stats,
    positions_to_hml263,
)
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK


@dataclass
class MhrToHml263Config:
    """Configuration for :class:`MhrToHml263Pipeline`.

    Attributes:
        mhr_estimator_config: Config for the SAM + MHR→SMPL estimator.
        hml_stats_dir: Directory containing ``Mean.npy`` and ``Std.npy`` for
            HumanML3D normalization.
        foot_height_thresh: Foot height (SMPL units, usually metres) below
            which a joint is treated as grounded.
        foot_vel_thresh: Per-frame foot speed below which a joint is treated
            as stationary.
    """

    mhr_estimator_config: MhrEstimatorConfig = field(default_factory=MhrEstimatorConfig)
    hml_stats_dir: Optional[Path] = None
    foot_height_thresh: float = 0.05
    foot_vel_thresh: float = 0.05


class MhrToHml263Pipeline:
    """Convert an image folder to a normalized HML263 motion array.

    Args:
        config: Pipeline configuration.
    """

    def __init__(self, config: MhrToHml263Config) -> None:
        self._config = config
        self._estimator = MhrPoseEstimator(config.mhr_estimator_config)
        self._tpose_22 = SmplLeftArmFK().tpose_all_joints  # (22, 3)

    def run(self, image_folder: Path) -> np.ndarray:
        """Run the full pipeline: images → HML263.

        Args:
            image_folder: Directory of image frames (treated as a video
                sequence; ordered by natural sort on filenames).

        Returns:
            ``(N, 263)`` normalized HML263 feature array, ``float32``.

        Raises:
            ValueError: If ``hml_stats_dir`` is not configured.
            RuntimeError: If SAM + MHR→SMPL worker fails.
        """
        if self._config.hml_stats_dir is None:
            raise ValueError("MhrToHml263Config.hml_stats_dir must be set.")

        result = self._estimator.run(image_folder)
        mean, std = load_hml_stats(self._config.hml_stats_dir)

        return positions_to_hml263(
            positions=result["smpl_positions"],
            mean=mean,
            std=std,
            tpose_22=self._tpose_22,
            foot_height_thresh=self._config.foot_height_thresh,
            foot_vel_thresh=self._config.foot_vel_thresh,
        )

    def run_to_smpl_positions(self, image_folder: Path) -> np.ndarray:
        """Return world-space 22-joint positions (debug/visualization helper).

        Args:
            image_folder: Directory of image frames.

        Returns:
            ``(N, 22, 3)`` world-space joint positions in SMPL units.
        """
        result = self._estimator.run(image_folder)
        return result["smpl_positions"]
