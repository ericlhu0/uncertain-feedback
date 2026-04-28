"""Data collection pipeline: images → HML263 for MDM fine-tuning.

The pipeline converts a folder of images (treated as a video sequence) to
263-dimensional HumanML3D feature vectors suitable for fine-tuning the MDM
motion diffusion model.

Stages
------
1. **SAM 3D Body inference** — estimates MHR (Momentum Human Rig) pose from
   each image, running inside the ``sam-3d-body`` conda environment via
   subprocess.
2. **MHR → SMPL conversion** — uses the official MHR repo
   (``~/MHR/tools/mhr_smpl_conversion``) to fit a SMPL body model to the MHR
   predictions via optimization-based surface mapping.
3. **SMPL → HML263** — converts SMPL ``body_pose`` / ``global_orient`` /
   ``transl`` to the 263-dim HumanML3D feature vector.

Quick start::

    from uncertain_feedback.data_collection import MhrToHml263Pipeline, MhrToHml263Config
    from uncertain_feedback.data_collection import MhrEstimatorConfig
    from pathlib import Path

    config = MhrToHml263Config(
        mhr_estimator_config=MhrEstimatorConfig(
            sam_checkpoint_path=Path("~/sam-3d-body/checkpoints/model.ckpt"),
            smpl_model_path=Path("~/MHR/tools/mhr_smpl_conversion/data/SMPL_NEUTRAL.pkl"),
        ),
        hml_stats_dir=Path("path/to/HumanML3D/Mean_Std/"),
    )
    hml263 = MhrToHml263Pipeline(config).run(Path("./video_frames/"))
"""

from uncertain_feedback.data_collection.mhr_pose_estimator import (
    MhrEstimatorConfig,
    MhrPoseEstimator,
)
from uncertain_feedback.data_collection.mhr_to_hml263_pipeline import (
    MhrToHml263Config,
    MhrToHml263Pipeline,
)
from uncertain_feedback.data_collection.smpl_to_hml263 import (
    load_hml_stats,
    positions_to_hml263,
    smpl_params_to_hml263,
    smpl_params_to_positions,
)

__all__ = [
    "MhrEstimatorConfig",
    "MhrPoseEstimator",
    "MhrToHml263Config",
    "MhrToHml263Pipeline",
    "load_hml_stats",
    "positions_to_hml263",
    "smpl_params_to_hml263",
    "smpl_params_to_positions",
]
