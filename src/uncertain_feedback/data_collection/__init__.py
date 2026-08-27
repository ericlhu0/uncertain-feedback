"""Data generation for MDM fine-tuning, one subpackage per method.

Every method ends in a HumanML3D dataset directory the MDM loader reads:
:mod:`~uncertain_feedback.data_collection.dataset_video` (recorded video →
frames → hand-captioned segments), :mod:`.dataset_auto_correction` (sampled MPC
reaches → oracle corrections → hand captions) and :mod:`.trajectory_editor` (a
hand-authoring UI that writes the dataset directly). :mod:`.common` holds the
HML263 encoder and the dataset-writing helpers they share;
:mod:`.pose_estimation` — re-exported here — is the images → HML263 stack the
video pipeline calls.

Pose-estimation stages
----------------------
1. **SAM 3D Body inference** — estimates MHR (Momentum Human Rig) pose from
   each image, running inside the ``sam-3d-body`` conda environment via
   subprocess.
2. **MHR → SMPL conversion** — uses the official MHR repo
   (``~/MHR/tools/mhr_smpl_conversion``) to fit a SMPL body model to the MHR
   predictions via optimization-based surface mapping.
3. **positions → HML263** — converts world-space 22-joint positions to the
   263-dim HumanML3D feature vector via the official HumanML3D
   ``process_file`` (MDM submodule).

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

from uncertain_feedback.data_collection.common.hml263 import (
    load_hml_stats,
    positions_to_hml263,
)
from uncertain_feedback.data_collection.pose_estimation.mhr_to_hml263_pipeline import (
    MhrToHml263Config,
    MhrToHml263Pipeline,
)
from uncertain_feedback.data_collection.pose_estimation.pose_estimator import (
    MhrEstimatorConfig,
    MhrPoseEstimator,
)

__all__ = [
    "MhrEstimatorConfig",
    "MhrPoseEstimator",
    "MhrToHml263Config",
    "MhrToHml263Pipeline",
    "load_hml_stats",
    "positions_to_hml263",
]
