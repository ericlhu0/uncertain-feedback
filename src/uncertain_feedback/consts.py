"""Project-wide constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
MDM_ROOT = PROJECT_ROOT / "motion_generators" / "mdm"
# MDM_MODEL_WEIGHTS_PATH = (
#     MDM_ROOT
#     / "motion-diffusion-model"
#     / "save"
#     / "customv2"
#     / "model000753000.pt"
# )
# Previous default, superseded 2026-08-09 (old homegrown encoding, seam 0.277 m):
# MDM_MODEL_WEIGHTS_PATH = (
#     MDM_ROOT
#     / "motion-diffusion-model"
#     / "save"
#     / "customv3_fixed"
#     / "model000750500.pt"
# )

# Fine-tuned on dataset/custom1_seatedcanon at lr 1e-7 for 9250 steps, so query
# poses (official process_file encoding) are on-manifold.  Checkpoints are
# gitignored — this file must exist on every host that runs the pipeline.
# MDM_MODEL_WEIGHTS_PATH = (
#     MDM_ROOT
#     / "motion-diffusion-model"
#     / "save"
#     / "custom_seatedcanon_lr1e7_10k"
#     / "model000759250.pt"
# )

MDM_MODEL_WEIGHTS_PATH = (
    MDM_ROOT
    / "motion-diffusion-model"
    / "save"
    / "correction_demo1_lr1e5_5k"
    / "model000752000.pt"
)

# Default whole-body HML263 start pose, used by every MPC config that does not
# set `pose:` itself.  It supplies the rest-of-body backdrop and the initial arm
# configuration, so it must be a seated body: the deployed checkpoint is
# fine-tuned on 44 seated clips (pelvis 0.656 m, knee flexion 95 deg, non-arm
# spread 0.046 m across the set).  See CODEBASE_MAP.md, "Checkpoint training
# bodies", for which body pose each checkpoint was trained on and how far each
# candidate start pose sits from it.
MDM_START_POSE_PATH = MDM_ROOT / "mdm_sit_pose.pt"
