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

MDM_MODEL_WEIGHTS_PATH = (
    MDM_ROOT
    / "motion-diffusion-model"
    / "save"
    / "customv3_fixed"
    / "model000750500.pt"
)

# kimodo (NVIDIA) text-to-motion backend. kimodo conflicts with the main env
# (pydantic>=2, transformers==5.1.0), so it runs in an isolated conda env via a
# subprocess worker — mirroring the SAM/MHR worker pattern in data_collection.
KIMODO_ROOT = PROJECT_ROOT / "motion_generators" / "kimodo"
KIMODO_CONDA_ENV = "kimodo"
# SMPL-X model variant: its get_amass_parameters() output (pose_body (T,63)) maps
# directly onto SMPL body_pose (T,21,3), reusing hml_smpl_conversion's helpers.
KIMODO_MODEL = "Kimodo-SMPLX-RP-v1"
# Default SMPL body_pose (21,3) start pose for kimodo. ``start_pose.npy`` is the
# original (front-back mirrored relative to kimodo's convention, from the MDM
# conversion); ``start_pose_kimodo.npy`` is the corrected pose (whole-pose Z
# mirror, re-IK'd) so legs/forearms point forward and match kimodo's distribution.
KIMODO_START_POSE_PATH = KIMODO_ROOT / "start_pose_kimodo.npy"
