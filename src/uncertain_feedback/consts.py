"""Project-wide constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
MDM_ROOT = PROJECT_ROOT / "motion_generators" / "mdm"
MDM_MODEL_WEIGHTS_PATH = (
    MDM_ROOT
    / "motion-diffusion-model"
    / "save"
    / "humanml_enc_512_50steps"
    / "model000750000.pt"
)
