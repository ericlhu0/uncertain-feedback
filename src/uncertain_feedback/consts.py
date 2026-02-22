"""Project-wide constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
MDM_MODEL_WEIGHTS_PATH = (
    PROJECT_ROOT
    / "motion-generators"
    / "mdm"
    / "motion-diffusion-model"
    / "save"
    / "humanml_enc_512_50steps"
    / "model000750000.pt"
)

print(PROJECT_ROOT)
print(MDM_MODEL_WEIGHTS_PATH)
