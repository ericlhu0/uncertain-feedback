"""Project-wide constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
MDM_ROOT = PROJECT_ROOT / "motion_generators" / "mdm"

# Host-shared storage for artifacts too big for git or home quota (MDM
# checkpoints live under DATA_DIR / "mdm_save").  Must exist on every host that
# runs the pipeline.
# DATA_DIR = Path("/share/bhattacharjee/eric_data")
DATA_DIR = PROJECT_ROOT.parent.parent.parent / "data"

MDM_MODEL_WEIGHTS_PATH = (
    # fine tuned on only corrections off one trajectory and "raise my arm up a bit"
    # DATA_DIR / "mdm_save" / "correction_demo1_lr1e5_5k" / "model000752000.pt"
    # fine tuned on 100 auto-generated trajectories and corrections
    # DATA_DIR / "mdm_save" / "correction_auto100_lr1e5_5k" / "model000755051.pt"
    # 1100 clips (600 default-paced + 500 elbow-rich), templated captions plus LLM
    # paraphrases; best alignment / violation / acceptable rate on the 80-case
    # grounding eval of 2026-09-09 (README, "Results (2026-09-09 ...)")
    # DATA_DIR / "mdm_save" / "correction_auto1100_paraphrased_lr1e5_26k" / "model000776650.pt"
    # the same 1100 clips recaptioned by the VLM with no fact filter (pure-VLM recipe),
    # 15k-step checkpoint: best alignment of its run on the 80-case grounding eval of
    # 2026-09-15 (README, "Results (2026-09-09 ...)" table)
    # DATA_DIR / "mdm_save" / "correction_auto1100_vlm_lr1e5_26k" / "model000765000.pt"
    # K10-ndsh: that checkpoint + 6500 steps of the replay recipe, HumanML3D prior restored
    # DATA_DIR / "mdm_save"
    # / "correction_auto1100_vlm_hml50_tagged_srcnull_lr3e6_kd10ndsh_k10" / "model000771500.pt"
    # the PC's speed recipe on top of that, then 6500 steps of the replay recipe: keeps the
    # HumanML3D prior and reads "slowly" (README, "Speed with the prior kept")
    DATA_DIR
    / "mdm_save"
    / "replay_kd10ndsh_hml50_on_pcspeed"
    / "model000778000.pt"
)

# The stock MDM checkpoint every fine-tune starts from.
MDM_BASE_WEIGHTS_PATH = (
    DATA_DIR / "mdm_save" / "humanml_enc_512_50steps" / "model000750000.pt"
)

# HumanML3D (MDM's pretraining set) rebuilt from AMASS with the official scripts; AMASS's
# license forbids redistributing it, so it exists only where it was built (see
# DATA_DIR / "humanml3d_build").
HUMANML3D_DIR = DATA_DIR / "HumanML3D" / "HumanML3D"

# The T2M text-motion evaluator every HumanML3D paper scores with (MDM's
# prepare/download_t2m_evaluators.sh).
T2M_EVALUATOR_DIR = DATA_DIR / "t2m_evaluator" / "t2m"

# Default whole-body HML263 start pose, used by every MPC config that does not
# set `pose:` itself.  It supplies the rest-of-body backdrop and the initial arm
# configuration, so it must be a seated body: the deployed checkpoint is
# fine-tuned on 44 seated clips (pelvis 0.656 m, knee flexion 95 deg, non-arm
# spread 0.046 m across the set).  See CODEBASE_MAP.md, "Checkpoint training
# bodies", for which body pose each checkpoint was trained on and how far each
# candidate start pose sits from it.
MDM_START_POSE_PATH = MDM_ROOT / "mdm_sit_pose.pt"
