"""Convert world-space 22-joint positions to 263-dim HumanML3D (HML263) features.

Feature construction is delegated to the **official** HumanML3D ``process_file``
from the MDM submodule (since 2026-07-07), so our data is encoded with exactly
the pipeline the *pretrained* ``humanml_enc_512_50steps`` checkpoint was trained
on.  Datasets built before that date — ``dataset/custom1``, and hence the
``save/customv3_fixed`` fine-tune (the default until 2026-08-09) — use the old
homegrown encoding instead; ``dataset/custom1_seatedcanon`` is the
``process_file`` re-encoding of the same clips, and the training data of the
current default checkpoint:

- ``uniform_skeleton`` retargeting onto the standard t2m skeleton
- floor grounding, root-XZ origin, initial facing to Z+
- quaternion IK (with forward-direction smoothing) for the 6D rotation block
- heading-local forward-difference joint velocities
- squared-velocity foot contacts

``process_file`` returns ``N-1`` feature frames for ``N`` position frames (the
last frame is dropped, per the HumanML3D convention).

The t2m target skeleton is read from ``t2m_example_frame.npy`` next to this
file (one ``(22, 3)`` frame from a retargeted HumanML3D motion; all retargeted
files share identical bone lengths).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import numpy as np

from uncertain_feedback.consts import MDM_ROOT

_T2M_EXAMPLE_FRAME = Path(__file__).parent / "t2m_example_frame.npy"


def _official_motion_process() -> ModuleType:
    """Import the MDM submodule's ``motion_process`` with its globals set up.

    ``process_file`` reads module-level globals that upstream only defines
    under ``if __name__ == "__main__"``; set them once here (values copied
    from that block).
    """
    mdm_dir = MDM_ROOT / "motion-diffusion-model"
    if str(mdm_dir) not in sys.path:
        sys.path.insert(0, str(mdm_dir))

    # pylint: disable=import-outside-toplevel,import-error
    import data_loaders.humanml.scripts.motion_process as mp
    import torch
    from data_loaders.humanml.common.skeleton import Skeleton
    from data_loaders.humanml.utils import paramUtil

    if not hasattr(mp, "tgt_offsets"):
        mp.l_idx1, mp.l_idx2 = 5, 8  # lower legs (skeleton scale reference)
        mp.fid_r, mp.fid_l = [8, 11], [7, 10]
        mp.face_joint_indx = [2, 1, 17, 16]  # r_hip, l_hip, sdr_r, sdr_l
        mp.n_raw_offsets = torch.from_numpy(paramUtil.t2m_raw_offsets)
        mp.kinematic_chain = paramUtil.t2m_kinematic_chain
        example_frame = torch.from_numpy(np.load(_T2M_EXAMPLE_FRAME))  # (22, 3)
        tgt_skel = Skeleton(mp.n_raw_offsets, mp.kinematic_chain, "cpu")
        mp.tgt_offsets = tgt_skel.get_offsets_joints(example_frame)
    return mp


def positions_to_hml263(
    positions: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    normalize: bool = True,
    feet_thre: float = 0.002,
) -> np.ndarray:
    """Convert world-space 22-joint positions to HML263 features.

    Args:
        positions: ``(N, 22, 3)`` world-space (Y-up, metres) joint positions.
        mean:      ``(263,)`` HumanML3D mean (from ``Mean.npy``).
        std:       ``(263,)`` HumanML3D std  (from ``Std.npy``).
        normalize: If True (default), return z-normalized HML263 using
                   ``mean``/``std``. If False, return raw HML263.
        feet_thre: Foot-contact squared-velocity threshold (HumanML3D default).

    Returns:
        ``(N-1, 263)`` HML263 feature array, ``float32``.
    """
    mp = _official_motion_process()
    positions = np.ascontiguousarray(positions, dtype=np.float32)
    if positions.ndim != 3 or positions.shape[1:] != (22, 3):
        raise ValueError(f"Expected (N, 22, 3) positions, got {positions.shape}")
    if len(positions) < 2:
        raise ValueError(f"Need at least 2 frames, got {len(positions)}.")

    features, _, _, _ = mp.process_file(positions, feet_thre)
    features = features.astype(np.float32)  # (N-1, 263)

    if normalize:
        features = (features - mean.astype(np.float32)) / (
            std.astype(np.float32) + 1e-8
        )
    return features


def load_hml_stats(stats_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load HumanML3D normalization statistics.

    Args:
        stats_dir: Directory containing ``Mean.npy`` and ``Std.npy``.

    Returns:
        ``(mean, std)`` each of shape ``(263,)``.
    """
    stats_dir = Path(stats_dir)
    mean = np.load(stats_dir / "Mean.npy")  # (263,)
    std = np.load(stats_dir / "Std.npy")  # (263,)
    return mean.astype(np.float32), std.astype(np.float32)
