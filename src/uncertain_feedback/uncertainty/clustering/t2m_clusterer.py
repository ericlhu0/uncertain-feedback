"""Clusterer over T2M motion-encoder embeddings (the FID/R-precision encoder)."""

from __future__ import annotations

import sys

import numpy as np

from uncertain_feedback.consts import MDM_ROOT
from uncertain_feedback.data_collection.smpl_to_hml263 import positions_to_hml263
from uncertain_feedback.uncertainty.clustering.base import (
    TrajectoryClusterer,
    agglomerative_labels,
)

_MAX_MOTION_LENGTH = 196
_UNIT_LENGTH = 4

_encoder: tuple | None = None


def _get_t2m_encoder():
    """Load the T2M movement/motion encoders and their norm stats once per process."""
    global _encoder
    if _encoder is not None:
        return _encoder

    mdm_dir = MDM_ROOT / "motion-diffusion-model"
    ckpt = mdm_dir / "t2m" / "text_mot_match" / "model" / "finest.tar"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"T2M evaluator weights not found at {ckpt}. Download them with "
            f"`bash prepare/download_t2m_evaluators.sh` from {mdm_dir}."
        )
    if str(mdm_dir) not in sys.path:
        sys.path.insert(0, str(mdm_dir))

    # pylint: disable=import-outside-toplevel,import-error
    import torch
    from data_loaders.humanml.networks.evaluator_wrapper import build_evaluators
    from data_loaders.humanml.utils.word_vectorizer import POS_enumerator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    opt = {
        "dataset_name": "humanml",
        "device": device,
        "dim_word": 300,
        "max_motion_length": _MAX_MOTION_LENGTH,
        "dim_pos_ohot": len(POS_enumerator),
        "dim_motion_hidden": 1024,
        "max_text_len": 20,
        "dim_text_hidden": 512,
        "dim_coemb_hidden": 512,
        "dim_pose": 263,
        "dim_movement_enc_hidden": 512,
        "dim_movement_latent": 512,
        "checkpoints_dir": str(mdm_dir),
        "unit_length": _UNIT_LENGTH,
    }
    _, motion_enc, movement_enc = build_evaluators(opt)
    movement_enc.to(device).eval()
    motion_enc.to(device).eval()

    t2m_mean = np.load(mdm_dir / "dataset" / "t2m_mean.npy")
    t2m_std = np.load(mdm_dir / "dataset" / "t2m_std.npy")
    _encoder = (movement_enc, motion_enc, device, t2m_mean, t2m_std)
    return _encoder


class T2mEmbeddingClusterer(TrajectoryClusterer):
    """Cluster trajectories by their 512-dim T2M motion-encoder embedding.

    Each sample's positions are encoded to normalized HML263 features (the
    evaluator's own ``t2m_mean``/``t2m_std`` convention), passed through the
    pretrained movement + motion encoders, and the embeddings are clustered
    with average-linkage agglomerative.
    """

    def _to_features(self, trajectories: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "T2mEmbeddingClusterer only supports positions."
        )

    def _positions_to_features(self, positions: np.ndarray) -> np.ndarray:
        movement_enc, motion_enc, device, t2m_mean, t2m_std = _get_t2m_encoder()

        # pylint: disable=import-outside-toplevel
        import torch

        n_frames = positions.shape[1]
        t4 = min((n_frames - 1) // _UNIT_LENGTH * _UNIT_LENGTH, _MAX_MOTION_LENGTH)
        features = np.stack(
            [
                positions_to_hml263(sample, t2m_mean, t2m_std)[:t4]
                for sample in positions
            ]
        )
        motions = torch.from_numpy(features).float().to(device)
        m_lens = torch.full((motions.shape[0],), t4, dtype=torch.long)
        with torch.no_grad():
            movements = movement_enc(motions[..., :-4])
            embeddings = motion_enc(movements, m_lens // _UNIT_LENGTH)
        return embeddings.cpu().numpy().astype(np.float64)

    def _fit_predict(self, features: np.ndarray) -> np.ndarray:
        return agglomerative_labels(features, self._n_clusters)
