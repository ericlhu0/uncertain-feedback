"""How well an MDM checkpoint still models HumanML3D, MDM's pretraining distribution.

MDM's protocol with the T2M text-motion evaluator every HumanML3D paper reports:
one whole-motion caption per test motion of the real test split
(``consts.HUMANML3D_DIR``), generated at that motion's own length, then R-precision,
matching score, FID against the real motions, diversity and multimodality. Two body
diagnostics name the failure a correction fine-tune drifts into: the fraction of
samples with the pelvis low (sitting, but also crawling or kneeling in real data), and
how far the root travels on walking prompts. Unlike MDM's ``eval_humanml.py``: no
sub-segment captions, no random length shortening or crop, generated frames past the
length zeroed like the ground truth's, all test motions once rather than 20 x 1000,
and 20 multimodality repeats.

Absolute numbers sit ~0.05 R@1 below published ones for ground truth and models alike,
while the ground-truth-to-base-MDM gap matches the published one; the cause is
unverified (MDM re-served its evaluator download in 2025-01 and the original link is
dead). Compare checkpoints with each other and with the ground-truth row, not with
papers.
"""

from __future__ import annotations

import sys
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from uncertain_feedback.consts import HUMANML3D_DIR, MDM_ROOT, T2M_EVALUATOR_DIR

_MDM_DIR = MDM_ROOT / "motion-diffusion-model"
if str(_MDM_DIR) not in sys.path:
    sys.path.insert(0, str(_MDM_DIR))

# pylint: disable=wrong-import-position,import-error,wrong-import-order
from data_loaders.humanml.networks.modules import (  # noqa: E402
    MotionEncoderBiGRUCo,
    MovementConvEncoder,
    TextEncoderBiGRUCo,
)
from data_loaders.humanml.scripts.motion_process import recover_from_ric  # noqa: E402
from data_loaders.humanml.utils import paramUtil  # noqa: E402
from data_loaders.humanml.utils.metrics import (  # noqa: E402
    calculate_activation_statistics,
    calculate_diversity,
    calculate_frechet_distance,
    calculate_multimodality,
    calculate_top_k,
    euclidean_distance_matrix,
)
from data_loaders.humanml.utils.word_vectorizer import (  # noqa: E402
    POS_enumerator,
    WordVectorizer,
)

MAX_FRAMES = 196
UNIT = 4
# A standing HumanML3D pelvis sits ~0.95 m up; the correction clips' seated body 0.60 m.
SEATED_PELVIS_Y = 0.75
LOCOMOTION_WORDS = ("walk", "run", "jog")
SHOWCASE = (
    "a person walks forward with his arms at his side slowly.",
    "man jumps twice in place",
    "a person slowly kicks with their right leg.",
    "a person waves with his right hand.",
    "a man sits down and then stays still.",
    "a person is doing jumping jacks.",
    "person raises left arm out to the side.",
)


@dataclass(frozen=True)
class TestSet:
    """HumanML3D test motions, one caption each, ready for the evaluator."""

    captions: list[str]
    lengths: np.ndarray  # (N,) frames, multiples of 4
    motions: np.ndarray  # (N, 196, 263) raw HML263
    word_embs: np.ndarray  # (N, 22, 300)
    pos_ohot: np.ndarray  # (N, 22, 15)
    cap_lens: np.ndarray  # (N,)


def load_test_set(limit: int | None = None, seed: int = 0) -> TestSet:
    """The 40-199-frame test motions MDM's loader keeps, each with a random caption."""
    rng = np.random.default_rng(seed)
    w_vec = WordVectorizer(str(_MDM_DIR / "glove"), "our_vab")
    captions, lengths, motions, words, poss, cap_lens = [], [], [], [], [], []
    for i in (HUMANML3D_DIR / "test.txt").read_text().split()[:limit]:
        motion = np.load(HUMANML3D_DIR / "new_joint_vecs" / f"{i}.npy")
        if not 40 <= len(motion) < 200:
            continue
        lines = [
            line.split("#")
            for line in (HUMANML3D_DIR / "texts" / f"{i}.txt").read_text().splitlines()
            if line.strip()
        ]
        whole = [line for line in lines if float(line[2]) == 0 and float(line[3]) == 0]
        caption, tokens = (whole or lines)[rng.integers(len(whole or lines))][:2]
        n = min(len(motion), MAX_FRAMES) // UNIT * UNIT
        padded = np.zeros((MAX_FRAMES, 263), np.float32)
        padded[:n] = motion[:n]
        token_list = ["sos/OTHER"] + tokens.split(" ")[:20] + ["eos/OTHER"]
        cap_lens.append(len(token_list))
        token_list += ["unk/OTHER"] * (22 - len(token_list))
        w, p = zip(
            *(w_vec[t if t.count("/") == 1 else "unk/OTHER"] for t in token_list)
        )
        captions.append(caption)
        lengths.append(n)
        motions.append(padded)
        words.append(np.stack(w))
        poss.append(np.stack(p))
    return TestSet(
        captions,
        np.array(lengths),
        np.stack(motions),
        np.stack(words).astype(np.float32),
        np.stack(poss).astype(np.float32),
        np.array(cap_lens),
    )


class T2MEvaluator:
    """The T2M text and motion encoders, loaded from ``consts.T2M_EVALUATOR_DIR``.

    cuDNN stays off around the GRUs: the host's LD_LIBRARY_PATH carries an older cuDNN
    than torch's, which fails torch's version check, and these GRUs are its only users.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        ckpt = torch.load(
            T2M_EVALUATOR_DIR / "text_mot_match" / "model" / "finest.tar",
            map_location=device,
            weights_only=True,
        )
        self.movement = MovementConvEncoder(263 - 4, 512, 512)
        self.text = TextEncoderBiGRUCo(300, len(POS_enumerator), 512, 512, device)
        self.motion = MotionEncoderBiGRUCo(512, 1024, 512, device)
        self.movement.load_state_dict(ckpt["movement_encoder"])
        self.text.load_state_dict(ckpt["text_encoder"])
        self.motion.load_state_dict(ckpt["motion_encoder"])
        with torch.backends.cudnn.flags(enabled=False):
            for module in (self.movement, self.text, self.motion):
                module.to(device).eval()
        meta = T2M_EVALUATOR_DIR / "Comp_v6_KLD01" / "meta"
        self.mean, self.std = np.load(meta / "mean.npy"), np.load(meta / "std.npy")

    def _sorted(
        self, fn: Callable[..., torch.Tensor], lens: np.ndarray, *arrays: np.ndarray
    ) -> np.ndarray:
        """Packed GRUs need length-sorted batches; results come back in input order."""
        out = np.empty((len(lens), 512), np.float32)
        for s in range(0, len(lens), 256):
            idx = np.arange(s, min(s + 256, len(lens)))
            idx = idx[np.argsort(-lens[idx], kind="stable")]
            tensors = [torch.from_numpy(a[idx]).float().to(self.device) for a in arrays]
            with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
                out[idx] = fn(*tensors, torch.from_numpy(lens[idx])).cpu().numpy()
        return out

    def text_embeddings(self, test: TestSet) -> np.ndarray:
        """One embedding per test caption."""
        return self._sorted(self.text, test.cap_lens, test.word_embs, test.pos_ohot)

    def motion_embeddings(self, raw: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """``raw`` un-normalized; frames past each length are zeroed like MDM's GT loader."""
        normed = ((raw - self.mean) / self.std).astype(np.float32)
        for motion, n in zip(normed, lengths):
            motion[n:] = 0.0
        return self._sorted(
            lambda m, n: self.motion(self.movement(m[..., :-4]), n // UNIT),
            lengths,
            normed,
        )


def body_usage(raw: np.ndarray, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean pelvis height and horizontal root path length of each motion."""
    pelvis_y, travel = [], []
    for s in range(0, len(raw), 256):
        positions = recover_from_ric(
            torch.from_numpy(raw[s : s + 256]).float(), 22
        ).numpy()
        for p, n in zip(positions, lengths[s : s + 256]):
            pelvis_y.append(p[:n, 0, 1].mean())
            travel.append(
                np.linalg.norm(np.diff(p[:n, 0, [0, 2]], axis=0), axis=-1).sum()
            )
    return np.array(pelvis_y), np.array(travel)


def prior_metrics(
    test: TestSet,
    text_emb: np.ndarray,
    gt_emb: np.ndarray,
    emb: np.ndarray,
    pelvis_y: np.ndarray,
    travel: np.ndarray,
    mm_emb: np.ndarray | None = None,
) -> dict[str, float]:
    """MDM's metrics for one set of samples paired with ``test``'s captions."""
    rng = np.random.default_rng(0)
    top_k, matching = np.zeros(3), 0.0
    n_batches = 10 * (len(emb) // 32)
    for _ in range(10):
        idx = rng.permutation(len(emb))
        for s in range(0, len(emb) // 32 * 32, 32):
            dist = euclidean_distance_matrix(
                text_emb[idx[s : s + 32]], emb[idx[s : s + 32]]
            )
            matching += dist.trace() / 32
            top_k += calculate_top_k(np.argsort(dist, axis=1), 3).mean(axis=0)
    np.random.seed(0)
    locomotion = np.array(
        [any(w in c.lower() for w in LOCOMOTION_WORDS) for c in test.captions]
    )
    metrics = {
        "R@1": top_k[0] / n_batches,
        "R@2": top_k[1] / n_batches,
        "R@3": top_k[2] / n_batches,
        "matching": matching / n_batches,
        # The ground-truth row is 0 by definition; computing it takes sqrtm of a covariance
        # times itself, which some GPUs' embeddings leave complex enough to raise.
        "FID": (
            0.0
            if emb is gt_emb
            else calculate_frechet_distance(
                *calculate_activation_statistics(gt_emb),
                *calculate_activation_statistics(emb),
            )
        ),
        "diversity": float(
            np.mean(
                [calculate_diversity(emb, min(300, len(emb) - 1)) for _ in range(10)]
            )
        ),
        "seated": float(np.mean(pelvis_y < SEATED_PELVIS_Y)),
        "walk_travel_m": float(np.median(travel[locomotion])),
    }
    if mm_emb is not None:
        metrics["multimodality"] = float(
            np.mean([calculate_multimodality(mm_emb, 10) for _ in range(10)])
        )
    return {k: float(v) for k, v in metrics.items()}


def render_showcase(samples: dict[str, np.ndarray], n_frames: int, path: Path) -> None:
    """Front and side keyframes of every model's ``SHOWCASE`` samples, side by side."""
    # pylint: disable=import-outside-toplevel
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(samples)
    positions = {
        n: recover_from_ric(torch.from_numpy(raw).float(), 22).numpy()[:, :n_frames]
        for n, raw in samples.items()
    }
    fig, axes = plt.subplots(
        len(SHOWCASE), 2 * len(names), figsize=(3.6 * len(names), 2.3 * len(SHOWCASE))
    )
    for row, caption in enumerate(SHOWCASE):
        for col, name in enumerate(names):
            p = positions[name][row]
            for view, (h, view_name) in enumerate(((0, "front"), (2, "side"))):
                ax = axes[row, 2 * col + view]
                for k, t in enumerate(np.linspace(0, n_frames - 1, 6).astype(int)):
                    for chain in paramUtil.t2m_kinematic_chain:
                        arm = chain[-1] == 20
                        ax.plot(
                            p[t, chain, h],
                            p[t, chain, 1],
                            color="#dd6b20" if arm else "#4a5568",
                            lw=2.0 if arm else 1.3,
                            alpha=0.25 + 0.75 * (k + 1) / 6,
                        )
                ax.plot(p[:, 0, h], p[:, 0, 1], color="k", lw=0.7, ls=":")
                ax.axhline(0, color="0.6", lw=0.8)
                ax.set_aspect("equal")
                ax.set_xlim(p[..., h].min() - 0.3, p[..., h].max() + 0.3)
                ax.set_ylim(-0.1, 2.0)
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(f"{name} — {view_name}", fontsize=9)
        axes[row, 0].set_ylabel("\n".join(textwrap.wrap(caption, 26)), fontsize=8)
    title = (
        "HumanML3D prompts, 6 frames each (faint → solid); orange = left arm, dotted = "
        "pelvis path; the front view shows the person's left side on the right"
    )
    fig.suptitle("\n".join(textwrap.wrap(title, 40 * len(names))), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=100)
    plt.close(fig)
