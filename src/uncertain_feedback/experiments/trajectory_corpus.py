"""Per-session on-disk corpus of executed arm trajectories.

Each logged trajectory is saved as a raw ``.npy`` plus a per-frame joint-feature
``.csv`` and recorded in ``manifest.json``. Codex-based cost generators read this
corpus to reason about which configurations were reached comfortably (executed
without a discomfort report) in past rounds.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from uncertain_feedback.planners.mpc.costs.base import MpcCostContext
from uncertain_feedback.simulated_users.base import FEATURE_NAMES, feature_series


@dataclass(frozen=True)
class TrajectoryCorpus:
    """On-disk log of executed and corrected trajectories for one session."""

    dir: Path
    context: MpcCostContext

    @classmethod
    def create(cls, corpus_dir: Path, context: MpcCostContext) -> "TrajectoryCorpus":
        """Open (creating if needed) the corpus rooted at ``corpus_dir``."""
        corpus_dir.mkdir(parents=True, exist_ok=True)
        manifest = corpus_dir / "manifest.json"
        if not manifest.exists():
            manifest.write_text("[]")
        return cls(dir=corpus_dir, context=context)

    def entries(self) -> list[dict[str, Any]]:
        """Every logged entry, in insertion order."""
        return json.loads((self.dir / "manifest.json").read_text())

    def log(
        self,
        trajectory: np.ndarray,
        *,
        kind: str,
        round_index: int,
        goal: tuple[float, float, float] | None,
        trigger_step: int | None,
        trigger_violation: float | None,
        feedback_text: str | None,
    ) -> dict[str, Any]:
        """Save a trajectory with its features and return the new manifest entry."""
        entries = self.entries()
        index = max((e["index"] for e in entries), default=-1) + 1

        trajectory = np.asarray(trajectory, dtype=np.float64)
        np.save(self.dir / f"traj_{index:03d}.npy", trajectory)

        feats = feature_series(self.context, trajectory)
        n_frames = int(trajectory.shape[0])
        with (self.dir / f"traj_{index:03d}_features.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", *FEATURE_NAMES])
            for frame in range(n_frames):
                writer.writerow(
                    [frame, *(feats[name][frame] for name in FEATURE_NAMES)]
                )

        entry = {
            "index": index,
            "kind": kind,
            "round": round_index,
            "goal": list(goal) if goal is not None else None,
            "n_frames": n_frames,
            "trigger_step": trigger_step,
            "trigger_violation": trigger_violation,
            "feedback_text": feedback_text,
            "comfortable_until": trigger_step if trigger_step is not None else n_frames,
            "traj_file": f"traj_{index:03d}.npy",
            "features_file": f"traj_{index:03d}_features.csv",
        }
        entries.append(entry)
        (self.dir / "manifest.json").write_text(json.dumps(entries, indent=2))
        return entry

    def remove(self, index: int) -> None:
        """Drop the entry with the given index and its artifacts."""
        entries = self.entries()
        kept = [e for e in entries if e["index"] != index]
        (self.dir / "manifest.json").write_text(json.dumps(kept, indent=2))
        (self.dir / f"traj_{index:03d}.npy").unlink(missing_ok=True)
        (self.dir / f"traj_{index:03d}_features.csv").unlink(missing_ok=True)
