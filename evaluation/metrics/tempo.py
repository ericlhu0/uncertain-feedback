"""Tempo of generated motions, measured as the planner would execute them.

Every sample is a ``(T, 3)`` spine3-relative wrist path after re-anchoring on
the feedback pose, so frame 0 is the arm's current position and the frame-0
seam is gone. The primary measure is ``t90``, the first frame at which the
wrist's displacement from its start reaches 90% of the sample's largest
displacement over the window: a scaled-down copy of a motion has the same t90,
so a smaller motion cannot pass as slower; a motion that finishes early and
holds has a small t90, one still moving at the end a t90 near the window's end.
Alongside, not tested: t50, the net displacement ``D`` and its direction, the
peak 5-frame wrist speed, and the ``rush``, the mean wrist speed over the first
five generated steps, which is where the pinned prefix's catch-up shows.

The pre-registered criteria (``outputs/speed_replay/PROTOCOL.md``, "Speed
test"), each a mean or median over cases, the means with a 90% case-bootstrap
CI: S1 tempo, S2 both words act, S3 direction kept, S4 not merely smaller.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

T90_FRACTION = 0.9
T50_FRACTION = 0.5
SPEED_WINDOW = 5
RUSH_STEPS = 5

S1_MIN_GAP_FRAMES = 8
S3_MIN_COSINE = 0.7
S4_MIN_RATIO = 0.3
N_BOOT = 10000
CI_LEVEL = 0.90


def displacement(wrist: np.ndarray) -> np.ndarray:
    """``(N, T)`` distance of each frame's wrist from the sample's first frame."""
    wrist = np.asarray(wrist, dtype=np.float64)
    return np.linalg.norm(wrist - wrist[:, :1], axis=-1)


def time_to_fraction(wrist: np.ndarray, fraction: float) -> np.ndarray:
    """``(N,)`` first frame at which each sample reaches ``fraction`` of its largest displacement."""
    disp = displacement(wrist)
    return (disp >= fraction * disp.max(axis=1, keepdims=True)).argmax(axis=1)


def net_vector(wrist: np.ndarray) -> np.ndarray:
    """``(N, 3)`` wrist displacement from the first frame to the last."""
    wrist = np.asarray(wrist, dtype=np.float64)
    return wrist[:, -1] - wrist[:, 0]


def peak_speed(wrist: np.ndarray) -> np.ndarray:
    """``(N,)`` largest wrist speed over ``SPEED_WINDOW``-frame steps, in m/frame."""
    wrist = np.asarray(wrist, dtype=np.float64)
    steps = wrist[:, SPEED_WINDOW:] - wrist[:, :-SPEED_WINDOW]
    return (np.linalg.norm(steps, axis=-1) / SPEED_WINDOW).max(axis=1)


def rush(wrist: np.ndarray) -> np.ndarray:
    """``(N,)`` mean wrist speed over the first ``RUSH_STEPS`` generated steps, in m/frame."""
    wrist = np.asarray(wrist, dtype=np.float64)
    steps = np.diff(wrist[:, : RUSH_STEPS + 1], axis=1)
    return np.linalg.norm(steps, axis=-1).mean(axis=1)


def sample_stats(wrist: np.ndarray) -> dict[str, Any]:
    """One prompt's samples summarized: means over samples, plus the mean net vector."""
    v = net_vector(wrist)
    return {
        "t90": float(time_to_fraction(wrist, T90_FRACTION).mean()),
        "t50": float(time_to_fraction(wrist, T50_FRACTION).mean()),
        "D": float(np.linalg.norm(v, axis=-1).mean()),
        "peak": float(peak_speed(wrist).mean()),
        "rush": float(rush(wrist).mean()),
        "v": v.mean(axis=0),
    }


def condition_stats(prompts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A condition's stats: the mean over its prompts of each prompt's stats."""
    return {
        key: (
            np.mean([p["v"] for p in prompts], axis=0)
            if key == "v"
            else float(np.mean([p[key] for p in prompts]))
        )
        for key in ("t90", "t50", "D", "peak", "rush", "v")
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine between two vectors, 0 when either is zero."""
    norms = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / norms) if norms > 0 else 0.0


def case_row(
    unmarked: Mapping[str, Any], slow: Mapping[str, Any], fast: Mapping[str, Any]
) -> dict[str, float]:
    """The per-case numbers the criteria read, from the three conditions' stats."""
    row = {
        f"{key}_{name}": float(stats[key])
        for name, stats in (("unmarked", unmarked), ("slow", slow), ("fast", fast))
        for key in ("t90", "t50", "D", "peak", "rush")
    }
    row["cos_slow"] = cosine(slow["v"], unmarked["v"])
    row["cos_fast"] = cosine(fast["v"], unmarked["v"])
    row["d_ratio_slow"] = (
        float(slow["D"] / unmarked["D"]) if unmarked["D"] > 0 else float("nan")
    )
    return row


def case_bootstrap(
    values: np.ndarray, n_boot: int = N_BOOT, seed: int = 0, level: float = CI_LEVEL
) -> dict[str, float]:
    """Mean of per-case ``values`` with a percentile bootstrap CI over cases."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[draws].mean(axis=1)
    tail = (1 - level) / 2
    return {
        "mean": float(values.mean()),
        "lo": float(np.quantile(means, tail)),
        "hi": float(np.quantile(means, 1 - tail)),
    }


def speed_verdict(rows: Sequence[Mapping[str, float]], seed: int = 0) -> dict[str, Any]:
    """S1-S4 over the cases' rows (:func:`case_row`), with the numbers behind each."""

    def col(name: str) -> np.ndarray:
        return np.array([r[name] for r in rows], dtype=np.float64)

    s1 = case_bootstrap(col("t90_slow") - col("t90_fast"), seed=seed)
    s2_slow = case_bootstrap(col("t90_slow") - col("t90_unmarked"), seed=seed)
    s2_fast = case_bootstrap(col("t90_unmarked") - col("t90_fast"), seed=seed)
    cos_slow, cos_fast = float(np.median(col("cos_slow"))), float(
        np.median(col("cos_fast"))
    )
    d_ratio = float(np.nanmedian(col("d_ratio_slow")))
    verdict = {
        "S1": s1["mean"] >= S1_MIN_GAP_FRAMES and s1["lo"] > 0,
        "S2": s2_slow["lo"] > 0 and s2_fast["lo"] > 0,
        "S3": min(cos_slow, cos_fast) >= S3_MIN_COSINE,
        "S4": d_ratio >= S4_MIN_RATIO,
    }
    return {
        **verdict,
        "pass": all(verdict.values()),
        "n_cases": len(rows),
        "s1_gap": s1,
        "s2_slow_vs_unmarked": s2_slow,
        "s2_unmarked_vs_fast": s2_fast,
        "s3_cos_slow": cos_slow,
        "s3_cos_fast": cos_fast,
        "s4_d_ratio_slow": d_ratio,
        "means": {
            name: float(np.mean(col(name)))
            for name in rows[0]
            if name.split("_")[0] in ("t90", "t50", "D", "peak", "rush")
        },
    }


def verdict_table(verdict: Mapping[str, Any]) -> str:
    """The verdict as a few readable lines."""
    s1, s2a, s2b = (
        verdict["s1_gap"],
        verdict["s2_slow_vs_unmarked"],
        verdict["s2_unmarked_vs_fast"],
    )
    means = verdict["means"]
    lines = [
        f"S1 tempo: t90(slow) - t90(fast) = {s1['mean']:.2f} frames "
        f"[{s1['lo']:.2f}, {s1['hi']:.2f}] (>= {S1_MIN_GAP_FRAMES}, CI > 0): "
        f"{'PASS' if verdict['S1'] else 'fail'}",
        f"S2 both act: slow - unmarked {s2a['mean']:.2f} [{s2a['lo']:.2f}, {s2a['hi']:.2f}], "
        f"unmarked - fast {s2b['mean']:.2f} [{s2b['lo']:.2f}, {s2b['hi']:.2f}]: "
        f"{'PASS' if verdict['S2'] else 'fail'}",
        f"S3 direction: median cos slow {verdict['s3_cos_slow']:.2f}, fast "
        f"{verdict['s3_cos_fast']:.2f} (>= {S3_MIN_COSINE}): "
        f"{'PASS' if verdict['S3'] else 'fail'}",
        f"S4 not smaller: median D(slow)/D(unmarked) {verdict['s4_d_ratio_slow']:.2f} "
        f"(>= {S4_MIN_RATIO}): {'PASS' if verdict['S4'] else 'fail'}",
        f"{'PASS' if verdict['pass'] else 'FAIL'} on {verdict['n_cases']} cases",
        "condition   t90    t50     D      peak    rush",
    ]
    for name in ("unmarked", "slow", "fast"):
        lines.append(
            f"{name:9s} {means[f't90_{name}']:6.2f} {means[f't50_{name}']:6.2f} "
            f"{means[f'D_{name}']:7.4f} {means[f'peak_{name}']:7.4f} "
            f"{means[f'rush_{name}']:7.4f}"
        )
    return "\n".join(lines)
