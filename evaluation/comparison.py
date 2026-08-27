"""Compare several grounding methods on one scenario: a table and one video.

Every approach plays the same episode from the same rig, so the nominal rollout
and the trigger step are identical across arms and the only difference is how
the utterance became motion. :func:`render_comparison` crops the Front (XY)
panel out of each arm's :meth:`ArmVisualizer.render_rollout_video` output and
tiles them, so all cells share one set of axes.

Colour key in every panel: ``royalblue`` planned motion the persona tolerates,
``red`` frames violating their hidden bounds, ``darkorange`` the correction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from evaluation.approaches.base import Approach
from evaluation.episode import run_episode
from evaluation.structs import InteractionTask
from uncertain_feedback.planners.mpc.kinematics import q_to_arm_aa
from uncertain_feedback.planners.rig import PlanningRig
from uncertain_feedback.simulated_users import SimulatedUser, first_violation_step
from uncertain_feedback.utils.plot import ArmVisualizer

_CROP = (slice(460, 912), slice(270, 650))
_LABEL_BAR = 46
_BLUE, _RED, _ORANGE = "royalblue", "red", "darkorange"

TABLE_COLUMNS = (
    "approach",
    "utterance_text",
    "n_candidates",
    "n_acceptable",
    "chosen_label",
    "magnitude",
    "correction_alignment",
    "retrigger_step",
    "continuation_mean_violation",
    "ground_seconds",
)


def run_arms(
    rig: PlanningRig,
    user: SimulatedUser,
    task: InteractionTask,
    approaches: Sequence[Approach],
    out_dir: Path,
) -> pd.DataFrame:
    """Play ``task`` through every approach, one episode directory each."""
    rows: list[dict] = []
    for approach in approaches:
        episode_dir = out_dir / approach.name
        approach.reset(rig, user, task, episode_dir)
        result = run_episode(rig, user, task, approach, episode_dir)
        rows.extend({"approach": approach.name, **row} for row in result["rows"])
    return pd.DataFrame(rows)


def comparison_table(rows: pd.DataFrame, round_index: int = 0) -> pd.DataFrame:
    """One line per approach for a single feedback round."""
    if rows.empty:
        return pd.DataFrame(columns=list(TABLE_COLUMNS))
    table = rows[rows["round_index"] == round_index]
    return table[list(TABLE_COLUMNS)].reset_index(drop=True)


def _method_frames(
    rig: PlanningRig,
    user: SimulatedUser,
    nominal: np.ndarray,
    trigger: int,
    round_dir: Path,
    threshold: float,
) -> tuple[np.ndarray, list[str]]:
    """The arm's whole story — approach, trigger, correction, continuation."""
    correction = np.load(round_dir / "correction.npy")
    continuation = np.load(round_dir / "continuation.npy")
    retrigger = first_violation_step(user, rig.context, continuation, threshold)
    tail = len(continuation) - 1
    kept = tail if retrigger is None else retrigger
    colors = (
        [_BLUE] * trigger
        + [_RED]
        + [_ORANGE] * (len(correction) - 1)
        + [_BLUE] * kept
        + [_RED] * (tail - kept)
    )
    states = np.concatenate([nominal[: trigger + 1], correction[1:], continuation[1:]])
    return states, colors


def render_comparison(
    rig: PlanningRig,
    user: SimulatedUser,
    goal: np.ndarray,
    out_dir: Path,
    approach_names: Sequence[str],
    round_index: int = 0,
    goal_index: int = 0,
    fps: int = 10,
) -> Path:
    """Render one video per approach plus the tiled side-by-side comparison."""
    import imageio.v2 as imageio  # pylint: disable=import-outside-toplevel
    from matplotlib import font_manager  # pylint: disable=import-outside-toplevel
    from PIL import (  # pylint: disable=import-outside-toplevel
        Image,
        ImageDraw,
        ImageFont,
    )

    viz = ArmVisualizer(fk=rig.fk)
    threshold = rig.cfg.corrections.trigger_threshold
    goal_dir = f"goal_{goal_index:02d}"
    round_name = f"round_{round_index:02d}"

    def render(states: np.ndarray, name: str, colors: list[str]) -> Path:
        path = out_dir / f"{name}.mp4"
        viz.render_rollout_video(
            q_to_arm_aa(states, rig.fk.elbow_hinge_axis),
            path,
            spine3_pos=rig.spine3_pos,
            spine3_aa=rig.spine3_aa,
            body_pos=rig.body_pos,
            cartesian_goal=goal,
            frame_colors=colors,
            fps=fps,
        )
        return path

    episodes = [
        (name, out_dir / name)
        for name in approach_names
        if (out_dir / name / goal_dir / round_name).is_dir()
    ]
    if not episodes:
        raise ValueError("No episode produced a feedback round to compare.")

    nominal = np.load(episodes[0][1] / goal_dir / "initial_rollout.npy")
    trigger = first_violation_step(user, rig.context, nominal, threshold)
    assert trigger is not None
    cells = [("nominal", "no correction (nominal)")]
    render(nominal, "nominal", [_BLUE] * trigger + [_RED] * (len(nominal) - trigger))

    for name, episode_dir in episodes:
        shared = np.load(episode_dir / goal_dir / "initial_rollout.npy")
        if not np.allclose(shared, nominal):
            raise ValueError(f"{name} planned a different nominal rollout.")
        states, colors = _method_frames(
            rig, user, nominal, trigger, episode_dir / goal_dir / round_name, threshold
        )
        render(states, name, colors)
        cells.append((name, name))

    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 22)
    clips = [
        np.stack(
            [
                frame[_CROP]
                for frame in imageio.mimread(out_dir / f"{n}.mp4", memtest=False)
            ]
        )
        for n, _ in cells
    ]
    height, width = clips[0].shape[1:3]
    frames = []
    for index in range(max(len(clip) for clip in clips)):
        row = []
        for clip, (_, label) in zip(clips, cells):
            cell = Image.new("RGB", (width, height + _LABEL_BAR), "white")
            cell.paste(
                Image.fromarray(clip[min(index, len(clip) - 1)]), (0, _LABEL_BAR)
            )
            held = "" if index < len(clip) else "  (ended)"
            ImageDraw.Draw(cell).text(
                (10, 12), f"{label}{held}", fill="black", font=font
            )
            row.append(np.asarray(cell))
        frames.append(np.concatenate(row, axis=1))

    path = out_dir / "comparison.mp4"
    imageio.mimsave(path, np.stack(frames), fps=fps)
    return path
