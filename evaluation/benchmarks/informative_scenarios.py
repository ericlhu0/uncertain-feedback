"""Select reaches and fixed or sampled preferences by their oracle corrections."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from evaluation.benchmarks.sampled_bounds import sample_bound

from uncertain_feedback.data_collection.dataset_auto_correction.clips import (
    arm_positions,
    sample_arm_q,
    wrist_goal,
)
from uncertain_feedback.planners.mpc.arm_features import arm_feature_series
from uncertain_feedback.planners.mpc.config import load_mpc_config
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    MpcCostContext,
    build_extra_costs,
)
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK
from uncertain_feedback.planners.mpc.rollout import goal_reach, rollout_to_goal
from uncertain_feedback.planners.rig import cfg_with_goal
from uncertain_feedback.simulated_users import HiddenCostTerm, get_persona
from uncertain_feedback.simulated_users.base import compute_violations
from uncertain_feedback.simulated_users.personas import UNRESTRICTED


@dataclass(frozen=True)
class ScenarioCriteria:
    window: int = 40
    min_history: int = 8
    min_nominal_violation: float = 0.15
    max_oracle_violation: float = 0.02
    min_position_rms: float = 0.04
    min_feature_rms: float = 0.15
    min_oracle_motion: float = 0.05


def generate_scenarios(
    geometry_dir: Path,
    config_path: Path,
    out_dir: Path,
    personas: list[str],
    seed: int,
    max_attempts: int,
    criteria: ScenarioCriteria,
    sampled_bounds: int = 0,
) -> None:
    """Keep the first passing reach per fixed persona or synthetic case slot."""
    out_dir.mkdir(parents=True, exist_ok=False)
    geo = np.load(geometry_dir / "geometry.npz")
    manifest = json.loads((geometry_dir / "manifest.json").read_text())
    fk = SmplLeftArmFK()
    fk.collar_aa = geo["collar_aa"]
    cfg = replace(load_mpc_config(config_path), seed=seed, max_angle_delta=0.0025)
    context = MpcCostContext(
        fk=fk,
        spine3_pos=geo["spine3_pos"],
        spine3_aa=geo["spine3_aa"],
        time_of_day=cfg.simulated_user.time_of_day,
    )
    body = geo["body_pos"]
    np.savez(out_dir / "geometry.npz", **{key: geo[key] for key in geo.files})
    base = CompositeTrajectoryCost(
        [*build_extra_costs(cfg.costs, context).terms(), UNRESTRICTED.limit_cost()]
    )
    audit: dict[str, object] = {
        "seed": seed,
        "criteria": asdict(criteria),
        "config": str(config_path.resolve()),
        "max_angle_delta": cfg.max_angle_delta,
        "personas": (
            [] if sampled_bounds else [asdict(get_persona(name)) for name in personas]
        ),
    }
    rows: list[dict[str, object]] = []
    audit["attempts"] = rows
    names = (
        [f"sampled_{index:03d}" for index in range(sampled_bounds)]
        if sampled_bounds
        else personas
    )
    audit["sampled_bounds"] = sampled_bounds
    for persona_index, name in enumerate(names):
        rng = np.random.default_rng([seed, persona_index])
        for attempt in range(max_attempts):
            user = UNRESTRICTED if sampled_bounds else get_persona(name)
            row: dict[str, object] = {"persona": name, "attempt": attempt}
            rows.append(row)
            q0 = sample_arm_q(rng, np.asarray(manifest["clavicle"]), context)
            q_goal = sample_arm_q(rng, np.asarray(manifest["clavicle"]), context)
            goal = wrist_goal(q_goal, context)
            if np.max(compute_violations(user, context, np.stack([q0, q_goal]))) > 0:
                row["rejected"] = "uncomfortable_endpoint"
                continue
            if np.linalg.norm(goal - wrist_goal(q0, context)) < 0.25:
                row["rejected"] = "short_reach"
                continue
            goal_cfg = cfg_with_goal(cfg, goal)

            def rollout(
                start: np.ndarray, extra: CompositeTrajectoryCost
            ) -> np.ndarray:
                return rollout_to_goal(
                    goal_cfg,
                    start,
                    goal,
                    context,
                    extra,
                    body,
                    context.spine3_pos,
                    context.spine3_aa,
                    steps=600,
                )

            naive = rollout(q0, base)
            if sampled_bounds:
                user = sample_bound(
                    rng,
                    naive,
                    q_goal,
                    context,
                    name,
                    "constant" if persona_index % 2 == 0 else "coupled",
                    criteria.min_history,
                    criteria.window,
                    criteria.min_nominal_violation,
                )
                if user is None:
                    row["rejected"] = "no_separating_bound"
                    continue
                row["user"] = asdict(user)
            # Joint-box costs already occur in the shared base stack.
            costs = CompositeTrajectoryCost(
                [
                    *base.terms(),
                    HiddenCostTerm(
                        user=replace(user, joint_limits=()), context=context
                    ),
                ]
            )
            violations = compute_violations(user, context, naive)
            crossing = np.flatnonzero(violations > 1e-8)
            if not goal_reach(context, cfg, naive, goal)["reached"] or not len(
                crossing
            ):
                row["rejected"] = "no_reached_violating_reach"
                continue
            trigger = int(crossing[0]) - 1
            if (
                trigger < criteria.min_history
                or len(naive) - trigger <= criteria.window
            ):
                row["rejected"] = "insufficient_history_or_future"
                continue
            nominal_full = rollout(naive[trigger], base)
            if (
                len(nominal_full) <= criteria.window
                or not goal_reach(context, cfg, nominal_full, goal)["reached"]
            ):
                row["rejected"] = "unusable_nominal_restart"
                continue
            nominal = nominal_full[: criteria.window + 1]
            nominal_peak = float(compute_violations(user, context, nominal).max())
            if nominal_peak < criteria.min_nominal_violation:
                row["rejected"] = "weak_nominal_violation"
                continue
            oracle = rollout(naive[trigger], costs)
            window = oracle[np.minimum(np.arange(criteria.window + 1), len(oracle) - 1)]
            nom_pos = arm_positions(nominal, fk, context.spine3_pos, context.spine3_aa)[
                :, -2:
            ]
            oracle_pos = arm_positions(
                window, fk, context.spine3_pos, context.spine3_aa
            )[:, -2:]
            position_rms = float(
                np.sqrt(np.mean(np.sum((nom_pos - oracle_pos) ** 2, axis=-1)))
            )
            nom_feats = arm_feature_series(nominal, context)
            ora_feats = arm_feature_series(window, context)
            features = {bound.feature for bound in user.bounds}
            features.update(
                bound.cond_feature
                for bound in user.bounds
                if hasattr(bound, "cond_feature")
            )
            feature_rms = max(
                float(np.sqrt(np.mean((nom_feats[key] - ora_feats[key]) ** 2)))
                for key in features
            )
            oracle_peak = float(compute_violations(user, context, oracle).max())
            motion = float(
                np.linalg.norm(oracle_pos[-1] - oracle_pos[0], axis=-1).max()
            )
            reach = goal_reach(context, cfg, oracle, goal)
            row.update(
                trigger=trigger,
                nominal_peak=nominal_peak,
                oracle_peak=oracle_peak,
                position_rms_m=position_rms,
                feature_rms_rad=feature_rms,
                oracle_motion_m=motion,
                oracle_reached=bool(reach["reached"]),
                final_distance_m=float(reach["distance"]),
            )
            failed = []
            if not reach["reached"]:
                failed.append("oracle_did_not_reach")
            if oracle_peak > criteria.max_oracle_violation:
                failed.append("oracle_violation")
            if (
                position_rms < criteria.min_position_rms
                or feature_rms < criteria.min_feature_rms
            ):
                failed.append("weak_contrast")
            if motion < criteria.min_oracle_motion:
                failed.append("oracle_stalls")
            row["rejected"] = ",".join(failed)
            print(f"[scenario] {name} {attempt}: {row}", flush=True)
            (out_dir / "selection.json").write_text(json.dumps(audit, indent=2))
            if failed:
                continue
            np.savez(
                out_dir / f"{name}.npz",
                naive=naive,
                oracle=oracle,
                nominal=nominal,
                nominal_full=nominal_full,
                correction=window,
                goal=goal,
                q_goal=q_goal,
                trigger=trigger,
            )
            break
        (out_dir / "selection.json").write_text(json.dumps(audit, indent=2))


def render_scenarios(out_dir: Path) -> None:
    """SMPL movies share the history, trigger, camera, and physical frame rate."""
    import subprocess

    import imageio_ffmpeg
    from uncertain_feedback.planners.mpc.kinematics import q_to_arm_aa
    from uncertain_feedback.utils.mesh_video import (
        MeshLayer,
        arm_mesh_vertices,
        goal_layer,
        render_layers,
    )

    geo = np.load(out_dir / "geometry.npz")
    fk = SmplLeftArmFK()
    fk.collar_aa = geo["collar_aa"]
    for path in sorted(out_dir.glob("*.npz")):
        if path.name == "geometry.npz":
            continue
        data = np.load(path)
        trigger = int(data["trigger"])
        history = data["naive"][max(0, trigger - 20) : trigger]
        futures = {
            "nominal": data["nominal"],
            "oracle": data["correction"],
            "oracle_full": data["oracle"],
        }
        meshes = {}
        for name, future in futures.items():
            q = np.concatenate([history, future])
            vertices, faces = arm_mesh_vertices(
                q_to_arm_aa(q, fk.elbow_hinge_axis),
                geo["body_pos"],
                fk,
                geo["spine3_aa"],
            )
            meshes[name] = vertices
        flat = np.concatenate([v.reshape(-1, 3) for v in meshes.values()])
        bounds = np.stack([flat.min(0), flat.max(0)])
        bounds[1, 1] += 0.12
        for name, vertices in meshes.items():
            render_layers(
                (
                    MeshLayer(vertices, faces),
                    goal_layer(data["goal"] + geo["spine3_pos"], len(vertices)),
                ),
                out_dir / f"{path.stem}_{name}.mp4",
                resolution=480,
                bounds=bounds,
                caption=f"{name.upper()} | preference boundary",
                caption_from=len(history),
                holds={len(history): 1.0, len(vertices) - 1: 1.0},
            )
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(out_dir / f"{path.stem}_nominal.mp4"),
                "-i",
                str(out_dir / f"{path.stem}_oracle.mp4"),
                "-filter_complex",
                "hstack=inputs=2",
                str(out_dir / f"{path.stem}_comparison.mp4"),
            ],
            check=True,
        )
