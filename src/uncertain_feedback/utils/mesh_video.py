"""Offscreen SMPL-mesh video of an executed arm trajectory.

The demo runner draws the same mesh in the browser (three.js); this renders it
headlessly so a finished ``executed_trajectory.npy`` from
:mod:`uncertain_feedback.planners.run` becomes an mp4.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# pyrender picks its GL backend at import time, and the compute nodes have no
# display; EGL renders against the GPU without one.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from uncertain_feedback import consts  # noqa: E402
from uncertain_feedback.motion_generators import make_motion_generator  # noqa: E402
from uncertain_feedback.planners.mpc.kinematics import (  # noqa: E402
    SmplLeftArmFK,
    q_to_arm_aa,
)
from uncertain_feedback.utils.smpl_mesh import SmplMeshCache  # noqa: E402

# The decoded body is Y-up and faces +Z (left shoulder at +X), so these are the
# person's own front and left-side views.
_VIEW_ROTATIONS = {
    "front": np.eye(3),
    "side": np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
}
_BODY_COLOR = (0.65, 0.68, 0.78, 1.0)
# Okabe-Ito: distinguishable from each other, from the neutral body, and in
# grayscale, which a paper figure eventually needs.
_CLUSTER_COLORS = (
    (0.00, 0.62, 0.45, 1.0),
    (0.90, 0.62, 0.00, 1.0),
    (0.34, 0.71, 0.91, 1.0),
    (0.80, 0.47, 0.65, 1.0),
    (0.75, 0.12, 0.15, 1.0),
    (0.35, 0.31, 0.55, 1.0),
)
# Alpha for the candidates that are not being tracked.
_GHOST_ALPHA = 0.5
# Marker for the Cartesian wrist goal: annotation, so it borrows the caption's
# charcoal rather than a colour a candidate could be confused with.
_GOAL_COLOR = (0.11, 0.12, 0.15, 1.0)
_GOAL_RADIUS = 0.03
_YFOV = np.pi / 4.0


@dataclass(frozen=True)
class MeshLayer:
    """One mesh sequence drawn over a window of the output video.

    Args:
        vertices:      ``(L, n_verts, 3)`` vertices, one entry per window frame.
        faces:         ``(n_faces, 3)`` triangle indices.
        color:         RGBA base colour, ignored when ``vertex_colors`` is set.
        start:         Video frame the window opens on; the layer is absent
                       before it and after ``start + L``.
        vertex_colors: ``(n_verts, 4)`` per-vertex RGBA, for tinting part of a
                       body (the left arm) without splitting it into a second
                       mesh that would z-fight against the first.
    """

    vertices: np.ndarray
    faces: np.ndarray
    color: tuple[float, float, float, float] = _BODY_COLOR
    start: int = 0
    vertex_colors: np.ndarray | None = None


def _mesh_frames(
    cache: SmplMeshCache,
    arm_aa: np.ndarray,
    fk: SmplLeftArmFK,
    spine3_pos: np.ndarray,
    spine3_aa: np.ndarray,
) -> np.ndarray:
    """``(T, n_verts, 3)`` mesh vertices for a ``(T, 3, 3)`` arm-angle sequence."""
    arm_pos = fk.fk_batch(np.asarray(arm_aa, dtype=np.float64), spine3_pos, spine3_aa)
    return cache.vertices(cache.register(arm_pos))


def arm_mesh_vertices(
    arm_aa: np.ndarray,
    body_pos: np.ndarray,
    fk: SmplLeftArmFK,
    spine3_aa: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame SMPL mesh for one arm-angle sequence.

    Args:
        arm_aa:    ``(T, 3, 3)`` ``[shoulder, elbow, wrist]`` axis-angles.
        body_pos:  ``(22, 3)`` decoded start-pose joint positions (torso fit).
        fk:        Arm FK with ``collar_aa`` already set from the same pose.
        spine3_aa: ``(3,)`` world axis-angle of spine3.

    Returns:
        ``(T, n_verts, 3)`` vertices in the body's world frame, and the
        ``(n_faces, 3)`` triangles they index.
    """
    cache = SmplMeshCache(body_pos, max_entries=1)
    return _mesh_frames(cache, arm_aa, fk, body_pos[9], spine3_aa), cache.faces


def _caption_font(height: int):  # type: ignore[no-untyped-def]
    """DejaVuSans at a size proportional to the frame, via matplotlib's copy."""
    # pylint: disable=import-outside-toplevel
    import matplotlib
    from PIL import ImageFont

    path = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    return ImageFont.truetype(str(path), max(12, height // 22))


def _draw_caption(frame: np.ndarray, text: str) -> np.ndarray:
    """Overlay ``text`` centred across the top of a rendered frame."""
    # pylint: disable=import-outside-toplevel
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = _caption_font(frame.shape[0])
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    pad = font.size // 2
    x = (frame.shape[1] - (right - left)) // 2
    y = pad
    draw.rounded_rectangle(
        (x - pad, y - pad // 2, x + (right - left) + pad, y + (bottom - top) + pad),
        radius=pad,
        fill=(28, 30, 38),
    )
    draw.text((x - left, y - top + pad // 2), text, font=font, fill=(245, 246, 250))
    return np.asarray(image)


def goal_layer(goal_world: np.ndarray, n_frames: int) -> MeshLayer:
    """A marker sphere at the Cartesian wrist goal, held for the whole clip.

    Args:
        goal_world: ``(3,)`` goal position in the body's world frame — the
                    config's spine3-relative goal plus ``body_pos[9]``.
        n_frames:   Frames to hold it for.
    """
    import trimesh  # pylint: disable=import-outside-toplevel

    sphere = trimesh.creation.icosphere(subdivisions=2, radius=_GOAL_RADIUS)
    vertices = np.tile(
        np.asarray(sphere.vertices, dtype=np.float64) + goal_world, (n_frames, 1, 1)
    )
    return MeshLayer(vertices, np.asarray(sphere.faces), color=_GOAL_COLOR)


def render_layers(
    layers: tuple[MeshLayer, ...],
    save_path: str | Path,
    *,
    fps: int = 20,
    views: tuple[str, ...] = ("front", "side"),
    resolution: int = 640,
    bounds: np.ndarray | None = None,
    caption: str | None = None,
    caption_from: int = 0,
    holds: dict[int, float] | None = None,
) -> None:
    """Render overlaid mesh layers to a video, one panel per view.

    Args:
        layers:       Meshes to draw, back to front, each with its own window.
        save_path:    Output ``.mp4`` / ``.gif``.
        fps:          Frames per second.
        views:        Keys of :data:`_VIEW_ROTATIONS`, left to right.
        resolution:   Pixel height/width of each panel.
        bounds:       ``(2, 3)`` min/max corners to frame the camera on instead
                      of the layers' own extent. Pass a shared box when several
                      clips are compared side by side, so a bigger motion reads
                      as bigger rather than being zoomed back out.
        caption:      Text drawn across the top from ``caption_from`` onward.
        caption_from: First video frame carrying the caption.
        holds:        ``{frame: seconds}`` to linger on, duplicating that frame
                      in the output rather than re-rendering it.
    """
    # pylint: disable=import-outside-toplevel
    import imageio
    import pyrender
    import trimesh

    n_frames = max(layer.start + len(layer.vertices) for layer in layers)
    # One camera framing for the whole clip so the body does not drift or rescale.
    flat = np.concatenate([layer.vertices.reshape(-1, 3) for layer in layers])
    box = (
        np.stack([flat.min(0), flat.max(0)])
        if bounds is None
        else np.asarray(bounds, dtype=np.float64)
    )
    center = box.mean(0)
    radius = float((box[1] - box[0]).max()) / 2.0
    distance = radius / np.tan(_YFOV / 2.0) * 1.08

    camera_poses = []
    for view in views:
        rotation = _VIEW_ROTATIONS[view]
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = center + rotation @ np.array([0.0, 0.0, distance])
        camera_poses.append(pose)

    renderer = pyrender.OffscreenRenderer(resolution, resolution)
    frames = []
    for t in range(n_frames):
        meshes = []
        for layer in layers:
            index = t - layer.start
            if not 0 <= index < len(layer.vertices):
                continue
            raw = trimesh.Trimesh(
                vertices=layer.vertices[index],
                faces=layer.faces,
                vertex_colors=layer.vertex_colors,
                process=False,
            )
            material = (
                None
                if layer.vertex_colors is not None
                else pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=layer.color,
                    metallicFactor=0.1,
                    roughnessFactor=0.7,
                    # Candidates overlap each other and the body; opaque ones
                    # would hide whichever arm happens to sit furthest back,
                    # including the chosen one.
                    alphaMode="BLEND" if layer.color[3] < 1.0 else "OPAQUE",
                )
            )
            meshes.append(
                pyrender.Mesh.from_trimesh(raw, material=material, smooth=True)
            )
        panels = []
        for pose in camera_poses:
            scene = pyrender.Scene(
                bg_color=(1.0, 1.0, 1.0, 1.0), ambient_light=(0.35, 0.35, 0.35)
            )
            for mesh in meshes:
                scene.add(mesh)
            scene.add(pyrender.PerspectiveCamera(yfov=_YFOV), pose=pose)
            scene.add(pyrender.DirectionalLight(intensity=4.0), pose=pose)
            color, _ = renderer.render(scene)
            panels.append(color)
        frame = np.hstack(panels)
        if caption is not None and t >= caption_from:
            frame = _draw_caption(frame, caption)
        frames.append(frame)
    renderer.delete()

    if holds:
        held = []
        for t, frame in enumerate(frames):
            held.extend([frame] * (1 + round(holds.get(t, 0.0) * fps)))
        frames = held

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(save_path), np.stack(frames), fps=fps)
    print(f"[mesh-video] saved {save_path}")


def render_correction_video(
    executed_arm_aa: np.ndarray,
    cluster_means: dict[int, np.ndarray],
    chosen_label: int,
    trigger_step: int,
    caption: str,
    *,
    body_pos: np.ndarray,
    fk: SmplLeftArmFK,
    spine3_aa: np.ndarray,
    save_path: str | Path,
    fps: int = 20,
    views: tuple[str, ...] = ("front", "side"),
    resolution: int = 640,
    hide: frozenset[int] = frozenset(),
    hold_seconds: float = 0.0,
    goal: np.ndarray | None = None,
) -> None:
    """Render one correction round: candidates fan out, the chosen one is kept.

    Through ``trigger_step`` the body is neutral. At the trigger the caption
    appears and every non-chosen cluster mean is drawn as a translucent-coloured
    arm alongside the body, so the alternatives the user was offered animate
    together; the body's arm takes the chosen colour on the frame after the
    trigger, so a hold on the trigger reads as the sentence landing rather than
    as the pick already made. The chosen cluster is not a ghost: it is the body's own arm, tinted
    its colour, because the executed motion *is* the chosen mean being tracked
    (rate-limited by ``feedback.max_playback_delta``), and a second mesh on the
    same arm would z-fight. The ghosts end with the candidates; the tint stays
    for the run out to the goal.

    Args:
        executed_arm_aa: ``(T, 3, 3)`` executed arm angles for the whole run.
        cluster_means:   ``{label: (L, 3, 3)}`` candidate means, anchored to the
                         arm pose at the trigger.
        chosen_label:    Key of the mean that was tracked.
        trigger_step:    Frame of ``executed_arm_aa`` the correction fired on.
        caption:         The feedback sentence, drawn from the trigger onward.
        hide:            Candidate labels to leave undrawn. Colours stay bound to
                         the label, so hiding one does not recolour the rest.
        hold_seconds:    Linger this long on the trigger frame and on the last
                         frame the candidates are up, so both beats are readable
                         at playback speed.
        goal:            ``(3,)`` spine3-relative Cartesian wrist goal to mark,
                         in the frame ``cartesian.goals`` uses.
    """
    cache = SmplMeshCache(body_pos, max_entries=1 + len(cluster_means))
    spine3_pos = np.asarray(body_pos[9], dtype=np.float64)
    body = _mesh_frames(cache, executed_arm_aa, fk, spine3_pos, spine3_aa)

    arm_faces = cache.left_arm_faces
    tint = np.tile((np.array(_BODY_COLOR) * 255).astype(np.uint8), (body.shape[1], 1))
    chosen_color = _CLUSTER_COLORS[chosen_label % len(_CLUSTER_COLORS)]
    tint[np.unique(arm_faces)] = (np.array(chosen_color) * 255).astype(np.uint8)

    # The trigger frame itself stays neutral: with a hold on it, that pause is
    # the moment the sentence lands, before any candidate has been committed to.
    layers = [
        MeshLayer(body[: trigger_step + 1], cache.faces),
        MeshLayer(
            body[trigger_step + 1 :],
            cache.faces,
            start=trigger_step + 1,
            vertex_colors=tint,
        ),
    ]
    candidate_frames = 0
    for label, mean in sorted(cluster_means.items()):
        if label == chosen_label or label in hide:
            continue
        candidate_frames = max(candidate_frames, len(mean))
        layers.append(
            MeshLayer(
                _mesh_frames(cache, mean, fk, spine3_pos, spine3_aa),
                arm_faces,
                color=_CLUSTER_COLORS[label % len(_CLUSTER_COLORS)][:3]
                + (_GHOST_ALPHA,),
                start=trigger_step,
            )
        )
    if goal is not None:
        layers.append(goal_layer(spine3_pos + np.asarray(goal), len(body)))
    render_layers(
        tuple(layers),
        save_path,
        fps=fps,
        views=views,
        resolution=resolution,
        caption=caption,
        caption_from=trigger_step,
        holds=(
            {
                trigger_step: hold_seconds,
                trigger_step + candidate_frames - 1: hold_seconds,
            }
            if hold_seconds > 0 and candidate_frames
            else None
        ),
    )


def _load_pose_context(
    pose_path: Path,
) -> tuple[np.ndarray, SmplLeftArmFK, np.ndarray]:
    """Decode the body pose the arm hangs off, as ``(body_pos, fk, spine3_aa)``."""
    gen = make_motion_generator("mdm", None, seed=0, lock_seed=True)
    _, body_pos, spine3_aa, collar_aa = gen.decode_pose(gen.load_pose(pose_path))
    fk = SmplLeftArmFK()
    fk.collar_aa = np.asarray(collar_aa, dtype=np.float64)
    return (
        np.asarray(body_pos, dtype=np.float64),
        fk,
        np.asarray(spine3_aa, dtype=np.float64),
    )


def main() -> None:
    """Render an ``executed_trajectory.npy`` as an SMPL-mesh video."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trajectory", type=Path, help="(T, 7) planner-state .npy trajectory"
    )
    parser.add_argument("save_path", type=Path, help="Output .mp4 or .gif")
    parser.add_argument(
        "--pose",
        type=Path,
        default=consts.MDM_START_POSE_PATH,
        help="HML263 body pose .pt supplying the torso the arm hangs off",
    )
    parser.add_argument(
        "--cluster-means",
        type=Path,
        default=None,
        dest="cluster_means",
        help=(
            "round_<N>/cluster_means.npz from the same run; adds the non-chosen "
            "candidates as coloured arms over the correction window"
        ),
    )
    parser.add_argument(
        "--trigger-step",
        type=int,
        default=None,
        dest="trigger_step",
        help="Frame the correction fired on (required with --cluster-means)",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default=None,
        help="Feedback sentence drawn from the trigger frame onward",
    )
    parser.add_argument(
        "--goal",
        nargs=3,
        type=float,
        default=None,
        help=(
            "Spine3-relative Cartesian wrist goal to mark, in the frame the "
            "config's cartesian.goals uses"
        ),
    )
    parser.add_argument(
        "--hide-clusters",
        nargs="+",
        type=int,
        default=(),
        dest="hide_clusters",
        help="Candidate labels to leave undrawn; each label keeps its colour",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=0.0,
        dest="hold_seconds",
        help=(
            "Linger this long on the trigger frame and on the candidates' last "
            "frame, so both beats read at playback speed"
        ),
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--views", nargs="+", default=["front", "side"], choices=list(_VIEW_ROTATIONS)
    )
    parser.add_argument("--resolution", type=int, default=640)
    args = parser.parse_args()
    # The MDM loader os.chdir()s into its submodule, so relative CLI paths must
    # be pinned to the launch cwd before the generator is built.
    trajectory_path = args.trajectory.resolve()
    save_path = args.save_path.resolve()
    pose_path = args.pose.resolve()
    means_path = None if args.cluster_means is None else args.cluster_means.resolve()

    body_pos, fk, spine3_aa = _load_pose_context(pose_path)
    executed = q_to_arm_aa(np.load(trajectory_path), fk.elbow_hinge_axis)

    if means_path is None:
        vertices, faces = arm_mesh_vertices(executed, body_pos, fk, spine3_aa)
        layers = [MeshLayer(vertices, faces)]
        if args.goal is not None:
            layers.append(
                goal_layer(body_pos[9] + np.asarray(args.goal), len(vertices))
            )
        render_layers(
            tuple(layers),
            save_path,
            fps=args.fps,
            views=tuple(args.views),
            resolution=args.resolution,
        )
        return

    assert args.trigger_step is not None, "--cluster-means needs --trigger-step"
    stored = np.load(means_path)
    render_correction_video(
        executed,
        {
            int(key.removeprefix("cluster_")): stored[key]
            for key in stored.files
            if key.startswith("cluster_")
        },
        int(stored["chosen_label"]),
        args.trigger_step,
        args.caption or "",
        body_pos=body_pos,
        fk=fk,
        spine3_aa=spine3_aa,
        save_path=save_path,
        fps=args.fps,
        views=tuple(args.views),
        resolution=args.resolution,
        hide=frozenset(args.hide_clusters),
        hold_seconds=args.hold_seconds,
        goal=None if args.goal is None else np.asarray(args.goal, dtype=np.float64),
    )


if __name__ == "__main__":
    main()
