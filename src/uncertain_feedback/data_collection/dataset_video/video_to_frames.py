"""Extract frames from a video file into a folder for use with the data_collection pipeline.

Usage::

    uv run python src/uncertain_feedback/data_collection/dataset_video/video_to_frames.py \\
        --video_path ./recording.mov \\
        --output_dir ./video_frames \\
        [--ext jpg]

Every frame is extracted at the video's native FPS.
The output folder can be passed directly to ``pose_estimation/_inference_worker.py`` via
``--image_folder``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2  # type: ignore[import]


def extract_frames(
    video_path: Path,
    output_dir: Path,
    ext: str = "jpg",
    start_sec: float = 0.0,
    end_sec: float | None = None,
) -> tuple[int, float]:
    """Extract every frame from *video_path* into *output_dir*.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory to write frames into (created if absent).
        ext: Output image extension, e.g. ``"jpg"`` or ``"png"``.
        start_sec: Start time in seconds (default: 0 — beginning of video).
        end_sec: End time in seconds (default: ``None`` — end of video).

    Returns:
        Tuple of (number of frames written, source FPS of the video).

    Raises:
        FileNotFoundError: If *video_path* does not exist.
        RuntimeError: If the video cannot be opened by OpenCV.
    """
    video_path = Path(video_path).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))  # pylint: disable=no-member
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)  # pylint: disable=no-member
    if source_fps <= 0:
        source_fps = 30.0  # fallback for containers that don't report FPS

    if start_sec > 0:
        start_frame = int(start_sec * source_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # pylint: disable=no-member
    end_frame = int(end_sec * source_fps) if end_sec is not None else None

    written = 0
    while True:
        raw_frame_pos = int(
            cap.get(cv2.CAP_PROP_POS_FRAMES)  # pylint: disable=no-member
        )
        if end_frame is not None and raw_frame_pos > end_frame:
            break
        ret, frame = cap.read()
        if not ret:
            break
        written += 1
        out_path = output_dir / f"frame_{written:06d}.{ext}"
        cv2.imwrite(str(out_path), frame.copy())

    cap.release()
    return written, source_fps


def main() -> None:
    """Parse arguments and run frame extraction."""
    parser = argparse.ArgumentParser(
        description="Extract all frames from a video at its native FPS."
    )
    parser.add_argument("--video_path", required=True, help="Path to the input video.")
    parser.add_argument(
        "--output_dir", required=True, help="Directory to write extracted frames into."
    )
    parser.add_argument(
        "--ext",
        default="jpg",
        choices=["jpg", "png"],
        help="Output image format (default: jpg).",
    )
    args = parser.parse_args()

    n, source_fps = extract_frames(
        video_path=Path(args.video_path),
        output_dir=Path(args.output_dir),
        ext=args.ext,
    )
    print(f"Wrote {n} frames @ {source_fps} fps to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
