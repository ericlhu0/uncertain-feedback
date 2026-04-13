"""Extract frames from a video file into a folder for use with the data_collection pipeline.

Usage::

    uv run python src/uncertain_feedback/data_collection/video_to_frames.py \\
        --video_path ./recording.mov \\
        --output_dir ./video_frames \\
        [--fps 10] \\
        [--ext jpg]

The output folder can be passed directly to ``_mhr_inference_worker.py`` via
``--image_folder``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2  # type: ignore[import]


def extract_frames(
    video_path: Path,
    output_dir: Path,
    fps: float = 10.0,
    ext: str = "jpg",
) -> int:
    """Extract frames from *video_path* into *output_dir* at the given frame rate.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory to write frames into (created if absent).
        fps: Target frames per second to extract.  The source video is decimated
            so that approximately *fps* frames are written per second of footage.
        ext: Output image extension, e.g. ``"jpg"`` or ``"png"``.

    Returns:
        Number of frames written.

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
    stride = max(1, round(source_fps / fps))

    frame_idx = 0
    written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            written += 1
            out_path = output_dir / f"frame_{written:06d}.{ext}"
            cv2.imwrite(str(out_path), cv2.flip(frame, 0))  # pylint: disable=no-member
        frame_idx += 1

    cap.release()
    return written


def main() -> None:
    """Parse arguments and run frame extraction."""
    parser = argparse.ArgumentParser(
        description="Extract frames from a video file for the data_collection pipeline."
    )
    parser.add_argument("--video_path", required=True, help="Path to the input video.")
    parser.add_argument(
        "--output_dir", required=True, help="Directory to write extracted frames into."
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Frames per second to extract.",
    )
    parser.add_argument(
        "--ext",
        default="jpg",
        choices=["jpg", "png"],
        help="Output image format (default: jpg).",
    )
    args = parser.parse_args()

    n = extract_frames(
        video_path=Path(args.video_path),
        output_dir=Path(args.output_dir),
        fps=args.fps,
        ext=args.ext,
    )
    print(f"Wrote {n} frames to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
