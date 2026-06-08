"""Batch-extract frames from all videos in a directory.

Run this once before launching the labeler::

    uv run python src/uncertain_feedback/data_collection/extract_all_frames.py \\
        --videos_dir ./recordings/ \\
        --frames_dir ./frames/

Every frame is extracted at the video's native FPS.  Each video produces a
sub-directory ``<frames_dir>/<video_stem>/`` containing ``frame_000001.jpg``,
``frame_000002.jpg``, … and a ``meta.json`` file with the source FPS and frame
count.  These directories are consumed directly by the labeler and dataset
builder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from uncertain_feedback.data_collection.video_to_frames import extract_frames

_VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv", ".webm", ".m4v"}


def extract_all(videos_dir: Path, frames_dir: Path) -> None:
    """Extract every frame from every video in *videos_dir* into *frames_dir*.

    Args:
        videos_dir: Directory containing source video files.
        frames_dir: Root directory to write per-video frame folders into.
    """
    videos = sorted(
        p for p in videos_dir.iterdir() if p.suffix.lower() in _VIDEO_EXTENSIONS
    )
    if not videos:
        print(f"No videos found in {videos_dir}")
        return

    print(f"Found {len(videos)} video(s).\n")
    for video_path in videos:
        out_dir = frames_dir / video_path.stem
        meta_path = out_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                existing = json.load(f)
            print(
                f"  [skip] {video_path.name} — already extracted "
                f"({existing['count']} frames @ {existing['fps']} fps)"
            )
            continue

        print(f"  {video_path.name} → {out_dir.name}/  ", end="", flush=True)
        count, source_fps = extract_frames(video_path, out_dir)
        meta = {"fps": source_fps, "count": count}
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        print(f"{count} frames @ {source_fps} fps")

    print("\nDone.")


def main() -> None:
    """Parse arguments and run batch frame extraction."""
    parser = argparse.ArgumentParser(
        description="Extract all frames from every video in a directory at native FPS."
    )
    _here = Path(__file__).parent
    parser.add_argument(
        "--videos_dir",
        default=str(_here / "data" / "videos"),
        help="Directory containing source video files (default: data_collection/data/videos/).",
    )
    parser.add_argument(
        "--frames_dir",
        default=str(_here / "data" / "frames"),
        help="Directory to write per-video frame folders into (default: data_collection/data/frames/).",
    )
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir).expanduser().resolve()
    frames_dir = Path(args.frames_dir).expanduser().resolve()
    frames_dir.mkdir(parents=True, exist_ok=True)

    if not videos_dir.is_dir():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")

    extract_all(videos_dir, frames_dir)


if __name__ == "__main__":
    main()
