#!/usr/bin/env python3
"""Standalone CLI: process two golf-shot videos into a results.json.

This is the "preferred first task" CLI from the project spec -- it does
NOT depend on the web backend or any session storage. Give it two video
paths and it writes a results.json describing what it actually knows.

Usage:
    python scripts/process_shot.py camera_a.mp4 camera_b.mp4 -o results.json
    python scripts/process_shot.py camera_a.mp4 camera_b.mp4 --club driver

v0 behavior (honest by design, spec section 19):
    - Reads REAL fps/resolution/duration from both videos via OpenCV.
    - Does NOT detect, track, triangulate, or estimate any shot metric --
      those are Milestones 1-7 and are not implemented yet.
    - Every metric in the output is explicitly source=not_available.
    - Warns if either camera is below the recommended 240fps, or if no
      calibration/sync info was supplied.

This intentionally produces a "boring" results.json today. That's the
point: it should never look more confident than the underlying pipeline
actually is. As each milestone lands, this script's job changes from
"prove the file I/O and schema work" to "actually run the pipeline."
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running this script directly from a repo checkout without an
# editable-install step, by adding the sibling packages/ dir to sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _pkg in ("golfie_core", "golfie_cv", "golfie_physics", "golfie_render"):
    _pkg_path = _REPO_ROOT / "packages" / _pkg
    if str(_pkg_path) not in sys.path and _pkg_path.exists():
        sys.path.insert(0, str(_pkg_path))

from golfie_core.schemas import CameraCapture, Session, build_placeholder_shot_result  # noqa: E402
from golfie_cv.video import VideoReadError, read_video_metadata  # noqa: E402


def _build_camera_capture(camera_id: str, video_path: Path, role_hint: str | None) -> CameraCapture:
    meta = read_video_metadata(video_path)
    return CameraCapture(
        camera_id=camera_id,
        fps=meta.fps,
        resolution=(meta.width, meta.height),
        video_path=str(video_path),
        role_hint=role_hint,
    )


def process_shot(
    video_a: Path,
    video_b: Path,
    club: str | None = None,
    role_hint_a: str | None = "down_the_line",
    role_hint_b: str | None = "face_on",
) -> Session:
    camera_a = _build_camera_capture("camera_a", video_a, role_hint_a)
    camera_b = _build_camera_capture("camera_b", video_b, role_hint_b)

    session = Session(club=club, camera_a=camera_a, camera_b=camera_b)
    
    try:
        from golfie_cv.sync import estimate_sync_offset
        session.sync = estimate_sync_offset(video_a, video_b)
    except Exception as e:
        from golfie_core.schemas import SyncResult, SyncMethod
        session.sync = SyncResult(
            offset_seconds=0.0,
            offset_frames=0.0,
            method=SyncMethod.MANUAL,
            confidence=0.0,
            notes=f"Auto audio sync failed: {e}",
        )

    session.shot = build_placeholder_shot_result(
        camera_a=camera_a,
        camera_b=camera_b,
        calibration=session.calibration,
        sync=session.sync,
    )
    return session


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video_a", type=Path, help="Path to camera A's video file")
    parser.add_argument("video_b", type=Path, help="Path to camera B's video file")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("results.json"), help="Output JSON path (default: results.json)"
    )
    parser.add_argument("--club", default=None, help="Optional club used for this shot (metadata only)")
    args = parser.parse_args()

    for label, path in (("video_a", args.video_a), ("video_b", args.video_b)):
        if not path.exists():
            print(f"error: {label} does not exist: {path}", file=sys.stderr)
            return 1

    try:
        session = process_shot(args.video_a, args.video_b, club=args.club)
    except VideoReadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(session.model_dump_json(indent=2))

    print(f"Wrote {args.output}")
    print(f"  camera_a: {session.camera_a.fps:.1f} fps, {session.camera_a.resolution[0]}x{session.camera_a.resolution[1]}")
    print(f"  camera_b: {session.camera_b.fps:.1f} fps, {session.camera_b.resolution[0]}x{session.camera_b.resolution[1]}")
    print(f"  generated: {datetime.now(timezone.utc).isoformat()}")
    print("  shot metrics: all source=not_available (no detection/tracking pipeline implemented yet)")
    for w in session.shot.warnings:
        print(f"  warning: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
