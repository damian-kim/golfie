"""Video ingestion: read real metadata from an uploaded video file.

This is the one piece of golfie_cv that is fully implemented in v0 --
everything downstream (sync, detection, tracking) needs accurate fps and
frame counts, and there is no honest way to fake this, so it's a good
place to start with real code instead of a stub.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


class VideoReadError(RuntimeError):
    """Raised when a video file can't be opened or has no readable frames."""


@dataclass
class VideoMetadata:
    path: str
    fps: float
    width: int
    height: int
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self.frame_count / self.fps


def read_video_metadata(video_path: str | Path) -> VideoMetadata:
    """Open a video file and read its real fps/resolution/frame count.

    Raises VideoReadError if the file is missing, unreadable, or not a
    video OpenCV understands -- callers should surface this to the user
    rather than substituting fake metadata (spec section 19, scientific
    honesty).
    """
    path = Path(video_path)
    if not path.exists():
        raise VideoReadError(f"Video file does not exist: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise VideoReadError(f"OpenCV could not open video file: {path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    if width <= 0 or height <= 0:
        raise VideoReadError(
            f"Video opened but reported invalid resolution ({width}x{height}): {path}"
        )

    return VideoMetadata(
        path=str(path),
        fps=float(fps),
        width=width,
        height=height,
        frame_count=frame_count,
    )


def extract_frame(video_path: str | Path, frame_index: int):
    """Extract a single frame as a BGR numpy array (OpenCV convention).

    Used by debug tooling (e.g. previewing the impact frame) and, later,
    by the detection pipeline. Raises VideoReadError on failure.
    """
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise VideoReadError(f"OpenCV could not open video file: {path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise VideoReadError(
                f"Could not read frame {frame_index} from {path}"
            )
        return frame
    finally:
        cap.release()
