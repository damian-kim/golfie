"""Debug visualization helpers.

Unlike most of golfie_cv, this module is partially real in v0: drawing an
overlay onto a frame is simple enough not to need a stub, and having it
available early makes every later milestone (detection, tracking,
triangulation) easier to debug visually from day one (spec section 22).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def draw_point_overlay(
    frame: np.ndarray,
    points_px: list[tuple[float, float]],
    color_bgr: tuple[int, int, int] = (0, 255, 0),
    radius: int = 6,
    labels: list[str] | None = None,
) -> np.ndarray:
    """Draw candidate/tracked points on a copy of a frame for debug review."""
    out = frame.copy()
    for i, (x, y) in enumerate(points_px):
        center = (int(round(x)), int(round(y)))
        cv2.circle(out, center, radius, color_bgr, thickness=2)
        if labels and i < len(labels):
            cv2.putText(
                out,
                labels[i],
                (center[0] + radius + 2, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color_bgr,
                1,
                cv2.LINE_AA,
            )
    return out


def save_debug_frame(frame: np.ndarray, out_path: str | Path) -> str:
    """Save a frame (e.g. with an overlay already drawn) to disk as PNG/JPG."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), frame)
    if not ok:
        raise RuntimeError(f"Failed to write debug frame to {out_path}")
    return str(out_path)
