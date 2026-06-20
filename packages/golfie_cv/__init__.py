"""golfie_cv

Computer-vision pipeline: video ingestion, calibration, synchronization,
ball detection/tracking, and triangulation.

Status (v0 / Milestone 0):
  - video: REAL. Extracts actual fps/resolution/duration/frame_count
    from uploaded files via OpenCV. This is what process_shot.py uses
    today.
  - calibration, sync, detection, tracking, triangulation, debug: STUBS.
    Each module defines its real input/output contract (matching the
    spec's data model) and raises NotImplementedError. They are filled
    in milestone-by-milestone (see project spec section 20) -- the
    point of stubbing them now is so the backend and CLI can already
    call a stable interface and swap in real implementations without
    changing any caller.
"""

__all__ = [
    "video",
    "calibration",
    "sync",
    "detection",
    "tracking",
    "triangulation",
    "debug",
]
