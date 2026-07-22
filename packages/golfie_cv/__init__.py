"""golfie_cv

Computer-vision pipeline: video ingestion, calibration, synchronization,
ball detection/tracking, and triangulation.

The rescue MVP implements real metadata/CFR handling, validation-gated stereo
calibration, audio synchronization, candidate detection, joint stereo-aware
tracking, and distortion-corrected triangulation. Debug rendering remains an
optional development aid rather than part of the scientific result.
"""

__all__ = [
    "video",
    "calibration",
    "sync",
    "detection",
    "tracking",
    "spin",
    "triangulation",
    "debug",
]
