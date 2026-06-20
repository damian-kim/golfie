"""Shared helper for building the v0 ("Milestone 0") placeholder
ShotResult: honestly-labeled-unavailable metrics plus useful warnings
about anything that's already checkable from metadata alone (fps,
missing calibration, missing sync).

Both the FastAPI backend (golfie_api.pipeline) and the standalone CLI
(scripts/process_shot.py) call this so their behavior can't drift apart.
"""

from __future__ import annotations

from typing import Optional

from golfie_core.schemas.calibration import CalibrationResult
from golfie_core.schemas.camera import CameraCapture
from golfie_core.schemas.shot import ShotMetrics, ShotResult
from golfie_core.schemas.sync import SyncResult

_MIN_RECOMMENDED_FPS = 200.0

_PIPELINE_NOT_IMPLEMENTED_WARNING = (
    "Ball detection, tracking, triangulation, calibration, synchronization, "
    "and launch parameter estimation are not implemented yet (see project "
    "milestones 1-7). This is a structural placeholder: it confirms both "
    "videos were read successfully and reports their real, measured video "
    "metadata, but every shot metric below is honestly marked unavailable."
)


def build_placeholder_shot_result(
    camera_a: CameraCapture,
    camera_b: CameraCapture,
    calibration: Optional[CalibrationResult] = None,
    sync: Optional[SyncResult] = None,
) -> ShotResult:
    warnings: list[str] = [_PIPELINE_NOT_IMPLEMENTED_WARNING]

    for label, cam in (("camera_a", camera_a), ("camera_b", camera_b)):
        if cam.fps < _MIN_RECOMMENDED_FPS:
            warnings.append(
                f"{label} was recorded at {cam.fps:.1f} fps. 240 fps is recommended "
                "for accurate ball tracking; slower footage will likely produce "
                "low-confidence or missing detections once tracking is implemented."
            )

    if calibration is None or not calibration.is_valid:
        warnings.append(
            "No valid camera calibration is attached to this session. "
            "3D triangulation will not be possible until calibration is provided."
        )

    if sync is None:
        warnings.append(
            "No synchronization result is attached to this session. "
            "Triangulation accuracy depends directly on sync accuracy."
        )

    return ShotResult(
        metrics=ShotMetrics(),
        measured_points_3d=[],
        fitted_points_3d=[],
        simulated_trajectory_3d=[],
        warnings=warnings,
        is_placeholder=True,
        notes=(
            "Placeholder pipeline output (Milestone 0). No ball tracking or "
            "physics fitting has run."
        ),
    )
