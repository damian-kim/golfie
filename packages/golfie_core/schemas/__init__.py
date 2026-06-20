"""Public schema exports for golfie_core.

Usage:
    from golfie_core.schemas import Session, ShotMetrics, MetricValue
"""

from golfie_core.schemas.calibration import CalibrationResult, CoordinateSystem
from golfie_core.schemas.camera import CameraCapture, CameraExtrinsics, CameraIntrinsics
from golfie_core.schemas.common import (
    Environment,
    MetricSource,
    MetricValue,
    ProcessingStage,
    SyncMethod,
)
from golfie_core.schemas.session import Session
from golfie_core.schemas.shot import ShotMetrics, ShotResult
from golfie_core.schemas.sync import SyncResult
from golfie_core.schemas.tracking import TrackedPoint2D, TrackedPoint3D
from golfie_core.schemas.placeholder import build_placeholder_shot_result

__all__ = [
    "CalibrationResult",
    "CoordinateSystem",
    "CameraCapture",
    "CameraExtrinsics",
    "CameraIntrinsics",
    "Environment",
    "MetricSource",
    "MetricValue",
    "ProcessingStage",
    "SyncMethod",
    "Session",
    "ShotMetrics",
    "ShotResult",
    "SyncResult",
    "TrackedPoint2D",
    "TrackedPoint3D",
    "build_placeholder_shot_result",
]
