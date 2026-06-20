"""Top-level Session schema -- one row per recorded golf shot attempt.

This is the object the FastAPI backend persists to data/shots/<id>.json
and is the single source of truth a frontend page needs to render the
upload, processing, review, and simulator screens.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from golfie_core.schemas.calibration import CalibrationResult
from golfie_core.schemas.camera import CameraCapture
from golfie_core.schemas.common import Environment, ProcessingStage
from golfie_core.schemas.shot import ShotResult
from golfie_core.schemas.sync import SyncResult


class Session(BaseModel):
    session_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    environment: Environment = Environment.UNKNOWN
    stage: ProcessingStage = ProcessingStage.CREATED
    error: Optional[str] = None

    # Optional shot metadata supplied by the user at upload time.
    club: Optional[str] = None
    handedness: Optional[str] = None  # "left" | "right"
    ball_type: Optional[str] = None

    camera_a: Optional[CameraCapture] = None
    camera_b: Optional[CameraCapture] = None
    calibration: Optional[CalibrationResult] = None
    sync: Optional[SyncResult] = None
    shot: Optional[ShotResult] = None
