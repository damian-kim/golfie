"""Tracked point schemas for ball detection/tracking/triangulation output.

See spec sections 9-11.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TrackedPoint2D(BaseModel):
    frame_index: int
    time_seconds: float
    x_px: float
    y_px: float
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class TrackedPoint3D(BaseModel):
    time_seconds: float
    x_m: float
    y_m: float
    z_m: float
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reprojection_error_px: float | None = None
