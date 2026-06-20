"""Video synchronization result schema. See spec section 8."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from golfie_core.schemas.common import SyncMethod


class SyncResult(BaseModel):
    offset_seconds: float
    offset_frames: float
    method: SyncMethod = SyncMethod.MANUAL
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    debug_plot_path: Optional[str] = None
    notes: Optional[str] = None
