"""Common, low-level schema types shared across the Golfie data model.

These are intentionally tiny and dependency-free (pydantic only) so that
golfie_core can be imported from the backend, the CV package, the physics
package, and test code without pulling in OpenCV/SciPy/etc.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class MetricSource(str, Enum):
    """Honesty label for every number Golfie reports.

    See spec section 19 ("Scientific honesty requirements"): every metric
    must be labeled as measured, estimated, experimental, or unavailable.
    Nothing is allowed to look more certain than it is.
    """

    MEASURED = "measured"
    ESTIMATED = "estimated"
    EXPERIMENTAL = "experimental"
    NOT_AVAILABLE = "not_available"


class MetricValue(BaseModel):
    """A single number plus an honest description of where it came from.

    `value` is optional because a NOT_AVAILABLE metric may have no number
    at all (e.g. spin on a build with no spin estimator implemented yet).
    """

    value: Optional[float] = None
    source: MetricSource = MetricSource.NOT_AVAILABLE
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    notes: Optional[str] = None

    @model_validator(mode="after")
    def _check_not_available_is_honest(self) -> "MetricValue":
        if self.source == MetricSource.NOT_AVAILABLE and self.value is not None:
            # A value with no real basis must not carry a confidence that
            # implies it was actually computed from data.
            raise ValueError(
                "MetricValue cannot have source=not_available while also "
                "carrying a numeric value; either supply a real source or "
                "drop the value."
            )
        return self

    @classmethod
    def unavailable(cls, notes: Optional[str] = None) -> "MetricValue":
        """Convenience constructor for 'we have nothing for this yet'."""
        return cls(value=None, source=MetricSource.NOT_AVAILABLE, confidence=0.0, notes=notes)


class Environment(str, Enum):
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    UNKNOWN = "unknown"


class SyncMethod(str, Enum):
    AUDIO_IMPACT = "audio_impact"
    VISUAL_IMPACT = "visual_impact"
    MANUAL = "manual"
    HYBRID = "hybrid"


class ProcessingStage(str, Enum):
    """Stages surfaced on the processing screen (spec section 16)."""

    CREATED = "created"
    EXTRACTING_FRAMES = "extracting_frames"
    SYNCING = "syncing"
    DETECTING_IMPACT = "detecting_impact"
    DETECTING_BALL = "detecting_ball"
    TRACKING_BALL = "tracking_ball"
    TRIANGULATING = "triangulating"
    FITTING_PHYSICS = "fitting_physics"
    RENDERING = "rendering"
    DONE = "done"
    FAILED = "failed"
