"""Shot-level outputs: honest, confidence-labeled metrics plus the
trajectories that back them up. See spec sections 5 and 19.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel

from golfie_core.schemas.common import MetricValue
from golfie_core.schemas.tracking import TrackedPoint3D


class ShotMetrics(BaseModel):
    """Every metric Golfie might show, each individually labeled.

    Required-by-v0 metrics (ball_speed, launch_angle, horizontal_launch,
    carry, total, apex, side_deviation) should normally have
    source=measured or source=estimated once the real pipeline is wired
    up. Spin/club metrics default to NOT_AVAILABLE and must stay
    EXPERIMENTAL at best until a real estimator exists -- never MEASURED,
    per spec section 12.
    """

    ball_speed_mps: MetricValue = MetricValue.unavailable()
    launch_angle_deg: MetricValue = MetricValue.unavailable()
    horizontal_launch_deg: MetricValue = MetricValue.unavailable()
    carry_m: MetricValue = MetricValue.unavailable()
    total_m: MetricValue = MetricValue.unavailable()
    apex_m: MetricValue = MetricValue.unavailable()
    side_deviation_m: MetricValue = MetricValue.unavailable()
    backspin_rpm: MetricValue = MetricValue.unavailable()
    sidespin_rpm: MetricValue = MetricValue.unavailable()
    spin_axis_deg: MetricValue = MetricValue.unavailable()
    club_speed_mps: MetricValue = MetricValue.unavailable()
    smash_factor: MetricValue = MetricValue.unavailable()


class ShotResult(BaseModel):
    """Everything produced by processing one shot."""

    metrics: ShotMetrics = ShotMetrics()
    measured_points_3d: List[TrackedPoint3D] = []
    fitted_points_3d: List[TrackedPoint3D] = []
    simulated_trajectory_3d: List[TrackedPoint3D] = []
    warnings: List[str] = []
    is_placeholder: bool = True
    notes: Optional[str] = None
