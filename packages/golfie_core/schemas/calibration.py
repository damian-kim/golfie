"""Calibration result + world coordinate system schema.

See spec section 6 (coordinate system) and section 7 (calibration phases).
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class CoordinateSystem(BaseModel):
    """Defines how the abstract stereo-rig frame maps to Golfie world axes.

    Golfie's world convention (see docs/coordinate_system.md):
      - origin: ball position at address
      - +X: target line direction
      - +Y: golfer's right-hand side in a right-handed frame (so +Z = up
        is consistent with X cross Y)
      - +Z: vertical up
      - units: meters / seconds / radians

    `origin_in_rig_frame` and `target_direction_in_rig_frame` describe how
    this world frame sits inside the raw stereo-calibration frame (i.e.
    the frame defined by camera A at the calibration step), so that 3D
    points triangulated in the rig frame can be rotated/translated into
    Golfie world coordinates.
    """

    origin_in_rig_frame_m: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    target_direction_in_rig_frame: List[float] = Field(
        default_factory=lambda: [1.0, 0.0, 0.0]
    )
    up_direction_in_rig_frame: List[float] = Field(
        default_factory=lambda: [0.0, 0.0, 1.0]
    )
    alignment_method: str = "manual"  # "manual" | "floor_markers" | "calibration_board"
    notes: Optional[str] = None


class CalibrationResult(BaseModel):
    """Output of intrinsic + stereo extrinsic calibration (spec section 7)."""

    coordinate_system: Optional[CoordinateSystem] = None
    camera_a_intrinsics: Optional[List[List[float]]] = None
    camera_b_intrinsics: Optional[List[List[float]]] = None
    camera_a_extrinsics: Optional[List[List[float]]] = None  # 4x4 or 3x4, TBD by golfie_cv
    camera_b_extrinsics: Optional[List[List[float]]] = None
    reprojection_error_px: Optional[float] = None
    confidence: float = 0.0
    calibration_target: Optional[str] = None  # "checkerboard" | "charuco" | "aruco" | "apriltag"
    is_valid: bool = False
