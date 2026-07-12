"""Calibration result + world coordinate system schema.

See spec section 6 (coordinate system) and section 7 (calibration phases).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

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

    calibration_version: int = 1
    coordinate_system: Optional[CoordinateSystem] = None
    camera_a_intrinsics: Optional[List[List[float]]] = None
    camera_b_intrinsics: Optional[List[List[float]]] = None
    camera_a_extrinsics: Optional[List[List[float]]] = None  # 4x4 or 3x4, TBD by golfie_cv
    camera_b_extrinsics: Optional[List[List[float]]] = None

    # A 3x3 intrinsic matrix is not a complete lens model.  These fields
    # deliberately live beside the legacy matrix fields so calibration JSON
    # written by older Golfie builds remains readable.
    camera_a_distortion: List[float] = Field(default_factory=list)
    camera_b_distortion: List[float] = Field(default_factory=list)
    camera_a_image_size: Optional[Tuple[int, int]] = None  # (width, height)
    camera_b_image_size: Optional[Tuple[int, int]] = None

    # Stereo convention: camera_a_extrinsics and camera_b_extrinsics are
    # rig/world-to-camera transforms.  For calibrations produced by Golfie,
    # camera A is the rig origin and stereo_rotation/stereo_translation map
    # camera-A coordinates into camera-B coordinates.
    stereo_rotation: Optional[List[List[float]]] = None
    stereo_translation_m: Optional[List[float]] = None
    essential_matrix: Optional[List[List[float]]] = None
    fundamental_matrix: Optional[List[List[float]]] = None

    # Quality diagnostics are persisted so callers can reject a calibration
    # for a concrete reason instead of trusting a single optimizer RMS value.
    intrinsic_error_a_px: Optional[float] = None
    intrinsic_error_b_px: Optional[float] = None
    epipolar_error_median_px: Optional[float] = None
    epipolar_error_p95_px: Optional[float] = None
    epipolar_inlier_ratio: Optional[float] = None
    baseline_m: Optional[float] = None
    validation_frame_count: int = 0
    validation_warnings: List[str] = Field(default_factory=list)
    reprojection_error_px: Optional[float] = None
    confidence: float = 0.0
    calibration_target: Optional[str] = None  # "checkerboard" | "charuco" | "aruco" | "apriltag"
    board_grid_size: Optional[Tuple[int, int]] = None
    board_square_length_m: Optional[float] = None
    board_marker_length_m: Optional[float] = None
    camera_a_calibration_fps: Optional[float] = None
    camera_b_calibration_fps: Optional[float] = None
    measured_baseline_m: Optional[float] = None
    baseline_source: str = "stereo_board_solve"
    is_valid: bool = False
