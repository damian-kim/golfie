"""Camera capture and calibration geometry schemas.

Matrices are stored as plain nested lists (not numpy arrays) so that this
module stays JSON-serializable and dependency-light. golfie_cv is
responsible for converting to/from numpy when it actually does math.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import BaseModel, field_validator


class CameraIntrinsics(BaseModel):
    """Pinhole camera intrinsics + distortion, OpenCV convention."""

    fx: float
    fy: float
    cx: float
    cy: float
    # OpenCV-style distortion coefficients: [k1, k2, p1, p2, k3, ...]
    distortion: List[float] = []
    image_width: int
    image_height: int
    reprojection_error_px: Optional[float] = None

    def as_matrix(self) -> List[List[float]]:
        """3x3 camera matrix K, as a plain nested list."""
        return [
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ]


class CameraExtrinsics(BaseModel):
    """Pose of a camera relative to the Golfie world coordinate system.

    rotation_matrix is row-major 3x3 (world-to-camera), translation_m is
    the camera position offset in meters, also world-to-camera convention.
    Keeping rotation as a full matrix (rather than e.g. quaternion) keeps
    this directly compatible with cv2.triangulatePoints / cv2.Rodrigues.
    """

    rotation_matrix: List[List[float]]
    translation_m: List[float]

    @field_validator("rotation_matrix")
    @classmethod
    def _check_rotation_shape(cls, v: List[List[float]]) -> List[List[float]]:
        if len(v) != 3 or any(len(row) != 3 for row in v):
            raise ValueError("rotation_matrix must be 3x3")
        return v

    @field_validator("translation_m")
    @classmethod
    def _check_translation_shape(cls, v: List[float]) -> List[float]:
        if len(v) != 3:
            raise ValueError("translation_m must have length 3 (x, y, z)")
        return v


class CameraCapture(BaseModel):
    """One phone's recording of a single shot."""

    camera_id: str
    device_model: Optional[str] = None
    fps: float
    resolution: Tuple[int, int]
    video_path: str
    role_hint: Optional[str] = None  # e.g. "down_the_line", "face_on"
    intrinsics: Optional[CameraIntrinsics] = None
    extrinsics: Optional[CameraExtrinsics] = None
