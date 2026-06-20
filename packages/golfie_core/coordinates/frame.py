"""Golfie world coordinate convention and the transform into/out of it.

Convention (spec section 6):
  - origin: golf ball position at address
  - +X: target line direction
  - +Z: vertical up
  - +Y: chosen so the frame is right-handed (X cross Y = Z)
  - units: meters / seconds / radians internally; UI converts to
    yards/mph/degrees/rpm at the edges.

Triangulation happens in an arbitrary "rig frame" defined by stereo
calibration (typically camera A's frame). `CoordinateTransformer` maps
points from that rig frame into the physically meaningful Golfie world
frame defined by a `CoordinateSystem` (target line + up vector + origin,
themselves derived from floor markers / calibration board / manual UI
confirmation -- see spec section 7 Phase 3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from golfie_core.schemas.calibration import CoordinateSystem


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-9:
        raise ValueError("Cannot normalize a near-zero vector")
    return v / norm


def build_right_handed_basis(
    target_direction: np.ndarray, up_direction: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build an orthonormal right-handed basis (x_hat, y_hat, z_hat).

    x_hat is exactly the (normalized) target direction. z_hat is the
    component of `up_direction` orthogonal to x_hat (so "up" need not be
    perfectly vertical relative to the target line -- it just can't be
    parallel to it). y_hat is derived so that x_hat cross y_hat == z_hat,
    matching the "+Y is whatever makes this right-handed" rule in the spec.
    """
    x_hat = _normalize(np.asarray(target_direction, dtype=float))
    up = np.asarray(up_direction, dtype=float)
    up_orth = up - np.dot(up, x_hat) * x_hat
    z_hat = _normalize(up_orth)
    y_hat = np.cross(z_hat, x_hat)
    return x_hat, y_hat, z_hat


@dataclass
class CoordinateTransformer:
    """Converts 3D points between the stereo rig frame and Golfie world frame."""

    origin_in_rig_frame_m: np.ndarray  # (3,)
    rotation_rig_to_world: np.ndarray  # (3, 3), rows are x_hat, y_hat, z_hat

    @classmethod
    def from_coordinate_system(cls, cs: CoordinateSystem) -> "CoordinateTransformer":
        x_hat, y_hat, z_hat = build_right_handed_basis(
            np.asarray(cs.target_direction_in_rig_frame, dtype=float),
            np.asarray(cs.up_direction_in_rig_frame, dtype=float),
        )
        rotation = np.stack([x_hat, y_hat, z_hat], axis=0)
        origin = np.asarray(cs.origin_in_rig_frame_m, dtype=float)
        return cls(origin_in_rig_frame_m=origin, rotation_rig_to_world=rotation)

    @classmethod
    def identity(cls) -> "CoordinateTransformer":
        return cls(
            origin_in_rig_frame_m=np.zeros(3),
            rotation_rig_to_world=np.eye(3),
        )

    def to_world(self, point_rig: np.ndarray) -> np.ndarray:
        """rig-frame point (meters) -> Golfie world-frame point (meters)."""
        p = np.asarray(point_rig, dtype=float) - self.origin_in_rig_frame_m
        return self.rotation_rig_to_world @ p

    def to_rig(self, point_world: np.ndarray) -> np.ndarray:
        """Golfie world-frame point (meters) -> rig-frame point (meters)."""
        p = self.rotation_rig_to_world.T @ np.asarray(point_world, dtype=float)
        return p + self.origin_in_rig_frame_m
