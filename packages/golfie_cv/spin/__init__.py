"""Marked-ball surface rotation and stereo spin-vector estimation.

Each camera measures the component of angular velocity about its optical
axis. Two camera components plus the golf-ball assumption that rifle spin
along the launch direction is negligible determine a 3D spin vector in the
Golfie world frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from golfie_core.coordinates import CoordinateTransformer
from golfie_core.schemas import CalibrationResult, TrackedPoint2D


@dataclass
class CameraSpinEstimate:
    optical_axis_rad_s: float
    confidence: float
    sample_count: int
    median_features: float
    dispersion_rad_s: float


@dataclass
class StereoSpinEstimate:
    spin_world_rad_s: np.ndarray
    confidence: float
    backspin_rpm: float
    sidespin_rpm: float
    spin_axis_deg: float
    total_spin_rpm: float
    projection_residual_rad_s: float


def _read_selected_frames(
    video_path: str | Path, frame_indices: set[int]
) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return {}
    frames: dict[int, np.ndarray] = {}
    for frame_index in sorted(frame_indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if ok and frame is not None:
            frames[frame_index] = frame
    capture.release()
    return frames


def estimate_camera_spin_from_frames(
    frames: dict[int, np.ndarray],
    track: list[TrackedPoint2D],
    *,
    min_feature_tracks: int = 5,
) -> CameraSpinEstimate | None:
    """Estimate image-plane angular velocity from markings inside the ball."""
    ordered = sorted(track, key=lambda point: point.time_seconds)
    angular_rates: list[float] = []
    feature_counts: list[int] = []
    pair_qualities: list[float] = []

    for previous, current in zip(ordered, ordered[1:]):
        dt = float(current.time_seconds - previous.time_seconds)
        frame_gap = current.frame_index - previous.frame_index
        if not (0.001 <= dt <= 0.05) or not (1 <= frame_gap <= 4):
            continue
        frame_a = frames.get(previous.frame_index)
        frame_b = frames.get(current.frame_index)
        if frame_a is None or frame_b is None:
            continue

        radius_values = [
            value for value in (previous.radius_px, current.radius_px)
            if value is not None and np.isfinite(value)
        ]
        if not radius_values:
            continue
        radius = float(np.clip(np.median(radius_values), 7.0, 100.0))
        if radius < 7.0:
            continue

        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray_a)
        center_a = np.array([previous.x_px, previous.y_px], dtype=np.float32)
        center_b = np.array([current.x_px, current.y_px], dtype=np.float32)
        cv2.circle(
            mask,
            tuple(np.round(center_a).astype(int)),
            max(4, int(round(0.82 * radius))),
            255,
            -1,
        )
        features = cv2.goodFeaturesToTrack(
            gray_a,
            maxCorners=80,
            qualityLevel=0.008,
            minDistance=max(2.0, 0.08 * radius),
            mask=mask,
            blockSize=3,
            useHarrisDetector=False,
        )
        if features is None or len(features) < min_feature_tracks:
            continue
        followed, status, errors = cv2.calcOpticalFlowPyrLK(
            gray_a,
            gray_b,
            features,
            None,
            winSize=(31, 31),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.005),
        )
        if followed is None or status is None:
            continue
        keep = status.reshape(-1).astype(bool)
        if errors is not None:
            keep &= errors.reshape(-1) < 35.0
        source = features.reshape(-1, 2)[keep]
        target = followed.reshape(-1, 2)[keep]
        if len(source) < min_feature_tracks:
            continue

        source_local = source - center_a
        target_local = target - center_b
        inside = (
            (np.linalg.norm(source_local, axis=1) <= 0.9 * radius)
            & (np.linalg.norm(target_local, axis=1) <= 0.95 * radius)
            & (np.linalg.norm(source_local, axis=1) >= 0.12 * radius)
        )
        source_local = source_local[inside]
        target_local = target_local[inside]
        if len(source_local) < min_feature_tracks:
            continue

        transform, inliers = cv2.estimateAffinePartial2D(
            source_local,
            target_local,
            method=cv2.RANSAC,
            ransacReprojThreshold=max(1.0, 0.06 * radius),
            maxIters=1000,
            confidence=0.995,
            refineIters=10,
        )
        if transform is None or inliers is None:
            continue
        inlier_count = int(np.sum(inliers))
        if inlier_count < min_feature_tracks:
            continue
        angle = float(np.arctan2(transform[1, 0], transform[0, 0]))
        scale = float(np.hypot(transform[0, 0], transform[1, 0]))
        if abs(angle) < 0.002 or abs(angle) > 2.8 or not (0.65 <= scale <= 1.45):
            continue
        inlier_fraction = inlier_count / len(source_local)
        angular_rates.append(angle / dt)
        feature_counts.append(inlier_count)
        pair_qualities.append(float(np.clip(inlier_fraction, 0.0, 1.0)))

    if len(angular_rates) < 2:
        return None
    rates = np.asarray(angular_rates, dtype=np.float64)
    median_rate = float(np.median(rates))
    mad = float(1.4826 * np.median(np.abs(rates - median_rate)))
    relative_dispersion = mad / max(abs(median_rate), 1.0)
    if relative_dispersion > 0.65:
        return None
    sample_score = min(1.0, len(rates) / 4.0)
    feature_score = min(1.0, float(np.median(feature_counts)) / 12.0)
    consistency_score = float(np.exp(-2.0 * relative_dispersion))
    confidence = float(
        np.clip(
            sample_score
            * feature_score
            * consistency_score
            * float(np.median(pair_qualities)),
            0.0,
            1.0,
        )
    )
    if confidence < 0.2:
        return None
    return CameraSpinEstimate(
        optical_axis_rad_s=median_rate,
        confidence=confidence,
        sample_count=len(rates),
        median_features=float(np.median(feature_counts)),
        dispersion_rad_s=mad,
    )


def estimate_camera_spin(
    video_path: str | Path, track: list[TrackedPoint2D]
) -> CameraSpinEstimate | None:
    indices = {point.frame_index for point in track}
    return estimate_camera_spin_from_frames(_read_selected_frames(video_path, indices), track)


def _camera_optical_axis_world(
    extrinsics: list[list[float]], calibration: CalibrationResult
) -> np.ndarray:
    matrix = np.asarray(extrinsics, dtype=np.float64)
    matrix = matrix[:3, :] if matrix.shape == (4, 4) else matrix
    axis_rig = matrix[:, :3].T @ np.array([0.0, 0.0, 1.0])
    transformer = (
        CoordinateTransformer.from_coordinate_system(calibration.coordinate_system)
        if calibration.coordinate_system is not None
        else CoordinateTransformer.identity()
    )
    axis_world = transformer.rotation_rig_to_world @ axis_rig
    return axis_world / max(float(np.linalg.norm(axis_world)), 1e-12)


def reconstruct_stereo_spin(
    camera_a: CameraSpinEstimate,
    camera_b: CameraSpinEstimate,
    calibration: CalibrationResult,
    launch_velocity_world_mps: np.ndarray,
) -> StereoSpinEstimate | None:
    """Recover world-frame spin using two views and a zero-rifle-spin constraint."""
    if calibration.camera_a_extrinsics is None or calibration.camera_b_extrinsics is None:
        return None
    velocity = np.asarray(launch_velocity_world_mps, dtype=np.float64)
    speed = float(np.linalg.norm(velocity))
    if speed < 1.0:
        return None
    flight_axis = velocity / speed
    axis_a = _camera_optical_axis_world(calibration.camera_a_extrinsics, calibration)
    axis_b = _camera_optical_axis_world(calibration.camera_b_extrinsics, calibration)
    system = np.stack([axis_a, axis_b, flight_axis])
    condition = float(np.linalg.cond(system))
    if not np.isfinite(condition) or condition > 30.0:
        return None
    observations = np.array(
        [camera_a.optical_axis_rad_s, camera_b.optical_axis_rad_s, 0.0],
        dtype=np.float64,
    )
    spin, *_ = np.linalg.lstsq(system, observations, rcond=None)
    predicted = system @ spin
    residual = float(np.sqrt(np.mean((predicted - observations) ** 2)))
    total_rpm = float(np.linalg.norm(spin) * 60.0 / (2.0 * np.pi))
    if not (100.0 <= total_rpm <= 15000.0):
        return None
    backspin_rpm = float(-spin[1] * 60.0 / (2.0 * np.pi))
    sidespin_rpm = float(spin[2] * 60.0 / (2.0 * np.pi))
    spin_axis = float(np.degrees(np.arctan2(sidespin_rpm, backspin_rpm)))
    geometry_score = min(1.0, 8.0 / max(condition, 1.0))
    residual_score = float(np.exp(-residual / 20.0))
    confidence = float(
        np.clip(
            min(camera_a.confidence, camera_b.confidence)
            * geometry_score
            * residual_score,
            0.0,
            1.0,
        )
    )
    if confidence < 0.2:
        return None
    return StereoSpinEstimate(
        spin_world_rad_s=spin,
        confidence=confidence,
        backspin_rpm=backspin_rpm,
        sidespin_rpm=sidespin_rpm,
        spin_axis_deg=spin_axis,
        total_spin_rpm=total_rpm,
        projection_residual_rad_s=residual,
    )


__all__ = [
    "CameraSpinEstimate",
    "StereoSpinEstimate",
    "estimate_camera_spin",
    "estimate_camera_spin_from_frames",
    "reconstruct_stereo_spin",
]
