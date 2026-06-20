"""Synthetic ground-truth generation and accuracy validation utilities.

STATUS: stub (mostly). Implemented in Milestone 10 (spec section 18, 20).

`generate_synthetic_flight` is a thin convenience wrapper around the
already-real `simulate_flight` -- useful today for building demo/sample
trajectories. The actual *validation* utilities (projecting a known 3D
trajectory into synthetic stereo camera views to test the calibration +
triangulation + sync pipeline end-to-end) are not implemented yet.
"""

from __future__ import annotations

import numpy as np

from golfie_physics.models.params import FlightParams
from golfie_physics.models.projectile import FlightResult, simulate_from_launch_conditions


def generate_synthetic_flight(
    ball_speed_mps: float,
    launch_angle_deg: float,
    horizontal_launch_deg: float,
    params: FlightParams | None = None,
) -> FlightResult:
    """Generate a known-ground-truth flight for use as demo data or as
    input to a (future) synthetic camera projection test."""
    return simulate_from_launch_conditions(
        ball_speed_mps=ball_speed_mps,
        launch_angle_deg=launch_angle_deg,
        horizontal_launch_deg=horizontal_launch_deg,
        params=params,
    )


def project_to_synthetic_cameras(
    flight: FlightResult,
    calibration: CalibrationResult,
    fps: float = 240.0,
) -> tuple[list[TrackedPoint2D], list[TrackedPoint2D]]:
    """Project a known 3D trajectory into two synthetic camera views to
    test calibration/triangulation/sync noise sensitivity end-to-end.

    Spec section 18 ("Synthetic tests").
    """
    from golfie_core.coordinates import CoordinateTransformer
    from golfie_core.schemas import TrackedPoint2D

    if not calibration.is_valid:
        raise ValueError("Calibration is not valid.")
    if calibration.camera_a_intrinsics is None or calibration.camera_b_intrinsics is None:
        raise ValueError("Missing camera intrinsics in calibration.")
    if calibration.camera_a_extrinsics is None or calibration.camera_b_extrinsics is None:
        raise ValueError("Missing camera extrinsics in calibration.")

    K_a = np.array(calibration.camera_a_intrinsics, dtype=np.float64)
    K_b = np.array(calibration.camera_b_intrinsics, dtype=np.float64)
    ext_a = np.array(calibration.camera_a_extrinsics, dtype=np.float64)
    ext_b = np.array(calibration.camera_b_extrinsics, dtype=np.float64)

    P_a = K_a @ ext_a[:3, :]
    P_b = K_b @ ext_b[:3, :]

    if calibration.coordinate_system is not None:
        transformer = CoordinateTransformer.from_coordinate_system(calibration.coordinate_system)
    else:
        transformer = CoordinateTransformer.identity()

    track_a = []
    track_b = []

    for sample in flight.samples:
        t = sample.time_s
        p_world = sample.position_m
        p_rig = transformer.to_rig(p_world)

        # Check positive depth in Camera A
        if p_rig[2] <= 0.05:
            continue

        # Check positive depth in Camera B
        p_rig_hom = np.append(p_rig, 1.0)
        p_cam_b = ext_b[:3, :] @ p_rig_hom
        if p_cam_b[2] <= 0.05:
            continue

        # Project to Camera A
        proj_a = P_a @ p_rig_hom
        u_a = proj_a[0] / proj_a[2]
        v_a = proj_a[1] / proj_a[2]

        # Project to Camera B
        proj_b = P_b @ p_rig_hom
        u_b = proj_b[0] / proj_b[2]
        v_b = proj_b[1] / proj_b[2]

        f_idx = int(round(t * fps))

        track_a.append(
            TrackedPoint2D(
                frame_index=f_idx,
                time_seconds=t,
                x_px=float(u_a),
                y_px=float(v_a),
                confidence=1.0,
            )
        )
        track_b.append(
            TrackedPoint2D(
                frame_index=f_idx,
                time_seconds=t,
                x_px=float(u_b),
                y_px=float(v_b),
                confidence=1.0,
            )
        )

    return track_a, track_b
