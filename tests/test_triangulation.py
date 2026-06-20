import numpy as np
import pytest

from golfie_core.schemas import (
    CalibrationResult,
    CoordinateSystem,
    TrackedPoint2D,
    TrackedPoint3D,
)
from golfie_cv.triangulation import triangulate_track


def test_triangulate_track_perfect_recovery():
    # Setup camera matrices
    # Camera A at origin, looking along Z
    K_a = [
        [1000.0, 0.0, 500.0],
        [0.0, 1000.0, 500.0],
        [0.0, 0.0, 1.0]
    ]
    ext_a = np.eye(4).tolist()

    # Camera B shifted by 1m in X (so T = [-1, 0, 0]), looking along Z
    K_b = [
        [1000.0, 0.0, 500.0],
        [0.0, 1000.0, 500.0],
        [0.0, 0.0, 1.0]
    ]
    ext_b = [
        [1.0, 0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ]

    calibration = CalibrationResult(
        coordinate_system=CoordinateSystem(
            origin_in_rig_frame_m=[0.0, 0.0, 0.0],
            target_direction_in_rig_frame=[1.0, 0.0, 0.0],
            up_direction_in_rig_frame=[0.0, 0.0, 1.0],
            alignment_method="manual",
        ),
        camera_a_intrinsics=K_a,
        camera_b_intrinsics=K_b,
        camera_a_extrinsics=ext_a,
        camera_b_extrinsics=ext_b,
        is_valid=True,
    )

    # Generate synthetic 3D trajectory where:
    # x is horizontal, y is vertical, z is depth (looking along Z-axis)
    fps = 100.0
    track_a = []
    track_b = []
    true_points = []

    for f in range(10):
        t = f / fps
        x = 0.5 + 2.0 * t
        y = 1.0 + 3.0 * t - 4.9 * (t ** 2)
        z = 5.0
        true_points.append((x, y, z))

        # Project to Camera A
        # P_A = [x, y, z]
        pt_a_u = 1000.0 * x / z + 500.0
        pt_a_v = 1000.0 * y / z + 500.0
        track_a.append(
            TrackedPoint2D(
                frame_index=f,
                time_seconds=t,
                x_px=pt_a_u,
                y_px=pt_a_v,
                confidence=0.9,
            )
        )

        # Project to Camera B
        # P_B = [x - 1.0, y, z]
        pt_b_u = 1000.0 * (x - 1.0) / z + 500.0
        pt_b_v = 1000.0 * y / z + 500.0
        track_b.append(
            TrackedPoint2D(
                frame_index=f,
                time_seconds=t,
                x_px=pt_b_u,
                y_px=pt_b_v,
                confidence=0.95,
            )
        )

    res = triangulate_track(track_a, track_b, calibration)

    assert len(res) == 10
    for idx, pt in enumerate(res):
        assert pt.time_seconds == pytest.approx(idx / fps)
        assert pt.x_m == pytest.approx(true_points[idx][0], abs=1e-5)
        assert pt.y_m == pytest.approx(true_points[idx][1], abs=1e-5)
        assert pt.z_m == pytest.approx(true_points[idx][2], abs=1e-5)
        assert pt.reprojection_error_px is not None
        assert pt.reprojection_error_px < 1e-3
        assert pt.confidence == pytest.approx(np.sqrt(0.9 * 0.95))


def test_triangulate_track_repro_error_filtering():
    # Setup same calibration
    K = [[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0], [0.0, 0.0, 1.0]]
    ext_a = np.eye(4).tolist()
    ext_b = [[1.0, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]

    calibration = CalibrationResult(
        camera_a_intrinsics=K,
        camera_b_intrinsics=K,
        camera_a_extrinsics=ext_a,
        camera_b_extrinsics=ext_b,
        is_valid=True,
    )

    # Frame 0: perfect point
    # Frame 1: mismatched point in Camera B (causes huge reprojection error because y-coordinates skew)
    track_a = [
        TrackedPoint2D(frame_index=0, time_seconds=0.0, x_px=600.0, y_px=700.0, confidence=0.8),
        TrackedPoint2D(frame_index=1, time_seconds=0.1, x_px=600.0, y_px=700.0, confidence=0.8),
    ]
    track_b = [
        TrackedPoint2D(frame_index=0, time_seconds=0.0, x_px=400.0, y_px=700.0, confidence=0.8),
        TrackedPoint2D(frame_index=1, time_seconds=0.1, x_px=400.0, y_px=900.0, confidence=0.8),  # shifted way off vertically
    ]

    res = triangulate_track(track_a, track_b, calibration)
    
    # Should only keep frame 0, frame 1 should be filtered out
    assert len(res) == 1
    assert res[0].time_seconds == pytest.approx(0.0)


def test_triangulate_track_invalid_calibration():
    calibration = CalibrationResult(is_valid=False)
    with pytest.raises(ValueError, match="Calibration is not valid"):
        triangulate_track([], [], calibration)

    calibration_partial = CalibrationResult(is_valid=True)
    with pytest.raises(ValueError, match="Missing camera intrinsics"):
        triangulate_track([], [], calibration_partial)
