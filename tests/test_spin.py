import cv2
import numpy as np
import pytest

from golfie_core.schemas import CalibrationResult, CoordinateSystem, TrackedPoint2D
from golfie_cv.spin import (
    CameraSpinEstimate,
    estimate_camera_spin_from_frames,
    reconstruct_stereo_spin,
)


def test_marked_ball_image_rotation_estimates_optical_axis_spin():
    center = (110, 110)
    radius = 42
    base = np.full((220, 220, 3), 35, dtype=np.uint8)
    cv2.circle(base, center, radius, (225, 225, 225), -1)
    cv2.line(base, (68, 110), (152, 110), (10, 10, 10), 3)
    cv2.line(base, (110, 68), (110, 152), (10, 10, 10), 3)
    for point in ((88, 88), (132, 91), (91, 132), (130, 130), (105, 82), (140, 112)):
        cv2.circle(base, point, 3, (15, 15, 15), -1)

    fps = 240.0
    degrees_per_frame = 3.0
    frames = {}
    track = []
    for frame_index in range(7):
        matrix = cv2.getRotationMatrix2D(center, frame_index * degrees_per_frame, 1.0)
        frames[frame_index] = cv2.warpAffine(
            base, matrix, (220, 220), flags=cv2.INTER_LINEAR, borderValue=(35, 35, 35)
        )
        track.append(
            TrackedPoint2D(
                frame_index=frame_index,
                time_seconds=frame_index / fps,
                x_px=center[0],
                y_px=center[1],
                radius_px=radius,
                confidence=0.95,
            )
        )

    estimate = estimate_camera_spin_from_frames(frames, track)

    assert estimate is not None
    measured_rpm = abs(estimate.optical_axis_rad_s) * 60.0 / (2.0 * np.pi)
    expected_rpm = degrees_per_frame * fps / 6.0
    assert measured_rpm == pytest.approx(expected_rpm, rel=0.2)
    assert estimate.sample_count >= 3
    assert estimate.confidence >= 0.2


def test_stereo_spin_reconstructs_world_backspin_and_sidespin():
    identity = np.eye(4)
    camera_b = np.eye(4)
    # Camera B's optical axis is world +Y (third row of world-to-camera R).
    camera_b[:3, :3] = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    calibration = CalibrationResult(
        coordinate_system=CoordinateSystem(
            origin_in_rig_frame_m=[0.0, 0.0, 0.0],
            target_direction_in_rig_frame=[1.0, 0.0, 0.0],
            up_direction_in_rig_frame=[0.0, 0.0, 1.0],
            alignment_method="test",
        ),
        camera_a_extrinsics=identity.tolist(),
        camera_b_extrinsics=camera_b.tolist(),
        is_valid=True,
    )
    camera_a_spin = CameraSpinEstimate(50.0, 0.9, 5, 12, 1.0)
    camera_b_spin = CameraSpinEstimate(-200.0, 0.85, 5, 12, 1.0)

    result = reconstruct_stereo_spin(
        camera_a_spin,
        camera_b_spin,
        calibration,
        np.array([60.0, 0.0, 0.0]),
    )

    assert result is not None
    assert result.spin_world_rad_s == pytest.approx([0.0, -200.0, 50.0], abs=1e-8)
    assert result.backspin_rpm == pytest.approx(200.0 * 60.0 / (2.0 * np.pi))
    assert result.sidespin_rpm == pytest.approx(50.0 * 60.0 / (2.0 * np.pi))
    assert result.confidence > 0.5

