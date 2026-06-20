import numpy as np
import pytest

from golfie_core.schemas import CalibrationResult, CoordinateSystem
from golfie_cv.triangulation import triangulate_track
from golfie_physics.fitting import fit_initial_conditions
from golfie_physics.models import FlightParams
from golfie_physics.validation import generate_synthetic_flight, project_to_synthetic_cameras


@pytest.fixture
def test_calibration():
    # Setup standard virtual camera system:
    # Camera A at origin, looking down Z
    K = [[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0], [0.0, 0.0, 1.0]]
    ext_a = np.eye(4).tolist()
    # Camera B shifted by 1.0 m in X, looking down Z
    ext_b = [
        [1.0, 0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ]

    return CalibrationResult(
        coordinate_system=CoordinateSystem(
            origin_in_rig_frame_m=[0.0, 0.0, 0.0],
            target_direction_in_rig_frame=[1.0, 0.0, 0.0],
            up_direction_in_rig_frame=[0.0, 0.0, 1.0],
            alignment_method="manual",
        ),
        camera_a_intrinsics=K,
        camera_b_intrinsics=K,
        camera_a_extrinsics=ext_a,
        camera_b_extrinsics=ext_b,
        is_valid=True,
    )


def test_validation_pipeline_perfect_recovery(test_calibration):
    # 1. Generate synthetic flight trajectory
    params = FlightParams(drag_enabled=True, lift_enabled=False)
    flight = generate_synthetic_flight(
        ball_speed_mps=50.0,
        launch_angle_deg=10.0,
        horizontal_launch_deg=3.0,
        params=params,
    )

    # 2. Project trajectory to synthetic cameras (simulate 240 fps)
    track_a, track_b = project_to_synthetic_cameras(flight, test_calibration, fps=240.0)

    # Ensure we got points and they are correctly ordered
    assert len(track_a) > 10
    assert len(track_b) > 10

    # 3. Triangulate the 2D tracks back to 3D rig/world points
    # (Since we only want early trajectory for fitting, we can slice it)
    triangulated_points = triangulate_track(track_a[:30], track_b[:30], test_calibration)
    
    assert len(triangulated_points) > 5

    # 4. Fit the launch conditions
    fit_res = fit_initial_conditions(triangulated_points)

    # 5. Assert perfect recovery
    assert fit_res.initial_position_m == pytest.approx([0.0, 0.0, 0.0], abs=1e-3)
    
    # Expected launch velocity vector
    vx = 50.0 * np.cos(np.radians(10.0)) * np.cos(np.radians(3.0))
    vy = 50.0 * np.cos(np.radians(10.0)) * np.sin(np.radians(3.0))
    vz = 50.0 * np.sin(np.radians(10.0))

    assert fit_res.initial_velocity_mps[0] == pytest.approx(vx, abs=1e-2)
    assert fit_res.initial_velocity_mps[1] == pytest.approx(vy, abs=1e-2)
    assert fit_res.initial_velocity_mps[2] == pytest.approx(vz, abs=1e-2)
    assert fit_res.residual_rms_m < 1e-3
    assert fit_res.confidence > 0.95


def test_validation_pipeline_noise_sensitivity(test_calibration):
    # 1. Generate synthetic flight
    params = FlightParams(drag_enabled=True, lift_enabled=False)
    flight = generate_synthetic_flight(
        ball_speed_mps=60.0,
        launch_angle_deg=14.0,
        horizontal_launch_deg=-2.0,
        params=params,
    )

    # 2. Project to 2D
    track_a, track_b = project_to_synthetic_cameras(flight, test_calibration, fps=240.0)

    # 3. Add Gaussian pixel jitter (1.0 pixel standard deviation noise)
    np.random.seed(1337)
    for pt in track_a:
        pt.x_px += np.random.normal(0.0, 1.0)
        pt.y_px += np.random.normal(0.0, 1.0)
    for pt in track_b:
        pt.x_px += np.random.normal(0.0, 1.0)
        pt.y_px += np.random.normal(0.0, 1.0)

    # 4. Triangulate the noisy tracks
    triangulated_points = triangulate_track(track_a[:25], track_b[:25], test_calibration)

    # 5. Fit the initial conditions
    fit_res = fit_initial_conditions(triangulated_points)

    # 6. Assert that the fit recovered initial conditions within reasonable bounds
    vx = 60.0 * np.cos(np.radians(14.0)) * np.cos(np.radians(-2.0))
    vy = 60.0 * np.cos(np.radians(14.0)) * np.sin(np.radians(-2.0))
    vz = 60.0 * np.sin(np.radians(14.0))

    # Noise introduces minor offsets, but constrained optimization must still converge
    assert fit_res.initial_position_m == pytest.approx([0.0, 0.0, 0.0], abs=0.05)
    assert fit_res.initial_velocity_mps[0] == pytest.approx(vx, abs=2.5)
    assert fit_res.initial_velocity_mps[1] == pytest.approx(vy, abs=1.5)
    assert fit_res.initial_velocity_mps[2] == pytest.approx(vz, abs=1.5)
    
    # Verify that the RMS error reflects the introduced noise
    assert fit_res.residual_rms_m > 0.0  # must have some error
    assert fit_res.confidence < 1.0      # confidence should not be perfect
