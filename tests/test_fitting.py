import numpy as np
import pytest

from golfie_core.schemas import TrackedPoint3D
from golfie_physics.fitting import fit_initial_conditions
from golfie_physics.models import FlightParams, simulate_from_launch_conditions


def test_fit_initial_conditions_perfect():
    # 1. Generate a ground truth trajectory using the real physics engine
    # Launch: speed=50.0 m/s, launch angle=12.0 deg, deviation=2.0 deg (along Y)
    params = FlightParams(drag_enabled=True, lift_enabled=False)
    flight = simulate_from_launch_conditions(
        ball_speed_mps=50.0,
        launch_angle_deg=12.0,
        horizontal_launch_deg=2.0,
        params=params,
    )
    
    # 2. Extract early tracked points (e.g. first 15 points at 5ms intervals)
    # The simulation step size is 1ms, so we sample every 5 steps
    samples = flight.samples[:75:5]
    measured_points = [
        TrackedPoint3D(
            time_seconds=s.time_s,
            x_m=s.position_m[0],
            y_m=s.position_m[1],
            z_m=s.position_m[2],
            confidence=0.95
        )
        for s in samples
    ]

    # 3. Fit the trajectory
    res = fit_initial_conditions(measured_points)

    # 4. Assert exact parameter recovery
    # Initial position should be at origin
    assert res.initial_position_m == pytest.approx([0.0, 0.0, 0.0], abs=1e-3)
    
    # Verify velocity vector matches the launch conditions
    vx = 50.0 * np.cos(np.radians(12.0)) * np.cos(np.radians(2.0))
    vy = 50.0 * np.cos(np.radians(12.0)) * np.sin(np.radians(2.0))
    vz = 50.0 * np.sin(np.radians(12.0))
    
    assert res.initial_velocity_mps[0] == pytest.approx(vx, abs=1e-2)
    assert res.initial_velocity_mps[1] == pytest.approx(vy, abs=1e-2)
    assert res.initial_velocity_mps[2] == pytest.approx(vz, abs=1e-2)
    
    # RMS error should be practically 0 and confidence should be near 1.0
    assert res.residual_rms_m < 5e-3
    assert res.confidence > 0.9


def test_fit_initial_conditions_with_noise():
    # 1. Generate a ground truth trajectory
    params = FlightParams(drag_enabled=True, lift_enabled=False)
    flight = simulate_from_launch_conditions(
        ball_speed_mps=45.0,
        launch_angle_deg=15.0,
        horizontal_launch_deg=0.0,
        params=params,
    )
    
    # 2. Sample 10 points and add Gaussian noise (e.g., standard deviation of 1 cm)
    np.random.seed(42)
    samples = flight.samples[:60:6]
    measured_points = []
    
    for s in samples:
        noise = np.random.normal(0, 0.01, 3)  # 1 cm noise in x, y, z
        measured_points.append(
            TrackedPoint3D(
                time_seconds=s.time_s,
                x_m=s.position_m[0] + noise[0],
                y_m=s.position_m[1] + noise[1],
                z_m=s.position_m[2] + noise[2],
                confidence=0.9
            )
        )

    # 3. Fit
    res = fit_initial_conditions(measured_points)

    # 4. Assert that values converged close to truth
    vx = 45.0 * np.cos(np.radians(15.0))
    vy = 0.0
    vz = 45.0 * np.sin(np.radians(15.0))

    assert res.initial_position_m == pytest.approx([0.0, 0.0, 0.0], abs=0.03)
    assert res.initial_velocity_mps[0] == pytest.approx(vx, abs=1.0)
    assert res.initial_velocity_mps[1] == pytest.approx(vy, abs=1.0)
    assert res.initial_velocity_mps[2] == pytest.approx(vz, abs=1.0)
    
    # RMS error should be close to the 1 cm noise we introduced, and confidence should be high
    assert res.residual_rms_m == pytest.approx(0.01, abs=0.01)
    assert res.confidence > 0.6


def test_fit_initial_conditions_insufficient_points():
    measured_points = [
        TrackedPoint3D(time_seconds=0.0, x_m=0.0, y_m=0.0, z_m=0.0),
        TrackedPoint3D(time_seconds=0.1, x_m=1.0, y_m=0.0, z_m=0.0),
    ]
    with pytest.raises(ValueError, match="fitting requires at least 3 points"):
        fit_initial_conditions(measured_points)
