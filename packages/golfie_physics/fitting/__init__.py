"""Fit physics parameters (initial velocity, optionally spin) to measured
early 3D trajectory points.

Fits launch position and velocity to early world-frame measurements using a
constrained drag-enabled flight model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from golfie_core.schemas import TrackedPoint3D
from golfie_physics.models.params import FlightParams


@dataclass
class FitResult:
    initial_position_m: np.ndarray
    initial_velocity_mps: np.ndarray
    params: FlightParams
    residual_rms_m: float
    confidence: float
    optimizer_success: bool = True
    optimizer_message: str = ""
    active_bounds: tuple[str, ...] = ()


def fit_initial_conditions(
    measured_points: list[TrackedPoint3D],
    time_origin_seconds: float | None = None,
) -> FitResult:
    """Fit launch position/velocity (and optionally drag/lift coefficients
    within realistic bounds) to a short early-flight 3D point sequence.

    Spec section 13: use constrained optimization (e.g.
    scipy.optimize.least_squares with bounds), penalize physically
    implausible values, and report confidence/uncertainty rather than a
    bare number. Must not silently extrapolate past the available data
    with false confidence.
    """
    from scipy.optimize import least_squares
    from golfie_physics.models.projectile import simulate_flight

    if len(measured_points) < 3:
        raise ValueError(
            f"Trajectory fitting requires at least 3 points, got {len(measured_points)}."
        )

    t_absolute = np.array([p.time_seconds for p in measured_points], dtype=np.float64)
    if np.any(np.diff(t_absolute) <= 0):
        raise ValueError("Trajectory timestamps must be strictly increasing.")
    # Synthetic/launch-relative tracks begin near t=0. Real video timestamps
    # can be minutes into a recording, so callers provide a local origin.
    if time_origin_seconds is None:
        time_origin_seconds = 0.0 if t_absolute[0] < 5.0 else float(t_absolute[0])
    t_meas = t_absolute - float(time_origin_seconds)
    if np.any(t_meas < -1e-7):
        raise ValueError("Physics time origin cannot be after the first measured point.")

    x_meas = np.array([p.x_m for p in measured_points], dtype=np.float64)
    y_meas = np.array([p.y_m for p in measured_points], dtype=np.float64)
    z_meas = np.array([p.z_m for p in measured_points], dtype=np.float64)

    # Estimate velocity using finite differences
    dt_total = t_meas[-1] - t_meas[0]
    if dt_total > 1e-6:
        vel_guess = np.array([
            (x_meas[-1] - x_meas[0]) / dt_total,
            (y_meas[-1] - y_meas[0]) / dt_total,
            (z_meas[-1] - z_meas[0]) / dt_total,
        ], dtype=np.float64)
    else:
        vel_guess = np.array([30.0, 0.0, 10.0], dtype=np.float64)

    # Estimate state at the chosen local time origin.
    pos_guess = (
        np.array([x_meas[0], y_meas[0], z_meas[0]], dtype=np.float64)
        - vel_guess * t_meas[0]
    )

    theta0 = np.concatenate([pos_guess, vel_guess])

    # Keep the initial state near the first triangulated point while allowing
    # measurement noise.  Tee/world alignment is a separate calibration step;
    # hard-coding the origin here made absolute video timestamps impossible.
    lower_bounds = [pos_guess[0] - 0.5, pos_guess[1] - 0.5, pos_guess[2] - 0.5, -95.0, -60.0, -50.0]
    upper_bounds = [pos_guess[0] + 0.5, pos_guess[1] + 0.5, pos_guess[2] + 0.5, 95.0, 60.0, 60.0]
    theta0 = np.clip(theta0, lower_bounds, upper_bounds)

    params = FlightParams(drag_enabled=True, lift_enabled=False)

    def residuals(theta):
        pos = theta[:3]
        vel = theta[3:]

        # Simulate trajectory starting from the tee (t=0.0)
        res_flight = simulate_flight(
            initial_position_m=pos,
            initial_velocity_mps=vel,
            params=params,
            max_time_s=float(t_meas[-1] + 0.05),
            dt=0.001
        )

        sim_times = np.array([s.time_s for s in res_flight.samples], dtype=np.float64)
        sim_pos = np.array([s.position_m for s in res_flight.samples], dtype=np.float64)

        if len(sim_pos) == 0:
            return np.ones(3 * len(t_meas)) * 999.0

        sim_x = sim_pos[:, 0]
        sim_y = sim_pos[:, 1]
        sim_z = sim_pos[:, 2]

        int_x = np.interp(t_meas, sim_times, sim_x)
        int_y = np.interp(t_meas, sim_times, sim_y)
        int_z = np.interp(t_meas, sim_times, sim_z)

        rx = int_x - x_meas
        ry = int_y - y_meas
        rz = int_z - z_meas

        return np.concatenate([rx, ry, rz])

    # Run optimization
    opt_res = least_squares(residuals, theta0, bounds=(lower_bounds, upper_bounds))
    theta_opt = opt_res.x

    # Calculate statistics
    opt_residuals = residuals(theta_opt)
    rms = float(np.sqrt(np.mean(opt_residuals ** 2)))

    # Scientific confidence score
    confidence = max(0.0, 1.0 - rms / 0.05)
    confidence *= min(1.0, len(measured_points) / 5.0)

    parameter_names = ("x0", "y0", "z0", "vx", "vy", "vz")
    active_bounds = tuple(
        name
        for name, value, lower, upper in zip(
            parameter_names, theta_opt, lower_bounds, upper_bounds
        )
        if np.isclose(value, lower, atol=1e-4, rtol=0.0)
        or np.isclose(value, upper, atol=1e-4, rtol=0.0)
    )

    return FitResult(
        initial_position_m=theta_opt[:3],
        initial_velocity_mps=theta_opt[3:],
        params=params,
        residual_rms_m=rms,
        confidence=float(confidence),
        optimizer_success=bool(opt_res.success),
        optimizer_message=str(opt_res.message),
        active_bounds=active_bounds,
    )
