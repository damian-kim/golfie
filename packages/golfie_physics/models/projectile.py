"""Ball flight model: gravity (Phase 1), drag (Phase 2), optional Magnus
lift (Phase 3, experimental). Ground interaction (Phase 4: bounce/roll)
is NOT implemented -- simulation stops at the first ground crossing.

State vector layout: y = [x, y, z, vx, vy, vz] in the Golfie world frame
(meters, meters/second; see golfie_core.coordinates).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from golfie_core.schemas import MetricSource, TrackedPoint3D
from golfie_physics.integrators.rk4 import integrate
from golfie_physics.models.params import FlightParams


@dataclass
class FlightSample:
    time_s: float
    position_m: np.ndarray  # shape (3,)
    velocity_mps: np.ndarray  # shape (3,)


@dataclass
class FlightResult:
    samples: list[FlightSample]
    params: FlightParams

    @property
    def landing_sample(self) -> FlightSample:
        return self.samples[-1]

    @property
    def carry_m(self) -> float:
        """Horizontal distance from launch to first ground contact."""
        start = self.samples[0].position_m
        end = self.landing_sample.position_m
        return float(np.linalg.norm(end[:2] - start[:2]))

    @property
    def apex_m(self) -> float:
        return float(max(s.position_m[2] for s in self.samples))

    @property
    def side_deviation_m(self) -> float:
        """Lateral (+/-Y) offset of the landing point from the target line."""
        return float(self.landing_sample.position_m[1])

    def to_tracked_points(
        self,
        source: "MetricSource" = MetricSource.EXPERIMENTAL,
        confidence: float = 1.0,
        max_points: int | None = 400,
    ) -> list[TrackedPoint3D]:
        """Convert to the shared TrackedPoint3D schema for storage/rendering.

        `confidence` here describes confidence in the *simulation*, not a
        measurement -- a pure physics simulation should normally be
        labeled experimental/estimated, never measured. Downsamples to
        `max_points` (evenly spaced) so a multi-second flight at 1ms
        steps doesn't ship thousands of points to the frontend.
        """
        samples = self.samples
        if max_points is not None and len(samples) > max_points:
            idx = np.linspace(0, len(samples) - 1, max_points).round().astype(int)
            samples = [samples[i] for i in idx]
        return [
            TrackedPoint3D(
                time_seconds=s.time_s,
                x_m=float(s.position_m[0]),
                y_m=float(s.position_m[1]),
                z_m=float(max(s.position_m[2], 0.0)),
                confidence=confidence,
            )
            for s in samples
        ]


def _acceleration(
    velocity_mps: np.ndarray,
    params: FlightParams,
    spin_rad_s: np.ndarray | None,
) -> np.ndarray:
    """Net acceleration on the ball: gravity + drag + (optional) Magnus."""
    accel = np.array([0.0, 0.0, -params.gravity_mps2])

    speed = float(np.linalg.norm(velocity_mps))
    if params.drag_enabled and speed > 1e-9:
        drag_force_mag = (
            0.5 * params.air_density_kg_m3 * params.drag_coefficient * params.cross_section_m2 * speed**2
        )
        accel += -(drag_force_mag / params.mass_kg) * (velocity_mps / speed)

    if params.lift_enabled and spin_rad_s is not None and speed > 1e-9:
        # Simplified Magnus model: lift acts along spin x velocity,
        # magnitude scaled by lift_coefficient. This is a coarse
        # approximation (no spin-ratio-dependent Cl curve) and must stay
        # labeled experimental until validated against real data.
        cross = np.cross(spin_rad_s, velocity_mps)
        cross_norm = np.linalg.norm(cross)
        if cross_norm > 1e-9:
            lift_force_mag = (
                0.5 * params.air_density_kg_m3 * params.lift_coefficient * params.cross_section_m2 * speed**2
            )
            accel += (lift_force_mag / params.mass_kg) * (cross / cross_norm)

    return accel


def simulate_flight(
    initial_position_m: np.ndarray,
    initial_velocity_mps: np.ndarray,
    params: FlightParams | None = None,
    spin_rad_s: np.ndarray | None = None,
    dt: float = 0.001,
    max_time_s: float = 12.0,
) -> FlightResult:
    """Integrate the ball flight forward until it first reaches ground level (z=0).

    Uses RK4 (spec section 13). `dt=0.001s` (1ms) by default, which is
    small enough to be stable for realistic golf launch speeds while
    staying fast for an MVP; tighten further if validation shows it's
    needed.
    """
    params = params or FlightParams()
    y0 = np.concatenate([np.asarray(initial_position_m, dtype=float), np.asarray(initial_velocity_mps, dtype=float)])

    def derivative(_t: float, y: np.ndarray) -> np.ndarray:
        vel = y[3:6]
        acc = _acceleration(vel, params, spin_rad_s)
        return np.concatenate([vel, acc])

    def hit_ground(t: float, y: np.ndarray) -> bool:
        # Stop once below ground level (after the first step away from
        # z=0 at launch) or once we exceed the time budget.
        return (t > 0.0 and y[2] <= 0.0) or t >= max_time_s

    raw_samples = integrate(derivative, y0, t0=0.0, dt=dt, stop_condition=hit_ground)

    samples = [
        FlightSample(time_s=t, position_m=y[0:3], velocity_mps=y[3:6]) for t, y in raw_samples
    ]
    return FlightResult(samples=samples, params=params)


def simulate_from_launch_conditions(
    ball_speed_mps: float,
    launch_angle_deg: float,
    horizontal_launch_deg: float,
    params: FlightParams | None = None,
    spin_rad_s: np.ndarray | None = None,
) -> FlightResult:
    """Convenience wrapper: build the initial velocity vector from launch
    monitor-style conditions (speed, vertical launch angle, horizontal
    launch direction) instead of a raw velocity vector.

    Angle convention matches golfie_core.coordinates: 0 deg horizontal
    launch = straight down the +X target line, positive = right of
    target (toward +Y) for a right-handed frame with +Z up.
    """
    theta = np.radians(launch_angle_deg)
    phi = np.radians(horizontal_launch_deg)
    vx = ball_speed_mps * np.cos(theta) * np.cos(phi)
    vy = ball_speed_mps * np.cos(theta) * np.sin(phi)
    vz = ball_speed_mps * np.sin(theta)
    return simulate_flight(
        initial_position_m=np.array([0.0, 0.0, 0.0]),
        initial_velocity_mps=np.array([vx, vy, vz]),
        params=params,
        spin_rad_s=spin_rad_s,
    )
