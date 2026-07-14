"""Experimental post-landing bounce and roll estimation.

Golfie does not currently measure spin or turf firmness, so a detailed contact
solve would imply precision the inputs cannot support.  The run-distance anchor
is the R&A/USGA proposed linear turf model (yards):

    run = 79.1 - 1.6 * terminal_angle_degrees

It is scaled by terminal horizontal speed outside the narrow Overall Distance
Standard launch regime.  The returned samples provide a visually continuous,
clearly experimental two-bounce-and-roll replay ending at that estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from golfie_physics.models.projectile import FlightResult, FlightSample


@dataclass
class GroundRunResult:
    samples: list[FlightSample]
    run_m: float
    landing_angle_deg: float


def estimate_ground_run(flight: FlightResult) -> GroundRunResult:
    landing = flight.landing_sample
    velocity = np.asarray(landing.velocity_mps, dtype=float)
    horizontal = velocity[:2]
    horizontal_speed = float(np.linalg.norm(horizontal))
    vertical_speed = abs(float(velocity[2]))
    landing_angle = float(np.degrees(np.arctan2(vertical_speed, max(horizontal_speed, 1e-6))))

    # R&A/USGA model is calibrated around driver terminal conditions. Retain
    # its terminal-angle dependence, but reduce rollout for materially slower
    # arrivals. A small minimum keeps total distinct from carry for soft,
    # steep landings without inventing a long wedge rollout.
    base_run_m = max(0.0, 79.1 - 1.6 * landing_angle) * 0.9144
    speed_scale = float(np.clip(horizontal_speed / 35.0, 0.35, 1.15))
    run_m = max(0.5, base_run_m * speed_scale)

    direction = horizontal / horizontal_speed if horizontal_speed > 1e-6 else np.array([1.0, 0.0])
    origin = np.asarray(landing.position_m, dtype=float).copy()
    origin[2] = 0.0
    start_time = float(landing.time_s)
    samples: list[FlightSample] = []

    phases = (
        (0.00, 0.52, 0.00, 0.52, min(1.4, max(0.12, vertical_speed**2 * 0.0028))),
        (0.52, 0.78, 0.52, 0.92, min(0.42, max(0.05, vertical_speed**2 * 0.0008))),
        (0.78, 1.00, 0.92, 2.15, 0.0),
    )
    previous_position = origin
    previous_time = start_time
    for distance_start, distance_end, time_start, time_end, bounce_height in phases:
        for fraction in np.linspace(0.0, 1.0, 25, endpoint=False):
            # Rolling eases out; airborne phases retain approximately uniform
            # horizontal motion and use a parabolic bounce arc.
            eased = 1.0 - (1.0 - fraction) ** 2 if bounce_height == 0.0 else fraction
            distance_fraction = distance_start + (distance_end - distance_start) * eased
            time_s = start_time + time_start + (time_end - time_start) * fraction
            position = origin.copy()
            position[:2] += direction * (run_m * distance_fraction)
            position[2] = bounce_height * 4.0 * fraction * (1.0 - fraction)
            dt = max(time_s - previous_time, 1e-6)
            sample_velocity = (position - previous_position) / dt
            samples.append(FlightSample(time_s=time_s, position_m=position, velocity_mps=sample_velocity))
            previous_position, previous_time = position, time_s

    final_position = origin.copy()
    final_position[:2] += direction * run_m
    samples.append(
        FlightSample(
            time_s=start_time + phases[-1][3],
            position_m=final_position,
            velocity_mps=np.zeros(3, dtype=float),
        )
    )
    return GroundRunResult(samples=samples, run_m=run_m, landing_angle_deg=landing_angle)
