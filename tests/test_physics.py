import numpy as np
import pytest

from golfie_physics.integrators.rk4 import integrate, rk4_step
from golfie_physics.models import FlightParams, simulate_from_launch_conditions


def test_rk4_step_matches_exponential_decay_closed_form():
    """dy/dt = -k*y has the closed form y(t) = y0 * exp(-k*t). A single
    RK4 step should match it to ~4th-order accuracy for a reasonable dt.
    """
    k = 2.0
    y0 = np.array([1.0])

    def f(_t, y):
        return -k * y

    dt = 0.01
    y1 = rk4_step(f, 0.0, y0, dt)
    expected = y0 * np.exp(-k * dt)
    assert np.allclose(y1, expected, atol=1e-6)


def test_integrate_stops_at_condition():
    y0 = np.array([0.0])

    def f(_t, _y):
        return np.array([1.0])  # dy/dt = 1 -> y(t) = t

    samples = integrate(f, y0, t0=0.0, dt=0.1, stop_condition=lambda t, _y: t >= 1.0)
    final_t, final_y = samples[-1]
    assert final_t >= 1.0
    assert np.isclose(final_y[0], final_t, atol=1e-9)


@pytest.mark.parametrize("speed,angle", [(30.0, 10.0), (50.0, 25.0), (20.0, 45.0)])
def test_no_drag_projectile_matches_closed_form_range_and_apex(speed, angle):
    params = FlightParams(drag_enabled=False, lift_enabled=False)
    result = simulate_from_launch_conditions(speed, angle, 0.0, params=params)

    theta = np.radians(angle)
    g = params.gravity_mps2
    expected_range = speed**2 * np.sin(2 * theta) / g
    expected_apex = speed**2 * np.sin(theta) ** 2 / (2 * g)

    assert result.carry_m == pytest.approx(expected_range, rel=0.01)
    assert result.apex_m == pytest.approx(expected_apex, rel=0.01)


def test_drag_reduces_carry_distance():
    no_drag = simulate_from_launch_conditions(
        60.0, 14.0, 0.0, params=FlightParams(drag_enabled=False, lift_enabled=False)
    )
    with_drag = simulate_from_launch_conditions(
        60.0, 14.0, 0.0, params=FlightParams(drag_enabled=True, lift_enabled=False)
    )
    assert with_drag.carry_m < no_drag.carry_m


def test_positive_horizontal_launch_lands_on_positive_y_side():
    result = simulate_from_launch_conditions(40.0, 12.0, 5.0, params=FlightParams())
    assert result.side_deviation_m > 0


def test_magnus_lift_increases_carry_for_backspin():
    """A ball with backspin (spin around -Y for +X-direction flight)
    should carry farther than the same launch with lift disabled --
    this is the whole reason real golf balls fly farther than a
    drag-only model predicts."""
    speed, angle = 50.0, 12.0
    no_lift = simulate_from_launch_conditions(
        speed, angle, 0.0, params=FlightParams(drag_enabled=True, lift_enabled=False)
    )
    spin = np.array([0.0, -400.0, 0.0])
    with_lift = simulate_from_launch_conditions(
        speed, angle, 0.0, params=FlightParams(drag_enabled=True, lift_enabled=True), spin_rad_s=spin
    )
    assert with_lift.carry_m > no_lift.carry_m
    assert with_lift.apex_m > no_lift.apex_m


def test_to_tracked_points_is_time_ordered_and_clips_height_at_zero():
    result = simulate_from_launch_conditions(35.0, 15.0, 0.0, params=FlightParams())
    points = result.to_tracked_points(max_points=50)
    times = [p.time_seconds for p in points]
    assert times == sorted(times)
    assert all(p.z_m >= 0.0 for p in points)
    assert points[0].time_seconds == pytest.approx(0.0)
