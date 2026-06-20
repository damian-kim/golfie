"""Generic, model-agnostic RK4 integrator.

Kept independent of any particular state representation so it can be
reused for the ball-flight ODE now and for anything else (club path
smoothing, etc.) later. Spec section 13 explicitly prefers RK4 over
simpler integrators for accuracy/debuggability.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# y: state vector (numpy array of any length)
# t: time in seconds
DerivativeFn = Callable[[float, np.ndarray], np.ndarray]


def rk4_step(f: DerivativeFn, t: float, y: np.ndarray, dt: float) -> np.ndarray:
    """Advance state `y` at time `t` by one RK4 step of size `dt`."""
    k1 = f(t, y)
    k2 = f(t + dt / 2.0, y + dt / 2.0 * k1)
    k3 = f(t + dt / 2.0, y + dt / 2.0 * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate(
    f: DerivativeFn,
    y0: np.ndarray,
    t0: float,
    dt: float,
    stop_condition: Callable[[float, np.ndarray], bool],
    max_steps: int = 100_000,
) -> list[tuple[float, np.ndarray]]:
    """Integrate forward in time until `stop_condition(t, y)` is True.

    Returns a list of (time, state) samples including the initial state
    and the final (stopping) state. `stop_condition` is checked *after*
    each step, so the last sample may have just crossed the threshold
    (e.g. slightly below ground level) -- callers needing an exact
    crossing point should interpolate between the last two samples.
    """
    samples: list[tuple[float, np.ndarray]] = [(t0, y0.copy())]
    t, y = t0, y0.copy()
    for _ in range(max_steps):
        if stop_condition(t, y):
            break
        y = rk4_step(f, t, y, dt)
        t += dt
        samples.append((t, y.copy()))
    return samples
