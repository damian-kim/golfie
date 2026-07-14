"""golfie_physics

Ball flight physics: state/parameter models, numerical integrators, and
(eventually) fitting of physics parameters to measured early trajectory.

RK4, gravity/drag flight, and constrained initial-state fitting are active.
Magnus lift remains experimental/off by default. Ground bounce/roll uses an
explicitly experimental terminal-angle estimate because spin and turf firmness
are not yet measured.
"""

__all__ = ["models", "integrators", "fitting", "validation"]
