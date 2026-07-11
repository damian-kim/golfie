"""golfie_physics

Ball flight physics: state/parameter models, numerical integrators, and
(eventually) fitting of physics parameters to measured early trajectory.

RK4, gravity/drag flight, and constrained initial-state fitting are active.
Magnus lift remains experimental/off by default, and ground bounce/roll is not
implemented; first ground contact is reported as carry.
"""

__all__ = ["models", "integrators", "fitting", "validation"]
