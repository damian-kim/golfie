"""golfie_physics

Ball flight physics: state/parameter models, numerical integrators, and
(eventually) fitting of physics parameters to measured early trajectory.

Status (v0 / Milestone 0):
  - integrators.rk4: REAL. Generic RK4 step, unit tested.
  - models.projectile: REAL for gravity + drag (spec section 13, Phases
    1-2). Magnus lift is implemented but optional/off by default and
    should be treated as experimental until validated (Phase 3). Ground
    interaction (Phase 4: bounce/roll) is NOT implemented -- simulation
    stops at first ground contact, which is reported as the carry
    landing point only.
  - fitting: STUB. Constrained optimization to fit initial velocity/spin
    to measured early 3D points is Milestone 7.
  - validation: STUB. Synthetic ground-truth generators for the test
    suite are Milestone 10.
"""

__all__ = ["models", "integrators", "fitting", "validation"]
