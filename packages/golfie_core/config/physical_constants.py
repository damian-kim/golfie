"""Physical constants shared by golfie_physics and golfie_cv.

Values are defaults, not commandments -- golfie_physics.fitting is
expected to adjust drag/lift coefficients per-shot within realistic
bounds (spec section 13). Keeping the *defaults* in one place avoids the
classic bug where three modules each hardcode a slightly different ball
mass.
"""

from __future__ import annotations

GRAVITY_MPS2 = 9.81

# USGA-regulation golf ball, approximate.
GOLF_BALL_MASS_KG = 0.0459
GOLF_BALL_RADIUS_M = 0.02135
GOLF_BALL_DIAMETER_M = GOLF_BALL_RADIUS_M * 2
GOLF_BALL_CROSS_SECTION_M2 = 3.14159265358979 * GOLF_BALL_RADIUS_M**2

# Sea level, ~20C dry air. Override per-session if altitude/temperature
# are known.
AIR_DENSITY_KG_M3 = 1.225

# Reasonable starting points for a dimpled golf ball at typical golf
# speeds; golfie_physics.fitting refines these per shot.
DEFAULT_DRAG_COEFFICIENT = 0.25
DEFAULT_LIFT_COEFFICIENT = 0.15

# Unit conversions (UI displays yards/mph/degrees/rpm; engine stays in
# meters/seconds/radians internally per spec section 6).
METERS_PER_YARD = 0.9144
MPS_PER_MPH = 0.44704
