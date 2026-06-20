"""Physics model parameters, with defaults from golfie_core.config."""

from __future__ import annotations

from dataclasses import dataclass

from golfie_core.config.physical_constants import (
    AIR_DENSITY_KG_M3,
    DEFAULT_DRAG_COEFFICIENT,
    DEFAULT_LIFT_COEFFICIENT,
    GOLF_BALL_CROSS_SECTION_M2,
    GOLF_BALL_MASS_KG,
    GRAVITY_MPS2,
)


@dataclass
class FlightParams:
    """Tunable parameters for the ball flight model (spec section 13).

    `fitting` adjusts drag_coefficient / lift_coefficient per shot within
    realistic bounds; everything else is a near-constant.
    """

    mass_kg: float = GOLF_BALL_MASS_KG
    cross_section_m2: float = GOLF_BALL_CROSS_SECTION_M2
    air_density_kg_m3: float = AIR_DENSITY_KG_M3
    gravity_mps2: float = GRAVITY_MPS2
    drag_coefficient: float = DEFAULT_DRAG_COEFFICIENT
    lift_coefficient: float = DEFAULT_LIFT_COEFFICIENT

    drag_enabled: bool = True
    # Magnus lift is implemented but should be treated as experimental
    # (spec section 13, Phase 3) -- off by default until spin estimation
    # is validated.
    lift_enabled: bool = False
