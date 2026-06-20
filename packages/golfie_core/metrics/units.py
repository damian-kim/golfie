"""Unit conversions between Golfie's internal metric units and the
yards/mph/degrees/rpm units the UI displays (spec section 6).

Internal storage is always meters/seconds/radians; convert only at the
edges (API serialization for display, or user-facing exports).
"""

from __future__ import annotations

import math

from golfie_core.config.physical_constants import METERS_PER_YARD, MPS_PER_MPH


def mps_to_mph(value_mps: float) -> float:
    return value_mps / MPS_PER_MPH


def mph_to_mps(value_mph: float) -> float:
    return value_mph * MPS_PER_MPH


def meters_to_yards(value_m: float) -> float:
    return value_m / METERS_PER_YARD


def yards_to_meters(value_yd: float) -> float:
    return value_yd * METERS_PER_YARD


def rad_to_deg(value_rad: float) -> float:
    return math.degrees(value_rad)


def deg_to_rad(value_deg: float) -> float:
    return math.radians(value_deg)
