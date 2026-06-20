"""Converts Golfie's internal Session/ShotResult schema into the JSON
shape the browser-based Three.js driving range scene consumes.

Coordinate mapping (documented here because it's the one place the
Golfie world frame and the Three.js scene frame meet):

    Golfie world frame (golfie_core.coordinates)   Three.js scene frame
    --------------------------------------------   ---------------------
    +X  target line / downrange                ->  scene X (downrange)
    +Z  vertical up                             ->  scene Y (Three.js "up")
    +Y  lateral / side                          ->  scene Z (side)

This keeps "up" on the Three.js Y axis (its convention) while keeping
downrange distance on a single, easy-to-reason-about scene axis.
"""

from __future__ import annotations

from golfie_core.schemas import ShotResult, TrackedPoint3D


def _to_scene_point(p: TrackedPoint3D) -> dict:
    return {
        "t": p.time_seconds,
        "x": p.x_m,
        "y": p.z_m,
        "z": p.y_m,
        "confidence": p.confidence,
    }


def _points_to_scene(points: list[TrackedPoint3D]) -> list[dict]:
    return [_to_scene_point(p) for p in points]


def build_trajectory_payload(session_id: str, shot: ShotResult, club: str | None = None) -> dict:
    """Build the full JSON payload the frontend's <DrivingRangeScene>
    component expects (see apps/web/frontend/src/lib/types.ts).
    """
    metrics = shot.metrics.model_dump(mode="json")
    return {
        "session_id": session_id,
        "club": club,
        "is_placeholder": shot.is_placeholder,
        "warnings": shot.warnings,
        "notes": shot.notes,
        "metrics": metrics,
        "measured_points": _points_to_scene(shot.measured_points_3d),
        "fitted_points": _points_to_scene(shot.fitted_points_3d),
        "simulated_trajectory": _points_to_scene(shot.simulated_trajectory_3d),
    }
