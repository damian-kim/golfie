from golfie_core.schemas import ShotMetrics, ShotResult, TrackedPoint3D
from golfie_render.threejs import build_trajectory_payload


def test_trajectory_payload_maps_golfie_axes_to_scene_axes():
    """Golfie world frame: +X downrange, +Y lateral, +Z up.
    Scene frame (Three.js convention): scene X = downrange, scene Y = up,
    scene Z = lateral. See golfie_render.threejs.payload docstring.
    """
    points = [TrackedPoint3D(time_seconds=1.0, x_m=10.0, y_m=-2.0, z_m=3.0, confidence=0.8)]
    shot = ShotResult(simulated_trajectory_3d=points, metrics=ShotMetrics())

    payload = build_trajectory_payload("s1", shot, club="driver")
    scene_point = payload["simulated_trajectory"][0]

    assert scene_point["x"] == 10.0  # downrange unchanged
    assert scene_point["y"] == 3.0  # world Z (up) -> scene Y (up)
    assert scene_point["z"] == -2.0  # world Y (lateral) -> scene Z (lateral)
    assert scene_point["t"] == 1.0
    assert scene_point["confidence"] == 0.8


def test_trajectory_payload_round_trips_metrics_as_json_safe_dict():
    shot = ShotResult(metrics=ShotMetrics())
    payload = build_trajectory_payload("s1", shot)
    # MetricSource enum should serialize to its plain string value, not
    # a Python enum repr, since this goes straight into an HTTP response.
    assert payload["metrics"]["ball_speed_mps"]["source"] == "not_available"


def test_trajectory_payload_preserves_warnings_and_placeholder_flag():
    shot = ShotResult(metrics=ShotMetrics(), warnings=["test warning"], is_placeholder=True)
    payload = build_trajectory_payload("s1", shot)
    assert payload["warnings"] == ["test warning"]
    assert payload["is_placeholder"] is True
