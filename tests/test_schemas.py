import pytest
from pydantic import ValidationError

from golfie_core.schemas import (
    CameraCapture,
    MetricSource,
    MetricValue,
    Session,
    ShotMetrics,
    build_placeholder_shot_result,
)


def test_metric_value_rejects_value_without_real_source():
    """The core honesty rule: you can't attach a number to
    source=not_available -- that would look measured without being so.
    """
    with pytest.raises(ValidationError):
        MetricValue(value=123.4, source=MetricSource.NOT_AVAILABLE)


def test_metric_value_unavailable_helper_has_no_value():
    mv = MetricValue.unavailable(notes="not implemented yet")
    assert mv.value is None
    assert mv.source == MetricSource.NOT_AVAILABLE
    assert mv.confidence == 0.0


def test_metric_value_allows_value_with_real_source():
    mv = MetricValue(value=42.0, source=MetricSource.MEASURED, confidence=0.95)
    assert mv.value == 42.0


def test_shot_metrics_defaults_are_all_unavailable():
    metrics = ShotMetrics()
    for field_name in ShotMetrics.model_fields:
        metric = getattr(metrics, field_name)
        assert metric.source == MetricSource.NOT_AVAILABLE
        assert metric.value is None


def test_session_round_trips_through_json():
    session = Session(club="7-iron")
    restored = Session.model_validate_json(session.model_dump_json())
    assert restored.session_id == session.session_id
    assert restored.club == "7-iron"
    assert restored.stage == session.stage


def _camera(fps: float, camera_id: str = "camera_a") -> CameraCapture:
    return CameraCapture(
        camera_id=camera_id,
        fps=fps,
        resolution=(1920, 1080),
        video_path=f"/tmp/{camera_id}.mp4",
    )


def test_placeholder_shot_result_is_honest_with_no_calibration_or_sync():
    shot = build_placeholder_shot_result(
        camera_a=_camera(240.0, "camera_a"),
        camera_b=_camera(240.0, "camera_b"),
    )
    assert shot.is_placeholder is True
    assert shot.metrics.ball_speed_mps.source == MetricSource.NOT_AVAILABLE
    assert shot.metrics.carry_m.value is None
    assert any("calibration" in w.lower() for w in shot.warnings)
    assert any("sync" in w.lower() for w in shot.warnings)


def test_placeholder_shot_result_warns_on_low_fps():
    shot = build_placeholder_shot_result(
        camera_a=_camera(240.0, "camera_a"),
        camera_b=_camera(60.0, "camera_b"),
    )
    fps_warnings = [w for w in shot.warnings if "camera_b" in w and "fps" in w]
    assert len(fps_warnings) == 1


def test_placeholder_shot_result_does_not_warn_about_fps_when_both_high():
    shot = build_placeholder_shot_result(
        camera_a=_camera(240.0, "camera_a"),
        camera_b=_camera(240.0, "camera_b"),
    )
    assert not any("fps" in w for w in shot.warnings)
