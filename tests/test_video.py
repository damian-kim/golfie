import shutil
import subprocess

import pytest

from golfie_cv.video import (
    VideoReadError,
    VideoTiming,
    read_video_metadata,
    resolve_stereo_capture_rates,
)

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None


@pytest.fixture
def sample_video(tmp_path):
    if not FFMPEG_AVAILABLE:
        pytest.skip("ffmpeg not available in this environment")
    out_path = tmp_path / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=0.5:size=640x360:rate=240",
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ],
        check=True,
    )
    return out_path


def test_read_video_metadata_reports_real_values(sample_video):
    meta = read_video_metadata(sample_video)
    assert meta.fps == pytest.approx(240.0, rel=0.02)
    assert meta.width == 640
    assert meta.height == 360
    assert meta.frame_count > 0
    assert meta.duration_seconds == pytest.approx(0.5, rel=0.1)


def test_read_video_metadata_raises_on_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.mp4"
    with pytest.raises(VideoReadError):
        read_video_metadata(missing)


def test_read_video_metadata_raises_on_non_video_file(tmp_path):
    bogus = tmp_path / "not_a_video.mp4"
    bogus.write_text("this is definitely not a video file")
    with pytest.raises(VideoReadError):
        read_video_metadata(bogus)


def test_stereo_timing_uses_native_phone_anchor_for_metadata_stripped_pair():
    samsung_export = VideoTiming(
        nominal_fps=30000 / 1001,
        average_fps=29.97,
        playback_duration_seconds=7.0,
        frame_timestamps=(),
        inference_reason="metadata stripped",
    )
    iphone_native = VideoTiming(
        nominal_fps=60.0,
        average_fps=44.7,
        playback_duration_seconds=10.7,
        frame_timestamps=(),
        device_make="Apple",
        device_model="iPhone 16 Pro Max",
        variable_frame_rate=True,
        slow_motion_hint=True,
        inferred_capture_fps=240.0,
        inference_confidence=0.9,
        inference_reason="native Apple Slo-mo metadata",
    )

    rate_a, rate_b, reason = resolve_stereo_capture_rates(
        samsung_export, iphone_native, 30000 / 1001, 60.0
    )

    assert rate_a == pytest.approx(240.0)
    assert rate_b == pytest.approx(240.0)
    assert "paired" in reason


def test_stereo_timing_does_not_invent_high_speed_without_an_anchor():
    timing_a = VideoTiming(30.0, 30.0, 5.0, ())
    timing_b = VideoTiming(60.0, 60.0, 5.0, ())

    rate_a, rate_b, reason = resolve_stereo_capture_rates(timing_a, timing_b, 30.0, 60.0)

    assert rate_a == 30.0
    assert rate_b == 60.0
    assert "no recoverable slow-motion clock" in reason
