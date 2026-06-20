import shutil
import subprocess

import pytest

from golfie_cv.video import VideoReadError, read_video_metadata

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
