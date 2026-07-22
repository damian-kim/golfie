import json
import shutil
import subprocess

import pytest

from golfie_cv.video.lossless_trim import trim_video_lossless


FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="FFmpeg and FFprobe are required")
def test_lossless_trim_preserves_stream_and_video_geometry(tmp_path):
    source = tmp_path / "source.mp4"
    output = tmp_path / "trimmed.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=duration=8:size=640x360:rate=30",
            "-c:v",
            "libx264",
            "-g",
            "60",
            "-keyint_min",
            "60",
            "-sc_threshold",
            "0",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )

    result = trim_video_lossless(
        source,
        output,
        start_seconds=2.4,
        end_seconds=5.0,
    )

    assert output.exists()
    assert result.lossless is True
    assert result.requested_start_seconds == pytest.approx(2.4)
    assert result.actual_start_seconds == pytest.approx(2.0, abs=0.04)
    assert result.actual_end_seconds >= 5.0 - 0.08
    assert result.actual_end_seconds <= 5.0 + 1.0 / 30.0 + 0.01
    assert result.fps == pytest.approx(30.0, rel=0.001)
    assert (result.width, result.height) == (640, 360)
    assert result.sample_aspect_ratio == "1:1"
    assert result.video_codec == "h264"

    def probe(path):
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,r_frame_rate,width,height,sample_aspect_ratio",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)["streams"][0]

    assert probe(output) == probe(source)

    def decoded_hashes(path):
        completed = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "framemd5", "-"],
            check=True,
            capture_output=True,
            text=True,
        )
        return [
            line.rsplit(",", 1)[-1].strip()
            for line in completed.stdout.splitlines()
            if line and not line.startswith("#")
        ]

    source_hashes = decoded_hashes(source)
    output_hashes = decoded_hashes(output)
    assert source_hashes.index(output_hashes[0]) == 60  # 2.0-second source keyframe
    assert source_hashes.index(output_hashes[-1]) <= 150  # no frames beyond the selected end


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="FFmpeg and FFprobe are required")
def test_manual_lossless_trim_uses_selected_marks(tmp_path):
    source = tmp_path / "source.mp4"
    output = tmp_path / "manual.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
            "-i", "testsrc2=duration=4:size=320x240:rate=60",
            "-c:v", "libx264", "-g", "60", "-pix_fmt", "yuv420p", str(source),
        ],
        check=True,
    )

    result = trim_video_lossless(
        source,
        output,
        start_seconds=1.25,
        end_seconds=2.75,
    )

    assert result.requested_start_seconds == pytest.approx(1.25)
    assert result.requested_end_seconds == pytest.approx(2.75)
    assert result.actual_start_seconds == pytest.approx(1.0, abs=0.03)
    assert result.output_duration_seconds == pytest.approx(1.75, abs=0.08)
    assert result.fps == pytest.approx(60.0, rel=0.001)
