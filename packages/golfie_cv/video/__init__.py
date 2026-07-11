"""Video ingestion: read real metadata from an uploaded video file.

This is the one piece of golfie_cv that is fully implemented in v0 --
everything downstream (sync, detection, tracking) needs accurate fps and
frame counts, and there is no honest way to fake this, so it's a good
place to start with real code instead of a stub.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2


class VideoReadError(RuntimeError):
    """Raised when a video file can't be opened or has no readable frames."""


@dataclass
class VideoMetadata:
    path: str
    fps: float
    width: int
    height: int
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self.frame_count / self.fps


@dataclass
class VideoTiming:
    """Presentation timing and recoverable high-speed capture semantics."""

    nominal_fps: float
    average_fps: float
    playback_duration_seconds: float
    frame_timestamps: tuple[float, ...]
    device_make: Optional[str] = None
    device_model: Optional[str] = None
    variable_frame_rate: bool = False
    slow_motion_hint: bool = False
    inferred_capture_fps: Optional[float] = None
    inference_confidence: float = 0.0
    inference_reason: str = ""
    slow_regions: tuple[tuple[int, int], ...] = ()

    def frame_index_at_time(self, seconds: float) -> int:
        if not self.frame_timestamps:
            return max(0, int(round(seconds * self.nominal_fps)))
        import bisect

        index = bisect.bisect_left(self.frame_timestamps, seconds)
        if index <= 0:
            return 0
        if index >= len(self.frame_timestamps):
            return len(self.frame_timestamps) - 1
        before = self.frame_timestamps[index - 1]
        after = self.frame_timestamps[index]
        return index - 1 if seconds - before <= after - seconds else index

    def contains_slow_window(self, start_frame: int, end_frame: int) -> bool:
        if not self.slow_regions:
            return self.slow_motion_hint
        return any(start_frame >= start and end_frame <= end for start, end in self.slow_regions)


def _ffprobe_path() -> str | None:
    import shutil

    executable = shutil.which("ffprobe")
    if executable:
        return executable
    try:
        from golfie_cv.sync import _find_ffmpeg_fallback

        ffmpeg = _find_ffmpeg_fallback()
        if ffmpeg:
            sibling = Path(ffmpeg).with_name("ffprobe.exe")
            if sibling.exists():
                return str(sibling)
    except Exception:
        pass
    return None


def _fraction(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator) if float(denominator) else 0.0
    except (TypeError, ValueError):
        return 0.0


def analyze_video_timing(video_path: str | Path) -> VideoTiming:
    """Inspect sample PTS and device tags; never confuse playback FPS with capture FPS."""
    import json
    import subprocess
    import numpy as np

    path = Path(video_path)
    probe = _ffprobe_path()
    if probe is None:
        metadata = read_video_metadata(path)
        return VideoTiming(
            nominal_fps=metadata.fps,
            average_fps=metadata.fps,
            playback_duration_seconds=metadata.duration_seconds,
            frame_timestamps=(),
            inference_reason="ffprobe unavailable; only container nominal rate is known",
        )

    command = [
        probe, "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=r_frame_rate,avg_frame_rate,duration:frame=best_effort_timestamp_time:format_tags",
        "-of", "json", str(path),
    ]
    try:
        payload = json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        metadata = read_video_metadata(path)
        return VideoTiming(
            nominal_fps=metadata.fps,
            average_fps=metadata.fps,
            playback_duration_seconds=metadata.duration_seconds,
            frame_timestamps=(),
            inference_reason=f"ffprobe timing analysis failed: {exc}",
        )

    stream = (payload.get("streams") or [{}])[0]
    tags = (payload.get("format") or {}).get("tags") or {}
    timestamps = tuple(
        float(frame["best_effort_timestamp_time"])
        for frame in payload.get("frames", [])
        if frame.get("best_effort_timestamp_time") is not None
    )
    nominal = _fraction(stream.get("r_frame_rate"))
    average = _fraction(stream.get("avg_frame_rate"))
    duration = float(stream.get("duration") or (timestamps[-1] if timestamps else 0.0))
    deltas = np.diff(np.asarray(timestamps, dtype=np.float64))
    positive = deltas[deltas > 1e-6]
    variable = bool(
        len(positive) >= 3
        and np.percentile(positive, 90) / max(np.percentile(positive, 10), 1e-9) > 1.35
    )
    make = tags.get("com.apple.quicktime.make") or tags.get("make")
    model = tags.get("com.apple.quicktime.model") or tags.get("model")
    apple_slowmo = (
        str(make).lower() == "apple"
        and tags.get("com.apple.quicktime.full-frame-rate-playback-intent") == "0"
    )

    capture_fps = None
    confidence = 0.0
    reason = "container exposes playback timing only"
    if apple_slowmo and model and "iphone" in model.lower():
        # Native rear-camera iPhone Slo-mo at FHD uses a 240 Hz capture clock.
        # The PTS table describes editable playback ramps, not sensor cadence.
        capture_fps = 240.0
        confidence = 0.9
        reason = f"Apple native Slo-mo metadata from {model}; PTS contains playback retiming"

    slow_regions: list[tuple[int, int]] = []
    if apple_slowmo and len(positive):
        # The slowed section has the longest stable sample durations. Include
        # transition-ramp samples close to that plateau and merge short gaps.
        slow_delta = float(np.percentile(positive, 90))
        mask = deltas >= slow_delta * 0.82
        start = None
        for index, is_slow in enumerate(mask):
            if is_slow and start is None:
                start = index
            if start is not None and (not is_slow or index == len(mask) - 1):
                end = index + 1 if is_slow else index
                if end - start >= 4:
                    slow_regions.append((start, end))
                start = None

    return VideoTiming(
        nominal_fps=nominal,
        average_fps=average,
        playback_duration_seconds=duration,
        frame_timestamps=timestamps,
        device_make=make,
        device_model=model,
        variable_frame_rate=variable,
        slow_motion_hint=apple_slowmo,
        inferred_capture_fps=capture_fps,
        inference_confidence=confidence,
        inference_reason=reason,
        slow_regions=tuple(slow_regions),
    )


def resolve_stereo_capture_rates(
    timing_a: VideoTiming,
    timing_b: VideoTiming,
    declared_fps_a: float,
    declared_fps_b: float,
) -> tuple[float, float, str]:
    """Resolve physical sample clocks, using a strong phone anchor for its paired view."""
    rate_a = timing_a.inferred_capture_fps
    rate_b = timing_b.inferred_capture_fps
    reasons = []

    # A user value is an override only when it materially differs from the
    # container playback rate. Auto-detected uploads store the playback rate.
    if rate_a is None and declared_fps_a > timing_a.nominal_fps * 1.5:
        rate_a = declared_fps_a
        reasons.append(f"Camera A explicit capture-rate override {rate_a:.1f} fps")
    if rate_b is None and declared_fps_b > timing_b.nominal_fps * 1.5:
        rate_b = declared_fps_b
        reasons.append(f"Camera B explicit capture-rate override {rate_b:.1f} fps")

    if rate_a is None and rate_b is not None and timing_a.nominal_fps <= 60.1:
        rate_a = rate_b
        reasons.append("Camera A inherited the anchored paired slow-motion capture clock")
    if rate_b is None and rate_a is not None and timing_b.nominal_fps <= 60.1:
        rate_b = rate_a
        reasons.append("Camera B inherited the anchored paired slow-motion capture clock")

    if rate_a is None:
        rate_a = timing_a.nominal_fps
        reasons.append("Camera A has no recoverable slow-motion clock; using playback rate")
    if rate_b is None:
        rate_b = timing_b.nominal_fps
        reasons.append("Camera B has no recoverable slow-motion clock; using playback rate")
    return float(rate_a), float(rate_b), "; ".join(reasons)


def _get_ffprobe_fps(path: Path) -> float | None:
    import shutil
    import subprocess
    
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        try:
            from golfie_cv.sync import _find_ffmpeg_fallback
            ffmpeg_bin = _find_ffmpeg_fallback()
            if ffmpeg_bin:
                ffmpeg_path = Path(ffmpeg_bin)
                ffprobe_sibling = ffmpeg_path.parent / "ffprobe.exe"
                if ffprobe_sibling.exists():
                    ffprobe_bin = str(ffprobe_sibling)
        except Exception:
            pass
            
    if not ffprobe_bin:
        return None
        
    try:
        cmd = [
            ffprobe_bin, "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        out = res.stdout.strip()
        if "/" in out:
            num, den = map(float, out.split("/"))
            if den > 0:
                return num / den
        else:
            return float(out)
    except Exception:
        pass
    return None


def read_video_metadata(video_path: str | Path) -> VideoMetadata:
    """Open a video file and read its real fps/resolution/frame count.

    Raises VideoReadError if the file is missing, unreadable, or not a
    video OpenCV understands -- callers should surface this to the user
    rather than substituting fake metadata (spec section 19, scientific
    honesty).
    """
    path = Path(video_path)
    if not path.exists():
        raise VideoReadError(f"Video file does not exist: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise VideoReadError(f"OpenCV could not open video file: {path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    if width <= 0 or height <= 0:
        raise VideoReadError(
            f"Video opened but reported invalid resolution ({width}x{height}): {path}"
        )

    ffprobe_fps = _get_ffprobe_fps(path)
    if ffprobe_fps is not None and ffprobe_fps > 0:
        fps = ffprobe_fps

    return VideoMetadata(
        path=str(path),
        fps=float(fps),
        width=width,
        height=height,
        frame_count=frame_count,
    )



def extract_frame(video_path: str | Path, frame_index: int):
    """Extract a single frame as a BGR numpy array (OpenCV convention).

    Used by debug tooling (e.g. previewing the impact frame) and, later,
    by the detection pipeline. Raises VideoReadError on failure.
    """
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise VideoReadError(f"OpenCV could not open video file: {path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise VideoReadError(
                f"Could not read frame {frame_index} from {path}"
            )
        return frame
    finally:
        cap.release()

from golfie_cv.video.cfr_transcoder import ensure_constant_frame_rate
