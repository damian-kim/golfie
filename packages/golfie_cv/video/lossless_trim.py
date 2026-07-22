"""Lossless, keyframe-safe trimming for single-camera golf swing clips."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from golfie_cv.sync import _find_ffmpeg_fallback
from golfie_cv.video import read_video_metadata


class LosslessTrimError(RuntimeError):
    """Raised when a source clip cannot be analyzed or stream-copied."""


@dataclass(frozen=True)
class LosslessTrimResult:
    source_duration_seconds: float
    requested_start_seconds: float
    requested_end_seconds: float
    actual_start_seconds: float
    actual_end_seconds: float
    output_duration_seconds: float
    fps: float
    width: int
    height: int
    sample_aspect_ratio: str
    display_aspect_ratio: str
    video_codec: str
    audio_codec: str | None
    keyframe_aligned: bool
    lossless: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _VideoPacket:
    pts_seconds: float
    is_keyframe: bool
    data_hash: str


def _ffmpeg_path() -> str:
    executable = _find_ffmpeg_fallback()
    if executable:
        return executable
    raise LosslessTrimError(
        "FFmpeg is required for lossless trimming but was not found. "
        "Install FFmpeg or place it in Golfie's bin directory."
    )


def _ffprobe_path(ffmpeg_path: str) -> str:
    executable = shutil.which("ffprobe")
    if executable:
        return executable
    sibling_name = "ffprobe.exe" if Path(ffmpeg_path).suffix.lower() == ".exe" else "ffprobe"
    sibling = Path(ffmpeg_path).with_name(sibling_name)
    if sibling.exists():
        return str(sibling)
    raise LosslessTrimError(
        "FFprobe is required to preserve and verify source video timing, but was not found."
    )


def _run_json(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=True)
        return json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        raise LosslessTrimError(f"Could not inspect the video with FFprobe: {detail}") from exc


def _fraction(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator) if float(denominator) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _probe_media(path: Path, ffprobe_path: str) -> dict[str, Any]:
    payload = _run_json(
        [
            ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            (
                "format=start_time,duration:"
                "stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,"
                "start_time,duration,sample_aspect_ratio,display_aspect_ratio"
            ),
            "-of",
            "json",
            str(path),
        ]
    )
    streams = payload.get("streams") or []
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if video is None:
        raise LosslessTrimError("The uploaded file does not contain a video stream.")
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    duration = float((payload.get("format") or {}).get("duration") or 0.0)
    return {
        "duration": duration,
        "format_start": float((payload.get("format") or {}).get("start_time") or 0.0),
        "video_start": float(video.get("start_time") or 0.0),
        # r_frame_rate is the encoded stream's nominal cadence. avg_frame_rate
        # can legitimately change when a VFR source is shortened.
        "fps": _fraction(video.get("r_frame_rate")) or _fraction(video.get("avg_frame_rate")),
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "sample_aspect_ratio": video.get("sample_aspect_ratio") or "1:1",
        "display_aspect_ratio": video.get("display_aspect_ratio") or "",
        "video_codec": str(video.get("codec_name") or "unknown"),
        "audio_codec": str(audio.get("codec_name")) if audio else None,
    }


def _probe_video_packets(path: Path, ffprobe_path: str) -> list[_VideoPacket]:
    payload = _run_json(
        [
            ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_packets",
            "-show_entries",
            "packet=pts_time,flags,data_hash",
            "-show_data_hash",
            "sha256",
            "-of",
            "json",
            str(path),
        ]
    )
    packets: list[_VideoPacket] = []
    for packet in payload.get("packets") or []:
        try:
            timestamp = float(packet.get("pts_time"))
        except (TypeError, ValueError):
            continue
        data_hash = str(packet.get("data_hash") or "")
        if not math.isfinite(timestamp) or not data_hash:
            continue
        packets.append(
            _VideoPacket(
                pts_seconds=timestamp,
                is_keyframe="K" in str(packet.get("flags") or ""),
                data_hash=data_hash,
            )
        )
    if not packets:
        raise LosslessTrimError("FFprobe could not read encoded video packets.")
    return packets


def _keyframes(packets: list[_VideoPacket]) -> list[float]:
    return sorted({packet.pts_seconds for packet in packets if packet.is_keyframe})


def _safe_keyframe_start(keyframes: list[float], requested_start: float) -> float:
    candidates = [timestamp for timestamp in keyframes if timestamp <= requested_start + 1e-6]
    return max(0.0, max(candidates, default=0.0))


def _run_ffmpeg(command: list[str], output_path: Path) -> None:
    output_path.unlink(missing_ok=True)
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise LosslessTrimError(f"FFmpeg could not be started: {exc}") from exc
    if completed.returncode != 0:
        output_path.unlink(missing_ok=True)
        detail = completed.stderr.strip() or "unknown FFmpeg error"
        raise LosslessTrimError(f"Lossless FFmpeg trim failed: {detail}")
    if not output_path.exists() or output_path.stat().st_size <= 0:
        output_path.unlink(missing_ok=True)
        raise LosslessTrimError("FFmpeg completed without producing a playable clip.")


def _source_packet_span(
    source_packets: list[_VideoPacket], output_packets: list[_VideoPacket]
) -> tuple[float, float]:
    """Map copied packet hashes back to source PTS for exact boundary verification."""
    source_by_hash = {packet.data_hash: packet for packet in source_packets}
    matched = [
        source_by_hash[packet.data_hash]
        for packet in output_packets
        if packet.data_hash in source_by_hash
    ]
    if len(matched) < max(1, round(len(output_packets) * 0.9)):
        raise LosslessTrimError(
            "Lossless verification failed: copied packets could not be matched to the source."
        )
    first_keyframe = next((packet for packet in matched if packet.is_keyframe), None)
    if first_keyframe is None:
        raise LosslessTrimError("Lossless output does not begin with a decodable keyframe.")
    return first_keyframe.pts_seconds, max(packet.pts_seconds for packet in matched)


def _stream_copy_command(
    ffmpeg: str,
    source: Path,
    output: Path,
    *,
    seek_start: float,
    end_cutoff: float,
    include_seek: bool = True,
) -> list[str]:
    command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
    ]
    if include_seek:
        command.extend(["-ss", f"{seek_start:.9f}"])
    command.extend(
        [
            "-to",
            f"{end_cutoff:.9f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-map_metadata",
            "0",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
        ]
    )
    if output.suffix.lower() in {".mp4", ".mov", ".m4v"}:
        command.extend(["-movflags", "+faststart"])
    command.append(str(output))
    return command


def trim_video_lossless(
    input_path: str | Path,
    output_path: str | Path,
    *,
    start_seconds: float,
    end_seconds: float,
) -> LosslessTrimResult:
    """Stream-copy a manually selected, independently playable interval.

    Inter-frame video can only start cleanly at an existing keyframe without
    re-encoding. The start is therefore moved backward to the nearest source
    keyframe, never forward past requested swing footage. No video/audio codec,
    frame-rate, resolution, sample aspect ratio, or pixels are changed.
    """
    source = Path(input_path)
    destination = Path(output_path)
    if not source.exists() or source.stat().st_size <= 0:
        raise LosslessTrimError("The uploaded video is empty or missing.")
    if not math.isfinite(start_seconds) or not math.isfinite(end_seconds):
        raise LosslessTrimError("Trim marks must be finite timestamps.")
    if start_seconds < 0:
        raise LosslessTrimError("The start mark cannot be negative.")
    if end_seconds <= start_seconds:
        raise LosslessTrimError("The end mark must be after the start mark.")

    ffmpeg = _ffmpeg_path()
    ffprobe = _ffprobe_path(ffmpeg)
    source_probe = _probe_media(source, ffprobe)
    metadata = read_video_metadata(source)
    source_duration = source_probe["duration"] or metadata.duration_seconds
    if source_duration <= 0:
        raise LosslessTrimError("The uploaded video reports an invalid duration.")

    requested_start = float(start_seconds)
    requested_end = min(source_duration, float(end_seconds))
    if requested_start >= source_duration:
        raise LosslessTrimError("The start mark is beyond the end of the video.")
    if requested_end - requested_start < 0.05:
        raise LosslessTrimError("Select at least 0.05 seconds of video.")

    source_packets = _probe_video_packets(source, ffprobe)
    keyframes = _keyframes(source_packets)
    actual_start = _safe_keyframe_start(keyframes, requested_start)
    target_keyframe_index = max(
        0, max((index for index, timestamp in enumerate(keyframes) if timestamp <= actual_start + 1e-6), default=0)
    )
    seek_keyframe_index = target_keyframe_index
    end_cutoff = requested_end
    source_fps = source_probe["fps"] or metadata.fps
    frame_duration = 1.0 / source_fps if source_fps > 0 else 1.0 / 30.0
    boundary_tolerance = max(0.002, frame_duration * 0.51)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.stem}.partial{destination.suffix}")
    candidate = destination.with_name(f"{destination.stem}.candidate{destination.suffix}")
    partial.unlink(missing_ok=True)
    candidate.unlink(missing_ok=True)

    copied_start = actual_start
    copied_end = requested_end
    try:
        for _ in range(12):
            seek_start = keyframes[seek_keyframe_index] if keyframes else 0.0
            _run_ffmpeg(
                _stream_copy_command(
                    ffmpeg,
                    source,
                    candidate,
                    seek_start=seek_start,
                    end_cutoff=end_cutoff,
                    include_seek=actual_start > 1e-6,
                ),
                candidate,
            )
            candidate_packets = _probe_video_packets(candidate, ffprobe)
            copied_start, copied_end = _source_packet_span(source_packets, candidate_packets)

            # Some HEVC/MOV demuxers compare the output seek against decode
            # timestamps and begin at the following GOP. Retry from one source
            # keyframe earlier until the desired safe keyframe is present.
            if copied_start > actual_start + boundary_tolerance:
                if seek_keyframe_index == 0:
                    raise LosslessTrimError(
                        "Lossless trim would omit frames at the selected start."
                    )
                seek_keyframe_index -= 1
                continue
            if copied_start < actual_start - boundary_tolerance:
                raise LosslessTrimError(
                    "Lossless trim included an unexpected GOP before the selected start."
                )

            # B-frame packet order can carry presentation frames past -to.
            # Tighten the mux cutoff and verify against source packet hashes.
            if copied_end > requested_end + boundary_tolerance:
                end_cutoff -= copied_end - requested_end
                if end_cutoff <= seek_start:
                    raise LosslessTrimError("The lossless trim window collapsed during verification.")
                continue
            break
        else:
            raise LosslessTrimError("Could not converge on the selected lossless trim boundaries.")

        candidate_probe = _probe_media(candidate, ffprobe)
        video_start = max(0.0, candidate_probe["video_start"])
        # A small positive start is normal codec composition delay (especially
        # with B-frames). Only remove material pre-roll, not that decoder delay;
        # seeking a very short clip by a few milliseconds can drop its sole GOP.
        if video_start > max(0.05, frame_duration * 3.0):
            normalize_command = [
                ffmpeg,
                "-y",
                "-v",
                "error",
                "-ss",
                f"{video_start:.9f}",
                "-i",
                str(candidate),
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-map_metadata",
                "0",
                "-c",
                "copy",
                "-avoid_negative_ts",
                "make_zero",
            ]
            if partial.suffix.lower() in {".mp4", ".mov", ".m4v"}:
                normalize_command.extend(["-movflags", "+faststart"])
            normalize_command.append(str(partial))
            _run_ffmpeg(normalize_command, partial)
        else:
            candidate.replace(partial)

        final_packets = _probe_video_packets(partial, ffprobe)
        final_start, final_end = _source_packet_span(source_packets, final_packets)
        if abs(final_start - copied_start) > boundary_tolerance:
            raise LosslessTrimError("Timestamp normalization changed the selected start frame.")
        if final_end > requested_end + boundary_tolerance:
            raise LosslessTrimError("Lossless output still contains frames after the selected end.")
        copied_start, copied_end = final_start, final_end
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    finally:
        candidate.unlink(missing_ok=True)

    output_probe = _probe_media(partial, ffprobe)
    for property_name in ("video_codec", "width", "height", "sample_aspect_ratio"):
        if output_probe[property_name] != source_probe[property_name]:
            partial.unlink(missing_ok=True)
            raise LosslessTrimError(
                f"Lossless verification failed: {property_name} changed during trimming."
            )
    output_fps = output_probe["fps"]
    if source_fps > 0 and (output_fps <= 0 or abs(output_fps - source_fps) > max(0.01, source_fps * 0.001)):
        partial.unlink(missing_ok=True)
        raise LosslessTrimError("Lossless verification failed: frame rate changed during trimming.")

    partial.replace(destination)
    output_duration = float(output_probe["duration"] or (copied_end - copied_start + frame_duration))
    return LosslessTrimResult(
        source_duration_seconds=source_duration,
        requested_start_seconds=requested_start,
        requested_end_seconds=requested_end,
        actual_start_seconds=copied_start,
        actual_end_seconds=min(source_duration, copied_end + frame_duration),
        output_duration_seconds=output_duration,
        fps=source_fps,
        width=source_probe["width"] or metadata.width,
        height=source_probe["height"] or metadata.height,
        sample_aspect_ratio=source_probe["sample_aspect_ratio"],
        display_aspect_ratio=source_probe["display_aspect_ratio"],
        video_codec=source_probe["video_codec"],
        audio_codec=source_probe["audio_codec"],
        keyframe_aligned=abs(actual_start - requested_start) <= 1e-3,
    )
