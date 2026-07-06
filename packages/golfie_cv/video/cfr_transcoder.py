import subprocess
from pathlib import Path
import re
import cv2
from golfie_cv.sync import _find_ffmpeg_fallback

def ensure_constant_frame_rate(
    video_path: Path, 
    output_path: Path, 
    target_fps: float = 240.0,
    progress_callback = None
) -> Path:
    """Transcode a variable frame rate (VFR) video into a constant frame rate (CFR) video.
    
    Removes audio (-an) to optimize transcoding speed, as audio envelope extraction
    is handled independently.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    ffmpeg_bin = _find_ffmpeg_fallback() or "ffmpeg"
    
    # Estimate total duration of input video to calculate progress percentage
    duration_s = 0.0
    try:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration_s = frame_count / fps if fps > 0 else 0.0
            cap.release()
    except Exception:
        pass

    # Regex to extract the time stamp from FFmpeg output (e.g. time=00:00:05.12)
    time_regex = re.compile(r"time=(\d+):(\d+):(\d+\.\d+)")

    def run_ffmpeg(args_list):
        process = subprocess.Popen(
            args_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Read stderr character-by-character to handle \r updates
        buffer = []
        last_pct = -1.0
        while True:
            char = process.stderr.read(1)
            if not char:
                break
            if char in ("\r", "\n"):
                line = "".join(buffer).strip()
                buffer = []
                if line:
                    match = time_regex.search(line)
                    if match and duration_s > 0 and progress_callback:
                        hours, minutes, seconds = match.groups()
                        current_s = float(hours) * 3600 + float(minutes) * 60 + float(seconds)
                        pct = min(100.0, (current_s / duration_s) * 100.0)
                        # Avoid flooding the log file with too many identical percentage prints
                        if round(pct) != last_pct:
                            last_pct = round(pct)
                            progress_callback(f"Transcoding: {pct:.0f}% ({current_s:.1f}s / {duration_s:.1f}s)")
            else:
                buffer.append(char)
                
        process.wait()
        return process.returncode, process.stdout.read() if process.stdout else "", "".join(buffer)

    # Try using modern -fps_mode first
    cmd = [
        ffmpeg_bin, "-y",
        "-i", str(video_path),
        "-r", str(target_fps),
        "-fps_mode", "cfr",
        "-vcodec", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        "-an",
        str(output_path)
    ]
    
    returncode, stdout, stderr = run_ffmpeg(cmd)
    if returncode != 0:
        # Check if the error is because the modern option isn't supported
        if "Unrecognized option 'fps_mode'" in stderr or "Option not found" in stderr:
            # Fall back to deprecated -vsync for older FFmpeg installations
            cmd_fallback = [
                ffmpeg_bin, "-y",
                "-i", str(video_path),
                "-r", str(target_fps),
                "-vsync", "cfr",
                "-vcodec", "libx264",
                "-preset", "ultrafast",
                "-crf", "23",
                "-an",
                str(output_path)
            ]
            returncode, stdout, stderr = run_ffmpeg(cmd_fallback)
            if returncode != 0:
                raise subprocess.CalledProcessError(returncode, cmd_fallback, output=stdout, stderr=stderr)
        else:
            # Raise the original error
            raise subprocess.CalledProcessError(returncode, cmd, output=stdout, stderr=stderr)
            
    return output_path
