import subprocess
from pathlib import Path
from golfie_cv.sync import _find_ffmpeg_fallback

def ensure_constant_frame_rate(video_path: Path, output_path: Path, target_fps: float = 240.0) -> Path:
    """Transcode a variable frame rate (VFR) video into a constant frame rate (CFR) video.
    
    Removes audio (-an) to optimize transcoding speed, as audio envelope extraction
    is handled independently.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    ffmpeg_bin = _find_ffmpeg_fallback() or "ffmpeg"
    
    cmd = [
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
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path
