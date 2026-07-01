import cv2
import numpy as np
import scipy.io.wavfile as wavfile
import subprocess
import tempfile
import shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Headless mode
import matplotlib.pyplot as plt

from golfie_core.schemas import SyncResult, SyncMethod

def _find_ffmpeg_fallback() -> str | None:
    import shutil
    import os
    import glob
    
    # 1. Check if already in PATH
    if shutil.which("ffmpeg"):
        return "ffmpeg"
        
    # 2. Check Overwolf extensions (contains Obs ffmpeg version 7.0.2)
    try:
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            pattern = os.path.join(local_app_data, "Overwolf", "**", "ffmpeg.exe")
            matches = glob.glob(pattern, recursive=True)
            if matches:
                ffmpeg_path = matches[0]
                ffmpeg_dir = os.path.dirname(ffmpeg_path)
                # Add OBS bin directory to PATH so ffmpeg can load avcodec.dll, avformat.dll, etc.
                os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
                return ffmpeg_path
    except Exception:
        pass
        
    return None


def _extract_audio(video_path: Path, output_wav_path: Path) -> bool:
    """Extract mono audio from video file to WAV using ffmpeg."""
    ffmpeg_bin = _find_ffmpeg_fallback()
    if not ffmpeg_bin:
        return False
    
    cmd = [
        ffmpeg_bin, "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1",
        str(output_wav_path)
    ]
    res = subprocess.run(cmd, capture_output=True)
    return res.returncode == 0


def align_audio_data(data_a: np.ndarray, rate_a: int, data_b: np.ndarray, rate_b: int) -> tuple[float, float, int, int]:
    """Helper to perform cross-correlation based synchronization on loaded audio signals."""
    # Ensure data is single-channel and floating point
    if len(data_a.shape) > 1:
        data_a = data_a.mean(axis=1)
    if len(data_b.shape) > 1:
        data_b = data_b.mean(axis=1)

    # Normalize values to [-1, 1] range
    if data_a.dtype != np.float32:
        # Convert integer types to float32
        if np.issubdtype(data_a.dtype, np.integer):
            data_a = data_a.astype(np.float32) / np.iinfo(data_a.dtype).max
        else:
            data_a = data_a.astype(np.float32)
    if data_b.dtype != np.float32:
        if np.issubdtype(data_b.dtype, np.integer):
            data_b = data_b.astype(np.float32) / np.iinfo(data_b.dtype).max
        else:
            data_b = data_b.astype(np.float32)

    # Absolute envelope for peak detection
    env_a = np.abs(data_a)
    env_b = np.abs(data_b)

    # Locate the primary transient spike (the impact)
    peak_a = int(np.argmax(env_a))
    peak_b = int(np.argmax(env_b))

    # Extract correlation windows (e.g. 0.02s before and 0.02s after the peak)
    win_sz_a = int(rate_a * 0.02)
    win_sz_b = int(rate_b * 0.02)

    start_a = max(0, peak_a - win_sz_a)
    end_a = min(len(data_a), peak_a + win_sz_a)
    win_a = data_a[start_a:end_a]

    start_b = max(0, peak_b - win_sz_b)
    end_b = min(len(data_b), peak_b + win_sz_b)
    win_b = data_b[start_b:end_b]

    # Ensure identical sample rates (standard for ffmpeg output config)
    if rate_a != rate_b:
        raise ValueError(f"Sample rates must match. Got rate_a={rate_a}Hz and rate_b={rate_b}Hz.")

    # Remove means for normalized cross-correlation
    win_a_norm = win_a - np.mean(win_a)
    win_b_norm = win_b - np.mean(win_b)

    # Compute normalized cross-correlation
    norm_a = np.linalg.norm(win_a_norm)
    norm_b = np.linalg.norm(win_b_norm)
    
    if norm_a > 1e-9 and norm_b > 1e-9:
        corr = np.correlate(win_a_norm, win_b_norm, mode='full') / (norm_a * norm_b)
    else:
        corr = np.correlate(win_a_norm, win_b_norm, mode='full')

    lag = int(np.argmax(corr) - (len(win_b_norm) - 1))

    # Calculate offset in seconds (t_a - t_b)
    offset_sec = (start_a - start_b + lag) / rate_a

    # Determine confidence based on normalized cross-correlation peak value
    if len(corr) > 0:
        peak_val = np.max(corr)
        # Scale peak correlation to a confidence score
        # For a clean correlation, peak value is close to 1.0. 
        # We map peak values [0.3, 0.8] to [0.0, 1.0] confidence.
        confidence = min(1.0, max(0.0, (peak_val - 0.3) / 0.5))
    else:
        confidence = 0.0

    return offset_sec, confidence, peak_a, peak_b


def generate_alignment_plot(
    data_a: np.ndarray, rate_a: int,
    data_b: np.ndarray, rate_b: int,
    peak_a: int, peak_b: int,
    offset_sec: float,
    output_path: Path
) -> None:
    """Save a debug plot comparing audio envelopes before and after alignment."""
    win_sz_a = int(rate_a * 0.5)
    win_sz_b = int(rate_b * 0.5)

    start_a = max(0, peak_a - win_sz_a)
    end_a = min(len(data_a), peak_a + win_sz_a)
    win_a = data_a[start_a:end_a]
    time_a = np.arange(start_a, end_a) / rate_a

    start_b = max(0, peak_b - win_sz_b)
    end_b = min(len(data_b), peak_b + win_sz_b)
    win_b = data_b[start_b:end_b]
    time_b = np.arange(start_b, end_b) / rate_b

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    # Plot unaligned
    ax1.plot(time_a, win_a, label="Camera A", color="royalblue", alpha=0.7)
    ax1.plot(time_b, win_b, label="Camera B", color="darkorange", alpha=0.7)
    ax1.set_title("Before Sync Alignment (Local Timelines)")
    ax1.set_xlabel("Time (seconds relative to video start)")
    ax1.set_ylabel("Amplitude")
    ax1.legend()
    ax1.grid(True)

    # Plot aligned (Camera B shifted by offset_seconds)
    ax2.plot(time_a, win_a, label="Camera A", color="royalblue", alpha=0.7)
    ax2.plot(time_b + offset_sec, win_b, label="Camera B (Aligned)", color="darkorange", alpha=0.7)
    ax2.set_title("After Sync Alignment (Common Timeline)")
    ax2.set_xlabel("Time (seconds relative to Camera A start)")
    ax2.set_ylabel("Amplitude")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()


def estimate_sync_offset(video_path_a: str | Path, video_path_b: str | Path) -> SyncResult:
    """Estimate the time offset between two independently-started recordings.

    Spec section 8.
    """
    video_path_a = Path(video_path_a)
    video_path_b = Path(video_path_b)
    
    # Get FPS of videos using opencv
    cap_a = cv2.VideoCapture(str(video_path_a))
    fps_a = cap_a.get(cv2.CAP_PROP_FPS)
    cap_a.release()
    
    cap_b = cv2.VideoCapture(str(video_path_b))
    fps_b = cap_b.get(cv2.CAP_PROP_FPS)
    cap_b.release()
    
    if fps_a <= 0 or fps_b <= 0:
        raise ValueError("Could not read frame rate (FPS) from video files.")

    # We use a temp directory to store WAV files
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_a = Path(tmpdir) / "audio_a.wav"
        wav_b = Path(tmpdir) / "audio_b.wav"
        
        success_a = _extract_audio(video_path_a, wav_a)
        success_b = _extract_audio(video_path_b, wav_b)
        
        if not (success_a and success_b):
            # If ffmpeg is not available, we return a manual result or fallback
            return SyncResult(
                offset_seconds=0.0,
                offset_frames=0.0,
                method=SyncMethod.MANUAL,
                confidence=0.0,
                notes="Ffmpeg extraction failed or was not available. Defaulting to 0 sync offset.",
            )

        # Load audio signals
        rate_a, data_a = wavfile.read(wav_a)
        rate_b, data_b = wavfile.read(wav_b)

        # Calculate alignment
        offset_sec, confidence, peak_a, peak_b = align_audio_data(data_a, rate_a, data_b, rate_b)

        # Average FPS
        fps = 0.5 * (fps_a + fps_b)
        offset_frames = offset_sec * fps

        # Plot debug figure if possible
        debug_plot_path = None
        try:
            plot_dir = video_path_a.parent
            plot_path = plot_dir / f"sync_debug_{video_path_a.stem}_{video_path_b.stem}.png"
            generate_alignment_plot(
                data_a, rate_a, data_b, rate_b,
                peak_a, peak_b, offset_sec,
                plot_path
            )
            debug_plot_path = str(plot_path.resolve())
        except Exception as e:
            # Carry on without plot
            pass

        return SyncResult(
            offset_seconds=float(offset_sec),
            offset_frames=float(offset_frames),
            method=SyncMethod.AUDIO_IMPACT,
            confidence=float(confidence),
            debug_plot_path=debug_plot_path,
            notes=f"Successfully aligned waveforms. Audio sample rate A={rate_a}Hz, B={rate_b}Hz. Peak alignment lag: {offset_sec:.6f}s.",
        )
