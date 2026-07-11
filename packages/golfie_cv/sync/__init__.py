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
    from pathlib import Path
    
    # 1. Check if already in PATH
    if shutil.which("ffmpeg"):
        return "ffmpeg"
        
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        local_path = Path(local_app_data)
        
        # Check Medal ffmpeg
        medal_bin = local_path / "Medal" / "ffmpeg7.exe"
        if medal_bin.exists():
            return str(medal_bin)
            
        # Check Overwolf obs 64bit extension ffmpeg
        ow_ext = local_path / "Overwolf" / "Extensions" / "ncfplpkmiejjaklknfnkgcpapnhkggmlcppckhcb"
        if ow_ext.exists():
            try:
                for sub in ow_ext.iterdir():
                    if sub.is_dir():
                        candidate = sub / "obs" / "bin" / "64bit" / "ffmpeg.exe"
                        if candidate.exists():
                            return str(candidate)
            except Exception:
                pass

    # 2. Check common Windows installation paths
    common_paths = [
        str(Path(__file__).resolve().parents[3] / "bin" / "ffmpeg.exe"),
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"E:\ffmpeg\bin\ffmpeg.exe",
        r"E:\Golfie\ffmpeg\bin\ffmpeg.exe",
        r"E:\Golfie\ffmpeg\ffmpeg.exe",
    ]
    for p in common_paths:
        if os.path.exists(p):
            return p
            
    # 3. Check if E:\ffmpeg contains a nested build directory (very common for zip extractions)
    try:
        ffmpeg_root = Path(r"E:\ffmpeg")
        if ffmpeg_root.exists():
            for sub in ffmpeg_root.iterdir():
                if sub.is_dir():
                    candidate = sub / "bin" / "ffmpeg.exe"
                    if candidate.exists():
                        return str(candidate)
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
    """Locate the same impulsive event from two independently processed mics.

    Phone AGC, clipping, echo cancellation, and codec phase often make the raw
    clap waveforms look quite different even though their energy transients are
    unambiguous.  Synchronization therefore uses a short RMS-energy envelope
    and scores peak prominence, rather than requiring raw-waveform correlation.
    """
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

    # Ensure identical sample rates (standard for ffmpeg output config)
    if rate_a != rate_b:
        raise ValueError(f"Sample rates must match. Got rate_a={rate_a}Hz and rate_b={rate_b}Hz.")

    def locate_transient(data: np.ndarray, rate: int) -> tuple[int, float]:
        if data.size == 0:
            raise ValueError("Audio stream is empty.")

        # A 2 ms RMS window suppresses isolated codec spikes while preserving
        # the sharp onset needed for sub-frame video alignment.
        window = max(3, int(round(rate * 0.002)))
        kernel = np.ones(window, dtype=np.float64) / window
        energy = np.convolve(np.square(data.astype(np.float64)), kernel, mode="same")
        rms = np.sqrt(np.maximum(energy, 0.0))
        coarse_peak = int(np.argmax(rms))

        baseline = float(np.median(rms))
        mad = float(np.median(np.abs(rms - baseline)))
        noise_scale = max(1.4826 * mad, float(np.percentile(rms, 75)) * 0.05, 1e-7)
        prominence = max(0.0, (float(rms[coarse_peak]) - baseline) / noise_scale)

        # Energy centroid around the onset is more stable across clipping and
        # echo than whichever single sample happened to hit the largest value.
        radius = max(window, int(round(rate * 0.006)))
        lo = max(0, coarse_peak - radius)
        hi = min(len(energy), coarse_peak + radius + 1)
        local = np.maximum(energy[lo:hi] - baseline * baseline, 0.0)
        if float(np.sum(local)) > 1e-12:
            peak = int(round(float(np.sum(np.arange(lo, hi) * local) / np.sum(local))))
        else:
            peak = coarse_peak

        # Prominence of roughly 8 robust noise scales is already a clear clap.
        confidence = float(np.clip((prominence - 3.0) / 8.0, 0.0, 1.0))
        return peak, confidence

    peak_a, confidence_a = locate_transient(data_a, rate_a)
    peak_b, confidence_b = locate_transient(data_b, rate_b)
    offset_sec = (peak_a / rate_a) - (peak_b / rate_b)
    confidence = min(confidence_a, confidence_b)

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
