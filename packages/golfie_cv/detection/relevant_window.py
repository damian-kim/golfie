import cv2
import numpy as np
import scipy.io.wavfile as wavfile
import tempfile
from pathlib import Path
from golfie_cv.sync import _find_ffmpeg_fallback, _extract_audio

def detect_relevant_window(video_path: Path, fps: float, frame_count: int) -> tuple[int, int, float]:
    """Detect the relevant parts of a video fast (start_frame, end_frame, impact_time).
    
    Checks audio peak transients, validates with a visual motion surge inside the golfer center ROI.
    Falls back to visual motion analysis if validation fails or audio is silent.
    We target a 5-second window: 3 seconds before impact/action, and 2 seconds after.
    """
    video_path = Path(video_path)
    t_impact = None
    
    # 1. Gather potential audio impact candidates
    audio_candidates = []
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "audio.wav"
        if _extract_audio(video_path, wav_path):
            try:
                rate, data = wavfile.read(wav_path)
                if len(data.shape) > 1:
                    data = data.mean(axis=1)
                
                # Normalize
                if np.issubdtype(data.dtype, np.integer):
                    data = data.astype(np.float32) / np.iinfo(data.dtype).max
                else:
                    data = data.astype(np.float32)
                
                env = np.abs(data)
                
                # Retrieve local peaks with 0.5s minimum separation
                min_peak_dist = int(rate * 0.5)
                peaks = []
                for i in range(min_peak_dist, len(env) - min_peak_dist, min_peak_dist):
                    local_max_idx = i + int(np.argmax(env[i:i+min_peak_dist]))
                    if env[local_max_idx] > 0.05:
                        peaks.append((local_max_idx, env[local_max_idx]))
                
                # Sort peaks by amplitude descending and keep top 3
                peaks.sort(key=lambda x: x[1], reverse=True)
                for p_idx, _ in peaks[:3]:
                    audio_candidates.append(p_idx / rate)
            except Exception:
                pass
                
    # 2. Visual motion scan on Golfer ROI (center 60% of screen width)
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Coarse sample of up to 150 frames to be extremely fast
            sample_step = max(1, total_frames // 150)
            
            motion_scores = []
            prev_roi_gray = None
            
            for frame_idx in range(0, total_frames, sample_step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                
                h, w = frame.shape[:2]
                # Golfer ROI: central 60% width, 10% to 90% height
                x_start, x_end = int(w * 0.2), int(w * 0.8)
                y_start, y_end = int(h * 0.1), int(h * 0.9)
                roi = frame[y_start:y_end, x_start:x_end]
                
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray_small = cv2.resize(gray, (160, 120))
                
                if prev_roi_gray is not None:
                    diff = cv2.absdiff(gray_small, prev_roi_gray)
                    _, thresholded = cv2.threshold(diff, 12, 255, cv2.THRESH_BINARY)
                    motion_score = float(np.sum(thresholded) / thresholded.size)
                    motion_scores.append((frame_idx, motion_score))
                    
                prev_roi_gray = gray_small
                
            if motion_scores:
                # Apply rolling average smoothing (width of 3 taps) to the motion energy curve
                scores_array = [score for idx, score in motion_scores]
                smoothed_scores = []
                for i in range(len(scores_array)):
                    start_i = max(0, i - 1)
                    end_i = min(len(scores_array), i + 2)
                    smoothed_scores.append(float(np.mean(scores_array[start_i:end_i])))
                
                # Cross-validate the audio candidates against visual motion energy
                best_audio_val = None
                for t_candidate in audio_candidates:
                    near_scores = []
                    for list_idx, (f_idx, _) in enumerate(motion_scores):
                        t_frame = f_idx / fps
                        if abs(t_frame - t_candidate) <= 0.75:
                            near_scores.append((smoothed_scores[list_idx], t_candidate))
                    
                    if near_scores:
                        near_scores.sort(key=lambda x: x[0], reverse=True)
                        # Validate if there is substantial visual motion around the audio peak
                        if near_scores[0][0] > 0.02:
                            best_audio_val = near_scores[0][1]
                            break
                            
                if best_audio_val is not None:
                    t_impact = best_audio_val
                else:
                    # Fallback to the maximum motion peak of the smoothed golfer ROI curve
                    max_motion_idx = int(np.argmax(smoothed_scores))
                    t_impact = motion_scores[max_motion_idx][0] / fps
        except Exception:
            pass
        finally:
            cap.release()

    # 3. Fallback to middle of video if everything failed
    if t_impact is None or t_impact <= 0:
        t_impact = (frame_count / fps) / 2.0

    # 4. Construct 5-second window: 3.0 seconds before, 2.0 seconds after impact
    start_time = max(0.0, t_impact - 3.0)
    end_time = min(frame_count / fps, t_impact + 2.0)
    
    # Adjust start and end to ensure a full window if video duration allows
    duration = frame_count / fps
    if duration >= 5.0:
        if start_time == 0.0:
            end_time = 5.0
        elif end_time == duration:
            start_time = max(0.0, duration - 5.0)
            
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    # Ensure indices are within bounds
    start_frame = max(0, min(start_frame, frame_count - 1))
    end_frame = max(start_frame + 1, min(end_frame, frame_count))
    
    return start_frame, end_frame, t_impact
