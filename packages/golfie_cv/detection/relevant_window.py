import cv2
import numpy as np
import scipy.io.wavfile as wavfile
import tempfile
from pathlib import Path
from golfie_cv.sync import _find_ffmpeg_fallback, _extract_audio


def detect_impact_frame_fast(video_path: Path) -> tuple[int, float]:
    """Locate club/ball launch with one sequential low-resolution decode.

    Shot clips are already short slow-motion renders. Extracting audio and
    repeatedly seeking a VFR H.264 stream took 20-40 seconds and can return
    inconsistent frame ordinals. Sequential visual motion is both faster and
    aligned with the ordinals used by candidate detection.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video for impact detection: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    scores: list[float] = []
    ball_scores: list[float] = []
    timestamps: list[float] = []
    previous = None
    previous_gray = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            # Exclude sky/top edge and extreme side clutter while retaining
            # both the DTL and face-on hitting areas.
            roi = cv2.GaussianBlur(gray[25:178, 12:308], (5, 5), 0)
            if previous is None:
                scores.append(0.0)
                ball_scores.append(0.0)
            else:
                delta = cv2.absdiff(roi, previous)
                # Top-tail motion emphasizes the club/ball burst without a
                # large golfer silhouette dominating the mean.
                active = delta[delta >= 18]
                score = float(np.mean(active) * np.sqrt(len(active))) if len(active) else 0.0
                scores.append(score)
                hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
                yellow = cv2.inRange(hsv, (8, 70, 90), (45, 255, 255))
                count, component_labels, stats, _ = cv2.connectedComponentsWithStats(yellow, 8)
                best_ball_score = 0.0
                full_delta = cv2.absdiff(gray, previous_gray)
                for label in range(1, count):
                    left, top, width, height, area = stats[label]
                    if area < 2 or area > 800 or width > 100 or height > 70:
                        continue
                    if top + 0.5 * height < 0.52 * gray.shape[0]:
                        continue
                    component = np.where(
                        component_labels[top:top + height, left:left + width] == label, 255, 0
                    ).astype(np.uint8)
                    motion = cv2.mean(
                        full_delta[top:top + height, left:left + width], mask=component
                    )[0]
                    best_ball_score = max(best_ball_score, float(motion * np.sqrt(area)))
                ball_scores.append(best_ball_score)
            timestamps.append(timestamp)
            previous = roi
            previous_gray = gray
    finally:
        cap.release()

    if len(scores) < 3:
        raise RuntimeError(f"Too few decoded frames for impact detection: {len(scores)}")
    smoothed = np.convolve(np.asarray(scores), np.ones(3) / 3.0, mode="same")
    # Downswing motion peaks just before contact. In phone Slo-mo the coloured
    # ball's first high-motion component provides a much sharper launch cue.
    peak = int(np.argmax(smoothed))
    ball_start = min(len(ball_scores) - 1, peak + 4)
    ball_end = min(len(ball_scores), peak + 26)
    local_ball_scores = ball_scores[ball_start:ball_end]
    maximum_ball_score = max(local_ball_scores, default=0.0)
    if ball_end > ball_start and maximum_ball_score > 40.0:
        ball_threshold = max(40.0, 0.35 * maximum_ball_score)
        first_flight = ball_start + int(np.argmax(local_ball_scores))
        for relative, score in enumerate(local_ball_scores):
            if score >= ball_threshold and max(local_ball_scores[relative:relative + 2]) >= ball_threshold:
                first_flight = ball_start + relative
                break
        impact = max(1, first_flight - 1)
    else:
        impact = peak
    timestamp = timestamps[impact]
    if timestamp <= 0.0 and fps > 0:
        timestamp = impact / fps
    return impact, float(timestamp)

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
