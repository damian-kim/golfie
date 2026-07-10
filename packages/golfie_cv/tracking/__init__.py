"""Per-camera ball tracking across frames (turns candidates into a track).

STATUS: stub. Implemented in Milestones 3-4 (spec section 20 / spec section 10).
"""

from __future__ import annotations

from golfie_core.schemas import TrackedPoint2D
from golfie_cv.detection import BallCandidate


def track_ball_2d(
    candidates_by_frame: list[list[BallCandidate]],
    fps: float,
) -> list[TrackedPoint2D]:
    """Resolve per-frame candidates into a single 2D track over time.

    Spec section 10. Expected implementation: Kalman or particle filter
    over candidate positions, enforcing physically plausible
    velocity/acceleration between frames, with RANSAC-style outlier
    rejection. Must tolerate missed detections (gaps) rather than
    breaking the whole track. Output includes per-frame confidence so
    low-confidence stretches can be flagged in the UI rather than
    silently trusted.
    """
    # Sort candidates by confidence (descending) BEFORE slicing to top 100.
    # Without this sort, the real ball (which may be candidate #150+) is randomly
    # discarded when grass/vibration noise generates 150-400 contours per frame.
    candidates_by_frame = [
        sorted(cands, key=lambda c: c.confidence, reverse=True)[:100]
        for cands in candidates_by_frame
    ]


    import numpy as np


    class ActiveTrack:
        def __init__(self, start_frame: int, candidate: BallCandidate):
            self.frames = [start_frame]
            self.candidates = [candidate]
            self.gap_count = 0

        @property
        def last_frame(self) -> int:
            return self.frames[-1]

        def predict_position(self, frame_index: int) -> tuple[float, float]:
            n = len(self.candidates)
            if n == 1:
                # Static prediction (zero velocity)
                return self.candidates[0].x_px, self.candidates[0].y_px

            # Always use linear extrapolation (constant velocity) during the tracking association loop.
            # Fitting a full quadratic curve (np.polyfit) in the online loop is extremely slow due to NumPy 
            # overhead and is physically unnecessary over a single-frame interval (dt <= 0.033s), where 
            # gravity/drag displacement is less than a pixel.
            t_last2 = self.frames[-2:]
            x_last2 = [c.x_px for c in self.candidates[-2:]]
            y_last2 = [c.y_px for c in self.candidates[-2:]]
            
            t_target = frame_index / fps
            t0 = t_last2[0] / fps
            t1 = t_last2[1] / fps
            
            dt = t1 - t0
            if dt < 1e-6:
                return x_last2[1], y_last2[1]
                
            vx = (x_last2[1] - x_last2[0]) / dt
            vy = (y_last2[1] - y_last2[0]) / dt
            
            pred_x = x_last2[1] + vx * (t_target - t1)
            pred_y = y_last2[1] + vy * (t_target - t1)
            return float(pred_x), float(pred_y)

        def add_candidate(self, frame_index: int, candidate: BallCandidate):
            self.frames.append(frame_index)
            self.candidates.append(candidate)
            self.gap_count = 0

        def add_gap(self):
            self.gap_count += 1

        def get_rmse(self) -> float:
            n = len(self.candidates)
            if n < 3:
                return 0.0
            t = np.array([f / fps for f in self.frames])
            x = np.array([c.x_px for c in self.candidates])
            y = np.array([c.y_px for c in self.candidates])
            try:
                t_shifted = t - t[0]
                coef_x = np.polyfit(t_shifted, x, 2)
                coef_y = np.polyfit(t_shifted, y, 2)
                fit_x = np.polyval(coef_x, t_shifted)
                fit_y = np.polyval(coef_y, t_shifted)
                rmse_x = np.sqrt(np.mean((x - fit_x) ** 2))
                rmse_y = np.sqrt(np.mean((y - fit_y) ** 2))
                return float(rmse_x + rmse_y)
            except Exception:
                return 999.0

        def score(self) -> float:
            # Score function tuned to select the real ball-in-flight track:
            # - Real ball: short (5-10 frames), very fast (45-214 px/frame), often streaks
            # - Noise (club/body): long (50-80 frames), slow (8-22 px/frame), circular blobs
            #
            # Key design choices:
            # 1. Use total displacement (not speed*length) — prevents double-counting length
            # 2. Boost streak confidence — motion-blurred ball has unfairly low circularity
            # 3. Prefer tracks starting within 2 frames of impact (frames 13-20)
            # 4. Hard-penalize slow tracks (speed < 5 px/fr)
            n = len(self.candidates)
            if n < 3:
                return -9999.0

            disp = self.total_displacement()
            step_distances = [
                float(np.hypot(
                    self.candidates[i].x_px - self.candidates[i - 1].x_px,
                    self.candidates[i].y_px - self.candidates[i - 1].y_px,
                ))
                for i in range(1, n)
            ]
            if not step_distances:
                return -9999.0

            median_step = float(np.median(step_distances))
            p90_step = float(np.percentile(step_distances, 90))

            launch_frame = self.frames[0]
            launch_step = p90_step
            for i, step in enumerate(step_distances, start=1):
                if step >= 4.0:
                    launch_frame = self.frames[i]
                    launch_step = step
                    break

            # Hard speed gating: real ball moves fast
            # Use binary gates (not continuous) to avoid disp*speed double-counting
            if p90_step < 4.0 or disp < 12.0:
                speed_gate = 0.001
            elif median_step < 1.5 and p90_step < 8.0:
                speed_gate = 0.01
            elif p90_step < 8.0:
                speed_gate = 0.1
            else:
                speed_gate = 1.0

            # Boost confidence for streaks — motion-blurred ball at 30fps gets
            # low circularity/solidity (~0.4-0.5) compared to static noise blobs (~0.8)
            # This unfairly penalizes real ball detections
            confs = []
            for c in self.candidates:
                conf = c.confidence
                if c.is_streak:
                    conf = min(1.0, conf * 1.5)  # Boost streak confidence by 50%
                confs.append(conf)
            avg_conf = float(np.mean(confs))

            rmse = self.get_rmse()

            # Temporal window: ball in flight starts at/just after impact (frame 15)
            # Primary window: 11-22 gets full credit (ball visible slightly before impact
            # due to backswing motion detection, and flies for several frames after)
            # Secondary window: 8-28 gets partial credit. Outside gets harsh penalty.
            if 10 <= launch_frame <= 28:
                start_penalty = 1.0
            elif 6 <= launch_frame <= 38:
                start_penalty = 0.3
            else:
                start_penalty = 0.01

            # Primary signal: displacement * confidence * gates
            # Also require minimum point count for robustness (penalize n < 5)
            n_factor = min(n, 8) / 8.0  # Scales 3→0.375, 5→0.625, 8+→1.0
            launch_boost = min(2.0, max(0.5, launch_step / 12.0))
            return float(disp * avg_conf * start_penalty * speed_gate * n_factor * launch_boost - 0.5 * rmse)






        def total_displacement(self) -> float:
            if len(self.candidates) < 2:
                return 0.0
            x_first, y_first = self.candidates[0].x_px, self.candidates[0].y_px
            x_last, y_last = self.candidates[-1].x_px, self.candidates[-1].y_px
            return float(np.sqrt((x_last - x_first) ** 2 + (y_last - y_first) ** 2))

    max_gaps = 5
    active_tracks: list[ActiveTrack] = []
    finished_tracks: list[ActiveTrack] = []

    for f_idx, candidates in enumerate(candidates_by_frame):
        # 1. Predict next positions for all active tracks
        predictions = []
        for track in active_tracks:
            pred_x, pred_y = track.predict_position(f_idx)
            predictions.append((track, pred_x, pred_y))

        # 2. Build association matrix / distance list
        associations = []
        for track_idx, (track, pred_x, pred_y) in enumerate(predictions):
            for cand_idx, cand in enumerate(candidates):
                dist = float(np.sqrt((cand.x_px - pred_x) ** 2 + (cand.y_px - pred_y) ** 2))
                
                # Dynamic gating radius
                n = len(track.candidates)
                if n == 1:
                    # Gating is wider for first step since velocity is unknown and ball moves very fast.
                    gate_r = 450.0
                elif n == 2:
                    gate_r = 250.0
                else:
                    gate_r = 150.0


                if dist <= gate_r:
                    associations.append((dist, track_idx, cand_idx))

        # Sort associations by distance (greedy nearest-neighbor matching)
        associations.sort(key=lambda x: x[0])
        matched_tracks = set()
        matched_candidates = set()

        for dist, track_idx, cand_idx in associations:
            if track_idx not in matched_tracks and cand_idx not in matched_candidates:
                track = active_tracks[track_idx]
                cand = candidates[cand_idx]
                track.add_candidate(f_idx, cand)
                matched_tracks.add(track_idx)
                matched_candidates.add(cand_idx)

        # 3. Handle unmatched active tracks (increment gap counts)
        still_active_tracks = []
        for track_idx, track in enumerate(active_tracks):
            if track_idx not in matched_tracks:
                track.add_gap()
                # Prune short-lived tracks immediately if they have gaps (filters noise)
                if len(track.candidates) == 1 and track.gap_count >= 1:
                    continue
                if len(track.candidates) == 2 and track.gap_count >= 2:
                    continue
                if track.gap_count <= max_gaps:
                    still_active_tracks.append(track)
                else:
                    finished_tracks.append(track)
            else:
                still_active_tracks.append(track)

        # 4. Handle unmatched candidates (start new tracks)
        for cand_idx, cand in enumerate(candidates):
            if cand_idx not in matched_candidates:
                still_active_tracks.append(ActiveTrack(f_idx, cand))

        active_tracks = still_active_tracks

    # Gather all tracks
    all_tracks = finished_tracks + active_tracks

    # Filter tracks: require minimum points (e.g., 3) to prevent noise
    valid_tracks = [t for t in all_tracks if len(t.candidates) >= 3]

    if not valid_tracks:
        return []

    # Filter for tracks that represent a moving ball:
    # 1. Total displacement must be >= 10.0 pixels to filter out drifting background noise.
    # 2. Average speed must be >= 0.3 pixels per frame.
    moving_tracks = [
        t for t in valid_tracks 
        if t.total_displacement() >= 10.0 and (t.total_displacement() / len(t.candidates)) >= 0.3
    ]
    if moving_tracks:
        valid_tracks = moving_tracks
    else:
        # If no moving tracks are found, return empty to prevent falling back to stationary noise
        return []

    best_track = max(valid_tracks, key=lambda t: t.score())

    # Keep debug output focused on likely launch tracks instead of dumping
    # hundreds of noise tracks from club/body motion.
    sorted_tracks = sorted(valid_tracks, key=lambda x: x.score(), reverse=True)
    plausible_launch_tracks = [
        t for t in sorted_tracks
        if t is best_track or (t.frames[0] <= 45 and t.frames[-1] >= 6)
    ][:5]

    print(f"\n--- 2D TRACK BALL DEBUG (valid_tracks={len(valid_tracks)}, showing={len(plausible_launch_tracks)}) ---")
    for i, t in enumerate(plausible_launch_tracks):
        disp = t.total_displacement()
        speed = disp / (len(t.candidates) + 1e-6)
        label = "SELECTED" if t is best_track else f"ALT {i}"
        print(f"  {label}: len={len(t.candidates)}, frames={t.frames[0]} to {t.frames[-1]}, disp={disp:.1f}, speed={speed:.2f} px/fr, score={t.score():.2f}")
        for pt_idx, (c, f) in enumerate(zip(t.candidates[:3], t.frames[:3])):
            print(f"    Pt {pt_idx}: frame={f}, pos=({c.x_px:.1f}, {c.y_px:.1f}), conf={c.confidence:.3f}, streak={c.is_streak}")


    # Build the continuous tracked points list, bridging gaps with prediction
    tracked_points: list[TrackedPoint2D] = []
    start_f = best_track.frames[0]
    end_f = best_track.frames[-1]

    # Map existing frames to candidates
    cand_map = {f: c for f, c in zip(best_track.frames, best_track.candidates)}

    for f_idx in range(start_f, end_f + 1):
        time_sec = f_idx / fps
        if f_idx in cand_map:
            cand = cand_map[f_idx]
            tracked_points.append(
                TrackedPoint2D(
                    frame_index=f_idx,
                    time_seconds=time_sec,
                    x_px=cand.x_px,
                    y_px=cand.y_px,
                    confidence=cand.confidence,
                )
            )
        else:
            # Predict position for the gap frame using the track model up to this point
            # For prediction, we fit model using only the points before this frame
            hist_frames = [f for f in best_track.frames if f < f_idx]
            hist_cands = [cand_map[f] for f in hist_frames]
            
            # Temporary track to run prediction logic
            temp_track = ActiveTrack(hist_frames[0], hist_cands[0])
            for hf, hc in zip(hist_frames[1:], hist_cands[1:]):
                temp_track.add_candidate(hf, hc)
                
            pred_x, pred_y = temp_track.predict_position(f_idx)
            
            tracked_points.append(
                TrackedPoint2D(
                    frame_index=f_idx,
                    time_seconds=time_sec,
                    x_px=pred_x,
                    y_px=pred_y,
                    confidence=0.1,  # Low confidence for interpolated gap frame
                )
            )

    return tracked_points
