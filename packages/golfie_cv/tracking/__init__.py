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
            # Score matches based on candidate count, average confidence, and quadratic fit RMSE
            n = len(self.candidates)
            avg_conf = np.mean([c.confidence for c in self.candidates]) if n > 0 else 0.0
            rmse = self.get_rmse()
            return float(n * avg_conf - 0.2 * rmse)

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
                    # Gating is wider for first step since velocity is unknown.
                    # If candidate is a streak, it's moving fast, so we expand the search.
                    gate_r = 150.0 if (cand.is_streak or track.candidates[0].is_streak) else 50.0
                elif n == 2:
                    gate_r = 30.0
                else:
                    gate_r = 15.0

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

    # Filter for tracks that represent a fast-moving ball (average speed >= 5.0 pixels per frame)
    # This filters out background noise/drifting tracks and slower descent tracks.
    moving_tracks = [t for t in valid_tracks if (t.total_displacement() / len(t.candidates)) >= 5.0]
    if moving_tracks:
        valid_tracks = moving_tracks

    # Select the track with the highest score
    best_track = max(valid_tracks, key=lambda t: t.score())

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
