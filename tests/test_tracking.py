import numpy as np
import pytest

from golfie_cv.detection import BallCandidate
from golfie_cv.tracking import track_ball_2d


def test_track_ball_2d_perfect_parabola():
    fps = 240.0
    # Simulate a ball flying for 20 frames
    # Trajectory: x(t) = 100 + 400 * t, y(t) = 300 - 200 * t - 150 * t^2
    candidates_by_frame = []
    true_positions = []
    
    for f in range(20):
        t = f / fps
        x = 100.0 + 400.0 * t
        y = 300.0 - 200.0 * t - 150.0 * (t ** 2)
        true_positions.append((x, y))
        
        # Add single perfect candidate
        candidates_by_frame.append([
            BallCandidate(x_px=x, y_px=y, radius_px=5.0, confidence=0.95, is_streak=False)
        ])

    track = track_ball_2d(candidates_by_frame, fps)
    
    assert len(track) == 20
    for i, pt in enumerate(track):
        assert pt.frame_index == i
        assert pt.time_seconds == pytest.approx(i / fps)
        assert pt.x_px == pytest.approx(true_positions[i][0], abs=1e-3)
        assert pt.y_px == pytest.approx(true_positions[i][1], abs=1e-3)
        assert pt.confidence == pytest.approx(0.95)


def test_track_ball_2d_with_gaps():
    fps = 240.0
    # Simulate a ball flying for 15 frames, but frames 5 and 6 have no detections
    candidates_by_frame = []
    true_positions = []
    
    for f in range(15):
        t = f / fps
        x = 50.0 + 600.0 * t
        y = 400.0 - 300.0 * t - 200.0 * (t ** 2)
        true_positions.append((x, y))
        
        if f in (5, 6):
            # Gap frame: no candidate
            candidates_by_frame.append([])
        else:
            candidates_by_frame.append([
                BallCandidate(x_px=x, y_px=y, radius_px=6.0, confidence=0.9, is_streak=True)
            ])

    track = track_ball_2d(candidates_by_frame, fps)
    
    assert len(track) == 15
    for i, pt in enumerate(track):
        assert pt.frame_index == i
        assert pt.time_seconds == pytest.approx(i / fps)
        assert pt.x_px == pytest.approx(true_positions[i][0], abs=1.0)
        assert pt.y_px == pytest.approx(true_positions[i][1], abs=1.0)
        
        if i in (5, 6):
            # Bridged gap frames must have low confidence
            assert pt.confidence == pytest.approx(0.1)
        else:
            assert pt.confidence == pytest.approx(0.9)


def test_track_ball_2d_with_outliers():
    fps = 240.0
    # Simulate a ball flying for 10 frames, with random noise candidates in each frame
    candidates_by_frame = []
    true_positions = []
    np.random.seed(42)
    
    for f in range(10):
        t = f / fps
        x = 200.0 + 300.0 * t
        y = 150.0 - 100.0 * t - 100.0 * (t ** 2)
        true_positions.append((x, y))
        
        cands = [
            # True candidate
            BallCandidate(x_px=x, y_px=y, radius_px=4.0, confidence=0.9, is_streak=False),
            # Noise outlier 1
            BallCandidate(x_px=float(np.random.uniform(0, 800)), y_px=float(np.random.uniform(0, 600)), radius_px=3.0, confidence=0.4, is_streak=False),
            # Noise outlier 2
            BallCandidate(x_px=float(np.random.uniform(0, 800)), y_px=float(np.random.uniform(0, 600)), radius_px=3.0, confidence=0.6, is_streak=False)
        ]
        # Shuffle candidates so the true one isn't always first
        np.random.shuffle(cands)
        candidates_by_frame.append(cands)

    track = track_ball_2d(candidates_by_frame, fps)
    
    assert len(track) == 10
    for i, pt in enumerate(track):
        assert pt.frame_index == i
        assert pt.x_px == pytest.approx(true_positions[i][0], abs=1e-3)
        assert pt.y_px == pytest.approx(true_positions[i][1], abs=1e-3)
        assert pt.confidence == pytest.approx(0.9)


def test_track_ball_2d_stationary_to_moving():
    fps = 240.0
    # Simulate a ball stationary at (100, 200) for 5 frames, then hit at frame 5 and flies for 10 frames
    candidates_by_frame = []
    true_positions = []
    
    for f in range(15):
        if f < 5:
            x = 100.0
            y = 200.0
        else:
            t_fly = (f - 5) / fps
            # Fast launching streak
            x = 100.0 + 800.0 * t_fly
            y = 200.0 - 500.0 * t_fly - 400.0 * (t_fly ** 2)
            
        true_positions.append((x, y))
        
        is_streak = (f >= 5)
        conf = 0.9 if is_streak else 0.95
        candidates_by_frame.append([
            BallCandidate(x_px=x, y_px=y, radius_px=5.0, confidence=conf, is_streak=is_streak)
        ])

    track = track_ball_2d(candidates_by_frame, fps)
    
    # The tracker should track the entire sequence from frame 0 to 14
    assert len(track) == 15
    for i, pt in enumerate(track):
        assert pt.frame_index == i
        assert pt.x_px == pytest.approx(true_positions[i][0], abs=1.5)
        assert pt.y_px == pytest.approx(true_positions[i][1], abs=1.5)
