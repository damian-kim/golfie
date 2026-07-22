import cv2
import numpy as np
import pytest

from golfie_cv.detection import detect_ball_candidates, build_background_model


def test_build_background_model():
    # Generate 5 frames of green turf background with random pixel noise
    np.random.seed(42)
    frames = []
    base_color = np.array([34, 139, 34], dtype=np.uint8)  # green turf color
    for _ in range(5):
        frame = np.zeros((100, 100, 3), dtype=np.uint8) + base_color
        # Add salt and pepper noise to random pixels
        noise = np.random.randint(-10, 10, (100, 100, 3))
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frames.append(frame)

    bg_model = build_background_model(frames)
    
    assert bg_model.shape == (100, 100, 3)
    assert bg_model.dtype == np.uint8
    # The noise should be heavily smoothed out, median close to base color
    assert np.allclose(np.mean(bg_model, axis=(0, 1)), base_color, atol=2.0)


def test_detect_ball_candidates_circular_ball():
    # 100x100 dark background
    bg_model = np.zeros((100, 100, 3), dtype=np.uint8)
    frame = bg_model.copy()
    
    # Draw a solid white circle at center (50, 60) with radius 6
    cv2.circle(frame, (50, 60), 6, (255, 255, 255), thickness=-1)

    candidates = detect_ball_candidates(frame, bg_model)

    assert len(candidates) >= 1
    best_cand = candidates[0]

    # Check candidate coordinates and radius
    assert best_cand.x_px == pytest.approx(50.0, abs=1.0)
    assert best_cand.y_px == pytest.approx(60.0, abs=1.0)
    assert best_cand.radius_px == pytest.approx(6.0, abs=1.5)
    assert best_cand.is_streak is False
    assert best_cand.confidence > 0.8


def test_detect_ball_candidates_streak_ball():
    bg_model = np.zeros((120, 120, 3), dtype=np.uint8)
    frame = bg_model.copy()

    # Draw a thick white line representing a motion blur streak from (30, 40) to (70, 80)
    cv2.line(frame, (30, 40), (70, 80), (255, 255, 255), thickness=6)

    candidates = detect_ball_candidates(frame, bg_model)

    assert len(candidates) >= 1
    best_cand = candidates[0]

    # Center should be around the midpoint of the line: (50, 60)
    assert best_cand.x_px == pytest.approx(50.0, abs=2.0)
    assert best_cand.y_px == pytest.approx(60.0, abs=2.0)
    
    # It must be classified as a streak
    assert best_cand.is_streak is True
    assert best_cand.confidence > 0.7


def test_detect_large_optic_yellow_phone_streak():
    bg_model = np.full((1080, 1920, 3), (45, 120, 45), dtype=np.uint8)
    frame = bg_model.copy()
    cv2.ellipse(frame, (1450, 850), (95, 14), -18, 0, 360, (0, 255, 255), -1)

    candidates = detect_ball_candidates(frame, bg_model)

    optic = [candidate for candidate in candidates if candidate.is_optic_color]
    assert optic
    assert optic[0].x_px == pytest.approx(1450, abs=3)
    assert optic[0].y_px == pytest.approx(850, abs=3)
    assert optic[0].is_streak
    assert optic[0].bbox_width_px > 150


def test_detect_backlit_dark_ball_silhouette():
    bg_model = np.full((180, 240, 3), (190, 150, 100), dtype=np.uint8)
    frame = bg_model.copy()
    cv2.circle(frame, (155, 62), 7, (25, 25, 25), thickness=-1)

    candidates = detect_ball_candidates(frame, bg_model)

    dark = [candidate for candidate in candidates if candidate.appearance_score == 0.45]
    assert dark
    assert dark[0].x_px == pytest.approx(155, abs=1)
    assert dark[0].y_px == pytest.approx(62, abs=1)
    assert dark[0].confidence >= 0.68
