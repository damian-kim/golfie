"""Per-frame ball candidate detection.

STATUS: stub. Implemented in Milestone 3 (spec section 20 / spec section 9).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BallCandidate:
    """One possible ball detection in a single frame.

    Detectors must return *multiple* candidates with confidence rather
    than a single answer (spec section 9) -- downstream tracking decides
    which candidate is physically plausible across the sequence.
    """

    x_px: float
    y_px: float
    radius_px: float
    confidence: float
    is_streak: bool = False  # True if detected as a motion-blur streak, not a circle


import cv2

def detect_ball_candidates(frame: np.ndarray, background_model: np.ndarray | None = None) -> list[BallCandidate]:
    """Find candidate ball detections in a single frame.

    Spec section 9.
    """
    # Convert frame to grayscale if it is BGR
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    diff = None

    # Background subtraction if available
    if background_model is not None:
        if len(background_model.shape) == 3:
            bg_gray = cv2.cvtColor(background_model, cv2.COLOR_BGR2GRAY)
        else:
            bg_gray = background_model
        
        # Compute absolute difference to detect both light-on-dark and dark-on-light motion
        diff = cv2.absdiff(gray, bg_gray)
        # Threshold difference to filter high-frequency flicker/camera vibration.
        _, motion_mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
    else:
        motion_mask = np.ones_like(gray) * 255

    # Prefer golf-ball-like pixels: bright/white, but do not require a fixed
    # global brightness because indoor turf/net lighting varies a lot.
    _, bright_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    if len(frame.shape) == 3:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        white_mask = cv2.inRange(hsv, (0, 0, 105), (179, 95, 255))
        specular_mask = cv2.inRange(value, 145, 255)
        color_mask = cv2.bitwise_or(white_mask, specular_mask)
    else:
        saturation = np.zeros_like(gray)
        value = gray
        color_mask = bright_mask

    # Combine masks
    mask = cv2.bitwise_and(motion_mask, cv2.bitwise_or(bright_mask, color_mask))

    # Clean one-pixel sensor noise without merging separate moving objects.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        # Ignore extremely small noise or large moving body/club regions. A
        # motion-blurred ball streak is still usually far below this size.
        if area < 4 or area > 2000:
            continue

        # Get enclosing circle parameters
        (x, y), radius = cv2.minEnclosingCircle(contour)
        
        # Circularity check
        perimeter = cv2.arcLength(contour, True)
        circularity = 0.0
        if perimeter > 0:
            circularity = (4.0 * np.pi * area) / (perimeter ** 2)

        # Check for motion-blurred streaks (needs >= 5 points for fitEllipse)
        is_streak = False
        if len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                (e_x, e_y), (ma, MA), angle = ellipse
                aspect_ratio = MA / (ma + 1e-6)
                # If elongated, classify as streak
                if aspect_ratio >= 1.5:
                    is_streak = True
                    x, y = e_x, e_y
                    # Streak radius is average axis half-length
                    radius = 0.25 * (ma + MA)
            except cv2.error:
                pass

        x_i = int(np.clip(round(x), 0, gray.shape[1] - 1))
        y_i = int(np.clip(round(y), 0, gray.shape[0] - 1))
        contour_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
        mean_value = float(cv2.mean(value, mask=contour_mask)[0])
        mean_saturation = float(cv2.mean(saturation, mask=contour_mask)[0])
        whiteness_score = np.clip((mean_value - 70.0) / 150.0, 0.0, 1.0) * np.clip(
            (140.0 - mean_saturation) / 140.0, 0.25, 1.0
        )
        if diff is not None:
            motion_strength = float(diff[y_i, x_i]) / 255.0
        else:
            motion_strength = 0.7

        # The in-flight ball is small; penalize bigger regions without throwing
        # away close or blurred balls outright.
        size_score = float(np.clip(1.0 - (area / 2000.0), 0.15, 1.0))

        # Compute confidence score
        if is_streak:
            # Scale confidence by contour solidity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / (hull_area + 1e-6)
            shape_confidence = float(solidity * 0.9)
        else:
            # Circle shape confidence based on circularity
            shape_confidence = float(min(1.0, circularity))

        confidence = float(
            0.45 * shape_confidence
            + 0.25 * whiteness_score
            + 0.20 * motion_strength
            + 0.10 * size_score
        )
        confidence = max(0.1, min(1.0, confidence))

        candidates.append(
            BallCandidate(
                x_px=float(x),
                y_px=float(y),
                radius_px=float(radius),
                confidence=confidence,
                is_streak=is_streak
            )
        )

    # Sort by confidence highest first
    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return candidates


def build_background_model(pre_impact_frames: list[np.ndarray]) -> np.ndarray:
    """Build a background model from frames captured before impact."""
    if not pre_impact_frames:
        raise ValueError("No frames provided for background modeling.")
    # Median along the frames list axis
    return np.median(pre_impact_frames, axis=0).astype(np.uint8)


from golfie_cv.detection.relevant_window import detect_relevant_window
from golfie_cv.detection.yolo_outlines import render_ball_detection_replay, render_stripped_outlines_video
