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

    # Background subtraction if available
    if background_model is not None:
        if len(background_model.shape) == 3:
            bg_gray = cv2.cvtColor(background_model, cv2.COLOR_BGR2GRAY)
        else:
            bg_gray = background_model
        
        # Compute absolute difference
        diff = cv2.absdiff(gray, bg_gray)
        # Threshold difference
        _, motion_mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
    else:
        motion_mask = np.ones_like(gray) * 255

    # Threshold absolute brightness for a white/highlight ball
    _, bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # Combine masks
    mask = cv2.bitwise_and(motion_mask, bright_mask)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        # Ignore extremely small noise or overly large shapes
        if area < 12 or area > 5000:
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

        # Compute confidence score
        if is_streak:
            # Scale confidence by contour solidity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / (hull_area + 1e-6)
            confidence = float(solidity * 0.9)
        else:
            # Circle shape confidence based on circularity
            confidence = float(min(1.0, circularity))

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
