"""Per-frame ball candidate detection.

Provides the lightweight motion/appearance candidate detector used by the MVP.
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
    appearance_score: float = 0.5  # white/yellow/orange golf-ball colour likelihood
    is_optic_color: bool = False
    bbox_width_px: float = 0.0
    bbox_height_px: float = 0.0


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

    # Candidate extraction has to be bounded. The old implementation accepted
    # every bright moving grass pixel and allocated a full-HD mask per contour,
    # producing 600+ candidates/frame and multi-minute runtimes. Require both
    # meaningful motion and golf-ball-like colour before connected components.
    diff = None
    if background_model is not None:
        bg_gray = (
            cv2.cvtColor(background_model, cv2.COLOR_BGR2GRAY)
            if len(background_model.shape) == 3 else background_model
        )
        diff = cv2.absdiff(gray, bg_gray)
        noise_floor = float(np.median(diff))
        motion_threshold = int(np.clip(noise_floor + 18.0, 22.0, 42.0))
        motion_mask = cv2.inRange(diff, motion_threshold, 255)
    else:
        motion_mask = np.full_like(gray, 255)

    if len(frame.shape) == 3:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        white_mask = cv2.inRange(hsv, (0, 0, 130), (179, 100, 255))
        yellow_orange_mask = cv2.inRange(hsv, (8, 70, 90), (45, 255, 255))
        optic_ball_mask = cv2.inRange(hsv, (18, 130, 180), (42, 255, 255))
        color_mask = cv2.bitwise_or(white_mask, yellow_orange_mask)
    else:
        saturation = np.zeros_like(gray)
        value = gray
        white_mask = cv2.inRange(gray, 130, 255)
        yellow_orange_mask = np.zeros_like(gray)
        optic_ball_mask = np.zeros_like(gray)
        color_mask = white_mask

    mask = cv2.bitwise_and(motion_mask, color_mask)
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    flat_labels = labels.ravel()
    areas = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    safe_areas = np.maximum(areas, 1.0)
    mean_motion = np.bincount(
        flat_labels, weights=(diff if diff is not None else motion_mask).ravel(),
        minlength=component_count,
    ) / safe_areas
    mean_value_by_component = np.bincount(
        flat_labels, weights=value.ravel(), minlength=component_count
    ) / safe_areas
    mean_saturation_by_component = np.bincount(
        flat_labels, weights=saturation.ravel(), minlength=component_count
    ) / safe_areas
    yellow_fraction = np.bincount(
        flat_labels, weights=(yellow_orange_mask > 0).ravel(), minlength=component_count
    ) / safe_areas
    white_likelihood = np.clip(
        (mean_value_by_component - 70.0) / 150.0, 0.0, 1.0
    ) * np.clip((140.0 - mean_saturation_by_component) / 140.0, 0.25, 1.0)
    appearance_by_component = np.maximum(white_likelihood, yellow_fraction)

    widths = stats[:, cv2.CC_STAT_WIDTH]
    heights = stats[:, cv2.CC_STAT_HEIGHT]
    valid = np.flatnonzero(
        (areas >= 3) & (areas <= 10000)
        & (widths <= 360) & (heights <= 180)
        & (mean_motion >= 24.0)
    )
    valid = valid[valid != 0]
    if len(valid):
        preliminary_score = (
            0.50 * appearance_by_component[valid]
            + 0.30 * np.clip(mean_motion[valid] / 160.0, 0.0, 1.0)
            + 0.20 * np.clip(np.sqrt(areas[valid] / 80.0), 0.0, 1.0)
        )
        valid = valid[np.argsort(preliminary_score)[::-1][:60]]
    
    candidates = []

    # A brightly coloured ball can become a near-black silhouette when the
    # down-the-line phone points toward a low sun. The broad colour mask above
    # intentionally excludes those pixels, so retain compact regions that
    # became substantially darker than the pre-impact background. These are
    # only candidates: calibrated stereo geometry and 3D launch physics still
    # decide whether any region is actually the ball.
    if diff is not None:
        dark_change = cv2.subtract(bg_gray, gray)
        dark_mask = cv2.inRange(dark_change, motion_threshold, 255)
        dark_mask = cv2.bitwise_and(dark_mask, motion_mask)
        dark_count, _, dark_stats, dark_centroids = cv2.connectedComponentsWithStats(
            dark_mask, 8
        )
        for dark_label in range(1, dark_count):
            left, top, width, height, dark_area = dark_stats[dark_label]
            if (
                dark_area < 6
                or dark_area > 3000
                or width > 180
                or height > 120
            ):
                continue
            aspect = max(width, height) / max(1.0, min(width, height))
            if aspect > 12.0:
                continue
            x, y = dark_centroids[dark_label]
            local_change = float(
                np.mean(dark_change[top:top + height, left:left + width])
            ) / 255.0
            if local_change < 0.10:
                continue
            confidence = float(np.clip(0.68 + 0.25 * local_change, 0.68, 0.93))
            candidates.append(
                BallCandidate(
                    x_px=float(x),
                    y_px=float(y),
                    radius_px=float(0.25 * (width + height)),
                    confidence=confidence,
                    is_streak=aspect >= 1.6,
                    appearance_score=0.45,
                    bbox_width_px=float(width),
                    bbox_height_px=float(height),
                )
            )

    # Optic-yellow balls deserve a dedicated pool. On outdoor grass the broad
    # motion mask can fragment or merge a long exposure streak, whereas the
    # high-saturation ball remains one clean component in both supplied phone
    # views. Keep these ahead of generic white/highlight candidates.
    # Colour alone is not enough outdoors: sunlit grass and range markers can
    # contain thousands of optic-yellow pixels and previously entered the
    # tracker as high-confidence stationary "balls". Require the optic colour
    # to overlap foreground motion, with a tiny dilation to keep blurred streak
    # edges connected.
    optic_foreground_mask = cv2.bitwise_and(
        optic_ball_mask,
        cv2.dilate(motion_mask, np.ones((3, 3), dtype=np.uint8), iterations=1),
    )
    optic_count, _, optic_stats, optic_centroids = cv2.connectedComponentsWithStats(
        optic_foreground_mask, 8
    )
    for optic_label in range(1, optic_count):
        left, top, width, height, optic_area = optic_stats[optic_label]
        if optic_area < 20 or optic_area > 15000 or width > 400 or height > 220:
            continue
        aspect = max(width, height) / max(1.0, min(width, height))
        is_streak = aspect >= 1.45 or optic_area >= 1400
        x, y = optic_centroids[optic_label]
        local_motion = 0.0
        if diff is not None:
            local_motion = float(np.mean(diff[top:top + height, left:left + width])) / 255.0
            if local_motion < 0.06:
                continue
        confidence = float(np.clip(0.88 + 0.10 * local_motion, 0.88, 0.99))
        candidates.append(
            BallCandidate(
                x_px=float(x),
                y_px=float(y),
                radius_px=float(0.25 * (width + height)),
                confidence=confidence,
                is_streak=is_streak,
                appearance_score=1.0,
                is_optic_color=True,
                bbox_width_px=float(width),
                bbox_height_px=float(height),
            )
        )
    for label in valid:
        left, top, width, height, pixel_area = stats[label]
        if pixel_area < 3 or pixel_area > 10000 or width > 360 or height > 180:
            continue

        local_labels = labels[top:top + height, left:left + width]
        local_mask = np.where(local_labels == label, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        area = max(float(cv2.contourArea(contour)), float(pixel_area) * 0.65)

        (x, y), radius = cv2.minEnclosingCircle(contour)
        x += left
        y += top
        
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
                (e_x, e_y), (ma, MA), _ = ellipse
                aspect_ratio = MA / (ma + 1e-6)
                # If elongated, classify as streak
                if aspect_ratio >= 1.5:
                    is_streak = True
                    x, y = e_x + left, e_y + top
                    # Streak radius is average axis half-length
                    radius = 0.25 * (ma + MA)
            except cv2.error:
                pass

        component_pixels = local_mask > 0
        mean_value = float(mean_value_by_component[label])
        mean_saturation = float(mean_saturation_by_component[label])
        whiteness_score = np.clip((mean_value - 70.0) / 150.0, 0.0, 1.0) * np.clip(
            (140.0 - mean_saturation) / 140.0, 0.25, 1.0
        )
        if len(frame.shape) == 3:
            colored_ball_score = float(yellow_fraction[label])
        else:
            colored_ball_score = 0.0
        appearance_score = float(max(whiteness_score, colored_ball_score))
        if diff is not None:
            motion_strength = float(mean_motion[label]) / 255.0
        else:
            motion_strength = 0.7

        # Suppress one-pixel grass/compression sparkle. A close 240 fps ball can
        # be a several-thousand-pixel streak in the face-on camera, so large
        # components are not intrinsically worse.
        size_score = float(np.clip(np.sqrt(area / 80.0), 0.1, 1.0))

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
            0.38 * shape_confidence
            + 0.30 * appearance_score
            + 0.24 * motion_strength
            + 0.08 * size_score
        )
        confidence = max(0.1, min(1.0, confidence))

        candidates.append(
            BallCandidate(
                x_px=float(x),
                y_px=float(y),
                radius_px=float(radius),
                confidence=confidence,
                is_streak=is_streak,
                appearance_score=appearance_score,
            )
        )

    # Sort by confidence highest first
    candidates.sort(key=lambda c: c.confidence, reverse=True)
    selected: list[BallCandidate] = []
    for candidate in candidates:
        if any(
            np.hypot(candidate.x_px - kept.x_px, candidate.y_px - kept.y_px)
            <= max(4.0, 0.35 * (candidate.radius_px + kept.radius_px))
            for kept in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= 40:
            break
    return selected


def build_background_model(pre_impact_frames: list[np.ndarray]) -> np.ndarray:
    """Build a background model from frames captured before impact."""
    if not pre_impact_frames:
        raise ValueError("No frames provided for background modeling.")
    # Median along the frames list axis
    return np.median(pre_impact_frames, axis=0).astype(np.uint8)


from golfie_cv.detection.relevant_window import detect_impact_frame_fast, detect_relevant_window
from golfie_cv.detection.yolo_outlines import render_ball_detection_replay, render_stripped_outlines_video
