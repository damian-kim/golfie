"""Conservative 2D pose checks for the swing-comparison replay.

These checks intentionally report screen-space evidence, not 3D joint angles.
They are coaching prompts that should be verified from the replay or by a coach.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _point(observation: dict, index: int, minimum_confidence: float = 0.35):
    xy = observation.get("xy") or []
    confidence = observation.get("confidence") or []
    if index >= len(xy) or index >= len(confidence) or confidence[index] < minimum_confidence:
        return None
    point = np.asarray(xy[index], dtype=float)
    return point if point.shape == (2,) and np.all(np.isfinite(point)) else None


def _distance(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def _joint_angle(a, b, c) -> float | None:
    if a is None or b is None or c is None:
        return None
    first = np.asarray(a) - np.asarray(b)
    second = np.asarray(c) - np.asarray(b)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator < 1e-6:
        return None
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def analyze_pose_sequence(
    observations: list[dict[str, Any]],
    *,
    camera_role: str,
    impact_frame: int,
    fps: float,
    handedness: str = "right",
) -> dict[str, Any]:
    usable = [item for item in observations if item.get("frame_index") is not None]
    result: dict[str, Any] = {
        "camera": camera_role,
        "observation_count": len(usable),
        "findings": [],
        "limitations": [
            "Screen-space pose estimates are sensitive to camera angle, clothing, occlusion, and mirrored video.",
            "Use these prompts to review the marked frames; they are not a diagnosis or a universal model swing.",
        ],
    }
    if len(usable) < 5 or fps <= 0:
        result["limitations"].append("Too few confident pose frames were available for tailored checks.")
        return result

    address = [item for item in usable if item["frame_index"] <= impact_frame - 0.55 * fps]
    top = [
        item for item in usable
        if impact_frame - 0.75 * fps <= item["frame_index"] <= impact_frame - 0.20 * fps
    ]
    release = [
        item for item in usable
        if impact_frame <= item["frame_index"] <= impact_frame + 0.30 * fps
    ]
    if not address:
        address = usable[: max(2, len(usable) // 4)]

    shoulder_widths = [
        _distance(left, right)
        for item in address
        if (left := _point(item, 5)) is not None and (right := _point(item, 6)) is not None
    ]
    scale = float(np.median(shoulder_widths)) if shoulder_widths else 0.0

    if camera_role == "face_on" and scale > 8.0:
        baseline_heads = [_point(item, 0) for item in address]
        baseline_heads = [point for point in baseline_heads if point is not None]
        swing_heads = [_point(item, 0) for item in usable]
        swing_heads = [point for point in swing_heads if point is not None]
        if baseline_heads and len(swing_heads) >= 4:
            baseline = np.median(np.asarray(baseline_heads), axis=0)
            excursion = max(_distance(point, baseline) for point in swing_heads) / scale
            if excursion > 0.48:
                result["findings"].append({
                    "trait": "Excessive head translation",
                    "severity": "warning" if excursion < 0.75 else "high",
                    "confidence": min(0.9, 0.45 + len(swing_heads) / 30.0),
                    "evidence": f"Head landmark moved about {excursion:.2f} shoulder widths from the address baseline.",
                    "fix": "Use a face-on rehearsal with a visual reference just outside the head. Turn around a stable swing center without trying to freeze the head; stop if the cue restricts a comfortable pivot.",
                    "source_key": "pga_swing_center",
                })

        hip_width_address = [
            _distance(left, right)
            for item in address
            if (left := _point(item, 11)) is not None and (right := _point(item, 12)) is not None
        ]
        hip_width_top = [
            _distance(left, right)
            for item in top
            if (left := _point(item, 11)) is not None and (right := _point(item, 12)) is not None
        ]
        if len(hip_width_address) >= 2 and len(hip_width_top) >= 2:
            address_width = float(np.median(hip_width_address))
            top_width = float(np.percentile(hip_width_top, 30))
            foreshortening = 1.0 - top_width / max(address_width, 1e-6)
            if foreshortening < 0.10:
                result["findings"].append({
                    "trait": "Limited hip-turn indication",
                    "severity": "warning",
                    "confidence": 0.52,
                    "evidence": f"Projected hip width changed only {max(0.0, foreshortening) * 100:.0f}% during the backswing window.",
                    "fix": "Rehearse a comfortable backswing pivot with the trail hip moving behind you while the trail knee stays supported. A club-across-the-hips turn drill can distinguish rotation from a lateral sway.",
                    "source_key": "pga_hip_turn",
                    "caveat": "Hip-width foreshortening is only a 2D proxy; confirm visually before changing your motion.",
                })

        lead_indices = (5, 7, 9) if handedness.lower() != "left" else (6, 8, 10)
        elbow_angles = [
            angle
            for item in release
            if (angle := _joint_angle(
                _point(item, lead_indices[0]),
                _point(item, lead_indices[1]),
                _point(item, lead_indices[2]),
            )) is not None
        ]
        if len(elbow_angles) >= 2:
            median_angle = float(np.median(elbow_angles))
            if median_angle < 148.0:
                result["findings"].append({
                    "trait": "Lead-arm collapse / chicken-wing indication",
                    "severity": "warning" if median_angle >= 125 else "high",
                    "confidence": min(0.88, 0.5 + len(elbow_angles) * 0.06),
                    "evidence": f"Median lead-elbow angle was about {median_angle:.0f}° in the first 0.30 s after impact.",
                    "fix": "Make slow waist-high swings feeling both arms extend past the strike, with the chest continuing to turn. Do not force the elbow straight; pair the width cue with lower-body rotation.",
                    "source_key": "tpi_chicken_wing",
                })

    if not result["findings"]:
        result["summary"] = "No supported warning crossed its conservative threshold in this camera view."
    else:
        result["summary"] = f"{len(result['findings'])} review prompt(s) detected from {len(usable)} pose observations."
    return result


__all__ = ["analyze_pose_sequence"]
