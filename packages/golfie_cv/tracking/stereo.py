"""Joint stereo-aware ball track selection.

Independent monocular trackers can each choose a plausible moving highlight
that is not the same physical object.  This module constructs candidate pairs
that satisfy the calibrated epipolar geometry, then follows those pairs through
time with a small beam search.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from golfie_core.schemas import CalibrationResult, TrackedPoint2D
from golfie_cv.detection import BallCandidate


@dataclass
class _Pair:
    frame: int
    a: BallCandidate
    b: BallCandidate
    epipolar_error_px: float


@dataclass
class _Hypothesis:
    pairs: list[_Pair]
    score: float


def _path_motion_quality(positions: np.ndarray) -> tuple[float, float, float]:
    """Return median step, path efficiency, and normalized acceleration."""
    vectors = np.diff(positions, axis=0)
    steps = np.linalg.norm(vectors, axis=1)
    if len(steps) == 0:
        return 0.0, 0.0, float("inf")
    median_step = float(np.median(steps))
    path_length = float(np.sum(steps))
    displacement = float(np.linalg.norm(positions[-1] - positions[0]))
    efficiency = displacement / max(path_length, 1e-6)
    if len(vectors) < 2:
        acceleration_ratio = 0.0
    else:
        accelerations = np.linalg.norm(np.diff(vectors, axis=0), axis=1)
        acceleration_ratio = float(np.median(accelerations)) / max(median_step, 1e-6)
    return median_step, efficiency, acceleration_ratio


def _extrinsic(value: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    return matrix[:3, :] if matrix.shape == (4, 4) else matrix


def _essential(calibration: CalibrationResult) -> np.ndarray:
    a = _extrinsic(calibration.camera_a_extrinsics)  # type: ignore[arg-type]
    b = _extrinsic(calibration.camera_b_extrinsics)  # type: ignore[arg-type]
    rotation = b[:, :3] @ a[:, :3].T
    translation = b[:, 3] - rotation @ a[:, 3]
    cross = np.array(
        [
            [0.0, -translation[2], translation[1]],
            [translation[2], 0.0, -translation[0]],
            [-translation[1], translation[0], 0.0],
        ]
    )
    return cross @ rotation


def _frame_pairs(
    frame: int,
    candidates_a: list[BallCandidate],
    candidates_b: list[BallCandidate],
    calibration: CalibrationResult,
    essential: np.ndarray,
    limit_px: float,
) -> list[_Pair]:
    # Confidence sorting limits a noisy real frame from creating an unbounded
    # Cartesian product while retaining streaks and small bright candidates.
    ca = sorted(candidates_a, key=lambda c: c.confidence, reverse=True)[:40]
    cb = sorted(candidates_b, key=lambda c: c.confidence, reverse=True)[:40]
    if not ca or not cb:
        return []

    ka = np.asarray(calibration.camera_a_intrinsics, dtype=np.float64)
    kb = np.asarray(calibration.camera_b_intrinsics, dtype=np.float64)
    da_values = np.asarray(calibration.camera_a_distortion or [], dtype=np.float64)
    db_values = np.asarray(calibration.camera_b_distortion or [], dtype=np.float64)
    da = da_values if da_values.size else None
    db = db_values if db_values.size else None
    raw_a = np.asarray([[c.x_px, c.y_px] for c in ca], dtype=np.float64)
    raw_b = np.asarray([[c.x_px, c.y_px] for c in cb], dtype=np.float64)
    na = cv2.undistortPoints(raw_a.reshape(-1, 1, 2), ka, da).reshape(-1, 2)
    nb = cv2.undistortPoints(raw_b.reshape(-1, 1, 2), kb, db).reshape(-1, 2)
    focal = float(np.mean([ka[0, 0], ka[1, 1], kb[0, 0], kb[1, 1]]))

    pairs: list[_Pair] = []
    for ia, point_a in enumerate(na):
        xa = np.array([point_a[0], point_a[1], 1.0])
        line_b = essential @ xa
        denom_b = max(float(np.linalg.norm(line_b[:2])), 1e-12)
        for ib, point_b in enumerate(nb):
            xb = np.array([point_b[0], point_b[1], 1.0])
            line_a = essential.T @ xb
            numerator = abs(float(xb @ line_b))
            error = 0.5 * (
                numerator / denom_b
                + numerator / max(float(np.linalg.norm(line_a[:2])), 1e-12)
            ) * focal
            pair_limit = max(limit_px, 16.0) if (ca[ia].is_streak or cb[ib].is_streak) else limit_px
            if error <= pair_limit:
                pairs.append(_Pair(frame, ca[ia], cb[ib], error))

    def pair_rank(pair: _Pair) -> float:
        confidence = np.sqrt(pair.a.confidence * pair.b.confidence)
        appearance = np.sqrt(pair.a.appearance_score * pair.b.appearance_score)
        streak_bonus = 0.15 * (int(pair.a.is_streak) + int(pair.b.is_streak))
        effective_limit = max(limit_px, 16.0) if (pair.a.is_streak or pair.b.is_streak) else limit_px
        return float(2.0 * confidence + 0.75 * appearance + streak_bonus - pair.epipolar_error_px / effective_limit)

    return sorted(pairs, key=pair_rank, reverse=True)[:40]


def _predict(pairs: list[_Pair], camera: str, frame: int) -> tuple[float, float]:
    latest = getattr(pairs[-1], camera)
    if len(pairs) < 2:
        return latest.x_px, latest.y_px
    previous = getattr(pairs[-2], camera)
    frame_delta = max(1, pairs[-1].frame - pairs[-2].frame)
    ahead = frame - pairs[-1].frame
    return (
        latest.x_px + (latest.x_px - previous.x_px) * ahead / frame_delta,
        latest.y_px + (latest.y_px - previous.y_px) * ahead / frame_delta,
    )


def _candidate_epipolar_error(
    candidate_a: BallCandidate,
    candidate_b: BallCandidate,
    calibration: CalibrationResult,
    essential: np.ndarray,
) -> float:
    ka = np.asarray(calibration.camera_a_intrinsics, dtype=np.float64)
    kb = np.asarray(calibration.camera_b_intrinsics, dtype=np.float64)
    da_values = np.asarray(calibration.camera_a_distortion or [], dtype=np.float64)
    db_values = np.asarray(calibration.camera_b_distortion or [], dtype=np.float64)
    da = da_values if da_values.size else None
    db = db_values if db_values.size else None
    point_a = cv2.undistortPoints(
        np.asarray([candidate_a.x_px, candidate_a.y_px], dtype=np.float64).reshape(1, 1, 2), ka, da
    ).reshape(2)
    point_b = cv2.undistortPoints(
        np.asarray([candidate_b.x_px, candidate_b.y_px], dtype=np.float64).reshape(1, 1, 2), kb, db
    ).reshape(2)
    xa = np.r_[point_a, 1.0]
    xb = np.r_[point_b, 1.0]
    line_b = essential @ xa
    line_a = essential.T @ xb
    numerator = abs(float(xb @ line_b))
    focal = float(np.mean([ka[0, 0], ka[1, 1], kb[0, 0], kb[1, 1]]))
    return float(
        0.5 * (
            numerator / max(float(np.linalg.norm(line_b[:2])), 1e-12)
            + numerator / max(float(np.linalg.norm(line_a[:2])), 1e-12)
        ) * focal
    )


def _optic_motion_hypotheses(
    candidates: list[list[BallCandidate]],
) -> list[tuple[float, list[tuple[int, BallCandidate]]]]:
    dominant: list[BallCandidate | None] = []
    for frame_candidates in candidates:
        optic = [candidate for candidate in frame_candidates if candidate.is_optic_color]
        dominant.append(
            max(
                optic,
                key=lambda candidate: (
                    candidate.bbox_width_px * candidate.bbox_height_px,
                    candidate.radius_px,
                ),
                default=None,
            )
        )

    hypotheses: list[tuple[float, list[tuple[int, BallCandidate]]]] = []
    for start in range(len(dominant)):
        for length in range(4, min(11, len(dominant) - start + 1)):
            segment = dominant[start:start + length]
            if any(candidate is None for candidate in segment):
                continue
            typed = [candidate for candidate in segment if candidate is not None]
            positions = np.asarray([(candidate.x_px, candidate.y_px) for candidate in typed])
            vectors = np.diff(positions, axis=0)
            steps = np.linalg.norm(vectors, axis=1)
            if len(steps) < 3 or float(np.median(steps)) < 4.0:
                continue
            path_length = float(np.sum(steps))
            displacement = float(np.linalg.norm(positions[-1] - positions[0]))
            efficiency = displacement / max(path_length, 1e-6)
            accelerations = np.linalg.norm(np.diff(vectors, axis=0), axis=1)
            acceleration_ratio = float(np.median(accelerations)) / max(float(np.median(steps)), 1e-6)
            if efficiency < 0.55 or acceleration_ratio > 1.25:
                continue
            mean_area = float(np.mean([
                min(candidate.bbox_width_px * candidate.bbox_height_px, 5000.0) / 5000.0
                for candidate in typed
            ]))
            score = (
                displacement + 15.0 * length + 100.0 * efficiency
                - 45.0 * acceleration_ratio + 30.0 * mean_area
            )
            hypotheses.append((score, list(zip(range(start, start + length), typed))))
    return sorted(hypotheses, key=lambda item: item[0], reverse=True)[:20]


def track_optic_ball_stereo(
    candidates_a: list[list[BallCandidate]],
    candidates_b: list[list[BallCandidate]],
    fps: float,
    calibration: CalibrationResult,
) -> tuple[list[TrackedPoint2D], list[TrackedPoint2D]]:
    """Fast physics-scored path for a high-saturation practice ball."""
    from golfie_cv.triangulation import triangulate_track

    hypotheses_a = _optic_motion_hypotheses(candidates_a)
    hypotheses_b = _optic_motion_hypotheses(candidates_b)
    if not hypotheses_a or not hypotheses_b or not calibration.is_valid:
        return [], []
    essential = _essential(calibration)
    best: tuple[float, list[TrackedPoint2D], list[TrackedPoint2D]] | None = None

    for score_a, path_a in hypotheses_a:
        for score_b, path_b in hypotheses_b:
            for trim_a in range(max(1, len(path_a) - 3)):
                for trim_b in range(max(1, len(path_b) - 3)):
                    length = min(len(path_a) - trim_a, len(path_b) - trim_b, 8)
                    if length < 4:
                        continue
                    paired = list(zip(path_a[trim_a:trim_a + length], path_b[trim_b:trim_b + length]))
                    errors = np.asarray([
                        _candidate_epipolar_error(a_item[1], b_item[1], calibration, essential)
                        for a_item, b_item in paired
                    ])
                    # A driver can turn the face-on ball into a 200px exposure
                    # streak. Its centroid is not the instantaneous ball centre,
                    # but the calibrated epipolar line still crosses the streak.
                    # Admit that bounded along-streak uncertainty; compact balls
                    # retain the tighter iron-shot gate.
                    pair_limits = np.asarray([
                        40.0 if (a_item[1].is_streak or b_item[1].is_streak) else 18.0
                        for a_item, b_item in paired
                    ])
                    geometry = errors <= pair_limits
                    if int(np.sum(geometry)) < 4:
                        continue
                    selected = [pair for pair, keep in zip(paired, geometry) if keep]
                    track_a = [
                        TrackedPoint2D(
                            frame_index=a_item[0], time_seconds=index / fps,
                            x_px=a_item[1].x_px, y_px=a_item[1].y_px,
                            confidence=a_item[1].confidence,
                            radius_px=a_item[1].radius_px,
                        )
                        for index, (a_item, _b_item) in enumerate(selected)
                    ]
                    track_b = [
                        TrackedPoint2D(
                            frame_index=b_item[0], time_seconds=index / fps,
                            x_px=b_item[1].x_px, y_px=b_item[1].y_px,
                            confidence=b_item[1].confidence,
                            radius_px=b_item[1].radius_px,
                        )
                        for index, (_a_item, b_item) in enumerate(selected)
                    ]
                    try:
                        points = triangulate_track(
                            track_a, track_b, calibration, epipolar_limit_px=40.0
                        )
                    except ValueError:
                        continue
                    if len(points) < 4:
                        continue
                    times = np.asarray([point.time_seconds for point in points])
                    positions = np.asarray([(point.x_m, point.y_m, point.z_m) for point in points])
                    design = np.column_stack([np.ones(len(times)), times - times[0]])
                    coefficients, *_ = np.linalg.lstsq(design, positions, rcond=None)
                    velocity = coefficients[1]
                    speed = float(np.linalg.norm(velocity))
                    residual = float(np.sqrt(np.mean((design @ coefficients - positions) ** 2)))
                    if not (10.0 <= speed <= 80.0) or velocity[2] <= 0.0 or residual > 0.08:
                        continue
                    score = (
                        score_a + score_b + 40.0 * len(points)
                        - 5.0 * float(np.median(errors[geometry])) - 1000.0 * residual
                    )
                    if best is None or score > best[0]:
                        best = (score, track_a, track_b)
    return (best[1], best[2]) if best is not None else ([], [])


def _track_ball_stereo_aligned(
    candidates_a: list[list[BallCandidate]],
    candidates_b: list[list[BallCandidate]],
    fps: float,
    calibration: CalibrationResult,
    epipolar_limit_px: float | None = None,
    require_physical_launch: bool = False,
) -> tuple[list[TrackedPoint2D], list[TrackedPoint2D]]:
    """Return a jointly selected pair of local-timeline 2D tracks."""
    if not calibration.is_valid:
        return [], []
    if any(
        value is None
        for value in (
            calibration.camera_a_intrinsics,
            calibration.camera_b_intrinsics,
            calibration.camera_a_extrinsics,
            calibration.camera_b_extrinsics,
        )
    ):
        return [], []

    essential = _essential(calibration)
    calibration_p95 = calibration.epipolar_error_p95_px
    default_limit_px = (
        3.0 if calibration_p95 is None
        else float(np.clip(2.0 * calibration_p95, 2.0, 5.0))
    )
    limit_px = (
        default_limit_px if epipolar_limit_px is None
        else float(max(default_limit_px, epipolar_limit_px))
    )
    hypotheses: list[_Hypothesis] = []
    completed: list[_Hypothesis] = []

    for frame in range(min(len(candidates_a), len(candidates_b))):
        pairs = _frame_pairs(
            frame, candidates_a[frame], candidates_b[frame], calibration, essential, limit_px
        )
        next_hypotheses: list[_Hypothesis] = []

        # Keep hypotheses alive across at most three missed detections.
        for hypothesis in hypotheses:
            if frame - hypothesis.pairs[-1].frame <= 3:
                next_hypotheses.append(_Hypothesis(hypothesis.pairs, hypothesis.score - 0.35))
            else:
                completed.append(hypothesis)

        for pair in pairs:
            confidence = float(np.sqrt(pair.a.confidence * pair.b.confidence))
            appearance = float(np.sqrt(pair.a.appearance_score * pair.b.appearance_score))
            effective_limit = max(limit_px, 16.0) if (pair.a.is_streak or pair.b.is_streak) else limit_px
            base = 2.0 * confidence + 0.75 * appearance - pair.epipolar_error_px / effective_limit
            next_hypotheses.append(_Hypothesis([pair], base))
            for hypothesis in hypotheses:
                gap = pair.frame - hypothesis.pairs[-1].frame
                if gap <= 0 or gap > 3:
                    continue
                pred_a = _predict(hypothesis.pairs, "a", pair.frame)
                pred_b = _predict(hypothesis.pairs, "b", pair.frame)
                error_a = float(np.hypot(pair.a.x_px - pred_a[0], pair.a.y_px - pred_a[1]))
                error_b = float(np.hypot(pair.b.x_px - pred_b[0], pair.b.y_px - pred_b[1]))
                gate = 350.0 if len(hypothesis.pairs) < 2 else 140.0 * gap
                if error_a <= gate and error_b <= gate:
                    motion_cost = (error_a + error_b) / max(gate, 1.0)
                    next_hypotheses.append(
                        _Hypothesis(hypothesis.pairs + [pair], hypothesis.score + base - motion_cost)
                    )

        def beam_rank(hypothesis: _Hypothesis) -> float:
            if len(hypothesis.pairs) < 2:
                return hypothesis.score
            first, last = hypothesis.pairs[0], hypothesis.pairs[-1]
            displacement = np.hypot(last.a.x_px - first.a.x_px, last.a.y_px - first.a.y_px)
            displacement += np.hypot(last.b.x_px - first.b.x_px, last.b.y_px - first.b.y_px)
            return float(hypothesis.score + min(displacement, 500.0) / 80.0)

        hypotheses = sorted(next_hypotheses, key=beam_rank, reverse=True)[:80]

    completed.extend(hypotheses)
    valid = []
    for hypothesis in completed:
        if len(hypothesis.pairs) < 4:
            continue
        first, last = hypothesis.pairs[0], hypothesis.pairs[-1]
        displacement_a = float(np.hypot(last.a.x_px - first.a.x_px, last.a.y_px - first.a.y_px))
        displacement_b = float(np.hypot(last.b.x_px - first.b.x_px, last.b.y_px - first.b.y_px))
        if displacement_a < 8.0 or displacement_b < 8.0:
            continue
        positions_a = np.asarray([(pair.a.x_px, pair.a.y_px) for pair in hypothesis.pairs])
        positions_b = np.asarray([(pair.b.x_px, pair.b.y_px) for pair in hypothesis.pairs])
        steps_a = np.linalg.norm(np.diff(positions_a, axis=0), axis=1)
        steps_b = np.linalg.norm(np.diff(positions_b, axis=0), axis=1)
        joint_stationary = float(np.mean((steps_a < 2.0) | (steps_b < 2.0)))
        if joint_stationary > 0.55:
            continue
        median_a, efficiency_a, acceleration_a = _path_motion_quality(positions_a)
        median_b, efficiency_b, acceleration_b = _path_motion_quality(positions_b)
        # A ball flight is nearly monotonic during this short post-impact
        # window. Club edges and codec blocks often win on raw displacement by
        # jumping between unrelated highlights, which produces a winding path
        # and abrupt velocity changes in one or both cameras.
        if min(efficiency_a, efficiency_b) < 0.5:
            continue
        if max(acceleration_a, acceleration_b) > 1.5:
            continue
        median_motion = median_a + median_b
        smoothness = efficiency_a + efficiency_b - 0.5 * (acceleration_a + acceleration_b)
        launch_frame = first.frame
        timing_score = 1.0 if 6 <= launch_frame <= 38 else 0.25
        mean_epi = float(np.mean([p.epipolar_error_px for p in hypothesis.pairs]))
        final_score = (
            hypothesis.score
            + timing_score * (displacement_a + displacement_b) / 15.0
            + 1.5 * median_motion
            + 20.0 * smoothness
            - 30.0 * joint_stationary
            - mean_epi
        )
        valid.append((final_score, hypothesis))

    if not valid:
        return [], []

    if not require_physical_launch:
        best = max(valid, key=lambda item: item[0])[1]
        return (
            [
                TrackedPoint2D(
                    frame_index=pair.frame,
                    time_seconds=pair.frame / fps,
                    x_px=pair.a.x_px,
                    y_px=pair.a.y_px,
                    confidence=pair.a.confidence,
                    radius_px=pair.a.radius_px,
                )
                for pair in best.pairs
            ],
            [
                TrackedPoint2D(
                    frame_index=pair.frame,
                    time_seconds=pair.frame / fps,
                    x_px=pair.b.x_px,
                    y_px=pair.b.y_px,
                    confidence=pair.b.confidence,
                    radius_px=pair.b.radius_px,
                )
                for pair in best.pairs
            ],
        )

    # Image-space smoothness is necessary but not sufficient: a club edge can
    # move smoothly and satisfy an epipolar line. Rank the strongest surviving
    # hypotheses by their reconstructed launch, so only a forward/upward path
    # with golf-ball-scale speed can be returned to the main physics fitter.
    from golfie_cv.triangulation import triangulate_track

    physical: list[tuple[float, list[TrackedPoint2D], list[TrackedPoint2D]]] = []
    for image_score, hypothesis in sorted(valid, key=lambda item: item[0], reverse=True)[:80]:
        track_a = [
            TrackedPoint2D(
                frame_index=pair.frame,
                time_seconds=pair.frame / fps,
                x_px=pair.a.x_px,
                y_px=pair.a.y_px,
                confidence=pair.a.confidence,
                radius_px=pair.a.radius_px,
            )
            for pair in hypothesis.pairs
        ]
        track_b = [
            TrackedPoint2D(
                frame_index=pair.frame,
                time_seconds=pair.frame / fps,
                x_px=pair.b.x_px,
                y_px=pair.b.y_px,
                confidence=pair.b.confidence,
                radius_px=pair.b.radius_px,
            )
            for pair in hypothesis.pairs
        ]
        try:
            points = triangulate_track(
                track_a, track_b, calibration, epipolar_limit_px=epipolar_limit_px
            )
        except ValueError:
            continue
        # Four points can be manufactured by a short club/grass coincidence
        # over only 25 ms. Require enough consecutive stereo evidence to
        # observe an actual launch segment before accepting the hypothesis.
        if len(points) < 6:
            continue
        times = np.asarray([point.time_seconds for point in points], dtype=np.float64)
        positions = np.asarray(
            [(point.x_m, point.y_m, point.z_m) for point in points], dtype=np.float64
        )
        design = np.column_stack([np.ones(len(times)), times - times[0]])
        coefficients, *_ = np.linalg.lstsq(design, positions, rcond=None)
        velocity = coefficients[1]
        speed = float(np.linalg.norm(velocity))
        residual = float(np.sqrt(np.mean((design @ coefficients - positions) ** 2)))
        if not (10.0 <= speed <= 95.0):
            continue
        if velocity[0] <= 1.0 or velocity[2] <= 0.0 or residual > 0.04:
            continue
        reprojection = np.asarray(
            [point.reprojection_error_px or 0.0 for point in points], dtype=np.float64
        )
        physics_score = (
            image_score
            + 30.0 * len(points)
            + 0.5 * speed
            - 1000.0 * residual
            - 3.0 * float(np.median(reprojection))
        )
        physical.append((physics_score, track_a, track_b))

    if not physical:
        return [], []
    best = max(physical, key=lambda item: item[0])
    return best[1], best[2]


def track_ball_stereo(
    candidates_a: list[list[BallCandidate]],
    candidates_b: list[list[BallCandidate]],
    fps: float,
    calibration: CalibrationResult,
    max_temporal_shift_frames: int = 10,
    epipolar_limit_px: float | None = None,
    require_physical_launch: bool = False,
) -> tuple[list[TrackedPoint2D], list[TrackedPoint2D]]:
    """Select the stereo track while correcting small impact/speed-ramp sync errors.

    Audio and visual impact detectors locate each phone independently. Native
    slow-motion edit ramps can move that estimate by several captured frames,
    so test nearby integer offsets and let calibrated epipolar continuity pick
    the alignment instead of averaging container FPS values.
    """
    best: tuple[float, list[TrackedPoint2D], list[TrackedPoint2D]] | None = None
    maximum = max(0, int(max_temporal_shift_frames))
    for shift in range(-maximum, maximum + 1):
        start_a = max(0, shift)
        start_b = max(0, -shift)
        available = min(len(candidates_a) - start_a, len(candidates_b) - start_b)
        if available < 4:
            continue
        track_a, track_b = _track_ball_stereo_aligned(
            candidates_a[start_a:start_a + available],
            candidates_b[start_b:start_b + available],
            fps,
            calibration,
            epipolar_limit_px=epipolar_limit_px,
            require_physical_launch=require_physical_launch,
        )
        if not track_a or not track_b:
            continue
        track_a = [
            TrackedPoint2D(
                frame_index=point.frame_index + start_a,
                time_seconds=(point.frame_index + start_a) / fps,
                x_px=point.x_px,
                y_px=point.y_px,
                confidence=point.confidence,
                radius_px=point.radius_px,
            )
            for point in track_a
        ]
        track_b = [
            TrackedPoint2D(
                frame_index=point.frame_index + start_b,
                time_seconds=(point.frame_index + start_b) / fps,
                x_px=point.x_px,
                y_px=point.y_px,
                confidence=point.confidence,
                radius_px=point.radius_px,
            )
            for point in track_b
        ]
        confidence = float(np.mean([p.confidence for p in track_a + track_b]))
        displacement = float(
            np.hypot(track_a[-1].x_px - track_a[0].x_px, track_a[-1].y_px - track_a[0].y_px)
            + np.hypot(track_b[-1].x_px - track_b[0].x_px, track_b[-1].y_px - track_b[0].y_px)
        )
        positions_a = np.asarray([(point.x_px, point.y_px) for point in track_a])
        positions_b = np.asarray([(point.x_px, point.y_px) for point in track_b])
        steps_a = np.linalg.norm(np.diff(positions_a, axis=0), axis=1)
        steps_b = np.linalg.norm(np.diff(positions_b, axis=0), axis=1)
        joint_stationary = float(np.mean((steps_a < 2.0) | (steps_b < 2.0)))
        median_a, efficiency_a, acceleration_a = _path_motion_quality(positions_a)
        median_b, efficiency_b, acceleration_b = _path_motion_quality(positions_b)
        median_motion = median_a + median_b
        smoothness = efficiency_a + efficiency_b - 0.5 * (acceleration_a + acceleration_b)
        score = (
            4.0 * len(track_a)
            + confidence
            + min(displacement, 800.0) / 20.0
            + 2.0 * median_motion
            + 20.0 * smoothness
            - 40.0 * joint_stationary
        )
        if best is None or score > best[0]:
            best = (score, track_a, track_b)
    return (best[1], best[2]) if best is not None else ([], [])
