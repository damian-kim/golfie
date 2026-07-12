"""Calibrated stereo triangulation for synchronized 2D ball tracks.

All triangulation is performed from undistorted normalized image rays.  Raw
pixel coordinates are used only for final reprojection diagnostics.  This is
important for phone cameras, where ignoring lens distortion can move a point
many pixels near the edge of the image.
"""

from __future__ import annotations

import cv2
import numpy as np

from golfie_core.coordinates import CoordinateTransformer
from golfie_core.schemas import CalibrationResult, TrackedPoint2D, TrackedPoint3D


def _as_extrinsic(value: list[list[float]]) -> np.ndarray:
    ext = np.asarray(value, dtype=np.float64)
    if ext.shape == (4, 4):
        return ext[:3, :]
    if ext.shape == (3, 4):
        return ext
    raise ValueError(f"Camera extrinsics must be 3x4 or 4x4, got {ext.shape}.")


def _positive_median_step(points: list[TrackedPoint2D]) -> float:
    times = np.asarray(sorted({float(p.time_seconds) for p in points}), dtype=np.float64)
    if len(times) < 2:
        return 1.0 / 240.0
    steps = np.diff(times)
    steps = steps[steps > 1e-7]
    return float(np.median(steps)) if len(steps) else 1.0 / 240.0


def _match_by_time(
    track_a: list[TrackedPoint2D], track_b: list[TrackedPoint2D]
) -> list[tuple[TrackedPoint2D, float, float, float]]:
    """Interpolate B at A timestamps without spanning missing video frames."""
    a = sorted(track_a, key=lambda p: p.time_seconds)
    b = sorted(track_b, key=lambda p: p.time_seconds)
    if not a or not b:
        return []

    step_a = _positive_median_step(a)
    step_b = _positive_median_step(b)
    nearest_limit = 0.75 * max(step_a, step_b)
    interpolation_span_limit = 2.5 * step_b
    times_b = np.asarray([p.time_seconds for p in b], dtype=np.float64)
    overlap_start = max(a[0].time_seconds, b[0].time_seconds)
    overlap_end = min(a[-1].time_seconds, b[-1].time_seconds)
    if overlap_end < overlap_start:
        return []

    matches: list[tuple[TrackedPoint2D, float, float, float]] = []
    for pa in a:
        t = float(pa.time_seconds)
        if t < overlap_start - 1e-9 or t > overlap_end + 1e-9:
            continue
        idx = int(np.searchsorted(times_b, t))

        # Exact/near-exact endpoint matches should not require a bracket.
        nearest = []
        if idx < len(b):
            nearest.append(b[idx])
        if idx > 0:
            nearest.append(b[idx - 1])
        if nearest:
            pb = min(nearest, key=lambda p: abs(p.time_seconds - t))
            if abs(pb.time_seconds - t) <= max(nearest_limit, 1e-5):
                matches.append((pa, pb.x_px, pb.y_px, pb.confidence))
                continue

        if 0 < idx < len(b):
            lo, hi = b[idx - 1], b[idx]
            span = float(hi.time_seconds - lo.time_seconds)
            if 1e-7 < span <= interpolation_span_limit:
                alpha = (t - lo.time_seconds) / span
                if 0.0 <= alpha <= 1.0:
                    matches.append(
                        (
                            pa,
                            float(lo.x_px + alpha * (hi.x_px - lo.x_px)),
                            float(lo.y_px + alpha * (hi.y_px - lo.y_px)),
                            float(lo.confidence + alpha * (hi.confidence - lo.confidence)),
                        )
                    )
    return matches


def _relative_pose(ext_a: np.ndarray, ext_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the transform X_b = R_ba X_a + t_ba."""
    ra, ta = ext_a[:, :3], ext_a[:, 3]
    rb, tb = ext_b[:, :3], ext_b[:, 3]
    r_ba = rb @ ra.T
    t_ba = tb - r_ba @ ta
    return r_ba, t_ba


def _normalized_epipolar_errors(
    points_a: np.ndarray,
    points_b: np.ndarray,
    essential: np.ndarray,
    focal_scale: float,
) -> np.ndarray:
    xa = np.column_stack([points_a, np.ones(len(points_a))])
    xb = np.column_stack([points_b, np.ones(len(points_b))])
    lines_b = (essential @ xa.T).T
    lines_a = (essential.T @ xb.T).T
    numerator = np.abs(np.sum(xb * lines_b, axis=1))
    distance_b = numerator / np.maximum(np.linalg.norm(lines_b[:, :2], axis=1), 1e-12)
    distance_a = numerator / np.maximum(np.linalg.norm(lines_a[:, :2], axis=1), 1e-12)
    return 0.5 * (distance_a + distance_b) * focal_scale


def _project_pixels(
    points_rig: np.ndarray,
    ext: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray | None,
) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(ext[:, :3])
    projected, _ = cv2.projectPoints(
        points_rig.reshape(-1, 1, 3), rvec, ext[:, 3], camera_matrix, distortion
    )
    return projected.reshape(-1, 2)


def triangulate_track(
    track_a: list[TrackedPoint2D],
    track_b: list[TrackedPoint2D],
    calibration: CalibrationResult,
    epipolar_limit_px: float | None = None,
) -> list[TrackedPoint3D]:
    """Triangulate synchronized tracks after timing and epipolar validation."""
    if not calibration.is_valid:
        raise ValueError("Calibration is not valid.")
    if calibration.camera_a_intrinsics is None or calibration.camera_b_intrinsics is None:
        raise ValueError("Missing camera intrinsics in calibration.")
    if calibration.camera_a_extrinsics is None or calibration.camera_b_extrinsics is None:
        raise ValueError("Missing camera extrinsics in calibration.")

    matches = _match_by_time(track_a, track_b)
    if not matches:
        return []

    ka = np.asarray(calibration.camera_a_intrinsics, dtype=np.float64)
    kb = np.asarray(calibration.camera_b_intrinsics, dtype=np.float64)
    da_values = np.asarray(calibration.camera_a_distortion or [], dtype=np.float64)
    db_values = np.asarray(calibration.camera_b_distortion or [], dtype=np.float64)
    da = da_values if da_values.size else None
    db = db_values if db_values.size else None
    ext_a = _as_extrinsic(calibration.camera_a_extrinsics)
    ext_b = _as_extrinsic(calibration.camera_b_extrinsics)

    raw_a = np.asarray([[m[0].x_px, m[0].y_px] for m in matches], dtype=np.float64)
    raw_b = np.asarray([[m[1], m[2]] for m in matches], dtype=np.float64)
    norm_a = cv2.undistortPoints(raw_a.reshape(-1, 1, 2), ka, da).reshape(-1, 2)
    norm_b = cv2.undistortPoints(raw_b.reshape(-1, 1, 2), kb, db).reshape(-1, 2)

    r_ba, t_ba = _relative_pose(ext_a, ext_b)
    tx = np.array(
        [[0.0, -t_ba[2], t_ba[1]], [t_ba[2], 0.0, -t_ba[0]], [-t_ba[1], t_ba[0], 0.0]],
        dtype=np.float64,
    )
    essential = tx @ r_ba
    focal_scale = float(np.mean([ka[0, 0], ka[1, 1], kb[0, 0], kb[1, 1]]))
    epipolar_errors = _normalized_epipolar_errors(norm_a, norm_b, essential, focal_scale)
    calibration_p95 = calibration.epipolar_error_p95_px
    # Compact board corners should remain near the calibration p95, but a
    # 240-fps ball can be a 100px exposure streak. Its centroid can sit 10-15px
    # from the line even when the line crosses the observed streak. Admit that
    # bounded uncertainty, then refine the correspondence onto the epipolar
    # manifold and validate the whole physical trajectory downstream.
    default_epipolar_limit = (
        3.0 if calibration_p95 is None
        else float(np.clip(2.0 * calibration_p95, 2.0, 5.0))
    )
    epipolar_limit = (
        default_epipolar_limit if epipolar_limit_px is None
        else float(max(default_epipolar_limit, epipolar_limit_px))
    )
    geometry_mask = epipolar_errors <= epipolar_limit
    if not np.any(geometry_mask):
        raise ValueError(
            "No time-matched detections satisfy the stereo geometry: "
            f"median epipolar error {float(np.median(epipolar_errors)):.2f}px, "
            f"minimum {float(np.min(epipolar_errors)):.2f}px, limit {epipolar_limit:.2f}px. "
            "Check calibration compatibility, synchronization, and the selected 2D tracks."
        )

    selected_indices = np.flatnonzero(geometry_mask)
    norm_a_selected = norm_a[geometry_mask]
    norm_b_selected = norm_b[geometry_mask]
    corrected_a, corrected_b = cv2.correctMatches(
        essential,
        norm_a_selected.reshape(1, -1, 2),
        norm_b_selected.reshape(1, -1, 2),
    )
    norm_a_selected = corrected_a.reshape(-1, 2)
    norm_b_selected = corrected_b.reshape(-1, 2)
    # Normalized observations pair with extrinsic-only projection matrices.
    homogeneous = cv2.triangulatePoints(
        ext_a, ext_b, norm_a_selected.T, norm_b_selected.T
    )
    safe_w = np.where(np.abs(homogeneous[3]) < 1e-12, np.nan, homogeneous[3])
    points_rig = (homogeneous[:3] / safe_w).T

    projected_a = _project_pixels(points_rig, ext_a, ka, da)
    projected_b = _project_pixels(points_rig, ext_b, kb, db)
    reproj_a = np.linalg.norm(projected_a - raw_a[geometry_mask], axis=1)
    reproj_b = np.linalg.norm(projected_b - raw_b[geometry_mask], axis=1)
    reprojection_errors = 0.5 * (reproj_a + reproj_b)
    reprojection_limit = (
        max(12.0, 0.65 * epipolar_limit)
        if epipolar_limit_px is not None
        else float(np.clip(1.5 * epipolar_limit, 3.0, 6.0))
    )

    ra, ta = ext_a[:, :3], ext_a[:, 3]
    rb, tb = ext_b[:, :3], ext_b[:, 3]
    center_a = -ra.T @ ta
    center_b = -rb.T @ tb
    transformer = (
        CoordinateTransformer.from_coordinate_system(calibration.coordinate_system)
        if calibration.coordinate_system is not None
        else CoordinateTransformer.identity()
    )

    result: list[TrackedPoint3D] = []
    for local_idx, source_idx in enumerate(selected_indices):
        point = points_rig[local_idx]
        if not np.all(np.isfinite(point)):
            continue
        depth_a = float((ext_a[:, :3] @ point + ext_a[:, 3])[2])
        depth_b = float((ext_b[:, :3] @ point + ext_b[:, 3])[2])
        if depth_a <= 0.05 or depth_b <= 0.05:
            continue

        ray_a = point - center_a
        ray_b = point - center_b
        cosine = float(
            np.dot(ray_a, ray_b)
            / max(np.linalg.norm(ray_a) * np.linalg.norm(ray_b), 1e-12)
        )
        parallax_deg = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
        if parallax_deg < 0.25 or reprojection_errors[local_idx] > reprojection_limit:
            continue

        pa, _bx, _by, confidence_b = matches[source_idx]
        point_world = transformer.to_world(point)
        geometry_confidence = float(np.exp(-reprojection_errors[local_idx] / reprojection_limit))
        confidence = float(np.sqrt(pa.confidence * confidence_b) * geometry_confidence)
        result.append(
            TrackedPoint3D(
                time_seconds=float(pa.time_seconds),
                x_m=float(point_world[0]),
                y_m=float(point_world[1]),
                z_m=float(point_world[2]),
                confidence=confidence,
                reprojection_error_px=float(reprojection_errors[local_idx]),
            )
        )

    if not result:
        raise ValueError(
            "Epipolar-consistent pairs were found, but all triangulated points failed "
            f"depth/parallax/reprojection validation (reprojection limit {reprojection_limit:.2f}px)."
        )
    return result
