"""2D-to-3D triangulation of paired, synced ball detections.

STATUS: stub. Implemented in Milestone 5 (spec section 20 / spec section 11).
"""

from __future__ import annotations

from golfie_core.schemas import CalibrationResult, TrackedPoint2D, TrackedPoint3D


def triangulate_track(
    track_a: list[TrackedPoint2D],
    track_b: list[TrackedPoint2D],
    calibration: CalibrationResult,
) -> list[TrackedPoint3D]:
    """Triangulate paired 2D tracks into a 3D point sequence.

    Spec section 11. Expected implementation: cv2.triangulatePoints (or
    a direct linear triangulation + nonlinear refinement) per matched
    frame pair, followed by reprojection-error filtering, physically
    plausible speed/acceleration filtering, and temporal smoothing.
    Optionally a full bundle adjustment over the whole trajectory.
    Requires `calibration.is_valid` -- callers should check that before
    calling this so failures are diagnosable (missing/garbage
    calibration vs. a triangulation bug).
    """
    import cv2
    import numpy as np
    from golfie_core.coordinates import CoordinateTransformer
    from datetime import datetime

    def log_tri(msg):
        print(msg)
        try:
            with open(r"E:\Golfie\golfie\progress.log", "a", encoding="utf-8") as f_log:
                f_log.write(f"[{datetime.now().isoformat()}] [Triangulation] {msg}\n")
        except Exception:
            pass

    # 1. Validation
    if not calibration.is_valid:
        raise ValueError("Calibration is not valid.")
    if calibration.camera_a_intrinsics is None or calibration.camera_b_intrinsics is None:
        raise ValueError("Missing camera intrinsics in calibration.")
    if calibration.camera_a_extrinsics is None or calibration.camera_b_extrinsics is None:
        raise ValueError("Missing camera extrinsics in calibration.")

    # 2. Extract matrices
    K_a = np.array(calibration.camera_a_intrinsics, dtype=np.float64)
    K_b = np.array(calibration.camera_b_intrinsics, dtype=np.float64)
    ext_a = np.array(calibration.camera_a_extrinsics, dtype=np.float64)
    ext_b = np.array(calibration.camera_b_extrinsics, dtype=np.float64)

    # Reconstruct projection matrices: P = K * [R | T]
    # Calibration extrinsics are 4x4 world-to-camera matrices.
    P_a = K_a @ ext_a[:3, :]
    P_b = K_b @ ext_b[:3, :]

    # 3. Time-based matching with linear interpolation
    #
    # The old approach matched by exact frame_index equality, which almost
    # never works when the two cameras have different recording start times
    # and the sync offset is a large non-integer number of frames.
    #
    # New approach: for each point in track_a, find the closest point(s) in
    # track_b by time_seconds and interpolate track_b's 2D position to
    # track_a's exact timestamp.

    if not track_a or not track_b:
        log_tri("\n--- TRIANGULATION DEBUG ---")
        log_tri(f"  Empty track(s): track_a={len(track_a)}, track_b={len(track_b)}")
        return []

    # Sort tracks by time for binary search
    sorted_a = sorted(track_a, key=lambda p: p.time_seconds)
    sorted_b = sorted(track_b, key=lambda p: p.time_seconds)

    times_b = np.array([p.time_seconds for p in sorted_b])

    # Maximum time gap to allow for matching.
    # Camera A runs at 30fps (33ms between frames), so we need at least
    # one full frame interval to find matching pairs across cameras.
    MAX_TIME_GAP = 1.0 / 30.0  # ~33ms (one frame at 30fps)

    matched_pairs: list[tuple[TrackedPoint2D, float, float, float]] = []
    # Each entry: (pt_a, interp_x_b, interp_y_b, interp_conf_b)

    log_tri("\n--- TRIANGULATION DEBUG ---")
    log_tri(f"  Track A: {len(sorted_a)} pts, time range [{sorted_a[0].time_seconds:.4f}s, {sorted_a[-1].time_seconds:.4f}s]")
    log_tri(f"  Track B: {len(sorted_b)} pts, time range [{sorted_b[0].time_seconds:.4f}s, {sorted_b[-1].time_seconds:.4f}s]")

    # Compute the overlapping time window
    t_start = max(sorted_a[0].time_seconds, sorted_b[0].time_seconds)
    t_end = min(sorted_a[-1].time_seconds, sorted_b[-1].time_seconds)
    overlap = t_end - t_start
    log_tri(f"  Time overlap: {overlap:.4f}s ({t_start:.4f}s to {t_end:.4f}s)")

    if overlap <= 0:
        log_tri("  ERROR: No temporal overlap between tracks! Sync offset may be wrong.")
        return []

    for pt_a in sorted_a:
        t_a = pt_a.time_seconds

        # Skip points outside the overlap region
        if t_a < t_start - MAX_TIME_GAP or t_a > t_end + MAX_TIME_GAP:
            continue

        # Find the insertion point in sorted_b times
        idx = np.searchsorted(times_b, t_a)

        # Consider the two nearest neighbours: idx-1 and idx
        best_interp = None
        best_dt = float("inf")

        # Case 1: Exact or near-exact match at idx
        if idx < len(sorted_b):
            dt = abs(sorted_b[idx].time_seconds - t_a)
            if dt < best_dt:
                best_dt = dt
                best_interp = (sorted_b[idx].x_px, sorted_b[idx].y_px, sorted_b[idx].confidence)

        # Case 2: Exact or near-exact match at idx-1
        if idx > 0:
            dt = abs(sorted_b[idx - 1].time_seconds - t_a)
            if dt < best_dt:
                best_dt = dt
                best_interp = (sorted_b[idx - 1].x_px, sorted_b[idx - 1].y_px, sorted_b[idx - 1].confidence)

        # Case 3: Linear interpolation between idx-1 and idx
        if 0 < idx < len(sorted_b):
            pb_lo = sorted_b[idx - 1]
            pb_hi = sorted_b[idx]
            dt_span = pb_hi.time_seconds - pb_lo.time_seconds
            if 0 < dt_span <= 0.1:  # Sanity: don't interpolate over gaps > 100ms
                alpha = (t_a - pb_lo.time_seconds) / dt_span
                if 0.0 <= alpha <= 1.0:
                    interp_x = pb_lo.x_px + alpha * (pb_hi.x_px - pb_lo.x_px)
                    interp_y = pb_lo.y_px + alpha * (pb_hi.y_px - pb_lo.y_px)
                    interp_conf = pb_lo.confidence + alpha * (pb_hi.confidence - pb_lo.confidence)
                    # Interpolation is always better than nearest-neighbor
                    # if the point falls between two track_b samples
                    best_interp = (interp_x, interp_y, interp_conf)
                    best_dt = 0.0  # Interpolated = effectively zero temporal error

        if best_interp is not None and best_dt <= MAX_TIME_GAP:
            matched_pairs.append((pt_a, best_interp[0], best_interp[1], best_interp[2]))

    log_tri(f"  Time-matched pairs: {len(matched_pairs)}")

    if not matched_pairs:
        log_tri("  ERROR: No time-matched pairs found!")
        # Print sample timestamps for debugging
        for i, pt in enumerate(sorted_a[:5]):
            log_tri(f"    Track A sample [{i}]: frame={pt.frame_index}, t={pt.time_seconds:.4f}s")
        for i, pt in enumerate(sorted_b[:5]):
            log_tri(f"    Track B sample [{i}]: frame={pt.frame_index}, t={pt.time_seconds:.4f}s")
        return []

    # Print first few matched pairs for debugging
    for i, (pt_a, bx, by, bc) in enumerate(matched_pairs[:5]):
        log_tri(f"    Pair [{i}]: t={pt_a.time_seconds:.4f}s, A=({pt_a.x_px:.1f}, {pt_a.y_px:.1f}), B=({bx:.1f}, {by:.1f})")

    # 4. Form coordinate matrices from matched pairs
    n_pairs = len(matched_pairs)
    pts_a_np = np.zeros((2, n_pairs), dtype=np.float64)
    pts_b_np = np.zeros((2, n_pairs), dtype=np.float64)

    for idx, (pt_a, bx, by, _bc) in enumerate(matched_pairs):
        pts_a_np[0, idx] = pt_a.x_px
        pts_a_np[1, idx] = pt_a.y_px
        pts_b_np[0, idx] = bx
        pts_b_np[1, idx] = by

    # 5. Triangulate points
    points_homogeneous = cv2.triangulatePoints(P_a, P_b, pts_a_np, pts_b_np)
    
    # Avoid division by zero
    w = points_homogeneous[3, :]
    w_safe = np.where(np.abs(w) < 1e-9, 1e-9, w)
    points_3d_rig = points_homogeneous[:3, :] / w_safe

    # 6. Calculate reprojection errors
    pts_3d_hom = np.vstack([points_3d_rig, np.ones(points_3d_rig.shape[1])])
    
    proj_a_hom = P_a @ pts_3d_hom
    proj_b_hom = P_b @ pts_3d_hom

    u_a_proj = proj_a_hom[0, :] / np.where(np.abs(proj_a_hom[2, :]) < 1e-9, 1e-9, proj_a_hom[2, :])
    v_a_proj = proj_a_hom[1, :] / np.where(np.abs(proj_a_hom[2, :]) < 1e-9, 1e-9, proj_a_hom[2, :])

    u_b_proj = proj_b_hom[0, :] / np.where(np.abs(proj_b_hom[2, :]) < 1e-9, 1e-9, proj_b_hom[2, :])
    v_b_proj = proj_b_hom[1, :] / np.where(np.abs(proj_b_hom[2, :]) < 1e-9, 1e-9, proj_b_hom[2, :])

    errors_a = np.sqrt((u_a_proj - pts_a_np[0, :]) ** 2 + (v_a_proj - pts_a_np[1, :]) ** 2)
    errors_b = np.sqrt((u_b_proj - pts_b_np[0, :]) ** 2 + (v_b_proj - pts_b_np[1, :]) ** 2)
    reproj_errors = 0.5 * (errors_a + errors_b)

    log_tri(f"  Reprojection errors: min={float(np.min(reproj_errors)):.2f}, median={float(np.median(reproj_errors)):.2f}, max={float(np.max(reproj_errors)):.2f} px")

    # 7. Transform to world frame
    if calibration.coordinate_system is not None:
        transformer = CoordinateTransformer.from_coordinate_system(calibration.coordinate_system)
    else:
        transformer = CoordinateTransformer.identity()

    points_3d_world = []
    for i in range(points_3d_rig.shape[1]):
        p_rig = points_3d_rig[:, i]
        p_world = transformer.to_world(p_rig)
        points_3d_world.append(p_world)

    # 8. Filter by reprojection error and physical constraints
    REPROJ_ERROR_THRESHOLD = 50.0  # px (relaxed to tolerate motion-blurred streak detections)
    candidates_3d = []
    rejected_reproj = 0
    rejected_depth = 0

    for idx, (pt_a, _bx, _by, bc) in enumerate(matched_pairs):
        p_world = points_3d_world[idx]
        p_rig = points_3d_rig[:, idx]
        err = float(reproj_errors[idx])
        
        # Average time and geometric mean of confidence
        t_sec = pt_a.time_seconds
        conf = float(np.sqrt(pt_a.confidence * bc))

        # Check reprojection error threshold
        if err > REPROJ_ERROR_THRESHOLD:
            rejected_reproj += 1
            continue

        # Check positive depth check (must be in front of both cameras)
        # For Camera A: depth is p_rig[2]
        # For Camera B: depth is the 3rd coordinate of ext_b * p_rig
        p_rig_hom = np.append(p_rig, 1.0)
        p_cam_b = ext_b[:3, :] @ p_rig_hom
        
        if p_rig[2] <= 0.1 or p_cam_b[2] <= 0.1:
            rejected_depth += 1
            continue

        candidates_3d.append(
            TrackedPoint3D(
                time_seconds=t_sec,
                x_m=float(p_world[0]),
                y_m=float(p_world[1]),
                z_m=float(p_world[2]),
                confidence=conf,
                reprojection_error_px=err,
            )
        )

    log_tri(f"  After reproj+depth filter: {len(candidates_3d)} pts (rejected: {rejected_reproj} reproj, {rejected_depth} depth)")

    # Enforce velocity and acceleration constraints
    filtered_3d: list[TrackedPoint3D] = []
    rejected_speed = 0
    rejected_accel = 0

    for pt in candidates_3d:
        if not filtered_3d:
            filtered_3d.append(pt)
            continue

        prev = filtered_3d[-1]
        dt = pt.time_seconds - prev.time_seconds
        if dt <= 0:
            continue

        p_curr = np.array([pt.x_m, pt.y_m, pt.z_m])
        p_prev = np.array([prev.x_m, prev.y_m, prev.z_m])
        vel = (p_curr - p_prev) / dt
        speed = float(np.linalg.norm(vel))

        # Reject speed > 100 m/s
        if speed > 100.0:
            rejected_speed += 1
            continue

        # Check acceleration if we have at least 2 previous points
        if len(filtered_3d) >= 2:
            prev_prev = filtered_3d[-2]
            dt_prev = prev.time_seconds - prev_prev.time_seconds
            p_prev_prev = np.array([prev_prev.x_m, prev_prev.y_m, prev_prev.z_m])
            vel_prev = (p_prev - p_prev_prev) / dt_prev

            accel = (vel - vel_prev) / dt
            accel_non_g = accel - np.array([0.0, 0.0, -9.8])
            accel_mag = float(np.linalg.norm(accel_non_g))

            # Reject non-gravitational acceleration > 150 m/s^2
            if accel_mag > 150.0:
                rejected_accel += 1
                continue

        filtered_3d.append(pt)

    log_tri(f"  After velocity/accel filter: {len(filtered_3d)} pts (rejected: {rejected_speed} speed, {rejected_accel} accel)")

    if filtered_3d:
        # Print sample 3D points for debugging
        for i, pt in enumerate(filtered_3d[:5]):
            log_tri(f"    3D [{i}]: t={pt.time_seconds:.4f}s, pos=({pt.x_m:.3f}, {pt.y_m:.3f}, {pt.z_m:.3f})m, reproj={pt.reprojection_error_px:.2f}px")

    return filtered_3d
