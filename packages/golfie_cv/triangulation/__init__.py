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

    # 3. Match track points by frame_index
    map_a = {pt.frame_index: pt for pt in track_a}
    map_b = {pt.frame_index: pt for pt in track_b}
    common_frames = sorted(list(set(map_a.keys()) & set(map_b.keys())))

    if not common_frames:
        return []

    # 4. Form coordinates matrices
    pts_a_np = np.zeros((2, len(common_frames)), dtype=np.float64)
    pts_b_np = np.zeros((2, len(common_frames)), dtype=np.float64)

    for idx, f in enumerate(common_frames):
        pts_a_np[0, idx] = map_a[f].x_px
        pts_a_np[1, idx] = map_a[f].y_px
        pts_b_np[0, idx] = map_b[f].x_px
        pts_b_np[1, idx] = map_b[f].y_px

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
    candidates_3d = []
    for idx, f in enumerate(common_frames):
        pt_a = map_a[f]
        pt_b = map_b[f]
        p_world = points_3d_world[idx]
        p_rig = points_3d_rig[:, idx]
        err = float(reproj_errors[idx])
        
        # Average time and geometric mean of confidence
        t_sec = 0.5 * (pt_a.time_seconds + pt_b.time_seconds)
        conf = float(np.sqrt(pt_a.confidence * pt_b.confidence))

        # Check reprojection error threshold (10.0 px)
        if err > 10.0:
            continue

        # Check positive depth check (must be in front of both cameras)
        # For Camera A: depth is p_rig[2]
        # For Camera B: depth is the 3rd coordinate of ext_b * p_rig
        p_rig_hom = np.append(p_rig, 1.0)
        p_cam_b = ext_b[:3, :] @ p_rig_hom
        
        if p_rig[2] <= 0.1 or p_cam_b[2] <= 0.1:
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

    # Enforce velocity and acceleration constraints
    filtered_3d: list[TrackedPoint3D] = []
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
                continue

        filtered_3d.append(pt)

    return filtered_3d
