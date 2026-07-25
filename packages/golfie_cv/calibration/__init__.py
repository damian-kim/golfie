"""Camera calibration: intrinsics + stereo extrinsics + world alignment.

Implements ChArUco/chessboard intrinsics plus validation-gated stereo
extrinsics. The persisted result contains the complete lens model needed by
triangulation, including distortion and calibrated image size.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from golfie_core.schemas import CalibrationResult, CameraIntrinsics, CoordinateSystem


import cv2
import numpy as np

MIN_CHARUCO_CORNERS = 4


def _solve_stable_phone_intrinsics(obj_points, img_points, image_size):
    """Calibrate a video camera without letting sparse board coverage invent
    extreme lens distortion.

    Phone video has square pixels and its encoded crop is centred.  Enforcing
    those known properties leaves k1/k2 to describe radial distortion while
    avoiding the high-order/pincipal-point overfit that catastrophically
    changes triangulation near the bottom of a 240-fps frame.
    """
    camera_matrix = cv2.initCameraMatrix2D(
        obj_points, img_points, image_size, aspectRatio=1.0
    ).astype(np.float64)
    focal = float(0.5 * (camera_matrix[0, 0] + camera_matrix[1, 1]))
    camera_matrix[0, 0] = focal
    camera_matrix[1, 1] = focal
    camera_matrix[0, 2] = (image_size[0] - 1.0) * 0.5
    camera_matrix[1, 2] = (image_size[1] - 1.0) * 0.5
    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_FIX_ASPECT_RATIO
        | cv2.CALIB_FIX_PRINCIPAL_POINT
        | cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_K3
    )
    return cv2.calibrateCamera(
        obj_points,
        img_points,
        image_size,
        camera_matrix,
        np.zeros((5, 1), dtype=np.float64),
        flags=flags,
    )

def calibrate_intrinsics(
    calibration_images: Sequence[str | Path],
    board_type: str = "charuco",
    grid_size: tuple[int, int] = (11, 8),  # (squares_x, squares_y)
    square_length: float = 0.04,          # in meters
    marker_length: float = 0.03,          # in meters (for charuco)
    dictionary_id: int = cv2.aruco.DICT_6X6_250,
    progress_callback: Callable[[float, str], None] | None = None,
) -> CameraIntrinsics:
    """Estimate one camera's intrinsics from checkerboard/ChArUco frames.

    Spec section 7, Phase 1.
    """
    if not calibration_images:
        raise ValueError("No calibration images provided.")

    def report(fraction: float, message: str) -> None:
        if progress_callback is not None:
            progress_callback(max(0.0, min(1.0, fraction)), message)

    report(0.01, f"Preparing {len(calibration_images)} selected frames.")

    obj_points = []  # 3d points in real world space
    img_points = []  # 2d points in image plane
    image_size = None

    # Chessboard grid size parameter represents interior corners
    chessboard_corners_size = (grid_size[0] - 1, grid_size[1] - 1)

    # Prepare 3D points for chessboard
    if board_type == "chessboard":
        objp = np.zeros((chessboard_corners_size[0] * chessboard_corners_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_corners_size[0], 0:chessboard_corners_size[1]].T.reshape(-1, 2) * square_length
    elif board_type == "charuco":
        # We try a list of standard dictionaries to support users who printed their board using non-default presets (e.g. DICT_4X4_50)
        seen_dicts = set()
        unique_dicts = []
        for d in [dictionary_id] + [
            cv2.aruco.DICT_4X4_50,
            cv2.aruco.DICT_4X4_100,
            cv2.aruco.DICT_4X4_250,
            cv2.aruco.DICT_4X4_1000,
            cv2.aruco.DICT_5X5_50,
            cv2.aruco.DICT_5X5_100,
            cv2.aruco.DICT_5X5_250,
            cv2.aruco.DICT_5X5_1000,
            cv2.aruco.DICT_6X6_50,
            cv2.aruco.DICT_6X6_100,
            cv2.aruco.DICT_6X6_250,
            cv2.aruco.DICT_6X6_1000,
            cv2.aruco.DICT_7X7_50,
            cv2.aruco.DICT_7X7_100,
            cv2.aruco.DICT_7X7_250,
            cv2.aruco.DICT_7X7_1000,
            cv2.aruco.DICT_ARUCO_ORIGINAL,
        ]:
            if d not in seen_dicts:
                seen_dicts.add(d)
                unique_dicts.append(d)
        
        # Configure detector parameters to detect small markers far away (lowering minMarkerPerimeterRate)
        detector_params = cv2.aruco.DetectorParameters()
        detector_params.minMarkerPerimeterRate = 0.005  # Allow smaller markers at a distance
        detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        
        # Try both grid orientations in case the board is rotated
        grid_sizes_to_try = [grid_size]
        if grid_size[0] != grid_size[1]:
            grid_sizes_to_try.append((grid_size[1], grid_size[0]))

        detected_dict = None
        detected_grid_size = None
        detector_attempts = len(unique_dicts) * len(grid_sizes_to_try)
        detector_attempt = 0
        for dict_id in unique_dicts:
            for g_size in grid_sizes_to_try:
                detector_attempt += 1
                report(
                    0.02 + 0.23 * detector_attempt / detector_attempts,
                    f"Testing board dictionary/layout {detector_attempt}/{detector_attempts}.",
                )
                dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
                board = cv2.aruco.CharucoBoard(g_size, square_length, marker_length, dictionary)
                detector = cv2.aruco.CharucoDetector(board, detectorParams=detector_params)
                
                any_detected = False
                for img_path in calibration_images:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
                    if charuco_corners is not None and len(charuco_corners) >= MIN_CHARUCO_CORNERS:
                        any_detected = True
                        break
                
                if any_detected:
                    detected_dict = dict_id
                    detected_grid_size = g_size
                    report(
                        0.25,
                        f"Board layout identified as {g_size[0]}x{g_size[1]} "
                        f"(dictionary {dict_id}); partial views are accepted.",
                    )
                    break
            if detected_dict is not None:
                break
                
        if detected_dict is None:
            dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
            board = cv2.aruco.CharucoBoard(grid_size, square_length, marker_length, dictionary)
            detector = cv2.aruco.CharucoDetector(board, detectorParams=detector_params)
        else:
            dictionary = cv2.aruco.getPredefinedDictionary(detected_dict)
            board = cv2.aruco.CharucoBoard(detected_grid_size, square_length, marker_length, dictionary)
            detector = cv2.aruco.CharucoDetector(board, detectorParams=detector_params)
    else:
        raise ValueError(f"Unknown board type: {board_type}")

    usable_views = 0
    for image_index, img_path in enumerate(calibration_images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        if board_type == "chessboard":
            ret, corners = cv2.findChessboardCorners(gray, chessboard_corners_size, None)
            if ret:
                obj_points.append(objp)
                # Refine corners
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                img_points.append(corners_refined)
                usable_views += 1
        elif board_type == "charuco":
            charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
            num_corners = len(charuco_corners) if charuco_corners is not None else 0
            print(f"[Calibration Debug] Image {Path(img_path).name}: detected {num_corners} ChArUco corners.")
            if charuco_corners is not None and len(charuco_corners) >= MIN_CHARUCO_CORNERS:
                # Get the 3D coordinates of detected ChArUco corners
                # board.getMatchPrediction (or manually matching charuco_ids to board.getChessboardCorners())
                # In modern OpenCV, CharucoBoard exposes chessboardCorners (which matches the IDs)
                # corners 3D positions are defined by board's corner coordinates:
                board_corners = board.getChessboardCorners()
                objp = board_corners[charuco_ids.flatten()]
                obj_points.append(objp)
                img_points.append(charuco_corners)
                usable_views += 1

        if (image_index + 1) % 4 == 0 or image_index + 1 == len(calibration_images):
            report(
                0.25 + 0.45 * (image_index + 1) / len(calibration_images),
                f"Detected {usable_views} usable views after checking "
                f"{image_index + 1}/{len(calibration_images)} frames.",
            )

    if not obj_points:
        raise ValueError(f"Could not detect any calibration board patterns in the provided images (board_type={board_type}).")
    if len(obj_points) < 4:
        raise ValueError(
            f"Only {len(obj_points)} usable calibration views were found; at least 4 varied board poses are required."
        )

    # Run calibration
    # Make object points and image points 32-bit floats
    obj_points = [np.array(p, dtype=np.float32) for p in obj_points]
    img_points = [np.array(p, dtype=np.float32) for p in img_points]

    try:
        report(0.75, f"Solving lens model from {len(obj_points)} usable views.")
        ret, mtx, dist, rvecs, tvecs = _solve_stable_phone_intrinsics(
            obj_points, img_points, image_size
        )

        # A few blurred or partially occluded frames can dominate an otherwise
        # sound solve.  Perform one conservative robust re-fit using per-view
        # reprojection errors.  Never prune below the minimum needed to
        # constrain a useful intrinsic model.
        if len(obj_points) >= 6:
            view_errors = []
            for obj, img, rvec, tvec in zip(obj_points, img_points, rvecs, tvecs):
                projected, _ = cv2.projectPoints(obj, rvec, tvec, mtx, dist)
                residual = projected.reshape(-1, 2) - img.reshape(-1, 2)
                view_errors.append(float(np.sqrt(np.mean(np.sum(residual * residual, axis=1)))))
            median = float(np.median(view_errors))
            mad = float(np.median(np.abs(np.asarray(view_errors) - median)))
            cutoff = max(1.5, median + 3.0 * max(mad, 0.05))
            keep = [i for i, err in enumerate(view_errors) if err <= cutoff]
            if 5 <= len(keep) < len(obj_points):
                report(
                    0.90,
                    f"Refitting after rejecting {len(obj_points) - len(keep)} high-error views.",
                )
                kept_obj = [obj_points[i] for i in keep]
                kept_img = [img_points[i] for i in keep]
                ret, mtx, dist, rvecs, tvecs = _solve_stable_phone_intrinsics(
                    kept_obj, kept_img, image_size
                )
    except cv2.error as e:
        raise ValueError(
            f"OpenCV camera calibration failed: {str(e)}. This usually happens when "
            "detected corners are degenerate (e.g., collinear) or there is not enough "
            "variation/frames with high-quality board detections in the calibration video."
        ) from e

    distortion = np.zeros(5, dtype=np.float64)
    flattened_distortion = np.asarray(dist, dtype=np.float64).reshape(-1)
    distortion[: min(5, len(flattened_distortion))] = flattened_distortion[:5]
    result = CameraIntrinsics(
        fx=float(mtx[0, 0]),
        fy=float(mtx[1, 1]),
        cx=float(mtx[0, 2]),
        cy=float(mtx[1, 2]),
        distortion=[float(x) for x in distortion],
        image_width=image_size[0],
        image_height=image_size[1],
        reprojection_error_px=float(ret),
    )
    report(1.0, f"Lens solve complete: RMS={result.reprojection_error_px:.3f}px.")
    return result


def calibrate_stereo(
    camera_a_intrinsics: CameraIntrinsics,
    camera_b_intrinsics: CameraIntrinsics,
    shared_board_images_a: Sequence[str | Path],
    shared_board_images_b: Sequence[str | Path],
    board_type: str = "charuco",
    grid_size: tuple[int, int] = (11, 8),
    square_length: float = 0.04,
    marker_length: float = 0.03,
    dictionary_id: int = cv2.aruco.DICT_6X6_250,
) -> CalibrationResult:
    """Estimate relative pose between camera A and camera B.

    Spec section 7, Phase 2.
    """
    if len(shared_board_images_a) != len(shared_board_images_b):
        raise ValueError("Stereo calibration requires corresponding pairs of images.")
    if (
        camera_a_intrinsics.image_width != camera_b_intrinsics.image_width
        or camera_a_intrinsics.image_height != camera_b_intrinsics.image_height
    ):
        raise ValueError(
            "Stereo calibration currently requires both camera streams to use the same "
            "resolution. Calibrate and record with matching orientation/resolution."
        )

    obj_points = []  # 3d points
    img_points_a = []  # 2d points from camera A
    img_points_b = []  # 2d points from camera B
    image_size = (camera_a_intrinsics.image_width, camera_a_intrinsics.image_height)

    chessboard_corners_size = (grid_size[0] - 1, grid_size[1] - 1)

    if board_type == "chessboard":
        objp = np.zeros((chessboard_corners_size[0] * chessboard_corners_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_corners_size[0], 0:chessboard_corners_size[1]].T.reshape(-1, 2) * square_length
    elif board_type == "charuco":
        # We try a list of standard dictionaries to support users who printed their board using non-default presets (e.g. DICT_4X4_50)
        seen_dicts = set()
        unique_dicts = []
        for d in [dictionary_id] + [
            cv2.aruco.DICT_4X4_50,
            cv2.aruco.DICT_4X4_100,
            cv2.aruco.DICT_4X4_250,
            cv2.aruco.DICT_4X4_1000,
            cv2.aruco.DICT_5X5_50,
            cv2.aruco.DICT_5X5_100,
            cv2.aruco.DICT_5X5_250,
            cv2.aruco.DICT_5X5_1000,
            cv2.aruco.DICT_6X6_50,
            cv2.aruco.DICT_6X6_100,
            cv2.aruco.DICT_6X6_250,
            cv2.aruco.DICT_6X6_1000,
            cv2.aruco.DICT_7X7_50,
            cv2.aruco.DICT_7X7_100,
            cv2.aruco.DICT_7X7_250,
            cv2.aruco.DICT_7X7_1000,
            cv2.aruco.DICT_ARUCO_ORIGINAL,
        ]:
            if d not in seen_dicts:
                seen_dicts.add(d)
                unique_dicts.append(d)
        
        # Configure detector parameters to detect small markers far away (lowering minMarkerPerimeterRate)
        detector_params = cv2.aruco.DetectorParameters()
        detector_params.minMarkerPerimeterRate = 0.005  # Allow smaller markers at a distance
        detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        
        # Try both grid orientations in case the board is rotated
        grid_sizes_to_try = [grid_size]
        if grid_size[0] != grid_size[1]:
            grid_sizes_to_try.append((grid_size[1], grid_size[0]))

        detected_dict = None
        detected_grid_size = None
        for dict_id in unique_dicts:
            for g_size in grid_sizes_to_try:
                dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
                board = cv2.aruco.CharucoBoard(g_size, square_length, marker_length, dictionary)
                detector = cv2.aruco.CharucoDetector(board, detectorParams=detector_params)
                
                any_detected = False
                for img_path_a, img_path_b in zip(shared_board_images_a, shared_board_images_b):
                    img_a = cv2.imread(str(img_path_a))
                    img_b = cv2.imread(str(img_path_b))
                    if img_a is None or img_b is None:
                        continue
                    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
                    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
                    charuco_corners_a, charuco_ids_a, _, _ = detector.detectBoard(gray_a)
                    charuco_corners_b, charuco_ids_b, _, _ = detector.detectBoard(gray_b)
                    
                    if (charuco_corners_a is not None and len(charuco_corners_a) >= MIN_CHARUCO_CORNERS and
                            charuco_corners_b is not None and len(charuco_corners_b) >= MIN_CHARUCO_CORNERS):
                        ids_a = charuco_ids_a.flatten()
                        ids_b = charuco_ids_b.flatten()
                        common_ids = np.intersect1d(ids_a, ids_b)
                        if len(common_ids) >= MIN_CHARUCO_CORNERS:
                            any_detected = True
                            break
                
                if any_detected:
                    detected_dict = dict_id
                    detected_grid_size = g_size
                    break
            if detected_dict is not None:
                break
                
        if detected_dict is None:
            dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
            board = cv2.aruco.CharucoBoard(grid_size, square_length, marker_length, dictionary)
            detector = cv2.aruco.CharucoDetector(board, detectorParams=detector_params)
        else:
            dictionary = cv2.aruco.getPredefinedDictionary(detected_dict)
            board = cv2.aruco.CharucoBoard(detected_grid_size, square_length, marker_length, dictionary)
            detector = cv2.aruco.CharucoDetector(board, detectorParams=detector_params)
    else:
        raise ValueError(f"Unknown board type: {board_type}")

    for img_path_a, img_path_b in zip(shared_board_images_a, shared_board_images_b):
        img_a = cv2.imread(str(img_path_a))
        img_b = cv2.imread(str(img_path_b))
        if img_a is None or img_b is None:
            continue

        gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

        if board_type == "chessboard":
            ret_a, corners_a = cv2.findChessboardCorners(gray_a, chessboard_corners_size, None)
            ret_b, corners_b = cv2.findChessboardCorners(gray_b, chessboard_corners_size, None)
            if ret_a and ret_b:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_a = cv2.cornerSubPix(gray_a, corners_a, (11, 11), (-1, -1), criteria)
                corners_b = cv2.cornerSubPix(gray_b, corners_b, (11, 11), (-1, -1), criteria)
                obj_points.append(objp)
                img_points_a.append(corners_a)
                img_points_b.append(corners_b)
        elif board_type == "charuco":
            charuco_corners_a, charuco_ids_a, _, _ = detector.detectBoard(gray_a)
            charuco_corners_b, charuco_ids_b, _, _ = detector.detectBoard(gray_b)

            len_a = len(charuco_corners_a) if charuco_corners_a is not None else 0
            len_b = len(charuco_corners_b) if charuco_corners_b is not None else 0
            print(
                f"[Stereo Debug] Frame {Path(img_path_a).name}/{Path(img_path_b).name}: "
                f"Cam A detected {len_a} corners, Cam B detected {len_b} corners."
            )

            if (charuco_corners_a is not None and len(charuco_corners_a) >= MIN_CHARUCO_CORNERS and
                    charuco_corners_b is not None and len(charuco_corners_b) >= MIN_CHARUCO_CORNERS):
                
                # We need to find corresponding corners observed in BOTH views
                ids_a = charuco_ids_a.flatten()
                ids_b = charuco_ids_b.flatten()
                common_ids = np.intersect1d(ids_a, ids_b)
                print(f"[Stereo Debug] Shared corners: {len(common_ids)}")

                if len(common_ids) >= MIN_CHARUCO_CORNERS:
                    pts_a = []
                    pts_b = []
                    board_corners = board.getChessboardCorners()
                    pts_obj = []

                    for cid in common_ids:
                        idx_a = np.where(ids_a == cid)[0][0]
                        idx_b = np.where(ids_b == cid)[0][0]
                        pts_a.append(charuco_corners_a[idx_a])
                        pts_b.append(charuco_corners_b[idx_b])
                        pts_obj.append(board_corners[cid])

                    obj_points.append(np.array(pts_obj, dtype=np.float32))
                    img_points_a.append(np.array(pts_a, dtype=np.float32))
                    img_points_b.append(np.array(pts_b, dtype=np.float32))

    if not obj_points:
        raise ValueError("Could not find simultaneous detections of the board in both views.")
    if len(obj_points) < 3:
        raise ValueError(
            f"Only {len(obj_points)} usable stereo board poses were found; at least 3 are required."
        )

    # Reconstruct Camera matrices from intrinsics
    mtx_a = np.array(camera_a_intrinsics.as_matrix(), dtype=np.float32)
    dist_a = np.array(camera_a_intrinsics.distortion, dtype=np.float32)
    mtx_b = np.array(camera_b_intrinsics.as_matrix(), dtype=np.float32)
    dist_b = np.array(camera_b_intrinsics.distortion, dtype=np.float32)

    # Reserve a deterministic subset of poses for geometry validation.  A
    # calibration should not grade the same correspondences it optimized.
    if len(obj_points) >= 6:
        validation_indices = {i for i in range(len(obj_points)) if i % 5 == 0}
        training_indices = [i for i in range(len(obj_points)) if i not in validation_indices]
    else:
        validation_indices = set(range(len(obj_points)))
        training_indices = list(range(len(obj_points)))
    solve_obj = [obj_points[i] for i in training_indices]
    solve_a = [img_points_a[i] for i in training_indices]
    solve_b = [img_points_b[i] for i in training_indices]

    # Perform stereo calibration. Keep intrinsics fixed.
    flags = cv2.CALIB_FIX_INTRINSIC
    try:
        ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            solve_obj, solve_a, solve_b,
            mtx_a, dist_a, mtx_b, dist_b,
            image_size, flags=flags
        )
    except cv2.error as e:
        raise ValueError(
            f"OpenCV stereo calibration failed: {str(e)}. "
            "Ensure the calibration board is simultaneously visible, well-lit, and in focus "
            "in both camera views."
        ) from e

    # Format camera A extrinsics as identity world coordinate frame by default,
    # and camera B extrinsics relative to it.
    # We will build transformation matrices: world to camera.
    # camera A is at the origin of the rig coordinate system.
    ext_a = np.eye(4)
    
    ext_b = np.eye(4)
    ext_b[:3, :3] = R
    ext_b[:3, 3] = T.flatten()

    # Symmetric point-to-epipolar-line error on all calibration observations.
    # This measures whether correspondences satisfy the recovered stereo
    # geometry; optimizer RMS alone can look acceptable for a degenerate rig.
    epipolar_errors: list[float] = []
    validation_a = [img_points_a[i] for i in sorted(validation_indices)]
    validation_b = [img_points_b[i] for i in sorted(validation_indices)]
    for pts_a, pts_b in zip(validation_a, validation_b):
        pa = np.asarray(pts_a, dtype=np.float64).reshape(-1, 2)
        pb = np.asarray(pts_b, dtype=np.float64).reshape(-1, 2)
        pa_norm = cv2.undistortPoints(pa.reshape(-1, 1, 2), mtx_a, dist_a).reshape(-1, 2)
        pb_norm = cv2.undistortPoints(pb.reshape(-1, 1, 2), mtx_b, dist_b).reshape(-1, 2)
        pa_h = np.column_stack([pa_norm, np.ones(len(pa_norm))])
        pb_h = np.column_stack([pb_norm, np.ones(len(pb_norm))])
        lines_b = (E @ pa_h.T).T
        lines_a = (E.T @ pb_h.T).T
        denom_b = np.linalg.norm(lines_b[:, :2], axis=1)
        denom_a = np.linalg.norm(lines_a[:, :2], axis=1)
        error_b = np.abs(np.sum(lines_b * pb_h, axis=1)) / np.maximum(denom_b, 1e-12)
        error_a = np.abs(np.sum(lines_a * pa_h, axis=1)) / np.maximum(denom_a, 1e-12)
        focal_scale = float(np.mean([mtx_a[0, 0], mtx_a[1, 1], mtx_b[0, 0], mtx_b[1, 1]]))
        epipolar_errors.extend((0.5 * (error_a + error_b) * focal_scale).tolist())

    epi_median = float(np.median(epipolar_errors)) if epipolar_errors else float("inf")
    epi_p95 = float(np.percentile(epipolar_errors, 95)) if epipolar_errors else float("inf")
    epi_inlier_ratio = (
        float(np.mean(np.asarray(epipolar_errors) <= 4.0)) if epipolar_errors else 0.0
    )
    baseline_m = float(np.linalg.norm(T))
    warnings: list[str] = []
    fatal_warnings: list[str] = []
    if len(obj_points) < 3:
        fatal_warnings.append("Fewer than 3 usable stereo board poses were found.")
    if float(ret) > 2.0:
        fatal_warnings.append(f"Stereo calibration RMS is high ({float(ret):.2f}px; expected <= 2px).")
    if epi_median > 1.5:
        fatal_warnings.append(
            f"Held-out median epipolar residual is high ({epi_median:.2f}px; expected <= 1.5px)."
        )
    if epi_inlier_ratio < 0.85:
        fatal_warnings.append(
            "Too few held-out ChArUco correspondences satisfy the stereo geometry "
            f"({epi_inlier_ratio * 100:.1f}% within 4px; expected at least 85%)."
        )
    elif epi_p95 > 4.0:
        warnings.append(
            f"A small held-out correspondence tail was rejected (p95 {epi_p95:.2f}px; "
            f"{epi_inlier_ratio * 100:.1f}% remain within 4px)."
        )
    if not 0.05 <= baseline_m <= 10.0:
        fatal_warnings.append(f"Recovered camera baseline is implausible ({baseline_m:.3f}m).")

    warnings = fatal_warnings + warnings
    is_valid = not fatal_warnings
    rms_score = max(0.0, 1.0 - float(ret) / 2.0)
    # A calibration at the acceptance boundary should be low-confidence, not
    # effectively zero-confidence. Map 0..3 px onto 1..0 while validation still
    # enforces the stricter 1.5 px median ceiling above.
    epi_score = max(0.0, 1.0 - epi_median / 3.0)
    confidence = float(min(rms_score, epi_score) * epi_inlier_ratio) if is_valid else 0.0

    # MVP world alignment assumes Camera A is an upright down-the-line view:
    # OpenCV camera +Z looks forward/downrange and image -Y is up.  The tee
    # origin is established from the first accepted 3D ball point in the shot
    # pipeline. A future floor-marker workflow can replace this approximation.
    cs = CoordinateSystem(
        origin_in_rig_frame_m=[0.0, 0.0, 0.0],
        target_direction_in_rig_frame=[0.0, 0.0, 1.0],
        up_direction_in_rig_frame=[0.0, -1.0, 0.0],
        alignment_method="camera_a_down_the_line_assumption",
        notes=(
            "MVP alignment: Camera A must be upright and down-the-line. "
            "Shot processing recenters the first accepted ball point at the origin."
        ),
    )

    return CalibrationResult(
        calibration_version=2,
        coordinate_system=cs,
        camera_a_intrinsics=camera_a_intrinsics.as_matrix(),
        camera_b_intrinsics=camera_b_intrinsics.as_matrix(),
        camera_a_extrinsics=ext_a.tolist(),
        camera_b_extrinsics=ext_b.tolist(),
        camera_a_distortion=list(camera_a_intrinsics.distortion),
        camera_b_distortion=list(camera_b_intrinsics.distortion),
        camera_a_image_size=(camera_a_intrinsics.image_width, camera_a_intrinsics.image_height),
        camera_b_image_size=(camera_b_intrinsics.image_width, camera_b_intrinsics.image_height),
        stereo_rotation=R.tolist(),
        stereo_translation_m=T.flatten().tolist(),
        essential_matrix=E.tolist(),
        fundamental_matrix=F.tolist(),
        intrinsic_error_a_px=camera_a_intrinsics.reprojection_error_px,
        intrinsic_error_b_px=camera_b_intrinsics.reprojection_error_px,
        epipolar_error_median_px=epi_median,
        epipolar_error_p95_px=epi_p95,
        epipolar_inlier_ratio=epi_inlier_ratio,
        baseline_m=baseline_m,
        validation_frame_count=len(validation_indices),
        validation_warnings=warnings,
        reprojection_error_px=float(ret),
        confidence=confidence,
        calibration_target=board_type,
        board_grid_size=tuple(
            int(value)
            for value in (
                detected_grid_size
                if board_type == "charuco" and detected_grid_size is not None
                else grid_size
            )
        ),
        board_square_length_m=float(square_length),
        board_marker_length_m=float(marker_length) if board_type == "charuco" else None,
        is_valid=is_valid,
    )
