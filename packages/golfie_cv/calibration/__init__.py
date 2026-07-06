"""Camera calibration: intrinsics + stereo extrinsics + world alignment.

STATUS: stub. Implemented in Milestone 1 (spec section 20).

These function signatures are the contract the rest of the system will
call against -- the backend's `/sessions/{id}/calibration` endpoint and
`scripts/calibrate_cameras.py` are written to call these names, so
filling them in later should not require touching call sites.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from golfie_core.schemas import CalibrationResult, CameraIntrinsics, CoordinateSystem


import cv2
import numpy as np

MIN_CHARUCO_CORNERS = 6

def calibrate_intrinsics(
    calibration_images: Sequence[str | Path],
    board_type: str = "charuco",
    grid_size: tuple[int, int] = (11, 8),  # (squares_x, squares_y)
    square_length: float = 0.04,          # in meters
    marker_length: float = 0.03,          # in meters (for charuco)
    dictionary_id: int = cv2.aruco.DICT_6X6_250,
) -> CameraIntrinsics:
    """Estimate one camera's intrinsics from checkerboard/ChArUco frames.

    Spec section 7, Phase 1.
    """
    if not calibration_images:
        raise ValueError("No calibration images provided.")

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
        for dict_id in unique_dicts:
            for g_size in grid_sizes_to_try:
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

    for img_path in calibration_images:
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
        elif board_type == "charuco":
            charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
            num_corners = len(charuco_corners) if charuco_corners is not None else 0
            print(f"[Calibration Debug] Image {img_path.name}: detected {num_corners} ChArUco corners.")
            if charuco_corners is not None and len(charuco_corners) >= MIN_CHARUCO_CORNERS:
                # Get the 3D coordinates of detected ChArUco corners
                # board.getMatchPrediction (or manually matching charuco_ids to board.getChessboardCorners())
                # In modern OpenCV, CharucoBoard exposes chessboardCorners (which matches the IDs)
                # corners 3D positions are defined by board's corner coordinates:
                board_corners = board.getChessboardCorners()
                objp = board_corners[charuco_ids.flatten()]
                obj_points.append(objp)
                img_points.append(charuco_corners)

    if not obj_points:
        raise ValueError(f"Could not detect any calibration board patterns in the provided images (board_type={board_type}).")

    # Run calibration
    # Make object points and image points 32-bit floats
    obj_points = [np.array(p, dtype=np.float32) for p in obj_points]
    img_points = [np.array(p, dtype=np.float32) for p in img_points]

    try:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, image_size, None, None
        )
    except cv2.error as e:
        raise ValueError(
            f"OpenCV camera calibration failed: {str(e)}. This usually happens when "
            "detected corners are degenerate (e.g., collinear) or there is not enough "
            "variation/frames with high-quality board detections in the calibration video."
        ) from e

    return CameraIntrinsics(
        fx=float(mtx[0, 0]),
        fy=float(mtx[1, 1]),
        cx=float(mtx[0, 2]),
        cy=float(mtx[1, 2]),
        distortion=[float(x) for x in dist[0]],
        image_width=image_size[0],
        image_height=image_size[1],
        reprojection_error_px=float(ret),
    )


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
            print(f"[Stereo Debug] Frame {img_path_a.name}/{img_path_b.name}: Cam A detected {len_a} corners, Cam B detected {len_b} corners.")

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

    # Reconstruct Camera matrices from intrinsics
    mtx_a = np.array(camera_a_intrinsics.as_matrix(), dtype=np.float32)
    dist_a = np.array(camera_a_intrinsics.distortion, dtype=np.float32)
    mtx_b = np.array(camera_b_intrinsics.as_matrix(), dtype=np.float32)
    dist_b = np.array(camera_b_intrinsics.distortion, dtype=np.float32)

    # Perform stereo calibration. Keep intrinsics fixed.
    flags = cv2.CALIB_FIX_INTRINSIC
    try:
        ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            obj_points, img_points_a, img_points_b,
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

    # Calculate alignment confidence (lower reprojection error -> higher confidence)
    # A simple mapping: 0 px error = 1.0 confidence, 2 px or more = 0.0 confidence
    confidence = max(0.0, 1.0 - (ret / 2.0))

    # Build default coordinate system
    cs = CoordinateSystem(
        origin_in_rig_frame_m=[0.0, 0.0, 0.0],
        target_direction_in_rig_frame=[1.0, 0.0, 0.0],
        up_direction_in_rig_frame=[0.0, 0.0, 1.0],
        alignment_method="manual",
    )

    return CalibrationResult(
        coordinate_system=cs,
        camera_a_intrinsics=camera_a_intrinsics.as_matrix(),
        camera_b_intrinsics=camera_b_intrinsics.as_matrix(),
        camera_a_extrinsics=ext_a.tolist(),
        camera_b_extrinsics=ext_b.tolist(),
        reprojection_error_px=float(ret),
        confidence=confidence,
        calibration_target=board_type,
        is_valid=True,
    )
