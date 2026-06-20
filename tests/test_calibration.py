import cv2
import numpy as np
import pytest
from pathlib import Path

from golfie_core.schemas import CameraIntrinsics
from golfie_cv.calibration import calibrate_intrinsics, calibrate_stereo


def create_synthetic_board_image(grid_size, square_size_px=100) -> np.ndarray:
    """Create a flat checkerboard pattern image.
    grid_size is (cols, rows) of squares.
    """
    cols, rows = grid_size
    img = np.zeros((rows * square_size_px, cols * square_size_px), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                img[r * square_size_px : (r + 1) * square_size_px, c * square_size_px : (c + 1) * square_size_px] = 255
    return img


def generate_perspective_warped_board(
    board_img: np.ndarray,
    grid_size: tuple[int, int],
    square_size_px: int,
    square_length: float,
    K: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    img_size: tuple[int, int],
) -> np.ndarray:
    """Project the board corners to the camera coordinate frame and warp the
    flat checkerboard image onto the camera plane.
    """
    cols, rows = grid_size
    # Corner coordinates of the board outline on the flat image
    src_pts = np.float32([
        [0, 0],
        [cols * square_size_px, 0],
        [cols * square_size_px, rows * square_size_px],
        [0, rows * square_size_px],
    ])

    # Corresponding 3D coordinates in the board frame (z=0)
    dst_pts_3d = np.float32([
        [0, 0, 0],
        [cols * square_length, 0, 0],
        [cols * square_length, rows * square_length, 0],
        [0, rows * square_length, 0],
    ])

    # Project the 3D outline corners using camera pose
    projected, _ = cv2.projectPoints(dst_pts_3d, rvec, tvec, K, distCoeffs=None)
    dst_pts_2d = projected.reshape(4, 2).astype(np.float32)

    # Warp the checkerboard image to match perspective projection
    H = cv2.getPerspectiveTransform(src_pts, dst_pts_2d)
    warped = cv2.warpPerspective(board_img, H, img_size, borderValue=128)
    return warped


def test_calibration_recovers_synthetic_parameters(tmp_path):
    # Configure board parameters
    grid_size = (8, 6)  # cols, rows of squares
    square_length = 0.04  # 4 cm
    square_size_px = 80
    img_size = (640, 480)

    # Generate base board image
    board_img = create_synthetic_board_image(grid_size, square_size_px)

    # Ground truth intrinsics
    K = np.array([
        [600.0, 0.0, 320.0],
        [0.0, 600.0, 240.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    # Intrinsic camera schema
    gt_intrinsics = CameraIntrinsics(
        fx=600.0,
        fy=600.0,
        cx=320.0,
        cy=240.0,
        distortion=[0.0, 0.0, 0.0, 0.0, 0.0],
        image_width=img_size[0],
        image_height=img_size[1]
    )

    # Generate a few calibration poses (more diverse angles help resolve focal length ambiguity)
    poses = [
        (np.array([0.1, -0.05, 0.2], dtype=np.float32), np.array([-0.1, -0.05, 0.45], dtype=np.float32)),
        (np.array([0.05, 0.1, -0.15], dtype=np.float32), np.array([-0.08, -0.08, 0.48], dtype=np.float32)),
        (np.array([-0.15, 0.05, 0.1], dtype=np.float32), np.array([-0.12, -0.02, 0.42], dtype=np.float32)),
        (np.array([0.02, -0.12, 0.05], dtype=np.float32), np.array([-0.1, -0.1, 0.5], dtype=np.float32)),
        (np.array([0.2, 0.1, -0.05], dtype=np.float32), np.array([-0.09, -0.04, 0.40], dtype=np.float32)),
        (np.array([-0.1, -0.2, 0.1], dtype=np.float32), np.array([-0.11, -0.06, 0.46], dtype=np.float32)),
        (np.array([0.05, 0.05, 0.25], dtype=np.float32), np.array([-0.07, -0.07, 0.44], dtype=np.float32)),
        (np.array([-0.05, -0.05, -0.25], dtype=np.float32), np.array([-0.13, -0.03, 0.43], dtype=np.float32)),
    ]

    # Save generated images to disk
    cam_a_paths = []
    cam_b_paths = []
    
    # We will position camera B shifted relative to camera A
    # R_rel, T_rel: relative transform from A to B
    R_rel = cv2.Rodrigues(np.array([0.05, -0.1, 0.02], dtype=np.float32))[0]
    T_rel = np.array([0.15, -0.02, 0.05], dtype=np.float32).reshape(3, 1)

    for idx, (rvec_a, tvec_a) in enumerate(poses):
        # Camera A warped image
        img_a = generate_perspective_warped_board(
            board_img, grid_size, square_size_px, square_length, K, rvec_a, tvec_a, img_size
        )
        path_a = tmp_path / f"cam_a_{idx}.png"
        cv2.imwrite(str(path_a), img_a)
        cam_a_paths.append(path_a)

        # Camera B pose in the same board frame
        # X_B = R_rel * X_A + T_rel = R_rel * (R_A * X_board + t_A) + T_rel
        # R_B = R_rel * R_A, t_B = R_rel * t_A + T_rel
        R_a = cv2.Rodrigues(rvec_a)[0]
        R_b = R_rel @ R_a
        tvec_b = (R_rel @ tvec_a.reshape(3, 1) + T_rel).flatten()
        rvec_b = cv2.Rodrigues(R_b)[0].flatten()

        # Camera B warped image
        img_b = generate_perspective_warped_board(
            board_img, grid_size, square_size_px, square_length, K, rvec_b, tvec_b, img_size
        )
        path_b = tmp_path / f"cam_b_{idx}.png"
        cv2.imwrite(str(path_b), img_b)
        cam_b_paths.append(path_b)

    # Perform intrinsics calibration
    calib_intrinsics_a = calibrate_intrinsics(
        cam_a_paths, board_type="chessboard", grid_size=grid_size, square_length=square_length
    )

    # Validate intrinsics recovery
    assert calib_intrinsics_a.fx == pytest.approx(600.0, abs=80.0)
    assert calib_intrinsics_a.fy == pytest.approx(600.0, abs=80.0)
    assert calib_intrinsics_a.cx == pytest.approx(320.0, abs=80.0)
    assert calib_intrinsics_a.cy == pytest.approx(240.0, abs=80.0)
    assert calib_intrinsics_a.reprojection_error_px is not None
    assert calib_intrinsics_a.reprojection_error_px < 1.0

    # Perform stereo extrinsic calibration
    stereo_res = calibrate_stereo(
        calib_intrinsics_a,
        calib_intrinsics_a,  # use same for both
        cam_a_paths,
        cam_b_paths,
        board_type="chessboard",
        grid_size=grid_size,
        square_length=square_length
    )

    # Validate stereo calibration extrinsics recovery
    assert stereo_res.is_valid is True
    assert stereo_res.confidence > 0.5

    # Camera B extrinsics matrix should match the relative transform relative to Camera A (identity)
    ext_b = np.array(stereo_res.camera_b_extrinsics)
    R_recovered = ext_b[:3, :3]
    T_recovered = ext_b[:3, 3]

    assert np.allclose(R_recovered, R_rel, atol=0.05)
    assert np.allclose(T_recovered, T_rel.flatten(), atol=0.02)
