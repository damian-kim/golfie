import numpy as np
import pytest

from golfie_cv.detection import BallCandidate
from golfie_cv.tracking import track_ball_2d
from golfie_cv.tracking import track_ball_stereo, track_optic_ball_stereo
from golfie_core.schemas import CalibrationResult, CoordinateSystem


def test_track_ball_2d_perfect_parabola():
    fps = 240.0
    # Simulate a ball flying for 20 frames
    # Trajectory: x(t) = 100 + 400 * t, y(t) = 300 - 200 * t - 150 * t^2
    candidates_by_frame = []
    true_positions = []
    
    for f in range(20):
        t = f / fps
        x = 100.0 + 400.0 * t
        y = 300.0 - 200.0 * t - 150.0 * (t ** 2)
        true_positions.append((x, y))
        
        # Add single perfect candidate
        candidates_by_frame.append([
            BallCandidate(x_px=x, y_px=y, radius_px=5.0, confidence=0.95, is_streak=False)
        ])

    track = track_ball_2d(candidates_by_frame, fps)
    
    assert len(track) == 20
    for i, pt in enumerate(track):
        assert pt.frame_index == i
        assert pt.time_seconds == pytest.approx(i / fps)
        assert pt.x_px == pytest.approx(true_positions[i][0], abs=1e-3)
        assert pt.y_px == pytest.approx(true_positions[i][1], abs=1e-3)
        assert pt.confidence == pytest.approx(0.95)


def test_track_ball_2d_with_gaps():
    fps = 240.0
    # Simulate a ball flying for 15 frames, but frames 5 and 6 have no detections
    candidates_by_frame = []
    true_positions = []
    
    for f in range(15):
        t = f / fps
        x = 50.0 + 600.0 * t
        y = 400.0 - 300.0 * t - 200.0 * (t ** 2)
        true_positions.append((x, y))
        
        if f in (5, 6):
            # Gap frame: no candidate
            candidates_by_frame.append([])
        else:
            candidates_by_frame.append([
                BallCandidate(x_px=x, y_px=y, radius_px=6.0, confidence=0.9, is_streak=True)
            ])

    track = track_ball_2d(candidates_by_frame, fps)
    
    assert len(track) == 15
    for i, pt in enumerate(track):
        assert pt.frame_index == i
        assert pt.time_seconds == pytest.approx(i / fps)
        assert pt.x_px == pytest.approx(true_positions[i][0], abs=1.0)
        assert pt.y_px == pytest.approx(true_positions[i][1], abs=1.0)
        
        if i in (5, 6):
            # Bridged gap frames must have low confidence
            assert pt.confidence == pytest.approx(0.1)
        else:
            assert pt.confidence == pytest.approx(0.9)


def test_track_ball_2d_with_outliers():
    fps = 240.0
    # Simulate a ball flying for 10 frames, with random noise candidates in each frame
    candidates_by_frame = []
    true_positions = []
    np.random.seed(42)
    
    for f in range(10):
        t = f / fps
        x = 200.0 + 300.0 * t
        y = 150.0 - 100.0 * t - 100.0 * (t ** 2)
        true_positions.append((x, y))
        
        cands = [
            # True candidate
            BallCandidate(x_px=x, y_px=y, radius_px=4.0, confidence=0.9, is_streak=False),
            # Noise outlier 1
            BallCandidate(x_px=float(np.random.uniform(0, 800)), y_px=float(np.random.uniform(0, 600)), radius_px=3.0, confidence=0.4, is_streak=False),
            # Noise outlier 2
            BallCandidate(x_px=float(np.random.uniform(0, 800)), y_px=float(np.random.uniform(0, 600)), radius_px=3.0, confidence=0.6, is_streak=False)
        ]
        # Shuffle candidates so the true one isn't always first
        np.random.shuffle(cands)
        candidates_by_frame.append(cands)

    track = track_ball_2d(candidates_by_frame, fps)
    
    assert len(track) == 10
    for i, pt in enumerate(track):
        assert pt.frame_index == i
        assert pt.x_px == pytest.approx(true_positions[i][0], abs=1e-3)
        assert pt.y_px == pytest.approx(true_positions[i][1], abs=1e-3)
        assert pt.confidence == pytest.approx(0.9)


def test_track_ball_2d_stationary_to_moving():
    fps = 240.0
    # Simulate a ball stationary at (100, 200) for 5 frames, then hit at frame 5 and flies for 10 frames
    candidates_by_frame = []
    true_positions = []
    
    for f in range(15):
        if f < 5:
            x = 100.0
            y = 200.0
        else:
            t_fly = (f - 5) / fps
            # Fast launching streak
            x = 100.0 + 800.0 * t_fly
            y = 200.0 - 500.0 * t_fly - 400.0 * (t_fly ** 2)
            
        true_positions.append((x, y))
        
        is_streak = (f >= 5)
        conf = 0.9 if is_streak else 0.95
        candidates_by_frame.append([
            BallCandidate(x_px=x, y_px=y, radius_px=5.0, confidence=conf, is_streak=is_streak)
        ])

    track = track_ball_2d(candidates_by_frame, fps)
    
    # The tracker should track the entire sequence from frame 0 to 14
    assert len(track) == 15
    for i, pt in enumerate(track):
        assert pt.frame_index == i
        assert pt.x_px == pytest.approx(true_positions[i][0], abs=1.5)
        assert pt.y_px == pytest.approx(true_positions[i][1], abs=1.5)


def test_stereo_tracker_rejects_high_confidence_epipolar_distractors():
    k = [[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0], [0.0, 0.0, 1.0]]
    ext_a = np.eye(4)
    ext_b = np.eye(4)
    ext_b[0, 3] = -1.0
    calibration = CalibrationResult(
        camera_a_intrinsics=k,
        camera_b_intrinsics=k,
        camera_a_extrinsics=ext_a.tolist(),
        camera_b_extrinsics=ext_b.tolist(),
        is_valid=True,
    )
    candidates_a = [[] for _ in range(30)]
    candidates_b = [[] for _ in range(30)]
    for frame in range(10, 20):
        x = 0.4 + 0.02 * (frame - 10)
        y = 0.8 - 0.015 * (frame - 10)
        u_a, v_a = 1000 * x / 5 + 500, 1000 * y / 5 + 500
        u_b, v_b = 1000 * (x - 1) / 5 + 500, v_a
        candidates_a[frame] = [
            BallCandidate(u_a, v_a, 4, 0.8),
            BallCandidate(100 + 8 * frame, 120, 5, 0.99),
        ]
        candidates_b[frame] = [
            BallCandidate(u_b, v_b, 4, 0.8),
            BallCandidate(900 - 5 * frame, 850, 5, 0.99),
        ]

    track_a, track_b = track_ball_stereo(candidates_a, candidates_b, 240.0, calibration)
    assert len(track_a) >= 8
    assert len(track_a) == len(track_b)
    assert all(abs(a.y_px - b.y_px) < 1e-6 for a, b in zip(track_a, track_b))


def test_stereo_tracker_recovers_small_slow_motion_frame_offset():
    k = [[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0], [0.0, 0.0, 1.0]]
    ext_a = np.eye(4)
    ext_b = np.eye(4)
    ext_b[0, 3] = -1.0
    calibration = CalibrationResult(
        camera_a_intrinsics=k,
        camera_b_intrinsics=k,
        camera_a_extrinsics=ext_a.tolist(),
        camera_b_extrinsics=ext_b.tolist(),
        is_valid=True,
    )
    candidates_a = [[] for _ in range(35)]
    candidates_b = [[] for _ in range(35)]
    for sample in range(10):
        x = 0.4 + 0.02 * sample
        y = 0.8 - 0.015 * sample
        u_a, v = 1000 * x / 5 + 500, 1000 * y / 5 + 500
        u_b = 1000 * (x - 1) / 5 + 500
        candidates_a[10 + sample] = [BallCandidate(u_a, v, 4, 0.9)]
        candidates_b[13 + sample] = [BallCandidate(u_b, v, 4, 0.9)]

    track_a, track_b = track_ball_stereo(candidates_a, candidates_b, 240.0, calibration)

    assert len(track_a) >= 8
    assert track_b[0].frame_index - track_a[0].frame_index == 3


def test_optic_tracker_recovers_real_iron_flight_over_long_distractors():
    calibration = CalibrationResult(
        calibration_version=2,
        coordinate_system=CoordinateSystem(
            target_direction_in_rig_frame=[0.0, 0.0, 1.0],
            up_direction_in_rig_frame=[0.0, -1.0, 0.0],
        ),
        camera_a_intrinsics=[[1491.4292, 0, 617.7333], [0, 1407.7747, 594.3919], [0, 0, 1]],
        camera_b_intrinsics=[[1694.9617, 0, 1035.0573], [0, 1720.0120, 459.3523], [0, 0, 1]],
        camera_a_extrinsics=np.eye(4).tolist(),
        camera_b_extrinsics=[
            [0.3674185, -0.1847797, 0.9115153, -1.3128251],
            [0.0586382, 0.9827177, 0.1755775, 0.0972559],
            [-0.9282054, -0.0110608, 0.3719038, 1.1987005],
            [0, 0, 0, 1],
        ],
        camera_a_distortion=[-0.0109689, -0.3496205, -0.0169869, -0.0544403, 0.4867264],
        camera_b_distortion=[-0.0226871, 3.0155413, -0.0134417, 0.0661154, -14.8383992],
        epipolar_error_p95_px=2.1306,
        is_valid=True,
    )
    points_a = [(688.6, 741.8), (724.9, 677.9), (753.5, 624.9), (779.7, 576.8), (802.7, 534.9), (821.8, 496.9)]
    points_b = [(1098.0, 996.3), (1219.8, 957.4), (1350.3, 918.4), (1487.3, 877.2), (1622.2, 833.9), (1757.8, 795.8)]
    candidates_a = [[] for _ in range(20)]
    candidates_b = [[] for _ in range(20)]
    for frame in range(20):
        candidates_a[frame].append(BallCandidate(200 + frame, 300, 8, 0.99))
        candidates_b[frame].append(BallCandidate(500 + frame, 400, 8, 0.99))
    for index, (point_a, point_b) in enumerate(zip(points_a, points_b), start=6):
        candidates_a[index].append(BallCandidate(*point_a, 20, 0.96, index > 6, 1.0, True, 40, 40))
        candidates_b[index].append(BallCandidate(*point_b, 45, 0.96, True, 1.0, True, 120, 55))

    track_a, track_b = track_optic_ball_stereo(candidates_a, candidates_b, 240.0, calibration)

    assert len(track_a) == 6
    assert len(track_b) == 6
    assert track_a[0].x_px == pytest.approx(points_a[0][0], abs=1)
    assert track_b[-1].x_px == pytest.approx(points_b[-1][0], abs=1)


def test_optic_tracker_accepts_four_frame_driver_streak():
    calibration = CalibrationResult(
        calibration_version=2,
        coordinate_system=CoordinateSystem(
            target_direction_in_rig_frame=[0.0, 0.0, 1.0],
            up_direction_in_rig_frame=[0.0, -1.0, 0.0],
        ),
        camera_a_intrinsics=[[1491.4292, 0, 617.7333], [0, 1407.7747, 594.3919], [0, 0, 1]],
        camera_b_intrinsics=[[1694.9617, 0, 1035.0573], [0, 1720.0120, 459.3523], [0, 0, 1]],
        camera_a_extrinsics=np.eye(4).tolist(),
        camera_b_extrinsics=[
            [0.3674185, -0.1847797, 0.9115153, -1.3128251],
            [0.0586382, 0.9827177, 0.1755775, 0.0972559],
            [-0.9282054, -0.0110608, 0.3719038, 1.1987005],
            [0, 0, 0, 1],
        ],
        camera_a_distortion=[-0.0109689, -0.3496205, -0.0169869, -0.0544403, 0.4867264],
        camera_b_distortion=[-0.0226871, 3.0155413, -0.0134417, 0.0661154, -14.8383992],
        epipolar_error_p95_px=2.1306,
        is_valid=True,
    )
    points_a = [(876.6, 743.2), (906.9, 672.3), (932.2, 614.6), (949.9, 570.9)]
    points_b = [(997.3, 1058.2), (1205.0, 1034.8), (1472.3, 1007.1), (1745.0, 974.9)]
    candidates_a = [[] for _ in range(12)]
    candidates_b = [[] for _ in range(12)]
    for index, (point_a, point_b) in enumerate(zip(points_a, points_b), start=4):
        candidates_a[index].append(BallCandidate(*point_a, 20, 0.92, True, 1.0, True, 40, 44))
        candidates_b[index].append(BallCandidate(*point_b, 60, 0.92, True, 1.0, True, 210, 56))

    track_a, track_b = track_optic_ball_stereo(candidates_a, candidates_b, 240.0, calibration)

    assert len(track_a) == 4
    assert len(track_b) == 4
    assert track_a[0].x_px == pytest.approx(points_a[0][0], abs=1)
    assert track_b[-1].x_px == pytest.approx(points_b[-1][0], abs=1)
