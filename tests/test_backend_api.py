import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Build a TestClient against an isolated data directory so tests
    don't pollute (or depend on) the real data/shots directory.

    Note: `golfie_api/storage/__init__.py` does `from .session_store
    import session_store`, which rebinds the `session_store` attribute
    on the *package* to the singleton instance -- shadowing the
    submodule of the same name. So we reach the actual submodule via
    sys.modules rather than attribute/dotted-import access.
    """
    import sys

    monkeypatch.chdir(tmp_path)

    import golfie_api.routers.sessions as sessions_module
    import golfie_api.routers.video_trimmer as video_trimmer_module
    import golfie_api.storage as storage_pkg
    import golfie_api.storage.session_store  # noqa: F401 (ensures it's imported)
    from golfie_api import main as main_module
    from golfie_api.storage.session_store import SessionStore

    session_store_module = sys.modules["golfie_api.storage.session_store"]

    isolated_store = SessionStore(tmp_path / "shots")
    monkeypatch.setattr(session_store_module, "session_store", isolated_store)
    monkeypatch.setattr(storage_pkg, "session_store", isolated_store)
    monkeypatch.setattr(sessions_module, "session_store", isolated_store)
    monkeypatch.setattr(sessions_module, "CALIBRATION_DIR", tmp_path / "calibration")
    monkeypatch.setattr(video_trimmer_module, "TRIMS_DIR", tmp_path / "trims")
    (tmp_path / "trims").mkdir()

    return TestClient(main_module.app)


@pytest.fixture
def sample_clip(tmp_path):
    if not FFMPEG_AVAILABLE:
        pytest.skip("ffmpeg not available in this environment")
    path = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
            "-i", "testsrc=duration=0.2:size=320x240:rate=240",
            "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )
    return path


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_manual_video_editor_is_downloadable_and_preserves_rate(client, sample_clip):
    with open(sample_clip, "rb") as video:
        response = client.post(
            "/video-trimmer/trim",
            files={"file": ("my-swing.mp4", video, "video/mp4")},
            data={"start_seconds": "0.05", "end_seconds": "0.15"},
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["lossless"] is True
    assert body["requested_start_seconds"] == pytest.approx(0.05)
    assert body["requested_end_seconds"] == pytest.approx(0.15)
    assert body["fps"] == pytest.approx(240.0, rel=0.02)
    assert body["width"] == 320
    assert body["height"] == 240
    assert body["output_filename"] == "my-swing-golfie-trim.mp4"

    metadata = client.get(f"/video-trimmer/{body['trim_id']}")
    assert metadata.status_code == 200
    assert metadata.json()["requested_start_seconds"] == pytest.approx(0.05)

    preview = client.get(body["preview_url"])
    assert preview.status_code == 200
    assert preview.headers["content-disposition"].startswith("inline")
    assert len(preview.content) > 0

    download = client.get(body["download_url"])
    assert download.status_code == 200
    assert download.headers["content-disposition"].startswith("attachment")


def test_create_session_returns_unprocessed_session(client):
    response = client.post("/sessions", json={"club": "driver"})
    assert response.status_code == 200
    body = response.json()
    assert body["club"] == "driver"
    assert body["stage"] == "created"
    assert body["shot"] is None


def test_get_unknown_session_returns_404(client):
    response = client.get("/sessions/does-not-exist")
    assert response.status_code == 404


def test_manual_spin_preview_changes_trajectory_without_mutating_session(client):
    import golfie_api.routers.sessions as sessions_module
    from golfie_core.schemas import MetricSource, MetricValue, ShotMetrics, ShotResult

    session_id = client.post("/sessions", json={"club": "driver"}).json()["session_id"]
    stored = sessions_module.session_store.load(session_id)
    stored.shot = ShotResult(
        is_placeholder=False,
        metrics=ShotMetrics(
            ball_speed_mps=MetricValue(value=55.0, source=MetricSource.ESTIMATED, confidence=0.7),
            launch_angle_deg=MetricValue(value=14.0, source=MetricSource.ESTIMATED, confidence=0.7),
            horizontal_launch_deg=MetricValue(value=0.0, source=MetricSource.ESTIMATED, confidence=0.7),
        ),
    )
    sessions_module.session_store.save(stored)

    no_spin = client.post(
        f"/sessions/{session_id}/trajectory/spin-preview",
        json={"backspin_rpm": 0, "sidespin_rpm": 0},
    )
    backspin = client.post(
        f"/sessions/{session_id}/trajectory/spin-preview",
        json={"backspin_rpm": 3000, "sidespin_rpm": 0},
    )

    assert no_spin.status_code == 200
    assert backspin.status_code == 200
    assert backspin.json()["carry_m"] > no_spin.json()["carry_m"]
    assert len(backspin.json()["simulated_trajectory"]) > 50
    assert sessions_module.session_store.load(session_id).shot.metrics.backspin_rpm.value is None


def test_process_without_both_cameras_returns_400(client):
    session = client.post("/sessions", json={}).json()
    response = client.post(f"/sessions/{session['session_id']}/process")
    assert response.status_code == 400


def test_full_session_lifecycle_is_honest_about_unimplemented_pipeline(client, sample_clip):
    session = client.post("/sessions", json={"club": "7-iron"}).json()
    sid = session["session_id"]

    with open(sample_clip, "rb") as f:
        resp_a = client.post(f"/sessions/{sid}/upload/camera-a", files={"file": ("a.mp4", f, "video/mp4")})
    assert resp_a.status_code == 200
    assert resp_a.json()["camera_a"]["fps"] == pytest.approx(240.0, rel=0.02)
    assert resp_a.json()["camera_a"]["original_filename"] == "a.mp4"

    with open(sample_clip, "rb") as f:
        resp_b = client.post(f"/sessions/{sid}/upload/camera-b", files={"file": ("b.mp4", f, "video/mp4")})
    assert resp_b.status_code == 200

    process_resp = client.post(f"/sessions/{sid}/process")
    assert process_resp.status_code == 200
    shot = process_resp.json()["shot"]
    assert process_resp.json()["processed_at"] is not None
    assert shot["is_placeholder"] is True
    assert shot["metrics"]["ball_speed_mps"]["source"] == "not_available"
    assert any("calibration" in w.lower() for w in shot["warnings"])

    trajectory = client.get(f"/sessions/{sid}/trajectory")
    assert trajectory.status_code == 200
    assert trajectory.json()["simulated_trajectory"] == []


def test_session_history_keeps_only_newest_result_for_same_upload_names(client, sample_clip):
    def create_processed_session(filename_a, filename_b, replayable=True):
        session = client.post("/sessions", json={}).json()
        session_id = session["session_id"]
        with open(sample_clip, "rb") as video_a:
            response_a = client.post(
                f"/sessions/{session_id}/upload/camera-a",
                files={"file": (filename_a, video_a, "video/mp4")},
            )
        with open(sample_clip, "rb") as video_b:
            response_b = client.post(
                f"/sessions/{session_id}/upload/camera-b",
                files={"file": (filename_b, video_b, "video/mp4")},
            )
        assert response_a.status_code == 200
        assert response_b.status_code == 200
        processed = client.post(f"/sessions/{session_id}/process")
        assert processed.status_code == 200
        if replayable:
            import golfie_api.routers.sessions as sessions_module
            from golfie_core.schemas import TrackedPoint3D

            stored_session = sessions_module.session_store.load(session_id)
            stored_session.shot.simulated_trajectory_3d = [
                TrackedPoint3D(
                    time_seconds=0.0,
                    x_m=0.0,
                    y_m=0.0,
                    z_m=0.0,
                    confidence=1.0,
                )
            ]
            sessions_module.session_store.save(stored_session)
        return session_id

    older_duplicate = create_processed_session("down_driver.mp4", "face_driver.mp4")
    newer_duplicate = create_processed_session("down_driver.mp4", "face_driver.mp4")
    distinct_session = create_processed_session("down_iron.mp4", "face_iron.mp4")
    empty_session = create_processed_session(
        "broken_down.mp4", "broken_face.mp4", replayable=False
    )

    response = client.get("/sessions/history")
    assert response.status_code == 200
    history = response.json()
    assert len(history) == 2
    assert older_duplicate not in {item["session_id"] for item in history}
    assert empty_session not in {item["session_id"] for item in history}

    driver_entry = next(
        item for item in history if item["upload_identifier"] == "down_driver.mp4 + face_driver.mp4"
    )
    assert driver_entry["session_id"] == newer_duplicate
    assert driver_entry["processed_at"]
    assert driver_entry["session_uid"].endswith(
        "__down_driver.mp4 + face_driver.mp4"
    )
    assert distinct_session in {item["session_id"] for item in history}


def test_upload_rejects_unreadable_video(client, tmp_path):
    session = client.post("/sessions", json={}).json()
    sid = session["session_id"]
    bogus = tmp_path / "bogus.mp4"
    bogus.write_text("not a real video")
    with open(bogus, "rb") as f:
        resp = client.post(f"/sessions/{sid}/upload/camera-a", files={"file": ("bogus.mp4", f, "video/mp4")})
    assert resp.status_code == 400


def test_full_session_lifecycle_with_real_pipeline(client, sample_clip, monkeypatch):
    # 1. Create a session
    session = client.post("/sessions", json={"club": "driver"}).json()
    sid = session["session_id"]

    # 2. Upload video clips
    with open(sample_clip, "rb") as f:
        client.post(f"/sessions/{sid}/upload/camera-a", files={"file": ("a.mp4", f, "video/mp4")})
    with open(sample_clip, "rb") as f:
        client.post(f"/sessions/{sid}/upload/camera-b", files={"file": ("b.mp4", f, "video/mp4")})

    # 3. Post a valid calibration result
    calibration_data = {
        "coordinate_system": {
            "origin_in_rig_frame_m": [0.0, 0.0, 0.0],
            "target_direction_in_rig_frame": [1.0, 0.0, 0.0],
            "up_direction_in_rig_frame": [0.0, 0.0, 1.0],
            "alignment_method": "manual"
        },
        "camera_a_intrinsics": [[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0], [0.0, 0.0, 1.0]],
        "camera_b_intrinsics": [[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0], [0.0, 0.0, 1.0]],
        "camera_a_extrinsics": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ],
        "camera_b_extrinsics": [
            [1.0, 0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ],
            "reprojection_error_px": 0.1,
            "confidence": 0.95,
            "measured_baseline_m": 1.0,
            "baseline_m": 1.0,
            "camera_a_calibration_fps": 240.0,
            "camera_b_calibration_fps": 240.0,
            "is_valid": True
    }
    calib_resp = client.post(f"/sessions/{sid}/calibration", json=calibration_data)
    assert calib_resp.status_code == 200

    # 4. Mock 2D tracking to return a clean simulated ball track for both cameras
    from golfie_core.schemas import TrackedPoint2D
    call_count = 0

    def mock_track_ball_2d(candidates, fps):
        nonlocal call_count
        call_count += 1
        # Generate 5 tracked points (frame indices 0 to 4)
        if call_count == 1:
            return [
                TrackedPoint2D(frame_index=i, time_seconds=i / fps, x_px=600.0 + 10.0 * i, y_px=700.0 + 5.0 * i, confidence=0.9)
                for i in range(5)
            ]
        else:
            return [
                TrackedPoint2D(frame_index=i, time_seconds=i / fps, x_px=400.0 + 10.0 * i, y_px=700.0 + 5.0 * i, confidence=0.9)
                for i in range(5)
            ]

    monkeypatch.setattr("golfie_cv.tracking.track_ball_2d", mock_track_ball_2d)

    # 5. Process the session
    process_resp = client.post(f"/sessions/{sid}/process")
    assert process_resp.status_code == 200
    shot = process_resp.json()["shot"]

    # 6. Verify that it ran the real pipeline (is_placeholder=False) and computed metrics
    assert shot["is_placeholder"] is False
    assert len(shot["measured_points_3d"]) == 5
    assert len(shot["simulated_trajectory_3d"]) > 0

    # The metrics must have been estimated successfully (not "not_available")
    assert shot["metrics"]["ball_speed_mps"]["source"] == "estimated"
    assert shot["metrics"]["ball_speed_mps"]["value"] > 0.0
    assert shot["metrics"]["carry_m"]["source"] == "estimated"
    assert shot["metrics"]["carry_m"]["value"] > 0.0

    # 7. Get trajectory payload for Three.js rendering
    traj_resp = client.get(f"/sessions/{sid}/trajectory")
    assert traj_resp.status_code == 200
    traj_body = traj_resp.json()
    assert len(traj_body["simulated_trajectory"]) > 0
    assert traj_body["club"] == "driver"


def test_calibration_endpoints(client, tmp_path, monkeypatch):
    from pathlib import Path
    
    # Isolate calibration directory to prevent reading/writing real active calibration from host disk
    monkeypatch.setattr("golfie_api.routers.calibration.CALIBRATION_DIR", tmp_path)
    monkeypatch.setattr("golfie_api.routers.calibration.ACTIVE_CALIBRATION_PATH", tmp_path / "active_calibration.json")
    monkeypatch.setattr("golfie_api.routers.sessions.CALIBRATION_DIR", tmp_path)
    
    # 1. Active calibration should initially return 404
    resp = client.get("/calibration/active")
    assert resp.status_code == 404

    # 2. Mock intrinsic & stereo calibration to avoid real OpenCV computation on sample test video
    from golfie_core.schemas import CameraIntrinsics, CalibrationResult, CoordinateSystem
    
    mock_intrinsics = CameraIntrinsics(
        fx=1000.0, fy=1000.0, cx=500.0, cy=500.0,
        distortion=[0.0, 0.0, 0.0, 0.0, 0.0],
        image_width=640, image_height=480, reprojection_error_px=0.1
    )
    
    mock_calibration = CalibrationResult(
        coordinate_system=CoordinateSystem(
            origin_in_rig_frame_m=[0.0, 0.0, 0.0],
            target_direction_in_rig_frame=[1.0, 0.0, 0.0],
            up_direction_in_rig_frame=[0.0, 0.0, 1.0],
            alignment_method="manual",
        ),
        camera_a_intrinsics=mock_intrinsics.as_matrix(),
        camera_b_intrinsics=mock_intrinsics.as_matrix(),
        camera_a_extrinsics=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            camera_b_extrinsics=[[1.0, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            stereo_translation_m=[-1.0, 0.0, 0.0],
            baseline_m=1.0,
        reprojection_error_px=0.15,
        confidence=0.92,
        calibration_target="charuco",
        is_valid=True
    )

    monkeypatch.setattr("golfie_api.routers.calibration.calibrate_intrinsics", lambda *args, **kwargs: mock_intrinsics)
    monkeypatch.setattr("golfie_api.routers.calibration.calibrate_stereo", lambda *args, **kwargs: mock_calibration)

    # Mock extract_synced_frames to return dummy paths
    dummy_paths = [Path("dummy_a.png")]
    monkeypatch.setattr(
        "golfie_api.routers.calibration.extract_synced_frames",
        lambda *args, **kwargs: (dummy_paths, dummy_paths, None),
    )

    # Create dummy video files
    dummy_clip = tmp_path / "dummy.mp4"
    dummy_clip.write_text("dummy clip content")

    # 3. Perform calibration upload
    with open(dummy_clip, "rb") as f_a, open(dummy_clip, "rb") as f_b:
        upload_resp = client.post(
            "/calibration/upload",
            files={
                "file_a": ("a.mp4", f_a, "video/mp4"),
                "file_b": ("b.mp4", f_b, "video/mp4"),
            },
            data={
                "board_type": "charuco",
                "grid_cols": "11",
                "grid_rows": "8",
                "square_size": "0.04",
                "marker_size": "0.03",
                "measured_baseline": "2.0",
            }
        )
    assert upload_resp.status_code == 200
    assert upload_resp.json()["baseline_m"] == 2.0
    assert upload_resp.json()["stereo_translation_m"] == [-2.0, 0.0, 0.0]
    assert upload_resp.json()["baseline_source"] == "tape_measured_lens_centres"
    assert upload_resp.json()["reprojection_error_px"] == 0.15

    # 4. Now active calibration should return 200
    active_resp = client.get("/calibration/active")
    assert active_resp.status_code == 200
    assert active_resp.json()["reprojection_error_px"] == 0.15

    # 5. Creating a new session should automatically attach active calibration
    session_resp = client.post("/sessions", json={})
    assert session_resp.status_code == 200
    assert session_resp.json()["calibration"] is not None
    assert session_resp.json()["calibration"]["reprojection_error_px"] == 0.15


def test_camera_upload_with_fps_override(client, sample_clip):
    session = client.post("/sessions", json={}).json()
    sid = session["session_id"]
    with open(sample_clip, "rb") as f:
        resp = client.post(
            f"/sessions/{sid}/upload/camera-a",
            files={"file": ("a.mp4", f, "video/mp4")},
            data={"fps_override": "240.0", "slow_motion_factor": "8"}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["camera_a"]["fps"] == 240.0
    assert body["camera_a"]["slow_motion_factor"] == 8.0


def test_slow_motion_candidate_frames_resample_to_common_physical_clock():
    from golfie_api.pipeline import _resample_frame_sequence

    camera_a, indices_a = _resample_frame_sequence(list(range(9)), 240.0, 120.0)
    camera_b, indices_b = _resample_frame_sequence(list(range(5)), 120.0, 120.0)

    assert camera_a == [0, 2, 4, 6, 8]
    assert indices_a == [0, 2, 4, 6, 8]
    assert camera_b == [0, 1, 2, 3, 4]
    assert indices_b == [0, 1, 2, 3, 4]


def test_mixed_rate_calibration_pairing_uses_snapped_a_timestamp():
    from golfie_api.routers.calibration import _paired_frame_indices

    offset = 6.326757369614512
    idx_a, idx_b = _paired_frame_indices(
        progress_time=1.0,
        start_time_a=offset,
        start_time_b=0.0,
        fps_a=30.0,
        fps_b=120.0,
    )
    aligned_a_time = idx_a / 30.0
    aligned_b_time = idx_b / 120.0 + offset
    assert abs(aligned_a_time - aligned_b_time) <= 0.5 / 120.0


def test_slow_motion_calibration_pairing_uses_common_physical_time():
    from golfie_api.routers.calibration import _slow_motion_frame_indices

    idx_a, idx_b = _slow_motion_frame_indices(
        progress_time=1.0,
        landmark_time_a=10.0,
        landmark_time_b=5.0,
        playback_fps_a=30.0,
        playback_fps_b=30.0,
        slow_motion_factor_a=4.0,
        slow_motion_factor_b=8.0,
    )

    assert idx_a == 420
    assert idx_b == 390
    assert (idx_a - 300) / (30.0 * 4.0) == pytest.approx(1.0)
    assert (idx_b - 150) / (30.0 * 8.0) == pytest.approx(1.0)


def test_calibration_pose_selection_keeps_full_timeline_coverage():
    from golfie_api.routers.calibration import _evenly_spaced_items

    selected = _evenly_spaced_items(list(range(64)), 16)

    assert len(selected) == 16
    assert selected[0] == 0
    assert selected[-1] == 63
    assert len(set(selected)) == 16
    assert max(b - a for a, b in zip(selected, selected[1:])) <= 5


def test_calibration_detector_uses_strongest_shared_board_view(monkeypatch):
    from golfie_api.routers import calibration as calibration_router

    class FakeDetector:
        def __init__(self, name):
            self.name = name

        def getBoard(self):
            return self

        def getChessboardCorners(self):
            return list(range(12))

    counts = {
        ("weak", "a"): 6,
        ("weak", "b"): 6,
        ("strong", "a"): 12,
        ("strong", "b"): 10,
        ("one_sided", "a"): 20,
        ("one_sided", "b"): 2,
    }

    monkeypatch.setattr(
        calibration_router,
        "_detected_corner_count",
        lambda gray, _board_type, detector: counts[(detector.name, gray)],
    )

    detectors = [FakeDetector(name) for name in ("weak", "strong", "one_sided")]
    best = calibration_router._best_shared_detector(
        "a",
        "b",
        "charuco",
        [(detector.name, detector) for detector in detectors],
    )

    assert best == ("strong", detectors[1], 12, 10)


def test_calibration_detector_accepts_partial_one_camera_view(monkeypatch):
    from golfie_api.routers import calibration as calibration_router

    class PartialDetector:
        def getBoard(self):
            return self

        def getChessboardCorners(self):
            return list(range(12))

    detector = PartialDetector()
    monkeypatch.setattr(
        calibration_router,
        "_detected_corner_count",
        lambda gray, _board_type, _detector: 5 if gray == "a" else 1,
    )

    assert calibration_router._best_shared_detector(
        "a", "b", "charuco", [("dictionary=0, grid=4x5", detector)]
    ) == ("dictionary=0, grid=4x5", detector, 5, 1)


def test_board_hint_preserves_detected_dictionary_and_rotation():
    from golfie_api.routers.calibration import _board_hint

    hint = _board_hint("dictionary=0, grid=4x5", "charuco")

    assert hint.dictionary_id == 0
    assert hint.grid_size == (4, 5)


def test_duplicate_calibration_run_is_rejected():
    from fastapi import HTTPException
    from golfie_api.routers.calibration import _exclusive_calibration_run

    first = _exclusive_calibration_run()
    next(first)
    try:
        second = _exclusive_calibration_run()
        with pytest.raises(HTTPException) as exc_info:
            next(second)
        assert exc_info.value.status_code == 409
    finally:
        first.close()
