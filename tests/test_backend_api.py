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
    import golfie_api.storage as storage_pkg
    import golfie_api.storage.session_store  # noqa: F401 (ensures it's imported)
    from golfie_api import main as main_module
    from golfie_api.storage.session_store import SessionStore

    session_store_module = sys.modules["golfie_api.storage.session_store"]

    isolated_store = SessionStore(tmp_path / "shots")
    monkeypatch.setattr(session_store_module, "session_store", isolated_store)
    monkeypatch.setattr(storage_pkg, "session_store", isolated_store)
    monkeypatch.setattr(sessions_module, "session_store", isolated_store)

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

    with open(sample_clip, "rb") as f:
        resp_b = client.post(f"/sessions/{sid}/upload/camera-b", files={"file": ("b.mp4", f, "video/mp4")})
    assert resp_b.status_code == 200

    process_resp = client.post(f"/sessions/{sid}/process")
    assert process_resp.status_code == 200
    shot = process_resp.json()["shot"]
    assert shot["is_placeholder"] is True
    assert shot["metrics"]["ball_speed_mps"]["source"] == "not_available"
    assert any("calibration" in w.lower() for w in shot["warnings"])

    trajectory = client.get(f"/sessions/{sid}/trajectory")
    assert trajectory.status_code == 200
    assert trajectory.json()["simulated_trajectory"] == []


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

