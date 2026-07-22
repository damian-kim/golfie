"""Session endpoints. See spec section 17 for the target API surface.

Implemented now:
    POST /sessions
    POST /sessions/{id}/upload/camera-a
    POST /sessions/{id}/upload/camera-b
    POST /sessions/{id}/calibration   (manual/no-op for v0)
    POST /sessions/{id}/process       (placeholder pipeline)
    GET  /sessions/{id}/status
    GET  /sessions/{id}/results
    GET  /sessions/{id}/debug/overlays (empty for v0, real from Milestone 3)
    GET  /sessions/{id}/trajectory     (renderer-ready JSON)

Added beyond the spec's literal list, because the frontend needs them:
    GET  /sessions            (list, for a home/review screen)
    GET  /sessions/{id}       (full session detail)
"""

from __future__ import annotations

import shutil
import threading
import json
import hashlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from golfie_core.schemas import CalibrationResult, CameraCapture, Environment, Session
from golfie_cv.video import VideoReadError, read_video_metadata
from golfie_render.threejs import build_trajectory_payload

from golfie_api.pipeline import PipelineError, advance_through_placeholder_stages
from golfie_api.storage import SessionNotFoundError, session_store
from golfie_api.config import CALIBRATION_DIR

router = APIRouter(prefix="/sessions", tags=["sessions"])
_processing_locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
_outline_jobs_lock = threading.Lock()
_outline_jobs: dict[str, dict] = {}
OUTLINE_RENDER_VERSION = 4


def _outline_artifacts_current(session_dir: Path) -> bool:
    required = (
        "camera_a_stripped.mp4", "camera_b_stripped.mp4",
        "camera_a_replay_original.mp4", "camera_b_replay_original.mp4",
    )
    if not all((session_dir / name).exists() and (session_dir / name).stat().st_size > 0 for name in required):
        return False
    marker = session_dir / "outline_render_version.json"
    try:
        return json.loads(marker.read_text(encoding="utf-8")).get("version") == OUTLINE_RENDER_VERSION
    except (OSError, ValueError, AttributeError):
        return False


class CreateSessionRequest(BaseModel):
    environment: Environment = Environment.UNKNOWN
    club: Optional[str] = None
    handedness: Optional[str] = None
    ball_type: Optional[str] = None


class PreviousSessionSummary(BaseModel):
    session_id: str
    session_uid: str
    processed_at: datetime
    camera_a_filename: str
    camera_b_filename: str
    upload_identifier: str
    is_placeholder: bool


def _get_session_or_404(session_id: str) -> Session:
    try:
        return session_store.load(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _outline_worker(session_id: str) -> None:
    try:
        from ultralytics import YOLO
        from golfie_cv.detection import detect_impact_frame_fast, render_stripped_outlines_video

        session = session_store.load(session_id)
        if session.camera_a is None or session.camera_b is None:
            raise RuntimeError("Both camera videos are required for outline rendering.")
        model = YOLO("yolov8n-seg.pt")
        try:
            pose_model = YOLO("yolov8n-pose.pt")
        except Exception as exc:
            # Body segmentation remains available if optional pose weights
            # cannot be downloaded or loaded on this machine.
            print(f"Golfie outline warning: pose model unavailable: {exc}")
            pose_model = None
        session_dir = session_store.session_dir(session_id)
        cameras = (
            ("camera_a", Path(session.camera_a.video_path)),
            ("camera_b", Path(session.camera_b.video_path)),
        )
        for camera_index, (camera_name, video_path) in enumerate(cameras):
            with _outline_jobs_lock:
                _outline_jobs[session_id].update(
                    stage=f"rendering_{camera_name}",
                    message=f"Rendering {camera_name.replace('_', ' ')} swing outlines...",
                    progress=10 + camera_index * 42,
                )
            impact_frame, _ = detect_impact_frame_fast(video_path)
            metadata = read_video_metadata(video_path)
            # Use time-based bounds so 30/120/240 fps phone videos cover the
            # same swing interval and remain comparable in the replay UI.
            start_frame = max(0, impact_frame - round(1.1 * metadata.fps))
            end_frame = min(metadata.frame_count, impact_frame + round(1.4 * metadata.fps))
            final_path = session_dir / f"{camera_name}_stripped.mp4"
            temporary_path = session_dir / f"{camera_name}_stripped.rendering.mp4"
            comparison_path = session_dir / f"{camera_name}_replay_original.mp4"
            comparison_temporary = session_dir / f"{camera_name}_replay_original.rendering.mp4"
            temporary_path.unlink(missing_ok=True)
            comparison_temporary.unlink(missing_ok=True)
            render_stripped_outlines_video(
                video_path=video_path,
                start_frame=start_frame,
                end_frame=end_frame,
                output_path=temporary_path,
                model=model,
                pose_model=pose_model,
                # Eight fresh masks per playback second is smooth enough for a
                # slow-motion body-outline replay and bounds CPU inference cost
                # independently of whether the phone stored 30, 120, or 240fps.
                inference_stride=max(1, round(metadata.fps / 8.0)),
                inference_size=320,
                comparison_output_path=comparison_temporary,
                person_only=True,
            )
            temporary_path.replace(final_path)
            comparison_temporary.replace(comparison_path)
            temporary_map = temporary_path.with_name(f"{temporary_path.stem}_frame_map.json")
            final_map = final_path.with_name(f"{final_path.stem}_frame_map.json")
            if temporary_map.exists():
                temporary_map.replace(final_map)
        (session_dir / "outline_render_version.json").write_text(
            json.dumps({"version": OUTLINE_RENDER_VERSION}), encoding="utf-8"
        )
        with _outline_jobs_lock:
            _outline_jobs[session_id].update(
                running=False,
                stage="complete",
                message="Swing outline videos are ready.",
                progress=100,
                error=None,
            )
    except Exception as exc:
        with _outline_jobs_lock:
            _outline_jobs.setdefault(session_id, {}).update(
                running=False,
                stage="failed",
                message="Swing outline rendering failed.",
                error=str(exc),
            )


def _queue_outline_render(session_id: str) -> dict:
    session_dir = session_store.session_dir(session_id)
    if _outline_artifacts_current(session_dir):
        return {
            "running": False,
            "stage": "complete",
            "message": "Swing outline videos are ready.",
            "progress": 100,
            "error": None,
        }
    with _outline_jobs_lock:
        current = _outline_jobs.get(session_id)
        if current and current.get("running"):
            return dict(current)
        status = {
            "running": True,
            "stage": "queued",
            "message": "Swing outline rendering queued.",
            "progress": 1,
            "error": None,
        }
        _outline_jobs[session_id] = status
    threading.Thread(target=_outline_worker, args=(session_id,), daemon=True).start()
    return dict(status)


@router.post("", response_model=Session)
def create_session(body: CreateSessionRequest = CreateSessionRequest()) -> Session:
    session = session_store.create(**body.model_dump())
    
    # Auto-attach active calibration if it exists
    active_cal_path = CALIBRATION_DIR / "active_calibration.json"
    if active_cal_path.exists():
        try:
            calibration = CalibrationResult.model_validate_json(active_cal_path.read_text())
            session.calibration = calibration
            session_store.save(session)
        except Exception:
            pass
            
    return session


@router.get("", response_model=list[str])
def list_sessions() -> list[str]:
    return session_store.list_ids()


def _capture_filename(capture: CameraCapture | None, fallback: str) -> str:
    if capture is None:
        return fallback
    return capture.original_filename or Path(capture.video_path).name or fallback


def _capture_identity(capture: CameraCapture | None, fallback: str) -> str:
    """Identify legacy uploads by content when their original name was not stored."""
    if capture is None:
        return fallback

    video_path = Path(capture.video_path)
    try:
        size = video_path.stat().st_size
        digest = hashlib.sha256()
        with video_path.open("rb") as video:
            digest.update(video.read(64 * 1024))
            if size > 64 * 1024:
                video.seek(max(0, size - 64 * 1024))
                digest.update(video.read(64 * 1024))
        return f"content:{size}:{digest.hexdigest()}"
    except OSError:
        return f"path:{video_path.name.casefold()}"


@router.get("/history", response_model=list[PreviousSessionSummary])
def list_previous_sessions() -> list[PreviousSessionSummary]:
    """Return newest completed simulation per unique pair of upload filenames."""
    candidates: list[tuple[datetime, Session, str, str]] = []
    for session_id in session_store.list_ids():
        try:
            session = session_store.load(session_id)
        except (SessionNotFoundError, OSError, ValueError):
            continue
        if (
            session.stage.value != "done"
            or session.shot is None
            or not session.shot.simulated_trajectory_3d
        ):
            continue
        processed_at = session.processed_at
        if processed_at is None:
            session_file = session_store.session_dir(session_id) / "session.json"
            try:
                processed_at = datetime.fromtimestamp(
                    session_file.stat().st_mtime, tz=timezone.utc
                )
            except OSError:
                processed_at = session.created_at
        if processed_at.tzinfo is None:
            processed_at = processed_at.replace(tzinfo=timezone.utc)
        camera_a_filename = _capture_filename(session.camera_a, "camera_a")
        camera_b_filename = _capture_filename(session.camera_b, "camera_b")
        candidates.append((processed_at, session, camera_a_filename, camera_b_filename))

    history: list[PreviousSessionSummary] = []
    seen_upload_names: set[tuple[str, str]] = set()
    seen_upload_content: set[tuple[str, str]] = set()
    for processed_at, session, camera_a_filename, camera_b_filename in sorted(
        candidates, key=lambda item: item[0], reverse=True
    ):
        content_key = (
            _capture_identity(session.camera_a, "camera_a"),
            _capture_identity(session.camera_b, "camera_b"),
        )
        name_key = None
        if session.camera_a.original_filename and session.camera_b.original_filename:
            name_key = (
                session.camera_a.original_filename.casefold(),
                session.camera_b.original_filename.casefold(),
            )
        if name_key is not None:
            if name_key in seen_upload_names:
                continue
        elif content_key in seen_upload_content:
            continue
        seen_upload_content.add(content_key)
        if name_key is not None:
            seen_upload_names.add(name_key)
        upload_identifier = f"{camera_a_filename} + {camera_b_filename}"
        timestamp_uid = processed_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        history.append(
            PreviousSessionSummary(
                session_id=session.session_id,
                session_uid=f"{timestamp_uid}__{upload_identifier}",
                processed_at=processed_at,
                camera_a_filename=camera_a_filename,
                camera_b_filename=camera_b_filename,
                upload_identifier=upload_identifier,
                is_placeholder=session.shot.is_placeholder,
            )
        )
    return history


@router.get("/{session_id}", response_model=Session)
def get_session(session_id: str) -> Session:
    return _get_session_or_404(session_id)


async def _save_upload_and_build_capture(
    session_id: str,
    camera_id: str,
    file: UploadFile,
    role_hint: Optional[str],
    device_model: Optional[str],
    fps_override: Optional[float] = None,
    slow_motion_factor: float = 1.0,
) -> CameraCapture:
    session_dir = session_store.session_dir(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "").suffix or ".mp4"
    dest_path = session_dir / f"{camera_id}{suffix}"

    with dest_path.open("wb") as out_file:
        shutil.copyfileobj(file.file, out_file)

    try:
        meta = read_video_metadata(dest_path)
    except VideoReadError as exc:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"Could not read uploaded video for {camera_id}: {exc}",
        ) from exc

    return CameraCapture(
        camera_id=camera_id,
        device_model=device_model,
        fps=fps_override if fps_override is not None else meta.fps,
        resolution=(meta.width, meta.height),
        video_path=str(dest_path),
        original_filename=Path(file.filename or dest_path.name).name,
        slow_motion_factor=slow_motion_factor,
        role_hint=role_hint,
    )


@router.post("/{session_id}/upload/camera-a", response_model=Session)
async def upload_camera_a(
    session_id: str,
    file: UploadFile = File(...),
    role_hint: Optional[str] = Form(default=None),
    device_model: Optional[str] = Form(default=None),
    fps_override: Optional[float] = Form(default=None),
    slow_motion_factor: float = Form(default=1.0, ge=1.0, le=32.0),
) -> Session:
    session = _get_session_or_404(session_id)
    session.camera_a = await _save_upload_and_build_capture(
        session_id, "camera_a", file, role_hint, device_model, fps_override,
        slow_motion_factor,
    )
    session_store.save(session)
    return session


@router.post("/{session_id}/upload/camera-b", response_model=Session)
async def upload_camera_b(
    session_id: str,
    file: UploadFile = File(...),
    role_hint: Optional[str] = Form(default=None),
    device_model: Optional[str] = Form(default=None),
    fps_override: Optional[float] = Form(default=None),
    slow_motion_factor: float = Form(default=1.0, ge=1.0, le=32.0),
) -> Session:
    session = _get_session_or_404(session_id)
    session.camera_b = await _save_upload_and_build_capture(
        session_id, "camera_b", file, role_hint, device_model, fps_override,
        slow_motion_factor,
    )
    session_store.save(session)
    return session



@router.post("/{session_id}/calibration", response_model=Session)
def set_calibration(session_id: str, calibration: CalibrationResult) -> Session:
    """Attach an externally computed or manually reviewed calibration."""
    session = _get_session_or_404(session_id)
    session.calibration = calibration
    session_store.save(session)
    return session


@router.post("/{session_id}/process", response_model=Session)
def process_session(session_id: str) -> Session:
    # React StrictMode intentionally re-runs mount effects in development.
    # Serializing by session makes POST /process idempotent and prevents two
    # ffmpeg/OpenCV pipelines from writing the same artifacts concurrently.
    with _processing_locks[session_id]:
        session = _get_session_or_404(session_id)
        if session.stage.value == "done" and session.shot is not None:
            return session
    
        # Auto-attach active calibration if it is missing and exists
        if session.calibration is None:
            active_cal_path = CALIBRATION_DIR / "active_calibration.json"
            if active_cal_path.exists():
                try:
                    calibration = CalibrationResult.model_validate_json(active_cal_path.read_text())
                    session.calibration = calibration
                except Exception:
                    pass

        try:
            session.error = None
            session = advance_through_placeholder_stages(session)
        except PipelineError as exc:
            from golfie_core.schemas import ProcessingStage

            session.error = str(exc)
            session.stage = ProcessingStage.FAILED
            session_store.save(session)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        session_store.save(session)
        return session


@router.get("/{session_id}/status")
def get_status(session_id: str) -> dict:
    session = _get_session_or_404(session_id)
    return {
        "session_id": session.session_id,
        "stage": session.stage,
        "error": session.error,
    }


@router.get("/{session_id}/results")
def get_results(session_id: str) -> dict:
    session = _get_session_or_404(session_id)
    if session.shot is None:
        raise HTTPException(
            status_code=404,
            detail="This session has not been processed yet. Call POST /sessions/{id}/process first.",
        )
    return session.shot.model_dump(mode="json")


@router.get("/{session_id}/debug/overlays")
def get_debug_overlays(session_id: str) -> dict:
    _get_session_or_404(session_id)
    return {
        "overlays": [],
        "notes": "Debug overlay videos are produced starting at Milestone 3 (ball detection).",
    }


@router.get("/{session_id}/trajectory")
def get_trajectory(session_id: str) -> dict:
    session = _get_session_or_404(session_id)
    if session.shot is None:
        raise HTTPException(
            status_code=404,
            detail="This session has not been processed yet. Call POST /sessions/{id}/process first.",
        )
    return build_trajectory_payload(session.session_id, session.shot, club=session.club)


@router.get("/{session_id}/video/{camera_id}/stripped")
def get_stripped_video(session_id: str, camera_id: str):
    from fastapi.responses import FileResponse
    _get_session_or_404(session_id)
    session_dir = session_store.session_dir(session_id)
    
    # Map camera-a/camera-b or camera_a/camera_b to the correct filename
    cam_name = camera_id.replace("-", "_")
    video_path = session_dir / f"{cam_name}_stripped.mp4"
    
    if not video_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Stripped video not found for {camera_id} in session {session_id}."
        )
    return FileResponse(str(video_path), media_type="video/mp4")


@router.get("/{session_id}/video/{camera_id}/replay-original")
def get_replay_original_video(session_id: str, camera_id: str):
    """Return the browser-compatible original frames matching an outline clip."""
    from fastapi.responses import FileResponse
    _get_session_or_404(session_id)
    cam_name = camera_id.replace("-", "_")
    video_path = session_store.session_dir(session_id) / f"{cam_name}_replay_original.mp4"
    if not video_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Original comparison replay not found for {camera_id} in session {session_id}.",
        )
    return FileResponse(str(video_path), media_type="video/mp4")


@router.get("/{session_id}/video/{camera_id}/frame_map")
def get_frame_map(session_id: str, camera_id: str) -> dict:
    _get_session_or_404(session_id)
    session_dir = session_store.session_dir(session_id)
    cam_name = camera_id.replace("-", "_")
    map_path = session_dir / f"{cam_name}_stripped_frame_map.json"
    
    if not map_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Frame alignment mapping not found for {camera_id} in session {session_id}."
        )
    import json
    try:
        with open(map_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read frame mapping data: {e}"
        )


@router.get("/{session_id}/artifacts")
def get_artifacts(session_id: str) -> dict:
    """Report optional render artifacts separately from shot-processing status."""
    _get_session_or_404(session_id)
    session_dir = session_store.session_dir(session_id)

    def exists(name: str) -> bool:
        path = session_dir / name
        return path.exists() and path.stat().st_size > 0

    outlines = {
        "camera_a": exists("camera_a_stripped.mp4"),
        "camera_b": exists("camera_b_stripped.mp4"),
    }
    originals = {
        "camera_a": exists("camera_a_replay_original.mp4"),
        "camera_b": exists("camera_b_replay_original.mp4"),
    }
    outline_ready = all(outlines.values()) and all(originals.values()) and _outline_artifacts_current(session_dir)
    with _outline_jobs_lock:
        job = dict(_outline_jobs.get(session_id, {}))
    if outline_ready:
        job = {
            "running": False,
            "stage": "complete",
            "message": "Swing outline videos are ready.",
            "progress": 100,
            "error": None,
        }
    elif not job:
        job = {
            "running": False,
            "stage": "idle",
            "message": "Swing outlines have not been generated yet.",
            "progress": 0,
            "error": None,
        }
    return {
        "outline_videos": outlines,
        "outline_original_videos": originals,
        "outline_ready": outline_ready,
        "outline_render_status": job,
        "outline_unavailable_reason": None if outline_ready else (
            job.get("error") or job.get("message")
        ),
        "ball_selection_replays": {
            "camera_a": exists("camera_a_ball_replay.mp4"),
            "camera_b": exists("camera_b_ball_replay.mp4"),
        },
    }


@router.post("/{session_id}/artifacts/outlines")
def generate_outlines(session_id: str) -> dict:
    """Queue bounded, asynchronous YOLO swing-outline rendering."""
    session = _get_session_or_404(session_id)
    if session.camera_a is None or session.camera_b is None:
        raise HTTPException(status_code=409, detail="Upload both camera videos first.")
    return _queue_outline_render(session_id)


@router.get("/{session_id}/logs", response_model=list[str])
def get_session_logs(session_id: str) -> list[str]:
    _get_session_or_404(session_id)
    log_path = session_store.session_dir(session_id) / "processing.log"
    if not log_path.exists():
        return []
    
    session_logs = []
    try:
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                marker = f"Session {session_id}:"
                session_logs.append(line.split(marker, 1)[-1].strip())
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read session logs: {e}"
        )
    return session_logs
