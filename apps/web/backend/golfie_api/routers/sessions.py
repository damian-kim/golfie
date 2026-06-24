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


class CreateSessionRequest(BaseModel):
    environment: Environment = Environment.UNKNOWN
    club: Optional[str] = None
    handedness: Optional[str] = None
    ball_type: Optional[str] = None


def _get_session_or_404(session_id: str) -> Session:
    try:
        return session_store.load(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


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


@router.get("/{session_id}", response_model=Session)
def get_session(session_id: str) -> Session:
    return _get_session_or_404(session_id)


async def _save_upload_and_build_capture(
    session_id: str, camera_id: str, file: UploadFile, role_hint: Optional[str], device_model: Optional[str]
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
        fps=meta.fps,
        resolution=(meta.width, meta.height),
        video_path=str(dest_path),
        role_hint=role_hint,
    )


@router.post("/{session_id}/upload/camera-a", response_model=Session)
async def upload_camera_a(
    session_id: str,
    file: UploadFile = File(...),
    role_hint: Optional[str] = Form(default=None),
    device_model: Optional[str] = Form(default=None),
) -> Session:
    session = _get_session_or_404(session_id)
    session.camera_a = await _save_upload_and_build_capture(
        session_id, "camera_a", file, role_hint, device_model
    )
    session_store.save(session)
    return session


@router.post("/{session_id}/upload/camera-b", response_model=Session)
async def upload_camera_b(
    session_id: str,
    file: UploadFile = File(...),
    role_hint: Optional[str] = Form(default=None),
    device_model: Optional[str] = Form(default=None),
) -> Session:
    session = _get_session_or_404(session_id)
    session.camera_b = await _save_upload_and_build_capture(
        session_id, "camera_b", file, role_hint, device_model
    )
    session_store.save(session)
    return session


@router.post("/{session_id}/calibration", response_model=Session)
def set_calibration(session_id: str, calibration: CalibrationResult) -> Session:
    """For v0 this just stores whatever CalibrationResult the client sends
    (e.g. a manually-entered or externally-computed one via
    scripts/calibrate_cameras.py). Real automatic calibration is
    Milestone 1; golfie_cv.calibration currently raises NotImplementedError
    if called directly.
    """
    session = _get_session_or_404(session_id)
    session.calibration = calibration
    session_store.save(session)
    return session


@router.post("/{session_id}/process", response_model=Session)
def process_session(session_id: str) -> Session:
    session = _get_session_or_404(session_id)
    
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
        session = advance_through_placeholder_stages(session)
    except PipelineError as exc:
        session.error = str(exc)
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
