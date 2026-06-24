from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from golfie_core.schemas import CalibrationResult, CameraIntrinsics
from golfie_cv.calibration import calibrate_intrinsics, calibrate_stereo
from golfie_api.config import CALIBRATION_DIR

router = APIRouter(prefix="/calibration", tags=["calibration"])

ACTIVE_CALIBRATION_PATH = CALIBRATION_DIR / "active_calibration.json"


def extract_synced_frames(
    video_path_a: Path, video_path_b: Path, max_frames: int = 25
) -> tuple[list[Path], list[Path]]:
    """Sync two videos and extract matching frames for calibration."""
    from golfie_cv.sync import estimate_sync_offset

    # Determine sync offset
    try:
        sync = estimate_sync_offset(video_path_a, video_path_b)
        offset_frames = int(round(sync.offset_frames))
    except Exception:
        offset_frames = 0

    cap_a = cv2.VideoCapture(str(video_path_a))
    cap_b = cv2.VideoCapture(str(video_path_b))

    total_a = int(cap_a.get(cv2.CAP_PROP_FRAME_COUNT))
    total_b = int(cap_b.get(cv2.CAP_PROP_FRAME_COUNT))

    # Overlap regions
    start_a = max(0, -offset_frames)
    start_b = max(0, offset_frames)

    overlap_len = min(total_a - start_a, total_b - start_b)
    if overlap_len <= 0:
        start_a = 0
        start_b = 0
        overlap_len = min(total_a, total_b)

    step = max(1, overlap_len // max_frames)

    # Subdirectories within temp directory
    tmp_dir_a = video_path_a.parent / "_tmp_cal_a"
    tmp_dir_b = video_path_b.parent / "_tmp_cal_b"
    tmp_dir_a.mkdir(exist_ok=True)
    tmp_dir_b.mkdir(exist_ok=True)

    paths_a = []
    paths_b = []

    saved = 0
    for i in range(max_frames):
        idx_a = start_a + i * step
        idx_b = start_b + i * step

        if idx_a >= total_a or idx_b >= total_b:
            break

        cap_a.set(cv2.CAP_PROP_POS_FRAMES, idx_a)
        ret_a, frame_a = cap_a.read()

        cap_b.set(cv2.CAP_PROP_POS_FRAMES, idx_b)
        ret_b, frame_b = cap_b.read()

        if ret_a and ret_b and frame_a is not None and frame_b is not None:
            path_a = tmp_dir_a / f"frame_{saved:04d}.png"
            path_b = tmp_dir_b / f"frame_{saved:04d}.png"
            cv2.imwrite(str(path_a), frame_a)
            cv2.imwrite(str(path_b), frame_b)
            paths_a.append(path_a)
            paths_b.append(path_b)
            saved += 1

    cap_a.release()
    cap_b.release()
    return paths_a, paths_b


@router.get("/active", response_model=CalibrationResult)
def get_active_calibration() -> CalibrationResult:
    if not ACTIVE_CALIBRATION_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No active camera calibration found. Please calibrate your cameras first.",
        )
    try:
        return CalibrationResult.model_validate_json(ACTIVE_CALIBRATION_PATH.read_text())
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading active calibration: {exc}",
        )


@router.post("/upload", response_model=CalibrationResult)
async def upload_and_calibrate(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
    board_type: str = Form(default="charuco"),
    grid_cols: int = Form(default=11),
    grid_rows: int = Form(default=8),
    square_size: float = Form(default=0.04),
    marker_size: float = Form(default=0.03),
) -> CalibrationResult:
    temp_dir = Path(tempfile.mkdtemp(prefix="golfie_cal_"))
    try:
        path_a = temp_dir / f"cal_a{Path(file_a.filename or '').suffix or '.mp4'}"
        path_b = temp_dir / f"cal_b{Path(file_b.filename or '').suffix or '.mp4'}"

        with path_a.open("wb") as f:
            shutil.copyfileobj(file_a.file, f)
        with path_b.open("wb") as f:
            shutil.copyfileobj(file_b.file, f)

        # Extract matching frames
        try:
            paths_a, paths_b = extract_synced_frames(path_a, path_b)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to read or sync calibration videos: {exc}",
            )

        if not paths_a or not paths_b:
            raise HTTPException(
                status_code=400,
                detail="Could not extract aligned frames from the calibration videos.",
            )

        # Calibrate Camera A
        try:
            intrinsics_a = calibrate_intrinsics(
                paths_a,
                board_type=board_type,
                grid_size=(grid_cols, grid_rows),
                square_length=square_size,
                marker_length=marker_size,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Camera A intrinsics calibration failed: {exc}",
            )

        # Calibrate Camera B
        try:
            intrinsics_b = calibrate_intrinsics(
                paths_b,
                board_type=board_type,
                grid_size=(grid_cols, grid_rows),
                square_length=square_size,
                marker_length=marker_size,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Camera B intrinsics calibration failed: {exc}",
            )

        # Run Stereo Extrinsics
        try:
            stereo_result = calibrate_stereo(
                intrinsics_a,
                intrinsics_b,
                paths_a,
                paths_b,
                board_type=board_type,
                grid_size=(grid_cols, grid_rows),
                square_length=square_size,
                marker_length=marker_size,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Stereo extrinsic calibration failed: {exc}",
            )

        # Persist results
        ACTIVE_CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        ACTIVE_CALIBRATION_PATH.write_text(stereo_result.model_dump_json(indent=2))
        return stereo_result

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
