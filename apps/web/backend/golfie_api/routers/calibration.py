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
from golfie_api.config import CALIBRATION_DIR, CALIBRATION_LOG_PATH

router = APIRouter(prefix="/calibration", tags=["calibration"])

ACTIVE_CALIBRATION_PATH = CALIBRATION_DIR / "active_calibration.json"


def _paired_frame_indices(
    progress_time: float,
    start_time_a: float,
    start_time_b: float,
    fps_a: float,
    fps_b: float,
) -> tuple[int, int]:
    """Pair B to the timestamp of A's actual decoded frame."""
    idx_a = int(round((start_time_a + progress_time) * fps_a))
    actual_progress_time = (idx_a / fps_a) - start_time_a
    idx_b = int(round((start_time_b + actual_progress_time) * fps_b))
    return idx_a, idx_b


def extract_synced_frames(
    video_path_a: Path, video_path_b: Path, max_frames: int = 25
) -> tuple[list[Path], list[Path]]:
    """Sync two videos and extract matching frames for calibration."""
    from golfie_cv.sync import estimate_sync_offset

    cap_a = cv2.VideoCapture(str(video_path_a))
    cap_b = cv2.VideoCapture(str(video_path_b))

    total_a = int(cap_a.get(cv2.CAP_PROP_FRAME_COUNT))
    total_b = int(cap_b.get(cv2.CAP_PROP_FRAME_COUNT))

    fps_a = cap_a.get(cv2.CAP_PROP_FPS)
    fps_b = cap_b.get(cv2.CAP_PROP_FPS)
    
    if fps_a <= 0: fps_a = 30.0
    if fps_b <= 0: fps_b = 30.0

    duration_a = total_a / fps_a
    duration_b = total_b / fps_b

    # Determine sync offset in seconds
    try:
        sync = estimate_sync_offset(video_path_a, video_path_b)
        candidate_offset = sync.offset_seconds
        
        # Verify that the offset produces a valid overlapping window
        s_a = max(0.0, candidate_offset)
        s_b = max(0.0, -candidate_offset)
        overlap = min(duration_a - s_a, duration_b - s_b)
        
        if sync.method.value == "manual":
            raise ValueError(sync.notes or "Audio could not be extracted from one or both videos.")
        if sync.confidence >= 0.3 and overlap > 1.0:
            offset_sec = candidate_offset
            log_calibration_progress(
                f"Audio sync accepted: confidence={sync.confidence:.2f}, "
                f"offset={offset_sec:.6f}s, Camera A={fps_a:.3f}fps, Camera B={fps_b:.3f}fps."
            )
        else:
            raise ValueError(
                "Calibration audio was extracted, but synchronization was rejected: "
                f"confidence={sync.confidence:.2f}, candidate_offset={candidate_offset:.3f}s, "
                f"overlap={overlap:.2f}s. The clap must be the strongest isolated transient "
                "in both recordings."
            )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"Calibration audio synchronization failed: {exc}"
        ) from exc

    # Overlap regions in seconds
    start_time_a = max(0.0, offset_sec)
    start_time_b = max(0.0, -offset_sec)

    overlap_duration = min(duration_a - start_time_a, duration_b - start_time_b)
    if overlap_duration <= 0:
        start_time_a = 0.0
        start_time_b = 0.0
        overlap_duration = min(duration_a, duration_b)

    time_step = overlap_duration / max_frames

    # Subdirectories within temp directory
    tmp_dir_a = video_path_a.parent / "_tmp_cal_a"
    tmp_dir_b = video_path_b.parent / "_tmp_cal_b"
    tmp_dir_a.mkdir(exist_ok=True)
    tmp_dir_b.mkdir(exist_ok=True)

    paths_a = []
    paths_b = []

    saved = 0
    for i in range(max_frames):
        t = i * time_step
        # Camera A and B can have very different rates (e.g. 30 vs 120 fps).
        # Pair B to A's snapped frame timestamp, not the unsnapped ideal time.
        idx_a, idx_b = _paired_frame_indices(t, start_time_a, start_time_b, fps_a, fps_b)

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


from datetime import datetime

def log_calibration_progress(msg: str):
    try:
        # Clear log if it's the start of calibration
        mode = "w" if "Starting" in msg else "a"
        with open(CALIBRATION_LOG_PATH, mode, encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat()}] {msg}\n")
    except Exception:
        pass


@router.get("/logs", response_model=list[str])
def get_calibration_logs() -> list[str]:
    if not CALIBRATION_LOG_PATH.exists():
        return []
    
    logs = []
    try:
        with CALIBRATION_LOG_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.split("] ", 1)
                if len(parts) == 2:
                    logs.append(parts[1].strip())
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read calibration logs: {e}"
        )
    return logs


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
    log_calibration_progress("Starting calibration process...")
    temp_dir = Path(tempfile.mkdtemp(prefix="golfie_cal_"))
    try:
        path_a = temp_dir / f"cal_a{Path(file_a.filename or '').suffix or '.mp4'}"
        path_b = temp_dir / f"cal_b{Path(file_b.filename or '').suffix or '.mp4'}"

        log_calibration_progress("Writing uploaded camera videos to temporary storage...")
        with path_a.open("wb") as f:
            shutil.copyfileobj(file_a.file, f)
        with path_b.open("wb") as f:
            shutil.copyfileobj(file_b.file, f)

        # Extract matching frames
        log_calibration_progress("Extracting synced calibration frames from videos...")
        try:
            paths_a, paths_b = extract_synced_frames(path_a, path_b)
        except Exception as exc:
            log_calibration_progress(f"Failed to read or sync calibration videos: {exc}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to read or sync calibration videos: {exc}",
            )

        if not paths_a or not paths_b:
            log_calibration_progress("Could not extract aligned frames from the calibration videos.")
            raise HTTPException(
                status_code=400,
                detail="Could not extract aligned frames from the calibration videos.",
            )

        # Calibrate Camera A
        log_calibration_progress("Calibrating Camera A intrinsic lens parameters...")
        try:
            intrinsics_a = calibrate_intrinsics(
                paths_a,
                board_type=board_type,
                grid_size=(grid_cols, grid_rows),
                square_length=square_size,
                marker_length=marker_size,
            )
            log_calibration_progress(
                f"Camera A intrinsics: RMS={intrinsics_a.reprojection_error_px:.3f}px, "
                f"size={intrinsics_a.image_width}x{intrinsics_a.image_height}."
            )
        except Exception as exc:
            log_calibration_progress(f"Camera A intrinsics calibration failed: {exc}")
            raise HTTPException(
                status_code=400,
                detail=f"Camera A intrinsics calibration failed: {exc}",
            )

        # Calibrate Camera B
        log_calibration_progress("Calibrating Camera B intrinsic lens parameters...")
        try:
            intrinsics_b = calibrate_intrinsics(
                paths_b,
                board_type=board_type,
                grid_size=(grid_cols, grid_rows),
                square_length=square_size,
                marker_length=marker_size,
            )
            log_calibration_progress(
                f"Camera B intrinsics: RMS={intrinsics_b.reprojection_error_px:.3f}px, "
                f"size={intrinsics_b.image_width}x{intrinsics_b.image_height}."
            )
        except Exception as exc:
            log_calibration_progress(f"Camera B intrinsics calibration failed: {exc}")
            raise HTTPException(
                status_code=400,
                detail=f"Camera B intrinsics calibration failed: {exc}",
            )

        # Run Stereo Extrinsics
        log_calibration_progress("Performing stereo extrinsic relative camera pose estimation...")
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
            if all(
                value is not None
                for value in (
                    stereo_result.epipolar_error_median_px,
                    stereo_result.epipolar_error_p95_px,
                    stereo_result.epipolar_inlier_ratio,
                    stereo_result.baseline_m,
                )
            ):
                log_calibration_progress(
                    "Stereo diagnostics: "
                    f"RMS={stereo_result.reprojection_error_px:.3f}px, "
                    f"epipolar median={stereo_result.epipolar_error_median_px:.3f}px, "
                    f"p95={stereo_result.epipolar_error_p95_px:.3f}px, "
                    f"inliers={stereo_result.epipolar_inlier_ratio * 100:.1f}%, "
                    f"baseline={stereo_result.baseline_m:.3f}m."
                )
            if not stereo_result.is_valid:
                reasons = stereo_result.validation_warnings or [
                    "Stereo geometry did not meet Golfie's validation thresholds."
                ]
                raise ValueError(" ".join(reasons))
        except Exception as exc:
            log_calibration_progress(f"Stereo extrinsic calibration failed: {exc}")
            raise HTTPException(
                status_code=400,
                detail=f"Stereo extrinsic calibration failed: {exc}",
            )

        # Persist results
        log_calibration_progress("Saving validated active calibration result to disk...")
        ACTIVE_CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        ACTIVE_CALIBRATION_PATH.write_text(stereo_result.model_dump_json(indent=2))
        log_calibration_progress("Rig calibration completed successfully!")
        return stereo_result

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
