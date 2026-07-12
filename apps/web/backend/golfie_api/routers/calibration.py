from __future__ import annotations

import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
_status_lock = threading.Lock()
_calibration_status = {
    "running": False,
    "progress": 0,
    "stage": "idle",
    "message": "No calibration is running.",
}


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
    video_path_a: Path, video_path_b: Path, max_frames: int = 18
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

def log_calibration_progress(
    msg: str, progress: int | None = None, stage: str | None = None
):
    with _status_lock:
        if "Starting calibration" in msg:
            _calibration_status.update(running=True, progress=1, stage="starting")
        if progress is not None:
            # Worker threads may finish in either order; progress must never
            # move backwards.
            _calibration_status["progress"] = max(
                int(_calibration_status["progress"]), int(progress)
            )
        if stage is not None:
            _calibration_status["stage"] = stage
        _calibration_status["message"] = msg
        if "completed successfully" in msg:
            _calibration_status.update(running=False, progress=100, stage="complete")
        elif "failed" in msg.lower():
            _calibration_status.update(running=False, stage="failed")
    try:
        # Clear log if it's the start of calibration
        mode = "w" if "Starting" in msg else "a"
        with open(CALIBRATION_LOG_PATH, mode, encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat()}] {msg}\n")
    except Exception:
        pass


@router.get("/status")
def get_calibration_status() -> dict:
    with _status_lock:
        return dict(_calibration_status)


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
    measured_baseline: float | None = Form(default=None),
) -> CalibrationResult:
    log_calibration_progress("Starting calibration process...", 1, "starting")
    log_calibration_progress(
        f"Board geometry: type={board_type}, grid={grid_cols}x{grid_rows}, "
        f"square={square_size * 1000.0:.2f}mm, marker={marker_size * 1000.0:.2f}mm."
    )
    if measured_baseline is not None:
        if not 0.1 <= measured_baseline <= 10.0:
            raise HTTPException(
                status_code=400,
                detail="Measured lens-to-lens baseline must be between 0.1m and 10m.",
            )
        log_calibration_progress(
            f"Measured camera lens-to-lens baseline: {measured_baseline:.4f}m."
        )
    temp_dir = Path(tempfile.mkdtemp(prefix="golfie_cal_"))
    try:
        path_a = temp_dir / f"cal_a{Path(file_a.filename or '').suffix or '.mp4'}"
        path_b = temp_dir / f"cal_b{Path(file_b.filename or '').suffix or '.mp4'}"

        log_calibration_progress(
            "Writing uploaded camera videos to temporary storage...", 4, "uploading"
        )
        with path_a.open("wb") as f:
            shutil.copyfileobj(file_a.file, f)
        with path_b.open("wb") as f:
            shutil.copyfileobj(file_b.file, f)

        # Extract matching frames
        log_calibration_progress(
            "Synchronizing videos and extracting calibration frames...", 9, "extracting"
        )
        try:
            paths_a, paths_b = extract_synced_frames(path_a, path_b)
            log_calibration_progress(
                f"Extracted {len(paths_a)} synchronized frame pairs.", 30, "frames_ready"
            )
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

        # The two intrinsic solves are independent and OpenCV releases the GIL,
        # so run them together instead of making the user wait for A then B.
        log_calibration_progress(
            "Calibrating both camera lens models in parallel...", 32, "intrinsics"
        )

        def solve_intrinsics(label: str, paths: list[Path]):
            result = calibrate_intrinsics(
                paths,
                board_type=board_type,
                grid_size=(grid_cols, grid_rows),
                square_length=square_size,
                marker_length=marker_size,
            )
            return label, result

        intrinsic_results = {}
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(solve_intrinsics, "Camera A", paths_a),
                    executor.submit(solve_intrinsics, "Camera B", paths_b),
                ]
                for completed_count, future in enumerate(as_completed(futures), start=1):
                    label, result = future.result()
                    intrinsic_results[label] = result
                    log_calibration_progress(
                        f"{label} intrinsics complete: RMS={result.reprojection_error_px:.3f}px, "
                        f"size={result.image_width}x{result.image_height}.",
                        32 + completed_count * 13,
                        "intrinsics",
                    )
        except Exception as exc:
            log_calibration_progress(f"Camera intrinsics calibration failed: {exc}")
            raise HTTPException(
                status_code=400,
                detail=f"Camera intrinsics calibration failed: {exc}",
            )
        intrinsics_a = intrinsic_results["Camera A"]
        intrinsics_b = intrinsic_results["Camera B"]

        # Run Stereo Extrinsics
        log_calibration_progress(
            "Solving stereo camera pose and validating held-out geometry...", 60, "stereo"
        )
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
                    f"baseline={stereo_result.baseline_m:.3f}m.",
                    92,
                    "validating",
                )
            if not stereo_result.is_valid:
                reasons = stereo_result.validation_warnings or [
                    "Stereo geometry did not meet Golfie's validation thresholds."
                ]
                raise ValueError(" ".join(reasons))

            # A planar board determines scale in principle, but phone-mode
            # intrinsics and limited pose coverage can bias translation while
            # retaining a deceptively small pixel RMS.  A tape-measured optical
            # centre separation is a stronger metric-scale constraint.
            if measured_baseline is not None:
                solved_baseline = float(stereo_result.baseline_m or 0.0)
                if solved_baseline <= 0.0:
                    raise ValueError("Stereo solve did not produce a usable camera baseline.")
                scale = measured_baseline / solved_baseline
                translation = np.asarray(stereo_result.stereo_translation_m, dtype=np.float64) * scale
                stereo_result.stereo_translation_m = translation.tolist()
                camera_b_extrinsics = np.asarray(
                    stereo_result.camera_b_extrinsics, dtype=np.float64
                )
                camera_b_extrinsics[:3, 3] = translation
                stereo_result.camera_b_extrinsics = camera_b_extrinsics.tolist()
                stereo_result.baseline_m = float(measured_baseline)
                stereo_result.measured_baseline_m = float(measured_baseline)
                stereo_result.baseline_source = "tape_measured_lens_centres"
                log_calibration_progress(
                    f"Applied measured-baseline scale correction {scale:.4f}x "
                    f"(board solve {solved_baseline:.4f}m -> {measured_baseline:.4f}m)."
                )

            # Do not perform the expensive full per-frame timing scan here:
            # calibration geometry does not consume it, and it added ~26s
            # after the stereo solve had already finished.  Persist the cheap
            # container rates for diagnostics instead.
            cap_a = cv2.VideoCapture(str(path_a))
            cap_b = cv2.VideoCapture(str(path_b))
            try:
                fps_a = float(cap_a.get(cv2.CAP_PROP_FPS))
                fps_b = float(cap_b.get(cv2.CAP_PROP_FPS))
                stereo_result.camera_a_calibration_fps = fps_a if fps_a > 0 else None
                stereo_result.camera_b_calibration_fps = fps_b if fps_b > 0 else None
            finally:
                cap_a.release()
                cap_b.release()
            log_calibration_progress(
                "Recorded calibration container rates.", 95, "metadata"
            )
        except Exception as exc:
            log_calibration_progress(f"Stereo extrinsic calibration failed: {exc}")
            raise HTTPException(
                status_code=400,
                detail=f"Stereo extrinsic calibration failed: {exc}",
            )

        # Persist results
        log_calibration_progress(
            "Saving validated active calibration result to disk...", 98, "saving"
        )
        ACTIVE_CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        ACTIVE_CALIBRATION_PATH.write_text(stereo_result.model_dump_json(indent=2))
        log_calibration_progress("Rig calibration completed successfully!")
        return stereo_result

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
