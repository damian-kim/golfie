from __future__ import annotations

import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple, Sequence

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from golfie_core.schemas import CalibrationResult, CameraIntrinsics
from golfie_cv.calibration import calibrate_intrinsics, calibrate_stereo
from golfie_api.config import CALIBRATION_DIR, CALIBRATION_LOG_PATH

router = APIRouter(prefix="/calibration", tags=["calibration"])

ACTIVE_CALIBRATION_PATH = CALIBRATION_DIR / "active_calibration.json"
_status_lock = threading.Lock()
_calibration_run_lock = threading.Lock()
_calibration_status = {
    "running": False,
    "progress": 0,
    "stage": "idle",
    "message": "No calibration is running.",
}

MIN_CALIBRATION_CORNERS = 4
CALIBRATION_SCAN_MULTIPLIER = 2
BOARD_SCAN_MAX_WIDTH = 800


class BoardDetectionHint(NamedTuple):
    dictionary_id: int | None
    grid_size: tuple[int, int]
    description: str


def _exclusive_calibration_run():
    """Reject duplicate uploads instead of running CPU-heavy solves together."""
    if not _calibration_run_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="A camera calibration is already running. Wait for it to finish before retrying.",
        )
    try:
        yield
    finally:
        _calibration_run_lock.release()


def _evenly_spaced_items(items: Sequence, limit: int) -> list:
    """Keep ``limit`` items while retaining the first/last available poses."""
    values = list(items)
    if limit <= 0:
        return []
    if len(values) <= limit:
        return values
    if limit == 1:
        return [values[len(values) // 2]]
    indices = [
        round(position * (len(values) - 1) / (limit - 1))
        for position in range(limit)
    ]
    return [values[index] for index in indices]


def _charuco_detector_candidates(
    grid_size: tuple[int, int],
    square_length: float,
    marker_length: float,
    dictionary_ids: Sequence[int] | None = None,
) -> list[tuple[str, cv2.aruco.CharucoDetector]]:
    """Build the same dictionary/orientation fallbacks used by calibration."""
    fallback_dictionary_ids = [
        cv2.aruco.DICT_4X4_50,
        cv2.aruco.DICT_6X6_250,
        cv2.aruco.DICT_4X4_100,
        cv2.aruco.DICT_4X4_250,
        cv2.aruco.DICT_4X4_1000,
        cv2.aruco.DICT_5X5_50,
        cv2.aruco.DICT_5X5_100,
        cv2.aruco.DICT_5X5_250,
        cv2.aruco.DICT_5X5_1000,
        cv2.aruco.DICT_6X6_50,
        cv2.aruco.DICT_6X6_100,
        cv2.aruco.DICT_6X6_1000,
        cv2.aruco.DICT_7X7_50,
        cv2.aruco.DICT_7X7_100,
        cv2.aruco.DICT_7X7_250,
        cv2.aruco.DICT_7X7_1000,
        cv2.aruco.DICT_ARUCO_ORIGINAL,
    ]
    dictionary_ids = list(dictionary_ids or fallback_dictionary_ids)
    orientations = [grid_size]
    if grid_size[0] != grid_size[1]:
        orientations.append((grid_size[1], grid_size[0]))
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.minMarkerPerimeterRate = 0.005
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    candidates = []
    for dictionary_id in dict.fromkeys(dictionary_ids):
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        for orientation in orientations:
            board = cv2.aruco.CharucoBoard(
                orientation, square_length, marker_length, dictionary
            )
            candidates.append(
                (
                    f"dictionary={dictionary_id}, grid={orientation[0]}x{orientation[1]}",
                    cv2.aruco.CharucoDetector(
                        board, detectorParams=detector_params
                    ),
                )
            )
    return candidates


def _find_charuco_dictionary(
    gray_a: np.ndarray, gray_b: np.ndarray
) -> tuple[int, int, int] | None:
    """Identify the marker family cheaply before ChArUco interpolation."""
    dictionary_ids = [
        cv2.aruco.DICT_4X4_50,
        cv2.aruco.DICT_6X6_250,
        cv2.aruco.DICT_4X4_100,
        cv2.aruco.DICT_4X4_250,
        cv2.aruco.DICT_4X4_1000,
        cv2.aruco.DICT_5X5_50,
        cv2.aruco.DICT_5X5_100,
        cv2.aruco.DICT_5X5_250,
        cv2.aruco.DICT_5X5_1000,
        cv2.aruco.DICT_6X6_50,
        cv2.aruco.DICT_6X6_100,
        cv2.aruco.DICT_6X6_1000,
        cv2.aruco.DICT_7X7_50,
        cv2.aruco.DICT_7X7_100,
        cv2.aruco.DICT_7X7_250,
        cv2.aruco.DICT_7X7_1000,
        cv2.aruco.DICT_ARUCO_ORIGINAL,
    ]
    params = cv2.aruco.DetectorParameters()
    params.minMarkerPerimeterRate = 0.005
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    for dictionary_id in dict.fromkeys(dictionary_ids):
        detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(dictionary_id), params
        )
        _, ids_a, _ = detector.detectMarkers(gray_a)
        _, ids_b, _ = detector.detectMarkers(gray_b)
        markers_a = 0 if ids_a is None else len(ids_a)
        markers_b = 0 if ids_b is None else len(ids_b)
        if max(markers_a, markers_b) >= 2:
            return dictionary_id, markers_a, markers_b
    return None


def _pattern_detector_candidates(
    board_type: str,
    grid_size: tuple[int, int],
    square_length: float,
    marker_length: float,
) -> list[tuple[str, object]]:
    if board_type == "charuco":
        return _charuco_detector_candidates(
            grid_size, square_length, marker_length
        )
    if board_type == "chessboard":
        pattern = (grid_size[0] - 1, grid_size[1] - 1)
        patterns = [pattern]
        if pattern[0] != pattern[1]:
            patterns.append((pattern[1], pattern[0]))
        return [(f"grid={cols}x{rows}", (cols, rows)) for cols, rows in patterns]
    raise ValueError(f"Unknown board type: {board_type}")


def _detected_corner_count(gray: np.ndarray, board_type: str, detector: object) -> int:
    if board_type == "charuco":
        corners, _, _, _ = detector.detectBoard(gray)
        return 0 if corners is None else len(corners)
    found, corners = cv2.findChessboardCorners(gray, detector, None)
    return len(corners) if found and corners is not None else 0


def _detection_thumbnail(frame: np.ndarray) -> np.ndarray:
    """Downscale a phone frame for fast board-presence scoring."""
    height, width = frame.shape[:2]
    if width > BOARD_SCAN_MAX_WIDTH:
        scale = BOARD_SCAN_MAX_WIDTH / width
        frame = cv2.resize(
            frame,
            (BOARD_SCAN_MAX_WIDTH, max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _board_hint(description: str, board_type: str) -> BoardDetectionHint:
    grid_text = description.rsplit("grid=", 1)[1]
    cols_text, rows_text = grid_text.split("x", 1)
    dictionary_id = None
    if board_type == "charuco":
        dictionary_id = int(description.split("dictionary=", 1)[1].split(",", 1)[0])
    return BoardDetectionHint(
        dictionary_id=dictionary_id,
        grid_size=(int(cols_text), int(rows_text)),
        description=description,
    )


def _best_shared_detector(
    gray_a: np.ndarray,
    gray_b: np.ndarray,
    board_type: str,
    candidates: Sequence[tuple[str, object]],
) -> tuple[str, object, int, int] | None:
    """Find the strongest board interpretation visible in either camera."""
    best = None
    best_score = None
    for description, detector in candidates:
        corners_a = _detected_corner_count(gray_a, board_type, detector)
        corners_b = _detected_corner_count(gray_b, board_type, detector)
        if max(corners_a, corners_b) < MIN_CALIBRATION_CORNERS:
            continue
        # Prefer an interpretation supported by both views, but allow a
        # one-sided partial board to lock the known physical target. Intrinsic
        # calibration can use that view and nearby visual-sync shifts may make
        # it a shared stereo pose later.
        score = (
            min(corners_a, corners_b) >= MIN_CALIBRATION_CORNERS,
            min(corners_a, corners_b),
            max(corners_a, corners_b),
            corners_a + corners_b,
        )
        if best_score is None or score > best_score:
            best_score = score
            best = (description, detector, corners_a, corners_b)
        if board_type == "charuco":
            maximum_corners = len(detector.getBoard().getChessboardCorners())
        else:
            maximum_corners = detector[0] * detector[1]
        if max(corners_a, corners_b) == maximum_corners:
            # A complete detection in either camera is enough to identify the
            # physical board. No later fallback can improve that view.
            return best
    return best

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


def _slow_motion_frame_indices(
    progress_time: float,
    landmark_time_a: float,
    landmark_time_b: float,
    playback_fps_a: float,
    playback_fps_b: float,
    slow_motion_factor_a: float,
    slow_motion_factor_b: float,
) -> tuple[int, int]:
    """Map common physical time to each retimed video's stored frame clock."""
    landmark_a = landmark_time_a * playback_fps_a
    landmark_b = landmark_time_b * playback_fps_b
    effective_fps_a = playback_fps_a * slow_motion_factor_a
    effective_fps_b = playback_fps_b * slow_motion_factor_b
    return (
        int(round(landmark_a + progress_time * effective_fps_a)),
        int(round(landmark_b + progress_time * effective_fps_b)),
    )


def extract_synced_frames(
    video_path_a: Path,
    video_path_b: Path,
    max_frames: int = 16,
    slow_motion_factor_a: float = 1.0,
    slow_motion_factor_b: float = 1.0,
    board_type: str = "charuco",
    grid_size: tuple[int, int] = (11, 8),
    square_length: float = 0.04,
    marker_length: float = 0.03,
) -> tuple[list[Path], list[Path], BoardDetectionHint | None]:
    """Sync videos, scan their shared timeline, and retain board-visible pairs."""
    from golfie_cv.sync import estimate_sync_landmarks
    from golfie_cv.video import analyze_video_timing

    cap_a = cv2.VideoCapture(str(video_path_a))
    cap_b = cv2.VideoCapture(str(video_path_b))

    total_a = int(cap_a.get(cv2.CAP_PROP_FRAME_COUNT))
    total_b = int(cap_b.get(cv2.CAP_PROP_FRAME_COUNT))

    fps_a = cap_a.get(cv2.CAP_PROP_FPS)
    fps_b = cap_b.get(cv2.CAP_PROP_FPS)
    
    if fps_a <= 0: fps_a = 30.0
    if fps_b <= 0: fps_b = 30.0

    if slow_motion_factor_a == 1.0 and slow_motion_factor_b == 1.0:
        # Real-time constant-rate calibration clips do not need an expensive
        # full ffprobe scan of every frame timestamp. Their stored frame clock
        # already is physical time.
        duration_a = total_a / fps_a
        duration_b = total_b / fps_b
        index_at_time_a = lambda seconds: max(0, int(round(seconds * fps_a)))
        index_at_time_b = lambda seconds: max(0, int(round(seconds * fps_b)))
        nominal_fps_a = fps_a
        nominal_fps_b = fps_b
    else:
        log_calibration_progress(
            "Reading Camera A slow-motion frame timeline...", 10, "timing_camera_a"
        )
        timing_a = analyze_video_timing(video_path_a)
        log_calibration_progress(
            "Camera A timeline ready; reading Camera B slow-motion frame timeline...",
            12,
            "timing_camera_b",
        )
        timing_b = analyze_video_timing(video_path_b)
        duration_a = timing_a.playback_duration_seconds
        duration_b = timing_b.playback_duration_seconds
        index_at_time_a = timing_a.frame_index_at_time
        index_at_time_b = timing_b.frame_index_at_time
        nominal_fps_a = timing_a.nominal_fps
        nominal_fps_b = timing_b.nominal_fps

    # Locate the same clap separately in each playback timeline. A constant
    # offset is insufficient when the files have different slow-motion scales.
    try:
        log_calibration_progress(
            "Locating the synchronization clap in both audio tracks...",
            14,
            "audio_sync",
        )
        landmark_time_a, landmark_time_b, confidence = estimate_sync_landmarks(
            video_path_a, video_path_b
        )
        overlap = min(
            (duration_a - landmark_time_a) / slow_motion_factor_a,
            (duration_b - landmark_time_b) / slow_motion_factor_b,
        )
        if confidence < 0.3 or overlap <= 1.0:
            raise ValueError(
                "Calibration audio was extracted, but synchronization was rejected: "
                f"confidence={confidence:.2f}, physical overlap={overlap:.2f}s. "
                "The clap must be the strongest isolated transient "
                "in both recordings."
            )
        log_calibration_progress(
            f"Audio landmarks accepted: confidence={confidence:.2f}, "
            f"Camera A clap={landmark_time_a:.3f}s, Camera B clap={landmark_time_b:.3f}s."
        )
        log_calibration_progress(
            "Slow-motion timeline: "
            f"Camera A={nominal_fps_a:.3f}fps x {slow_motion_factor_a:g}; "
            f"Camera B={nominal_fps_b:.3f}fps x {slow_motion_factor_b:g}; "
            f"overlap after clap={overlap:.3f}s."
            , 17, "board_scan"
        )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"Calibration audio synchronization failed: {exc}"
        ) from exc

    scan_frames = max(max_frames, max_frames * CALIBRATION_SCAN_MULTIPLIER)
    time_step = overlap / scan_frames

    # Subdirectories within temp directory
    tmp_dir_a = video_path_a.parent / "_tmp_cal_a"
    tmp_dir_b = video_path_b.parent / "_tmp_cal_b"
    tmp_dir_a.mkdir(exist_ok=True)
    tmp_dir_b.mkdir(exist_ok=True)

    paths_a = []
    paths_b = []
    # Audio timestamps can trail the corresponding image by several stored
    # frames even in real-time phone video. Always retain nearby Camera B
    # candidates so stereo geometry can refine that offset.
    refinement_shifts = tuple(range(-12, 13, 2))
    refinement_dirs = {
        shift: video_path_b.parent / f"_tmp_cal_b_shift_{shift:+d}"
        for shift in refinement_shifts
        if shift != 0
    }
    for directory in refinement_dirs.values():
        directory.mkdir(exist_ok=True)

    detector_candidates = (
        _pattern_detector_candidates(board_type, grid_size, square_length, marker_length)
        if board_type != "charuco"
        else []
    )
    dictionary_hint: int | None = None
    selected_detector: tuple[str, object] | None = None
    selected_hint: BoardDetectionHint | None = None
    usable_frames: list[tuple[int, int, float, int, int]] = []
    usable_a = 0
    usable_b = 0
    usable_shared = 0
    scan_started = time.monotonic()

    # First pass: inspect a denser, evenly spaced set of timestamps. Store only
    # frame indices, then decode the selected poses again below. This avoids
    # retaining dozens of full-resolution phone frames in memory or on disk.
    for i in range(scan_frames):
        # Use the midpoint of each interval to avoid the clap itself and the
        # last potentially incomplete frame in the shared physical window.
        t = (i + 0.5) * time_step
        # Convert common physical time back to each file's playback PTS, then
        # locate the nearest actual frame. This handles iPhone VFR/ramp edits;
        # multiplying a timestamp by average FPS does not.
        idx_a = index_at_time_a(landmark_time_a + t * slow_motion_factor_a)
        idx_b = index_at_time_b(landmark_time_b + t * slow_motion_factor_b)

        if idx_a >= total_a or idx_b >= total_b:
            break

        cap_a.set(cv2.CAP_PROP_POS_FRAMES, idx_a)
        ret_a, frame_a = cap_a.read()

        cap_b.set(cv2.CAP_PROP_POS_FRAMES, idx_b)
        ret_b, frame_b = cap_b.read()
        if not ret_a or not ret_b or frame_a is None or frame_b is None:
            continue

        gray_a = _detection_thumbnail(frame_a)
        gray_b = _detection_thumbnail(frame_b)
        if selected_detector is None:
            if board_type == "charuco" and dictionary_hint is None:
                dictionary_match = _find_charuco_dictionary(gray_a, gray_b)
                if dictionary_match is None:
                    if (i + 1) % 4 == 0:
                        elapsed = time.monotonic() - scan_started
                        log_calibration_progress(
                            f"Board scan {i + 1}/{scan_frames} (marker search, {elapsed:.1f}s): "
                            "no frame has two recognized board markers yet.",
                            17 + int(11 * (i + 1) / scan_frames),
                            "board_scan",
                        )
                    continue
                dictionary_hint, markers_a, markers_b = dictionary_match
                detector_candidates = _charuco_detector_candidates(
                    grid_size,
                    square_length,
                    marker_length,
                    dictionary_ids=[dictionary_hint],
                )
                log_calibration_progress(
                    f"Marker family identified as dictionary {dictionary_hint} from "
                    f"Camera A={markers_a}, Camera B={markers_b} markers; checking board rotation.",
                    18,
                    "board_scan",
                )
            best = _best_shared_detector(
                gray_a, gray_b, board_type, detector_candidates
            )
            if best is None:
                continue
            description, detector, corners_a, corners_b = best
            selected_detector = (description, detector)
            selected_hint = _board_hint(description, board_type)
            log_calibration_progress(
                f"Board detector locked from a partial or complete view: {description}; "
                f"current view "
                f"has Camera A={corners_a}, Camera B={corners_b} corners."
            )
        else:
            _, detector = selected_detector
            corners_a = _detected_corner_count(gray_a, board_type, detector)
            corners_b = _detected_corner_count(gray_b, board_type, detector)

        good_a = corners_a >= MIN_CALIBRATION_CORNERS
        good_b = corners_b >= MIN_CALIBRATION_CORNERS
        usable_a += int(good_a)
        usable_b += int(good_b)
        usable_shared += int(good_a and good_b)
        if good_a or good_b:
            usable_frames.append((idx_a, idx_b, t, corners_a, corners_b))

        if (i + 1) % 4 == 0 or i + 1 == scan_frames:
            detector_state = "locked" if selected_detector is not None else "searching"
            elapsed = time.monotonic() - scan_started
            log_calibration_progress(
                f"Board scan {i + 1}/{scan_frames} ({detector_state}, {elapsed:.1f}s): "
                f"Camera A usable={usable_a}, Camera B usable={usable_b}, "
                f"shared={usable_shared}; {MIN_CALIBRATION_CORNERS}+ corners accepted.",
                17 + int(11 * (i + 1) / scan_frames),
                "board_scan",
            )

    chosen_frames = _evenly_spaced_items(usable_frames, max_frames)
    log_calibration_progress(
        f"Board scan found Camera A={usable_a}, Camera B={usable_b}, "
        f"shared={usable_shared} usable timestamps across {scan_frames} checks; "
        f"selected {len(chosen_frames)} evenly spaced pairs."
        , 28, "saving_frames"
    )

    # Second pass: save the chosen base frames and all nearby Camera B visual
    # sync candidates under matching sequential names.
    for idx_a, idx_b, _, _, _ in chosen_frames:
        cap_a.set(cv2.CAP_PROP_POS_FRAMES, idx_a)
        ret_a, frame_a = cap_a.read()
        candidate_frames: dict[int, np.ndarray] = {}
        first_shift = min(refinement_shifts)
        last_shift = max(refinement_shifts)
        cap_b.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx_b + first_shift))
        for shifted_index in range(first_shift, last_shift + 1):
            ret_candidate, candidate = cap_b.read()
            if not ret_candidate or candidate is None:
                break
            if shifted_index in refinement_shifts:
                candidate_frames[shifted_index] = candidate
        frame_b = candidate_frames.get(0)
        if not ret_a or frame_a is None or frame_b is None:
            continue

        saved = len(paths_a)
        path_a = tmp_dir_a / f"frame_{saved:04d}.png"
        path_b = tmp_dir_b / f"frame_{saved:04d}.png"
        cv2.imwrite(str(path_a), frame_a)
        cv2.imwrite(str(path_b), frame_b)
        paths_a.append(path_a)
        paths_b.append(path_b)
        for shift, directory in refinement_dirs.items():
            candidate = candidate_frames.get(shift)
            if candidate is not None:
                cv2.imwrite(
                    str(directory / f"frame_{saved:04d}.jpg"),
                    candidate,
                    [cv2.IMWRITE_JPEG_QUALITY, 97],
                )

    log_calibration_progress(
        f"Saved {len(paths_a)} selected frame pairs and nearby visual-sync candidates.",
        29,
        "saving_frames",
    )

    cap_a.release()
    cap_b.release()
    return paths_a, paths_b, selected_hint


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
def upload_and_calibrate(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
    board_type: str = Form(default="charuco"),
    grid_cols: int = Form(default=11),
    grid_rows: int = Form(default=8),
    square_size: float = Form(default=0.04),
    marker_size: float = Form(default=0.03),
    measured_baseline: float | None = Form(default=None),
    slow_motion_factor_a: float = Form(default=1.0),
    slow_motion_factor_b: float = Form(default=1.0),
    _calibration_guard: None = Depends(_exclusive_calibration_run),
) -> CalibrationResult:
    # This endpoint performs long, blocking OpenCV/FFmpeg work. Defining it as
    # a synchronous route lets FastAPI run it in its worker thread pool so the
    # event loop can continue serving progress, health, and recovery requests.
    log_calibration_progress("Starting calibration process...", 1, "starting")
    log_calibration_progress(
        f"Board geometry: type={board_type}, grid={grid_cols}x{grid_rows}, "
        f"square={square_size * 1000.0:.2f}mm, marker={marker_size * 1000.0:.2f}mm."
    )
    if not 1.0 <= slow_motion_factor_a <= 32.0 or not 1.0 <= slow_motion_factor_b <= 32.0:
        raise HTTPException(
            status_code=400,
            detail="Slow-motion playback factors must be between 1x and 32x.",
        )
    log_calibration_progress(
        f"Playback slowdown: Camera A={slow_motion_factor_a:g}x, "
        f"Camera B={slow_motion_factor_b:g}x."
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
            paths_a, paths_b, board_hint = extract_synced_frames(
                path_a,
                path_b,
                slow_motion_factor_a=slow_motion_factor_a,
                slow_motion_factor_b=slow_motion_factor_b,
                board_type=board_type,
                grid_size=(grid_cols, grid_rows),
                square_length=square_size,
                marker_length=marker_size,
            )
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
            log_calibration_progress(
                "Calibration failed: no frames contained at least four detectable board corners "
                "in either camera after the clap.",
                30,
                "failed",
            )
            raise HTTPException(
                status_code=400,
                detail=(
                    "No usable board frames were found after the clap. Four or more detected "
                    "corners in either camera are sufficient; the complete board is not required."
                ),
            )

        # The two intrinsic solves are independent and OpenCV releases the GIL,
        # so run them together instead of making the user wait for A then B.
        log_calibration_progress(
            "Calibrating both camera lens models in parallel...", 32, "intrinsics"
        )

        intrinsic_progress = {"Camera A": 0.0, "Camera B": 0.0}
        intrinsic_progress_lock = threading.Lock()

        def report_intrinsic_progress(label: str, fraction: float, message: str):
            with intrinsic_progress_lock:
                intrinsic_progress[label] = fraction
                combined = sum(intrinsic_progress.values()) / len(intrinsic_progress)
                progress = 32 + int(25 * combined)
            log_calibration_progress(
                f"{label}: {message}", progress, "intrinsics"
            )

        def solve_intrinsics(label: str, paths: list[Path]):
            hinted_grid_size = board_hint.grid_size if board_hint else (grid_cols, grid_rows)
            hinted_dictionary = (
                board_hint.dictionary_id
                if board_hint and board_hint.dictionary_id is not None
                else cv2.aruco.DICT_6X6_250
            )
            result = calibrate_intrinsics(
                paths,
                board_type=board_type,
                grid_size=hinted_grid_size,
                square_length=square_size,
                marker_length=marker_size,
                dictionary_id=hinted_dictionary,
                progress_callback=lambda fraction, message: report_intrinsic_progress(
                    label, fraction, message
                ),
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
            log_calibration_progress(
                "Testing the audio alignment and nearby Camera B frames in parallel...",
                68,
                "refining_sync",
            )
            candidates = []

            def solve_shift_candidate(shift: int):
                if shift == 0:
                    candidate_paths_b = paths_b
                else:
                    candidate_dir = path_b.parent / f"_tmp_cal_b_shift_{shift:+d}"
                    candidate_paths_b = sorted(candidate_dir.glob("frame_*.jpg"))
                if len(candidate_paths_b) != len(paths_a):
                    return shift, None
                hinted_grid_size = board_hint.grid_size if board_hint else (grid_cols, grid_rows)
                hinted_dictionary = (
                    board_hint.dictionary_id
                    if board_hint and board_hint.dictionary_id is not None
                    else cv2.aruco.DICT_6X6_250
                )
                return shift, calibrate_stereo(
                    intrinsics_a,
                    intrinsics_b,
                    paths_a,
                    candidate_paths_b,
                    board_type=board_type,
                    grid_size=hinted_grid_size,
                    square_length=square_size,
                    marker_length=marker_size,
                    dictionary_id=hinted_dictionary,
                )

            # Audio is normally within four stored frames. Include the base
            # alignment in this same concurrent batch instead of first paying
            # for a separate serial stereo solve.
            shift_batches = [(0, -4, -2, 2, 4), (-12, -10, -8, -6, 6, 8, 10, 12)]
            total_shift_candidates = sum(len(batch) for batch in shift_batches)
            evaluated_shift_candidates = 0
            for batch in shift_batches:
                with ThreadPoolExecutor(max_workers=min(5, len(batch))) as executor:
                    futures = {
                        executor.submit(solve_shift_candidate, shift): shift
                        for shift in batch
                    }
                    for future in as_completed(futures):
                        shift = futures[future]
                        evaluated_shift_candidates += 1
                        candidate_progress = 68 + int(
                            20 * evaluated_shift_candidates / total_shift_candidates
                        )
                        try:
                            _, candidate_result = future.result()
                        except Exception as candidate_exc:
                            log_calibration_progress(
                                f"Visual sync candidate {shift:+d} frames rejected: "
                                f"{candidate_exc}",
                                candidate_progress,
                                "refining_sync",
                            )
                            continue
                        if candidate_result is None:
                            log_calibration_progress(
                                f"Visual sync candidate {shift:+d} frames skipped: "
                                "not every selected pose had a nearby decoded frame.",
                                candidate_progress,
                                "refining_sync",
                            )
                            continue
                        candidates.append((shift, candidate_result))
                        candidate_median = candidate_result.epipolar_error_median_px
                        candidate_inliers = candidate_result.epipolar_inlier_ratio
                        median_text = (
                            f"{candidate_median:.3f}px"
                            if candidate_median is not None
                            else "n/a"
                        )
                        inlier_text = (
                            f"{candidate_inliers * 100:.1f}%"
                            if candidate_inliers is not None
                            else "n/a"
                        )
                        log_calibration_progress(
                            f"Visual sync candidate {shift:+d} frames: "
                            f"RMS={candidate_result.reprojection_error_px:.3f}px, "
                            f"median={median_text}, inliers={inlier_text}.",
                            candidate_progress,
                            "refining_sync",
                        )
                if any(result.is_valid for _, result in candidates):
                    break
            if not candidates:
                raise ValueError("No stereo sync candidate produced usable board geometry.")
            selected_frame_shift_b, stereo_result = min(
                candidates,
                key=lambda item: (
                    not item[1].is_valid,
                    item[1].reprojection_error_px,
                    item[1].epipolar_error_median_px
                    if item[1].epipolar_error_median_px is not None
                    else float("inf"),
                    -(
                        item[1].epipolar_inlier_ratio
                        if item[1].epipolar_inlier_ratio is not None
                        else 0.0
                    ),
                ),
            )
            log_calibration_progress(
                f"Selected Camera B visual sync refinement {selected_frame_shift_b:+d} "
                "frames."
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
            from golfie_cv.video import VideoReadError, read_video_metadata

            try:
                metadata_a = read_video_metadata(path_a)
                metadata_b = read_video_metadata(path_b)
                stereo_result.camera_a_calibration_fps = (
                    metadata_a.fps * slow_motion_factor_a
                )
                stereo_result.camera_b_calibration_fps = (
                    metadata_b.fps * slow_motion_factor_b
                )
            except VideoReadError:
                # The geometry result remains valid even when a damaged or
                # synthetic container cannot provide optional rate metadata.
                stereo_result.camera_a_calibration_fps = None
                stereo_result.camera_b_calibration_fps = None
            log_calibration_progress(
                "Recorded effective physical calibration rates.", 95, "metadata"
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
