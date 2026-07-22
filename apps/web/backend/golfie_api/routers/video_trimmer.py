"""Single-video golf swing impact detection and lossless trimming endpoints."""

from __future__ import annotations

import json
import re
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from golfie_api.config import TRIMS_DIR
from golfie_cv.video import LosslessTrimError, trim_video_lossless


router = APIRouter(prefix="/video-trimmer", tags=["video-trimmer"])
_TRIM_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_SAFE_STEM_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
_SAFE_SUFFIX_PATTERN = re.compile(r"^\.[A-Za-z0-9]{1,9}$")


class TrimJobResponse(BaseModel):
    trim_id: str
    created_at: str
    original_filename: str
    output_filename: str
    source_duration_seconds: float
    requested_start_seconds: float
    requested_end_seconds: float
    actual_start_seconds: float
    actual_end_seconds: float
    output_duration_seconds: float
    fps: float
    width: int
    height: int
    sample_aspect_ratio: str
    display_aspect_ratio: str
    video_codec: str
    audio_codec: str | None
    keyframe_aligned: bool
    lossless: bool
    preview_url: str
    download_url: str


def _job_dir(trim_id: str) -> Path:
    if not _TRIM_ID_PATTERN.fullmatch(trim_id):
        raise HTTPException(status_code=404, detail="Trim result not found.")
    return TRIMS_DIR / trim_id


def _load_job(trim_id: str) -> tuple[Path, dict[str, Any]]:
    directory = _job_dir(trim_id)
    metadata_path = directory / "result.json"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise HTTPException(status_code=404, detail="Trim result not found.") from None
    return directory, payload


def _safe_upload_names(filename: str | None) -> tuple[str, str, str]:
    original = Path(filename or "golf-swing.mp4").name
    source_suffix = Path(original).suffix
    suffix = source_suffix.lower() if _SAFE_SUFFIX_PATTERN.fullmatch(source_suffix) else ".mp4"
    stem = _SAFE_STEM_PATTERN.sub("-", Path(original).stem).strip("-._") or "golf-swing"
    return original, f"source{suffix}", f"{stem}-golfie-trim{suffix}"


def _save_result(
    directory: Path,
    trim_id: str,
    original_filename: str,
    output_filename: str,
    result: Any,
) -> dict[str, Any]:
    payload = {
        "trim_id": trim_id,
        "created_at": datetime.now(UTC).isoformat(),
        "original_filename": original_filename,
        "output_filename": output_filename,
        **result.to_dict(),
        "preview_url": f"/video-trimmer/{trim_id}/video",
        "download_url": f"/video-trimmer/{trim_id}/download",
    }
    (directory / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


@router.post("/trim", response_model=TrimJobResponse)
async def create_manual_lossless_trim(
    file: UploadFile = File(...),
    start_seconds: float = Form(ge=0.0),
    end_seconds: float = Form(gt=0.0),
) -> dict[str, Any]:
    trim_id = uuid.uuid4().hex
    directory = TRIMS_DIR / trim_id
    directory.mkdir(parents=True, exist_ok=False)
    original_filename, source_name, output_filename = _safe_upload_names(file.filename)
    source_path = directory / source_name
    output_path = directory / output_filename

    try:
        with source_path.open("wb") as destination:
            shutil.copyfileobj(file.file, destination)
        result = trim_video_lossless(
            source_path,
            output_path,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )
        payload = _save_result(
            directory, trim_id, original_filename, output_filename, result
        )
        source_path.unlink(missing_ok=True)
        return payload
    except LosslessTrimError as exc:
        shutil.rmtree(directory, ignore_errors=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    finally:
        await file.close()


@router.get("/{trim_id}", response_model=TrimJobResponse)
def get_trim_result(trim_id: str) -> dict[str, Any]:
    _, payload = _load_job(trim_id)
    return payload


def _video_response(trim_id: str, *, download: bool) -> FileResponse:
    directory, payload = _load_job(trim_id)
    video_path = directory / payload["output_filename"]
    if not video_path.exists() or not video_path.is_file():
        raise HTTPException(status_code=404, detail="Trimmed video file not found.")
    return FileResponse(
        video_path,
        filename=payload["output_filename"],
        content_disposition_type="attachment" if download else "inline",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@router.get("/{trim_id}/video")
def preview_trimmed_video(trim_id: str) -> FileResponse:
    return _video_response(trim_id, download=False)


@router.get("/{trim_id}/download")
def download_trimmed_video(trim_id: str) -> FileResponse:
    return _video_response(trim_id, download=True)
