"""Golfie FastAPI backend entrypoint.


Run with:
    uvicorn golfie_api.main:app --reload --port 8000"""

from __future__ import annotations

import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_root / "packages"))

import os
import tempfile
from golfie_api.config import TEMP_DIR

# Keep large decoded/transcoded intermediates in a configurable repository
# runtime directory instead of a developer-specific drive path.
os.environ["TEMP"] = str(TEMP_DIR)
os.environ["TMP"] = str(TEMP_DIR)
os.environ["TMPDIR"] = str(TEMP_DIR)
tempfile.tempdir = str(TEMP_DIR)

import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from golfie_api.config import SAMPLE_SESSIONS_DIR
from golfie_api.routers import sessions_router, calibration_router

app = FastAPI(
    title="Golfie API",
    description="Dual-iPhone golf shot reconstruction and simulation backend.",
    version="0.1.0",
)

# Local-first development: Vite selects the next available port when another
# local app already owns 5173. Permit loopback origins on any port so that
# harmless port selection does not turn a successful request into a browser
# CORS/network error. This still excludes non-local origins.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(?:localhost|127\.0\.0\.1|\[::1\])(?::\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions_router)
app.include_router(calibration_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/demo/sample-trajectory")
def sample_trajectory() -> dict:
    """Serves the hardcoded, clearly-synthetic demo shot used by the
    frontend's driving range scene before any real session has been
    processed (spec section 26: "Three.js driving range that can render
    a hardcoded sample trajectory from JSON").
    """
    sample_path = SAMPLE_SESSIONS_DIR / "sample_shot.json"
    if not sample_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"No sample trajectory found at {sample_path}. "
                "Run scripts/generate_sample_session.py first."
            ),
        )
    return json.loads(sample_path.read_text())

# Trigger reload
