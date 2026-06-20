"""Golfie FastAPI backend entrypoint.

Run with:
    uvicorn golfie_api.main:app --reload --port 8000
"""

from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from golfie_api.config import SAMPLE_SESSIONS_DIR
from golfie_api.routers import sessions_router

app = FastAPI(
    title="Golfie API",
    description="Dual-iPhone golf shot reconstruction and simulation backend (v0 / Milestone 0).",
    version="0.1.0",
)

# Local-first dev setup: Vite's default port plus a couple of common
# alternates. Tighten this before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions_router)


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
