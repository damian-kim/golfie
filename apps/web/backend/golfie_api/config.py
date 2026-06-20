"""Backend configuration: where session/shot data lives on disk.

Per spec section 4, all persisted data lives under the repo-level
`data/` directory (not inside the backend package itself), so that CLI
scripts (scripts/process_shot.py) and the API agree on one location.
"""

from __future__ import annotations

from pathlib import Path

# golfie_api/config.py -> golfie_api -> backend -> web -> apps -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[4]

DATA_DIR = REPO_ROOT / "data"
SHOTS_DIR = DATA_DIR / "shots"
SAMPLE_SESSIONS_DIR = DATA_DIR / "sample_sessions"
CALIBRATION_DIR = DATA_DIR / "calibration"

for _dir in (SHOTS_DIR, SAMPLE_SESSIONS_DIR, CALIBRATION_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
