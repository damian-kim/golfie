# Golfie

Dual-iPhone golf shot reconstruction and simulation. See `docs/` and the
original project spec for the full design; this README covers what
exists today and how to run it.

**Status: Milestone 0 (repository + basic web app).** Nothing below
detects, tracks, triangulates, or estimates a real shot yet -- that's
Milestones 1-7. What *is* real: video metadata extraction, the full
session/upload/process API lifecycle, the RK4 ball-flight physics
engine (gravity + drag + experimental Magnus lift), and a browser-based
3D driving range that can render any trajectory this system produces.

## Repository layout

```
packages/
  golfie_core/      schemas, coordinate frame, physical constants, units
  golfie_cv/         video metadata (real); calibration/sync/detection/
                      tracking/triangulation (stubs with defined contracts)
  golfie_physics/    RK4 integrator + projectile model (real);
                      fitting/validation (stubs)
  golfie_render/     Session/ShotResult -> Three.js JSON payload
apps/web/
  backend/golfie_api/  FastAPI app (sessions, upload, process, trajectory)
  frontend/            Vite + React + TypeScript + react-three-fiber
scripts/
  process_shot.py            standalone CLI: two videos -> results.json
  generate_sample_session.py synthetic demo shot for the renderer
  setup_dev_env.sh           installs everything below
data/
  shots/             one folder per session (videos + session.json)
  sample_sessions/   the synthetic demo trajectory
tests/               pytest: schemas, coordinates, physics, video, API
```

## Setup

Requires Python 3.10+, Node 18+, and `ffmpeg` (for video I/O; OpenCV
uses it under the hood on most platforms).

```bash
./scripts/setup_dev_env.sh
```

This installs all four Python packages and the backend in editable
mode, installs frontend dependencies, and generates the synthetic demo
session.

## Running it

**Backend** (FastAPI, port 8000):
```bash
cd apps/web/backend
uvicorn golfie_api.main:app --reload --port 8000
```

**Frontend** (Vite dev server, port 5173):
```bash
cd apps/web/frontend
npm run dev
```

Then open `http://localhost:5173`. The "Demo range" link in the header
renders the synthetic sample shot without needing the upload flow --
use it to sanity-check the 3D scene. The "Upload" flow exercises the
real (if currently honest-but-boring) end-to-end pipeline: create a
session, upload two videos, process, review, simulate.

**Standalone CLI** (no backend/frontend needed):
```bash
python scripts/process_shot.py camera_a.mp4 camera_b.mp4 --club driver -o results.json
```

**Tests:**
```bash
pytest tests/ -v
```

## What's real vs. placeholder right now

| Capability | Status |
|---|---|
| Video metadata (fps/resolution/duration) | Real (OpenCV) |
| Session storage, upload, API lifecycle | Real |
| RK4 projectile physics (gravity + drag) | Real, validated against closed-form projectile motion |
| Magnus lift | Implemented, but experimental/unvalidated -- off by default |
| Camera calibration | Stub (Milestone 1) |
| Video sync | Stub (Milestone 2) |
| Ball detection/tracking | Stub (Milestones 3-4) |
| Triangulation | Stub (Milestone 5) |
| Launch parameter estimation / fitting | Stub (Milestones 6-7) |
| Every shot metric you'll see today | Honestly `not_available` (see `golfie_core.schemas.MetricValue`) |

The one exception is the synthetic demo session
(`data/sample_sessions/sample_shot.json`), which exists purely to give
the renderer something to draw and is clearly labeled as synthetic
everywhere it appears (warnings, `is_placeholder`, per-metric `notes`).
It is never confused with `scripts/process_shot.py`'s output, which
only reports what was actually read from the input videos.

## Known limitations (v0)

- No ground-roll model: simulated `total_m` always equals `carry_m`.
- Magnus lift uses a simplified spin-independent-of-Cl model; treat any
  spin-driven shot shape as illustrative, not validated.
- Backend storage is per-process JSON files (no concurrency control);
  fine for one local user, not for multiple simultaneous writers.
- Frontend has no automated UI/visual test in this environment (no
  headless browser available); it's verified via `tsc -b`, `vite
  build`, `eslint`, and manual dev-server + API smoke tests instead.

## Next recommended step

Milestone 1: intrinsic + stereo camera calibration
(`golfie_cv.calibration`), starting with `scripts/calibrate_cameras.py`
(not yet created) that takes a folder of ChArUco board images/video per
camera and produces a `CalibrationResult` your real iPhone footage can
actually use. That's the dependency every later milestone (sync uses
fps already; triangulation needs calibration) is blocked on.
