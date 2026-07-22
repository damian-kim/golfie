# Golfie

Dual-iPhone golf shot reconstruction and simulation. See `docs/` and the
original project spec for the full design; this README covers what
exists today and how to run it.

**Status: calibration/triangulation rescue MVP.** The repository now has
a real, validation-gated ChArUco stereo calibration, stereo-aware ball
tracking, distortion-corrected triangulation, launch fitting, and the
original RK4/browser rendering path. Real footage is still sensitive to
capture quality, so calibration and epipolar diagnostics are surfaced
and bad geometry is rejected instead of being presented as a shot.

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

The **Video Editor** page at `http://localhost:5173/video-editor` is a separate
single-video utility. Preview the original file, place start and end marks with
sliders or exact timestamps, and export a downloadable FFmpeg stream-copy trim.
Because no video frames are re-encoded, the source codec, FPS, resolution,
pixel aspect ratio, and encoded image data are preserved. For inter-frame
codecs, Golfie safely moves the selected start back to the nearest existing
keyframe so the result remains independently playable and includes the full
selection.

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
| Magnus lift | Enabled only when marked-ball stereo spin passes confidence gates |
| Camera calibration | Implemented; full distortion model + held-out epipolar validation |
| Video sync | Implemented; audio transient, with confidence gating |
| Ball detection/tracking | Implemented MVP; joint stereo-aware hypothesis selection |
| Triangulation | Implemented; undistorted rays, tight timing, epipolar/depth/parallax gates |
| Launch parameter estimation / fitting | Implemented MVP; marked-ball spin estimation is confidence-gated |
| Every shot metric you'll see today | Honestly `not_available` (see `golfie_core.schemas.MetricValue`) |

The one exception is the synthetic demo session
(`data/sample_sessions/sample_shot.json`), which exists purely to give
the renderer something to draw and is clearly labeled as synthetic
everywhere it appears (warnings, `is_placeholder`, per-metric `notes`).
It is never confused with `scripts/process_shot.py`'s output, which
only reports what was actually read from the input videos.

## Known limitations (v0)

- No ground-roll model: simulated `total_m` always equals `carry_m`.
- Marked-ball spin uses two measured camera-axis components and assumes zero
  rifle spin along the launch direction. Magnus lift scales with spin ratio,
  but remains an estimated/unvalidated aerodynamic model.
- Backend storage is per-process JSON files (no concurrency control);
  fine for one local user, not for multiple simultaneous writers.
- Frontend has no automated UI/visual test in this environment (no
  headless browser available); it's verified via `tsc -b`, `vite
  build`, `eslint`, and manual dev-server + API smoke tests instead.

## Capture requirements for the rescue MVP

- Use the exact same lens, orientation, resolution, and stabilization mode
  for calibration and shot capture.
- Camera A must be the upright down-the-line camera; Camera B is face-on.
- Start both calibration recordings, make one sharp synchronization clap,
  then move the board through varied angles, depths, and image positions.
- A pre-v2 active calibration is intentionally rejected; recalibrate once
  after updating so distortion and validation data are present.
