# Golfie coordinate system

Golfie uses one physically meaningful world coordinate system everywhere
above the raw stereo-calibration layer (spec section 6).

## Convention

- **Origin**: golf ball position at address.
- **+X**: target line direction (downrange).
- **+Z**: vertical up.
- **+Y**: chosen so the frame is right-handed: `X x Y = Z`. For a
  right-handed golfer this points toward their right; the sign is a
  consequence of the right-hand rule, not a separate per-handedness
  setting -- `handedness` on a `Session` is metadata only and does not
  change the math.
- **Units**: meters, seconds, radians internally, everywhere in
  `golfie_core`, `golfie_cv`, and `golfie_physics`. The UI converts to
  yards / mph / degrees / rpm only at the display edge
  (`golfie_core.metrics.units`, `apps/web/frontend/src/lib/units.ts`).

## Why a separate "rig frame"

Stereo calibration (Milestone 1) produces 3D points in whatever frame
falls out of `cv2.stereoCalibrate` -- typically camera A's frame, with
arbitrary origin/orientation relative to the golfer. That frame has no
physical meaning on its own.

`golfie_core.coordinates.CoordinateTransformer` maps points from that
"rig frame" into the Golfie world frame once we know, in rig-frame
coordinates:

- where the ball was at address (the world origin),
- which direction the target line points,
- which direction is "up" (doesn't need to be exactly vertical --
  only the component orthogonal to the target line is used).

For v0, those three things come from manual UI confirmation or floor
markers (spec section 7, Phase 3). Later, they can be inferred
automatically from the calibration board's known pose or from detected
floor markers.

## Mapping into the Three.js renderer

The browser-based driving range scene uses Three.js's own convention
(Y is "up"), not Golfie's. `golfie_render.threejs.payload` is the one
place that conversion happens:

| Golfie world frame | Three.js scene frame |
|---|---|
| +X (downrange)      | scene X (downrange) |
| +Z (up)              | scene Y (up) |
| +Y (lateral/side)    | scene Z (lateral/side) |

If you ever need to debug "why is the ball flying sideways" in the
renderer, check this table before suspecting the physics engine.
