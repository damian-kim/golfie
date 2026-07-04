#!/usr/bin/env python3
"""Generate synthetic golf ball test videos for pipeline testing.

Creates stereo camera pairs (Camera A + Camera B) with a golf ball following
a physics-simulated trajectory. Includes ChArUco calibration board images,
synthetic audio for sync testing, and ground truth JSON for validation.

Usage:
    python scripts/generate_test_videos.py [--output-dir data/test_videos] [--scenario all|straight|draw|fade]

Dependencies: opencv-python, numpy, scipy, golfie_physics, golfie_core, ffmpeg
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Add packages to sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _pkg in ("golfie_core", "golfie_physics", "golfie_cv"):
    _pkg_path = _REPO_ROOT / "packages" / _pkg
    if str(_pkg_path) not in sys.path and _pkg_path.exists():
        sys.path.insert(0, str(_pkg_path))

from golfie_physics.models import FlightParams, simulate_from_launch_conditions  # noqa: E402


# ── Constants ──────────────────────────────────────────────────────────────────

WIDTH = 1920
HEIGHT = 1080
FPS = 240.0
BALL_RADIUS_M = 0.02135  # golf ball radius in meters (1.68 inches / 2)
BALL_COLOR = (255, 255, 255)  # BGR white
BG_COLOR_GRASS = (34, 85, 40)  # BGR dark green (grass)
BG_COLOR_SKY = (180, 210, 230)  # BGR light blue (sky)
IMPACT_FRAME = 30  # frame index where ball launches
TOTAL_FRAMES = 360  # ~1.5s at 240fps — enough for a full shot


# ── Camera Model ───────────────────────────────────────────────────────────────

@dataclass
class VirtualCamera:
    """A virtual pinhole camera with known intrinsics and extrinsics."""
    name: str
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = WIDTH
    height: int = HEIGHT
    # World-to-camera transform (4x4)
    extrinsics: np.ndarray = field(default_factory=lambda: np.eye(4))

    @property
    def K(self) -> np.ndarray:
        """3x3 camera intrinsic matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1],
        ], dtype=np.float64)

    def project(self, point_3d: np.ndarray) -> tuple[float, float, float]:
        """Project a 3D world point to 2D pixel coordinates.

        Returns (u, v, depth) where depth is the Z coordinate in camera frame.
        Returns (-1, -1, -1) if point is behind the camera.
        """
        p_hom = np.append(point_3d, 1.0)
        p_cam = self.extrinsics @ p_hom
        depth = p_cam[2]
        if depth < 0.01:  # behind camera
            return -1, -1, -1
        p_img = self.K @ p_cam[:3]
        u = p_img[0] / p_img[2]
        v = p_img[1] / p_img[2]
        return float(u), float(v), float(depth)

    def ball_radius_px(self, depth_m: float) -> float:
        """Compute ball radius in pixels at a given depth."""
        if depth_m < 0.01:
            return 0
        return float(BALL_RADIUS_M * self.fx / depth_m)

    def intrinsics_matrix(self) -> list[list[float]]:
        return self.K.tolist()

    def extrinsics_matrix(self) -> list[list[float]]:
        return self.extrinsics.tolist()


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Build a 4x4 world-to-camera transform matrix.

    Camera at `eye` looking at `target` with `up` as the approximate up vector.
    Convention: camera looks along +Z in camera space (positive depth = in front).
    """
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    actual_up = np.cross(right, forward)

    # Rotation (world-to-camera): rows are right, up, forward
    # Points in front of the camera have positive Z (depth > 0).
    R = np.array([right, actual_up, forward])  # 3x3
    t = -R @ eye

    ext = np.eye(4)
    ext[:3, :3] = R
    ext[:3, 3] = t
    return ext


def setup_stereo_cameras() -> tuple[VirtualCamera, VirtualCamera]:
    """Set up two cameras for stereo golf ball recording.

    Camera A: "down the line" — behind the golfer, looking along the target line.
    Camera B: "face on" — to the golfer's right side, angled to see both the
              tee and the ball flight path.

    Both cameras are ~2.5m from the tee, providing good stereo coverage.
    """
    up = np.array([0.0, 0.0, 1.0])

    # Camera A: behind the golfer, looking along +X (target line)
    eye_a = np.array([-2.0, 1.0, 1.5])
    target_a = np.array([15.0, 0.0, 3.0])
    ext_a = _look_at(eye_a, target_a, up)

    # Camera B: to the right side (golfer's right = +Y), looking back toward
    # the tee area so it can see the ball from launch.
    eye_b = np.array([1.0, -2.0, 1.5])
    target_b = np.array([15.0, 0.0, 3.0])
    ext_b = _look_at(eye_b, target_b, up)

    cam_a = VirtualCamera(
        name="camera_a",
        fx=1400.0, fy=1400.0,
        cx=WIDTH / 2, cy=HEIGHT / 2,
        extrinsics=ext_a,
    )
    cam_b = VirtualCamera(
        name="camera_b",
        fx=1400.0, fy=1400.0,
        cx=WIDTH / 2, cy=HEIGHT / 2,
        extrinsics=ext_b,
    )
    return cam_a, cam_b


# ── Background Rendering ──────────────────────────────────────────────────────

def render_background() -> np.ndarray:
    """Create a static grass/sky background frame (1920x1080 BGR)."""
    bg = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    # Sky gradient (top 40% of frame)
    sky_line = int(HEIGHT * 0.4)
    for y in range(sky_line):
        t = y / sky_line
        r = int(BG_COLOR_SKY[2] * (1 - t * 0.3))
        g = int(BG_COLOR_SKY[1] * (1 - t * 0.2))
        b = int(BG_COLOR_SKY[0] * (1 - t * 0.1))
        bg[y, :] = [b, g, r]

    # Grass (bottom 60%)
    for y in range(sky_line, HEIGHT):
        depth_factor = (y - sky_line) / (HEIGHT - sky_line)
        variation = int(10 * np.sin(y * 0.1))
        r = max(0, min(255, BG_COLOR_GRASS[2] + variation + int(depth_factor * 15)))
        g = max(0, min(255, BG_COLOR_GRASS[1] + variation + int(depth_factor * 10)))
        b = max(0, min(255, BG_COLOR_GRASS[0] + variation + int(depth_factor * 5)))
        bg[y, :] = [b, g, r]

    # Add subtle noise for realism
    rng = np.random.RandomState(42)
    noise = rng.randint(-5, 6, bg.shape, dtype=np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return bg


# ── Ball Rendering ────────────────────────────────────────────────────────────

def render_ball(
    frame: np.ndarray,
    u: float,
    v: float,
    radius_px: float,
    speed_px_per_frame: float = 0.0,
    angle_rad: float = 0.0,
) -> np.ndarray:
    """Draw a white golf ball on the frame.

    If the ball is moving fast, draws a motion-blur streak (elongated ellipse)
    to simulate what a real high-speed camera captures.
    """
    out = frame.copy()
    u_int, v_int = int(round(u)), int(round(v))
    r = max(2, int(round(radius_px)))

    # Motion blur threshold: if moving > 3 pixels per frame, add streak
    streak_length = min(speed_px_per_frame * 0.5, 40.0)  # cap streak length

    if streak_length > 3.0:
        # Draw motion-blur streak (elongated ellipse)
        streak_r = max(2, int(round(streak_length / 2)))
        axes = (streak_r, max(1, r // 2))
        angle_deg = np.degrees(angle_rad)
        # Draw the streak (slightly transparent via blending)
        streak_layer = out.copy()
        cv2.ellipse(streak_layer, (u_int, v_int), axes, angle_deg, 0, 360, BALL_COLOR, -1)
        # Also draw a brighter core
        cv2.circle(streak_layer, (u_int, v_int), max(1, r // 2), BALL_COLOR, -1)
        cv2.addWeighted(streak_layer, 0.8, out, 0.2, 0, out)
    else:
        # Normal ball: filled white circle with a slight glow
        glow_r = int(r * 1.5)
        cv2.circle(out, (u_int, v_int), glow_r, (240, 240, 240), -1)
        cv2.circle(out, (u_int, v_int), r, BALL_COLOR, -1)
        # Highlight (small bright spot offset from center)
        highlight_offset = max(1, r // 3)
        cv2.circle(out, (u_int - highlight_offset, v_int - highlight_offset),
                   max(1, r // 3), (255, 255, 255), -1)

    return out


# ── Audio Generation ──────────────────────────────────────────────────────────

def generate_impact_audio(wav_path: Path, fps: float, total_frames: int, impact_frame: int) -> None:
    """Generate a WAV file with a short impact click at the specified frame.

    The click is a brief burst of noise + tone, simulating a golf ball impact.
    Both cameras get the same audio so cross-correlation sync works.
    """
    from scipy.io import wavfile as scipy_wavfile

    sample_rate = 44100
    duration_s = total_frames / fps
    total_samples = int(sample_rate * duration_s)

    # Create silence
    audio = np.zeros(total_samples, dtype=np.float32)

    # Impact click: short burst of noise + harmonic at the impact frame
    impact_sample = int(impact_frame / fps * sample_rate)
    click_duration = int(0.01 * sample_rate)  # 10ms click

    t = np.linspace(0, 0.01, click_duration, dtype=np.float32)
    noise = np.random.randn(click_duration).astype(np.float32) * 0.5
    tone = np.sin(2 * np.pi * 2000 * t).astype(np.float32) * 0.5
    click = (noise + tone) * np.exp(-t * 300)  # fast decay

    end = min(impact_sample + click_duration, total_samples)
    audio[impact_sample:end] += click[:end - impact_sample]

    # Normalize to int16 range
    audio = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    scipy_wavfile.write(str(wav_path), sample_rate, audio)


def mux_audio_into_video(video_path: Path, audio_path: Path, output_path: Path) -> bool:
    """Combine a silent video with a WAV audio track using ffmpeg."""
    ffmpeg_bin = _find_ffmpeg()
    if not ffmpeg_bin:
        return False

    cmd = [
        ffmpeg_bin, "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        str(output_path),
    ]
    res = subprocess.run(cmd, capture_output=True)
    return res.returncode == 0


def _find_ffmpeg() -> str | None:
    """Find ffmpeg binary (PATH or Overwolf fallback)."""
    import shutil
    import glob

    if shutil.which("ffmpeg"):
        return "ffmpeg"

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        pattern = os.path.join(local_app_data, "Overwolf", "**", "ffmpeg.exe")
        matches = glob.glob(pattern, recursive=True)
        if matches:
            ffmpeg_path = matches[0]
            ffmpeg_dir = os.path.dirname(ffmpeg_path)
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
            return ffmpeg_path

    return None


# ── ChArUco Calibration Board ────────────────────────────────────────────────

def generate_calibration_board(
    cam_a: VirtualCamera,
    cam_b: VirtualCamera,
    output_dir: Path,
    grid_size: tuple[int, int] = (11, 8),
    square_length: float = 0.04,
    marker_length: float = 0.03,
) -> None:
    """Generate ChArUco board images and simulated detections for calibration.

    Creates:
    - charuco_board.png: the board itself
    - camera_a_board.png: board as seen from Camera A
    - camera_b_board.png: board as seen from Camera B
    """
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard(grid_size, square_length, marker_length, dictionary)

    # Generate board image
    board_img = board.generateImage((grid_size[0] * 100, grid_size[1] * 100))
    cv2.imwrite(str(output_dir / "charuco_board.png"), board_img)

    # Place the board upright, angled so both cameras can see the face.
    # Board center at (5, 0.5, 0.7): 5m downrange, slightly right of target line.
    board_center = np.array([5.0, 0.5, 0.7])

    # Get board corner positions in board-local coords
    board_corners_3d = board.getChessboardCorners()  # Nx3 array

    # Board local: X right, Y down, Z into board face.
    # Stand upright and angle the face to point toward the midpoint between cameras.
    # Compute the bisector of the two camera-to-board directions as the face normal.
    # This ensures both cameras can see the board face.
    # (Camera positions are set up in setup_stereo_cameras; we reconstruct them here.)
    eye_a = np.array([-2.0, 1.0, 1.5])
    eye_b = np.array([1.0, -2.0, 1.5])
    to_a = eye_a - board_center
    to_a = to_a / np.linalg.norm(to_a)
    to_b = eye_b - board_center
    to_b = to_b / np.linalg.norm(to_b)
    face_dir = to_a + to_b
    face_dir = face_dir / np.linalg.norm(face_dir)

    # Build rotation: face normal = face_dir, local Y = down (-Z world), local X = right
    down_dir = np.array([0.0, 0.0, -1.0])
    right_dir = np.cross(down_dir, face_dir)
    right_dir = right_dir / np.linalg.norm(right_dir)
    actual_down = np.cross(face_dir, right_dir)

    R_board_to_world = np.column_stack([right_dir, actual_down, face_dir])

    world_corners = []
    for corner in board_corners_3d:
        p = R_board_to_world @ corner + board_center
        world_corners.append(p)

    # Project corners onto both cameras
    pts_a = []
    pts_b = []
    valid_indices = []

    for i, p in enumerate(world_corners):
        ua, va, da = cam_a.project(p)
        ub, vb, db = cam_b.project(p)
        if (da > 0 and db > 0
                and 0 <= ua < WIDTH and 0 <= va < HEIGHT
                and 0 <= ub < WIDTH and 0 <= vb < HEIGHT):
            pts_a.append([ua, va])
            pts_b.append([ub, vb])
            valid_indices.append(i)

    # Draw the board from each camera's perspective
    for cam, pts, name in [(cam_a, pts_a, "camera_a"), (cam_b, pts_b, "camera_b")]:
        view_img = render_background()
        for pt in pts:
            cv2.circle(view_img, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
        cv2.imwrite(str(output_dir / f"{name}_board.png"), view_img)

    print(f"  Generated calibration board with {len(valid_indices)} visible corners per camera")


# ── Trajectory Simulation ────────────────────────────────────────────────────

@dataclass
class ScenarioConfig:
    name: str
    ball_speed_mps: float
    launch_angle_deg: float
    horizontal_launch_deg: float
    backspin_rpm: float = 3000.0
    description: str = ""


SCENARIOS = [
    ScenarioConfig(
        name="straight_drive",
        ball_speed_mps=60.0,
        launch_angle_deg=12.0,
        horizontal_launch_deg=0.0,
        backspin_rpm=3000.0,
        description="Straight drive, ~134 mph ball speed",
    ),
    ScenarioConfig(
        name="draw",
        ball_speed_mps=55.0,
        launch_angle_deg=15.0,
        horizontal_launch_deg=-3.0,
        backspin_rpm=3500.0,
        description="Draw shot (right-to-left for righty), ~123 mph",
    ),
    ScenarioConfig(
        name="fade",
        ball_speed_mps=55.0,
        launch_angle_deg=14.0,
        horizontal_launch_deg=3.0,
        backspin_rpm=2500.0,
        description="Fade shot (left-to-right for righty), ~123 mph",
    ),
]


def simulate_trajectory(config: ScenarioConfig) -> list[dict]:
    """Simulate a golf ball trajectory and return 3D points at 240fps."""
    spin_rad_s = np.array([0.0, -config.backspin_rpm * 2 * np.pi / 60.0, 0.0])
    params = FlightParams(drag_enabled=True, lift_enabled=True)

    flight = simulate_from_launch_conditions(
        ball_speed_mps=config.ball_speed_mps,
        launch_angle_deg=config.launch_angle_deg,
        horizontal_launch_deg=config.horizontal_launch_deg,
        params=params,
        spin_rad_s=spin_rad_s,
    )

    # Sample at 240fps intervals
    dt_frame = 1.0 / FPS
    points = []
    sample_idx = 0
    t_target = 0.0

    while t_target < flight.samples[-1].time_s and len(points) < TOTAL_FRAMES - IMPACT_FRAME:
        # Find closest sample
        while sample_idx < len(flight.samples) - 1 and flight.samples[sample_idx + 1].time_s < t_target:
            sample_idx += 1

        s = flight.samples[sample_idx]
        points.append({
            "time_seconds": round(t_target, 6),
            "x_m": round(float(s.position_m[0]), 4),
            "y_m": round(float(s.position_m[1]), 4),
            "z_m": round(float(max(s.position_m[2], 0.0)), 4),
        })
        t_target += dt_frame

    return points


# ── Video Generation ──────────────────────────────────────────────────────────

def generate_scenario(
    config: ScenarioConfig,
    cam_a: VirtualCamera,
    cam_b: VirtualCamera,
    output_dir: Path,
    generate_calibration: bool = True,
) -> None:
    """Generate a complete test scenario: videos + ground truth."""
    scenario_dir = output_dir / config.name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Generating scenario: {config.name}")
    print(f"  {config.description}")
    print(f"  Output: {scenario_dir}")

    # 1. Simulate trajectory
    trajectory = simulate_trajectory(config)
    if not trajectory:
        print("  ERROR: Trajectory simulation produced no points!")
        return
    print(f"  Trajectory: {len(trajectory)} points, {trajectory[-1]['time_seconds']:.3f}s flight time")

    # 2. Generate calibration board images
    if generate_calibration:
        cal_dir = scenario_dir / "calibration"
        cal_dir.mkdir(exist_ok=True)
        generate_calibration_board(cam_a, cam_b, cal_dir)

    # 3. Project trajectory onto both cameras and write video frames
    track_a = []
    track_b = []
    bg = render_background()

    video_a_path = scenario_dir / "camera_a_silent.mp4"
    video_b_path = scenario_dir / "camera_b_silent.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_a = cv2.VideoWriter(str(video_a_path), fourcc, FPS, (WIDTH, HEIGHT))
    writer_b = cv2.VideoWriter(str(video_b_path), fourcc, FPS, (WIDTH, HEIGHT))

    prev_u_a, prev_v_a = None, None
    prev_u_b, prev_v_b = None, None

    for frame_idx in range(TOTAL_FRAMES):
        frame_a = bg.copy()
        frame_b = bg.copy()

        if frame_idx >= IMPACT_FRAME:
            traj_idx = frame_idx - IMPACT_FRAME
            if traj_idx < len(trajectory):
                pt = trajectory[traj_idx]
                p_world = np.array([pt["x_m"], pt["y_m"], pt["z_m"]])

                # Project to Camera A
                ua, va, da = cam_a.project(p_world)
                if da > 0 and 0 <= ua < WIDTH and 0 <= va < HEIGHT:
                    ra = cam_a.ball_radius_px(da)
                    speed_a = 0.0
                    angle_a = 0.0
                    if prev_u_a is not None:
                        dx = ua - prev_u_a
                        dy = va - prev_v_a
                        speed_a = np.sqrt(dx**2 + dy**2)
                        angle_a = np.arctan2(dy, dx)
                    frame_a = render_ball(frame_a, ua, va, ra, speed_a, angle_a)

                    track_a.append({
                        "frame_index": frame_idx,
                        "time_seconds": round(frame_idx / FPS, 6),
                        "x_px": round(ua, 2),
                        "y_px": round(va, 2),
                        "confidence": 0.95,
                    })
                    prev_u_a, prev_v_a = ua, va

                # Project to Camera B
                ub, vb, db = cam_b.project(p_world)
                if db > 0 and 0 <= ub < WIDTH and 0 <= vb < HEIGHT:
                    rb = cam_b.ball_radius_px(db)
                    speed_b = 0.0
                    angle_b = 0.0
                    if prev_u_b is not None:
                        dx = ub - prev_u_b
                        dy = vb - prev_v_b
                        speed_b = np.sqrt(dx**2 + dy**2)
                        angle_b = np.arctan2(dy, dx)
                    frame_b = render_ball(frame_b, ub, vb, rb, speed_b, angle_b)

                    track_b.append({
                        "frame_index": frame_idx,
                        "time_seconds": round(frame_idx / FPS, 6),
                        "x_px": round(ub, 2),
                        "y_px": round(vb, 2),
                        "confidence": 0.95,
                    })
                    prev_u_b, prev_v_b = ub, vb

        writer_a.write(frame_a)
        writer_b.write(frame_b)

    writer_a.release()
    writer_b.release()

    print(f"  Track A: {len(track_a)} points")
    print(f"  Track B: {len(track_b)} points")

    # 4. Generate audio and mux into videos
    print("  Generating audio and muxing...")
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "impact.wav"
        generate_impact_audio(wav_path, FPS, TOTAL_FRAMES, IMPACT_FRAME)

        final_a = scenario_dir / "camera_a.mp4"
        final_b = scenario_dir / "camera_b.mp4"

        success_a = mux_audio_into_video(video_a_path, wav_path, final_a)
        success_b = mux_audio_into_video(video_b_path, wav_path, final_b)

        if not success_a:
            print("  WARNING: ffmpeg audio mux failed for camera A, using silent video")
            video_a_path.rename(final_a)
        else:
            video_a_path.unlink(missing_ok=True)

        if not success_b:
            print("  WARNING: ffmpeg audio mux failed for camera B, using silent video")
            video_b_path.rename(final_b)
        else:
            video_b_path.unlink(missing_ok=True)

    # 5. Write ground truth JSON
    gt_path = scenario_dir / "ground_truth.json"
    gt_data = {
        "scenario": config.name,
        "description": config.description,
        "fps": FPS,
        "resolution": [WIDTH, HEIGHT],
        "total_frames": TOTAL_FRAMES,
        "impact_frame": IMPACT_FRAME,
        "launch_conditions": {
            "ball_speed_mps": config.ball_speed_mps,
            "launch_angle_deg": config.launch_angle_deg,
            "horizontal_launch_deg": config.horizontal_launch_deg,
            "backspin_rpm": config.backspin_rpm,
        },
        "calibration": {
            "coordinate_system": {
                "origin_in_rig_frame_m": [0.0, 0.0, 0.0],
                "target_direction_in_rig_frame": [1.0, 0.0, 0.0],
                "up_direction_in_rig_frame": [0.0, 0.0, 1.0],
                "alignment_method": "synthetic",
            },
            "camera_a_intrinsics": cam_a.intrinsics_matrix(),
            "camera_b_intrinsics": cam_b.intrinsics_matrix(),
            "camera_a_extrinsics": cam_a.extrinsics_matrix(),
            "camera_b_extrinsics": cam_b.extrinsics_matrix(),
            "reprojection_error_px": 0.0,
            "confidence": 1.0,
            "calibration_target": "synthetic",
            "is_valid": True,
        },
        "trajectory_3d": trajectory,
        "track_a_2d": track_a,
        "track_b_2d": track_b,
    }
    gt_path.write_text(json.dumps(gt_data, indent=2))
    print(f"  Ground truth: {gt_path}")

    # Also write a standalone calibration.json
    cal_json_path = scenario_dir / "calibration.json"
    cal_json_path.write_text(json.dumps(gt_data["calibration"], indent=2))
    print(f"  Calibration: {cal_json_path}")

    print(f"  Done! Videos at:")
    print(f"    {scenario_dir / 'camera_a.mp4'}")
    print(f"    {scenario_dir / 'camera_b.mp4'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic golf ball test videos")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "test_videos",
        help="Output directory (default: data/test_videos)",
    )
    parser.add_argument(
        "--scenario",
        choices=["all", "straight", "draw", "fade"],
        default="all",
        help="Which scenario(s) to generate (default: all)",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Skip calibration board image generation",
    )
    args = parser.parse_args()

    scenarios = SCENARIOS
    if args.scenario == "straight":
        scenarios = [SCENARIOS[0]]
    elif args.scenario == "draw":
        scenarios = [SCENARIOS[1]]
    elif args.scenario == "fade":
        scenarios = [SCENARIOS[2]]

    cam_a, cam_b = setup_stereo_cameras()

    print(f"Stereo camera setup:")
    print(f"  Camera A position: {np.linalg.inv(cam_a.extrinsics)[:3, 3].round(2)}")
    print(f"  Camera B position: {np.linalg.inv(cam_b.extrinsics)[:3, 3].round(2)}")
    print(f"  Focal length: {cam_a.fx}px")
    print(f"  Resolution: {WIDTH}x{HEIGHT} @ {FPS}fps")

    for config in scenarios:
        generate_scenario(
            config, cam_a, cam_b,
            args.output_dir,
            generate_calibration=not args.no_calibration,
        )

    print(f"\n{'=' * 60}")
    print("All scenarios generated successfully!")
    print(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
