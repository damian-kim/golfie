// Mirrors packages/golfie_core/schemas (Python/pydantic) so the frontend
// and backend never silently drift. If you change a field there, change
// it here too.

export type MetricSource = "measured" | "estimated" | "experimental" | "not_available";

export interface MetricValue {
  value: number | null;
  source: MetricSource;
  confidence: number;
  notes: string | null;
}

export interface ShotMetrics {
  ball_speed_mps: MetricValue;
  launch_angle_deg: MetricValue;
  horizontal_launch_deg: MetricValue;
  carry_m: MetricValue;
  total_m: MetricValue;
  apex_m: MetricValue;
  side_deviation_m: MetricValue;
  backspin_rpm: MetricValue;
  sidespin_rpm: MetricValue;
  spin_axis_deg: MetricValue;
  club_speed_mps: MetricValue;
  smash_factor: MetricValue;
}

export type Environment = "indoor" | "outdoor" | "unknown";

export type ProcessingStage =
  | "created"
  | "extracting_frames"
  | "syncing"
  | "detecting_impact"
  | "detecting_ball"
  | "tracking_ball"
  | "triangulating"
  | "fitting_physics"
  | "rendering"
  | "done"
  | "failed";

export interface CameraCapture {
  camera_id: string;
  device_model: string | null;
  fps: number;
  resolution: [number, number];
  video_path: string;
  role_hint: string | null;
  intrinsics: unknown | null;
  extrinsics: unknown | null;
}

export interface CalibrationResult {
  coordinate_system: unknown | null;
  camera_a_intrinsics: number[][] | null;
  camera_b_intrinsics: number[][] | null;
  camera_a_extrinsics: number[][] | null;
  camera_b_extrinsics: number[][] | null;
  reprojection_error_px: number | null;
  confidence: number;
  calibration_target: string | null;
  is_valid: boolean;
}

export interface SyncResult {
  offset_seconds: number;
  offset_frames: number;
  method: "audio_impact" | "visual_impact" | "manual" | "hybrid";
  confidence: number;
  debug_plot_path: string | null;
  notes: string | null;
}

export interface ShotResult {
  metrics: ShotMetrics;
  measured_points_3d: unknown[];
  fitted_points_3d: unknown[];
  simulated_trajectory_3d: unknown[];
  warnings: string[];
  is_placeholder: boolean;
  notes: string | null;
}

export interface Session {
  session_id: string;
  created_at: string;
  environment: Environment;
  stage: ProcessingStage;
  error: string | null;
  club: string | null;
  handedness: string | null;
  ball_type: string | null;
  camera_a: CameraCapture | null;
  camera_b: CameraCapture | null;
  calibration: CalibrationResult | null;
  sync: SyncResult | null;
  shot: ShotResult | null;
}

// --- Renderer payload (golfie_render.threejs.build_trajectory_payload) ---

export interface ScenePoint {
  t: number;
  /** downrange distance, meters */
  x: number;
  /** height, meters (Three.js "up") */
  y: number;
  /** lateral/side deviation, meters */
  z: number;
  confidence: number;
}

export interface TrajectoryPayload {
  session_id: string;
  club: string | null;
  is_placeholder: boolean;
  warnings: string[];
  notes: string | null;
  metrics: ShotMetrics;
  measured_points: ScenePoint[];
  fitted_points: ScenePoint[];
  simulated_trajectory: ScenePoint[];
}
