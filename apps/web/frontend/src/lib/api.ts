import type {
  CalibrationResult,
  PreviousSessionSummary,
  Session,
  ShotResult,
  SpinPreview,
  TrajectoryPayload,
  TrimJobResult,
} from "./types";

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

export function apiUrl(path: string): string {
  return path.startsWith("http://") || path.startsWith("https://")
    ? path
    : `${API_BASE_URL}${path}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init);
  } catch {
    throw new ApiError(
      0,
      `Could not reach the Golfie backend at ${API_BASE_URL}. Is it running? ` +
        `(uvicorn golfie_api.main:app --port 8000)`
    );
  }
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch {
      // response body wasn't JSON; fall back to statusText
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

export interface CreateSessionInput {
  environment?: "indoor" | "outdoor" | "unknown";
  club?: string;
  handedness?: string;
  ball_type?: string;
}

export const api = {
  trimVideoManually(
    file: File,
    startSeconds: number,
    endSeconds: number
  ): Promise<TrimJobResult> {
    const form = new FormData();
    form.append("file", file);
    form.append("start_seconds", startSeconds.toString());
    form.append("end_seconds", endSeconds.toString());
    return request<TrimJobResult>("/video-trimmer/trim", {
      method: "POST",
      body: form,
    });
  },

  getTrimResult(trimId: string): Promise<TrimJobResult> {
    return request<TrimJobResult>(`/video-trimmer/${trimId}`);
  },

  createSession(input: CreateSessionInput): Promise<Session> {
    return request<Session>("/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    });
  },

  listSessions(): Promise<string[]> {
    return request<string[]>("/sessions");
  },

  listPreviousSessions(): Promise<PreviousSessionSummary[]> {
    return request<PreviousSessionSummary[]>("/sessions/history");
  },

  getSession(sessionId: string): Promise<Session> {
    return request<Session>(`/sessions/${sessionId}`);
  },

  async uploadCamera(
    sessionId: string,
    camera: "camera-a" | "camera-b",
    file: File,
    roleHint?: string,
    deviceModel?: string,
    fpsOverride?: number,
    slowMotionFactor: number = 1
  ): Promise<Session> {
    const form = new FormData();
    form.append("file", file);
    if (roleHint) form.append("role_hint", roleHint);
    if (deviceModel) form.append("device_model", deviceModel);
    if (fpsOverride) form.append("fps_override", fpsOverride.toString());
    form.append("slow_motion_factor", slowMotionFactor.toString());
    return request<Session>(`/sessions/${sessionId}/upload/${camera}`, {
      method: "POST",
      body: form,
    });
  },


  processSession(sessionId: string): Promise<Session> {
    return request<Session>(`/sessions/${sessionId}/process`, { method: "POST" });
  },

  getStatus(sessionId: string): Promise<{ session_id: string; stage: string; error: string | null }> {
    return request(`/sessions/${sessionId}/status`);
  },

  getLogs(sessionId: string): Promise<string[]> {
    return request<string[]>(`/sessions/${sessionId}/logs`);
  },

  getResults(sessionId: string): Promise<ShotResult> {
    return request<ShotResult>(`/sessions/${sessionId}/results`);
  },

  getTrajectory(sessionId: string): Promise<TrajectoryPayload> {
    return request<TrajectoryPayload>(`/sessions/${sessionId}/trajectory`);
  },

  previewSpin(sessionId: string, backspinRpm: number, sidespinRpm: number): Promise<SpinPreview> {
    return request<SpinPreview>(`/sessions/${sessionId}/trajectory/spin-preview`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ backspin_rpm: backspinRpm, sidespin_rpm: sidespinRpm }),
    });
  },

  getDebugOverlays(sessionId: string): Promise<{ overlays: unknown[]; notes: string }> {
    return request(`/sessions/${sessionId}/debug/overlays`);
  },

  getSampleTrajectory(): Promise<TrajectoryPayload> {
    return request<TrajectoryPayload>("/demo/sample-trajectory");
  },

  getActiveCalibration(): Promise<CalibrationResult> {
    return request<CalibrationResult>("/calibration/active");
  },

  getCalibrationLogs(): Promise<string[]> {
    return request<string[]>("/calibration/logs");
  },

  getCalibrationStatus(): Promise<{ running: boolean; progress: number; stage: string; message: string }> {
    return request("/calibration/status");
  },

  async uploadAndCalibrate(
    fileA: File,
    fileB: File,
    boardType: string = "charuco",
    gridCols: number = 11,
    gridRows: number = 8,
    squareSize: number = 0.04,
    markerSize: number = 0.03,
      slowMotionFactorA: number = 1,
      slowMotionFactorB: number = 1,
    measuredBaseline?: number
  ): Promise<CalibrationResult> {
    const form = new FormData();
    form.append("file_a", fileA);
    form.append("file_b", fileB);
    form.append("board_type", boardType);
    form.append("grid_cols", gridCols.toString());
    form.append("grid_rows", gridRows.toString());
    form.append("square_size", squareSize.toString());
    form.append("marker_size", markerSize.toString());
    form.append("slow_motion_factor_a", slowMotionFactorA.toString());
    form.append("slow_motion_factor_b", slowMotionFactorB.toString());
    if (measuredBaseline !== undefined) {
      form.append("measured_baseline", measuredBaseline.toString());
    }

    return request<CalibrationResult>("/calibration/upload", {
      method: "POST",
      body: form,
    });
  },
};
