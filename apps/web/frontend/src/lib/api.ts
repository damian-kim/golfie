import type { Session, ShotResult, TrajectoryPayload } from "./types";

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
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

  getSession(sessionId: string): Promise<Session> {
    return request<Session>(`/sessions/${sessionId}`);
  },

  async uploadCamera(
    sessionId: string,
    camera: "camera-a" | "camera-b",
    file: File,
    roleHint?: string,
    deviceModel?: string
  ): Promise<Session> {
    const form = new FormData();
    form.append("file", file);
    if (roleHint) form.append("role_hint", roleHint);
    if (deviceModel) form.append("device_model", deviceModel);
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

  getResults(sessionId: string): Promise<ShotResult> {
    return request<ShotResult>(`/sessions/${sessionId}/results`);
  },

  getTrajectory(sessionId: string): Promise<TrajectoryPayload> {
    return request<TrajectoryPayload>(`/sessions/${sessionId}/trajectory`);
  },

  getDebugOverlays(sessionId: string): Promise<{ overlays: unknown[]; notes: string }> {
    return request(`/sessions/${sessionId}/debug/overlays`);
  },

  getSampleTrajectory(): Promise<TrajectoryPayload> {
    return request<TrajectoryPayload>("/demo/sample-trajectory");
  },
};
