import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { api, ApiError } from "../lib/api";
import type { TrajectoryPayload } from "../lib/types";
import { ShotSimulatorView } from "../components/ShotSimulatorView";
import "../styles/forms.css";

export function SimulatorPage() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();
  const [payload, setPayload] = useState<TrajectoryPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!sessionId) return;
    api
      .getTrajectory(sessionId)
      .then(setPayload)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Failed to load trajectory."));
  }, [sessionId]);

  return (
    <div className="page page--full">
      <button
        type="button"
        className="primary-button"
        style={{
          position: "absolute",
          top: "20px",
          left: "20px",
          zIndex: 100,
          fontSize: "12px",
          padding: "6px 12px",
          background: "rgba(8, 12, 16, 0.72)",
          borderColor: "rgba(255, 255, 255, 0.15)",
          borderRadius: "var(--radius-sm)",
          cursor: "pointer",
        }}
        onClick={() => navigate(`/sessions/${sessionId}/review`)}
      >
        ← Back to Review
      </button>
      {error && (
        <div
          className="error-banner"
          style={{
            position: "absolute",
            top: "20px",
            left: "50%",
            transform: "translateX(-50%)",
            zIndex: 100,
            border: "1px solid var(--color-danger)",
            boxShadow: "0 4px 20px rgba(0, 0, 0, 0.5)",
            borderRadius: "var(--radius-sm)",
          }}
        >
          {error}
        </div>
      )}
      {payload && (
        <ShotSimulatorView
          payload={payload}
          title="Shot Simulator"
          subtitle={`Telemetry & Trajectory Replay · Session ${sessionId}`}
        />
      )}
    </div>
  );
}
