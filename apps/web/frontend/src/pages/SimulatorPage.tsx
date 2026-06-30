import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { api, ApiError } from "../lib/api";
import type { TrajectoryPayload } from "../lib/types";
import { ShotSimulatorView } from "../components/ShotSimulatorView";
import "../styles/forms.css";

export function SimulatorPage() {
  const { sessionId } = useParams<{ sessionId: string }>();
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
      {error && (
        <div className="error-banner" style={{ position: "absolute", top: "20px", left: "50%", transform: "translateX(-50%)", zIndex: 100, boxShadow: "0 4px 20px rgba(226, 87, 76, 0.3)" }}>
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
