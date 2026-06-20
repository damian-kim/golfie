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
    <div className="page">
      <div>
        <h1 className="page__title">Simulator</h1>
        <p className="page__subtitle mono">session {sessionId}</p>
      </div>
      {error && <div className="error-banner">{error}</div>}
      {payload && <ShotSimulatorView payload={payload} />}
    </div>
  );
}
