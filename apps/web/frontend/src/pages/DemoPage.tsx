import { useEffect, useState } from "react";
import { api, ApiError } from "../lib/api";
import type { TrajectoryPayload } from "../lib/types";
import { ShotSimulatorView } from "../components/ShotSimulatorView";
import "../styles/forms.css";

export function DemoPage() {
  const [payload, setPayload] = useState<TrajectoryPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getSampleTrajectory()
      .then(setPayload)
      .catch((err) =>
        setError(
          err instanceof ApiError
            ? `${err.message} (run: python scripts/generate_sample_session.py)`
            : "Failed to load demo trajectory."
        )
      );
  }, []);

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
          title="Demo Driving Range"
          subtitle="Synthetic flight profile utilized for WebGL rendering validation."
        />
      )}
    </div>
  );
}
