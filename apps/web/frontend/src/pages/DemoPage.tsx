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
    <div className="page">
      <div>
        <h1 className="page__title">Demo driving range</h1>
        <p className="page__subtitle">
          A synthetic example shot used to develop and showcase the renderer -- not derived from
          any real video. Use this to verify the 3D scene before reconstructed shots exist.
        </p>
      </div>
      {error && <div className="error-banner">{error}</div>}
      {payload && <ShotSimulatorView payload={payload} />}
    </div>
  );
}
