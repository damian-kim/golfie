import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, ApiError } from "../lib/api";
import type { Session } from "../lib/types";
import { WarningsList } from "../components/WarningsList";
import "../styles/forms.css";

export function ReviewPage() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const [session, setSession] = useState<Session | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!sessionId) return;
    api
      .getSession(sessionId)
      .then(setSession)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Failed to load session."));
  }, [sessionId]);

  if (error) {
    return (
      <div className="page">
        <div className="error-banner">{error}</div>
      </div>
    );
  }

  if (!session) {
    return (
      <div className="page">
        <p className="page__subtitle">Loading session…</p>
      </div>
    );
  }

  return (
    <div className="page">
      <div>
        <h1 className="page__title">Review &amp; debug</h1>
        <p className="page__subtitle mono">
          session {session.session_id} · stage: {session.stage}
        </p>
      </div>

      {session.shot && <WarningsList warnings={session.shot.warnings} />}

      <div className="form-grid">
        <CameraSummary label="Camera A" camera={session.camera_a} />
        <CameraSummary label="Camera B" camera={session.camera_b} />
      </div>

      <div className="form-grid">
        <div className="card">
          <h2 className="card__title">Calibration</h2>
          {session.calibration && session.calibration.is_valid ? (
            <p>
              Valid · reprojection error{" "}
              {session.calibration.reprojection_error_px?.toFixed(2) ?? "—"} px
            </p>
          ) : (
            <p className="page__subtitle" style={{ margin: 0 }}>
              No valid calibration attached yet. (Milestone 1 -- intrinsic + stereo calibration --
              is not implemented yet.)
            </p>
          )}
        </div>
        <div className="card">
          <h2 className="card__title">Synchronization</h2>
          {session.sync ? (
            <p>
              Offset {session.sync.offset_seconds.toFixed(3)}s ({session.sync.method}), confidence{" "}
              {Math.round(session.sync.confidence * 100)}%
            </p>
          ) : (
            <p className="page__subtitle" style={{ margin: 0 }}>
              No sync result attached yet. (Milestone 2 -- audio-based sync -- is not implemented
              yet.)
            </p>
          )}
        </div>
      </div>

      <div className="card">
        <h2 className="card__title">Debug overlays</h2>
        <p className="page__subtitle" style={{ margin: 0 }}>
          Ball-detection overlay videos will appear here starting at Milestone 3.
        </p>
      </div>

      <div>
        <Link to={`/sessions/${session.session_id}/simulator`} className="primary-button" style={{ display: "inline-block" }}>
          Continue to simulator →
        </Link>
      </div>
    </div>
  );
}

function CameraSummary({ label, camera }: { label: string; camera: Session["camera_a"] }) {
  return (
    <div className="card">
      <h2 className="card__title">{label}</h2>
      {camera ? (
        <ul style={{ margin: 0, padding: 0, listStyle: "none", display: "flex", flexDirection: "column", gap: 6 }}>
          <li className="mono">{camera.fps.toFixed(1)} fps</li>
          <li className="mono">
            {camera.resolution[0]}×{camera.resolution[1]}
          </li>
          {camera.role_hint && <li>{camera.role_hint.replace("_", " ")}</li>}
          {camera.device_model && <li>{camera.device_model}</li>}
        </ul>
      ) : (
        <p className="page__subtitle" style={{ margin: 0 }}>
          Not uploaded.
        </p>
      )}
    </div>
  );
}
