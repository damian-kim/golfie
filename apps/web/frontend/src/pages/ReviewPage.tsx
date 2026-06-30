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
    <div className="page page--focused" style={{ padding: "40px 24px" }}>
      <div>
        <h1 className="page__title">Review &amp; debug</h1>
        <p className="page__subtitle mono" style={{ fontSize: "11px", letterSpacing: "0.05em", color: "var(--color-muted)" }}>
          SESSION ID: {session.session_id} · STAGE: {session.stage.toUpperCase()}
        </p>
      </div>

      {session.shot && <WarningsList warnings={session.shot.warnings} />}

      <div className="form-grid">
        <CameraSummary label="Camera A" camera={session.camera_a} />
        <CameraSummary label="Camera B" camera={session.camera_b} />
      </div>

      <div className="form-grid">
        <div className="card" style={{ borderColor: "rgba(139, 124, 224, 0.25)" }}>
          <h2 className="card__title" style={{ color: "var(--color-violet)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span>Calibration</span>
            <span className="field__label" style={{ color: "rgba(139, 124, 224, 0.6)", fontSize: "9px" }}>FITTED GEOMETRY</span>
          </h2>
          {session.calibration && session.calibration.is_valid ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: "var(--color-muted)" }}>Status</span>
                <span style={{ color: "var(--color-turf-bright)", fontWeight: "bold" }}>VALID RIG</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: "var(--color-muted)" }}>Reprojection Error</span>
                <span className="mono" style={{ color: "var(--color-ink)", fontWeight: "bold" }}>
                  {session.calibration.reprojection_error_px?.toFixed(3) ?? "—"} px
                </span>
              </div>
            </div>
          ) : (
            <p className="page__subtitle" style={{ margin: 0, color: "var(--color-muted-dim)" }}>
              No active calibration configuration. Fallback simulated metrics active.
            </p>
          )}
        </div>

        <div className="card" style={{ borderColor: "rgba(76, 194, 115, 0.25)" }}>
          <h2 className="card__title" style={{ color: "var(--color-turf-bright)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span>Synchronization</span>
            <span className="field__label" style={{ color: "rgba(76, 194, 115, 0.6)", fontSize: "9px" }}>ESTIMATED ALIGNMENT</span>
          </h2>
          {session.sync ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: "var(--color-muted)" }}>Time Offset</span>
                <span className="mono" style={{ color: "var(--color-ink)", fontWeight: "bold" }}>
                  {session.sync.offset_seconds >= 0 ? "+" : ""}{session.sync.offset_seconds.toFixed(4)}s
                </span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: "var(--color-muted)" }}>Sync Method</span>
                <span style={{ color: "var(--color-ink)", fontWeight: "500" }}>
                  {session.sync.method.toUpperCase()}
                </span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: "var(--color-muted)" }}>Confidence</span>
                <span className="mono" style={{ color: "var(--color-turf-bright)", fontWeight: "bold" }}>
                  {Math.round(session.sync.confidence * 100)}%
                </span>
              </div>
            </div>
          ) : (
            <p className="page__subtitle" style={{ margin: 0, color: "var(--color-muted-dim)" }}>
              No synchronization details computed.
            </p>
          )}
        </div>
      </div>

      <div className="card" style={{ borderColor: "rgba(255, 255, 255, 0.08)", background: "rgba(10, 16, 13, 0.45)" }}>
        <h2 className="card__title" style={{ color: "var(--color-muted)" }}>Computer Vision Debug Feeds</h2>
        <p className="page__subtitle" style={{ margin: 0, color: "var(--color-muted-dim)" }}>
          Overlay video feeds highlighting impact frames, ball detection regions, and 2D tracking trajectories will stream here in future milestones.
        </p>
      </div>

      <div style={{ marginTop: 8 }}>
        <Link to={`/sessions/${session.session_id}/simulator`} className="primary-button" style={{ display: "inline-block" }}>
          Continue to simulator →
        </Link>
      </div>
    </div>
  );
}

function CameraSummary({ label, camera }: { label: string; camera: Session["camera_a"] }) {
  return (
    <div className="card" style={{ borderColor: "rgba(232, 162, 58, 0.2)" }}>
      <h2 className="card__title" style={{ color: "var(--color-amber)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span>{label}</span>
        <span className="field__label" style={{ color: "rgba(232, 162, 58, 0.6)", fontSize: "9px" }}>MEASURED FEED</span>
      </h2>
      {camera ? (
        <ul style={{ margin: 0, padding: 0, listStyle: "none", display: "flex", flexDirection: "column", gap: 8 }}>
          <li style={{ display: "flex", justifyContent: "space-between", fontSize: "13px" }}>
            <span style={{ color: "var(--color-muted)" }}>Frame Rate</span>
            <span className="mono" style={{ color: "var(--color-ink)", fontWeight: "bold" }}>{camera.fps.toFixed(1)} fps</span>
          </li>
          <li style={{ display: "flex", justifyContent: "space-between", fontSize: "13px" }}>
            <span style={{ color: "var(--color-muted)" }}>Resolution</span>
            <span className="mono" style={{ color: "var(--color-ink)", fontWeight: "bold" }}>{camera.resolution[0]}×{camera.resolution[1]}</span>
          </li>
          {camera.role_hint && (
            <li style={{ display: "flex", justifyContent: "space-between", fontSize: "13px" }}>
              <span style={{ color: "var(--color-muted)" }}>Role Hint</span>
              <span style={{ color: "var(--color-ink)", fontWeight: "500" }}>{camera.role_hint.replace("_", " ")}</span>
            </li>
          )}
          {camera.device_model && (
            <li style={{ display: "flex", justifyContent: "space-between", fontSize: "13px" }}>
              <span style={{ color: "var(--color-muted)" }}>Device</span>
              <span style={{ color: "var(--color-ink)", fontWeight: "500" }}>{camera.device_model}</span>
            </li>
          )}
        </ul>
      ) : (
        <p className="page__subtitle" style={{ margin: 0, color: "var(--color-muted-dim)" }}>
          No recording uploaded.
        </p>
      )}
    </div>
  );
}
