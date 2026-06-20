import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api, ApiError } from "../lib/api";
import "../styles/forms.css";

const STAGES: { key: string; label: string }[] = [
  { key: "extracting_frames", label: "Extracting frames" },
  { key: "syncing", label: "Syncing videos" },
  { key: "detecting_impact", label: "Detecting impact" },
  { key: "detecting_ball", label: "Detecting ball" },
  { key: "tracking_ball", label: "Tracking ball" },
  { key: "triangulating", label: "Triangulating 3D position" },
  { key: "fitting_physics", label: "Fitting physics model" },
  { key: "rendering", label: "Preparing render" },
];

export function ProcessingPage() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);

  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;
    api
      .processSession(sessionId)
      .then(() => {
        if (!cancelled) setDone(true);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof ApiError ? err.message : "Processing failed.");
      });
    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  useEffect(() => {
    if (done && sessionId) {
      const t = setTimeout(() => navigate(`/sessions/${sessionId}/review`), 700);
      return () => clearTimeout(t);
    }
  }, [done, sessionId, navigate]);

  return (
    <div className="page">
      <div>
        <h1 className="page__title">Processing shot</h1>
        <p className="page__subtitle">
          v0 pipeline note: most stages below are structural placeholders (Milestones 1-7 are
          not implemented yet) -- this confirms both videos and reports honest, label-checked
          results rather than pretending to track or triangulate anything.
        </p>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div className="card" style={{ maxWidth: 480 }}>
        <ul style={{ listStyle: "none", margin: 0, padding: 0, display: "flex", flexDirection: "column", gap: 12 }}>
          {STAGES.map((stage) => (
            <li key={stage.key} style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <span
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  background: done || error ? "var(--color-turf-bright)" : "var(--color-muted-dim)",
                  flex: "none",
                }}
              />
              <span style={{ color: done ? "var(--color-ink)" : "var(--color-muted)" }}>{stage.label}</span>
            </li>
          ))}
        </ul>
      </div>

      {done && <p className="page__subtitle">Done. Taking you to the review screen…</p>}
    </div>
  );
}
