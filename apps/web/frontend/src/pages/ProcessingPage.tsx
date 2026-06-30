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
  const [activeIndex, setActiveIndex] = useState(0);

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
    if (done) {
      setActiveIndex(STAGES.length);
      const t = setTimeout(() => navigate(`/sessions/${sessionId}/review`), 1200);
      return () => clearTimeout(t);
    }
  }, [done, sessionId, navigate]);

  useEffect(() => {
    if (done || error) return;
    const interval = setInterval(() => {
      setActiveIndex((prev) => {
        if (prev < STAGES.length - 1) {
          return prev + 1;
        }
        return prev;
      });
    }, 900);
    return () => clearInterval(interval);
  }, [done, error]);

  return (
    <div className="page page--focused" style={{ padding: "40px 24px" }}>
      <div>
        <h1 className="page__title">Processing shot</h1>
        <p className="page__subtitle">
          v0 pipeline note: most stages below are structural placeholders (Milestones 1-7 are
          not implemented yet) -- this confirms both videos and reports honest, label-checked
          results rather than pretending to track or triangulate anything.
        </p>
      </div>

      {error && (
        <div className="error-banner" style={{ marginBottom: 10 }}>
          <h4 style={{ margin: "0 0 6px 0", fontWeight: "bold" }}>Processing Pipeline Interrupted</h4>
          <p style={{ margin: 0 }}>{error}</p>
        </div>
      )}

      <div className="card" style={{ maxWidth: 520, background: "rgba(10, 16, 13, 0.85)", borderColor: "rgba(76, 194, 115, 0.2)" }}>
        <div style={{ display: "flex", justifyContent: "space-between", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: 10, marginBottom: 12 }}>
          <span className="field__label" style={{ color: "var(--color-muted)" }}>Pipeline Phase</span>
          <span className="field__label" style={{ color: "var(--color-muted)" }}>Telemetry Status</span>
        </div>

        <ul style={{ listStyle: "none", margin: 0, padding: 0, display: "flex", flexDirection: "column", gap: 14 }}>
          {STAGES.map((stage, idx) => {
            let statusText = "PENDING";
            let color = "var(--color-muted-dim)";
            let isCurrent = idx === activeIndex && !done && !error;
            let isCompleted = idx < activeIndex || done;
            let isFailed = idx === activeIndex && error;

            if (isCompleted) {
              statusText = "COMPLETED";
              color = "var(--color-turf-bright)";
            } else if (isCurrent) {
              statusText = "RUNNING";
              color = "var(--color-amber)";
            } else if (isFailed) {
              statusText = "FAILED";
              color = "var(--color-danger)";
            }

            return (
              <li
                key={stage.key}
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  fontSize: "13px",
                  color: isCurrent ? "var(--color-ink)" : isCompleted ? "rgba(255,255,255,0.85)" : "var(--color-muted-dim)",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                  <span
                    className={isCurrent || isFailed ? "pulse-glowing" : ""}
                    style={{
                      width: 8,
                      height: 8,
                      borderRadius: "50%",
                      background: color,
                      color: color,
                      flex: "none",
                    }}
                  />
                  <span>{stage.label}</span>
                </div>
                <span
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: "11px",
                    fontWeight: "bold",
                    color: color,
                    letterSpacing: "0.05em",
                  }}
                >
                  [{statusText}]
                </span>
              </li>
            );
          })}
        </ul>
      </div>

      {done && (
        <p className="page__subtitle" style={{ color: "var(--color-turf-bright)", fontWeight: "500", animation: "pulse-glowing 1.5s infinite" }}>
          ✓ Process completed successfully. Transferring to review dashboard…
        </p>
      )}
    </div>
  );
}
