import { useEffect, useState, useRef } from "react";
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
  const [activeStage, setActiveStage] = useState<string>("created");
  const [logs, setLogs] = useState<string[]>([]);
  const logContainerRef = useRef<HTMLDivElement>(null);

  // Trigger processing
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

  // Handle auto-redirect on done
  useEffect(() => {
    if (done) {
      const t = setTimeout(() => navigate(`/sessions/${sessionId}/review`), 1200);
      return () => clearTimeout(t);
    }
  }, [done, sessionId, navigate]);

  // Real-time status and logs polling
  useEffect(() => {
    if (!sessionId || done || error) return;

    const poll = () => {
      api.getStatus(sessionId)
        .then((status) => {
          if (status.stage) {
            setActiveStage(status.stage);
            if (status.stage === "done") {
              setDone(true);
            } else if (status.stage === "failed") {
              setError(status.error || "Processing failed.");
            }
          }
        })
        .catch(() => {
          // Ignore transient network errors
        });

      api.getLogs(sessionId)
        .then((res) => {
          setLogs(res);
        })
        .catch(() => {});
    };

    poll();
    const interval = setInterval(poll, 1200);
    return () => clearInterval(interval);
  }, [sessionId, done, error]);

  // Auto-scroll telemetry log to bottom
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]);

  const stageIndex = STAGES.findIndex((s) => s.key === activeStage);
  const activeIndex = done ? STAGES.length : stageIndex === -1 ? 0 : stageIndex;
  const progressPercent = done ? 100 : Math.max(5, Math.round((activeIndex / STAGES.length) * 100));

  return (
    <div className="page page--focused" style={{ padding: "40px 24px", display: "flex", flexDirection: "column", gap: 24 }}>
      <div>
        <h1 className="page__title">Processing shot</h1>
        <p className="page__subtitle">
          Real-time dual-camera telemetry processing. Watch progress metrics and transcode logs below.
        </p>
      </div>

      {error && (
        <div className="error-banner" style={{ marginBottom: 10 }}>
          <h4 style={{ margin: "0 0 6px 0", fontWeight: "bold" }}>Processing Pipeline Interrupted</h4>
          <p style={{ margin: 0 }}>{error}</p>
        </div>
      )}

      {/* Progress / Status Bar */}
      <div style={{ maxWidth: 520, width: "100%", margin: "0" }}>
        <div style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: "12px",
          color: "var(--color-muted)",
          marginBottom: "8px",
          fontWeight: 600
        }}>
          <span>Pipeline Progress</span>
          <span>{progressPercent}%</span>
        </div>
        <div style={{
          height: "8px",
          width: "100%",
          background: "rgba(255,255,255,0.04)",
          border: "1px solid rgba(255,255,255,0.06)",
          borderRadius: "4px",
          overflow: "hidden",
          position: "relative"
        }}>
          <div style={{
            height: "100%",
            width: `${progressPercent}%`,
            background: "linear-gradient(90deg, var(--color-turf) 0%, var(--color-turf-bright) 100%)",
            boxShadow: "0 0 8px var(--color-turf-bright)",
            transition: "width 0.4s cubic-bezier(0.1, 0.8, 0.2, 1)",
            borderRadius: "4px"
          }} />
        </div>
      </div>

      {/* Stage List Card */}
      <div className="card" style={{ maxWidth: 520, background: "rgba(8, 12, 16, 0.45)", borderColor: "rgba(152, 175, 199, 0.2)", margin: 0 }}>
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

      {/* Telemetry log terminal console */}
      <div className="card" style={{ maxWidth: 520, background: "#080c10", borderColor: "rgba(152, 175, 199, 0.2)", padding: "16px", margin: 0 }}>
        <h3 className="card__title" style={{ fontSize: "11px", color: "var(--color-muted)", marginBottom: "8px", display: "flex", justifyContent: "space-between", textTransform: "uppercase" }}>
          <span>Telemetry Stream</span>
          <span style={{ fontSize: "10px", color: "var(--color-turf-bright)", fontFamily: "var(--font-mono)" }}>LIVE_FEED</span>
        </h3>
        <div 
          ref={logContainerRef}
          style={{
            height: "160px",
            overflowY: "auto",
            background: "rgba(0,0,0,0.5)",
            border: "1px solid rgba(255,255,255,0.05)",
            borderRadius: "var(--radius-sm)",
            padding: "12px",
            fontFamily: "var(--font-mono)",
            fontSize: "11px",
            lineHeight: "1.6",
            color: "#e2e8f0",
            whiteSpace: "pre-wrap",
            display: "flex",
            flexDirection: "column",
            gap: "4px"
          }}
        >
          {logs.length === 0 ? (
            <span style={{ color: "var(--color-muted-dim)" }}>[system] Connecting to telemetry feed...</span>
          ) : (
            logs.map((log, i) => {
              let color = "#e2e8f0";
              if (log.toLowerCase().includes("fail") || log.toLowerCase().includes("error")) {
                color = "var(--color-danger)";
              } else if (log.toLowerCase().includes("complete") || log.toLowerCase().includes("success") || log.startsWith("✓")) {
                color = "var(--color-turf-bright)";
              } else if (log.toLowerCase().includes("starting") || log.toLowerCase().includes("processing") || log.toLowerCase().includes("transcoding")) {
                color = "var(--color-amber)";
              }
              return (
                <div key={i} style={{ color, display: "flex", alignItems: "flex-start" }}>
                  <span style={{ color: "var(--color-muted-dim)", marginRight: "8px", userSelect: "none" }}>&gt;</span>
                  <span style={{ flex: 1 }}>{log}</span>
                </div>
              );
            })
          )}
        </div>
      </div>

      {done && (
        <p className="page__subtitle" style={{ color: "var(--color-turf-bright)", fontWeight: "500", animation: "pulse-glowing 1.5s infinite" }}>
          ✓ Process completed successfully. Transferring to review dashboard…
        </p>
      )}
    </div>
  );
}
