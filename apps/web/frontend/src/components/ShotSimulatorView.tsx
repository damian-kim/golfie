import { useEffect, useState } from "react";
import type { TrajectoryPayload } from "../lib/types";
import { MetricCard } from "./MetricCard";
import { formatMetric } from "../lib/units";
import { DrivingRangeScene } from "../scenes/DrivingRangeScene";
import { API_BASE_URL } from "../lib/api";
import { SwingReplayModal } from "./SwingReplayModal";
import "./ShotSimulatorView.css";

interface ShotSimulatorViewProps {
  payload: TrajectoryPayload;
  title?: string;
  subtitle?: string;
}

export function ShotSimulatorView({ payload, title, subtitle }: ShotSimulatorViewProps) {
  const { metrics } = payload;
  const [showPrecursor, setShowPrecursor] = useState(false);
  const [videoError, setVideoError] = useState<string | null>(null);
  const [outlinesReady, setOutlinesReady] = useState(false);
  const [outlineStatus, setOutlineStatus] = useState("Checking outline availability...");
  const [outlinesRendering, setOutlinesRendering] = useState(false);
  const [outlineRefreshToken, setOutlineRefreshToken] = useState(0);
  const [playToken, setPlayToken] = useState(0);

  useEffect(() => {
    if (payload.session_id === "sample") return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const poll = async (queueIfIdle: boolean) => {
      try {
        const response = await fetch(`${API_BASE_URL}/sessions/${payload.session_id}/artifacts`);
        if (!response.ok) throw new Error(`artifact status request failed (${response.status})`);
        let status = await response.json();
        if (queueIfIdle && !status.outline_ready && status.outline_render_status?.stage === "idle") {
          const queued = await fetch(`${API_BASE_URL}/sessions/${payload.session_id}/artifacts/outlines`, { method: "POST" });
          if (!queued.ok) throw new Error(`outline generation request failed (${queued.status})`);
          status = { ...status, outline_render_status: await queued.json() };
        }
        if (cancelled) return;
        const ready = Boolean(status.outline_ready);
        const job = status.outline_render_status || {};
        setOutlinesReady(ready);
        setOutlinesRendering(Boolean(job.running));
        setOutlineStatus(ready
          ? "Swing outline videos are ready."
          : job.error || job.message || status.outline_unavailable_reason || "Swing outline videos are unavailable.");
        if (!ready && job.running) timer = setTimeout(() => void poll(false), 2000);
      } catch (error) {
        if (cancelled) return;
        setOutlinesReady(false);
        setOutlinesRendering(false);
        setOutlineStatus(`Could not generate swing outlines: ${error instanceof Error ? error.message : String(error)}`);
      }
    };
    void poll(true);
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [payload.session_id, outlineRefreshToken]);

  const handleOutlineClick = async () => {
    if (outlinesReady) {
      setVideoError(null);
      setShowPrecursor(true);
      return;
    }
    setOutlinesRendering(true);
    setOutlineStatus("Swing outline rendering queued.");
    try {
      const response = await fetch(`${API_BASE_URL}/sessions/${payload.session_id}/artifacts/outlines`, { method: "POST" });
      if (!response.ok) throw new Error(`request failed (${response.status})`);
      setOutlineRefreshToken((token) => token + 1);
    } catch (error) {
      setOutlinesRendering(false);
      setOutlineStatus(`Could not generate swing outlines: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  const handleSkip = () => {
    setShowPrecursor(false);
    setVideoError(null);
  };

  const handleVideoError = () => {
    setVideoError("Outline artifacts were reported ready, but a video could not be decoded. Check the session artifact status and processing log.");
  };

  const handleReplayClick = () => {
    setPlayToken((t) => t + 1);
  };

  return (
    <div className="shot-simulator-hud">
      {/* Immersive Full-Screen Canvas behind the HUD overlay */}
      <div className="shot-simulator-hud__canvas-container">
        <DrivingRangeScene
          key={payload.session_id}
          simulated={payload.simulated_trajectory}
          measured={payload.measured_points}
          fitted={payload.fitted_points}
          playToken={playToken}
          onReplayClick={handleReplayClick}
        />
      </div>

      {/* Top HUD Alert Deck */}
      {(payload.is_placeholder || payload.warnings.length > 0) && (
        <div className="shot-simulator-hud__alerts">
          {payload.is_placeholder && (
            <div className="hud-alert-banner hud-alert-banner--placeholder">
              <span className="hud-alert-banner__tag">METRIC SOURCE: DEMO</span>
              <p className="hud-alert-banner__message">
                {payload.notes ?? "Telemetry calculated utilizing simulation parameters."}
              </p>
            </div>
          )}
          {payload.warnings.map((w, idx) => (
            <div key={idx} className="hud-alert-banner hud-alert-banner--warning">
              <span className="hud-alert-banner__tag">WARN //</span>
              <p className="hud-alert-banner__message">{w}</p>
            </div>
          ))}
        </div>
      )}

      {/* Left Control Panel */}
      <div className="shot-simulator-hud__panel shot-simulator-hud__panel--left">
        <div className="hud-header">
          <h2 className="hud-header__title">{title || "Shot Replay"}</h2>
          <p className="hud-header__subtitle mono">{subtitle || "Telemetry Dashboard"}</p>
        </div>

        <div className="hud-section">
          <div className="hud-section__title">Equipment Profile</div>
          <div className="hud-club-widget">
            <div className="hud-club-widget__badge">CLUB</div>
            <div className="hud-club-widget__details">
              <div className="hud-club-widget__name">{payload.club || "Driver"}</div>
              <div className="hud-club-widget__class">Graphite Shaft / Standard Grip</div>
            </div>
          </div>
        </div>

        <div className="hud-section">
          <div className="hud-section__title">Environmental Factors</div>
          <div className="hud-env-grid">
            <div className="hud-env-item">
              <span className="hud-env-item__label">Wind</span>
              <span className="hud-env-item__val">0.0 mph</span>
            </div>
            <div className="hud-env-item">
              <span className="hud-env-item__label">Elevation</span>
              <span className="hud-env-item__val">Sea Level</span>
            </div>
            <div className="hud-env-item">
              <span className="hud-env-item__label">Temp</span>
              <span className="hud-env-item__val">70 °F</span>
            </div>
          </div>
        </div>

        {payload.session_id !== "sample" && (
          <div className="hud-section" style={{ marginTop: "8px" }}>
            <button 
              className="primary-button" 
              style={{
                width: "100%",
                padding: "8px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: "8px",
                background: "rgba(76, 194, 115, 0.12)",
                borderColor: "rgba(76, 194, 115, 0.4)",
                color: "#ffffff"
              }}
              onClick={handleOutlineClick}
              disabled={outlinesRendering}
              title={outlineStatus}
            >
              <span>{outlinesReady ? "Open Swing Comparison" : outlinesRendering ? "Rendering Swing Comparison..." : "Generate Swing Comparison"}</span>
            </button>
            {!outlinesReady && (
              <p className="mono" style={{ fontSize: "10px", lineHeight: 1.4, opacity: 0.7, margin: "8px 0 0" }}>
                {outlineStatus}
              </p>
            )}
          </div>
        )}
      </div>

      {/* Right Telemetry Panel */}
      <div className="shot-simulator-hud__panel shot-simulator-hud__panel--right">
        <div className="hud-section__title" style={{ marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.08)", paddingBottom: "6px" }}>
          Ball Launch Telemetry
        </div>
        
        <div className="shot-simulator-hud__metrics-scroll">
          <MetricCard label="Ball speed" metric={metrics.ball_speed_mps} unit="mph" format={formatMetric} />
          <MetricCard label="Launch angle" metric={metrics.launch_angle_deg} unit="deg" format={formatMetric} />
          <MetricCard label="Launch direction" metric={metrics.horizontal_launch_deg} unit="deg" format={formatMetric} />
          <MetricCard label="Carry distance" metric={metrics.carry_m} unit="yd" format={formatMetric} />
          <MetricCard label="Total distance" metric={metrics.total_m} unit="yd" format={formatMetric} />
          <MetricCard label="Apex height" metric={metrics.apex_m} unit="yd" format={formatMetric} />
          <MetricCard label="Side deviation" metric={metrics.side_deviation_m} unit="yd" format={formatMetric} />
        </div>
      </div>

      {/* Visual Telemetry Radar Widget */}
      <div className="hud-radar-widget">
        <div className="hud-radar-header">LAUNCH RADAR</div>
        <svg viewBox="0 0 100 100" className="hud-radar-svg">
          {/* Outer Ring */}
          <circle cx="50" cy="50" r="45" stroke="var(--hud-border)" strokeWidth="1" fill="rgba(10, 16, 13, 0.4)" />
          {/* Inner Grid Rings */}
          <circle cx="50" cy="50" r="30" stroke="rgba(255, 255, 255, 0.05)" strokeWidth="1" strokeDasharray="3 3" fill="none" />
          <circle cx="50" cy="50" r="15" stroke="rgba(255, 255, 255, 0.03)" strokeWidth="1" fill="none" />
          
          {/* Crosshairs */}
          <line x1="50" y1="5" x2="50" y2="95" stroke="rgba(255, 255, 255, 0.08)" strokeWidth="0.8" />
          <line x1="5" y1="50" x2="95" y2="50" stroke="rgba(255, 255, 255, 0.08)" strokeWidth="0.8" />
          
          {/* 45-deg Guidelines */}
          <line x1="18.2" y1="18.2" x2="81.8" y2="81.8" stroke="rgba(255, 255, 255, 0.03)" strokeWidth="0.6" strokeDasharray="1 3" />
          <line x1="18.2" y1="81.8" x2="81.8" y2="18.2" stroke="rgba(255, 255, 255, 0.03)" strokeWidth="0.6" strokeDasharray="1 3" />
          
          {/* Direction Indicator Line */}
          {metrics.horizontal_launch_deg?.value !== null && (
            <g>
              <line 
                x1="50" 
                y1="50" 
                x2={50 + 38 * Math.sin(((metrics.horizontal_launch_deg.value || 0)) * Math.PI / 180)} 
                y2={50 - 38 * Math.cos(((metrics.horizontal_launch_deg.value || 0)) * Math.PI / 180)} 
                stroke="var(--color-turf-bright)" 
                strokeWidth="2.5" 
                strokeLinecap="round"
              />
              <circle 
                cx={50 + 38 * Math.sin(((metrics.horizontal_launch_deg.value || 0)) * Math.PI / 180)} 
                cy={50 - 38 * Math.cos(((metrics.horizontal_launch_deg.value || 0)) * Math.PI / 180)} 
                r="3.5" 
                fill="var(--color-turf-bright)"
              />
            </g>
          )}
        </svg>
        <div className="hud-radar-footer mono">
          <div>DIR: {metrics.horizontal_launch_deg?.value !== null ? `${(metrics.horizontal_launch_deg.value || 0).toFixed(1)}°` : "N/A"}</div>
          <div>LA: {metrics.launch_angle_deg?.value !== null ? `${(metrics.launch_angle_deg.value || 0).toFixed(1)}°` : "N/A"}</div>
        </div>
      </div>

      {/* Precursor Outlines Replay Overlay */}
      {showPrecursor && (
        <SwingReplayModal
          sessionId={payload.session_id}
          onClose={handleSkip}
          onVideoError={handleVideoError}
          error={videoError}
          onPlayBallFlight={() => {
            handleSkip();
            setPlayToken((token) => token + 1);
          }}
        />
      )}
    </div>
  );
}
