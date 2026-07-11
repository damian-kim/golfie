import { useEffect, useState } from "react";
import type { TrajectoryPayload } from "../lib/types";
import { MetricCard } from "./MetricCard";
import { formatMetric } from "../lib/units";
import { DrivingRangeScene } from "../scenes/DrivingRangeScene";
import { API_BASE_URL } from "../lib/api";
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
  const [playToken, setPlayToken] = useState(0);

  const videoAUrl = `${API_BASE_URL}/sessions/${payload.session_id}/video/camera_a/stripped`;
  const videoBUrl = `${API_BASE_URL}/sessions/${payload.session_id}/video/camera_b/stripped`;

  useEffect(() => {
    if (payload.session_id === "sample") return;
    let cancelled = false;
    fetch(`${API_BASE_URL}/sessions/${payload.session_id}/artifacts`)
      .then(async (response) => {
        if (!response.ok) throw new Error(`artifact status request failed (${response.status})`);
        return response.json();
      })
      .then((status) => {
        if (cancelled) return;
        const ready = Boolean(status.outline_ready);
        setOutlinesReady(ready);
        setOutlineStatus(ready
          ? "Swing outline videos are ready."
          : status.outline_unavailable_reason || "Swing outline videos are unavailable.");
      })
      .catch((error) => {
        if (cancelled) return;
        setOutlinesReady(false);
        setOutlineStatus(`Could not check outline availability: ${error.message}`);
      });
    return () => { cancelled = true; };
  }, [payload.session_id]);

  const handleVideoEnded = () => {
    setShowPrecursor(false);
    setVideoError(null);
    setPlayToken((t) => t + 1);
  };

  const handleSkip = () => {
    setShowPrecursor(false);
    setVideoError(null);
    setPlayToken((t) => t + 1);
  };

  const handleVideoError = () => {
    setVideoError("Outline artifacts were reported ready, but a video could not be decoded. Check the session artifact status and processing log.");
  };

  const handleReplayClick = () => {
    if (payload.session_id !== "sample" && outlinesReady) {
      setVideoError(null);
      setShowPrecursor(true);
    } else {
      setPlayToken((t) => t + 1);
    }
  };

  return (
    <div className="shot-simulator-hud">
      {/* Immersive Full-Screen Canvas behind the HUD overlay */}
      <div className="shot-simulator-hud__canvas-container">
        <DrivingRangeScene
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
              onClick={handleReplayClick}
              disabled={!outlinesReady}
              title={outlineStatus}
            >
              <span>{outlinesReady ? "Replay Swing Outlines" : "Swing Outlines Unavailable"}</span>
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
        <div className="shot-simulator-precursor">
          <div className="precursor-overlay__content">
            <div className="precursor-header">
              <div className="precursor-header__title">Swing Motion Outlines</div>
              <div className="precursor-header__subtitle mono">YOLOv8 DETECTED SWING PATHS</div>
            </div>
            
            {videoError ? (
              <div style={{ padding: "48px 24px", textAlign: "center", display: "flex", flexDirection: "column", alignItems: "center", gap: 16 }}>
                <span style={{ fontSize: "32px" }}>⚠️</span>
                <p style={{ color: "var(--color-danger)", fontSize: "14px", maxWidth: "400px", margin: 0, lineHeight: 1.5 }}>
                  {videoError}
                </p>
                <button className="primary-button" style={{ marginTop: "12px" }} onClick={handleSkip}>
                  Continue to Simulator
                </button>
              </div>
            ) : (
              <div className="precursor-videos">
                <div className="precursor-video-wrapper camera-a">
                  <div className="precursor-video-label">CAMERA A · DOWN-THE-LINE</div>
                  <video 
                    src={videoAUrl} 
                    autoPlay 
                    muted 
                    playsInline
                    onEnded={handleVideoEnded}
                    onError={handleVideoError}
                    className="precursor-video"
                  />
                </div>
                <div className="precursor-video-wrapper camera-b">
                  <div className="precursor-video-label">CAMERA B · FACE-ON</div>
                  <video 
                    src={videoBUrl} 
                    autoPlay 
                    muted 
                    playsInline
                    onError={handleVideoError}
                    className="precursor-video"
                  />
                </div>
              </div>
            )}

            <div className="precursor-footer">
              <div className="precursor-progress-bar">
                <div className="precursor-progress-fill" style={{ animationDuration: '5s' }} />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '12px' }}>
                <span className="precursor-telemetry mono">SWING SYNC PATTERNS ACTIVE ... 100%</span>
                <button className="precursor-skip-btn" onClick={handleSkip}>
                  SKIP REPLAY & FLY →
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
