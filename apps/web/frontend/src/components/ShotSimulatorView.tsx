import type { TrajectoryPayload } from "../lib/types";
import { MetricCard } from "./MetricCard";
import { formatMetric } from "../lib/units";
import { DrivingRangeScene } from "../scenes/DrivingRangeScene";
import "./ShotSimulatorView.css";

interface ShotSimulatorViewProps {
  payload: TrajectoryPayload;
  title?: string;
  subtitle?: string;
}

export function ShotSimulatorView({ payload, title, subtitle }: ShotSimulatorViewProps) {
  const { metrics } = payload;
  return (
    <div className="shot-simulator-hud">
      {/* Immersive Full-Screen Canvas behind the HUD overlay */}
      <div className="shot-simulator-hud__canvas-container">
        <DrivingRangeScene
          simulated={payload.simulated_trajectory}
          measured={payload.measured_points}
          fitted={payload.fitted_points}
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
    </div>
  );
}
