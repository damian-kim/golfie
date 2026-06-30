import type { MetricValue } from "../lib/types";
import { SourceBadge } from "./SourceBadge";
import "./MetricCard.css";

interface MetricCardProps {
  label: string;
  metric: MetricValue;
  unit: "mph" | "yd" | "deg" | "rpm" | "x";
  format: (value: number | null, unit: MetricCardProps["unit"]) => string;
}

export function MetricCard({ label, metric, unit, format }: MetricCardProps) {
  const isAvailable = metric.source !== "not_available" && metric.value !== null;
  
  // Calculate telemetry gauge percentages based on reasonable bounds
  let pct = 0;
  if (isAvailable && metric.value !== null) {
    if (unit === "mph") {
      pct = (metric.value / 180) * 100;
    } else if (unit === "deg") {
      // Handles both vertical launch angles and horizontal/side deviation angles
      pct = (Math.abs(metric.value) / 45) * 100;
    } else if (unit === "yd") {
      pct = (metric.value / 320) * 100;
    } else if (unit === "x") {
      // Smash factor typically ranges from 1.0 to 1.5
      pct = ((metric.value - 1.0) / 0.5) * 100;
    } else if (unit === "rpm") {
      pct = (Math.abs(metric.value) / 6000) * 100;
    }
    pct = Math.min(100, Math.max(0, pct));
  }

  const sourceClass = isAvailable ? `metric-card--${metric.source}` : "metric-card--empty";

  return (
    <div className={`metric-card ${sourceClass}`}>
      <div className="metric-card__header">
        <div className="metric-card__label">{label}</div>
        <SourceBadge source={metric.source} confidence={metric.confidence} />
      </div>
      <div className="metric-card__value mono">{format(metric.value, unit)}</div>
      
      {isAvailable && pct > 0 && (
        <div className="metric-card__gauge-track">
          <div className="metric-card__gauge-bar" style={{ width: `${pct}%` }} />
        </div>
      )}
      {metric.notes && <div className="metric-card__notes">{metric.notes}</div>}
    </div>
  );
}
