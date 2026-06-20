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
  return (
    <div className={`metric-card${isAvailable ? "" : " metric-card--empty"}`}>
      <div className="metric-card__label">{label}</div>
      <div className="metric-card__value mono">{format(metric.value, unit)}</div>
      <SourceBadge source={metric.source} confidence={metric.confidence} />
      {metric.notes && <div className="metric-card__notes">{metric.notes}</div>}
    </div>
  );
}
