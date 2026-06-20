import type { MetricSource } from "../lib/types";
import "./SourceBadge.css";

const LABELS: Record<MetricSource, string> = {
  measured: "Measured",
  estimated: "Estimated",
  experimental: "Experimental",
  not_available: "Unavailable",
};

interface SourceBadgeProps {
  source: MetricSource;
  confidence?: number;
}

/**
 * Every number Golfie shows is labeled by where it actually came from
 * (spec section 19). This badge is the one visual element repeated
 * everywhere a metric appears -- solid dot = measured from real data,
 * hollow/dashed = estimated or experimental, dim = not available at all.
 */
export function SourceBadge({ source, confidence }: SourceBadgeProps) {
  return (
    <span className={`source-badge source-badge--${source}`}>
      <span className="source-badge__dot" aria-hidden="true" />
      {LABELS[source]}
      {confidence !== undefined && source !== "not_available" && (
        <span className="source-badge__confidence">{Math.round(confidence * 100)}%</span>
      )}
    </span>
  );
}
