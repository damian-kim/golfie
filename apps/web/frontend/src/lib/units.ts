// Mirrors packages/golfie_core/metrics/units.py. Internal/API values are
// always meters/seconds/m-per-second; the UI converts to yards/mph at
// the display edge only (spec section 6).

const METERS_PER_YARD = 0.9144;
const MPS_PER_MPH = 0.44704;

export const mpsToMph = (mps: number): number => mps / MPS_PER_MPH;
export const metersToYards = (m: number): number => m / METERS_PER_YARD;

export function formatMetric(value: number | null, unit: "mph" | "yd" | "deg" | "rpm" | "x"): string {
  if (value === null) return "—";
  switch (unit) {
    case "mph":
      return `${mpsToMph(value).toFixed(1)} mph`;
    case "yd":
      return `${metersToYards(value).toFixed(1)} yd`;
    case "deg":
      return `${value.toFixed(1)}°`;
    case "rpm":
      return `${value.toFixed(0)} rpm`;
    case "x":
      return value.toFixed(2);
    default:
      return `${value}`;
  }
}
