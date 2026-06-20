import type { TrajectoryPayload } from "../lib/types";
import { MetricCard } from "./MetricCard";
import { WarningsList } from "./WarningsList";
import { formatMetric } from "../lib/units";
import { DrivingRangeScene } from "../scenes/DrivingRangeScene";
import "./ShotSimulatorView.css";

interface ShotSimulatorViewProps {
  payload: TrajectoryPayload;
}

export function ShotSimulatorView({ payload }: ShotSimulatorViewProps) {
  const { metrics } = payload;
  return (
    <div className="shot-simulator">
      {payload.is_placeholder && (
        <div className="shot-simulator__placeholder-banner">
          {payload.notes ?? "This shot has not been fully processed yet."}
        </div>
      )}

      <WarningsList warnings={payload.warnings} />

      <div className="shot-simulator__metrics">
        <MetricCard label="Ball speed" metric={metrics.ball_speed_mps} unit="mph" format={formatMetric} />
        <MetricCard label="Launch angle" metric={metrics.launch_angle_deg} unit="deg" format={formatMetric} />
        <MetricCard
          label="Launch direction"
          metric={metrics.horizontal_launch_deg}
          unit="deg"
          format={formatMetric}
        />
        <MetricCard label="Carry" metric={metrics.carry_m} unit="yd" format={formatMetric} />
        <MetricCard label="Total" metric={metrics.total_m} unit="yd" format={formatMetric} />
        <MetricCard label="Apex" metric={metrics.apex_m} unit="yd" format={formatMetric} />
        <MetricCard label="Side deviation" metric={metrics.side_deviation_m} unit="yd" format={formatMetric} />
        <MetricCard label="Backspin" metric={metrics.backspin_rpm} unit="rpm" format={formatMetric} />
        <MetricCard label="Sidespin" metric={metrics.sidespin_rpm} unit="rpm" format={formatMetric} />
        <MetricCard label="Smash factor" metric={metrics.smash_factor} unit="x" format={formatMetric} />
      </div>

      <DrivingRangeScene
        simulated={payload.simulated_trajectory}
        measured={payload.measured_points}
        fitted={payload.fitted_points}
      />
    </div>
  );
}
