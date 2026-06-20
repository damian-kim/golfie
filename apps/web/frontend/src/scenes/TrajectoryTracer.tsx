import { Line } from "@react-three/drei";
import type { ScenePoint } from "../lib/types";

interface TrajectoryTracerProps {
  points: ScenePoint[];
  color: string;
  dashed?: boolean;
  lineWidth?: number;
}

export function TrajectoryTracer({ points, color, dashed = false, lineWidth = 2.5 }: TrajectoryTracerProps) {
  if (points.length < 2) return null;
  const vertices: [number, number, number][] = points.map((p) => [p.x, p.y, p.z]);
  return (
    <Line
      points={vertices}
      color={color}
      lineWidth={lineWidth}
      dashed={dashed}
      dashSize={dashed ? 0.6 : undefined}
      gapSize={dashed ? 0.4 : undefined}
    />
  );
}
