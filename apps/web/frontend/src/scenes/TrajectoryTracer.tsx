import { Line } from "@react-three/drei";
import type { ScenePoint } from "../lib/types";

interface TrajectoryTracerProps {
  points: ScenePoint[];
  color: string;
  dashed?: boolean;
  lineWidth?: number;
  showPoints?: boolean;
}

export function TrajectoryTracer({ points, color, dashed = false, lineWidth = 2.5, showPoints = false }: TrajectoryTracerProps) {
  if (points.length < 2) return null;
  const vertices: [number, number, number][] = points.map((p) => [p.x, p.y, p.z]);
  return (
    <group>
      <Line
        points={vertices}
        color={color}
        lineWidth={lineWidth}
        dashed={dashed}
        dashSize={dashed ? 0.6 : undefined}
        gapSize={dashed ? 0.4 : undefined}
      />
      {showPoints && points.map((point, index) => (
        <mesh key={`${point.t}-${index}`} position={[point.x, point.y, point.z]}>
          <sphereGeometry args={[index === points.length - 1 ? 0.22 : 0.14, 12, 12]} />
          <meshBasicMaterial color={color} toneMapped={false} />
        </mesh>
      ))}
    </group>
  );
}
