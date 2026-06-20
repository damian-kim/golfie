import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import type { ScenePoint } from "../lib/types";
import { samplePointAtTime } from "./sceneMath";

interface AnimatedBallProps {
  points: ScenePoint[];
  playToken: number; // bump this number to (re)start playback from t=0
}

export function AnimatedBall({ points, playToken }: AnimatedBallProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const startTimeRef = useRef<number | null>(null);
  const lastPlayTokenRef = useRef(playToken);

  const duration = points.length > 0 ? points[points.length - 1].t : 0;

  useFrame((state) => {
    if (!meshRef.current || points.length === 0) return;

    if (lastPlayTokenRef.current !== playToken) {
      lastPlayTokenRef.current = playToken;
      startTimeRef.current = state.clock.elapsedTime;
    }

    const restPoint = points[points.length - 1];
    if (startTimeRef.current === null) {
      meshRef.current.position.set(restPoint.x, restPoint.y, restPoint.z);
      return;
    }

    const elapsed = state.clock.elapsedTime - startTimeRef.current;
    const sample = elapsed >= duration ? restPoint : samplePointAtTime(points, elapsed);
    if (sample) {
      meshRef.current.position.set(sample.x, sample.y, sample.z);
    }
  });

  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[0.35, 24, 24]} />
      <meshStandardMaterial color="#f4f4f0" roughness={0.35} metalness={0.05} />
    </mesh>
  );
}
