import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";
import type { ScenePoint } from "../lib/types";
import { samplePointAtTime } from "./sceneMath";

interface AnimatedBallProps {
  points: ScenePoint[];
  playToken: number; // bump this number to (re)start playback from t=0
  chaseCamera?: boolean;
}

export function AnimatedBall({ points, playToken, chaseCamera = false }: AnimatedBallProps) {
  const ballRef = useRef<THREE.Group>(null);
  const startTimeRef = useRef<number | null>(null);
  const lastPlayTokenRef = useRef(playToken);

  const firstPointTime = points.length > 0 ? points[0].t : 0;
  const duration = points.length > 0 ? points[points.length - 1].t - firstPointTime : 0;

  useFrame((state, delta) => {
    if (!ballRef.current || points.length === 0) return;

    if (lastPlayTokenRef.current !== playToken) {
      lastPlayTokenRef.current = playToken;
      startTimeRef.current = state.clock.elapsedTime;
    }

    const restPoint = points[points.length - 1];
    if (startTimeRef.current === null) {
      ballRef.current.position.set(restPoint.x, restPoint.y, restPoint.z);
      return;
    }

    const elapsed = state.clock.elapsedTime - startTimeRef.current;
    const sample = elapsed >= duration ? restPoint : samplePointAtTime(points, firstPointTime + elapsed);
    if (sample) {
      ballRef.current.position.set(sample.x, sample.y, sample.z);
      if (chaseCamera && elapsed < duration) {
        const ahead = samplePointAtTime(points, firstPointTime + Math.min(elapsed + 0.18, duration)) || sample;
        const direction = new THREE.Vector3(
          ahead.x - sample.x,
          Math.max(-0.15, ahead.y - sample.y),
          ahead.z - sample.z,
        );
        if (direction.lengthSq() < 1e-6) direction.set(1, 0, 0);
        direction.normalize();
        const ball = new THREE.Vector3(sample.x, sample.y, sample.z);
        const desired = ball.clone().addScaledVector(direction, -10).add(new THREE.Vector3(0, 3.8, 0));
        state.camera.position.lerp(desired, 1 - Math.exp(-delta * 4.5));
        state.camera.lookAt(
          ball.clone().addScaledVector(direction, 7).add(new THREE.Vector3(0, 0.8, 0)),
        );
      }
    }
  });

  return (
    <group ref={ballRef}>
      <mesh castShadow>
        <sphereGeometry args={[0.02135, 20, 16]} />
        <meshStandardMaterial
          color="#f7f7f2"
          emissive="#ffffff"
          emissiveIntensity={0.12}
          roughness={0.42}
          metalness={0}
        />
      </mesh>
      <Html center zIndexRange={[2, 0]} style={{ pointerEvents: "none" }}>
        <span className="ball-tracker" aria-hidden="true" />
      </Html>
    </group>
  );
}
