import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";
import type { ScenePoint } from "../lib/types";
import { samplePointAtTime } from "./sceneMath";

interface AnimatedBallProps {
  points: ScenePoint[];
  playToken: number;
  chaseCamera?: boolean;
}

const UP = new THREE.Vector3(0, 1, 0);
const MAX_LAUNCH_RPM = 3200;
const MIN_FLIGHT_RPM = 520;

export function AnimatedBall({ points, playToken, chaseCamera = false }: AnimatedBallProps) {
  const ballRef = useRef<THREE.Group>(null);
  const ballMeshRef = useRef<THREE.Mesh>(null);
  const trackerRef = useRef<HTMLSpanElement>(null);
  const startTimeRef = useRef<number | null>(null);
  const lastPlayTokenRef = useRef(playToken);
  const lastChaseCameraRef = useRef(chaseCamera);
  const initialSpeedRef = useRef(1);
  const spinAxisRef = useRef(new THREE.Vector3(0, 0, 1));
  const visualSpinRef = useRef(0);
  const incrementalRotation = useRef(new THREE.Quaternion());

  const firstPointTime = points.length > 0 ? points[0].t : 0;
  const duration = points.length > 0 ? points[points.length - 1].t - firstPointTime : 0;

  useFrame((state, delta) => {
    if (!ballRef.current || points.length === 0) return;

    const newLaunch = lastPlayTokenRef.current !== playToken;
    if (newLaunch) {
      lastPlayTokenRef.current = playToken;
      startTimeRef.current = state.clock.elapsedTime;
      ballMeshRef.current?.quaternion.identity();
      visualSpinRef.current = 0;

      const launchAhead = samplePointAtTime(points, Math.min(firstPointTime + 0.08, firstPointTime + duration)) ?? points[1] ?? points[0];
      const launchVelocity = new THREE.Vector3(
        launchAhead.x - points[0].x,
        launchAhead.y - points[0].y,
        launchAhead.z - points[0].z,
      ).divideScalar(Math.max(0.001, launchAhead.t - points[0].t));
      initialSpeedRef.current = Math.max(launchVelocity.length(), 0.01);

      // Backspin is about the horizontal axis perpendicular to launch velocity.
      // Keeping this spin axis stable through flight matches launch-monitor convention.
      spinAxisRef.current.crossVectors(launchVelocity, UP).normalize();
      if (spinAxisRef.current.lengthSq() < 0.5) spinAxisRef.current.set(0, 0, 1);
    }

    const chaseActivated = chaseCamera && (!lastChaseCameraRef.current || newLaunch);
    lastChaseCameraRef.current = chaseCamera;
    const restPoint = points[points.length - 1];
    const elapsed = startTimeRef.current === null ? duration : state.clock.elapsedTime - startTimeRef.current;
    const flightTime = THREE.MathUtils.clamp(elapsed, 0, duration);
    const sample = startTimeRef.current === null || elapsed >= duration
      ? restPoint
      : samplePointAtTime(points, firstPointTime + flightTime);
    if (!sample) return;

    ballRef.current.position.set(sample.x, sample.y, sample.z);

    const sampleWindow = 0.055;
    const behindTime = Math.max(0, flightTime - sampleWindow);
    const aheadTime = Math.min(duration, flightTime + sampleWindow);
    const behind = samplePointAtTime(points, firstPointTime + behindTime) ?? sample;
    const ahead = samplePointAtTime(points, firstPointTime + aheadTime) ?? sample;
    const velocity = new THREE.Vector3(ahead.x - behind.x, ahead.y - behind.y, ahead.z - behind.z)
      .divideScalar(Math.max(0.001, aheadTime - behindTime));
    const speed = velocity.length();

    if (ballMeshRef.current && startTimeRef.current !== null && elapsed < duration) {
      const speedRatio = THREE.MathUtils.clamp(speed / initialSpeedRef.current, 0, 1);
      const rpm = MIN_FLIGHT_RPM + (MAX_LAUNCH_RPM - MIN_FLIGHT_RPM) * Math.pow(speedRatio, 0.78);
      const radiansPerSecond = rpm * Math.PI * 2 / 60;
      incrementalRotation.current.setFromAxisAngle(spinAxisRef.current, radiansPerSecond * Math.min(delta, 0.04));
      ballMeshRef.current.quaternion.premultiply(incrementalRotation.current);

      // The enlarged tracker is a readability aid. Cap its apparent rate below
      // the physical mesh rate to avoid frame-rate aliasing while preserving deceleration.
      const visibleRate = THREE.MathUtils.lerp(18, 74, Math.pow(speedRatio, 0.72));
      visualSpinRef.current -= visibleRate * Math.min(delta, 0.04);
      trackerRef.current?.style.setProperty("--ball-spin", `${visualSpinRef.current}rad`);
    }

    if (chaseCamera) {
      const direction = velocity.lengthSq() > 1e-6
        ? velocity.normalize()
        : new THREE.Vector3(restPoint.x - points[Math.max(0, points.length - 2)].x, 0, restPoint.z - points[Math.max(0, points.length - 2)].z).normalize();
      if (direction.lengthSq() < 1e-6) direction.set(1, 0, 0);
      const ball = new THREE.Vector3(sample.x, sample.y, sample.z);
      const desired = ball.clone().addScaledVector(direction, -12.5).add(new THREE.Vector3(0, 4.8, 0));

      // Snap on activation/replay before the frame is painted. Subsequent frames
      // damp smoothly, eliminating the one-frame backward orbit-camera flash.
      if (chaseActivated) state.camera.position.copy(desired);
      else if (elapsed < duration) state.camera.position.lerp(desired, 1 - Math.exp(-delta * 6.5));
      state.camera.up.set(0, 1, 0);
      state.camera.lookAt(ball);
      state.camera.updateMatrixWorld();
    }
  });

  return (
    <group ref={ballRef}>
      <mesh ref={ballMeshRef}>
        <sphereGeometry args={[0.02135, 40, 32]} />
        <meshStandardMaterial color="#f5f5ef" roughness={0.82} metalness={0} />
      </mesh>
      <Html center zIndexRange={[2, 0]} style={{ pointerEvents: "none" }}>
        <span ref={trackerRef} className="ball-tracker" aria-hidden="true" />
      </Html>
    </group>
  );
}
