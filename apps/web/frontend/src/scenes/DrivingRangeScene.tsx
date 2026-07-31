import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ComponentRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { AdaptiveDpr, ContactShadows, Html, OrbitControls, useTexture } from "@react-three/drei";
import * as THREE from "three";
import type { ScenePoint } from "../lib/types";
import { RangeEnvironment } from "./RangeEnvironment";
import { TrajectoryTracer } from "./TrajectoryTracer";
import { AnimatedBall } from "./AnimatedBall";
import { computeSceneBounds, fitCameraToBounds, rangeGroundHeight } from "./sceneMath";
import "./DrivingRangeScene.css";

interface DrivingRangeSceneProps {
  simulated: ScenePoint[];
  measured?: ScenePoint[];
  fitted?: ScenePoint[];
  playToken?: number;
  onReplayClick?: () => void;
}

type LayerKey = "measured" | "fitted";
type CameraVector = [number, number, number];

function conformToRange(points: ScenePoint[]): ScenePoint[] {
  return points.map((point) => ({
    ...point,
    y: point.y + rangeGroundHeight(point.x, point.z),
  }));
}

function SkyBackground() {
  const skySource = useTexture("/textures/range-sky-visible-clouds-v4.png");
  const skyTexture = useMemo(() => {
    const texture = skySource.clone();
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.mapping = THREE.EquirectangularReflectionMapping;
    texture.anisotropy = 8;
    texture.needsUpdate = true;
    return texture;
  }, [skySource]);

  useEffect(() => () => skyTexture.dispose(), [skyTexture]);

  return <primitive attach="background" object={skyTexture} />;
}

/** Keep the physical orbit pivot on the stereo handoff and continuously align
 * that exact world point with the HUD's 50% screen centerline. */
function CenteredHandoffProjection({ target, enabled }: { target: CameraVector; enabled: boolean }) {
  const { camera, size } = useThree();
  const viewOffset = useRef(0);
  const projectedTarget = useRef(new THREE.Vector3());

  useEffect(() => {
    viewOffset.current = 0;
    if (!enabled && camera instanceof THREE.PerspectiveCamera) {
      camera.clearViewOffset();
      camera.updateProjectionMatrix();
    }
    return () => {
      if (!(camera instanceof THREE.PerspectiveCamera)) return;
      camera.clearViewOffset();
      camera.updateProjectionMatrix();
    };
  }, [camera, enabled, size.height, size.width, target]);

  useFrame(() => {
    if (!(camera instanceof THREE.PerspectiveCamera)) return;
    if (!enabled) {
      if (camera.view?.enabled) {
        camera.clearViewOffset();
        camera.updateProjectionMatrix();
      }
      return;
    }
    if (size.width <= 0 || size.height <= 0) return;
    camera.updateMatrixWorld();
    projectedTarget.current.set(...target).project(camera);
    // The rendered handoff marker sits one stage-grid unit to the right of the
    // raw orbit coordinate. Center the visible join against the same 50% line
    // used by the shot selector, launch radar, and transport controls.
    const visibleHandoffBias = size.width / 96;
    const horizontalError = projectedTarget.current.x * size.width * 0.5 + visibleHandoffBias;
    if (Math.abs(horizontalError) < 0.2) return;

    viewOffset.current = THREE.MathUtils.clamp(
      viewOffset.current + horizontalError * 0.65,
      -size.width * 0.25,
      size.width * 0.25,
    );
    camera.setViewOffset(size.width, size.height, viewOffset.current, 0, size.width, size.height);
    camera.updateProjectionMatrix();
  });

  return null;
}

function FreeCameraControls({
  enabled,
  position,
  target,
}: {
  enabled: boolean;
  position: CameraVector;
  target: CameraVector;
}) {
  const { camera } = useThree();
  const controlsRef = useRef<ComponentRef<typeof OrbitControls>>(null);
  const pressedKeys = useRef(new Set<string>());
  const lockedTarget = useRef(new THREE.Vector3(...target));

  const resetCamera = useCallback(() => {
    camera.position.set(...position);
    camera.up.set(0, 1, 0);
    camera.lookAt(...target);
    camera.updateProjectionMatrix();
    lockedTarget.current.set(...target);

    const controls = controlsRef.current;
    if (controls) {
      controls.target.copy(lockedTarget.current);
      controls.update();
    }
  }, [camera, position, target]);

  useEffect(() => {
    if (enabled) resetCamera();
  }, [enabled, resetCamera]);

  useEffect(() => {
    if (!enabled) {
      pressedKeys.current.clear();
      return;
    }

    const isEditableTarget = (target: EventTarget | null) => {
      const element = target as HTMLElement | null;
      return Boolean(element?.isContentEditable || element?.closest("input, textarea, select, button, a"));
    };
    const movementKeys = new Set(["KeyW", "KeyA", "KeyS", "KeyD", "Space", "ControlLeft", "ControlRight"]);
    const handleKeyDown = (event: KeyboardEvent) => {
      const isVerticalControl = event.code === "ControlLeft" || event.code === "ControlRight";
      if (!movementKeys.has(event.code) || isEditableTarget(event.target) || event.metaKey || event.altKey || (event.ctrlKey && !isVerticalControl)) return;
      pressedKeys.current.add(event.code);
      event.preventDefault();
    };
    const handleKeyUp = (event: KeyboardEvent) => pressedKeys.current.delete(event.code);
    const clearKeys = () => pressedKeys.current.clear();

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    window.addEventListener("blur", clearKeys);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
      window.removeEventListener("blur", clearKeys);
      clearKeys();
    };
  }, [enabled]);

  useFrame((_, delta) => {
    if (!enabled) return;

    const controls = controlsRef.current;
    if (!controls) return;
    const controlledCamera = controls.object;
    if (!controls.target.equals(lockedTarget.current)) {
      controls.target.copy(lockedTarget.current);
      controls.update();
    }

    const forwardAmount = Number(pressedKeys.current.has("KeyW")) - Number(pressedKeys.current.has("KeyS"));
    const rightAmount = Number(pressedKeys.current.has("KeyD")) - Number(pressedKeys.current.has("KeyA"));
    const verticalAmount = Number(pressedKeys.current.has("Space")) - Number(pressedKeys.current.has("ControlLeft") || pressedKeys.current.has("ControlRight"));
    if (forwardAmount || rightAmount || verticalAmount) {
      const forward = new THREE.Vector3();
      controlledCamera.getWorldDirection(forward);
      forward.y = 0;
      if (forward.lengthSq() < 0.0001) forward.set(1, 0, 0);
      else forward.normalize();

      const right = new THREE.Vector3().crossVectors(forward, controlledCamera.up).normalize();
      const movement = forward.multiplyScalar(forwardAmount).addScaledVector(right, rightAmount);
      movement.y = verticalAmount;
      if (movement.lengthSq() > 1) movement.normalize();
      movement.multiplyScalar(24 * Math.min(delta, 0.05));
      controlledCamera.position.add(movement);
      lockedTarget.current.add(movement);
      controls.target.copy(lockedTarget.current);
      controls.update();
    }

    if (
      !Number.isFinite(controlledCamera.position.x) ||
      !Number.isFinite(controlledCamera.position.y) ||
      !Number.isFinite(controlledCamera.position.z)
    ) {
      controlledCamera.position.set(...position);
      controlledCamera.up.set(0, 1, 0);
      lockedTarget.current.set(...target);
      controls.target.copy(lockedTarget.current);
      controls.update();
      return;
    }

    const minimumCameraY = rangeGroundHeight(controlledCamera.position.x, controlledCamera.position.z) + 1.1;
    let corrected = false;

    if (controlledCamera.position.y < minimumCameraY) {
      controlledCamera.position.y = minimumCameraY;
      corrected = true;
    }

    const minimumTargetY = rangeGroundHeight(controls.target.x, controls.target.z) + 0.15;
    if (controls.target.y < minimumTargetY) {
      controls.target.y = minimumTargetY;
      lockedTarget.current.y = minimumTargetY;
      corrected = true;
    }
    if (corrected) controls.update();
  });

  return (
    <OrbitControls
      ref={controlsRef}
      enabled={enabled}
      target={target}
      enableDamping
      enablePan={false}
      dampingFactor={0.075}
      rotateSpeed={0.42}
      panSpeed={0.34}
      zoomSpeed={0.55}
      screenSpacePanning={false}
      minPolarAngle={Math.PI * 0.3}
      minDistance={4}
      maxDistance={460}
      maxPolarAngle={Math.PI / 2 - 0.02}
    />
  );
}

export function DrivingRangeScene({ 
  simulated, 
  measured = [], 
  fitted = [], 
  playToken: externalPlayToken, 
  onReplayClick 
}: DrivingRangeSceneProps) {
  const [visibleLayers, setVisibleLayers] = useState<Record<LayerKey, boolean>>({
    measured: true,
    fitted: true,
  });
  const [localPlayToken, setLocalPlayToken] = useState(0);
  const [cameraMode, setCameraMode] = useState<"chase" | "orbit">("orbit");
  const [rendererRevision, setRendererRevision] = useState(0);
  const playToken = externalPlayToken !== undefined ? externalPlayToken : localPlayToken;
  const previousPlayToken = useRef(playToken);
  const hasAutoFollowed = useRef(false);
  const scenePoints = useMemo(
    () => ({
      simulated: conformToRange(simulated),
      measured: conformToRange(measured),
      fitted: conformToRange(fitted),
    }),
    [fitted, measured, simulated],
  );

  // Three.js resources held by a Fast Refresh boundary can outlive the GPU
  // buffers that react-three-fiber disposed during the update. Recreate the
  // entire Canvas after any development edit so stale geometry can never be
  // reused. Production builds do not register this listener.
  useEffect(() => {
    const hot = import.meta.hot;
    if (!hot) return;
    const resetRenderer = () => setRendererRevision((revision) => revision + 1);
    hot.on("vite:afterUpdate", resetRenderer);
    return () => hot.off("vite:afterUpdate", resetRenderer);
  }, []);

  useEffect(() => {
    if (previousPlayToken.current === playToken) return;
    previousPlayToken.current = playToken;
    if (!hasAutoFollowed.current) {
      hasAutoFollowed.current = true;
      setCameraMode("chase");
    }
  }, [playToken]);

  const bounds = useMemo(
    () => computeSceneBounds([scenePoints.simulated, scenePoints.measured, scenePoints.fitted]),
    [scenePoints]
  );
  const { position: cameraPosition, target: fittedCameraTarget } = useMemo(
    () => fitCameraToBounds(bounds),
    [bounds]
  );
  const cameraTarget = useMemo<CameraVector>(() => {
    const stereoHandoff = scenePoints.measured.at(-1);
    if (!stereoHandoff) return fittedCameraTarget;
    return [stereoHandoff.x, stereoHandoff.y, stereoHandoff.z];
  }, [fittedCameraTarget, scenePoints.measured]);

  const toggle = (key: LayerKey) => setVisibleLayers((prev) => ({ ...prev, [key]: !prev[key] }));
  const replayShot = () => {
    if (onReplayClick) onReplayClick();
    else setLocalPlayToken((token) => token + 1);
  };

  return (
    <div className="driving-range">
      <div className="driving-range__toolbar">
        <div className="driving-range__layers">
          <LayerToggle
            label="Stereo observations"
            active={visibleLayers.measured}
            onClick={() => toggle("measured")}
            count={measured.length}
            colorVar="--color-stereo"
            description="Paired two-camera 3D points."
          />
          <LayerToggle
            label="Physics prediction"
            active={visibleLayers.fitted}
            onClick={() => toggle("fitted")}
            count={fitted.length}
            colorVar="--color-physics"
            description="Modeled flight after stereo ends."
          />
        </div>
        <button className="driving-range__replay" onClick={replayShot}>
          ▶ Replay shot
        </button>
        <button
          className="driving-range__camera"
          onClick={() => setCameraMode((mode) => mode === "chase" ? "orbit" : "chase")}
          title={cameraMode === "orbit" ? "WASD move · drag to look · scroll to zoom" : "Switch to free camera"}
          aria-keyshortcuts="W A S D Space Control"
        >
          {cameraMode === "chase" ? "Ball cam" : "Free cam"}
        </button>
      </div>

      <div className="driving-range__canvas-wrap">
        <Canvas
          key={rendererRevision}
          shadows={{ type: THREE.PCFShadowMap }}
          dpr={[1, 1.7]}
          camera={{ position: cameraPosition, fov: 46, near: 0.08, far: 2200 }}
          gl={{ antialias: true, powerPreference: "high-performance", toneMapping: THREE.ACESFilmicToneMapping }}
          onCreated={({ gl }) => {
            gl.outputColorSpace = THREE.SRGBColorSpace;
            gl.toneMappingExposure = 1.08;
          }}
        >
          <CenteredHandoffProjection target={cameraTarget} enabled={cameraMode === "orbit"} />
          <AdaptiveDpr pixelated />
          <fogExp2 attach="fog" args={["#b7d1d9", 0.0019]} />
          <SkyBackground />
          <ambientLight intensity={0.28} color="#dce9ed" />
          <hemisphereLight args={["#d9edff", "#334529", 1.12]} />
          <directionalLight
            position={[75, 95, -52]}
            intensity={2.65}
            color="#fff2d2"
            castShadow
            shadow-mapSize-width={2048}
            shadow-mapSize-height={2048}
            shadow-camera-near={1}
            shadow-camera-far={440}
            shadow-camera-left={-125}
            shadow-camera-right={125}
            shadow-camera-top={135}
            shadow-camera-bottom={-85}
            shadow-bias={-0.00012}
          />
          <RangeEnvironment bounds={bounds} />
          <ContactShadows position={[0, 0.055, 0]} opacity={0.32} scale={55} blur={2.8} far={14} color="#172012" />
          {visibleLayers.measured && (
            <TrajectoryTracer points={scenePoints.measured} color="#d8c49a" lineWidth={4} showPoints />
          )}
          {visibleLayers.fitted && (
            <TrajectoryTracer points={scenePoints.fitted} color="#090909" dashed lineWidth={3.5} />
          )}
          {visibleLayers.measured && visibleLayers.fitted && scenePoints.measured.length > 0 && (
            <TrajectoryHandoff point={scenePoints.measured[scenePoints.measured.length - 1]} />
          )}
          {simulated.length > 0 && (
            <AnimatedBall points={scenePoints.simulated} playToken={playToken} chaseCamera={cameraMode === "chase"} />
          )}
          <FreeCameraControls
            enabled={cameraMode === "orbit"}
            position={cameraPosition}
            target={cameraTarget}
          />
        </Canvas>
      </div>
    </div>
  );
}

function TrajectoryHandoff({ point }: { point: ScenePoint }) {
  return (
    <group position={[point.x, point.y, point.z]}>
      <mesh>
        <torusGeometry args={[0.72, 0.08, 12, 28]} />
        <meshBasicMaterial color="#e9e4d8" toneMapped={false} />
      </mesh>
      <Html center distanceFactor={16} position={[0, 1.55, 0]} className="trajectory-handoff-label">
        <span>LAST TWO-CAMERA FRAME</span>
        <strong>STEREO ENDS · PHYSICS TAKES OVER →</strong>
      </Html>
    </group>
  );
}

function LayerToggle({
  label,
  active,
  onClick,
  count,
  colorVar,
  description,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
  count: number;
  colorVar: string;
  description?: string;
}) {
  const disabled = count === 0;
  return (
    <button
      className={`layer-toggle${active && !disabled ? " layer-toggle--active" : ""}${colorVar === "--color-physics" ? " layer-toggle--dark" : ""}`}
      onClick={onClick}
      disabled={disabled}
      style={{ "--toggle-color": `var(${colorVar})` } as React.CSSProperties}
      title={disabled ? `No data for this layer. ${description || ""}`.trim() : description}
    >
      <span className="layer-toggle__dot" />
      {label}
    </button>
  );
}
