import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ComponentRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { AdaptiveDpr, ContactShadows, OrbitControls, SoftShadows, useTexture } from "@react-three/drei";
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

type LayerKey = "simulated" | "measured" | "fitted";
type CameraVector = [number, number, number];

function conformToRange(points: ScenePoint[]): ScenePoint[] {
  return points.map((point) => ({
    ...point,
    y: point.y + rangeGroundHeight(point.x, point.z),
  }));
}

function SkyBackground() {
  const skySource = useTexture("/textures/range-sky-only-v1.webp");
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

  const resetCamera = useCallback(() => {
    camera.position.set(...position);
    camera.up.set(0, 1, 0);
    camera.lookAt(...target);
    camera.updateProjectionMatrix();

    const controls = controlsRef.current;
    if (controls) {
      controls.target.set(...target);
      controls.update();
    }
  }, [camera, position, target]);

  useEffect(() => {
    if (enabled) resetCamera();
  }, [enabled, resetCamera]);

  useFrame(() => {
    if (!enabled) return;

    const controls = controlsRef.current;
    if (!controls) return;
    const controlledCamera = controls.object;

    if (
      !Number.isFinite(controlledCamera.position.x) ||
      !Number.isFinite(controlledCamera.position.y) ||
      !Number.isFinite(controlledCamera.position.z)
    ) {
      controlledCamera.position.set(...position);
      controlledCamera.up.set(0, 1, 0);
      controls.target.set(...target);
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
    simulated: true,
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
  const { position: cameraPosition, target: cameraTarget } = useMemo(
    () => fitCameraToBounds(bounds),
    [bounds]
  );

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
            label="Predicted flight + rollout"
            active={visibleLayers.simulated}
            onClick={() => toggle("simulated")}
            count={simulated.length}
            colorVar="--color-turf-bright"
            description="Physics prediction from launch through first landing, two estimated bounces, and rollout."
          />
          <LayerToggle
            label="Stereo observations"
            active={visibleLayers.measured}
            onClick={() => toggle("measured")}
            count={measured.length}
            colorVar="--color-amber"
            description="Raw 3D ball positions reconstructed from frames where both cameras observed the ball."
          />
          <LayerToggle
            label="Physics launch fit"
            active={visibleLayers.fitted}
            onClick={() => toggle("fitted")}
            count={fitted.length}
            colorVar="--color-violet"
            description="The smooth physics model evaluated at the observation times before it is extrapolated downrange."
          />
        </div>
        <button className="driving-range__replay" onClick={replayShot}>
          ▶ Replay shot
        </button>
        <button className="driving-range__camera" onClick={() => setCameraMode((mode) => mode === "chase" ? "orbit" : "chase")}>
          {cameraMode === "chase" ? "Ball cam" : "Free cam"}
        </button>
      </div>

      <div className="driving-range__canvas-wrap">
        <Canvas
          key={rendererRevision}
          shadows={{ type: THREE.PCFSoftShadowMap }}
          dpr={[1, 1.7]}
          camera={{ position: cameraPosition, fov: 46, near: 0.08, far: 2200 }}
          gl={{ antialias: true, powerPreference: "high-performance", toneMapping: THREE.ACESFilmicToneMapping }}
          onCreated={({ gl }) => {
            gl.outputColorSpace = THREE.SRGBColorSpace;
            gl.toneMappingExposure = 1.08;
          }}
        >
          <AdaptiveDpr pixelated />
          <SoftShadows size={24} samples={14} focus={0.35} />
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
          {visibleLayers.simulated && (
            <TrajectoryTracer points={scenePoints.simulated} color="#4cc273" lineWidth={3} />
          )}
          {visibleLayers.measured && (
            <TrajectoryTracer points={scenePoints.measured} color="#e8a23a" dashed lineWidth={2} />
          )}
          {visibleLayers.fitted && (
            <TrajectoryTracer points={scenePoints.fitted} color="#8b7ce0" dashed lineWidth={2} />
          )}
          {visibleLayers.simulated && simulated.length > 0 && (
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
      className={`layer-toggle${active && !disabled ? " layer-toggle--active" : ""}`}
      onClick={onClick}
      disabled={disabled}
      style={{ "--toggle-color": `var(${colorVar})` } as React.CSSProperties}
      title={disabled ? `No data for this layer. ${description || ""}`.trim() : description}
    >
      <span className="layer-toggle__dot" />
      {label}
      {disabled && <span className="layer-toggle__empty">(none)</span>}
    </button>
  );
}
