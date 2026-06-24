import { useMemo, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Sky, Cloud } from "@react-three/drei";
import type { ScenePoint } from "../lib/types";
import { RangeEnvironment } from "./RangeEnvironment";
import { TrajectoryTracer } from "./TrajectoryTracer";
import { AnimatedBall } from "./AnimatedBall";
import { computeSceneBounds, fitCameraToBounds } from "./sceneMath";
import "./DrivingRangeScene.css";

interface DrivingRangeSceneProps {
  simulated: ScenePoint[];
  measured?: ScenePoint[];
  fitted?: ScenePoint[];
}

type LayerKey = "simulated" | "measured" | "fitted";

export function DrivingRangeScene({ simulated, measured = [], fitted = [] }: DrivingRangeSceneProps) {
  const [visibleLayers, setVisibleLayers] = useState<Record<LayerKey, boolean>>({
    simulated: true,
    measured: true,
    fitted: true,
  });
  const [playToken, setPlayToken] = useState(0);

  const bounds = useMemo(
    () => computeSceneBounds([simulated, measured, fitted]),
    [simulated, measured, fitted]
  );
  const { position: cameraPosition, target: cameraTarget } = useMemo(
    () => fitCameraToBounds(bounds),
    [bounds]
  );

  const toggle = (key: LayerKey) => setVisibleLayers((prev) => ({ ...prev, [key]: !prev[key] }));

  return (
    <div className="driving-range">
      <div className="driving-range__toolbar">
        <div className="driving-range__layers">
          <LayerToggle
            label="Simulated flight"
            active={visibleLayers.simulated}
            onClick={() => toggle("simulated")}
            count={simulated.length}
            colorVar="--color-turf-bright"
          />
          <LayerToggle
            label="Measured points"
            active={visibleLayers.measured}
            onClick={() => toggle("measured")}
            count={measured.length}
            colorVar="--color-amber"
          />
          <LayerToggle
            label="Fitted curve"
            active={visibleLayers.fitted}
            onClick={() => toggle("fitted")}
            count={fitted.length}
            colorVar="--color-violet"
          />
        </div>
        <button className="driving-range__replay" onClick={() => setPlayToken((t) => t + 1)}>
          ▶ Replay shot
        </button>
      </div>

      <div className="driving-range__canvas-wrap">
        <Canvas shadows camera={{ position: cameraPosition, fov: 42, far: 2000 }}>
          <color attach="background" args={["#cde3f2"]} />
          <fog attach="fog" args={["#cde3f2", 150, Math.max(bounds.maxDownrangeM * 2.5, 300)]} />
          <Sky distance={450000} sunPosition={[50, 75, 25]} inclination={0} azimuth={0.25} />
          <Cloud position={[-15, 30, -35]} opacity={0.5} speed={0.15} segments={8} scale={1.2} />
          <Cloud position={[35, 45, 35]} opacity={0.4} speed={0.12} segments={10} scale={1.5} />
          <Cloud position={[90, 55, -25]} opacity={0.3} speed={0.2} segments={12} scale={2.0} />
          <RangeEnvironment bounds={bounds} />
          {visibleLayers.simulated && (
            <TrajectoryTracer points={simulated} color="#4cc273" lineWidth={3} />
          )}
          {visibleLayers.measured && (
            <TrajectoryTracer points={measured} color="#e8a23a" dashed lineWidth={2} />
          )}
          {visibleLayers.fitted && (
            <TrajectoryTracer points={fitted} color="#8b7ce0" dashed lineWidth={2} />
          )}
          {visibleLayers.simulated && simulated.length > 0 && (
            <AnimatedBall points={simulated} playToken={playToken} />
          )}
          <OrbitControls target={cameraTarget} maxPolarAngle={Math.PI / 2 - 0.02} />
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
}: {
  label: string;
  active: boolean;
  onClick: () => void;
  count: number;
  colorVar: string;
}) {
  const disabled = count === 0;
  return (
    <button
      className={`layer-toggle${active && !disabled ? " layer-toggle--active" : ""}`}
      onClick={onClick}
      disabled={disabled}
      style={{ "--toggle-color": `var(${colorVar})` } as React.CSSProperties}
      title={disabled ? "No data for this layer" : undefined}
    >
      <span className="layer-toggle__dot" />
      {label}
      {disabled && <span className="layer-toggle__empty">(none)</span>}
    </button>
  );
}
