import { Html } from "@react-three/drei";
import { metersToYards } from "../lib/units";
import type { SceneBounds } from "./sceneMath";

interface RangeEnvironmentProps {
  bounds: SceneBounds;
}

const YARD_TO_M = 0.9144;

export function RangeEnvironment({ bounds }: RangeEnvironmentProps) {
  const lengthM = Math.max(bounds.maxDownrangeM * 1.25, 30);
  const widthM = Math.max(bounds.maxLateralAbsM * 3, 24);

  // Distance rings every 25 yards out to a bit past the shot length.
  const maxYards = Math.ceil(metersToYards(lengthM) / 25) * 25;
  const ringYards: number[] = [];
  for (let y = 25; y <= maxYards; y += 25) ringYards.push(y);

  return (
    <group>
      {/* Ground */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[lengthM / 2 - 5, 0, 0]} receiveShadow>
        <planeGeometry args={[lengthM + 10, widthM]} />
        <meshStandardMaterial color="#16291e" roughness={1} />
      </mesh>

      {/* Faint distance grid lines along the ground, every 25 yards */}
      {ringYards.map((yd) => {
        const x = yd * YARD_TO_M;
        return (
          <group key={yd}>
            <mesh position={[x, 0.005, 0]} rotation={[-Math.PI / 2, 0, 0]}>
              <planeGeometry args={[0.05, widthM]} />
              <meshBasicMaterial color="#2c4434" transparent opacity={0.6} />
            </mesh>
            <Html position={[x, 0, widthM / 2 + 1.2]} center distanceFactor={40}>
              <div className="range-label">{yd} yd</div>
            </Html>
          </group>
        );
      })}

      {/* Target line down the middle (+X, z=0) */}
      <mesh position={[lengthM / 2 - 5, 0.01, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[lengthM + 10, 0.08]} />
        <meshBasicMaterial color="#4cc273" transparent opacity={0.35} />
      </mesh>

      {/* Tee marker */}
      <mesh position={[0, 0.05, 0]}>
        <cylinderGeometry args={[0.18, 0.18, 0.05, 24]} />
        <meshStandardMaterial color="#e8a23a" />
      </mesh>

      <ambientLight intensity={0.55} />
      <directionalLight position={[20, 35, 10]} intensity={1.1} castShadow />
      <hemisphereLight args={["#3a5c46", "#0b1410", 0.4]} />
    </group>
  );
}
