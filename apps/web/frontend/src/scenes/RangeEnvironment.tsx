import { Html } from "@react-three/drei";
import { metersToYards } from "../lib/units";
import type { SceneBounds } from "./sceneMath";

interface RangeEnvironmentProps {
  bounds: SceneBounds;
}

const YARD_TO_M = 0.9144;

// Staggered target setups
const TARGETS = [
  { yards: 50, z: -2.5, color: "#ef4444" },   // Red flag
  { yards: 100, z: 3.5, color: "#3b82f6" },   // Blue flag
  { yards: 150, z: -4.0, color: "#eab308" },  // Yellow flag
  { yards: 200, z: 0.0, color: "#10b981" },   // Green flag
  { yards: 250, z: 5.0, color: "#a855f7" },   // Purple flag
  { yards: 300, z: -3.0, color: "#ec4899" },  // Pink flag
];

function LowPolyTree({ position }: { position: [number, number, number] }) {
  // Use coordinate-based determinism for slight size/height variations
  const seed = position[0] + position[2];
  const heightScale = 0.85 + (Math.sin(seed * 100) * 0.15 + 0.15);
  const widthScale = 0.9 + (Math.cos(seed * 50) * 0.1 + 0.1);

  return (
    <group position={position} scale={[widthScale, heightScale, widthScale]}>
      {/* Trunk */}
      <mesh position={[0, 1.0, 0]} castShadow>
        <cylinderGeometry args={[0.15, 0.25, 2.0, 8]} />
        <meshStandardMaterial color="#5c4033" roughness={0.95} />
      </mesh>
      {/* Bottom foliage layer */}
      <mesh position={[0, 2.7, 0]} castShadow receiveShadow>
        <coneGeometry args={[1.1, 2.4, 8]} />
        <meshStandardMaterial color="#2d5e2e" roughness={0.8} />
      </mesh>
      {/* Top foliage layer */}
      <mesh position={[0, 3.8, 0]} castShadow>
        <coneGeometry args={[0.8, 1.8, 8]} />
        <meshStandardMaterial color="#367338" roughness={0.8} />
      </mesh>
    </group>
  );
}

interface TargetGreenProps {
  yards: number;
  x: number;
  z: number;
  color: string;
}

function TargetGreen({ yards, x, z, color }: TargetGreenProps) {
  return (
    <group position={[x, 0.005, z]}>
      {/* Circular green turf */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <circleGeometry args={[5.0, 32]} />
        <meshStandardMaterial color="#295b2c" roughness={0.7} />
      </mesh>

      {/* Target white ring */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.002, 0]}>
        <ringGeometry args={[4.7, 5.0, 32]} />
        <meshBasicMaterial color="#ffffff" transparent opacity={0.65} />
      </mesh>

      {/* Bullseye inner ring */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.002, 0]}>
        <ringGeometry args={[1.8, 2.1, 32]} />
        <meshBasicMaterial color="#ffffff" transparent opacity={0.45} />
      </mesh>

      {/* Flagpole */}
      <mesh position={[0, 1.6, 0]} castShadow>
        <cylinderGeometry args={[0.04, 0.04, 3.2, 8]} />
        <meshStandardMaterial color="#d1d5db" metalness={0.7} roughness={0.2} />
      </mesh>

      {/* Flag banner */}
      <mesh position={[0.38, 2.8, 0]} castShadow>
        <boxGeometry args={[0.76, 0.45, 0.015]} />
        <meshStandardMaterial color={color} roughness={0.4} />
      </mesh>

      {/* Floating yardage badge */}
      <Html position={[0, 3.8, 0]} center distanceFactor={22}>
        <div className="target-badge" style={{ borderLeft: `5px solid ${color}` }}>
          {yards} <span className="target-badge__unit">yd</span>
        </div>
      </Html>
    </group>
  );
}

export function RangeEnvironment({ bounds }: RangeEnvironmentProps) {
  const lengthM = Math.max(bounds.maxDownrangeM * 1.25, 30);
  const widthM = Math.max(bounds.maxLateralAbsM * 3, 24);

  // Fairway width is 70% of the total scene bounds width
  const fairwayWidth = widthM * 0.7;
  const roughWidth = (widthM - fairwayWidth) / 2;

  // Render distance labels every 25 yards up to maximum downrange distance
  const maxYards = Math.ceil(metersToYards(lengthM) / 25) * 25;
  const ringYards: number[] = [];
  for (let y = 25; y <= maxYards; y += 25) ringYards.push(y);

  // Distribute trees along the rough boundaries, spaced out every 22 meters
  const treeSpacing = 22;
  const numTrees = Math.ceil(lengthM / treeSpacing);
  const treePositions: [number, number, number][] = [];

  for (let i = 0; i < numTrees; i++) {
    const x = i * treeSpacing + 10;
    if (x > lengthM) break;
    // Left boundary
    treePositions.push([x, 0, -widthM / 2 + roughWidth / 2]);
    // Right boundary
    treePositions.push([x, 0, widthM / 2 - roughWidth / 2]);
  }

  return (
    <group>
      {/* 1. Fairway Lawn Stripes */}
      {Array.from({ length: 40 }).map((_, i) => {
        const stripWidth = (lengthM + 20) / 40;
        const x = i * stripWidth - 10 + stripWidth / 2;
        const isEven = i % 2 === 0;
        return (
          <mesh key={i} rotation={[-Math.PI / 2, 0, 0]} position={[x, 0.001, 0]} receiveShadow>
            <planeGeometry args={[stripWidth, fairwayWidth]} />
            <meshStandardMaterial
              color={isEven ? "#387a32" : "#3b8035"}
              roughness={0.85}
              metalness={0.05}
            />
          </mesh>
        );
      })}

      {/* 2. Left Rough Boundary */}
      <mesh
        rotation={[-Math.PI / 2, 0, 0]}
        position={[lengthM / 2 - 5, 0, -widthM / 2 + roughWidth / 2]}
        receiveShadow
      >
        <planeGeometry args={[lengthM + 10, roughWidth]} />
        <meshStandardMaterial color="#204d25" roughness={0.95} />
      </mesh>

      {/* 3. Right Rough Boundary */}
      <mesh
        rotation={[-Math.PI / 2, 0, 0]}
        position={[lengthM / 2 - 5, 0, widthM / 2 - roughWidth / 2]}
        receiveShadow
      >
        <planeGeometry args={[lengthM + 10, roughWidth]} />
        <meshStandardMaterial color="#204d25" roughness={0.95} />
      </mesh>

      {/* 4. Trees */}
      {treePositions.map((pos, idx) => (
        <LowPolyTree key={idx} position={pos} />
      ))}

      {/* 5. Target Greens with Flags */}
      {TARGETS.map((target) => {
        const x = target.yards * YARD_TO_M;
        // Only render if target is within our bounds length
        if (x < lengthM) {
          return (
            <TargetGreen
              key={target.yards}
              yards={target.yards}
              x={x}
              z={target.z}
              color={target.color}
            />
          );
        }
        return null;
      })}

      {/* Distance rings every 25 yards along the fairway */}
      {ringYards.map((yd) => {
        const x = yd * YARD_TO_M;
        // Don't render distance labels if target greens overlap
        return (
          <group key={yd}>
            <mesh position={[x, 0.004, 0]} rotation={[-Math.PI / 2, 0, 0]}>
              <planeGeometry args={[0.08, fairwayWidth]} />
              <meshBasicMaterial color="#ffffff" transparent opacity={0.15} />
            </mesh>
            <Html position={[x, 0, fairwayWidth / 2 + 0.8]} center distanceFactor={35}>
              <div className="range-label">{yd} yd</div>
            </Html>
          </group>
        );
      })}

      {/* Target line down the middle (+X, z=0) */}
      <mesh position={[lengthM / 2 - 5, 0.003, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[lengthM + 10, 0.06]} />
        <meshBasicMaterial color="#ffffff" transparent opacity={0.08} />
      </mesh>

      {/* Premium Tee Box */}
      <group position={[-0.5, 0.01, 0]}>
        {/* Wood platform */}
        <mesh position={[0, 0, 0]} receiveShadow castShadow>
          <boxGeometry args={[1.5, 0.06, 2.0]} />
          <meshStandardMaterial color="#402a1b" roughness={0.8} />
        </mesh>
        {/* Ball tray */}
        <mesh position={[0.5, 0.05, 0.8]} castShadow>
          <boxGeometry args={[0.3, 0.06, 0.5]} />
          <meshStandardMaterial color="#1f2937" roughness={0.5} />
        </mesh>
        {/* Synthetic turf strip inside platform */}
        <mesh position={[0, 0.035, 0]} receiveShadow>
          <boxGeometry args={[1.2, 0.01, 1.4]} />
          <meshStandardMaterial color="#1b4d1b" roughness={0.9} />
        </mesh>
      </group>

      {/* Tee Marker Spheres */}
      <mesh position={[0, 0.12, 1.2]} castShadow>
        <sphereGeometry args={[0.1, 16, 16]} />
        <meshStandardMaterial color="#3b82f6" metalness={0.1} roughness={0.3} />
      </mesh>
      <mesh position={[0, 0.12, -1.2]} castShadow>
        <sphereGeometry args={[0.1, 16, 16]} />
        <meshStandardMaterial color="#3b82f6" metalness={0.1} roughness={0.3} />
      </mesh>

      {/* Optimized natural lighting */}
      <ambientLight intensity={0.6} />
      <directionalLight
        position={[50, 75, 25]}
        intensity={1.3}
        castShadow
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
        shadow-camera-far={600}
        shadow-camera-left={-60}
        shadow-camera-right={60}
        shadow-camera-top={60}
        shadow-camera-bottom={-60}
        shadow-bias={-0.0001}
      />
      <hemisphereLight args={["#cde3f2", "#2e5b2c", 0.45]} />
    </group>
  );
}
