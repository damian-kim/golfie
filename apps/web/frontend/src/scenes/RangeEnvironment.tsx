import { useLayoutEffect, useMemo, useRef } from "react";
import { Html, useTexture } from "@react-three/drei";
import * as THREE from "three";
import { metersToYards } from "../lib/units";
import { rangeGroundHeight, type SceneBounds } from "./sceneMath";

interface RangeEnvironmentProps {
  bounds: SceneBounds;
}

interface TreeSpec {
  position: THREE.Vector3;
  scale: number;
  rotation: number;
  species: number;
}

interface TreeSpecies {
  texture: string;
  width: number;
  height: number;
}

const YARD_TO_M = 0.9144;
const TARGETS = [50, 100, 150, 200, 250, 300];
const TREE_SPECIES: TreeSpecies[] = [
  { texture: "/textures/deciduous-tree-v1.webp", width: 10.4, height: 15.6 },
  { texture: "/textures/pine-tree-v1.webp", width: 9.6, height: 19.5 },
  { texture: "/textures/birch-tree-v1.webp", width: 8.8, height: 17.8 },
];

function fairwayCenter(x: number): number {
  return Math.sin(x * 0.016 - 0.7) * Math.min(7, x * 0.025);
}

function seeded(seed: number): () => number {
  let value = seed >>> 0;
  return () => {
    value += 0x6d2b79f5;
    let result = value;
    result = Math.imul(result ^ (result >>> 15), result | 1);
    result ^= result + Math.imul(result ^ (result >>> 7), result | 61);
    return ((result ^ (result >>> 14)) >>> 0) / 4294967296;
  };
}

function makeGround(length: number, width: number, startX: number): THREE.BufferGeometry {
  const segmentsX = 192;
  const segmentsZ = 128;
  const vertices: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];

  for (let zIndex = 0; zIndex <= segmentsZ; zIndex += 1) {
    const v = zIndex / segmentsZ;
    const z = (v - 0.5) * width;
    for (let xIndex = 0; xIndex <= segmentsX; xIndex += 1) {
      const u = xIndex / segmentsX;
      const x = startX + u * length;
      vertices.push(x, rangeGroundHeight(x, z), z);
      uvs.push(u, v);
    }
  }

  const stride = segmentsX + 1;
  for (let zIndex = 0; zIndex < segmentsZ; zIndex += 1) {
    for (let xIndex = 0; xIndex < segmentsX; xIndex += 1) {
      const a = zIndex * stride + xIndex;
      const b = a + 1;
      const c = a + stride;
      const d = c + 1;
      indices.push(a, c, b, b, c, d);
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
  geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

function makeFairway(length: number, baseWidth: number): THREE.BufferGeometry {
  const rows = 150;
  const vertices: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];
  for (let row = 0; row <= rows; row += 1) {
    const fraction = row / rows;
    const x = -10 + fraction * (length + 10);
    const center = fairwayCenter(Math.max(0, x));
    const width = baseWidth * (0.82 + 0.15 * Math.sin(x * 0.024 + 0.9));
    for (const side of [-1, 1]) {
      const z = center + side * width * 0.5;
      vertices.push(x, rangeGroundHeight(x, z) + 0.035, z);
      uvs.push(fraction * 12, side < 0 ? 0 : 3.6);
    }
    if (row < rows) {
      const start = row * 2;
      indices.push(start, start + 2, start + 1, start + 1, start + 2, start + 3);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
  geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

function TreeBillboards({ trees, species }: { trees: TreeSpec[]; species: TreeSpecies }) {
  const billboards = useRef<THREE.InstancedMesh>(null);
  const treeTextureSource = useTexture(species.texture);
  const treeTexture = useMemo(() => {
    const texture = treeTextureSource.clone();
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = 8;
    texture.needsUpdate = true;
    return texture;
  }, [treeTextureSource]);

  useLayoutEffect(() => {
    if (!billboards.current) return;
    const matrix = new THREE.Matrix4();
    const quaternion = new THREE.Quaternion();
    const scale = new THREE.Vector3();
    trees.forEach((tree, index) => {
      const renderedHeight = species.height * tree.scale * 1.1;
      scale.set(species.width * tree.scale, renderedHeight, 1);
      for (let plane = 0; plane < 2; plane += 1) {
        quaternion.setFromEuler(new THREE.Euler(0, tree.rotation + plane * Math.PI * 0.5, 0));
        matrix.compose(
          tree.position.clone().add(new THREE.Vector3(0, renderedHeight * 0.5, 0)),
          quaternion,
          scale,
        );
        billboards.current!.setMatrixAt(index * 2 + plane, matrix);
      }
    });
    billboards.current.instanceMatrix.needsUpdate = true;
  }, [species.height, species.width, trees]);

  return (
    <instancedMesh ref={billboards} args={[undefined, undefined, trees.length * 2]} castShadow receiveShadow>
      <planeGeometry args={[1, 1]} />
      <meshBasicMaterial
        map={treeTexture}
        alphaTest={0.3}
        depthWrite
        side={THREE.DoubleSide}
        toneMapped={false}
      />
    </instancedMesh>
  );
}

function CourseForest({ length, width }: { length: number; width: number }) {
  const trees = useMemo(() => {
    const random = seeded(481516);
    const specs: TreeSpec[] = [];
    const pushTree = (x: number, z: number, scaleBias = 1, baseYOffset = 0) => {
      const scale = (0.84 + random() * 0.34) * scaleBias;
      const speciesRoll = random();
      const species = speciesRoll < 0.5 ? 0 : speciesRoll < 0.78 ? 1 : 2;
      specs.push({
        position: new THREE.Vector3(x, rangeGroundHeight(x, z) + baseYOffset, z),
        scale,
        rotation: random() * Math.PI * 2,
        species,
      });
    };

    // Four overlapping canopy rows, emergent tall trees, and a sunk understory
    // produce the uneven but visually continuous woodland edge of a real range.
    for (let x = -88; x < length + 36; x += 3.6) {
      for (const side of [-1, 1]) {
        const edge = width * 0.35;
        const heightWave = 1 + Math.sin(x * 0.067 + side * 0.8) * 0.16 + Math.sin(x * 0.19) * 0.08;
        for (let depth = 0; depth < 4; depth += 1) {
          const z = side * (edge + depth * 5.4 + random() * 2.8);
          const rowHeight = [0.9, 1.06, 1.2, 1.1][depth];
          pushTree(
            x + (random() - 0.5) * 3.2,
            z,
            rowHeight * heightWave,
          );
        }
        pushTree(
          x + (random() - 0.5) * 2.8,
          side * (edge - 2 + random() * 2.5),
          0.58 + random() * 0.12,
          -3,
        );
        if (random() < 0.2) {
          pushTree(x + (random() - 0.5) * 2.5, side * (edge + 16 + random() * 7), 1.46 + random() * 0.24);
        }
      }
    }

    for (let z = -width * 0.58; z <= width * 0.58; z += 3.6) {
      const heightWave = 1 + Math.sin(z * 0.09) * 0.17 + Math.sin(z * 0.21 + 1.2) * 0.07;
      for (let depth = 0; depth < 4; depth += 1) {
        const rowHeight = [1.02, 1.18, 1.36, 1.24][depth];
        pushTree(
          length + 7 + depth * 5.4 + random() * 2.5,
          z + (random() - 0.5) * 3.2,
          rowHeight * heightWave,
        );
      }
      pushTree(length + 4 + random() * 2, z, 0.58 + random() * 0.12, -3);
      if (random() < 0.22) pushTree(length + 24 + random() * 5, z, 1.55 + random() * 0.22);
    }

    for (let z = -width * 0.56; z <= width * 0.56; z += 3.6) {
      const heightWave = 1 + Math.sin(z * 0.083 - 0.6) * 0.15;
      for (let depth = 0; depth < 3; depth += 1) {
        pushTree(-66 - depth * 5.5 - random() * 2.5, z + (random() - 0.5) * 3.2, (0.88 + depth * 0.1) * heightWave);
      }
      pushTree(-63, z, 0.58 + random() * 0.1, -3);
    }
    return specs;
  }, [length, width]);
  const groupedTrees = useMemo(
    () => TREE_SPECIES.map((_, species) => trees.filter((tree) => tree.species === species)),
    [trees],
  );

  return (
    <group>
      {TREE_SPECIES.map((species, index) => (
        <TreeBillboards key={species.texture} trees={groupedTrees[index]} species={species} />
      ))}
    </group>
  );
}

function TargetFlag({ yards, x, z }: { yards: number; x: number; z: number }) {
  const y = rangeGroundHeight(x, z);
  const color = yards % 100 === 0 ? "#f4f0dc" : "#c83e35";
  return (
    <group position={[x, y + 0.06, z]}>
      <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <circleGeometry args={[5.8, 64]} />
        <meshStandardMaterial color="#4c8b43" roughness={0.92} />
      </mesh>
      <mesh position={[0, 1.65, 0]} castShadow>
        <cylinderGeometry args={[0.025, 0.035, 3.3, 10]} />
        <meshStandardMaterial color="#e6e2d5" roughness={0.35} metalness={0.25} />
      </mesh>
      <mesh position={[0.42, 2.9, 0]} castShadow>
        <planeGeometry args={[0.84, 0.45, 4, 2]} />
        <meshStandardMaterial color={color} side={THREE.DoubleSide} roughness={0.65} />
      </mesh>
      <Html position={[0, 3.75, 0]} center distanceFactor={30}>
        <div className="target-badge" style={{ borderLeft: `4px solid ${color}` }}>
          {yards} <span className="target-badge__unit">yd</span>
        </div>
      </Html>
    </group>
  );
}

export function RangeEnvironment({ bounds }: RangeEnvironmentProps) {
  const length = Math.max(330, bounds.maxDownrangeM * 1.32 + 35);
  const width = Math.max(145, bounds.maxLateralAbsM * 6 + 95);
  const terrainStart = -700;
  const terrainLength = length + 1400;
  const terrainWidth = Math.max(1400, width + 1200);
  const fairwayWidth = Math.max(24, Math.min(43, width * 0.26));
  const grassSource = useTexture("/textures/fairway-grass-v1.webp");
  const materials = useMemo(() => {
    const prepare = (repeatX: number, repeatY: number) => {
      const texture = grassSource.clone();
      texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
      texture.repeat.set(repeatX, repeatY);
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.anisotropy = 8;
      texture.needsUpdate = true;
      return texture;
    };
    return { rough: prepare(96, 82), fairway: prepare(12, 3.6), tee: prepare(2.4, 1.5) };
  }, [grassSource]);
  const ground = useMemo(
    () => makeGround(terrainLength, terrainWidth, terrainStart),
    [terrainLength, terrainStart, terrainWidth],
  );
  const fairway = useMemo(() => makeFairway(length, fairwayWidth), [length, fairwayWidth]);

  const maxYards = Math.floor(metersToYards(length - 10) / 50) * 50;
  const distanceYards = Array.from({ length: Math.floor(maxYards / 50) }, (_, index) => (index + 1) * 50);

  return (
    <group>
      {/* Permanent underlay: even if a GPU/texture resource is lost during a
          development refresh, the sky can never become the range floor. */}
      <mesh
        name="range-ground-failsafe"
        position={[0, -1.75, 0]}
        rotation={[-Math.PI / 2, 0, 0]}
        receiveShadow
        frustumCulled={false}
      >
        <planeGeometry args={[5000, 5000, 1, 1]} />
        <meshBasicMaterial color="#355f2d" side={THREE.DoubleSide} depthWrite />
      </mesh>
      <mesh
        name="range-detailed-ground"
        geometry={ground}
        receiveShadow
        frustumCulled={false}
        dispose={null}
      >
        <meshStandardMaterial
          map={materials.rough}
          bumpMap={materials.rough}
          bumpScale={0.075}
          color="#aec29b"
          roughness={0.98}
          side={THREE.DoubleSide}
          depthWrite
        />
      </mesh>
      <mesh name="range-fairway" geometry={fairway} receiveShadow dispose={null}>
        <meshStandardMaterial map={materials.fairway} bumpMap={materials.fairway} bumpScale={0.045} color="#d3e2c8" roughness={0.91} />
      </mesh>

      <mesh position={[-1.5, 0.045, 0]} receiveShadow>
        <boxGeometry args={[9.5, 0.09, 9]} />
        <meshStandardMaterial map={materials.tee} bumpMap={materials.tee} bumpScale={0.035} color="#dce8d3" roughness={0.9} />
      </mesh>
      <mesh position={[-1, 0.14, -2.7]} castShadow><sphereGeometry args={[0.12, 20, 14]} /><meshStandardMaterial color="#f3f0e6" /></mesh>
      <mesh position={[-1, 0.14, 2.7]} castShadow><sphereGeometry args={[0.12, 20, 14]} /><meshStandardMaterial color="#f3f0e6" /></mesh>

      {TARGETS.map((yards, index) => {
        const x = yards * YARD_TO_M;
        if (x > length - 10) return null;
        return <TargetFlag key={yards} yards={yards} x={x} z={fairwayCenter(x) + (index % 2 ? 4 : -3)} />;
      })}

      {distanceYards.map((yards) => {
        const x = yards * YARD_TO_M;
        return (
          <Html
            key={yards}
            position={[x, rangeGroundHeight(x, -fairwayWidth * 0.62) + 0.18, -fairwayWidth * 0.62]}
            center
            distanceFactor={38}
          >
            <div className="range-label">{yards} yd</div>
          </Html>
        );
      })}

      <CourseForest length={length} width={width} />
    </group>
  );
}
