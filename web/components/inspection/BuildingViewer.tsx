"use client";

import { Html, Line, OrbitControls, useGLTF } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { type ComponentRef, type RefObject, Suspense, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { LOW_CONF_HATCH } from "@/components/ui";
import { inspectionAssetUrl } from "@/lib/api";
import { filterStyle, LEVEL_STYLES, UNANALYZED_STYLE } from "@/lib/palette";
import {
  type CleaningLevel,
  type Facade,
  type InspectionResult,
  type LevelFilter,
  LEVELS,
  levelOf,
  type Window,
} from "@/lib/types";

type Controls = ComponentRef<typeof OrbitControls>;
type View = { position: THREE.Vector3; target: THREE.Vector3 };

const FOV = 45;

/** 외벽 사각형 네 귀퉁이 (origin, +u, +u+v, +v). */
function facadeCorners(f: Facade): THREE.Vector3[] {
  const o = new THREE.Vector3(...f.origin);
  const u = new THREE.Vector3(...f.u_axis).multiplyScalar(f.width);
  const v = new THREE.Vector3(...f.v_axis).multiplyScalar(f.height);
  return [o, o.clone().add(u), o.clone().add(u).add(v), o.clone().add(v)];
}

function quadGeometry(pts: THREE.Vector3[], uv = false): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry().setFromPoints(pts);
  g.setIndex([0, 1, 2, 0, 2, 3]);
  if (uv) g.setAttribute("uv", new THREE.Float32BufferAttribute([0, 0, 1, 0, 1, 1, 0, 1], 2));
  g.computeVertexNormals();
  return g;
}

/** 단일 이미지(신뢰도 낮음) 패널 빗금 — 대각선 줄무늬 alphaMap (모든 패널이 공유) */
let hatchTexture: THREE.CanvasTexture | null = null;
function getHatch(): THREE.CanvasTexture {
  if (hatchTexture) return hatchTexture;
  const c = document.createElement("canvas");
  c.width = c.height = 32;
  const g = c.getContext("2d")!;
  g.fillStyle = "#555"; // 줄 사이도 옅게 보이게
  g.fillRect(0, 0, 32, 32);
  g.strokeStyle = "#fff";
  g.lineWidth = 6;
  for (let k = -32; k <= 64; k += 16) {
    g.beginPath();
    g.moveTo(k, 0);
    g.lineTo(k + 32, 32);
    g.stroke();
  }
  hatchTexture = new THREE.CanvasTexture(c);
  hatchTexture.wrapS = hatchTexture.wrapT = THREE.RepeatWrapping;
  hatchTexture.repeat.set(3, 3);
  return hatchTexture;
}

type Panel = { window: Window; corners: THREE.Vector3[]; center: THREE.Vector3; normal: THREE.Vector3; size: number };

function buildPanels(result: InspectionResult, lift: number): Panel[] {
  const normals = new Map(result.building.facades.map((f) => [f.id, new THREE.Vector3(...f.normal).normalize()]));
  return result.windows.map((w) => {
    const normal = normals.get(w.facade_id) ?? new THREE.Vector3(0, 0, 1);
    // 외벽 법선 방향으로 살짝 띄워 z-fighting 방지
    const corners = w.corners_3d.map((c) => new THREE.Vector3(...c).addScaledVector(normal, lift));
    const center = corners.reduce((a, c) => a.add(c), new THREE.Vector3()).divideScalar(4);
    const size = Math.max(corners[0].distanceTo(corners[1]), corners[1].distanceTo(corners[2]));
    return { window: w, corners, center, normal, size };
  });
}

/** 창문 정면 시점: 창 크기의 ~3배가 화면에 들어오는 거리. */
function frontView(p: Panel): View {
  const dist = Math.max((p.size * 1.6) / Math.tan(THREE.MathUtils.degToRad(FOV / 2)), 3);
  return { position: p.center.clone().addScaledVector(p.normal, dist), target: p.center.clone() };
}

function BuildingMesh({ url }: { url: string }) {
  const { scene } = useGLTF(url);
  const obj = useMemo(() => scene.clone(), [scene]);
  return <primitive object={obj} />;
}

/** 외벽 정사영 텍스처 (glb는 박스만 있다). 텍스처가 없으면 그리지 않는다. */
function FacadeTexture({ facade, url, lift }: { facade: Facade; url: string; lift: number }) {
  const [tex, setTex] = useState<THREE.Texture | null>(null);
  useEffect(() => {
    let t: THREE.Texture | null = null;
    new THREE.TextureLoader().load(url, (loaded) => {
      loaded.colorSpace = THREE.SRGBColorSpace;
      t = loaded;
      setTex(loaded);
    });
    return () => t?.dispose();
  }, [url]);
  const geom = useMemo(() => {
    const n = new THREE.Vector3(...facade.normal).normalize();
    return quadGeometry(facadeCorners(facade).map((c) => c.addScaledVector(n, lift)), true);
  }, [facade, lift]);
  if (!tex) return null;
  return (
    <mesh geometry={geom}>
      <meshBasicMaterial map={tex} side={THREE.DoubleSide} toneMapped={false} />
    </mesh>
  );
}

function WindowPanel({
  panel,
  active,
  onHover,
  onSelect,
}: {
  panel: Panel;
  active: boolean;
  onHover: (id: string | null) => void;
  onSelect: (id: string) => void;
}) {
  const geom = useMemo(() => quadGeometry(panel.corners, true), [panel]);
  const low = !!panel.window.low_confidence && panel.window.cleaning_level !== null;
  const color = filterStyle(levelOf(panel.window)).color;
  const urgent = panel.window.cleaning_level === 5; // 우선 청소는 평소에도 진하고 굵게
  const id = panel.window.id;
  return (
    <group>
      <mesh
        geometry={geom}
        onPointerOver={(e) => {
          e.stopPropagation();
          onHover(id);
        }}
        onPointerOut={() => onHover(null)}
        onClick={(e) => {
          e.stopPropagation();
          onSelect(id);
        }}
      >
        <meshBasicMaterial
          color={color}
          transparent
          opacity={active || urgent ? 0.9 : 0.55}
          alphaMap={low ? getHatch() : null}
          side={THREE.DoubleSide}
          depthWrite={false}
          toneMapped={false}
        />
      </mesh>
      <Line
        points={[...panel.corners, panel.corners[0]]}
        color={active ? "#ffffff" : color}
        lineWidth={active ? 3 : urgent ? 2.5 : 1}
        dashed={low}
        dashSize={panel.size * 0.06}
        gapSize={panel.size * 0.04}
      />
    </group>
  );
}

/** 선택 시 카메라·OrbitControls 타깃을 목표 시점으로 부드럽게 옮긴다. 사용자가 드래그하면 중단. */
function CameraRig({ goal, controls }: { goal: View | null; controls: RefObject<Controls | null> }) {
  const camera = useThree((s) => s.camera);
  const anim = useRef<View | null>(null);
  useEffect(() => {
    anim.current = goal;
  }, [goal]);
  useEffect(() => {
    const c = controls.current;
    if (!c) return;
    const stop = () => (anim.current = null);
    c.addEventListener("start", stop);
    return () => c.removeEventListener("start", stop);
  }, [controls]);
  useFrame((_, dt) => {
    const g = anim.current;
    const c = controls.current;
    if (!g || !c) return;
    const k = 1 - Math.exp(-dt * 5);
    camera.position.lerp(g.position, k);
    c.target.lerp(g.target, k);
    c.update();
    if (camera.position.distanceTo(g.position) < 0.01 && c.target.distanceTo(g.target) < 0.01) anim.current = null;
  });
  return null;
}

export default function BuildingViewer({
  inspectionId,
  result,
  visibleLevels,
  selectedId,
  hoveredId,
  onHover,
  onSelect,
}: {
  inspectionId: string;
  result: InspectionResult;
  visibleLevels: Set<LevelFilter>;
  selectedId: string | null;
  hoveredId: string | null;
  onHover: (id: string | null) => void;
  onSelect: (id: string | null) => void;
}) {
  const controls = useRef<Controls>(null);
  const { home, size } = useMemo(() => {
    const box = new THREE.Box3().setFromPoints(result.building.facades.flatMap(facadeCorners));
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3()).length();
    // 첫 외벽(F1)의 앞 오른쪽 위에서 — 목업(F1 법선 +z, u +x)이면 (0.9, 0.45, 1.3) 방향
    const f1 = result.building.facades[0];
    const dir = new THREE.Vector3(...f1.normal)
      .multiplyScalar(1.3)
      .addScaledVector(new THREE.Vector3(...f1.u_axis), 0.9)
      .addScaledVector(new THREE.Vector3(...f1.v_axis), 0.45);
    const position = center.clone().add(dir.normalize().multiplyScalar(size * 1.4));
    return { home: { position, target: center }, size };
  }, [result]);
  const panels = useMemo(() => buildPanels(result, size * 0.002), [result, size]);
  const [homeTick, setHomeTick] = useState(0);
  const goal = useMemo(() => {
    const p = panels.find((x) => x.window.id === selectedId);
    return p ? frontView(p) : homeTick ? { ...home } : null;
  }, [panels, selectedId, home, homeTick]);
  const hovered = panels.find((p) => p.window.id === hoveredId);

  useEffect(() => {
    document.body.style.cursor = hoveredId ? "pointer" : "";
    return () => {
      document.body.style.cursor = "";
    };
  }, [hoveredId]);

  return (
    <div className="relative h-full w-full">
      <Canvas
        camera={{ fov: FOV, near: size * 0.001, far: size * 20, position: home.position.toArray() }}
        dpr={[1, 2]}
      >
        <color attach="background" args={["#0a0a0a"]} />
        <ambientLight intensity={1.2} />
        <directionalLight position={[size, size * 1.5, size]} intensity={1.5} />
        <Suspense fallback={null}>
          <BuildingMesh url={inspectionAssetUrl(inspectionId, result.building.mesh)} />
        </Suspense>
        {result.building.facades.map((f) => (
          <FacadeTexture key={f.id} facade={f} url={inspectionAssetUrl(inspectionId, f.texture)} lift={size * 0.0005} />
        ))}
        {panels
          .filter((p) => visibleLevels.has(levelOf(p.window)))
          .map((p) => (
            <WindowPanel
              key={p.window.id}
              panel={p}
              active={p.window.id === hoveredId || p.window.id === selectedId}
              onHover={onHover}
              onSelect={onSelect}
            />
          ))}
        {hovered && visibleLevels.has(levelOf(hovered.window)) && (
          <Html position={hovered.center} center style={{ pointerEvents: "none" }} zIndexRange={[20, 0]}>
            <div className="-translate-y-8 whitespace-nowrap rounded-md border border-neutral-700 bg-neutral-900/95 px-2 py-1 text-xs text-neutral-100 shadow-lg">
              <span className="font-mono">{hovered.window.id}</span>
              <span className="ml-2 font-bold" style={{ color: filterStyle(levelOf(hovered.window)).color }}>
                {hovered.window.cleaning_level ?? ""} {hovered.window.cleaning_label}
              </span>
              {hovered.window.low_confidence && hovered.window.cleaning_level !== null && (
                <span className="ml-2 text-amber-300">신뢰도 낮음 (단일 이미지)</span>
              )}
            </div>
          </Html>
        )}
        <OrbitControls ref={controls} makeDefault target={home.target} enableDamping />
        <CameraRig goal={goal} controls={controls} />
        <gridHelper args={[size * 2, 20, "#333", "#1f1f1f"]} position={[home.target.x, 0, home.target.z]} />
      </Canvas>
      <button
        type="button"
        onClick={() => {
          onSelect(null);
          setHomeTick((t) => t + 1);
        }}
        className="absolute right-3 top-3 rounded-md border border-neutral-700 bg-neutral-900/80 px-2.5 py-1 text-xs text-neutral-300 hover:bg-neutral-800"
      >
        전체 보기
      </button>
      <Legend
        unanalyzed={result.windows.some((w) => w.cleaning_level === null)}
        lowConfidence={result.windows.some((w) => w.low_confidence && w.cleaning_level !== null)}
      />
    </div>
  );
}

/** 패널 색 범례: level · 라벨 · CSI 구간 (높은 level이 위). 미분석 창문이 있으면 맨 아래 회색. */
function Legend({ unanalyzed, lowConfidence }: { unanalyzed: boolean; lowConfidence: boolean }) {
  return (
    <ul className="pointer-events-none absolute bottom-3 left-3 hidden space-y-1 rounded-md border border-neutral-800 bg-neutral-900/85 px-2.5 py-2 text-xs sm:block">
      {LEVELS.toReversed().map((lv) => {
        const { label, min, color } = LEVEL_STYLES[lv];
        const max = LEVEL_STYLES[(lv + 1) as CleaningLevel]?.min;
        return (
          <li key={lv} className={`flex items-center gap-2 ${lv === 5 ? "font-bold text-red-300" : "text-neutral-300"}`}>
            <span className="h-3 w-3 rounded-sm" style={{ backgroundColor: color }} />
            <span className="w-20">
              {lv} {label}
            </span>
            <span className="tabular-nums text-neutral-500">{max === undefined ? `≥ ${min}` : `${min}–${max}`}</span>
          </li>
        );
      })}
      {lowConfidence && (
        <li className="flex items-center gap-2 text-amber-300">
          <span className="h-3 w-3 rounded-sm border border-dashed border-neutral-300" style={{ backgroundImage: LOW_CONF_HATCH }} />
          <span>빗금 = 신뢰도 낮음 (단일 이미지)</span>
        </li>
      )}
      {unanalyzed && (
        <li className="flex items-center gap-2 text-neutral-400">
          <span className="h-3 w-3 rounded-sm" style={{ backgroundColor: UNANALYZED_STYLE.color }} />
          <span className="w-20">{UNANALYZED_STYLE.label}</span>
          <span className="text-neutral-500">오염 분석 전</span>
        </li>
      )}
    </ul>
  );
}
