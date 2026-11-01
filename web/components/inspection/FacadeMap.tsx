"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ToggleChip } from "@/components/ui";
import { inspectionAssetUrl } from "@/lib/api";
import { filterStyle, LEVEL_STYLES } from "@/lib/palette";
import {
  CONFIDENCE_LABELS,
  confidenceOf,
  type Facade,
  type InspectionResult,
  type LevelFilter,
  levelOf,
  rawCoveragePct,
  type Window,
} from "@/lib/types";

/* 2D 외벽 지도. 좌표는 외벽 평면 (u, v) — SfM 스케일을 모르므로 상대값이라 화면에 크기·단위를 쓰지 않는다 (눈금은 층·열 번호만).
 * 텍스처는 외벽 전체 (0..width)×(0..height)를 덮고 픽셀 y = height − v (recon3d/ortho.py rect_to_px) → SVG y = height − v.
 * 배치: "실제" = 텍스처 위 facade_rect, "격자" = 층×열 같은 크기 칸 (긴 띠 외벽에서 창이 너무 작을 때 기본). */

type Layout = "geo" | "grid";
type Rect = { x: number; y: number; w: number; h: number };
type View = { cx: number; cy: number; s: number }; // 화면 중심의 world 좌표, world 1 = s px

const GUTTER_L = 40; // 층 눈금 자리 (px)
const GUTTER_T = 24; // 열 눈금 자리
const PAD = 16;
const PAD_B = 44; // 아래 범례 자리
const MIN_WIN_PX = 16; // 실제 배치에서 창 폭 중앙값이 이보다 작으면 격자로 시작
const GRID_GAP = 0.12; // 격자 칸 사이 (칸 폭 대비)
const TICK_MIN_PX = 20; // 눈금 글자 최소 간격

const median = (xs: number[]) => {
  const s = xs.toSorted((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)] : 0;
};

/** 우선 청소·청소 필요가 많은 외벽 먼저 (기본 탭) */
const urgency = (ws: Window[]) => ws.filter((w) => w.cleaning_level === 5).length * 1000 + ws.filter((w) => w.cleaning_level === 4).length;

export default function FacadeMap({
  inspectionId,
  result,
  visibleLevels,
  multiOnly,
  onToggleMulti,
  selectedId,
  hoveredId,
  onHover,
  onSelect,
}: {
  inspectionId: string;
  result: InspectionResult;
  visibleLevels: Set<LevelFilter>;
  multiOnly: boolean;
  onToggleMulti: () => void;
  selectedId: string | null;
  hoveredId: string | null;
  onHover: (id: string | null) => void;
  onSelect: (id: string) => void;
}) {
  const byFacade = useMemo(() => {
    const m = new Map<string, Window[]>(result.building.facades.map((f) => [f.id, []]));
    for (const w of result.windows) m.get(w.facade_id)?.push(w);
    return m;
  }, [result]);
  const facades = useMemo(
    () => result.building.facades.toSorted((a, b) => urgency(byFacade.get(b.id)!) - urgency(byFacade.get(a.id)!)),
    [result, byFacade],
  );
  const photos = result.meta.recon_status?.mode === "degraded" && result.meta.recon_status.warnings.some((w) => w.startsWith("photos"));
  const [facadeId, setFacadeId] = useState(facades[0].id);
  const facade = facades.find((f) => f.id === facadeId) ?? facades[0];
  const windows = byFacade.get(facade.id)!;
  // 상세에서 다른 외벽 창을 고르면 (작업 목록 등) 그 외벽으로
  const selectedFacade = result.windows.find((w) => w.id === selectedId)?.facade_id;
  if (selectedFacade && selectedFacade !== facadeId && byFacade.has(selectedFacade)) setFacadeId(selectedFacade);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex flex-wrap items-center gap-1.5 border-b border-neutral-800 px-3 py-2" role="tablist" aria-label={photos ? "사진" : "외벽"}>
        <span className="mr-1 text-xs text-neutral-500">{photos ? "사진" : "외벽"}</span>
        {facades.map((f) => {
          const ws = byFacade.get(f.id)!.filter((w) => !multiOnly || confidenceOf(w) === "multi"); // 요약 바와 같은 기준
          const n5 = ws.filter((w) => w.cleaning_level === 5).length;
          const n4 = ws.filter((w) => w.cleaning_level === 4).length;
          const idx = result.building.facades.indexOf(f) + 1;
          return (
            <button
              key={f.id}
              role="tab"
              aria-selected={f.id === facade.id}
              onClick={() => setFacadeId(f.id)}
              className={`inline-flex items-center gap-2 rounded-md border px-2.5 py-1 text-xs ${
                f.id === facade.id ? "border-neutral-500 bg-neutral-800 text-neutral-100" : "border-neutral-800 text-neutral-400 hover:border-neutral-600"
              }`}
            >
              <span className="font-medium">{photos ? `사진 ${idx}` : f.id}</span>
              <span className="tabular-nums text-neutral-500">{ws.length}창</span>
              {n5 > 0 && <Count color={LEVEL_STYLES[5].color} n={n5} />}
              {n4 > 0 && <Count color={LEVEL_STYLES[4].color} n={n4} />}
            </button>
          );
        })}
      </div>
      {/* key: 외벽이 바뀌면 배치·확대 상태를 새로 */}
      <FacadeCanvas
        key={facade.id}
        inspectionId={inspectionId}
        facade={facade}
        windows={windows}
        visibleLevels={visibleLevels}
        multiOnly={multiOnly}
        onToggleMulti={onToggleMulti}
        selectedId={selectedId}
        hoveredId={hoveredId}
        onHover={onHover}
        onSelect={onSelect}
      />
    </div>
  );
}

function Count({ color, n }: { color: string; n: number }) {
  return (
    <span className="inline-flex items-center gap-1 tabular-nums" style={{ color }}>
      <span className="h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
      {n}
    </span>
  );
}

function FacadeCanvas({
  inspectionId,
  facade,
  windows,
  visibleLevels,
  multiOnly,
  onToggleMulti,
  selectedId,
  hoveredId,
  onHover,
  onSelect,
}: {
  inspectionId: string;
  facade: Facade;
  windows: Window[];
  visibleLevels: Set<LevelFilter>;
  multiOnly: boolean;
  onToggleMulti: () => void;
  selectedId: string | null;
  hoveredId: string | null;
  onHover: (id: string | null) => void;
  onSelect: (id: string) => void;
}) {
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);
  const box = useRef<HTMLDivElement | null>(null);
  const observer = useRef<ResizeObserver | null>(null);
  // 콜백 ref: 준비 전 자리표시 div → 실제 지도 div로 바뀌어도 크기 관찰을 옮긴다
  const setBox = useCallback((el: HTMLDivElement | null) => {
    box.current = el;
    observer.current?.disconnect();
    if (!el) return;
    observer.current = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }));
    observer.current.observe(el);
  }, []);
  const [layout, setLayout] = useState<Layout | null>(null); // null = 아직 자동 결정 전
  const [texture, setTexture] = useState(true);
  const [view, setView] = useState<View | null>(null);
  // 지금 view가 "맞춤"이면 그때의 크기 키 — 크기가 바뀌면 (상세 패널 열림 등) 다시 맞춘다. 사용자가 확대·이동하면 null
  const [fitted, setFitted] = useState<string | null>(null);
  const [tip, setTip] = useState<{ x: number; y: number; w: Window } | null>(null);
  const drag = useRef<{ x: number; y: number; moved: number } | null>(null);

  const rows = Math.max(1, ...windows.map((w) => w.row));
  const cols = Math.max(1, ...windows.map((w) => w.col));
  // 격자 칸 높이 (칸 폭 = 1): 화면을 가로·세로 모두 채우도록, 1~4 사이
  const cellH = size ? Math.min(4, Math.max(1, (size.h - GUTTER_T - PAD - PAD_B) / rows / ((size.w - GUTTER_L - 2 * PAD) / cols))) : 1;

  const rectOf = (w: Window, lay: Layout): Rect => {
    if (lay === "grid") return { x: w.col - 1 + GRID_GAP / 2, y: (rows - w.row) * cellH + GRID_GAP / 2, w: 1 - GRID_GAP, h: cellH - GRID_GAP };
    const [u0, v0, u1, v1] = w.facade_rect;
    return { x: u0, y: facade.height - v1, w: u1 - u0, h: v1 - v0 };
  };
  const worldOf = (lay: Layout) => (lay === "grid" ? { w: cols, h: rows * cellH } : { w: facade.width, h: facade.height });
  const fitView = (lay: Layout): View | null => {
    if (!size) return null;
    const world = worldOf(lay);
    const aw = size.w - GUTTER_L - 2 * PAD;
    const ah = size.h - GUTTER_T - PAD - PAD_B;
    const s = Math.max(1e-6, Math.min(aw / world.w, ah / world.h));
    // 내용 중심을 눈금·범례 자리를 뺀 영역의 중심에 둔다
    return { s, cx: world.w / 2 - GUTTER_L / 2 / s, cy: world.h / 2 - (GUTTER_T + PAD - PAD_B) / 2 / s };
  };

  function fit(lay: Layout) {
    setView(fitView(lay));
    setFitted(size ? `${size.w}x${size.h}` : null);
  }

  // 크기를 처음 알게 되면: 배치 자동 결정 + 맞춤
  if (size && layout === null) {
    const geo = fitView("geo")!;
    const medW = median(windows.map((w) => w.facade_rect[2] - w.facade_rect[0])) * geo.s;
    const lay: Layout = medW < MIN_WIN_PX ? "grid" : "geo";
    setLayout(lay);
    fit(lay);
  } else if (size && layout && fitted !== null && fitted !== `${size.w}x${size.h}`) fit(layout);

  const zoomAt = (f: number, px: number, py: number) => {
    if (!size) return;
    const { w: W, h: H } = size;
    setFitted(null);
    setView((v) => {
      if (!v) return v;
      const s = Math.min(Math.max(v.s * f, 1e-3), 1e7);
      // 커서 아래 world 점을 고정
      const wx = (px - W / 2) / v.s + v.cx;
      const wy = (py - H / 2) / v.s + v.cy;
      return { s, cx: wx - (px - W / 2) / s, cy: wy - (py - H / 2) / s };
    });
  };

  // 휠 확대 (passive 아님 — 페이지 스크롤 막기)
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      zoomAt(Math.exp(-e.deltaY * 0.0015), e.clientX - r.left, e.clientY - r.top);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  });

  if (!size || !layout || !view) return <div ref={setBox} className="min-h-0 flex-1" />;
  const W = size.w;
  const H = size.h;
  const toScreen = (x: number, y: number) => ({ x: (x - view.cx) * view.s + W / 2, y: (y - view.cy) * view.s + H / 2 });
  const switchLayout = (lay: Layout) => {
    setLayout(lay);
    fit(lay);
  };

  // 눈금: 층(row)·열(col) 번호를 그 줄의 중심 위치에. 겹치면 건너뛴다
  const centers = (key: "row" | "col") => {
    const g = new Map<number, number[]>();
    for (const w of windows) {
      const r = rectOf(w, layout);
      const c = key === "row" ? r.y + r.h / 2 : r.x + r.w / 2;
      g.set(w[key], [...(g.get(w[key]) ?? []), c]);
    }
    if (layout === "grid") for (let k = 1; k <= (key === "row" ? rows : cols); k++) if (!g.has(k)) g.set(k, [key === "row" ? (rows - k + 0.5) * cellH : k - 0.5]);
    const pts = [...g].map(([k, cs]) => ({ k, p: key === "row" ? toScreen(0, median(cs)).y : toScreen(median(cs), 0).x }));
    pts.sort((a, b) => a.p - b.p);
    const kept: typeof pts = [];
    for (const t of pts) if (!kept.length || t.p - kept[kept.length - 1].p >= TICK_MIN_PX) kept.push(t);
    return kept;
  };
  const rowTicks = centers("row");
  const colTicks = centers("col");
  const vb = `${view.cx - W / 2 / view.s} ${view.cy - H / 2 / view.s} ${W / view.s} ${H / view.s}`;
  const hatch = 6 / view.s; // 빗금 간격 6px (확대해도 같은 굵기)
  const world = worldOf(layout);
  const occupied = new Set(windows.map((w) => `${w.row}-${w.col}`));

  const btn = "rounded-md border border-neutral-700 px-2 py-1 text-xs text-neutral-300 hover:border-neutral-500 disabled:opacity-40";
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex flex-wrap items-center gap-2 px-3 py-2 text-xs">
        <div className="inline-flex overflow-hidden rounded-md border border-neutral-700" role="group" aria-label="배치">
          {(["geo", "grid"] as const).map((l) => (
            <button
              key={l}
              aria-pressed={layout === l}
              onClick={() => switchLayout(l)}
              className={`px-2.5 py-1 ${layout === l ? "bg-neutral-200 font-semibold text-neutral-900" : "text-neutral-400 hover:text-neutral-200"}`}
            >
              {l === "geo" ? "실제 배치" : "격자"}
            </button>
          ))}
        </div>
        <label className={`inline-flex items-center gap-1.5 ${layout === "grid" ? "text-neutral-600" : "text-neutral-300"}`} title={layout === "grid" ? "격자에서는 위치가 맞지 않아 배경을 쓰지 않습니다" : undefined}>
          <input type="checkbox" checked={texture && layout === "geo"} disabled={layout === "grid"} onChange={(e) => setTexture(e.target.checked)} />
          배경 텍스처
        </label>
        <ToggleChip on={multiOnly} onClick={onToggleMulti}>
          다시점만 보기
        </ToggleChip>
        <span className="flex-1" />
        <button className={btn} onClick={() => zoomAt(1.4, W / 2, H / 2)} aria-label="확대">＋</button>
        <button className={btn} onClick={() => zoomAt(1 / 1.4, W / 2, H / 2)} aria-label="축소">－</button>
        <button className={btn} onClick={() => fit(layout)}>맞춤</button>
      </div>
      <div
        ref={setBox}
        className="relative min-h-0 flex-1 cursor-grab touch-none select-none overflow-hidden bg-neutral-950 active:cursor-grabbing"
        onPointerDown={(e) => {
          drag.current = { x: e.clientX, y: e.clientY, moved: 0 };
          e.currentTarget.setPointerCapture(e.pointerId);
        }}
        onPointerMove={(e) => {
          const d = drag.current;
          if (!d) return;
          const dx = e.clientX - d.x;
          const dy = e.clientY - d.y;
          d.moved += Math.abs(dx) + Math.abs(dy);
          d.x = e.clientX;
          d.y = e.clientY;
          if (d.moved > 4) {
            setFitted(null);
            setView((v) => v && { ...v, cx: v.cx - dx / v.s, cy: v.cy - dy / v.s });
          }
        }}
        onPointerUp={(e) => {
          const d = drag.current;
          drag.current = null;
          if (d && d.moved <= 4) {
            // 클릭: pointer capture 때문에 rect의 onClick 대신 좌표로 찾는다
            const id = document.elementsFromPoint(e.clientX, e.clientY).find((el) => el instanceof SVGElement && el.dataset.win)?.getAttribute("data-win");
            if (id) onSelect(id);
          }
        }}
        onPointerLeave={() => {
          setTip(null);
          onHover(null);
        }}
      >
        <svg width={W} height={H} viewBox={vb} className="absolute inset-0" role="img" aria-label={`외벽 ${facade.id} 창문 지도`}>
          <defs>
            <pattern id="lowconf-hatch" patternUnits="userSpaceOnUse" width={hatch} height={hatch} patternTransform="rotate(45)">
              <rect width={hatch / 3} height={hatch} fill="rgba(255,255,255,0.22)" />
            </pattern>
          </defs>
          {layout === "geo" ? (
            texture ? (
              <image href={inspectionAssetUrl(inspectionId, facade.texture)} x={0} y={0} width={facade.width} height={facade.height} preserveAspectRatio="none" />
            ) : (
              <rect x={0} y={0} width={world.w} height={world.h} fill="#171717" stroke="#404040" vectorEffect="non-scaling-stroke" />
            )
          ) : (
            // 격자: 창이 없는 칸은 점선 테두리만
            Array.from({ length: rows * cols }, (_, i) => {
              const row = Math.floor(i / cols) + 1;
              const col = (i % cols) + 1;
              if (occupied.has(`${row}-${col}`)) return null;
              return (
                <rect
                  key={i}
                  x={col - 1 + GRID_GAP / 2}
                  y={(rows - row) * cellH + GRID_GAP / 2}
                  width={1 - GRID_GAP}
                  height={cellH - GRID_GAP}
                  fill="none"
                  stroke="#262626"
                  strokeDasharray="3 3"
                  vectorEffect="non-scaling-stroke"
                />
              );
            })
          )}
          {windows.map((w) => {
            const r = rectOf(w, layout);
            const lv = levelOf(w);
            const shown = visibleLevels.has(lv) && (!multiOnly || confidenceOf(w) === "multi");
            const active = w.id === selectedId || w.id === hoveredId;
            const { color } = filterStyle(lv);
            const label = `${w.id} ${w.cleaning_level ? `${w.cleaning_level}단계 ` : ""}${w.cleaning_label}`;
            return (
              <g
                key={w.id}
                data-win={shown ? w.id : undefined}
                role={shown ? "button" : undefined}
                tabIndex={shown ? 0 : undefined}
                aria-label={shown ? label : undefined}
                onKeyDown={(e) => shown && (e.key === "Enter" || e.key === " ") && onSelect(w.id)}
                onPointerEnter={(e) => {
                  if (!shown) return;
                  onHover(w.id);
                  const b = box.current!.getBoundingClientRect();
                  setTip({ x: e.clientX - b.left, y: e.clientY - b.top, w });
                }}
                onPointerLeave={() => {
                  onHover(null);
                  setTip(null);
                }}
                onFocus={() => shown && onHover(w.id)} // 키보드 포커스 = 흰 테두리 (브라우저 포커스 링 대신)
                onBlur={() => onHover(null)}
                className="outline-none"
                style={{ cursor: shown ? "pointer" : "default" }}
              >
                {/* 필터(단계·다시점만)로 뺀 창: 위치만 흐린 회색 윤곽 */}
                <rect
                  data-win={shown ? w.id : undefined}
                  x={r.x}
                  y={r.y}
                  width={r.w}
                  height={r.h}
                  fill={shown ? color : "none"}
                  fillOpacity={shown ? (layout === "geo" && texture ? 0.72 : 0.9) : 0}
                  stroke={shown ? (active ? "#ffffff" : color) : "#525252"}
                  strokeOpacity={shown ? 1 : 0.5}
                  strokeWidth={active ? 3 : shown ? 1.5 : 1}
                  strokeDasharray={shown && confidenceOf(w) === "single" ? "4 2" : undefined}
                  vectorEffect="non-scaling-stroke"
                />
                {shown && confidenceOf(w) === "single" && (
                  <rect data-win={w.id} x={r.x} y={r.y} width={r.w} height={r.h} fill="url(#lowconf-hatch)" pointerEvents="none" />
                )}
              </g>
            );
          })}
        </svg>
        {/* 눈금 (화면 고정 글자 크기) */}
        <div className="pointer-events-none absolute inset-y-0 left-0 bg-neutral-950/80" style={{ width: GUTTER_L }} aria-hidden>
          {rowTicks.map((t) => (
            <span key={t.k} className="absolute right-1.5 -translate-y-1/2 text-[11px] tabular-nums text-neutral-400" style={{ top: t.p }}>
              {t.k}층
            </span>
          ))}
        </div>
        <div className="pointer-events-none absolute inset-x-0 top-0 bg-neutral-950/80" style={{ height: GUTTER_T, paddingLeft: GUTTER_L }} aria-hidden>
          {colTicks.map((t) =>
            t.p > GUTTER_L ? (
              <span key={t.k} className="absolute top-1 -translate-x-1/2 text-[11px] tabular-nums text-neutral-400" style={{ left: t.p }}>
                {t.k}
              </span>
            ) : null,
          )}
        </div>
        <span className="pointer-events-none absolute left-1 top-1 text-[10px] text-neutral-600" aria-hidden>
          층＼열
        </span>
        {tip && <Tooltip x={tip.x} y={tip.y} w={tip.w} W={W} />}
        <div className="pointer-events-none absolute bottom-2 left-12 flex flex-wrap gap-3 rounded-md bg-neutral-950/85 px-2.5 py-1.5 text-[11px] text-neutral-400">
          <span className="inline-flex items-center gap-1.5">
            <svg width="14" height="10" aria-hidden>
              <rect width="14" height="10" fill={LEVEL_STYLES[3].color} />
              <path d="M0 10L10 0M4 10L14 0M-4 6L2 0" stroke="rgba(255,255,255,0.6)" strokeWidth="1.5" />
            </svg>
            빗금 = 신뢰도 낮음 (단일 이미지)
          </span>
          <span>휠 확대 · 드래그 이동 · 클릭 상세</span>
        </div>
      </div>
    </div>
  );
}

function Tooltip({ x, y, w, W }: { x: number; y: number; w: Window; W: number }) {
  const lv = levelOf(w);
  const raw = rawCoveragePct(w);
  const left = x > W - 260 ? x - 250 : x + 14;
  return (
    <div className="pointer-events-none absolute z-10 w-60 rounded-md border border-neutral-700 bg-neutral-900/95 px-3 py-2 text-xs shadow-lg" style={{ left, top: y + 14 }}>
      <div className="font-mono text-neutral-200">{w.id}</div>
      <div className="mt-0.5 font-semibold" style={{ color: filterStyle(lv).color }}>
        {w.cleaning_level ? `${w.cleaning_level}단계 ` : ""}
        {w.cleaning_label}
      </div>
      {raw !== null && <div className="text-neutral-300">오염 면적 {raw.toFixed(raw < 10 ? 1 : 0)}%</div>}
      <div className="text-neutral-500">{CONFIDENCE_LABELS[confidenceOf(w)]}</div>
    </div>
  );
}
