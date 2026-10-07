"use client";

/* eslint-disable @next/next/no-img-element -- 원본 프레임 픽셀 좌표에 SVG를 맞추므로 크기 그대로의 <img>가 필요하다 */
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";
import { SourceBadge, Status, ToggleChip } from "@/components/ui";
import { frameUrl, getDetections, listDetectionRuns, rectifiedUrl } from "@/lib/api";
import type { DetectedWindow, FrameDetections } from "@/lib/types";
import { useData } from "@/lib/useData";

type Layer = "window" | "flagged" | "fill" | "removed";
const LAYERS: Record<Layer, { label: string; color: string }> = {
  window: { label: "확정 창문", color: "#22c55e" },
  flagged: { label: "플래그", color: "#f97316" },
  fill: { label: "보완 후보", color: "#22d3ee" },
  removed: { label: "제거됨", color: "#ef4444" },
};
const FLAG_LABELS: Record<string, string> = {
  off_grid: "격자 밖",
  odd_width: "폭 이상",
  odd_height: "높이 이상",
  fill_candidate: "보완 후보",
};
const REASON_LABELS: Record<string, string> = { small: "너무 작음", not_rect: "사각형 아님", outside_facade: "외벽 밖" };
const WARNING_LABELS: Record<string, string> = {
  facade_filter_skipped: "외벽 필터 건너뜀",
  rectification_failed: "정면화 실패",
  grid_unreliable: "격자 신뢰 낮음",
};

type Item = { layer: Layer; corners: [number, number][]; conf: number; label: string; detail: string };

function items(f: FrameDetections): Item[] {
  const cell = (w: DetectedWindow) => (w.row === null ? "–" : `r${w.row}c${w.col}`);
  return [
    ...f.removed.map((w) => ({
      layer: "removed" as const,
      corners: w.corners,
      conf: w.conf,
      label: REASON_LABELS[w.reason],
      detail: `제거: ${REASON_LABELS[w.reason]}`,
    })),
    ...f.windows.map((w) => ({
      layer: (w.flags.length ? "flagged" : "window") as Layer,
      corners: w.corners,
      conf: w.conf,
      label: [cell(w), ...w.flags.map((x) => FLAG_LABELS[x])].join(" "),
      detail: w.flags.length ? `플래그: ${w.flags.map((x) => FLAG_LABELS[x]).join(", ")}` : "확정 창문",
    })),
    ...f.fill_candidates.map((w) => ({
      layer: "fill" as const,
      corners: w.corners,
      conf: w.conf,
      label: `${cell(w)} 보완`,
      detail: "보완 후보 (저신뢰 검출이 격자 빈칸을 채움)",
    })),
  ];
}

function FrameOverlay({
  src,
  frame,
  layers,
  labels,
  hovered,
  onHover,
}: {
  src: string;
  frame: FrameDetections;
  layers: Set<Layer>;
  labels: boolean;
  hovered: number | null;
  onHover: (i: number | null) => void;
}) {
  const [size, setSize] = useState<[number, number] | null>(null);
  const all = items(frame);
  const font = size ? Math.max(...size) / 75 : 0;
  return (
    <div className="relative overflow-hidden rounded-lg border border-neutral-800 bg-neutral-900">
      <img
        src={src}
        alt={frame.image}
        className="block w-full"
        onLoad={(e) => setSize([e.currentTarget.naturalWidth, e.currentTarget.naturalHeight])}
      />
      {size && (
        <svg viewBox={`0 0 ${size[0]} ${size[1]}`} className="absolute inset-0 h-full w-full" onMouseLeave={() => onHover(null)}>
          {all.map((it, i) =>
            layers.has(it.layer) ? (
              <g key={i} onMouseEnter={() => onHover(i)}>
                <polygon
                  points={it.corners.map((p) => p.join(",")).join(" ")}
                  fill={LAYERS[it.layer].color}
                  fillOpacity={hovered === i ? 0.35 : 0.08}
                  stroke={LAYERS[it.layer].color}
                  strokeWidth={hovered === i ? 4 : 2}
                  strokeDasharray={it.layer === "removed" || it.layer === "fill" ? "8 5" : undefined}
                  vectorEffect="non-scaling-stroke"
                />
                {labels && (
                  <text
                    x={it.corners[0][0] + font * 0.3}
                    y={it.corners[0][1] + font * 1.1}
                    fontSize={font}
                    fill={LAYERS[it.layer].color}
                    stroke="#000"
                    strokeWidth={font / 5}
                    paintOrder="stroke"
                    className="pointer-events-none select-none font-mono"
                  >
                    {it.label}
                  </text>
                )}
              </g>
            ) : null,
          )}
        </svg>
      )}
    </div>
  );
}

function RectifiedImage({ src }: { src: string }) {
  const [failed, setFailed] = useState(false);
  if (failed) return <Status>정면화 이미지 없음 (detect.run --debug 결과가 필요)</Status>;
  return (
    <img
      src={src}
      alt="정면화 이미지"
      onError={() => setFailed(true)}
      className="block w-full rounded-lg border border-neutral-800"
    />
  );
}

export default function DetectionsView() {
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const index = useData("detection-runs", listDetectionRuns);
  const runId = params.get("run") ?? index.data?.[0]?.id ?? null;
  const dets = useData(runId && `detections/${runId}`, () => getDetections(runId!));
  const [layers, setLayers] = useState<Set<Layer>>(() => new Set(Object.keys(LAYERS) as Layer[]));
  const [labels, setLabels] = useState(true);
  const [sideBySide, setSideBySide] = useState(false);
  const [hovered, setHovered] = useState<number | null>(null);

  const error = index.error ?? dets.error;
  if (error) return <Status error={error} />;
  if (!index.data || !dets.data || !runId) return <Status />;
  const frames = dets.data.images;
  const frame = frames.find((f) => f.image === params.get("frame")) ?? frames[0];
  if (!frame) return <Status>프레임이 없습니다</Status>;

  const go = (next: Record<string, string>) => {
    const q = new URLSearchParams(params);
    for (const [k, v] of Object.entries(next)) q.set(k, v);
    if (next.run) q.delete("frame");
    setHovered(null);
    router.replace(`${pathname}?${q}`, { scroll: false });
  };
  const toggle = (l: Layer) =>
    setLayers((s) => {
      const n = new Set(s);
      if (!n.delete(l)) n.add(l);
      return n;
    });
  const all = items(frame);
  const count = (l: Layer) => all.filter((it) => it.layer === l).length;
  const c = frame.counts;
  const hoveredItem = hovered === null ? null : all[hovered];

  return (
    <div className="flex flex-1 flex-col lg:flex-row">
      <nav className="space-y-4 border-b border-neutral-800 p-4 lg:w-64 lg:shrink-0 lg:border-b-0 lg:border-r">
        <div>
          <h2 className="mb-2 text-xs font-medium text-neutral-400">영상</h2>
          <div className="flex flex-wrap gap-1.5 lg:flex-col">
            {index.data.map((r) => (
              <button
                key={r.id}
                type="button"
                onClick={() => go({ run: r.id })}
                className={`rounded-md border px-2.5 py-1.5 text-left font-mono text-xs ${
                  r.id === runId ? "border-neutral-400 bg-neutral-800" : "border-neutral-800 hover:border-neutral-600"
                }`}
              >
                {r.id.split("/").pop()}
              </button>
            ))}
          </div>
        </div>
        <div>
          <h2 className="mb-2 text-xs font-medium text-neutral-400">프레임 ({frames.length})</h2>
          <div className="grid grid-cols-3 gap-2 lg:grid-cols-2">
            {frames.map((f) => (
              <button
                key={f.image}
                type="button"
                onClick={() => go({ frame: f.image })}
                className={`overflow-hidden rounded-md border text-left ${
                  f.image === frame.image ? "border-neutral-300" : "border-neutral-800 hover:border-neutral-600"
                }`}
              >
                <img src={frameUrl(runId, f.image)} alt={f.image} loading="lazy" className="block aspect-video w-full object-cover" />
                <div className="px-1.5 py-1 text-[10px] tabular-nums text-neutral-400">
                  #{f.image.match(/frame_(\d+)/)?.[1] ?? f.image} · {f.counts.raw}→{f.counts.final}
                </div>
              </button>
            ))}
          </div>
        </div>
      </nav>

      <section className="min-w-0 flex-1 space-y-3 p-4">
        <SourceBadge kind="real">실제 검출 결과 · {modelLabel(dets.data.weights)}</SourceBadge>
        <div className="flex flex-wrap items-center gap-1.5">
          {(Object.keys(LAYERS) as Layer[]).map((l) => (
            <ToggleChip key={l} on={layers.has(l)} onClick={() => toggle(l)} color={LAYERS[l].color}>
              {LAYERS[l].label} <span className="tabular-nums text-neutral-400">{count(l)}</span>
            </ToggleChip>
          ))}
          <span className="mx-1 h-4 w-px bg-neutral-800" />
          <ToggleChip on={labels} onClick={() => setLabels((v) => !v)}>
            row/col 라벨
          </ToggleChip>
          <ToggleChip on={sideBySide} onClick={() => setSideBySide((v) => !v)}>
            정면화 나란히
          </ToggleChip>
        </div>
        <div className={`grid items-start gap-3 ${sideBySide ? "xl:grid-cols-2" : ""}`}>
          <FrameOverlay
            key={`${runId}/${frame.image}`}
            src={frameUrl(runId, frame.image)}
            frame={frame}
            layers={layers}
            labels={labels}
            hovered={hovered}
            onHover={setHovered}
          />
          {sideBySide &&
            (frame.rectified ? (
              <RectifiedImage key={`${runId}/${frame.image}`} src={rectifiedUrl(runId, frame.image)} />
            ) : (
              <Status>이 프레임은 정면화에 실패했습니다</Status>
            ))}
        </div>
        <p className="font-mono text-xs text-neutral-500">{frame.image}</p>
      </section>

      <aside className="space-y-5 border-t border-neutral-800 p-4 text-sm lg:w-72 lg:shrink-0 lg:border-l lg:border-t-0">
        <div>
          <h2 className="mb-2 text-xs font-medium text-neutral-400">후처리 전 → 후</h2>
          <div className="flex items-baseline gap-2 text-2xl font-semibold tabular-nums">
            {c.raw} <span className="text-base text-neutral-500">→</span> {c.final}
            {c.fill_candidates > 0 && <span className="text-base text-cyan-400">+{c.fill_candidates} 보완</span>}
          </div>
          <dl className="mt-3 grid grid-cols-[1fr_auto] gap-y-1 text-neutral-300">
            <dt>제거됨</dt>
            <dd className="tabular-nums">{c.removed}</dd>
            {Object.entries(REASON_LABELS).map(([k, v]) => {
              const n = frame.removed.filter((w) => w.reason === k).length;
              return n ? (
                <Row key={k} label={`└ ${v}`} value={n} />
              ) : null;
            })}
            <dt>플래그</dt>
            <dd className="tabular-nums">{c.flagged}</dd>
            {["off_grid", "odd_width", "odd_height"].map((k) => {
              const n = frame.windows.filter((w) => w.flags.includes(k as DetectedWindow["flags"][number])).length;
              return n ? <Row key={k} label={`└ ${FLAG_LABELS[k]}`} value={n} /> : null;
            })}
            <dt>격자 빈칸</dt>
            <dd className="tabular-nums">{c.empty_cells}</dd>
            <dt>외벽 필터</dt>
            <dd>{frame.facade_filter ? "적용" : "건너뜀"}</dd>
            <dt>정면화</dt>
            <dd>{frame.rectified ? `성공 (f=${frame.focal_px?.toFixed(0)}px)` : "실패"}</dd>
          </dl>
        </div>

        <div>
          <h2 className="mb-2 text-xs font-medium text-neutral-400">경고 ({frame.warnings.length})</h2>
          {frame.warnings.length ? (
            <ul className="space-y-2">
              {frame.warnings.map((w) => {
                const [key, ...rest] = w.split(":");
                return (
                  <li key={w} className="rounded-md border border-amber-900/60 bg-amber-950/30 p-2 text-xs text-amber-200">
                    <div className="font-medium">{WARNING_LABELS[key] ?? key}</div>
                    {rest.length > 0 && <div className="mt-0.5 text-amber-200/70">{rest.join(":").trim()}</div>}
                  </li>
                );
              })}
            </ul>
          ) : (
            <p className="text-xs text-neutral-500">없음</p>
          )}
        </div>

        <div>
          <h2 className="mb-2 text-xs font-medium text-neutral-400">마우스 위 검출</h2>
          {hoveredItem ? (
            <div className="rounded-md border border-neutral-800 p-2 text-xs">
              <div style={{ color: LAYERS[hoveredItem.layer].color }}>{hoveredItem.detail}</div>
              <div className="mt-1 tabular-nums text-neutral-400">conf {hoveredItem.conf.toFixed(3)}</div>
              <div className="font-mono text-neutral-500">{hoveredItem.label}</div>
            </div>
          ) : (
            <p className="text-xs text-neutral-500">프레임 위 사각형에 마우스를 올리세요</p>
          )}
        </div>
        <p className="break-all text-[10px] text-neutral-600">weights: {dets.data.weights.split("/").pop()}</p>
      </aside>
    </div>
  );
}

/** seg_window_facade_v1.pt → "모델 v1" (버전 표기가 없으면 파일명). */
function modelLabel(weights: string) {
  const name = weights.split("/").pop() ?? weights;
  const v = name.match(/_v(\d+)\.pt$/)?.[1];
  return v ? `모델 v${v}` : name;
}

function Row({ label, value }: { label: string; value: number }) {
  return (
    <>
      <dt className="pl-2 text-neutral-500">{label}</dt>
      <dd className="tabular-nums text-neutral-500">{value}</dd>
    </>
  );
}
