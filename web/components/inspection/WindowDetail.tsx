"use client";

/* eslint-disable @next/next/no-img-element -- 정적/백엔드 원본 이미지를 그대로 쓴다 (최적화 불필요, 크기 미상) */
import { useEffect, useState } from "react";
import { GradeBadge, ToggleChip } from "@/components/ui";
import { inspectionAssetUrl } from "@/lib/api";
import { pct, TYPE_COLORS, TYPE_LABELS } from "@/lib/palette";
import { CONTAMINATION_TYPES, type Window } from "@/lib/types";

type Layer = "crop" | "heatmap" | "polygons";
const LAYERS: { key: Layer; label: string }[] = [
  { key: "crop", label: "원본" },
  { key: "heatmap", label: "히트맵" },
  { key: "polygons", label: "오염 영역" },
];

export default function WindowDetail({
  inspectionId,
  window: w,
  onClose,
}: {
  inspectionId: string;
  window: Window | null;
  onClose: () => void;
}) {
  const [layers, setLayers] = useState<Set<Layer>>(() => new Set(["crop", "polygons"]));
  const [hoverPoly, setHoverPoly] = useState<number | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!w) return null;
  const toggle = (l: Layer) =>
    setLayers((s) => {
      const n = new Set(s);
      if (!n.delete(l)) n.add(l);
      return n;
    });
  const hovered = hoverPoly === null ? null : w.contamination[hoverPoly];
  const maxRatio = Math.max(0.05, ...Object.values(w.type_ratios));

  return (
    <aside
      aria-label={`창문 ${w.id} 상세`}
      className="fixed inset-x-0 bottom-0 z-30 max-h-[85dvh] overflow-y-auto rounded-t-2xl border-t border-neutral-800 bg-neutral-950/95 shadow-2xl backdrop-blur lg:inset-x-auto lg:bottom-0 lg:right-0 lg:top-14 lg:max-h-none lg:w-[420px] lg:rounded-none lg:border-l lg:border-t-0"
    >
      <header className="sticky top-0 z-10 flex items-center gap-3 border-b border-neutral-800 bg-neutral-950/95 px-4 py-3">
        <GradeBadge grade={w.grade} size="lg" />
        <div className="min-w-0 flex-1">
          <div className="truncate font-mono text-sm">{w.id}</div>
          <div className="text-xs text-neutral-500">
            외벽 {w.facade_id} · {w.row}층 · {w.col}열 · 관측 뷰 {w.num_views}개
          </div>
        </div>
        <div className="text-right">
          <div className="text-xs text-neutral-500">점수</div>
          <div className="font-semibold tabular-nums">{w.score.toFixed(3)}</div>
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="닫기"
          className="rounded-md px-2 py-1 text-lg text-neutral-400 hover:bg-neutral-800 hover:text-neutral-100"
        >
          ×
        </button>
      </header>

      <div className="space-y-4 p-4">
        <div className="flex flex-wrap gap-1.5">
          {LAYERS.map((l) => (
            <ToggleChip key={l.key} on={layers.has(l.key)} onClick={() => toggle(l.key)}>
              {l.label}
            </ToggleChip>
          ))}
        </div>

        <div className="relative overflow-hidden rounded-lg border border-neutral-800 bg-neutral-900">
          {/* 원본을 꺼도 크기를 유지하도록 crop은 항상 배치하고 투명도만 바꾼다 */}
          <img
            src={inspectionAssetUrl(inspectionId, w.images.crop)}
            alt={`${w.id} 정면화 crop`}
            className="block w-full"
            style={{ opacity: layers.has("crop") ? 1 : 0 }}
          />
          {layers.has("heatmap") && (
            <img
              src={inspectionAssetUrl(inspectionId, w.images.heatmap)}
              alt=""
              className="absolute inset-0 h-full w-full opacity-80"
            />
          )}
          {layers.has("polygons") && (
            <svg
              viewBox="0 0 1 1"
              preserveAspectRatio="none"
              className="absolute inset-0 h-full w-full"
              onMouseLeave={() => setHoverPoly(null)}
            >
              {w.contamination.map((c, i) => (
                <polygon
                  key={i}
                  points={c.polygon.map(([u, v]) => `${u},${v}`).join(" ")}
                  fill={TYPE_COLORS[c.type]}
                  fillOpacity={hoverPoly === i ? 0.55 : 0.3}
                  stroke={TYPE_COLORS[c.type]}
                  strokeWidth={hoverPoly === i ? 3 : 1.5}
                  vectorEffect="non-scaling-stroke"
                  onMouseEnter={() => setHoverPoly(i)}
                  className="cursor-crosshair"
                />
              ))}
            </svg>
          )}
          {hovered && (
            <div className="pointer-events-none absolute left-2 top-2 rounded-md bg-neutral-950/90 px-2 py-1 text-xs">
              <span className="font-semibold" style={{ color: TYPE_COLORS[hovered.type] }}>
                {TYPE_LABELS[hovered.type]}
              </span>
              <span className="ml-2 tabular-nums text-neutral-300">면적 {pct(hovered.area_ratio)}</span>
            </div>
          )}
        </div>
        <p className="text-xs text-neutral-500">
          오염 영역 {w.contamination.length}개 — 영역에 마우스를 올리면 유형·면적비를 표시합니다.
        </p>

        <section>
          <h3 className="mb-2 text-xs font-medium text-neutral-400">유형별 면적 비율</h3>
          <ul className="space-y-1.5">
            {CONTAMINATION_TYPES.map((t) => (
              <li key={t} className="flex items-center gap-2 text-sm">
                <span className="h-3 w-3 shrink-0 rounded-sm" style={{ backgroundColor: TYPE_COLORS[t] }} />
                <span className="w-16 shrink-0">{TYPE_LABELS[t]}</span>
                <span className="h-2 flex-1 overflow-hidden rounded-full bg-neutral-800">
                  <span
                    className="block h-full rounded-full"
                    style={{ width: `${(w.type_ratios[t] / maxRatio) * 100}%`, backgroundColor: TYPE_COLORS[t] }}
                  />
                </span>
                <span className="w-14 text-right tabular-nums text-neutral-300">{pct(w.type_ratios[t])}</span>
              </li>
            ))}
          </ul>
        </section>
      </div>
    </aside>
  );
}
