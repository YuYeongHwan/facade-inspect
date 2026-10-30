"use client";

/* eslint-disable @next/next/no-img-element -- 정적/백엔드 원본 이미지를 그대로 쓴다 (최적화 불필요, 크기 미상) */
import { useEffect, useState } from "react";
import { LevelBadge, ToggleChip } from "@/components/ui";
import { inspectionAssetUrl } from "@/lib/api";
import { LEVEL_STYLES, pct, SEVERITY_LABELS, TYPE_COLORS, TYPE_LABELS } from "@/lib/palette";
import { CONTAMINATION_TYPES, type CsiParams, csiTerms, type Window } from "@/lib/types";

type Layer = "crop" | "heatmap" | "polygons";
const LAYERS: { key: Layer; label: string }[] = [
  { key: "crop", label: "원본" },
  { key: "heatmap", label: "히트맵" },
  { key: "polygons", label: "오염 영역" },
];

export default function WindowDetail({
  inspectionId,
  window: w,
  csiParams,
  onClose,
}: {
  inspectionId: string;
  window: Window | null;
  csiParams: CsiParams | null;
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
  const ratios = w.type_ratios; // null = 미분석 (T7 전)
  const maxRatio = ratios ? Math.max(0.05, ...Object.values(ratios)) : 1;
  const heatmap = w.images.heatmap;
  const layerOptions = heatmap ? LAYERS : LAYERS.filter((l) => l.key !== "heatmap");
  // 오염 면적 (배수 없음). 이전 result.json엔 raw_coverage_pct가 없으므로 영역 면적비 합으로 대신한다
  const rawPct = w.raw_coverage_pct ?? (ratios ? Math.min(100, Object.values(ratios).reduce((a, b) => a + b, 0) * 100) : null);
  const kMul = csiParams ? csiParams.severity_max * Math.max(...Object.values(csiParams.weights)) : null;

  return (
    <aside
      aria-label={`창문 ${w.id} 상세`}
      className="fixed inset-x-0 bottom-0 z-30 max-h-[85dvh] overflow-y-auto rounded-t-2xl border-t border-neutral-800 bg-neutral-950/95 shadow-2xl backdrop-blur lg:inset-x-auto lg:bottom-0 lg:right-0 lg:top-14 lg:max-h-none lg:w-[420px] lg:rounded-none lg:border-l lg:border-t-0"
    >
      <header className="sticky top-0 z-10 flex items-center gap-3 border-b border-neutral-800 bg-neutral-950/95 px-4 py-3">
        <LevelBadge level={w.cleaning_level} size="lg" lowConfidence={!!w.low_confidence} />
        <div className="min-w-0 flex-1">
          <div className="truncate font-mono text-sm">{w.id}</div>
          <div className="text-xs text-neutral-500">
            외벽 {w.facade_id} · {w.row}층 · {w.col}열 · 관측 뷰 {w.num_views}개
          </div>
        </div>
        <div className="text-right">
          <div className="text-xs text-neutral-500">CSI</div>
          <div className="font-semibold tabular-nums">{w.csi === null ? "—" : w.csi.toFixed(3)}</div>
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
          {layerOptions.map((l) => (
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
          {heatmap && layers.has("heatmap") && (
            <img
              src={inspectionAssetUrl(inspectionId, heatmap)}
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
              {(w.reflection_suspects ?? []).map((r, i) => (
                <polygon
                  key={`r${i}`}
                  points={r.polygon.map(([u, v]) => `${u},${v}`).join(" ")}
                  fill="none"
                  stroke="#facc15"
                  strokeWidth={2}
                  strokeDasharray="6 4"
                  vectorEffect="non-scaling-stroke"
                >
                  <title>반사 의심 (유리 면적의 {pct(r.area_ratio)}) — CSI에서 제외</title>
                </polygon>
              ))}
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
              <span className="ml-2 tabular-nums text-neutral-300">
                유리 면적의 {pct(hovered.area_ratio)} · 심각도 {hovered.severity}
              </span>
            </div>
          )}
        </div>
        <p className="text-xs text-neutral-500">
          {ratios
            ? `오염 영역 ${w.contamination.length}개 — 영역에 마우스를 올리면 유형·면적비를 표시합니다.` +
              (w.reflection_suspects?.length
                ? ` 노란 점선 ${w.reflection_suspects.length}곳은 반사 의심(비친 나무·건물)으로 CSI에서 뺐습니다.`
                : "")
            : "미분석 — 오염 분석(T7) 전입니다. 3D 복원 결과의 창문 위치·이미지만 있습니다."}
        </p>

        {w.cleaning_level !== null && rawPct !== null && (
          <section className="rounded-lg border border-neutral-800 px-3 py-2">
            <p className="text-base">
              오염 면적 <span className="font-semibold tabular-nums">{rawPct.toFixed(rawPct < 10 ? 1 : 0)}%</span> ·{" "}
              <span className="font-semibold" style={{ color: LEVEL_STYLES[w.cleaning_level].color }}>
                {w.cleaning_level}단계 {LEVEL_STYLES[w.cleaning_level].label}
              </span>
            </p>
            {w.analysis_mode && (
              <p className="mt-1 flex flex-wrap gap-1.5 text-[11px] text-neutral-400">
                <span className="rounded bg-neutral-800 px-1.5 py-0.5">
                  {w.analysis_mode === "multi" ? `다시점 ${w.num_views}장 정렬` : "단일 이미지"}
                </span>
                {w.low_confidence && (
                  <span
                    className="rounded bg-amber-900/60 px-1.5 py-0.5 text-amber-200"
                    title="한 장으로 분석해 반사·실내 비침을 오염과 구분하지 못했을 수 있습니다"
                  >
                    신뢰도 낮음
                  </span>
                )}
              </p>
            )}
            {w.weighted_coverage_pct != null && (
              <details className="mt-1.5 text-xs text-neutral-400">
                <summary className="cursor-pointer select-none">
                  상세: 가중 오염 면적률 <span className="tabular-nums">{w.weighted_coverage_pct.toFixed(1)}%</span> · CSI{" "}
                  <span className="tabular-nums">{w.csi?.toFixed(3)}</span>
                </summary>
                <p className="mt-1 text-[11px] leading-relaxed text-neutral-500">
                  오염 면적 = Σ 오염 면적 ÷ 유리 면적 (배수 없음, 최대 100%). 가중 오염 면적률 = Σ (오염 면적/유리 면적 × 심각도 k_s ×
                  종류 가중치 k_t) × 100 = CSI × {kMul ?? "k_s_max × k_t_max"} × 100 (최대 {kMul !== null ? kMul * 100 : "—"}%). 등급은
                  CSI 구간으로 정한다 — 아래 계산 내역.
                </p>
              </details>
            )}
          </section>
        )}

        {ratios && (
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
                      style={{ width: `${(ratios[t] / maxRatio) * 100}%`, backgroundColor: TYPE_COLORS[t] }}
                    />
                  </span>
                  <span className="w-14 text-right tabular-nums text-neutral-300">{pct(ratios[t])}</span>
                </li>
              ))}
            </ul>
          </section>
        )}
        {w.csi !== null && csiParams && <CsiBreakdown w={w} p={csiParams} hover={hoverPoly} onHover={setHoverPoly} />}
      </div>
    </aside>
  );
}

/** CSI 계산 내역: CSI = Σ (A_j/A_g × k_s × k_t) / (k_s_max × k_t_max) — docs/scoring.md */
function CsiBreakdown({
  w,
  p,
  hover,
  onHover,
}: {
  w: Window;
  p: CsiParams;
  hover: number | null;
  onHover: (i: number | null) => void;
}) {
  const kMax = Math.max(...Object.values(p.weights));
  const terms = csiTerms(w.contamination, p);
  const total = terms.reduce((s, t) => s + t, 0);
  const level = w.cleaning_level;
  const next = level !== null && level < 5 ? p.level_min_csi[level] : null;
  return (
    <section>
      <h3 className="mb-1 text-xs font-medium text-neutral-400">CSI 계산 내역</h3>
      <p className="mb-2 font-mono text-[11px] leading-relaxed text-neutral-500">
        CSI = Σ (A_j/A_g × k_s × k_t) ÷ (k_s_max {p.severity_max} × k_t_max {kMax})
      </p>
      <dl className="mb-2 grid grid-cols-2 gap-x-3 gap-y-0.5 text-xs">
        <dt className="text-neutral-500">유리 면적 A_g</dt>
        <dd className="text-right tabular-nums">crop의 {pct(w.glass_ratio ?? 0)}</dd>
        <dt className="text-neutral-500">창틀 마진 (각 변)</dt>
        <dd className="text-right tabular-nums">{pct(p.frame_margin)}</dd>
        <dt className="text-neutral-500">가림 (나무·전선·기둥)</dt>
        <dd className={`text-right tabular-nums ${w.occluded_ratio ? "text-amber-300" : ""}`}>
          crop의 {pct(w.occluded_ratio ?? 0)} — 계산에서 제외
        </dd>
      </dl>
      {w.contamination.length ? (
        <table className="w-full text-xs tabular-nums">
          <thead className="text-neutral-500">
            <tr className="border-b border-neutral-800">
              <th className="py-1 text-left font-normal">#</th>
              <th className="text-left font-normal">종류 (k_t)</th>
              <th className="text-right font-normal">A_j/A_g</th>
              <th className="text-right font-normal" title="깨끗한 유리 대비 색차 ΔE*ab">심각도 (ΔE)</th>
              <th className="text-right font-normal">기여</th>
            </tr>
          </thead>
          <tbody>
            {w.contamination.map((c, i) => (
              <tr
                key={i}
                onMouseEnter={() => onHover(i)}
                onMouseLeave={() => onHover(null)}
                className={`border-b border-neutral-900 ${hover === i ? "bg-neutral-800/60" : ""}`}
              >
                <td className="py-1 text-neutral-500">{i + 1}</td>
                <td>
                  <span className="mr-1 inline-block h-2 w-2 rounded-sm" style={{ backgroundColor: TYPE_COLORS[c.type] }} />
                  {TYPE_LABELS[c.type]} ({p.weights[c.type]})
                </td>
                <td className="text-right">{pct(c.area_ratio, 2)}</td>
                <td className="text-right" title={SEVERITY_LABELS[c.severity]}>
                  {c.severity} ({c.delta_e === null ? "—" : c.delta_e.toFixed(1)})
                </td>
                <td className="text-right">
                  {terms[i].toFixed(4)}
                  <span className="ml-1 text-neutral-500">{total > 0 ? pct(terms[i] / total, 0) : ""}</span>
                </td>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr>
              <td colSpan={4} className="py-1 text-right text-neutral-400">
                합계 CSI
              </td>
              <td className="text-right font-semibold">{total.toFixed(4)}</td>
            </tr>
          </tfoot>
        </table>
      ) : (
        <p className="text-xs text-neutral-500">오염 영역 없음 → CSI 0</p>
      )}
      {level !== null && (
        <p className="mt-2 text-xs text-neutral-400">
          → level {level} {LEVEL_STYLES[level].label} (CSI ≥ {p.level_min_csi[level - 1]}
          {next !== null && `, 다음 단계 ${next}`}) · 기준값: {p.status}
        </p>
      )}
    </section>
  );
}
