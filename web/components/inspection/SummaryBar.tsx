import { SourceBadge, ToggleChip } from "@/components/ui";
import { filterStyle } from "@/lib/palette";
import { type InspectionResult, LEVEL_FILTERS, type LevelFilter, levelOf } from "@/lib/types";

export default function SummaryBar({
  result,
  source,
  visibleLevels,
  onToggleLevel,
}: {
  result: InspectionResult;
  source?: "mock" | "run";
  visibleLevels: Set<LevelFilter>;
  onToggleLevel: (lv: LevelFilter) => void;
}) {
  const total = result.windows.length;
  const counts = Object.fromEntries(
    LEVEL_FILTERS.map((lv) => [lv, result.windows.filter((w) => levelOf(w) === lv).length]),
  ) as Record<LevelFilter, number>;
  const csis = result.windows.flatMap((w) => (w.csi === null ? [] : [w.csi]));
  const avg = csis.length ? csis.reduce((s, x) => s + x, 0) / csis.length : null;
  const analyzed = result.windows.filter((w) => w.cleaning_level !== null);
  const multi = analyzed.filter((w) => w.analysis_mode === "multi").length;
  const single = analyzed.filter((w) => w.analysis_mode === "single").length;
  // 미분석은 있을 때만 막대·칩에 보인다
  const shown = LEVEL_FILTERS.filter((lv) => lv !== "unanalyzed" || counts[lv]);
  return (
    <div className="flex flex-wrap items-center gap-x-6 gap-y-3 border-b border-neutral-800 px-4 py-3">
      <div>
        <div className="text-xs text-neutral-500">검사</div>
        <div className="flex items-center gap-2">
          <span className="font-mono text-sm">{result.inspection_id}</span>
          {source === "mock" && <SourceBadge kind="mock">목업 데이터 · 오염 분석 미구현</SourceBadge>}
        </div>
      </div>
      <Stat label="전체 창문" value={total.toString()} />
      <Stat label="평균 CSI" value={avg === null ? "—" : avg.toFixed(3)} />
      {(multi > 0 || single > 0) && (
        <div title="다시점 = 여러 프레임 정렬로 반사를 거른 결과 · 단일 = 한 장 분석 (신뢰도 낮음, 목록·지도에 빗금)">
          <div className="text-xs text-neutral-500">분석 방식</div>
          <div className="text-sm tabular-nums">
            다시점 <span className="font-semibold">{multi}</span> ·{" "}
            <span className="text-amber-300">
              단일 <span className="font-semibold">{single}</span>
            </span>
          </div>
        </div>
      )}
      <div className={`rounded-lg border px-3 py-1 ${counts[5] ? "border-red-500 bg-red-500/15" : "border-neutral-800"}`}>
        <div className={`text-xs ${counts[5] ? "font-semibold text-red-300" : "text-neutral-500"}`}>우선 청소</div>
        <div className={`text-lg font-bold tabular-nums ${counts[5] ? "text-red-400" : ""}`}>{counts[5]}개</div>
      </div>
      <div className="min-w-56 flex-1">
        <div className="mb-1 text-xs text-neutral-500">청소 우선순위 분포</div>
        <div className="flex h-3 overflow-hidden rounded-full bg-neutral-800" role="img" aria-label="청소 우선순위 분포">
          {shown.map((lv) =>
            counts[lv] ? (
              <div
                key={lv}
                title={`${filterStyle(lv).label}: ${counts[lv]}`}
                style={{
                  width: `${(counts[lv] / total) * 100}%`,
                  backgroundColor: filterStyle(lv).color,
                  opacity: visibleLevels.has(lv) ? 1 : 0.2,
                }}
              />
            ) : null,
          )}
        </div>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {shown.map((lv) => (
          <ToggleChip key={lv} on={visibleLevels.has(lv)} onClick={() => onToggleLevel(lv)} color={filterStyle(lv).color}>
            {lv === "unanalyzed" ? "" : `${lv} `}
            {filterStyle(lv).label} <span className="tabular-nums text-neutral-400">{counts[lv]}</span>
          </ToggleChip>
        ))}
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs text-neutral-500">{label}</div>
      <div className="text-lg font-semibold tabular-nums">{value}</div>
    </div>
  );
}
