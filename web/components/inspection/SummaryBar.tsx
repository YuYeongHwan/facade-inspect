import { SourceBadge, ToggleChip } from "@/components/ui";
import { LEVEL_STYLES } from "@/lib/palette";
import { type CleaningLevel, type InspectionResult, LEVELS } from "@/lib/types";

export default function SummaryBar({
  result,
  source,
  visibleLevels,
  onToggleLevel,
}: {
  result: InspectionResult;
  source?: "mock" | "run";
  visibleLevels: Set<CleaningLevel>;
  onToggleLevel: (lv: CleaningLevel) => void;
}) {
  const total = result.windows.length;
  const counts = Object.fromEntries(
    LEVELS.map((lv) => [lv, result.windows.filter((w) => w.cleaning_level === lv).length]),
  ) as Record<CleaningLevel, number>;
  const avg = total ? result.windows.reduce((s, w) => s + w.score, 0) / total : 0;
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
      <Stat label="평균 점수" value={avg.toFixed(3)} />
      <div className={`rounded-lg border px-3 py-1 ${counts[5] ? "border-red-500 bg-red-500/15" : "border-neutral-800"}`}>
        <div className={`text-xs ${counts[5] ? "font-semibold text-red-300" : "text-neutral-500"}`}>우선 청소</div>
        <div className={`text-lg font-bold tabular-nums ${counts[5] ? "text-red-400" : ""}`}>{counts[5]}개</div>
      </div>
      <div className="min-w-56 flex-1">
        <div className="mb-1 text-xs text-neutral-500">청소 우선순위 분포</div>
        <div className="flex h-3 overflow-hidden rounded-full bg-neutral-800" role="img" aria-label="청소 우선순위 분포">
          {LEVELS.map((lv) =>
            counts[lv] ? (
              <div
                key={lv}
                title={`${lv} ${LEVEL_STYLES[lv].label}: ${counts[lv]}`}
                style={{
                  width: `${(counts[lv] / total) * 100}%`,
                  backgroundColor: LEVEL_STYLES[lv].color,
                  opacity: visibleLevels.has(lv) ? 1 : 0.2,
                }}
              />
            ) : null,
          )}
        </div>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {LEVELS.map((lv) => (
          <ToggleChip key={lv} on={visibleLevels.has(lv)} onClick={() => onToggleLevel(lv)} color={LEVEL_STYLES[lv].color}>
            {lv} {LEVEL_STYLES[lv].label} <span className="tabular-nums text-neutral-400">{counts[lv]}</span>
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
