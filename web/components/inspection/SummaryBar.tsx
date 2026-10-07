import { SourceBadge, ToggleChip } from "@/components/ui";
import { GRADE_COLORS } from "@/lib/palette";
import { GRADES, type Grade, type InspectionResult } from "@/lib/types";

export default function SummaryBar({
  result,
  source,
  visibleGrades,
  onToggleGrade,
}: {
  result: InspectionResult;
  source?: "mock" | "run";
  visibleGrades: Set<Grade>;
  onToggleGrade: (g: Grade) => void;
}) {
  const total = result.windows.length;
  const counts = Object.fromEntries(GRADES.map((g) => [g, result.windows.filter((w) => w.grade === g).length])) as Record<
    Grade,
    number
  >;
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
      <div className="min-w-56 flex-1">
        <div className="mb-1 text-xs text-neutral-500">등급 분포</div>
        <div className="flex h-3 overflow-hidden rounded-full bg-neutral-800" role="img" aria-label="등급 분포">
          {GRADES.map((g) =>
            counts[g] ? (
              <div
                key={g}
                title={`${g}: ${counts[g]}`}
                style={{
                  width: `${(counts[g] / total) * 100}%`,
                  backgroundColor: GRADE_COLORS[g],
                  opacity: visibleGrades.has(g) ? 1 : 0.2,
                }}
              />
            ) : null,
          )}
        </div>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {GRADES.map((g) => (
          <ToggleChip key={g} on={visibleGrades.has(g)} onClick={() => onToggleGrade(g)} color={GRADE_COLORS[g]}>
            {g} <span className="tabular-nums text-neutral-400">{counts[g]}</span>
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
