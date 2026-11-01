import { SourceBadge, ToggleChip } from "@/components/ui";
import { filterStyle, LEVEL_STYLES } from "@/lib/palette";
import { confidenceOf, type InspectionResult, LEVEL_FILTERS, type LevelFilter, levelOf } from "@/lib/types";

const URGENT: LevelFilter[] = [4, 5];

export default function SummaryBar({
  result,
  source,
  visibleLevels,
  onToggleLevel,
  onSetLevels,
  multiOnly,
}: {
  result: InspectionResult;
  source?: "mock" | "run";
  visibleLevels: Set<LevelFilter>;
  onToggleLevel: (lv: LevelFilter) => void;
  onSetLevels: (lvs: Set<LevelFilter>) => void;
  multiOnly: boolean; // 켜면 창 수·단계 개수·평균 CSI를 다시점 창만으로
}) {
  const base = multiOnly ? result.windows.filter((w) => confidenceOf(w) === "multi") : result.windows;
  const total = base.length;
  const counts = Object.fromEntries(
    LEVEL_FILTERS.map((lv) => [lv, base.filter((w) => levelOf(w) === lv).length]),
  ) as Record<LevelFilter, number>;
  const csis = base.flatMap((w) => (w.csi === null ? [] : [w.csi]));
  const avg = csis.length ? csis.reduce((s, x) => s + x, 0) / csis.length : null;
  const conf = { multi: 0, single: 0, unanalyzed: 0 };
  for (const w of result.windows) conf[confidenceOf(w)]++;
  const urgentOnly = visibleLevels.size === URGENT.length && URGENT.every((lv) => visibleLevels.has(lv));
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
      <Stat label={multiOnly ? "다시점 창문" : "전체 창문"} value={total.toString()} />
      {multiOnly && <SourceBadge kind="real">다시점 기준 숫자</SourceBadge>}
      {/* 첫 화면 핵심: 먼저 청소할 창 수 */}
      <div className="flex items-stretch overflow-hidden rounded-lg border border-neutral-700" aria-label={`우선 청소 ${counts[5]}개, 청소 필요 ${counts[4]}개`}>
        {([5, 4] as const).map((lv) => (
          <div
            key={lv}
            className="px-4 py-1.5"
            style={counts[lv] ? { backgroundColor: `${LEVEL_STYLES[lv].color}26`, boxShadow: `inset 0 -3px 0 ${LEVEL_STYLES[lv].color}` } : undefined}
          >
            <div className="text-xs font-semibold" style={{ color: counts[lv] ? LEVEL_STYLES[lv].color : undefined }}>
              {lv} {LEVEL_STYLES[lv].label}
            </div>
            <div className="text-2xl font-bold tabular-nums leading-tight">{counts[lv]}개</div>
          </div>
        ))}
      </div>
      <div title="다시점 = 여러 프레임 정렬로 반사를 거른 결과 · 단일 = 한 장 분석 (신뢰도 낮음, 지도·목록에 빗금) · 미분석 = 해상도 부족 등">
        <div className="text-xs text-neutral-500">분석 신뢰도</div>
        <div className="text-sm tabular-nums">
          다시점 <span className="font-semibold">{conf.multi}</span> ·{" "}
          <span className="text-amber-300">
            단일 <span className="font-semibold">{conf.single}</span>
          </span>{" "}
          · <span className="text-neutral-400">미분석 {conf.unanalyzed}</span>
        </div>
        <div className="text-[11px] text-neutral-600 tabular-nums">평균 CSI {avg === null ? "—" : avg.toFixed(3)}</div>
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
      <div className="flex flex-wrap items-center gap-1.5">
        <button
          type="button"
          aria-pressed={urgentOnly}
          onClick={() => onSetLevels(new Set(urgentOnly ? LEVEL_FILTERS : URGENT))}
          className={`rounded-full border px-2.5 py-1 text-xs font-semibold ${urgentOnly ? "border-neutral-300 bg-neutral-200 text-neutral-900" : "border-neutral-600 text-neutral-200 hover:border-neutral-400"}`}
        >
          {urgentOnly ? "전체 보기" : "4~5단계만"}
        </button>
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
