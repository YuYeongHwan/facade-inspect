import type { ReactNode } from "react";
import { LEVEL_STYLES } from "@/lib/palette";
import type { CleaningLevel } from "@/lib/types";

/** 청소 우선순위 배지 "5 우선 청소". 5(우선 청소)만 채운 빨강 + 글로우로 가장 눈에 띄게, 나머지는 옅은 틴트. */
export function LevelBadge({ level, size = "sm" }: { level: CleaningLevel; size?: "sm" | "lg" }) {
  const { label, color } = LEVEL_STYLES[level];
  const urgent = level === 5;
  const cls = size === "lg" ? "gap-1.5 px-2.5 py-1 text-sm" : "gap-1 px-1.5 py-0.5 text-[11px]";
  return (
    <span
      className={`inline-flex shrink-0 items-center whitespace-nowrap rounded-md border font-semibold ${cls} ${
        urgent ? "text-white shadow-[0_0_10px_rgba(239,68,68,0.6)]" : ""
      }`}
      style={urgent ? { backgroundColor: color, borderColor: color } : { color, borderColor: `${color}66`, backgroundColor: `${color}1a` }}
      aria-label={`청소 우선순위 ${level}단계 ${label}`}
    >
      <span className="tabular-nums">{level}</span>
      {label}
    </span>
  );
}

/** 켜고 끄는 칩. color가 있으면 왼쪽에 색 점. */
export function ToggleChip({
  on,
  onClick,
  color,
  children,
}: {
  on: boolean;
  onClick: () => void;
  color?: string;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      aria-pressed={on}
      onClick={onClick}
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition-colors ${
        on ? "border-neutral-500 bg-neutral-800 text-neutral-100" : "border-neutral-800 text-neutral-500 line-through"
      }`}
    >
      {color && <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: on ? color : "transparent", boxShadow: `inset 0 0 0 1.5px ${color}` }} />}
      {children}
    </button>
  );
}

/** 데이터 출처 배지 — 목업(노랑)과 실제(초록)를 늘 구분해 보이게. */
export function SourceBadge({ kind, children }: { kind: "mock" | "real"; children: ReactNode }) {
  const cls =
    kind === "mock"
      ? "border-amber-700 bg-amber-950/60 text-amber-300"
      : "border-emerald-800 bg-emerald-950/60 text-emerald-300";
  return (
    <span className={`inline-flex shrink-0 items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium ${cls}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${kind === "mock" ? "bg-amber-400" : "bg-emerald-400"}`} />
      {children}
    </span>
  );
}

export function Status({ error, children }: { error?: string; children?: ReactNode }) {
  return (
    <div className="flex flex-1 items-center justify-center p-8 text-sm">
      {error ? (
        <pre className="max-w-2xl whitespace-pre-wrap rounded-lg border border-red-900 bg-red-950/40 p-4 text-red-300">
          {error}
          {"\n\n"}데이터가 없다면 web/에서 `pnpm sync-data`를 먼저 실행하세요.
        </pre>
      ) : (
        <span className="text-neutral-500">{children ?? "불러오는 중…"}</span>
      )}
    </div>
  );
}
