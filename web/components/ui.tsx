import type { ReactNode } from "react";
import { GRADE_COLORS } from "@/lib/palette";
import type { Grade } from "@/lib/types";

export function GradeBadge({ grade, size = "sm" }: { grade: Grade; size?: "sm" | "lg" }) {
  const cls = size === "lg" ? "h-9 w-9 text-lg" : "h-6 w-6 text-xs";
  return (
    <span
      className={`inline-flex shrink-0 items-center justify-center rounded-md font-bold text-neutral-950 ${cls}`}
      style={{ backgroundColor: GRADE_COLORS[grade] }}
      aria-label={`등급 ${grade}`}
    >
      {grade}
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
