import type { CleaningLevel, ContaminationType, LevelFilter } from "./types";

// config/grading.yaml levels와 같아야 한다 (cv/tests/test_schema.py가 검사). min = CSI 하한 (보정 전 초기값)
export const LEVEL_STYLES: Record<CleaningLevel, { label: string; min: number; color: string }> = {
  1: { label: "청소 불필요", min: 0.0, color: "#22c55e" },
  2: { label: "경과 관찰", min: 0.009, color: "#84cc16" },
  3: { label: "가급적 청소", min: 0.022, color: "#eab308" },
  4: { label: "청소 필요", min: 0.044, color: "#f97316" },
  5: { label: "우선 청소", min: 0.089, color: "#ef4444" },
};
// 오염 분석(T7) 전 창문 — 회색
export const UNANALYZED_STYLE = { label: "미분석", color: "#6b7280" };
export const filterStyle = (lv: LevelFilter) => (lv === "unanalyzed" ? UNANALYZED_STYLE : LEVEL_STYLES[lv]);

// level 색(초록·연두·노랑·주황·빨강)과 겹치지 않게
export const TYPE_COLORS: Record<ContaminationType, string> = {
  dust: "#cbd5e1",
  stain: "#38bdf8",
  grime: "#a78bfa",
  heavy: "#ec4899",
};
export const TYPE_LABELS: Record<ContaminationType, string> = {
  dust: "먼지",
  stain: "얼룩",
  grime: "때",
  heavy: "심한 오염",
};

export const SEVERITY_LABELS: Record<number, string> = { 1: "옅음", 2: "뚜렷함", 3: "불투명" };

export const pct = (x: number, digits = 1) => `${(x * 100).toFixed(digits)}%`;
