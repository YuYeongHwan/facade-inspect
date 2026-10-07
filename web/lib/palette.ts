import type { ContaminationType, Grade } from "./types";

export const GRADE_COLORS: Record<Grade, string> = { A: "#22c55e", B: "#eab308", C: "#f97316", D: "#ef4444" };

// 등급 색(초록·노랑·주황·빨강)과 겹치지 않게
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

export const pct = (x: number, digits = 1) => `${(x * 100).toFixed(digits)}%`;
