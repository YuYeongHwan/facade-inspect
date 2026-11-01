"use client";

import { useMemo } from "react";
import { LevelBadge, ToggleChip } from "@/components/ui";
import { TYPE_LABELS } from "@/lib/palette";
import { byPriority, CONFIDENCE_LABELS, confidenceOf, mainType, rawCoveragePct, type Window } from "@/lib/types";

/** 청소 작업 목록 — 우선순위 순 표 + CSV("청소 작업 지시서"). 화면과 CSV는 같은 필터 (level 칩 + 신뢰도). */
export default function WorkList({
  inspectionId,
  windows,
  multiOnly,
  onToggleMulti,
  selectedId,
  onSelect,
}: {
  inspectionId: string;
  windows: Window[]; // level 필터 적용된 창
  multiOnly: boolean; // 다시점만 (지도·요약 바와 공유)
  onToggleMulti: () => void;
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const rows = useMemo(
    () => windows.filter((w) => !multiOnly || confidenceOf(w) === "multi").toSorted(byPriority),
    [windows, multiOnly],
  );
  const nMulti = windows.filter((w) => confidenceOf(w) === "multi").length;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex flex-wrap items-center gap-3 border-b border-neutral-800 px-4 py-2">
        <ToggleChip on={multiOnly} onClick={onToggleMulti}>
          신뢰도 높은 것만 (다시점) <span className="tabular-nums text-neutral-400">{nMulti}</span>
        </ToggleChip>
        <span className="text-xs text-neutral-500">
          {rows.length}개 · 위 단계 칩과 이 토글이 CSV에도 적용됩니다
        </span>
        <span className="flex-1" />
        <button
          type="button"
          onClick={() => downloadCsv(inspectionId, rows)}
          disabled={!rows.length}
          className="rounded-md border border-neutral-600 bg-neutral-100 px-3 py-1.5 text-sm font-semibold text-neutral-900 hover:bg-white disabled:opacity-40"
        >
          CSV 내보내기 · 청소 작업 지시서
        </button>
      </div>
      <div className="min-h-0 flex-1 overflow-auto">
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-neutral-950 text-xs text-neutral-500">
            <tr className="border-b border-neutral-800 text-left">
              <th className="px-4 py-2 text-right font-normal">순위</th>
              <th className="px-2 py-2 font-normal">창문 ID</th>
              <th className="px-2 py-2 text-right font-normal">외벽</th>
              <th className="px-2 py-2 text-right font-normal">층</th>
              <th className="px-2 py-2 text-right font-normal">열</th>
              <th className="px-2 py-2 font-normal">단계</th>
              <th className="px-2 py-2 text-right font-normal">오염 면적</th>
              <th className="px-2 py-2 font-normal">주요 유형</th>
              <th className="px-2 py-2 font-normal">신뢰도</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((w, i) => {
              const raw = rawCoveragePct(w);
              const t = mainType(w);
              const conf = confidenceOf(w);
              return (
                <tr
                  key={w.id}
                  onClick={() => onSelect(w.id)}
                  className={`cursor-pointer border-b border-neutral-900 hover:bg-neutral-900 ${w.id === selectedId ? "bg-neutral-800/70" : ""}`}
                >
                  <td className="px-4 py-1.5 text-right tabular-nums text-neutral-400">{i + 1}</td>
                  <td className="px-2 py-1.5 font-mono">
                    <button type="button" className="hover:underline" onClick={() => onSelect(w.id)}>
                      {w.id}
                    </button>
                  </td>
                  <td className="px-2 py-1.5 text-right">{w.facade_id}</td>
                  <td className="px-2 py-1.5 text-right tabular-nums">{w.row}</td>
                  <td className="px-2 py-1.5 text-right tabular-nums">{w.col}</td>
                  <td className="px-2 py-1.5">
                    <LevelBadge level={w.cleaning_level} lowConfidence={conf === "single"} />
                  </td>
                  <td className="px-2 py-1.5 text-right tabular-nums">{raw === null ? "—" : `${raw.toFixed(1)}%`}</td>
                  <td className="px-2 py-1.5">{t ? TYPE_LABELS[t] : "—"}</td>
                  <td className={`px-2 py-1.5 text-xs ${conf === "single" ? "text-amber-300" : conf === "multi" ? "text-neutral-200" : "text-neutral-500"}`}>
                    {CONFIDENCE_LABELS[conf]}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {!rows.length && <p className="p-6 text-center text-sm text-neutral-500">조건에 맞는 창문이 없습니다.</p>}
      </div>
    </div>
  );
}

const csvCell = (v: string | number) => (/[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));

function downloadCsv(inspectionId: string, rows: Window[]) {
  const head = ["순위", "창문ID", "외벽", "층", "열", "단계", "라벨", "오염면적%", "주요유형", "신뢰도", "CSI"];
  const body = rows.map((w, i) => {
    const raw = rawCoveragePct(w);
    const t = mainType(w);
    return [i + 1, w.id, w.facade_id, w.row, w.col, w.cleaning_level ?? "", w.cleaning_label, raw === null ? "" : raw.toFixed(1),
      t ? TYPE_LABELS[t] : "", CONFIDENCE_LABELS[confidenceOf(w)], w.csi ?? ""];
  });
  // BOM: 엑셀에서 한글이 깨지지 않게
  const text = "﻿" + [head, ...body].map((r) => r.map(csvCell).join(",")).join("\r\n") + "\r\n";
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "text/csv;charset=utf-8" }));
  a.download = `청소작업지시서_${inspectionId}.csv`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
