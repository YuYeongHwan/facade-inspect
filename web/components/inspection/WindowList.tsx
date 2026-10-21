"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { LevelBadge } from "@/components/ui";
import type { Window } from "@/lib/types";

type SortKey = "id" | "row" | "cleaning_level" | "csi";
const COLUMNS: { key: SortKey; label: string; className: string }[] = [
  { key: "id", label: "ID", className: "flex-1 text-left" },
  { key: "row", label: "층", className: "w-10 text-right" },
  { key: "cleaning_level", label: "우선순위", className: "w-24 text-left" },
  { key: "csi", label: "CSI", className: "w-14 text-right" },
];

export default function WindowList({
  windows,
  selectedId,
  hoveredId,
  onHover,
  onSelect,
}: {
  windows: Window[];
  selectedId: string | null;
  hoveredId: string | null;
  onHover: (id: string | null) => void;
  onSelect: (id: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: "cleaning_level", desc: true });
  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    const sign = sort.desc ? -1 : 1;
    return windows
      .filter((w) => !q || w.id.toLowerCase().includes(q))
      .toSorted((a, b) => {
        const x = a[sort.key];
        const y = b[sort.key];
        // 미분석(null)은 정렬 방향과 관계없이 맨 뒤
        if ((x === null) !== (y === null)) return x === null ? 1 : -1;
        const c = typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y));
        // 같은 level 안에서는 CSI 높은 순
        const byScore = sort.key === "cleaning_level" ? (b.csi ?? 0) - (a.csi ?? 0) : 0;
        return sign * c || byScore || a.id.localeCompare(b.id);
      });
  }, [windows, query, sort]);

  const itemRefs = useRef(new Map<string, HTMLButtonElement>());
  useEffect(() => {
    if (selectedId) itemRefs.current.get(selectedId)?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [selectedId]);

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-neutral-800 p-3">
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="창문 ID 검색 (예: F1-R02)"
          className="w-full rounded-md border border-neutral-700 bg-neutral-900 px-3 py-1.5 text-sm placeholder:text-neutral-600 focus:border-neutral-500 focus:outline-none"
        />
      </div>
      <div className="flex gap-2 border-b border-neutral-800 px-3 py-2 text-xs text-neutral-500">
        {COLUMNS.map((c) => (
          <button
            key={c.key}
            type="button"
            className={`${c.className} hover:text-neutral-200 ${sort.key === c.key ? "text-neutral-200" : ""}`}
            onClick={() => setSort((s) => ({ key: c.key, desc: s.key === c.key ? !s.desc : c.key === "csi" || c.key === "cleaning_level" }))}
          >
            {c.label}
            {sort.key === c.key && (sort.desc ? " ↓" : " ↑")}
          </button>
        ))}
      </div>
      <ul className="flex-1 overflow-y-auto" onMouseLeave={() => onHover(null)}>
        {rows.map((w) => (
          <li key={w.id}>
            <button
              ref={(el) => {
                if (el) itemRefs.current.set(w.id, el);
                else itemRefs.current.delete(w.id);
              }}
              type="button"
              onClick={() => onSelect(w.id)}
              onMouseEnter={() => onHover(w.id)}
              className={`flex w-full items-center gap-2 border-l-2 px-3 py-1.5 text-left text-sm ${
                w.id === selectedId
                  ? "border-neutral-100 bg-neutral-800"
                  : w.id === hoveredId
                    ? "border-transparent bg-neutral-900"
                    : "border-transparent"
              }`}
            >
              <span className="flex-1 truncate font-mono text-xs">{w.id}</span>
              <span className="w-10 text-right tabular-nums text-neutral-400">{w.row}층</span>
              <span className="flex w-24">
                <LevelBadge level={w.cleaning_level} />
              </span>
              <span className="w-14 text-right tabular-nums">{w.csi === null ? "—" : w.csi.toFixed(3)}</span>
            </button>
          </li>
        ))}
        {!rows.length && <li className="p-4 text-center text-sm text-neutral-500">표시할 창문이 없습니다</li>}
      </ul>
    </div>
  );
}
