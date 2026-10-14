"use client";

import { useCallback, useMemo, useState } from "react";
import { Status } from "@/components/ui";
import { getInspection, listInspections } from "@/lib/api";
import { LEVEL_FILTERS, type LevelFilter, levelOf } from "@/lib/types";
import { useData } from "@/lib/useData";
import BuildingViewer from "./BuildingViewer";
import SummaryBar from "./SummaryBar";
import WindowDetail from "./WindowDetail";
import WindowList from "./WindowList";

export default function InspectionView({ id }: { id: string }) {
  const { data: result, error } = useData(`inspection/${id}`, () => getInspection(id));
  const index = useData("inspections", listInspections);
  const source = index.data?.find((i) => i.id === id)?.source;
  const [visibleLevels, setVisibleLevels] = useState<Set<LevelFilter>>(() => new Set(LEVEL_FILTERS));
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const close = useCallback(() => setSelectedId(null), []);

  const visibleWindows = useMemo(
    () => result?.windows.filter((w) => visibleLevels.has(levelOf(w))) ?? [],
    [result, visibleLevels],
  );
  if (!result) return <Status error={error} />;
  const selected = result.windows.find((w) => w.id === selectedId) ?? null;

  const toggleLevel = (lv: LevelFilter) => {
    const n = new Set(visibleLevels);
    if (!n.delete(lv)) n.add(lv);
    setVisibleLevels(n);
    if (selected && !n.has(levelOf(selected))) setSelectedId(null); // 숨긴 level의 창문은 상세도 닫는다
  };

  return (
    <div className={`flex flex-col lg:h-[calc(100dvh-3.5rem)] ${selected ? "lg:pr-[420px]" : ""}`}>
      {/* 상세 drawer(fixed, 420px)가 요약 막대·3D를 가리지 않도록 그 폭만큼 비운다 */}
      <SummaryBar result={result} source={source} visibleLevels={visibleLevels} onToggleLevel={toggleLevel} />
      <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
        {/* 모바일: 3D 위, 리스트 아래 / 데스크톱: 리스트 왼쪽 */}
        <section className="order-1 h-[55dvh] min-w-0 lg:order-2 lg:h-auto lg:flex-1">
          <BuildingViewer
            inspectionId={id}
            result={result}
            visibleLevels={visibleLevels}
            selectedId={selectedId}
            hoveredId={hoveredId}
            onHover={setHoveredId}
            onSelect={setSelectedId}
          />
        </section>
        <aside className="order-2 h-[60dvh] border-t border-neutral-800 lg:order-1 lg:h-auto lg:w-80 lg:shrink-0 lg:border-r lg:border-t-0">
          <WindowList
            windows={visibleWindows}
            selectedId={selectedId}
            hoveredId={hoveredId}
            onHover={setHoveredId}
            onSelect={setSelectedId}
          />
        </aside>
      </div>
      <WindowDetail inspectionId={id} window={selected} onClose={close} />
    </div>
  );
}
