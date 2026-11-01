"use client";

import { useCallback, useMemo, useState } from "react";
import { Status } from "@/components/ui";
import { getInspection, listInspections } from "@/lib/api";
import { LEVEL_FILTERS, type LevelFilter, levelOf } from "@/lib/types";
import { useData } from "@/lib/useData";
import { PHOTO_SOURCES, photoSourceOf } from "@/lib/photoSources";
import BuildingViewer from "./BuildingViewer";
import FacadeMap from "./FacadeMap";
import SummaryBar from "./SummaryBar";
import WindowDetail from "./WindowDetail";
import WindowList from "./WindowList";
import WorkList from "./WorkList";

type Tab = "map" | "3d" | "work";
const TABS: { key: Tab; label: string }[] = [
  { key: "map", label: "외벽 지도" },
  { key: "3d", label: "3D 보기" },
  { key: "work", label: "청소 작업 목록" },
];

export default function InspectionView({ id }: { id: string }) {
  const { data: result, error } = useData(`inspection/${id}`, () => getInspection(id));
  const index = useData("inspections", listInspections);
  const source = index.data?.find((i) => i.id === id)?.source;
  const [tab, setTab] = useState<Tab>("map");
  const [visibleLevels, setVisibleLevels] = useState<Set<LevelFilter>>(() => new Set(LEVEL_FILTERS));
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  // 다시점만: 지도는 단일·미분석 창을 회색 윤곽으로, 요약 바 숫자·작업 목록·CSV는 다시점 창만
  const [multiOnly, setMultiOnly] = useState(false);
  const toggleMulti = useCallback(() => setMultiOnly((v) => !v), []);
  const close = useCallback(() => setSelectedId(null), []);

  const visibleWindows = useMemo(
    () => result?.windows.filter((w) => visibleLevels.has(levelOf(w))) ?? [],
    [result, visibleLevels],
  );
  if (!result) return <Status error={error} />;
  const selected = result.windows.find((w) => w.id === selectedId) ?? null;

  const setLevels = (n: Set<LevelFilter>) => {
    setVisibleLevels(n);
    if (selected && !n.has(levelOf(selected))) setSelectedId(null); // 숨긴 level의 창문은 상세도 닫는다
  };
  const toggleLevel = (lv: LevelFilter) => {
    const n = new Set(visibleLevels);
    if (!n.delete(lv)) n.add(lv);
    setLevels(n);
  };

  return (
    <div className={`flex flex-col lg:h-[calc(100dvh-3.5rem)] ${selected ? "lg:pr-[420px]" : ""}`}>
      {/* 상세 drawer(fixed, 420px)가 요약 막대·지도를 가리지 않도록 그 폭만큼 비운다 */}
      <SummaryBar result={result} source={source} visibleLevels={visibleLevels} onToggleLevel={toggleLevel} onSetLevels={setLevels} multiOnly={multiOnly} />
      {PHOTO_SOURCES[result.meta.video] && <PhotoCredit video={result.meta.video} facadeId={selected?.facade_id ?? null} />}
      <div className="flex gap-1 border-b border-neutral-800 px-3 pt-2" role="tablist" aria-label="보기">
        {TABS.map((t) => (
          <button
            key={t.key}
            role="tab"
            aria-selected={tab === t.key}
            onClick={() => setTab(t.key)}
            className={`-mb-px rounded-t-md border px-4 py-1.5 text-sm ${
              tab === t.key ? "border-neutral-700 border-b-neutral-950 bg-neutral-950 font-semibold text-neutral-100" : "border-transparent text-neutral-400 hover:text-neutral-200"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>
      <div className="flex min-h-[70dvh] flex-1 flex-col lg:min-h-0">
        {tab === "map" && (
          <FacadeMap
            inspectionId={id}
            result={result}
            visibleLevels={visibleLevels}
            multiOnly={multiOnly}
            onToggleMulti={toggleMulti}
            selectedId={selectedId}
            hoveredId={hoveredId}
            onHover={setHoveredId}
            onSelect={setSelectedId}
          />
        )}
        {tab === "3d" && (
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
        )}
        {tab === "work" && (
          <WorkList
            inspectionId={id}
            windows={visibleWindows}
            multiOnly={multiOnly}
            onToggleMulti={toggleMulti}
            selectedId={selectedId}
            onSelect={setSelectedId}
          />
        )}
      </div>
      <WindowDetail inspectionId={id} window={selected} csiParams={result.meta.csi_params ?? null} onClose={close} />
    </div>
  );
}

/** 사진 데모(공개 이미지) 결과의 출처 표시 [R33]. 선택한 창이 있으면 그 사진의 작성자·라이선스 */
function PhotoCredit({ video, facadeId }: { video: string; facadeId: string | null }) {
  const src = facadeId ? photoSourceOf(video, facadeId) : null;
  return (
    <p className="border-b border-neutral-800 px-4 py-1.5 text-xs text-neutral-400">
      <span className="mr-2 rounded bg-neutral-800 px-1.5 py-0.5 font-semibold text-neutral-200">공개 이미지 예시 · 출처</span>
      {src ? (
        <>
          {src.file} · {src.author} · {src.license} ·{" "}
          <a href={src.page} target="_blank" rel="noreferrer" className="underline hover:text-neutral-200">Wikimedia Commons</a>
        </>
      ) : (
        <>Wikimedia Commons {PHOTO_SOURCES[video].length}장 (사진마다 작성자·라이선스 다름) — 창을 누르면 그 사진의 출처를 표시합니다</>
      )}
    </p>
  );
}
