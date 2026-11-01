"use client";

import Link from "next/link";
import { SourceBadge, Status } from "@/components/ui";
import { listDetectionRuns, listInspections } from "@/lib/api";
import type { InspectionIndex } from "@/lib/types";
import { useData } from "@/lib/useData";

/** 임시 첫 화면: 동기화된 데이터로 가는 링크만. (검사 목록 페이지는 나중에) */
export default function Home() {
  const inspections = useData("inspections", listInspections);
  const runs = useData("detection-runs", listDetectionRuns);
  if (inspections.error || runs.error) return <Status error={inspections.error ?? runs.error} />;
  // 실제 결과(runs/) 최신순 먼저, 목업(fixtures/)은 "예시"로 따로
  const all = (inspections.data ?? []).toSorted((a, b) => b.created_at.localeCompare(a.created_at));
  const real = all.filter((i) => i.source === "run");
  const examples = all.filter((i) => i.source === "mock");
  return (
    <div className="mx-auto grid w-full max-w-4xl gap-8 p-6 md:grid-cols-2">
      <section className="space-y-6">
        <InspectionLinks title="검사 결과" items={real} />
        {examples.length > 0 && <InspectionLinks title="예시 데이터 (목업)" items={examples} example />}
      </section>
      <section>
        <h2 className="mb-3 text-sm font-medium text-neutral-400">실제 영상 검출</h2>
        <ul className="space-y-2">
          {runs.data?.map((r) => (
            <li key={r.id}>
              <Link
                href={`/detections?run=${encodeURIComponent(r.id)}`}
                className="block rounded-lg border border-neutral-800 p-3 hover:border-neutral-600 hover:bg-neutral-900"
              >
                <div className="font-mono text-sm">{r.id}</div>
                <div className="text-xs text-neutral-500">프레임 {r.images.length}장</div>
              </Link>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}

function InspectionLinks({ title, items, example = false }: { title: string; items: InspectionIndex; example?: boolean }) {
  return (
    <div>
      <h2 className="mb-3 text-sm font-medium text-neutral-400">{title}</h2>
      <ul className="space-y-2">
        {items.map((i) => (
          <li key={i.id}>
            <Link
              href={`/inspections/${encodeURIComponent(i.id)}`}
              className={`block rounded-lg border p-3 hover:border-neutral-600 hover:bg-neutral-900 ${example ? "border-neutral-900 opacity-80" : "border-neutral-800"}`}
            >
              <div className="flex items-center gap-2">
                <span className="font-mono text-sm">{i.id}</span>
                {example && <SourceBadge kind="mock">예시</SourceBadge>}
              </div>
              <div className="text-xs text-neutral-500">
                창문 {i.total}개 · {new Date(i.created_at).toLocaleDateString("ko-KR")}
              </div>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
