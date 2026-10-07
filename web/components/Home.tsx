"use client";

import Link from "next/link";
import { Status } from "@/components/ui";
import { listDetectionRuns, listInspections } from "@/lib/api";
import { useData } from "@/lib/useData";

/** 임시 첫 화면: 동기화된 데이터로 가는 링크만. (검사 목록 페이지는 나중에) */
export default function Home() {
  const inspections = useData("inspections", listInspections);
  const runs = useData("detection-runs", listDetectionRuns);
  if (inspections.error || runs.error) return <Status error={inspections.error ?? runs.error} />;
  return (
    <div className="mx-auto grid w-full max-w-4xl gap-8 p-6 md:grid-cols-2">
      <section>
        <h2 className="mb-3 text-sm font-medium text-neutral-400">검사 결과</h2>
        <ul className="space-y-2">
          {inspections.data?.map((i) => (
            <li key={i.id}>
              <Link
                href={`/inspections/${encodeURIComponent(i.id)}`}
                className="block rounded-lg border border-neutral-800 p-3 hover:border-neutral-600 hover:bg-neutral-900"
              >
                <div className="font-mono text-sm">{i.id}</div>
                <div className="text-xs text-neutral-500">
                  창문 {i.total}개 · {new Date(i.created_at).toLocaleDateString("ko-KR")}
                </div>
              </Link>
            </li>
          ))}
        </ul>
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
