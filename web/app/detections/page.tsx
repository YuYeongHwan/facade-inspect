import { Suspense } from "react";
import DetectionsView from "@/components/detections/DetectionsView";
import { Status } from "@/components/ui";

export const metadata = { title: "검출 뷰어 · Facade Inspect" };

export default function DetectionsPage() {
  // DetectionsView가 useSearchParams(?run=&frame=)를 쓰므로 Suspense 경계가 필요하다
  return (
    <Suspense fallback={<Status />}>
      <DetectionsView />
    </Suspense>
  );
}
