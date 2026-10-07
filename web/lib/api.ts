// 데이터 로딩은 전부 여기서. 지금은 scripts/sync-data.mjs가 만든 public/data 정적 파일을 읽는다.
// 백엔드가 생기면 NEXT_PUBLIC_DATA_BASE를 API URL로 바꾸고, 경로가 다르면 아래 *Url 함수만 고친다.
import type { z } from "zod";
import {
  DetectionRunIndex,
  DetectionsFile,
  InspectionIndex,
  InspectionResult,
} from "./types";

const BASE = (process.env.NEXT_PUBLIC_DATA_BASE ?? "/data").replace(/\/$/, "");

async function fetchJson<T extends z.ZodType>(url: string, schema: T): Promise<z.infer<T>> {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}: ${url}`);
  const parsed = schema.safeParse(await res.json());
  if (!parsed.success) {
    throw new Error(`invalid data from ${url}:\n${parsed.error.issues.map((i) => `${i.path.join(".")}: ${i.message}`).join("\n")}`);
  }
  return parsed.data;
}

const encodePath = (id: string) => id.split("/").map(encodeURIComponent).join("/");

// 검사 결과 (result.json)
export const inspectionAssetUrl = (id: string, rel: string) => `${BASE}/inspections/${encodePath(id)}/${rel}`;
export const listInspections = () => fetchJson(`${BASE}/inspections/index.json`, InspectionIndex);
export const getInspection = (id: string) => fetchJson(inspectionAssetUrl(id, "result.json"), InspectionResult);

// 실제 영상 검출 결과 (detect/detections.json). runId 예: t1_check/20261007_b01_ground_01
export const listDetectionRuns = () => fetchJson(`${BASE}/detections/index.json`, DetectionRunIndex);
export const getDetections = (runId: string) =>
  fetchJson(`${BASE}/detections/${encodePath(runId)}/detections.json`, DetectionsFile);
export const frameUrl = (runId: string, image: string) =>
  `${BASE}/detections/${encodePath(runId)}/frames/${encodeURIComponent(image)}`;
export const rectifiedUrl = (runId: string, image: string) =>
  `${BASE}/detections/${encodePath(runId)}/rectified/${encodeURIComponent(image.replace(/\.[^.]+$/, ""))}.jpg`;
