// fixtures/ 와 runs/ 의 결과 파일을 web/public/data/ 로 복사한다 (백엔드가 생기기 전까지 정적 파일로 서빙).
//
//   node scripts/sync-data.mjs        (pnpm sync-data)
//
// public/data/
//   inspections/index.json                         [{ id, total, created_at, source }]  source: fixtures/ = "mock", runs/ = "run"
//   inspections/<inspection_id>/result.json + result.json이 참조하는 파일 (mesh, texture, windows/*)
//   detections/index.json                          [{ id, images: [{ image, raw, final }] }]
//   detections/<run>/detections.json, frames/<image>, rectified/<stem>.jpg
//     <run> = runs/ 기준 detect/ 폴더의 부모 경로 (예: t1_check/20261007_b01_ground_01)
import { cpSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const OUT = join(ROOT, "web/public/data");

const readJson = (p) => JSON.parse(readFileSync(p, "utf-8"));
const writeJson = (p, v) => writeFileSync(p, JSON.stringify(v, null, 2));

function copy(src, dst) {
  if (!existsSync(src)) {
    console.warn(`  ! missing ${relative(ROOT, src)}`);
    return false;
  }
  mkdirSync(dirname(dst), { recursive: true });
  cpSync(src, dst);
  return true;
}

function* walk(dir, depth = 4) {
  if (!existsSync(dir) || depth < 0) return;
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) yield* walk(p, depth - 1);
    else yield p;
  }
}

function syncInspection(resultPath, source) {
  const src = dirname(resultPath);
  const r = readJson(resultPath);
  const dst = join(OUT, "inspections", r.inspection_id);
  const files = [
    r.building.mesh,
    ...r.building.facades.map((f) => f.texture),
    ...r.windows.flatMap((w) => Object.values(w.images).filter(Boolean)), // 미분석 창문은 heatmap·overlay = null
  ];
  copy(resultPath, join(dst, "result.json"));
  for (const f of new Set(files)) copy(join(src, f), join(dst, f));
  console.log(`inspection ${r.inspection_id}: ${r.windows.length} windows ← ${relative(ROOT, src)}`);
  return { id: r.inspection_id, total: r.summary.total, created_at: r.meta.created_at, source };
}

function syncDetections(detPath) {
  const runDir = dirname(dirname(detPath));
  const id = relative(join(ROOT, "runs"), runDir).split(sep).join("/");
  const dst = join(OUT, "detections", ...id.split("/"));
  const det = readJson(detPath);
  copy(detPath, join(dst, "detections.json"));
  for (const { image } of det.images) {
    copy(join(runDir, "frames", image), join(dst, "frames", image));
    const stem = image.replace(/\.[^.]+$/, "");
    const rect = join(runDir, "debug/detect", `${stem}_rectified.jpg`);
    if (existsSync(rect)) copy(rect, join(dst, "rectified", `${stem}.jpg`));
  }
  console.log(`detections ${id}: ${det.images.length} frames`);
  return { id, images: det.images.map((i) => ({ image: i.image, raw: i.counts.raw, final: i.counts.final })) };
}

rmSync(OUT, { recursive: true, force: true });
mkdirSync(join(OUT, "inspections"), { recursive: true });
mkdirSync(join(OUT, "detections"), { recursive: true });

const FIXTURES = join(ROOT, "fixtures");
const results = [join(FIXTURES, "result.json"), ...walk(join(ROOT, "runs"))].filter(
  (p) => p.endsWith(`${sep}result.json`) && existsSync(p),
);
writeJson(
  join(OUT, "inspections/index.json"),
  results.map((p) => syncInspection(p, p.startsWith(FIXTURES + sep) ? "mock" : "run")),
);

const dets = [...walk(join(ROOT, "runs"))].filter((p) => p.endsWith(`${sep}detect${sep}detections.json`)).sort();
writeJson(join(OUT, "detections/index.json"), dets.map(syncDetections));
console.log(`→ ${relative(ROOT, OUT)}`);
