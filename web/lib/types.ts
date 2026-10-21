// result.json 데이터 계약 — cv/facade_cv/schema.py (pydantic)를 그대로 옮긴 zod 스키마. 바꿀 땐 둘 다 고친다.
// 좌표: 이미지 픽셀(원점 좌상단) / crop 내부 (u, v) 0~1 / 3D 미터(scale_known=false면 상대 단위), Y-up.
// row/col은 1부터, row는 v(위) 방향으로 증가. 경로는 result.json 디렉토리 기준 상대경로.
import { z } from "zod";

// 청소 우선순위 level 1(청소 불필요)~5(우선 청소). 임계값·라벨·색은 config/grading.yaml → lib/palette.ts LEVELS
export const LEVELS = [1, 2, 3, 4, 5] as const;
export const CONTAMINATION_TYPES = ["dust", "stain", "grime", "heavy"] as const;

export const CleaningLevel = z.literal(LEVELS);
const LevelKey = z.enum(["1", "2", "3", "4", "5", "unanalyzed"]);
// 오염 분석(T7) 전 창문: cleaning_level/csi/type_ratios/glass_ratio/occluded_ratio/images.heatmap·overlay = null
// CSI(창문 오염 심각도 지수, 0~1) = Σ area_ratio × severity × k_t / (k_s_max × k_t_max) — docs/scoring.md
export const UNANALYZED_LABEL = "미분석";
/** 필터·범례·분포 단위: level 1~5 + 미분석 */
export type LevelFilter = z.infer<typeof CleaningLevel> | "unanalyzed";
export const LEVEL_FILTERS: readonly LevelFilter[] = [...LEVELS, "unanalyzed"];
export const levelOf = (w: { cleaning_level: z.infer<typeof CleaningLevel> | null }): LevelFilter => w.cleaning_level ?? "unanalyzed";
export const ContaminationType = z.enum(CONTAMINATION_TYPES);
const Ratio = z.number().min(0).max(1);
const Vec3 = z.tuple([z.number(), z.number(), z.number()]);
const UV = z.tuple([Ratio, Ratio]);
const NonNegInt = z.number().int().min(0);
const CSI_TOL = 1e-3; // csi와 Σ 영역 기여도 허용 차 (반올림)

const WINDOW_ID_RE = /^(?<facade>.+)-R(?<row>\d{2,})-C(?<col>\d{2,})$/;

// 3D 복원 결과 (facade_cv.recon3d). degraded = SfM/평면 실패로 가상 평면 1개 + 2D 격자
export const ReconStatus = z.strictObject({
  mode: z.enum(["3d", "degraded"]),
  num_facades: NonNegInt,
  registered_ratio: Ratio.nullable(),
  warnings: z.array(z.string()),
});

// CSI 계산에 쓴 값 (config/grading.yaml 스냅샷) — 상세 패널 계산 내역
export const CsiParams = z.strictObject({
  weights: z.record(ContaminationType, z.number()), // k_t
  severity_max: z.number().int().min(1), // k_s_max
  severity_delta_e: z.array(z.number()), // k_s 2, 3의 ΔE*ab 하한
  frame_margin: Ratio,
  level_min_csi: z.array(z.number()).length(5),
  status: z.string(),
});

export const Meta = z.strictObject({
  video: z.string(),
  fps: z.number().gt(0),
  frames_used: NonNegInt,
  scale_known: z.boolean(),
  created_at: z.iso.datetime({ offset: true, local: true }),
  recon_status: ReconStatus.nullish(),
  csi_params: CsiParams.nullish(), // 오염 분석 결과가 있으면 필수
});

export const Facade = z.strictObject({
  id: z.string(),
  normal: Vec3,
  origin: Vec3,
  u_axis: Vec3,
  v_axis: Vec3,
  width: z.number().gt(0),
  height: z.number().gt(0),
  texture: z.string(),
});

export const Building = z.strictObject({
  mesh: z.string(),
  facades: z.array(Facade).min(1),
});

export const Contamination = z.strictObject({
  type: ContaminationType,
  area_ratio: Ratio, // A_j / A_g (유리 안 면적만)
  severity: z.number().int().min(1).max(3), // k_s
  delta_e: z.number().min(0).nullable(), // 깨끗한 유리 대비 ΔE*ab
  polygon: z.array(UV).min(3),
});

export const TypeRatios = z.strictObject({ dust: Ratio, stain: Ratio, grime: Ratio, heavy: Ratio });

export const WindowImages = z.strictObject({ crop: z.string(), heatmap: z.string().nullable(), overlay: z.string().nullable() });

const pad2 = (n: number) => String(n).padStart(2, "0");

export const Window = z
  .strictObject({
    id: z.string(),
    facade_id: z.string(),
    row: z.number().int().min(1),
    col: z.number().int().min(1),
    facade_rect: z.tuple([z.number(), z.number(), z.number(), z.number()]),
    corners_3d: z.tuple([Vec3, Vec3, Vec3, Vec3]),
    cleaning_level: CleaningLevel.nullable(),
    cleaning_label: z.string().min(1),
    csi: Ratio.nullable(),
    glass_ratio: Ratio.nullable(), // A_g / crop 면적
    occluded_ratio: Ratio.nullable(), // 나무·전선·기둥에 가린 면적 / crop 면적
    contamination: z.array(Contamination),
    type_ratios: TypeRatios.nullable(),
    images: WindowImages,
    num_views: NonNegInt,
  })
  .superRefine((w, ctx) => {
    const m = WINDOW_ID_RE.exec(w.id)?.groups;
    if (!m || m.facade !== w.facade_id || Number(m.row) !== w.row || Number(m.col) !== w.col) {
      ctx.addIssue({ code: "custom", message: `id ${w.id} != ${w.facade_id}-R${pad2(w.row)}-C${pad2(w.col)}` });
    }
    const [u0, v0, u1, v1] = w.facade_rect;
    if (!(u0 < u1 && v0 < v1)) {
      ctx.addIssue({ code: "custom", message: `facade_rect must satisfy u0<u1, v0<v1: ${w.facade_rect}` });
    }
    const analysis = [w.csi, w.type_ratios, w.glass_ratio, w.occluded_ratio];
    if (w.cleaning_level === null) {
      if (w.cleaning_label !== UNANALYZED_LABEL || analysis.some((v) => v !== null)) {
        ctx.addIssue({
          code: "custom",
          message: `window ${w.id}: unanalyzed needs label ${UNANALYZED_LABEL}, csi/type_ratios/glass_ratio/occluded_ratio null`,
        });
      }
    } else if (analysis.some((v) => v === null)) {
      ctx.addIssue({ code: "custom", message: `window ${w.id}: analyzed window needs csi, type_ratios, glass_ratio, occluded_ratio` });
    }
  });

export const Summary = z.strictObject({
  total: NonNegInt,
  level_counts: z.partialRecord(LevelKey, z.number().int()),
});

export const InspectionResult = z
  .strictObject({
    inspection_id: z.string(),
    meta: Meta,
    building: Building,
    windows: z.array(Window),
    summary: Summary,
  })
  .superRefine((r, ctx) => {
    const issue = (message: string) => ctx.addIssue({ code: "custom", message });
    const facadeIds = r.building.facades.map((f) => f.id);
    if (new Set(facadeIds).size !== facadeIds.length) issue("duplicate facade id");
    const windowIds = r.windows.map((w) => w.id);
    if (new Set(windowIds).size !== windowIds.length) issue("duplicate window id");
    for (const w of r.windows) {
      if (!facadeIds.includes(w.facade_id)) issue(`window ${w.id}: unknown facade_id ${w.facade_id}`);
    }
    const p = r.meta.csi_params;
    const analyzed = r.windows.filter((w) => w.csi !== null);
    if (analyzed.length && !p) issue("meta.csi_params required when windows are analyzed");
    if (p) {
      for (const w of analyzed) {
        const sum = csiTerms(w.contamination, p).reduce((s, t) => s + t, 0);
        if (Math.abs(sum - (w.csi ?? 0)) > CSI_TOL) issue(`window ${w.id}: csi ${w.csi} != Σ contamination terms ${sum.toFixed(5)}`);
        if (levelOfCsi(w.csi ?? 0, p) !== w.cleaning_level) issue(`window ${w.id}: cleaning_level != level of csi ${w.csi}`);
      }
    }
    if (r.summary.total !== r.windows.length) issue(`summary.total ${r.summary.total} != ${r.windows.length} windows`);
    for (const lv of LEVEL_FILTERS) {
      const actual = r.windows.filter((w) => levelOf(w) === lv).length;
      if ((r.summary.level_counts[`${lv}`] ?? 0) !== actual) issue(`summary.level_counts.${lv} != actual ${actual}`);
    }
  });

/** 영역별 CSI 항 = area_ratio × k_s × k_t / (k_s_max × k_t_max) (schema.py csi_of와 같은 식) */
export function csiTerms(regions: z.infer<typeof Contamination>[], p: z.infer<typeof CsiParams>): number[] {
  const denom = p.severity_max * Math.max(...Object.values(p.weights));
  return regions.map((c) => (c.area_ratio * c.severity * (p.weights[c.type] ?? 0)) / denom);
}
export const levelOfCsi = (csi: number, p: z.infer<typeof CsiParams>) =>
  Math.max(...p.level_min_csi.flatMap((m, i) => (csi >= m ? [i + 1] : [])));

export type CleaningLevel = z.infer<typeof CleaningLevel>;
export type CsiParams = z.infer<typeof CsiParams>;
export type ContaminationType = z.infer<typeof ContaminationType>;
export type Facade = z.infer<typeof Facade>;
export type Contamination = z.infer<typeof Contamination>;
export type Window = z.infer<typeof Window>;
export type InspectionResult = z.infer<typeof InspectionResult>;

// detections.json — cv/facade_cv/detect/postprocess.py PostResult.to_dict() + detect/run.py manifest
const Corners = z.array(z.tuple([z.number(), z.number()])).length(4); // TL, TR, BR, BL 원본 픽셀
export const DetectionFlag = z.enum(["off_grid", "odd_width", "odd_height", "fill_candidate"]);
export const RemovedReason = z.enum([
  "small",
  "not_rect",
  "outside_facade",
  "on_vegetation", // v2 장애물 위 오검출 (detect.postprocess obstacles)
  "on_wire",
  "on_pole",
  "on_door",
]);

const DetectedWindow = z.object({
  corners: Corners,
  conf: z.number(),
  row: z.number().int().nullable(),
  col: z.number().int().nullable(),
  flags: z.array(DetectionFlag),
});

export const FrameDetections = z.object({
  image: z.string(),
  counts: z.object({
    raw: NonNegInt,
    final: NonNegInt,
    removed: NonNegInt,
    flagged: NonNegInt,
    fill_candidates: NonNegInt,
    empty_cells: NonNegInt,
  }),
  rectified: z.boolean(),
  focal_px: z.number().nullable(),
  facade_filter: z.boolean(),
  warnings: z.array(z.string()),
  windows: z.array(DetectedWindow),
  removed: z.array(z.object({ corners: Corners, conf: z.number(), reason: RemovedReason })),
  fill_candidates: z.array(DetectedWindow),
  empty_cells: z.array(z.tuple([z.number().int(), z.number().int()])),
});

export const DetectionsFile = z.object({ weights: z.string(), images: z.array(FrameDetections) });

export const DetectionRunIndex = z.array(
  z.object({
    id: z.string(),
    images: z.array(z.object({ image: z.string(), raw: NonNegInt, final: NonNegInt })),
  }),
);
// source: sync-data.mjs가 기록하는 출처 — fixtures/ = "mock" (목업), runs/ = "run" (실제 파이프라인)
export const InspectionIndex = z.array(
  z.object({ id: z.string(), total: NonNegInt, created_at: z.string(), source: z.enum(["mock", "run"]) }),
);

export type DetectedWindow = z.infer<typeof DetectedWindow>;
export type FrameDetections = z.infer<typeof FrameDetections>;
export type DetectionsFile = z.infer<typeof DetectionsFile>;
export type DetectionRunIndex = z.infer<typeof DetectionRunIndex>;
export type InspectionIndex = z.infer<typeof InspectionIndex>;
