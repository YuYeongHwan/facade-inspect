// result.json 데이터 계약 — cv/facade_cv/schema.py (pydantic)를 그대로 옮긴 zod 스키마. 바꿀 땐 둘 다 고친다.
// 좌표: 이미지 픽셀(원점 좌상단) / crop 내부 (u, v) 0~1 / 3D 미터(scale_known=false면 상대 단위), Y-up.
// row/col은 1부터, row는 v(위) 방향으로 증가. 경로는 result.json 디렉토리 기준 상대경로.
import { z } from "zod";

// 청소 우선순위 level 1(청소 불필요)~5(우선 청소). 임계값·라벨·색은 config/grading.yaml → lib/palette.ts LEVELS
export const LEVELS = [1, 2, 3, 4, 5] as const;
export const CONTAMINATION_TYPES = ["dust", "stain", "grime", "heavy"] as const;

export const CleaningLevel = z.literal(LEVELS);
const LevelKey = z.enum(["1", "2", "3", "4", "5"]);
export const ContaminationType = z.enum(CONTAMINATION_TYPES);
const Ratio = z.number().min(0).max(1);
const Vec3 = z.tuple([z.number(), z.number(), z.number()]);
const UV = z.tuple([Ratio, Ratio]);
const NonNegInt = z.number().int().min(0);

const WINDOW_ID_RE = /^(?<facade>.+)-R(?<row>\d{2,})-C(?<col>\d{2,})$/;

export const Meta = z.strictObject({
  video: z.string(),
  fps: z.number().gt(0),
  frames_used: NonNegInt,
  scale_known: z.boolean(),
  created_at: z.iso.datetime({ offset: true, local: true }),
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
  area_ratio: Ratio,
  polygon: z.array(UV).min(3),
});

export const TypeRatios = z.strictObject({ dust: Ratio, stain: Ratio, grime: Ratio, heavy: Ratio });

export const WindowImages = z.strictObject({ crop: z.string(), heatmap: z.string(), overlay: z.string() });

const pad2 = (n: number) => String(n).padStart(2, "0");

export const Window = z
  .strictObject({
    id: z.string(),
    facade_id: z.string(),
    row: z.number().int().min(1),
    col: z.number().int().min(1),
    facade_rect: z.tuple([z.number(), z.number(), z.number(), z.number()]),
    corners_3d: z.tuple([Vec3, Vec3, Vec3, Vec3]),
    cleaning_level: CleaningLevel,
    cleaning_label: z.string().min(1),
    score: z.number().min(0),
    contamination: z.array(Contamination),
    type_ratios: TypeRatios,
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
    if (r.summary.total !== r.windows.length) issue(`summary.total ${r.summary.total} != ${r.windows.length} windows`);
    for (const lv of LEVELS) {
      const actual = r.windows.filter((w) => w.cleaning_level === lv).length;
      if ((r.summary.level_counts[`${lv}`] ?? 0) !== actual) issue(`summary.level_counts.${lv} != actual ${actual}`);
    }
  });

export type CleaningLevel = z.infer<typeof CleaningLevel>;
export type ContaminationType = z.infer<typeof ContaminationType>;
export type Facade = z.infer<typeof Facade>;
export type Contamination = z.infer<typeof Contamination>;
export type Window = z.infer<typeof Window>;
export type InspectionResult = z.infer<typeof InspectionResult>;

// detections.json — cv/facade_cv/detect/postprocess.py PostResult.to_dict() + detect/run.py manifest
const Corners = z.array(z.tuple([z.number(), z.number()])).length(4); // TL, TR, BR, BL 원본 픽셀
export const DetectionFlag = z.enum(["off_grid", "odd_width", "odd_height", "fill_candidate"]);
export const RemovedReason = z.enum(["small", "not_rect", "outside_facade"]);

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
