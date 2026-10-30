"""runs/<inspection_id>/result.json 데이터 계약 (CLAUDE.md 참조).

좌표 규칙
- 이미지 좌표: 픽셀, 원점 좌상단
- 창문 crop 내부 좌표 (u, v): 0~1 정규화
- 3D 좌표: 미터 (meta.scale_known=false면 상대 단위), Y-up
- facade_rect: 외벽 로컬 좌표 [u0, v0, u1, v1] (origin 기준, u_axis/v_axis 방향, 3D와 같은 단위)
- row/col: 1부터. row는 v(위) 방향, col은 u 방향으로 증가
- 경로(mesh, texture, images.*)는 result.json이 있는 디렉토리 기준 상대경로
- 오염 분석 전(미분석) 창문: cleaning_level/csi/type_ratios/glass_ratio/occluded_ratio/images.heatmap·overlay = null,
  cleaning_label = "미분석"
- CSI (창문 오염 심각도 지수, 0~1) = Σ area_ratio × severity × k_t / (k_s_max × k_t_max) — docs/scoring.md.
  area_ratio·type_ratios는 유리 면적 A_g(창틀 마진·가림 제외) 대비. 계산에 쓴 k_t 등은 meta.csi_params
"""

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# 청소 우선순위 level 1(청소 불필요)~5(우선 청소). CSI 경계·라벨은 config/grading.yaml
CleaningLevel = Annotated[int, Field(ge=1, le=5)]
LevelKey = Literal["1", "2", "3", "4", "5", "unanalyzed"]
UNANALYZED_LABEL = "미분석"
ContaminationType = Literal["dust", "stain", "grime", "heavy"]
Ratio = Annotated[float, Field(ge=0.0, le=1.0)]
Severity = Annotated[int, Field(ge=1, le=3)]
CSI_TOL = 1e-3  # csi와 Σ 영역 기여도(meta.csi_params로 다시 계산)의 허용 차 (반올림)
Vec3 = tuple[float, float, float]
UV = tuple[Ratio, Ratio]

WINDOW_ID_RE = re.compile(r"^(?P<facade>.+)-R(?P<row>\d{2,})-C(?P<col>\d{2,})$")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReconStatus(_Model):
    """3D 복원 결과 (facade_cv.recon3d). degraded = SfM/평면 실패로 가상 평면 1개 + 2D 격자."""

    mode: Literal["3d", "degraded"]
    num_facades: int = Field(ge=0)
    registered_ratio: Ratio | None  # SfM 등록 프레임 / 입력 프레임 (SfM 실패면 null)
    warnings: list[str]


class CsiParams(_Model):
    """CSI 계산에 쓴 값 (config/grading.yaml 스냅샷). 웹이 계산 내역을 보여 주고, 보정 후에도 옛 결과를 해석할 수 있게."""

    weights: dict[ContaminationType, float]  # k_t
    severity_max: int = Field(ge=1)  # k_s_max
    severity_delta_e: list[float]  # k_s 2, 3의 ΔE*ab 하한
    frame_margin: Ratio
    level_min_csi: list[float] = Field(min_length=5, max_length=5)  # level 1~5 CSI 하한
    status: str  # "보정 전 초기값" 또는 보정 기록


class Meta(_Model):
    video: str
    fps: float = Field(gt=0)
    frames_used: int = Field(ge=0)
    scale_known: bool
    created_at: datetime
    recon_status: ReconStatus | None = None
    csi_params: CsiParams | None = None  # 오염 분석 결과가 있으면 필수


class Facade(_Model):
    id: str
    normal: Vec3
    origin: Vec3
    u_axis: Vec3
    v_axis: Vec3
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    texture: str


class Building(_Model):
    mesh: str
    facades: list[Facade] = Field(min_length=1)


class Contamination(_Model):
    type: ContaminationType
    area_ratio: Ratio  # A_j / A_g (유리 안 면적만)
    severity: Severity  # k_s
    delta_e: float | None = Field(ge=0)  # 깨끗한 유리 대비 ΔE*ab (기준 유리가 부족하면 null → severity = max)
    polygon: list[UV] = Field(min_length=3)
    area_px: int | None = Field(None, ge=0)  # A_j (분석 해상도 픽셀, T7)
    k_t: float | None = None  # 계산에 쓴 종류 가중치 (= meta.csi_params.weights[type])


class ReflectionSuspect(_Model):
    """반사 의심 영역 (넓고 윤곽·직선이 많음 — 비친 나무·건물). CSI에 넣지 않고 따로 표시만 한다."""

    area_ratio: Ratio  # 면적 / A_g
    polygon: list[UV] = Field(min_length=3)


class TypeRatios(_Model):
    dust: Ratio
    stain: Ratio
    grime: Ratio
    heavy: Ratio


class WindowImages(_Model):
    crop: str
    heatmap: str | None  # 미분석이면 null
    overlay: str | None


class Window(_Model):
    id: str
    facade_id: str
    row: int = Field(ge=1)
    col: int = Field(ge=1)
    facade_rect: tuple[float, float, float, float]
    corners_3d: tuple[Vec3, Vec3, Vec3, Vec3]
    cleaning_level: CleaningLevel | None  # null = 미분석
    cleaning_label: str = Field(min_length=1)
    csi: Ratio | None
    glass_ratio: Ratio | None  # A_g / crop 면적
    occluded_ratio: Ratio | None  # 창틀 안쪽에서 나무·전선·기둥에 가린 면적 / crop 면적
    contamination: list[Contamination]
    type_ratios: TypeRatios | None
    images: WindowImages
    num_views: int = Field(ge=0)
    # T7 오염 분석 부가 정보 (없으면 null — 이전 result.json 호환)
    glass_area_px: int | None = Field(None, ge=0)  # A_g (분석 해상도 픽셀)
    raw_coverage_pct: float | None = Field(None, ge=0, le=100)  # Σ A_j / A_g × 100 (배수 없음)
    weighted_coverage_pct: float | None = Field(None, ge=0)  # CSI × k_s_max × k_t_max × 100 = Σ 면적비 × k_s × k_t × 100 (%, 최대 k_s_max×k_t_max×100)
    low_confidence: bool | None = None  # 단일 이미지 분석 (다시점으로 반사를 거르지 못함)
    analysis_mode: Literal["multi", "single"] | None = None
    reflection_suspects: list[ReflectionSuspect] | None = None  # CSI에서 뺀 반사 의심 영역

    @model_validator(mode="after")
    def _check(self) -> "Window":
        m = WINDOW_ID_RE.match(self.id)
        if not m or (m["facade"], int(m["row"]), int(m["col"])) != (self.facade_id, self.row, self.col):
            raise ValueError(f"id {self.id!r} != {self.facade_id}-R{self.row:02d}-C{self.col:02d}")
        u0, v0, u1, v1 = self.facade_rect
        if not (u0 < u1 and v0 < v1):
            raise ValueError(f"facade_rect must satisfy u0<u1, v0<v1: {self.facade_rect}")
        analysis = (self.csi, self.type_ratios, self.glass_ratio, self.occluded_ratio)
        if self.cleaning_level is None:
            if self.cleaning_label != UNANALYZED_LABEL or any(v is not None for v in analysis):
                raise ValueError(f"window {self.id}: unanalyzed needs label {UNANALYZED_LABEL!r}, "
                                 "csi/type_ratios/glass_ratio/occluded_ratio null")
        elif any(v is None for v in analysis):
            raise ValueError(f"window {self.id}: analyzed window needs csi, type_ratios, glass_ratio, occluded_ratio")
        return self


def csi_of(w: Window, p: CsiParams) -> float:
    """영역 표에서 CSI를 다시 계산 (docs/scoring.md 식)."""
    return sum(c.area_ratio * c.severity * p.weights[c.type] for c in w.contamination) / (
        p.severity_max * max(p.weights.values()))


def level_of(csi: float, p: CsiParams) -> int:
    return max(i + 1 for i, m in enumerate(p.level_min_csi) if csi >= m)


class Summary(_Model):
    total: int = Field(ge=0)
    level_counts: dict[LevelKey, int]


class InspectionResult(_Model):
    inspection_id: str
    meta: Meta
    building: Building
    windows: list[Window]
    summary: Summary

    @model_validator(mode="after")
    def _check(self) -> "InspectionResult":
        facade_ids = [f.id for f in self.building.facades]
        if len(set(facade_ids)) != len(facade_ids):
            raise ValueError("duplicate facade id")
        window_ids = [w.id for w in self.windows]
        if len(set(window_ids)) != len(window_ids):
            raise ValueError("duplicate window id")
        for w in self.windows:
            if w.facade_id not in facade_ids:
                raise ValueError(f"window {w.id}: unknown facade_id {w.facade_id}")
        analyzed = [w for w in self.windows if w.csi is not None]
        p = self.meta.csi_params
        if analyzed and p is None:
            raise ValueError("meta.csi_params required when windows are analyzed")
        for w in analyzed:
            if abs(csi_of(w, p) - w.csi) > CSI_TOL:
                raise ValueError(f"window {w.id}: csi {w.csi} != Σ contamination terms {csi_of(w, p):.5f}")
            if level_of(w.csi, p) != w.cleaning_level:
                raise ValueError(f"window {w.id}: cleaning_level {w.cleaning_level} != level of csi {w.csi}")
        if self.summary.total != len(self.windows):
            raise ValueError(f"summary.total {self.summary.total} != {len(self.windows)} windows")
        counts = {str(lv): sum(w.cleaning_level == lv for w in self.windows) for lv in range(1, 6)}
        counts["unanalyzed"] = sum(w.cleaning_level is None for w in self.windows)
        if {k: self.summary.level_counts.get(k, 0) for k in counts} != counts:
            raise ValueError(f"summary.level_counts != actual {counts}")
        return self


def load_result(path: str | Path) -> InspectionResult:
    return InspectionResult.model_validate_json(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    # python -m facade_cv.schema runs/<id>/result.json
    r = load_result(sys.argv[1])
    print(f"OK {r.inspection_id}: {r.summary.total} windows, {r.summary.level_counts}")
