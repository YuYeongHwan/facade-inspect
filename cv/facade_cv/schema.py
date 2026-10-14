"""runs/<inspection_id>/result.json 데이터 계약 (CLAUDE.md 참조).

좌표 규칙
- 이미지 좌표: 픽셀, 원점 좌상단
- 창문 crop 내부 좌표 (u, v): 0~1 정규화
- 3D 좌표: 미터 (meta.scale_known=false면 상대 단위), Y-up
- facade_rect: 외벽 로컬 좌표 [u0, v0, u1, v1] (origin 기준, u_axis/v_axis 방향, 3D와 같은 단위)
- row/col: 1부터. row는 v(위) 방향, col은 u 방향으로 증가
- 경로(mesh, texture, images.*)는 result.json이 있는 디렉토리 기준 상대경로
"""

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# 청소 우선순위 level 1(청소 불필요)~5(우선 청소). 임계값·라벨은 config/grading.yaml
CleaningLevel = Annotated[int, Field(ge=1, le=5)]
LevelKey = Literal["1", "2", "3", "4", "5"]
ContaminationType = Literal["dust", "stain", "grime", "heavy"]
Ratio = Annotated[float, Field(ge=0.0, le=1.0)]
Vec3 = tuple[float, float, float]
UV = tuple[Ratio, Ratio]

WINDOW_ID_RE = re.compile(r"^(?P<facade>.+)-R(?P<row>\d{2,})-C(?P<col>\d{2,})$")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Meta(_Model):
    video: str
    fps: float = Field(gt=0)
    frames_used: int = Field(ge=0)
    scale_known: bool
    created_at: datetime


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
    area_ratio: Ratio
    polygon: list[UV] = Field(min_length=3)


class TypeRatios(_Model):
    dust: Ratio
    stain: Ratio
    grime: Ratio
    heavy: Ratio


class WindowImages(_Model):
    crop: str
    heatmap: str
    overlay: str


class Window(_Model):
    id: str
    facade_id: str
    row: int = Field(ge=1)
    col: int = Field(ge=1)
    facade_rect: tuple[float, float, float, float]
    corners_3d: tuple[Vec3, Vec3, Vec3, Vec3]
    cleaning_level: CleaningLevel
    cleaning_label: str = Field(min_length=1)
    score: float = Field(ge=0)
    contamination: list[Contamination]
    type_ratios: TypeRatios
    images: WindowImages
    num_views: int = Field(ge=0)

    @model_validator(mode="after")
    def _check(self) -> "Window":
        m = WINDOW_ID_RE.match(self.id)
        if not m or (m["facade"], int(m["row"]), int(m["col"])) != (self.facade_id, self.row, self.col):
            raise ValueError(f"id {self.id!r} != {self.facade_id}-R{self.row:02d}-C{self.col:02d}")
        u0, v0, u1, v1 = self.facade_rect
        if not (u0 < u1 and v0 < v1):
            raise ValueError(f"facade_rect must satisfy u0<u1, v0<v1: {self.facade_rect}")
        return self


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
        if self.summary.total != len(self.windows):
            raise ValueError(f"summary.total {self.summary.total} != {len(self.windows)} windows")
        counts = {str(lv): sum(w.cleaning_level == lv for w in self.windows) for lv in range(1, 6)}
        if {k: self.summary.level_counts.get(k, 0) for k in counts} != counts:
            raise ValueError(f"summary.level_counts != actual {counts}")
        return self


def load_result(path: str | Path) -> InspectionResult:
    return InspectionResult.model_validate_json(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    # python -m facade_cv.schema runs/<id>/result.json
    r = load_result(sys.argv[1])
    print(f"OK {r.inspection_id}: {r.summary.total} windows, {r.summary.level_counts}")
