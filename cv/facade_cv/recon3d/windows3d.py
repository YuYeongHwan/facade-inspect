"""2D 창문 관측 → 외벽 평면 위 창문 (facade_rect, row/col, ID).

입력은 WindowObservation 목록 하나뿐: 지금은 T4 detections.json에서 (observations_from_detections),
T5 추적이 생기면 tracklet마다 track_id를 채워 넣으면 group_observations가 클러스터링 대신 track_id로 묶는다.

관측 4코너 → 카메라 광선 → 가장 가까이 맞는 facade 평면과 교차 → 평면 좌표 사각형 →
(track_id 또는 중심 거리 클러스터)로 같은 창문끼리 묶기 → 중앙값 사각형 → row/col (row 1 = 아래층) →
ID "F{n}-R{rr}-C{cc}". 같은 칸에 둘이면 관측이 많은 쪽만 남긴다.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from facade_cv.detect.postprocess import cluster_1d
from facade_cv.recon3d.geometry import CameraView, FacadeFrame


@dataclass
class WindowObservation:
    frame: str  # 이미지 파일명 (CameraView.name)
    corners: np.ndarray  # (4,2) TL TR BR BL 픽셀
    conf: float
    track_id: int | None = None  # T5 tracklet id. 있으면 클러스터링 대신 이걸로 묶는다


@dataclass
class LiftedObservation:
    obs: WindowObservation
    facade_id: str
    rect: np.ndarray  # [u0, v0, u1, v1] 평면 좌표
    uv: np.ndarray = field(repr=False, default=None)  # (4,2) TL TR BR BL 평면 좌표 (기울기 추정용)


def vertical_tilt(lifted: list[LiftedObservation]) -> float | None:
    """창문 세로 변 (BL→TL, BR→TR)이 평면 v축에서 기운 각의 중앙값 (rad, +면 위로 갈수록 +u 쪽).
    면 1개일 때 업벡터의 '면 안 회전'은 외벽 법선으로 못 정하므로 이걸로 보정한다."""
    if not lifted:
        return None
    a = []
    for x in lifted:
        for top, bottom in ((0, 3), (1, 2)):
            du, dv = x.uv[top] - x.uv[bottom]
            if dv > 0:
                a.append(np.arctan2(du, dv))
    return float(np.median(a)) if a else None


@dataclass
class Window3D:
    facade_id: str
    rect: np.ndarray  # [u0, v0, u1, v1]
    observations: list[LiftedObservation] = field(repr=False)
    row: int = 0
    col: int = 0

    @property
    def id(self) -> str:
        return f"{self.facade_id}-R{self.row:02d}-C{self.col:02d}"

    @property
    def num_views(self) -> int:
        return len({o.obs.frame for o in self.observations})

    def corners_3d(self, f: FacadeFrame) -> np.ndarray:
        """왼쪽 아래, 오른쪽 아래, 오른쪽 위, 왼쪽 위 (밖에서 봤을 때)."""
        u0, v0, u1, v1 = self.rect
        return f.to_world([[u0, v0], [u1, v0], [u1, v1], [u0, v1]])


def observations_from_detections(det: dict) -> list[WindowObservation]:
    """T4 detections.json → 확정 창문만 (보완 후보·제거된 것 제외)."""
    return [WindowObservation(img["image"], np.asarray(w["corners"], float), float(w["conf"]))
            for img in det["images"] for w in img["windows"]]


def lift_observations(obs: list[WindowObservation], cameras: dict[str, CameraView], facades: list[FacadeFrame],
                      margin: float) -> list[LiftedObservation]:
    """관측마다 4코너 광선이 모두 앞에서 맞고 (margin 안에서) 범위 안인 facade 중 가장 가까운 것."""
    out = []
    for o in obs:
        cam = cameras.get(o.frame)
        if cam is None:  # SfM 미등록 프레임
            continue
        dirs = cam.rays(o.corners)
        origins = np.repeat(cam.center[None], 4, 0)
        best = None
        for f in facades:
            if (dirs @ f.normal >= 0).any():  # 외벽 뒤에서 본 광선
                continue
            X, s = f.intersect(origins, dirs)
            if np.isnan(s).any():
                continue
            uv = f.to_uv(X)
            mu, mv = margin * f.width, margin * f.height
            if not ((uv[:, 0] >= -mu).all() and (uv[:, 0] <= f.width + mu).all()
                    and (uv[:, 1] >= -mv).all() and (uv[:, 1] <= f.height + mv).all()):
                continue
            if best is None or s.mean() < best[0]:
                best = (s.mean(), f.id, uv)
        if best is None:
            continue
        uv = best[2]  # TL TR BR BL → 왼쪽 u, 오른쪽 u, 아래 v, 위 v
        u0, u1 = sorted([(uv[0, 0] + uv[3, 0]) / 2, (uv[1, 0] + uv[2, 0]) / 2])
        v0, v1 = sorted([(uv[2, 1] + uv[3, 1]) / 2, (uv[0, 1] + uv[1, 1]) / 2])
        out.append(LiftedObservation(o, best[1], np.array([u0, v0, u1, v1]), uv))
    return out


def group_observations(lifted: list[LiftedObservation], dist_ratio: float) -> list[list[LiftedObservation]]:
    """facade별로 묶기. track_id가 모두 있으면 track_id로, 아니면 중심 거리 (평균 연결) 클러스터링."""
    groups: list[list[LiftedObservation]] = []
    for fid in sorted({x.facade_id for x in lifted}):
        items = [x for x in lifted if x.facade_id == fid]
        if all(x.obs.track_id is not None for x in items):
            by: dict[int, list[LiftedObservation]] = {}
            for x in items:
                by.setdefault(x.obs.track_id, []).append(x)
            groups += list(by.values())
            continue
        if len(items) == 1:
            groups.append(items)
            continue
        r = np.array([x.rect for x in items])
        centers = np.stack([(r[:, 0] + r[:, 2]) / 2, (r[:, 1] + r[:, 3]) / 2], 1)
        size = np.median(((r[:, 2] - r[:, 0]) + (r[:, 3] - r[:, 1])) / 2)
        labels = fcluster(linkage(centers, "average"), dist_ratio * size, criterion="distance")
        groups += [[x for x, lb in zip(items, labels) if lb == k] for k in np.unique(labels)]
    return groups


def assign_grid(windows: list[Window3D], gap_ratio: float) -> int:
    """facade별 row/col (1부터, row 1 = 가장 낮은 층, col 1 = u가 작은 쪽). 같은 칸 중복은 관측 많은 것만 남김.
    버린 수를 반환."""
    for fid in {w.facade_id for w in windows}:
        ws = [w for w in windows if w.facade_id == fid]
        r = np.array([w.rect for w in ws])
        cu, cv = (r[:, 0] + r[:, 2]) / 2, (r[:, 1] + r[:, 3]) / 2
        rows = cluster_1d(cv, gap_ratio * np.median(r[:, 3] - r[:, 1]))  # 오름차순 = 아래층부터
        cols = cluster_1d(cu, gap_ratio * np.median(r[:, 2] - r[:, 0]))
        for w, rr, cc in zip(ws, rows, cols):
            w.row, w.col = int(rr) + 1, int(cc) + 1
    return dedupe_cells(windows)


def dedupe_cells(windows: list[Window3D]) -> int:
    """같은 ID(같은 칸)가 여럿이면 관측이 많은 (같으면 큰) 것만 남긴다. 버린 수를 반환."""
    seen: dict[str, Window3D] = {}
    area = lambda w: (w.rect[2] - w.rect[0]) * (w.rect[3] - w.rect[1])  # noqa: E731
    for w in sorted(windows, key=lambda w: (-w.num_views, -area(w))):
        seen.setdefault(w.id, w)
    dropped = len(windows) - len(seen)
    windows[:] = sorted(seen.values(), key=lambda w: w.id)
    return dropped


def build_windows(obs: list[WindowObservation], cameras: list[CameraView], facades: list[FacadeFrame],
                  cfg: dict) -> tuple[list[Window3D], dict]:
    """(창문 목록, 통계)."""
    lifted = lift_observations(obs, {c.name: c for c in cameras}, facades, cfg["max_inside_margin"])
    groups = group_observations(lifted, cfg["cluster_dist_ratio"])
    by_id = {f.id: f for f in facades}
    windows, outside = [], 0
    for g in groups:
        if len({x.obs.frame for x in g}) < cfg["min_observations"]:
            continue
        f = by_id[g[0].facade_id]
        r = np.median([x.rect for x in g], axis=0)
        cu, cv = (r[0] + r[2]) / 2, (r[1] + r[3]) / 2
        if not (0 <= cu <= f.width and 0 <= cv <= f.height):  # 중심이 외벽 밖 (모서리 너머 다른 면 등)
            outside += 1
            continue
        windows.append(Window3D(f.id, np.clip(r, 0, [f.width, f.height, f.width, f.height]), g))
    dropped = assign_grid(windows, cfg["grid_gap_ratio"]) if windows else 0
    stats = {"observations": len(obs), "lifted": len(lifted), "groups": len(groups), "windows": len(windows),
             "dropped_few_views": sum(len({x.obs.frame for x in g}) < cfg["min_observations"] for g in groups),
             "dropped_outside": outside, "dropped_same_cell": dropped}
    return windows, stats
