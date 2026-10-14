"""창문 검출 기하 후처리: 마스크 → 4점 사각형 → 외벽 필터 → 정면화 격자 정규화.

a) quad: approxPolyDP 4점(볼록)이 되면 그것, 아니면 minAreaRect. 면적 < min_area_px, 직사각형도 < min_rectangularity → 제거
b) 외벽 필터: 창문이 (구멍을 메우고 팽창한) 외벽 마스크 밖이면 제거. 외벽 마스크가 창문 bbox 합집합보다 작거나
   창문 대부분이 밖이면 외벽 검출 실패로 보고 필터를 건너뛴다 (warnings에 기록)
c) 격자: 정면화 좌표(실패 시 원본 좌표)에서 창문 중심 y → row, x → col 1D 클러스터링. 제거하지 않고 flag만:
   off_grid(행·열 모두 혼자), odd_width(같은 열 폭 중앙값과 다름), odd_height(같은 행 높이 중앙값과 다름).
   격자 빈칸에 conf × fill_conf_ratio 이상인 저신뢰 검출이 있으면 fill_candidates(창문 수와 별도), 없으면 empty_cells.
   격자 점유율이 min_occupancy 미만이면 빈칸·보완 후보는 만들지 않는다 (한 평면 가정이 깨진 경우).
   ("마스크 확률"은 YOLO 검출 conf로 본다)
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from facade_cv.config import load_config
from facade_cv.detect.segmenter import Detection, split_classes
from facade_cv.rectify.homography import Rectification, estimate_rectification


@dataclass
class Window:
    corners: np.ndarray  # (4, 2) TL, TR, BR, BL 원본 픽셀
    conf: float
    row: int | None = None
    col: int | None = None
    flags: list[str] = field(default_factory=list)
    reason: str | None = None  # 제거 사유 (small / not_rect / outside_facade) — 제거된 것만

    def to_dict(self) -> dict:
        d = {"corners": np.round(self.corners, 1).tolist(), "conf": round(self.conf, 4)}
        return d | ({"reason": self.reason} if self.reason else {"row": self.row, "col": self.col, "flags": self.flags})


@dataclass
class PostResult:
    windows: list[Window]  # 남은 것 (flag 포함, 보완 후보 제외)
    removed: list[Window]
    empty_cells: list[tuple[int, int]]
    fill_candidates: list[Window]  # 격자 빈칸을 메우는 저신뢰 검출 — 창문 수에 넣지 않는다
    warnings: list[str]
    rect: Rectification | None
    facade_mask: np.ndarray | None  # 외벽 필터에 쓴 팽창 마스크 (건너뛰었으면 None)
    num_raw: int  # conf 이상 창문 검출 수
    line_region: np.ndarray | None = None  # 정면화 직선 검출 범위 (외벽 마스크 또는 창문 껍질)

    def to_dict(self) -> dict:
        return {
            "counts": {"raw": self.num_raw, "final": len(self.windows), "removed": len(self.removed),
                       "flagged": sum(bool(w.flags) for w in self.windows),
                       "fill_candidates": len(self.fill_candidates), "empty_cells": len(self.empty_cells)},
            "rectified": self.rect is not None,
            "focal_px": None if self.rect is None else round(self.rect.focal, 1),
            "facade_filter": self.facade_mask is not None,
            "warnings": self.warnings,
            "windows": [w.to_dict() for w in self.windows],
            "removed": [w.to_dict() for w in self.removed],
            "fill_candidates": [w.to_dict() for w in self.fill_candidates],
            "empty_cells": [list(c) for c in self.empty_cells],
        }


def order_corners(pts: np.ndarray) -> np.ndarray:
    """4점 → TL, TR, BR, BL (이미지 좌표, y 아래). 중심 기준 각도 정렬 후 x+y 최소를 TL로."""
    pts = np.asarray(pts, np.float64).reshape(4, 2)
    c = pts.mean(0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
    return np.roll(pts, -int(np.argmin(pts.sum(1))), axis=0)


def mask_to_quad(mask: np.ndarray, cfg: dict) -> tuple[np.ndarray | None, float, float]:
    """(corners 또는 None, 마스크 면적, 직사각형도 = 마스크 면적 / quad 면적). 가장 큰 연결 성분만 본다."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0, 0.0
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    approx = cv2.approxPolyDP(cnt, cfg["approx_eps"] * cv2.arcLength(cnt, True), True)
    quad = approx.reshape(4, 2) if len(approx) == 4 and cv2.isContourConvex(approx) else \
        cv2.boxPoints(cv2.minAreaRect(cnt))
    quad = order_corners(quad)
    quad_area = cv2.contourArea(quad.astype(np.float32))
    return quad, area, min(1.0, area / quad_area) if quad_area > 0 else 0.0


def _to_window(d: Detection, cfg: dict) -> Window | None:
    quad, area, rectangularity = mask_to_quad(d.mask, cfg)
    if quad is None:
        return None
    w = Window(quad, d.conf)
    if area < cfg["min_area_px"]:
        w.reason = "small"
    elif rectangularity < cfg["min_rectangularity"]:
        w.reason = "not_rect"
    return w


def _quad_mask(corners: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, np.round(corners).astype(np.int32), 1)
    return m.astype(bool)


def _inside_fraction(w: Window, region: np.ndarray) -> float:
    m = _quad_mask(w.corners, region.shape)
    return float(region[m].mean()) if m.any() else 0.0


def facade_filter(
    windows: list[Window], facade_mask: np.ndarray | None, cfg: dict
) -> tuple[list[Window], list[Window], str | None, np.ndarray | None]:
    """(남은 것, 제거된 것, 건너뛴 이유 또는 None, 팽창한 외벽 마스크 또는 None)."""
    if not windows:
        return windows, [], None, None
    if facade_mask is None or not facade_mask.any():
        return windows, [], "facade_filter_skipped: no facade detection", None
    fc = cfg["facade_filter"]
    # 분할 모델은 창문 자리를 외벽에서 뺀다 → 바깥 윤곽으로 구멍을 메워야 창문이 외벽 '안'이 된다
    cnts, _ = cv2.findContours(facade_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    facade_mask = cv2.drawContours(np.zeros(facade_mask.shape, np.uint8), cnts, -1, 1, cv2.FILLED).astype(bool)
    boxes = np.zeros(facade_mask.shape, bool)
    for w in windows:
        (x0, y0), (x1, y1) = np.floor(w.corners.min(0)).astype(int), np.ceil(w.corners.max(0)).astype(int)
        boxes[max(y0, 0):y1, max(x0, 0):x1] = True
    if facade_mask.sum() < boxes.sum():
        return windows, [], "facade_filter_skipped: facade mask smaller than window bbox union", None
    r = max(1, round(fc["margin_ratio"] * max(facade_mask.shape)))
    region = cv2.dilate(facade_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
    region = region.astype(bool)
    outside = [_inside_fraction(w, region) < fc["min_inside"] for w in windows]
    if sum(outside) > fc["max_outside"] * len(windows):
        return windows, [], f"facade_filter_skipped: {sum(outside)}/{len(windows)} windows outside facade mask", None
    for w, out in zip(windows, outside):
        if out:
            w.reason = "outside_facade"
    return [w for w in windows if not w.reason], [w for w in windows if w.reason], None, region


def cluster_1d(values: np.ndarray, gap: float) -> np.ndarray:
    """정렬 후 이웃 간격 > gap에서 끊는다. 라벨은 값 오름차순 (row 0 = 위, col 0 = 왼쪽)."""
    order = np.argsort(values)
    labels = np.empty(len(values), int)
    labels[order] = np.concatenate([[0], np.cumsum(np.diff(values[order]) > gap)])
    return labels


def _geometry(windows: list[Window], rect: Rectification | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(중심 (N,2), 폭 (N,), 높이 (N,)) — rect가 있으면 정면화 좌표."""
    q = np.array([rect.apply(w.corners) if rect else w.corners for w in windows]).reshape(-1, 4, 2)
    tl, tr, br, bl = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    width = (np.linalg.norm(tr - tl, axis=1) + np.linalg.norm(br - bl, axis=1)) / 2
    height = (np.linalg.norm(bl - tl, axis=1) + np.linalg.norm(br - tr, axis=1)) / 2
    return q.mean(1), width, height


def regularize_grid(
    windows: list[Window], rect: Rectification | None, candidates: list[Window], cfg: dict
) -> tuple[list[tuple[int, int]], list[Window], str | None]:
    """windows에 row/col/flags를 채운다. (남은 빈칸, 빈칸을 메우는 보완 후보, 경고) 반환 — 후보는 windows에 넣지 않는다.
    격자 점유율 < min_occupancy면 (여러 외벽이 섞인 코너 건물 등) 빈칸·보완 후보를 만들지 않고 경고만."""
    gc = cfg["grid"]
    if not windows:
        return [], [], None
    c, wd, ht = _geometry(windows, rect)
    rows = cluster_1d(c[:, 1], gc["gap_ratio"] * np.median(ht))
    cols = cluster_1d(c[:, 0], gc["gap_ratio"] * np.median(wd))
    on = np.ones(len(windows), bool)
    if len(windows) >= 3:
        on = (np.bincount(rows)[rows] > 1) | (np.bincount(cols)[cols] > 1)
    for w, ok in zip(windows, on):
        if not ok:
            w.flags.append("off_grid")
    if not on.any():
        return [], [], None

    # off_grid를 뺀 창문으로 다시 묶어 row/col 번호를 0부터 연속으로
    idx = np.flatnonzero(on)
    rows = cluster_1d(c[idx, 1], gc["gap_ratio"] * np.median(ht[idx]))
    cols = cluster_1d(c[idx, 0], gc["gap_ratio"] * np.median(wd[idx]))
    for i, r, k in zip(idx, rows, cols):
        windows[i].row, windows[i].col = int(r), int(k)
    for labels, size, flag in ((cols, wd[idx], "odd_width"), (rows, ht[idx], "odd_height")):
        for g in np.unique(labels):
            members = idx[labels == g]
            if len(members) < gc["min_size_members"]:
                continue
            med = np.median(size[labels == g])
            for i, s in zip(members, size[labels == g]):
                if abs(s - med) > gc["size_tol"] * med:
                    windows[i].flags.append(flag)

    row_y = {r: np.median(c[idx[rows == r], 1]) for r in np.unique(rows)}
    row_h = {r: np.median(ht[idx[rows == r]]) for r in np.unique(rows)}
    col_x = {k: np.median(c[idx[cols == k], 0]) for k in np.unique(cols)}
    col_w = {k: np.median(wd[idx[cols == k]]) for k in np.unique(cols)}
    occupied = set(zip(rows.tolist(), cols.tolist()))
    occupancy = len(occupied) / (len(row_y) * len(col_x))
    if occupancy < gc["min_occupancy"]:
        return [], [], f"grid_unreliable: occupancy {occupancy:.2f} ({len(row_y)}x{len(col_x)}) — no empty cells/fill"
    cand = list(candidates)
    cc = _geometry(cand, rect)[0] if cand else np.empty((0, 2))
    empty, fills, used = [], [], set()
    for r in row_y:
        for k in col_x:
            if (r, k) in occupied:
                continue
            d = np.abs(cc - [col_x[k], row_y[r]]) / [col_w[k], row_h[r]] if len(cc) else np.empty((0, 2))
            hit = [j for j in np.flatnonzero((d <= gc["fill_tol"]).all(1))] if len(d) else []
            hit = [j for j in hit if j not in used]
            if len(hit):
                best = hit[np.argmax([cand[j].conf for j in hit])]
                w = cand[best]
                used.add(best)
                w.row, w.col, w.flags = int(r), int(k), ["fill_candidate"]
                fills.append(w)
            else:
                empty.append((int(r), int(k)))
    return empty, fills, None


def window_hull(windows: list[Window], shape: tuple[int, int]) -> np.ndarray | None:
    """창문들의 볼록 껍질을 창 크기 중앙값만큼 팽창한 영역 (외벽 마스크 대신 직선 검출 범위)."""
    if not windows:
        return None
    pts = np.vstack([w.corners for w in windows]).astype(np.float32)
    m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(pts).astype(np.int32), 1)
    size = np.median([np.ptp(w.corners, axis=0).max() for w in windows])
    k = max(3, int(size) | 1)
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)


def postprocess(image: np.ndarray, dets: list[Detection], cfg: dict | None = None,
                rect_cfg: dict | None = None) -> PostResult:
    """dets는 conf × fill_conf_ratio까지 낮춰 뽑은 검출. conf 미만은 격자 빈칸 보완 후보로만 쓴다."""
    cfg = cfg or load_config("detect")
    wins, facades = split_classes(dets)
    warnings: list[str] = []

    # a) quad
    strong = [w for w in (_to_window(d, cfg) for d in wins if d.conf >= cfg["conf"]) if w]
    weak = [w for w in (_to_window(d, cfg) for d in wins if d.conf < cfg["conf"]) if w and not w.reason]
    kept, removed = [w for w in strong if not w.reason], [w for w in strong if w.reason]

    # b) 외벽 필터
    facades = [f for f in facades if f.conf >= cfg["conf"]]  # 저신뢰는 창문 보완 후보용으로만 뽑은 것
    facade_mask = np.logical_or.reduce([f.mask for f in facades]) if facades else None
    kept, out, skipped, region = facade_filter(kept, facade_mask, cfg)
    removed += out
    if skipped:
        warnings.append(skipped)
    if region is not None:
        weak = [w for w in weak if _inside_fraction(w, region) >= cfg["facade_filter"]["min_inside"]]

    # c) 정면화 + 격자
    line_region = region if region is not None else window_hull(kept, image.shape[:2])
    rect = estimate_rectification(image, line_region, rect_cfg)
    if rect is None:
        warnings.append("rectification_failed: grid on image coordinates")
    empty, fills, grid_warning = regularize_grid(kept, rect, weak, cfg)
    if grid_warning:
        warnings.append(grid_warning)
    return PostResult(kept, removed, empty, fills, warnings, rect, region, sum(d.conf >= cfg["conf"] for d in wins),
                      line_region)
