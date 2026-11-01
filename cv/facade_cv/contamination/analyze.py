"""창문 오염 영역 검출·유형 분류 → CSI (규칙 기반, 학습 없음). 설명: docs/scoring.md "오염 영역 검출과 유형 분류"

설정 config/contamination.yaml, CSI·k_s·level은 grade.py (config/grading.yaml)를 그대로 쓴다.
영상 기반 유리 오염 검출 선행 연구 [R7]이 지적한 밝기·배경 민감성이 단일 이미지 모드의 한계 → low_confidence로 표시.

  정면화 창문 → 유리 A_g (창틀 마진·가림 제외) → 창틀·창살 막대 제거 (행·열 투영) → 칸별 2차 곡면 배경
  → 후보 = 칸 배경 대비 ΔE ≥ t_g 또는 국소 ΔE ≥ t_l [R22], 반사광(밝고 무채색) 제외 → 흩어진 점 묶기 → 작은 잡음 제거
  → 물기 제외(시도) → 유형 규칙 (heavy → grime → stain → dust) → 깨끗한 유리 기준색 대비 ΔE → k_s → compute_csi → level

python -m facade_cv.contamination.analyze run runs/<id>              result.json 창문 분석 (recon/window_obs.json 있으면 원본 프레임)
python -m facade_cv.contamination.analyze image X.jpg --out runs/x   사진 1장 = 유리 전체
python -m facade_cv.contamination.analyze photos --images data/own/dirty_windows --out runs/demo_<이름>
                                                                      T4 창문 검출 → 창별 분석 → 웹용 result.json
"""

import json
import time
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import typer

from facade_cv.config import load_config
from facade_cv.contamination.clip_score import clip_dirty
from facade_cv.contamination.grade import (
    Region,
    cleaning_level,
    compute_csi,
    csi_params,
    glass_mask,
    k_t_max,
    lab,
    severity_level,
)
from facade_cv.schema import InspectionResult

TYPES = ("dust", "stain", "grime", "heavy")


@dataclass
class Found:
    type: str
    mask: np.ndarray  # bool, 유리 안 오염 픽셀 (A_j)
    group: np.ndarray  # bool, 묶음 영역 (polygon용)
    delta_e: float | None
    severity: int
    features: dict


@dataclass
class Analysis:
    crop: np.ndarray  # 분석한 정면화 BGR
    glass: np.ndarray  # bool A_g
    occluded_ratio: float
    regions: list[Found]
    csi: float
    level: int
    label: str
    delta_e_map: np.ndarray  # 기준색 대비 픽셀 ΔE (heatmap)
    glare: np.ndarray  # bool, 제외한 반사광
    suspects: list[Found] = field(default_factory=list)  # 반사 의심 (CSI 제외, 따로 표시) — reflection.enabled일 때만


def _odd(n: float) -> int:
    n = max(3, int(n))
    return n if n % 2 else n + 1


def _disk(px: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, px), max(1, px)))


def resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    s = max_side / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def rectify_quad(img: np.ndarray, corners, max_side: int) -> tuple[np.ndarray, np.ndarray]:
    """4코너 (TL TR BR BL) → 정면화 직사각형 이미지와 호모그래피 (원본 → 정면화). 크기 = 대변 평균, 긴 변 ≤ max_side."""
    q = np.asarray(corners, np.float32).reshape(4, 2)
    w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    s = min(1.0, max_side / max(w, h, 1))
    W, H = max(1, round(w * s)), max(1, round(h * s))
    M = cv2.getPerspectiveTransform(q, np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]]))
    return cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_AREA), M


def warp_polygons(polys, M: np.ndarray, shape: tuple[int, int]) -> np.ndarray | None:
    """원본 픽셀 polygon 목록 → 정면화 좌표 bool 마스크 (없으면 None)."""
    if not polys:
        return None
    m = np.zeros(shape, np.uint8)
    for p in polys:
        q = cv2.perspectiveTransform(np.asarray(p, np.float64).reshape(-1, 1, 2), M)
        cv2.fillPoly(m, [np.round(q).astype(np.int32)], 1)
    return m.astype(bool)


def reference_lab(img_lab: np.ndarray, mask: np.ndarray, q: tuple[float, float]) -> np.ndarray | None:
    """mask 픽셀 중 L*가 [q0, q1] 분위 사이인 것의 LAB 중앙값 — 반사·짙은 얼룩을 뺀 강건한 기준색."""
    px = img_lab[mask]
    if not len(px):
        return None
    lo, hi = np.quantile(px[:, 0], q)
    band = px[(px[:, 0] >= lo) & (px[:, 0] <= hi)]
    return np.median(band if len(band) else px, 0)


def background_surface(img_lab: np.ndarray, mask: np.ndarray, cfg: dict) -> np.ndarray:
    """유리 픽셀에 채널별 2차 곡면을 강건하게 맞춘 배경 (H,W,3) — 하늘·조명의 완만한 밝기 기울기를 흡수한다.
    맞춘 뒤 잔차가 큰 (분위 밖) 픽셀을 빼고 다시 맞추기를 반복 (얼룩이 곡면을 끌어당기지 않게)."""
    rc = cfg["reference"]
    h, w = mask.shape
    yy, xx = np.mgrid[0:h, 0:w]
    x, y = xx / w - 0.5, yy / h - 0.5
    A = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], -1)
    ys, xs = np.nonzero(mask)
    step = max(1, len(ys) // rc["fit_samples"])
    ys, xs = ys[::step], xs[::step]
    keep = np.ones(len(ys), bool)
    for _ in range(rc["fit_iters"]):
        coef = np.linalg.lstsq(A[ys[keep], xs[keep]], img_lab[ys[keep], xs[keep]], rcond=None)[0]
        r = np.linalg.norm(img_lab[ys, xs] - A[ys, xs] @ coef, axis=1)
        keep = r <= np.quantile(r, rc["fit_keep"])
    return A @ coef


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """연속 True 구간 [a, b) 목록."""
    d = np.diff(np.r_[0, flags.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _edge_step(L0: np.ndarray, a: int, b: int, lo: int, hi: int, step: int) -> float:
    """1차원 막대 [a, b) (L0의 1축 방향) 양쪽 경계의 밝기 계단 크기 = |L(경계 안) − L(경계 밖 step px)| 평균 중 큰 쪽.
    유리 범위 [lo, hi) 밖이면 그쪽 경계는 보지 않는다 (막대가 유리 끝에 붙은 경우)."""
    out = [0.0]
    if a - step >= lo:
        out.append(float(np.abs(L0[:, a] - L0[:, a - step]).mean()))
    if b - 1 + step < hi:
        out.append(float(np.abs(L0[:, b - 1] - L0[:, b - 1 + step]).mean()))
    return max(out)


def _straight_cover(segs: np.ndarray, a: int, b: int, lo: int, hi: int, tol: int) -> float:
    """막대 띠 [a−tol, b+tol) 안에 양 끝점이 모두 있는 선분(= 막대와 나란한 직선)의 축 방향 투영 합집합 / (hi − lo).
    segs: (N, 4) [가로 좌표1, 축 좌표1, 가로 좌표2, 축 좌표2] — 수평 막대는 x·y를 바꿔 넘긴다."""
    if not len(segs):
        return 0.0
    inside = ((segs[:, 0] >= a - tol) & (segs[:, 0] < b + tol) & (segs[:, 2] >= a - tol) & (segs[:, 2] < b + tol))
    cover = np.zeros(hi - lo, bool)
    for _, p1, _, p2 in segs[inside]:
        cover[max(int(min(p1, p2)) - lo, 0):max(int(np.ceil(max(p1, p2))) - lo, 0)] = True
    return float(cover.mean()) if len(cover) else 0.0


def bar_structures(diff: np.ndarray, glass: np.ndarray, L: np.ndarray, cfg: dict) -> np.ndarray:
    """정면화 창문의 수평·수직 막대 (창틀·창살·비친 건물 모서리): 유리 높이(폭)의 line_span 이상이 '다름'(diff)인
    열(행) 묶음을 두께와 상관없이 막대로 보되, 경계가 날카로운 것만 (경계 밝기 계단 ≥ line_min_edge).
    노을·하늘 그라데이션처럼 경계가 부드러운 띠는 막대가 아니다. 정면화 오차로 조금 기운 막대를 위해 diff를
    짧은 변 × line_tol_ratio만큼 옆으로 번지게 해서 센다.
    기하 규칙: 오염은 유리 끝에서 끝까지 곧은 직선으로 이어지는 일이 드물다 (유리 폭 전체를 덮는 균일한 띠는 예외 — 한계)."""
    cc = cfg["candidate"]
    tol = max(1, round(min(glass.shape) * cc["line_tol_ratio"]))
    ys, xs = np.nonzero(glass)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    d = diff.astype(np.uint8)
    dv = cv2.dilate(d, np.ones((1, 2 * tol + 1), np.uint8))[y0:y1, x0:x1].mean(0) >= cc["line_span"]
    dh = cv2.dilate(d, np.ones((2 * tol + 1, 1), np.uint8))[y0:y1, x0:x1].mean(1) >= cc["line_span"]
    L0 = L[..., 0]
    out = np.zeros_like(glass)
    st, me = cc["line_edge_step"], cc["line_min_edge"]
    segs = None
    if cc["line_straight"]:  # 곧은 막대만: 나란한 LSD 선분 [R32]이 유리 높이(폭)의 line_span 이상을 덮어야 한다
        s = cv2.createLineSegmentDetector().detect(np.clip(L0 * 2.55, 0, 255).astype(np.uint8))[0]
        segs = np.zeros((0, 4), np.float32) if s is None else s.reshape(-1, 4)

    def sharp(L1: np.ndarray, a: int, b: int, lo: int, hi: int) -> bool:  # 번지게 한 tol을 빼고 원래 경계에서 잰다
        a, b = (a + tol, b - tol) if b - a > 2 * tol else (a, b)
        return _edge_step(L1, a, b, lo, hi, st) >= me

    def straight(sg: np.ndarray | None, a: int, b: int, lo: int, hi: int) -> bool:
        return sg is None or _straight_cover(sg, a, b, lo, hi, tol) >= cc["line_span"]

    vsegs = segs  # 수직 막대: [x1, y1, x2, y2] 그대로
    hsegs = None if segs is None else segs[:, [1, 0, 3, 2]]  # 수평 막대: 축을 바꿔서
    for a, b in _runs(dv):  # 수직 막대: 열 묶음, 경계는 좌우
        if sharp(L0[y0:y1], x0 + a, x0 + b, x0, x1) and straight(vsegs, x0 + a, x0 + b, y0, y1):
            out[y0:y1, x0 + a:x0 + b] = True
    for a, b in _runs(dh):  # 수평 막대: 행 묶음, 경계는 위아래
        if sharp(L0[:, x0:x1].T, y0 + a, y0 + b, y0, y1) and straight(hsegs, y0 + a, y0 + b, x0, x1):
            out[y0 + a:y0 + b, x0:x1] = True
    return out & glass & diff


def glare_mask(L: np.ndarray, glass: np.ndarray, cfg: dict) -> np.ndarray:
    """반사광 = 매우 밝고 무채색 (팽창). glare.keep_sharp가 켜져 있으면 그중 '윤곽이 날카롭고 덩어리지고 작은' 조각은
    반사가 아니라 흰 오염(새똥 등)으로 보고 남긴다 — 실제 반사광은 가장자리가 부드럽게 옅어진다."""
    gc = cfg["glare"]
    raw = (L[..., 0] >= gc["L_min"]) & (np.hypot(L[..., 1], L[..., 2]) <= gc["C_max"]) & glass
    keep = np.zeros_like(raw)
    ks = gc["keep_sharp"]
    if ks["enabled"] and raw.any():
        gx = cv2.Sobel(L[..., 0], cv2.CV_32F, 1, 0, ksize=3) / 8
        gy = cv2.Sobel(L[..., 0], cv2.CV_32F, 0, 1, ksize=3) / 8
        grad = np.hypot(gx, gy)  # L*/px
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(raw.astype(np.uint8))
        a_g = glass.sum()
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] > ks["max_area_ratio"] * a_g:
                continue
            m = (lbl == i).astype(np.uint8)
            ring = cv2.dilate(m, np.ones((3, 3), np.uint8)).astype(bool) & ~m.astype(bool)
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c = max(cs, key=cv2.contourArea)
            solidity = cv2.contourArea(c) / max(cv2.contourArea(cv2.convexHull(c)), 1.0)
            if ring.any() and grad[ring].mean() >= ks["min_edge"] and solidity >= ks["min_solidity"]:
                keep |= m.astype(bool)
    glare = cv2.dilate((raw & ~keep).astype(np.uint8), _disk(gc["dilate_px"])).astype(bool)
    return glare & glass & ~keep


def panes(glass: np.ndarray, lines: np.ndarray, min_ratio: float) -> list[np.ndarray]:
    """창살로 나뉜 유리 칸 — glass에서 직선 구조를 뺀 연결 요소 중 유리의 min_ratio 이상인 것."""
    n, lbl = cv2.connectedComponents((glass & ~lines).astype(np.uint8))
    out = [lbl == i for i in range(1, n)]
    return [m for m in out if m.sum() >= min_ratio * glass.sum()]


def candidates(bgr: np.ndarray, glass: np.ndarray, cfg: dict,
               reference: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(후보 bool, 배경 대비 ΔE 맵, 제외 bool = 반사광 + 창살·창틀 직선 구조).

    1) 국소·전체 배경 대비 '다른' 픽셀의 행·열 투영으로 수평·수직 막대(창틀·창살)를 찾아 유리를 칸으로 나눈다
    2) 칸마다 2차 곡면 배경을 따로 맞춘다 — 칸마다 유리 뒤 실내·커튼이 달라 한 배경으로는 칸 전체가 후보가 된다
    3) 후보 = 배경 대비 ΔE ≥ global 또는 국소 ΔE ≥ local, 반사광·직선 구조 제외
    """
    cc = cfg["candidate"]
    sm = cv2.GaussianBlur(bgr, (0, 0), cc["blur_sigma"]) if cc["blur_sigma"] > 0 else bgr
    L = lab(sm)
    if reference is not None:  # 같은 위치 깨끗한 사진 (정렬됨) = 픽셀별 배경. 유리 뒤 풍경이 같으니 막대·칸·국소 배경이 필요 없다
        rs = cv2.GaussianBlur(reference, (0, 0), cc["blur_sigma"]) if cc["blur_sigma"] > 0 else reference
        de = np.linalg.norm(L - lab(rs), axis=2).astype(np.float32)
        glare = glare_mask(L, glass, cfg)
        c = cv2.morphologyEx(((de >= cc["global_delta_e"]) & glass & ~glare).astype(np.uint8), cv2.MORPH_OPEN, _disk(cc["open_px"]))
        c = cv2.morphologyEx(c, cv2.MORPH_CLOSE, _disk(cc["close_px"]))
        return c.astype(bool) & glass & ~glare, de * glass, glare
    short = min(bgr.shape[:2])
    bg = cv2.medianBlur(sm, _odd(short * cc["local_bg_ratio"]))
    local = (np.linalg.norm(L - lab(bg), axis=2) >= cc["local_delta_e"]) & glass
    whole = (np.linalg.norm(L - background_surface(L, glass, cfg), axis=2) >= cc["global_delta_e"]) & glass
    lines = bar_structures(local | whole, glass, L, cfg)
    lines = cv2.dilate(lines.astype(np.uint8), _disk(cc["line_dilate_px"])).astype(bool) & glass
    de_g = np.zeros(glass.shape, np.float32)
    g = np.zeros_like(glass)
    for pane in panes(glass, lines, cc["min_pane_ratio"]) or [glass & ~lines]:
        de = np.linalg.norm(L - background_surface(L, pane, cfg), axis=2)
        de_g[pane] = de[pane]
        gp = (de >= cc["global_delta_e"]) & pane
        if gp.sum() <= cc["max_area_ratio"] * pane.sum():  # 칸 거의 전부 '다름' = 배경 추정 실패 → 국소만
            g |= gp
    glare = glare_mask(L, glass, cfg)
    excluded = glare | lines
    c = cv2.morphologyEx(((g | local) & ~excluded).astype(np.uint8), cv2.MORPH_OPEN, _disk(cc["open_px"]))
    c = cv2.morphologyEx(c, cv2.MORPH_CLOSE, _disk(cc["close_px"]))
    return c.astype(bool) & glass & ~excluded, de_g, excluded


def region_features(m: np.ndarray, glass: np.ndarray, img_lab: np.ndarray, ref: np.ndarray | None, cfg: dict) -> dict:
    """유형 규칙에 쓰는 특징값 (모두 debug용으로 결과에 남길 수 있게 dict)."""
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    big = lbl == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cs, hier = cv2.findContours(big.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    outer = [c for c, h in zip(cs, hier[0]) if h[3] < 0]
    holes = sum(cv2.contourArea(c) for c, h in zip(cs, hier[0]) if h[3] >= 0)
    o = max(outer, key=cv2.contourArea)
    outer_area = max(cv2.contourArea(o), 1.0)
    hull = max(cv2.contourArea(cv2.convexHull(o)), 1.0)
    mo = cv2.moments(m.astype(np.uint8), binaryImage=True)
    mu20, mu02, mu11 = mo["mu20"], mo["mu02"], mo["mu11"]
    tr, det = mu20 + mu02, mu20 * mu02 - mu11 ** 2
    disc = np.sqrt(max(tr * tr / 4 - det, 0))
    l1, l2 = tr / 2 + disc, max(tr / 2 - disc, 1e-6)
    theta = np.degrees(0.5 * np.arctan2(2 * mu11, mu20 - mu02))  # 주축과 x축 각 (−90~90)
    ys, xs = np.nonzero(glass)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    eb = cfg["grime"]["edge_band"]
    yy, xx = np.nonzero(m)
    band = (yy >= y1 - eb * (y1 - y0)) | (xx <= x0 + eb * (x1 - x0)) | (xx >= x1 - eb * (x1 - x0))
    px = img_lab[m]
    inner = cv2.erode(m.astype(np.uint8), _disk(3)).astype(bool)  # 경계 픽셀(흐림·보간으로 섞인 색) 빼고 평균
    mean = img_lab[inner if inner.any() else m].mean(0)
    return {
        "components": n - 1,
        "solidity": float(cv2.contourArea(o) / hull),
        "ring": float(holes / outer_area),
        "elongation": float(np.sqrt(l1 / l2)),
        "vertical_dev": float(abs(abs(theta) - 90)),
        "edge_frac": float(band.mean()),
        "dL": None if ref is None else float(mean[0] - ref[0]),
        "highlight": float((px[:, 0] >= cfg["wet"]["highlight_L"]).mean()),
        "delta_e": None if ref is None else float(np.linalg.norm(mean - ref)),
        "chroma_gain": None if ref is None else float(np.hypot(*mean[1:]) - np.hypot(*ref[1:])),
        "dL_mean": None if ref is None else float(mean[0] - ref[0]),
        "chroma": float(np.hypot(*mean[1:])),
    }


def is_wet(f: dict, cfg: dict) -> bool:
    w = cfg["wet"]
    return f["components"] >= w["min_components"] and f["highlight"] >= w["highlight_ratio"]


def classify(f: dict, area_ratio: float, cfg: dict) -> str:
    """규칙 위에서부터 첫 일치: heavy → grime → stain → dust."""
    h, g, s = cfg["heavy"], cfg["grime"], cfg["stain"]
    if (f["delta_e"] is not None and f["delta_e"] >= h["delta_e"] and f["solidity"] >= h["solidity"]
            and area_ratio >= h["min_area_ratio"] and f["components"] <= h["max_components"]
            and f["elongation"] <= h["max_elongation"]):
        return "heavy"
    if f["edge_frac"] >= g["edge_frac"] and f["dL"] is not None and f["dL"] < g["max_dL"]:
        return "grime"
    if (f["elongation"] >= s["elongation"] and f["vertical_dev"] <= s["vertical_tol_deg"]) or f["ring"] >= s["ring_hole"]:
        return "stain"
    return "dust"


def scene_structure(crop: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """반사 의심 판정용: (강한 윤곽 bool — Canny, 긴 직선 bool — LSD [R32] 선분 중 짧은 변 × line_len_ratio 이상을 그린 것).
    유리에 비친 나무·건물은 윤곽·직선이 많고, 유리 위 오염(먼지·얼룩)은 흐릿해 적다."""
    rc = cfg["reflection"]
    gray = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (0, 0), 1.0)
    edges = cv2.Canny(gray, rc["canny_low"], rc["canny_high"]) > 0
    lines = np.zeros(gray.shape, np.uint8)
    segs = cv2.createLineSegmentDetector().detect(gray)[0]
    if segs is not None:
        min_len = rc["line_len_ratio"] * min(gray.shape)
        for x1, y1, x2, y2 in segs.reshape(-1, 4):
            if np.hypot(x2 - x1, y2 - y1) >= min_len:
                cv2.line(lines, (round(x1), round(y1)), (round(x2), round(y2)), 1, 1)
    return edges, lines.astype(bool)


def is_reflection(m: np.ndarray, g: np.ndarray, a_g: int, edges: np.ndarray, lines: np.ndarray, cfg: dict) -> bool:
    """넓고 (A_g × min_area_ratio 이상) 묶음 안쪽에 강한 윤곽 밀도 또는 긴 직선 픽셀 밀도가 높으면 반사 의심.
    영역 자신의 외곽선(덩어리·띠의 가장자리도 윤곽·직선이다)은 세지 않도록 interior_erode_px만큼 깎은 안쪽만 센다."""
    rc = cfg["reflection"]
    if m.sum() < rc["min_area_ratio"] * a_g:
        return False
    inner = cv2.erode(g.astype(np.uint8), _disk(2 * rc["interior_erode_px"] + 1)).astype(bool)  # 반지름 → 지름
    area = inner.sum()
    if area == 0:
        return False
    return edges[inner].sum() / area >= rc["min_edge_density"] or lines[inner].sum() / area >= rc["min_line_density"]


def suspect_reason(m: np.ndarray, f: dict, a_g: int, cfg: dict) -> str | None:
    """색·크기 기반 반사 의심 사유 (윤곽·직선 밀도 규칙 is_reflection은 따로): 'chroma' = 채도가 기준보다 크게 오른 번짐
    (비친 나무·하늘), 'large' = 국소 후보 하나가 유리의 큰 부분 (실내 벽·창턱 — 기준색 실패). 아니면 None."""
    cg, mr = cfg["reflection"]["min_chroma_gain"], cfg["candidate"]["max_region_ratio"]
    if cg is not None and f["chroma_gain"] is not None and f["chroma_gain"] >= cg:
        return "chroma"
    if mr is not None and m.sum() > mr * a_g:
        return "large"
    return None


def sharp_white(m: np.ndarray, grad: np.ndarray, f: dict, a_g: int, cfg: dict) -> bool:
    """반사 의심 예외: glare.keep_sharp 조건 (윤곽 날카로움·덩어리) + 흰색 계열 (기준보다 밝고 채도 ≤ glare.C_max) → 새똥 등 흰 오염.
    면적 상한은 keep_sharp.exception_max_area (null = 없음)."""
    ks, gc = cfg["glare"]["keep_sharp"], cfg["glare"]
    if not ks["reflection_exception"] or f["dL_mean"] is None or f["dL_mean"] <= 0:
        return False
    if ks["exception_max_area"] is not None and m.sum() > ks["exception_max_area"] * a_g:
        return False
    if f["solidity"] < ks["min_solidity"]:
        return False
    ring = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~m
    return bool(ring.any() and grad[ring].mean() >= ks["min_edge"] and f["chroma"] <= gc["C_max"])


def film_map(crop: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """유리 전체 먼지막(film) 정도 맵과 film이 없을 때의 추정 이미지 대비 ΔE 맵 (둘 다 (H, W) float).
    veil: 어두운 채널 사전 [R34] — haze = 1 − t, t = 1 − ω·dark(I/A). ΔE = |I − J|, J = (I − A)/max(t, t0) + A.
    texture: 미세 질감 밀도 — |Laplacian| (회색, 0~255)의 국소 평균. ΔE = I와 작은 median (점·줄 제거) 대비."""
    fc = cfg["film"]
    img = crop.astype(np.float32) / 255
    if fc["metric"] == "veil":
        k = np.ones((fc["patch_px"], fc["patch_px"]), np.uint8)
        dark = cv2.erode(img.min(2), k)
        top = dark >= np.quantile(dark, 0.999)
        A = np.maximum(img[top].mean(0), 1e-3)  # 대기광: 어두운 채널 상위 0.1% 픽셀의 평균색 [R34]
        t = 1 - fc["omega"] * cv2.erode((img / A).min(2), k)
        J = np.clip((img - A) / np.maximum(t, fc["t0"])[..., None] + A, 0, 1)
        de = np.linalg.norm(lab((img * 255).astype(np.uint8)) - lab((J * 255).astype(np.uint8)), axis=2)
        return 1 - t, de
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dens = cv2.blur(np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3)), (fc["patch_px"], fc["patch_px"]))
    de = np.linalg.norm(lab(crop) - lab(cv2.medianBlur(crop, 5)), axis=2)
    return dens, de


def analyze_crop(crop: np.ndarray, occlusion: np.ndarray | None = None, cfg: dict | None = None,
                 grading: dict | None = None, exclude: np.ndarray | None = None,
                 reference: np.ndarray | None = None) -> Analysis:
    """정면화 창문 BGR (+ 가림 bool) → 오염 영역·CSI. 크면 rectify.max_side로 줄여서 분석한다.
    exclude: 오염 후보·깨끗한 유리 기준에서 뺄 픽셀 (다시점 모드: 시점에 따라 바뀌는 픽셀 = 반사·실내). crop과 같은 크기.
    reference: 같은 위치의 깨끗한 사진을 crop에 정렬한 것 (crop과 같은 크기) — 후보 배경·기준색으로 쓴다."""
    cfg = cfg or load_config("contamination")
    grading = grading or load_config("grading")
    if max(crop.shape[:2]) > cfg["rectify"]["max_side"]:
        crop = resize_max(crop, cfg["rectify"]["max_side"])
        if reference is not None:
            reference = cv2.resize(reference, crop.shape[1::-1], interpolation=cv2.INTER_AREA)
        if occlusion is not None:
            occlusion = cv2.resize(occlusion.astype(np.uint8), crop.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0
    h, w = crop.shape[:2]
    frame = glass_mask((h, w), grading["frame_margin"])
    glass = frame & ~occlusion if occlusion is not None else frame
    a_g = int(glass.sum())
    if a_g == 0:
        raise ValueError("glass area is 0 (fully occluded or frame_margin too large)")
    cand, de_map, glare = candidates(crop, glass, cfg, reference)
    if exclude is not None:  # 반사광·창살 제외 마스크와 같이 다룬다 (overlay 흰 외곽)
        cand &= ~exclude
        glare = glare | (exclude & glass)
    img = lab(crop)
    clean = glass & ~cand & ~glare
    sc = grading["severity"]
    if reference is not None:  # 기준색 = 깨끗한 사진의 유리 (반사광 제외)
        ref = reference_lab(lab(reference), glass & ~glare, cfg["reference"]["l_quantiles"])
    else:
        ref = reference_lab(img, clean, cfg["reference"]["l_quantiles"]) if clean.sum() >= sc["clean_min_ratio"] * a_g else None

    refl = cfg["reflection"]["enabled"]
    edges, lines = scene_structure(crop, cfg) if refl else (None, None)
    grad = np.hypot(cv2.Sobel(img[..., 0], cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(img[..., 0], cv2.CV_32F, 0, 1, ksize=3)) / 8
    suspects: list[Found] = []
    merge = _disk(round(min(h, w) * cfg["dust"]["merge_ratio"]))
    groups = cv2.connectedComponents(cv2.dilate(cand.astype(np.uint8), merge))[1]
    found: list[Found] = []
    for i in range(1, groups.max() + 1):
        m = cand & (groups == i)
        g = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, merge).astype(bool) & glass | m  # 묶음 외곽 (팽창 후 원래 크기로)
        if m.sum() < cfg["candidate"]["min_area_ratio"] * a_g:
            continue
        f = region_features(m, glass, img, ref, cfg)
        if is_wet(f, cfg):
            continue
        de = f["delta_e"]
        r = Found(classify(f, m.sum() / a_g, cfg), m, g, None if de is None else round(de, 2),
                  severity_level(de, grading) if de is not None else sc["max"], f)
        why = suspect_reason(m, f, a_g, cfg) or ("structure" if refl and is_reflection(m, g, a_g, edges, lines, cfg) else None)
        r.features["suspect"] = why
        (suspects if why and not sharp_white(m, grad, f, a_g, cfg) else found).append(r)
    if cfg["film"]["enabled"]:  # 먼지막: 덮인 비율만큼 dust 영역 1개 (국소 영역·반사 의심·반사광 밖)
        fm, fde = film_map(crop, cfg)
        free = glass & ~glare
        for r in found + suspects:
            free &= ~r.group
        film = cv2.morphologyEx(((fm >= cfg["film"]["min"]) & free).astype(np.uint8), cv2.MORPH_OPEN, _disk(cfg["candidate"]["open_px"]))
        film = cv2.morphologyEx(film, cv2.MORPH_CLOSE, _disk(cfg["candidate"]["close_px"])).astype(bool) & free
        if film.sum() >= cfg["candidate"]["min_area_ratio"] * a_g:
            de = float(np.median(fde[film]))
            found.append(Found("dust", film, film, round(de, 2), severity_level(de, grading), {"film": True}))
    regions = [Region(r.type, float(r.mask.sum() / a_g), r.severity, r.delta_e) for r in found]
    csi = compute_csi(regions, grading)
    level, label = cleaning_level(csi, grading)
    occ = float((frame & occlusion).sum() / (h * w)) if occlusion is not None else 0.0
    return Analysis(crop, glass, occ, found, csi, level, label, de_map, glare, suspects)


def polygon_uv(group: np.ndarray, eps: float) -> list[list[float]]:
    """묶음 영역 가장 큰 외곽 → 단순화 polygon (u, v) 0~1."""
    h, w = group.shape
    cs, _ = cv2.findContours(group.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)
    p = cv2.approxPolyDP(c, eps * cv2.arcLength(c, True), True).reshape(-1, 2)
    if len(p) < 3:
        x, y, bw, bh = cv2.boundingRect(c)
        p = np.array([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]])
    return [[round(float(x) / w, 4), round(float(y) / h, 4)] for x, y in np.clip(p, 0, [w, h])]


def window_fields(a: Analysis, mode: str, low_confidence: bool, cfg: dict, grading: dict) -> dict:
    """result.json 창문 필드 (schema.Window의 분석 부분)."""
    h, w = a.crop.shape[:2]
    a_g = int(a.glass.sum())
    regions = [{"type": r.type, "area_ratio": round(float(r.mask.sum() / a_g), 5), "severity": r.severity,
                "delta_e": r.delta_e, "polygon": polygon_uv(r.group, cfg["polygon_eps"]),
                "area_px": int(r.mask.sum()), "k_t": grading["weights"][r.type]} for r in a.regions]
    ratios = {t: round(sum(r["area_ratio"] for r in regions if r["type"] == t), 5) for t in TYPES}
    return {"cleaning_level": a.level, "cleaning_label": a.label, "csi": round(a.csi, 5),
            "glass_ratio": round(a_g / (h * w), 4), "occluded_ratio": round(a.occluded_ratio, 4),
            "contamination": regions, "type_ratios": ratios, "glass_area_px": a_g,
            "raw_coverage_pct": round(min(sum(r["area_ratio"] for r in regions) * 100, 100.0), 2),
            "weighted_coverage_pct": round(a.csi * grading["severity"]["max"] * k_t_max(grading) * 100, 2),
            "low_confidence": low_confidence, "analysis_mode": mode,
            "reflection_suspects": [{"area_ratio": round(float(r.mask.sum() / a_g), 5),
                                     "polygon": polygon_uv(r.group, cfg["polygon_eps"])} for r in a.suspects]}


def _bgr(hex_color: str) -> tuple[int, int, int]:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return b, g, r


def draw_overlay(a: Analysis, cfg: dict) -> np.ndarray:
    """crop 위에 유형 색으로 오염 픽셀 + 묶음 외곽선, 제외한 반사광·창살은 흰 외곽, 반사 의심은 노란 굵은 외곽."""
    out = a.crop.copy()
    paint = out.copy()
    for r in a.regions:
        paint[r.mask] = _bgr(cfg["type_colors"][r.type])
    out = cv2.addWeighted(paint, cfg["overlay_alpha"], out, 1 - cfg["overlay_alpha"], 0)
    for r in a.regions:
        cs, _ = cv2.findContours(r.group.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, _bgr(cfg["type_colors"][r.type]), 1, cv2.LINE_AA)
    cs, _ = cv2.findContours(a.glare.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cs, -1, (255, 255, 255), 1, cv2.LINE_AA)
    for r in a.suspects:  # 반사 의심 (CSI 제외): 노란 외곽
        cs, _ = cv2.findContours(r.group.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, (0, 220, 255), 2, cv2.LINE_AA)
    return out


def heatmap_rgba(a: Analysis, cfg: dict) -> np.ndarray:
    """ΔE 맵 → BGRA (유리 밖 투명, 진할수록 불투명) — 웹에서 crop 위에 겹친다."""
    t = np.clip(a.delta_e_map / cfg["heatmap_max_delta_e"], 0, 1)
    rgb = cv2.applyColorMap((t * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    alpha = (t * 255 * a.glass).astype(np.uint8)
    return np.dstack([rgb, alpha])


def save_window_images(a: Analysis, d: Path, cfg: dict, jpeg_quality: int = 92) -> None:
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "crop.jpg"), a.crop, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    cv2.imwrite(str(d / "heatmap.png"), heatmap_rgba(a, cfg))
    cv2.imwrite(str(d / "overlay.png"), draw_overlay(a, cfg))


def _quad_area(c) -> float:
    return float(cv2.contourArea(np.asarray(c, np.float32).reshape(-1, 1, 2)))


def analyze_run(run_dir: str | Path, cfg: dict | None = None, grading: dict | None = None, single: bool = False,
                references: dict[str, np.ndarray] | None = None) -> dict:
    """runs/<id>/result.json의 창문을 분석해 같은 파일을 갱신. report(dict) 반환.

    recon/window_obs.json ({"frames": 폴더(run_dir 기준), "windows": {id: [{frame, corners, occlusion?}]}})이 있으면
    - 다시점 모드 (기본): 관측 ≥ multiview.min_views면 정렬 스택 중앙값으로 분석하고 시점마다 바뀌는 픽셀은 제외
      (multiview.py). 정렬 실패·관측 부족이면 단일 모드로 내려간다 (사유별 개수를 report에)
    - 단일 모드 (single=True 또는 대안): 가장 큰 (= 가장 가깝고 정면) 관측 1장 — low_confidence
    없으면 기존 windows/<id>/crop.jpg (외벽 정사영에서 자른 저해상도)로 단일 분석하고 경고를 남긴다.
    references: {프레임 이름: 그 프레임에 정렬한 깨끗한 사진} — 단일 모드에서 같은 코너로 정면화해 기준으로 쓴다 (photos --reference).
    """
    from facade_cv.contamination.multiview import stack_views

    cfg = cfg or load_config("contamination")
    grading = grading or load_config("grading")
    run_dir = Path(run_dir)
    res = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    obs_path = run_dir / "recon" / "window_obs.json"
    obs = json.loads(obs_path.read_text(encoding="utf-8")) if obs_path.exists() else {"frames": None, "windows": {}}
    frames = run_dir / obs["frames"] if obs["frames"] else None

    @lru_cache(maxsize=32)
    def frame(name: str) -> np.ndarray:
        img = cv2.imread(str(frames / name))
        if img is None:
            raise FileNotFoundError(frames / name)
        return img

    stats = {"windows": len(res["windows"]), "analyzed": 0, "multi": 0, "single_from_frames": 0, "from_ortho_crop": 0,
             "too_small": 0, "multi_fallback": {}}
    times = []
    for w in res["windows"]:
        t = time.perf_counter()
        views = obs["windows"].get(w["id"], [])
        mode, exclude, mv, ref_crop = "single", None, None, None
        if views and frames is not None and not single:
            mv = stack_views(frame, views, cfg, grading["frame_margin"])
            if not mv.ok:
                stats["multi_fallback"][mv.reason] = stats["multi_fallback"].get(mv.reason, 0) + 1
        if mv is not None and mv.ok:
            crop, occ, exclude, mode = mv.median, mv.occlusion, mv.unstable, "multi"
            stats["multi"] += 1
        elif views and frames is not None:
            v = max(views, key=lambda v: _quad_area(v["corners"]))
            crop, M = rectify_quad(frame(v["frame"]), v["corners"], cfg["rectify"]["max_side"])
            occ = warp_polygons(v.get("occlusion"), M, crop.shape[:2])
            if references and v["frame"] in references:
                ref_crop = rectify_quad(references[v["frame"]], v["corners"], cfg["rectify"]["max_side"])[0]
            stats["single_from_frames"] += 1
        else:
            crop = cv2.imread(str(run_dir / w["images"]["crop"]))
            occ = None
            stats["from_ortho_crop"] += 1
        if crop is None or min(crop.shape[:2]) < cfg["rectify"]["min_side"]:
            stats["too_small"] += 1  # 미분석으로 남긴다
            continue
        a = analyze_crop(crop, occ, cfg, grading, exclude, ref_crop)
        d = run_dir / "windows" / w["id"]
        save_window_images(a, d, cfg)
        w.update(window_fields(a, mode, mode == "single", cfg, grading))
        w["clip_dirty"] = clip_dirty([a.crop], cfg)[0]  # 참고 점수 [R35] (CSI·level 무관)
        if mode == "multi":
            w["num_views"] = mv.n_used  # 중앙값 스택에 실제로 쓴 관측 수 (recon의 관측 수 대신)
        w["images"] = {"crop": f"windows/{w['id']}/crop.jpg", "heatmap": f"windows/{w['id']}/heatmap.png",
                       "overlay": f"windows/{w['id']}/overlay.png"}
        stats["analyzed"] += 1
        times.append((time.perf_counter() - t) * 1000)

    res["meta"]["csi_params"] = csi_params(grading)
    counts = {str(lv): sum(w["cleaning_level"] == lv for w in res["windows"]) for lv in range(1, 6)}
    counts["unanalyzed"] = sum(w["cleaning_level"] is None for w in res["windows"])
    res["summary"]["level_counts"] = {k: v for k, v in counts.items() if v or k != "unanalyzed"}
    st = res["meta"].get("recon_status")
    if st is not None:
        msgs = []
        if stats["from_ortho_crop"]:
            msgs.append(f"contamination_low_res: {stats['from_ortho_crop']} windows analyzed from ortho crop (no window_obs)")
        if stats["multi_fallback"]:
            msgs.append("contamination_single_fallback: " + ", ".join(f"{k} {v}" for k, v in stats["multi_fallback"].items()))
        st["warnings"] = [x for x in st["warnings"] if not x.startswith("contamination_")] + msgs
    result = InspectionResult.model_validate(res)
    (run_dir / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    stats |= {"level_counts": counts, "ms_per_window": round(float(np.mean(times)), 1) if times else None}
    return stats


def align_reference(dirty: np.ndarray, clean: np.ndarray, cfg: dict) -> np.ndarray | None:
    """깨끗한 사진을 오염 사진에 ECC 호모그래피 [R31]로 정렬 (사진 전체로 상관). 실패·상관 < multiview.ecc_min이면 None."""
    from facade_cv.contamination.multiview import ecc_align

    clean = cv2.resize(clean, dirty.shape[1::-1], interpolation=cv2.INTER_AREA) if clean.shape != dirty.shape else clean
    out, corr = ecc_align(cv2.cvtColor(dirty, cv2.COLOR_BGR2GRAY), clean, cfg)
    return out if out is not None and corr >= cfg["multiview"]["ecc_min"] else None


def reference_truth(dirty: np.ndarray, aligned_clean: np.ndarray, cfg: dict, grading: dict) -> np.ndarray:
    """정답 오염 마스크 = 정렬한 깨끗한 사진 대비 ΔE ≥ grading 심각도 첫 경계 (지각 가능한 차이 [R23]) — 흐림·열림·닫힘·작은 조각 제거는
    후보 검출과 같은 값. 반사광(밝고 무채색)은 뺀다. (H, W) bool, 원본 크기."""
    cc = cfg["candidate"]
    blur = (lambda x: cv2.GaussianBlur(x, (0, 0), cc["blur_sigma"])) if cc["blur_sigma"] > 0 else (lambda x: x)
    L = lab(blur(dirty))
    de = np.linalg.norm(L - lab(blur(aligned_clean)), axis=2)
    full = np.ones(de.shape, bool)
    m = (de >= grading["severity"]["delta_e"][0]) & ~glare_mask(L, full, cfg)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, _disk(cc["open_px"]))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _disk(cc["close_px"]))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m)
    keep = np.flatnonzero(st[1:, cv2.CC_STAT_AREA] >= cc["min_area_ratio"] * de.size) + 1
    return np.isin(lbl, keep)


def mask_polygons(m: np.ndarray, eps: float) -> list[list[list[float]]]:
    """bool 마스크 → 외곽 polygon (u, v) 0~1 목록 (evaluate --labels 형식)."""
    h, w = m.shape
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        p = cv2.approxPolyDP(c, eps * cv2.arcLength(c, True), True).reshape(-1, 2)
        if len(p) >= 3:
            out.append([[round(float(x) / w, 4), round(float(y) / h, 4)] for x, y in p])
    return out


def roi_window(roi, w: int, h: int) -> tuple[np.ndarray, list[list[list[float]]]]:
    """ROI ([u0, v0, u1, v1] 사각형 또는 [[u, v], ...] 다각형, 0~1) → (창 코너 = ROI 외접 사각형 TL TR BR BL 픽셀,
    가림 polygon 목록 = 외접 사각형 안 ROI 밖 조각들). 다각형은 외접 사각형 네 변에 닿으므로 밖 조각에 구멍이 없다."""
    pts = np.array([[roi[0], roi[1]], [roi[2], roi[1]], [roi[2], roi[3]], [roi[0], roi[3]]] if np.ndim(roi) == 1 else roi, float)
    pts = pts * [w - 1, h - 1]
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    m = np.zeros((h, w), np.uint8)
    cv2.rectangle(m, (round(x0), round(y0)), (round(x1), round(y1)), 1, -1)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 0)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return corners, [c.reshape(-1, 2).astype(float).tolist() for c in cs if len(c) >= 3]


def whole_image_facade(img: np.ndarray, name: str, height: float, max_side: int, roi=None):
    """사진 전체 = 창문 1개 (row 1, col 1)인 가상 외벽. 정면화하지 않는다 (근접 사진엔 외벽 직선이 없다).
    roi: 사진 중 유리 부분 (roi_window 형식) — 창 = ROI 외접 사각형, ROI 밖은 가림.
    (facade, 텍스처 — 긴 변 ≤ max_side로 축소, ppu, [Window3D], T4 형식 검출 1건)."""
    from facade_cv.recon3d.geometry import FacadeFrame
    from facade_cv.recon3d.windows3d import LiftedObservation, Window3D, WindowObservation

    h, w = img.shape[:2]
    tex = resize_max(img, max_side)
    ppu = tex.shape[0] / height
    facade = FacadeFrame("F1", np.array([0.0, 0.0, 1.0]), np.zeros(3), tex.shape[1] / ppu, height)
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float)  # TL TR BR BL (원본 픽셀)
    occ: list = []
    if roi is not None:
        corners, occ = roi_window(roi, w, h)
    fx, fy = facade.width / w, height / h  # 원본 픽셀 → 외벽 단위 (v는 위로)
    rect = np.array([corners[0, 0] * fx, (h - corners[2, 1] - 1) * fy, (corners[1, 0] + 1) * fx, (h - corners[0, 1]) * fy])  # 픽셀 경계 기준
    win = Window3D("F1", rect, [LiftedObservation(WindowObservation(name, corners, 1.0), "F1", rect)], row=1, col=1)
    det = {"image": name, "counts": {"final": 1},
           "windows": [{"corners": corners.tolist(), "conf": 1.0, "row": 0, "col": 0, "occlusion": occ}]}
    return facade, tex, ppu, [win], det


def photos_to_run(images: str | Path, out: str | Path, weights: str | Path | None = None,
                  device: str | None = None, whole_image: bool = False, roi: dict | None = None,
                  reference: str | Path | None = None) -> dict:
    """사진 폴더 → 사진마다 T4 창문 검출 → 사진 1장 = 가상 외벽 1개 (degrade 경로 재사용) → result.json → 창별 분석.
    3D 복원이 아니다: 외벽 판들을 옆으로 늘어놓은 메시라 웹에서 b04와 같은 화면으로 볼 수 있게 하는 시연용.
    whole_image: 창문 검출 없이 사진 전체를 유리 1장으로 (차 앞유리·유리 근접 사진처럼 창문 검출이 안 되는 사진용).
    roi: {파일명: 유리 ROI (roi_window 형식)} — whole_image에서만. 없는 사진은 전체.
    reference: 같은 위치의 깨끗한 사진 (images 폴더 안이면 분석에서 뺀다). 사진마다 ECC로 정렬해 기준으로 쓰고,
      정렬한 깨끗한 사진 대비 차이로 정답 오염 마스크를 만들어 <out>/reference_labels.json (evaluate --labels 형식)과
      <out>/reference_truth/<사진>.png에 저장, 단일 모드(기준 없음)·기준 모드 검출의 면적 IoU를 report에 남긴다."""
    import trimesh

    from facade_cv.detect.datasets.yolo import list_images
    from facade_cv.detect.postprocess import postprocess
    from facade_cv.detect.segmenter import WindowSegmenter
    from facade_cv.recon3d.degrade import degrade_facade
    from facade_cv.recon3d.export import export_result
    from facade_cv.recon3d.mesh import build_mesh
    from facade_cv.recon3d.run import write_window_obs

    images, out = Path(images), Path(out)
    dcfg, rcfg = load_config("detect"), load_config("recon")
    seg = None if whole_image else WindowSegmenter(weights, dcfg, device)
    facades, bodies, textures, windows, det_all, skipped = [], [], {}, [], {"images": []}, []
    cfg, grading = load_config("contamination"), load_config("grading")
    clean = cv2.imread(str(reference)) if reference else None
    if reference and clean is None:
        raise FileNotFoundError(reference)
    refs, truth, ious, unaligned = {}, {}, {}, []
    x = 0.0
    for p in list_images(images):
        if reference and p.resolve() == Path(reference).resolve():
            continue
        img = cv2.imread(str(p))
        if clean is not None:
            al = align_reference(img, clean, cfg)
            if al is None:
                unaligned.append(p.name)
            else:
                refs[p.name] = al
                gt = reference_truth(img, al, cfg, grading)
                truth[p.name] = gt
                ious[p.name] = {k: _iou(analyze_crop(img, None, cfg, grading, reference=r), gt)
                                for k, r in (("single", None), ("reference", al))}
        if whole_image:
            f, tex, ppu, wins, det = whole_image_facade(img, p.name, rcfg["degrade"]["facade_height"],
                                                        rcfg["ortho"]["max_side_px"], (roi or {}).get(p.name))
        else:
            res = postprocess(img, seg.predict(img, conf=dcfg["conf"] * dcfg["fill_conf_ratio"]), dcfg)
            if not res.windows:
                skipped.append(p.name)
                continue
            det = {"image": p.name, "counts": {"final": len(res.windows)},
                   "windows": [{"corners": w.corners.tolist(), "conf": w.conf, "row": w.row, "col": w.col,
                                "occlusion": [q.tolist() for q in w.occlusion]} for w in res.windows]}
            f, tex, ppu, wins, _ = degrade_facade({"images": [det]}, images, rcfg["degrade"])
        det_all["images"].append(det)
        fid = f"F{len(facades) + 1}"
        f = replace(f, id=fid, origin=np.array([x, 0.0, 0.0]))
        body, (f,), _ = build_mesh([f], rcfg["mesh"])
        x += f.width * 1.15
        facades.append(f)
        bodies.append(body)
        textures[fid] = (tex, ppu)
        for w in wins:
            w.facade_id = fid
        windows += wins
    if not facades:
        raise ValueError(f"no windows detected in {images}")
    status = {"mode": "degraded", "num_facades": len(facades), "registered_ratio": None,
              "warnings": ["photos: 사진마다 가상 외벽 1개 (3D 복원 아님, 시연용)"]
              + (["whole_image: 창문 검출 없이 사진 전체를 유리로 분석"] if whole_image else [])
              + ([f"roi: {len(roi)}장 유리 ROI 지정"] if roi else [])
              + ([f"reference: 깨끗한 사진 {Path(reference).name} 기준 (정렬 실패 {len(unaligned)}장)"] if reference else [])
              + ([f"no_windows: {', '.join(skipped)}"] if skipped else [])}
    meta = {"video": str(images), "fps": 1.0, "frames_used": len(facades)}
    export_result(out, out.name, meta, trimesh.util.concatenate(bodies), facades, textures, windows, status,
                  rcfg["ortho"]["jpeg_quality"])
    write_window_obs(out, images, windows, det_all)
    rep = analyze_run(out, cfg, grading, references=refs) | {"photos": len(facades), "skipped": skipped}
    if reference:
        (out / "reference_truth").mkdir(exist_ok=True)
        labels = {}
        for n, gt in truth.items():
            cv2.imwrite(str(out / "reference_truth" / f"{Path(n).stem}.png"), gt.astype(np.uint8) * 255)
            labels[str(images / n)] = [{"class": "dirt", "polygon": poly} for poly in mask_polygons(gt, cfg["polygon_eps"])]
        (out / "reference_labels.json").write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
        rep |= {"reference_iou": ious, "reference_unaligned": unaligned}
    return rep


def _iou(a: "Analysis", gt: np.ndarray) -> float:
    """분석 검출(모든 오염 영역) vs 정답 마스크 (원본 크기 → 분석 크기로 줄여서) 면적 IoU, 유리 안."""
    det = np.zeros(a.glass.shape, bool)
    for r in a.regions:
        det |= r.mask
    g = cv2.resize(gt.astype(np.uint8), det.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool) & a.glass
    return round(float((det & g).sum() / max((det | g).sum(), 1)), 3)


app = typer.Typer(add_completion=False)


@app.command()
def run(run_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="runs/<id> (result.json 있는 폴더)"),
        single: bool = typer.Option(False, help="다시점 정렬 없이 창문마다 가장 큰 관측 1장만 (비교용)")) -> None:
    """result.json 창문 오염 분석 (같은 파일 갱신)."""
    print(json.dumps(analyze_run(run_dir, single=single), ensure_ascii=False))


@app.command()
def image(path: Path = typer.Argument(..., exists=True, dir_okay=False),
          out: Path = typer.Option(None, help="crop/heatmap/overlay 저장 폴더")) -> None:
    """사진 1장 = 유리 전체로 보고 분석."""
    cfg, grading = load_config("contamination"), load_config("grading")
    a = analyze_crop(cv2.imread(str(path)), None, cfg, grading)
    f = window_fields(a, "single", True, cfg, grading)
    if out:
        save_window_images(a, out, cfg)
    print(f"CSI {f['csi']:.4f} (가중 오염 면적률 {f['weighted_coverage_pct']}%) → {a.level} {a.label}")
    for r in f["contamination"]:
        print(f"  {r['type']:6s} {r['area_ratio']:.4f} ΔE {r['delta_e']} k_s {r['severity']}")


@app.command()
def photos(images: Path = typer.Option(..., exists=True, file_okay=False, help="창문 사진 폴더 (근접·원거리)"),
           out: Path = typer.Option(..., help="runs/demo_<이름>"),
           weights: Path = typer.Option(None, help="기본: config/detect.yaml weights"),
           device: str = typer.Option(None),
           whole_image: bool = typer.Option(False, "--whole-image", help="창문 검출 없이 사진 전체를 유리로 (차 앞유리 등 근접 사진)"),
           roi: Path = typer.Option(None, exists=True, dir_okay=False,
                                    help='--whole-image용 유리 ROI json {"파일명": [u0,v0,u1,v1] 또는 [[u,v],...]} (0~1)'),
           reference: Path = typer.Option(None, exists=True, dir_okay=False,
                                          help="같은 위치의 깨끗한 사진 — 기준색·배경 + 정답 오염 마스크 (pairs/<세트>/clean.jpg)")) -> None:
    """사진 폴더 → T4 검출 → 창별 단일 모드 분석 → runs/demo_<이름>/result.json (웹에서 그대로 보기)."""
    rois = json.loads(roi.read_text(encoding="utf-8")) if roi else None
    if rois and not whole_image:
        raise typer.BadParameter("--roi는 --whole-image와 함께")
    print(json.dumps(photos_to_run(images, out, weights, device, whole_image, rois, reference), ensure_ascii=False))


if __name__ == "__main__":
    app()
