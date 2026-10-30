"""다시점 정렬 — 같은 창문을 여러 프레임에서 정면화·정렬해 "시점이 바뀌어도 고정된 패턴"만 남긴다.

오염은 유리 표면에 붙어 있어 정면화하면 모든 관측에서 같은 자리에 있고, 반사(하늘·나무·건물)와 유리 뒤 실내는 드론이
움직이면 시차로 자리가 바뀐다. 그래서
  1) 관측마다 검출 4코너 → 공통 직사각형 호모그래피 (analyze.rectify_quad와 같은 정면화)
  2) 4코너 검출 오차를 ECC [R31] 호모그래피로 보정 — 기준 = 1차 정렬 스택의 중앙값, 상관은 창틀 띠에서만 잰다
     (유리 안은 반사가 시점마다 달라 기준이 못 된다 — b04: 창틀 띠 상관 중앙값 0.79 vs 전체 0.70). 상관이 낮은 관측은 버린다
  3) 픽셀별 중앙값 M (BGR) → 오염 분석 입력. 시점마다 다른 반사는 중앙값에서 대부분 사라진다
  4) 픽셀별 시간 변동 S = 채널별 MAD의 LAB 거리 → S가 큰 픽셀(시점에 따라 바뀜 = 반사·실내)은 오염 후보에서 뺀다
실패 대안 (analyze_run이 처리): 관측 < min_views, ECC 통과 < min_views, 정렬 품질(유리 S 중앙값) > misalign_mad → 단일 모드.
설정: config/contamination.yaml multiview.
"""

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from facade_cv.contamination.analyze import rectify_quad, warp_polygons
from facade_cv.contamination.grade import glass_mask, lab


@dataclass
class MultiView:
    ok: bool
    reason: str | None  # 실패 사유 (ok=False): few_views | ecc | misaligned
    median: np.ndarray | None = None  # 정렬 스택 중앙값 BGR (분석 입력)
    spread: np.ndarray | None = None  # 픽셀별 시간 변동 (LAB MAD 거리)
    unstable: np.ndarray | None = None  # spread ≥ mad_max (반사·실내 의심 → 후보 제외)
    occlusion: np.ndarray | None = None  # 관측 과반에서 가려진 픽셀
    n_used: int = 0
    n_total: int = 0
    glass_spread: float | None = None  # 유리 안 spread 중앙값 (정렬 품질)


def _quad_size(c) -> tuple[float, float]:
    q = np.asarray(c, float).reshape(4, 2)
    return ((np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2,
            (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2)


def target_size(views: list[dict], max_side: int) -> tuple[int, int]:
    """공통 정면화 크기 (W, H) = 관측 사각형 크기의 중앙값, 긴 변 ≤ max_side."""
    w, h = np.median([_quad_size(v["corners"]) for v in views], axis=0)
    s = min(1.0, max_side / max(w, h, 1))
    return max(1, round(w * s)), max(1, round(h * s))


def ecc_align(template: np.ndarray, image: np.ndarray, cfg: dict,
              mask: np.ndarray | None = None) -> tuple[np.ndarray | None, float]:
    """image(BGR)를 template(회색)에 맞추는 호모그래피 ECC [R31] — 축소 영상에서 구해 원래 크기로 환산.
    mask: 상관을 계산할 픽셀 (창틀 띠 — 유리 안은 반사가 시점마다 달라 정렬 기준으로 나쁘다).
    (정렬된 BGR, 상관계수). 수렴 실패면 (None, 0)."""
    mc = cfg["multiview"]
    h, w = template.shape
    s = min(1.0, mc["ecc_max_side"] / max(h, w))
    small = lambda g: cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g  # noqa: E731
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    W = np.eye(3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, mc["ecc_iters"], mc["ecc_eps"])
    try:
        m = None if mask is None else small(mask.astype(np.uint8) * 255)
        corr, W = cv2.findTransformECC(small(template).astype(np.float32), small(gray).astype(np.float32), W,
                                       cv2.MOTION_HOMOGRAPHY, crit, m, mc["ecc_gauss"])
    except cv2.error:
        return None, 0.0
    S = np.diag([s, s, 1.0]).astype(np.float32)
    W = np.linalg.inv(S) @ W @ S
    # ECC는 코너 검출 오차 보정용 — 코너가 크게 움직이는 해는 퇴화(엉뚱한 상관 최대)로 보고 버린다
    c = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
    if np.abs(cv2.perspectiveTransform(c, W.astype(np.float64)) - c).max() > mc["ecc_max_shift"] * min(h, w):
        return None, 0.0
    return cv2.warpPerspective(image, W, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE), float(corr)  # 검은 테두리가 때로 잡히지 않게


def stack_views(load: Callable[[str], np.ndarray], views: list[dict], cfg: dict, frame_margin: float) -> MultiView:
    """관측들 → 정렬 스택 → 중앙값·시간 변동. 실패하면 ok=False와 사유."""
    mc = cfg["multiview"]
    n = len(views)
    if n < mc["min_views"]:
        return MultiView(False, "few_views", n_total=n)
    W, H = target_size(views, cfg["rectify"]["max_side"])
    warped, occs = [], []
    for v in views:
        img, M = rectify_quad(load(v["frame"]), v["corners"], max(W, H))
        warped.append(cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA) if img.shape[1::-1] != (W, H) else img)
        o = warp_polygons(v.get("occlusion"), M, img.shape[:2])
        occs.append(None if o is None else cv2.resize(o.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0)

    template = cv2.cvtColor(np.median(warped, axis=0).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    band = ~glass_mask((H, W), mc["ecc_band"])  # 창틀 띠 (고정 구조)
    aligned, kept_occ = [], []
    for img, o in zip(warped, occs):
        a, corr = ecc_align(template, img, cfg, band)
        if a is not None and corr >= mc["ecc_min"]:
            aligned.append(a)
            kept_occ.append(o)
    if len(aligned) < mc["min_views"]:
        return MultiView(False, "ecc", n_used=len(aligned), n_total=n)

    stack = np.stack(aligned)
    median = np.median(stack, axis=0).astype(np.uint8)
    labs = np.stack([lab(a) for a in aligned])
    spread = np.linalg.norm(np.median(np.abs(labs - np.median(labs, axis=0)), axis=0), axis=2)
    glass = glass_mask((H, W), frame_margin)
    g_spread = float(np.median(spread[glass]))
    occ = None
    if any(o is not None for o in kept_occ):
        votes = np.sum([o for o in kept_occ if o is not None], axis=0)
        occ = votes * 2 > len(kept_occ)
    mv = MultiView(True, None, median, spread, spread >= mc["mad_max"], occ, len(aligned), n, g_spread)
    if g_spread > mc["misalign_mad"]:  # 유리 전체가 흔들림 = 정렬 실패
        mv.ok, mv.reason = False, "misaligned"
    return mv

