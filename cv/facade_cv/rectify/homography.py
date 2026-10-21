"""외벽 정면화 호모그래피 — 수평/수직 직선의 소실점 두 개로 카메라 회전을 추정한다 (detect 후처리, T6 공용).

python -m facade_cv.rectify.homography --image X [--out rectified.jpg]

수평 소실점 v_h, 수직 소실점 v_v, 초점거리 f(K = diag(f, f, 1), 주점 = 이미지 중심)에서
r_h = K⁻¹v_h, r_v = K⁻¹v_v (직교화), R = [r_h r_v r_h×r_v] → H = K Rᵀ K⁻¹ 는 외벽 축을 이미지 축에 맞춘다.
f는 두 소실점이 유한하면 직교 조건 (v_h·v_v = -f²) 으로 추정, 아니면 config 기본값. [R11]
선분 LSD [R12], 소실점 RANSAC [R13] (docs/REFERENCES.md).
"""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import typer

from facade_cv.config import load_config


@dataclass
class Rectification:
    H: np.ndarray  # 3x3, 원본 픽셀 → 정면화 픽셀
    size: tuple[int, int]  # 정면화 이미지 (w, h)
    focal: float  # 픽셀 단위
    focal_estimated: bool  # False면 default_focal_ratio 사용
    lines_h: np.ndarray = field(repr=False)  # 수평 소실점 inlier 선분 (N, 4) x1 y1 x2 y2
    lines_v: np.ndarray = field(repr=False)

    def warp(self, image: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(image, self.H, self.size)

    def apply(self, pts: np.ndarray) -> np.ndarray:
        return cv2.perspectiveTransform(np.asarray(pts, np.float64).reshape(-1, 1, 2), self.H).reshape(-1, 2)


def find_axis_lines(gray: np.ndarray, region: np.ndarray | None, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """LSD 선분 → (수평 후보, 수직 후보) 각 (N, 4). region(bool 마스크)이 있으면 중점이 그 안인 선분만."""
    segs = cv2.createLineSegmentDetector().detect(gray)[0]
    if segs is None:
        return np.empty((0, 4)), np.empty((0, 4))
    segs = segs.reshape(-1, 4).astype(np.float64)
    d = segs[:, 2:] - segs[:, :2]
    keep = np.hypot(*d.T) >= cfg["min_line_ratio"] * max(gray.shape)
    if region is not None:
        mid = ((segs[:, :2] + segs[:, 2:]) / 2).astype(int)
        mid = np.clip(mid, 0, [gray.shape[1] - 1, gray.shape[0] - 1])
        keep &= region[mid[:, 1], mid[:, 0]]
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180  # 0 = 수평, 90 = 수직
    tol = cfg["axis_tol_deg"]
    horiz = (np.minimum(ang, 180 - ang) <= tol) & keep
    vert = (np.abs(ang - 90) <= tol) & keep
    return segs[horiz], segs[vert]


def _vp_errors(segs: np.ndarray, vp: np.ndarray) -> np.ndarray:
    """선분 방향과 (중점→소실점) 방향의 각도 차 (도). vp는 동차좌표 — 무한원점(w≈0)도 된다."""
    mid = (segs[:, :2] + segs[:, 2:]) / 2
    d = segs[:, 2:] - segs[:, :2]
    to_vp = vp[:2] - mid * vp[2]  # w로 나누지 않은 방향 (w 부호에 따라 반대일 수 있어 아래에서 abs)
    cos = np.abs((d * to_vp).sum(1)) / (np.linalg.norm(d, axis=1) * np.linalg.norm(to_vp, axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(cos, 0, 1)))


def _fit_vp(segs: np.ndarray, cfg: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    """RANSAC(선분 두 개의 교점) → inlier 길이 가중 최소제곱. (동차 vp, inlier 마스크) 또는 None.
    segs는 정규화 좌표 (estimate_rectification 참고)."""
    if len(segs) < cfg["min_lines"]:
        return None
    ones = np.ones((len(segs), 1))
    lines = np.cross(np.hstack([segs[:, :2], ones]), np.hstack([segs[:, 2:], ones]))
    lines /= np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    length = np.linalg.norm(segs[:, 2:] - segs[:, :2], axis=1)
    best, best_score = None, 0.0
    for _ in range(cfg["ransac_iters"]):
        i, j = rng.choice(len(segs), 2, replace=False)
        vp = np.cross(lines[i], lines[j])
        if not np.any(vp):
            continue
        inl = _vp_errors(segs, vp) <= cfg["vp_inlier_deg"]
        score = length[inl].sum()
        if score > best_score:
            best, best_score = inl, score
    if best is None or best.sum() < cfg["min_lines"]:
        return None
    vp = np.linalg.svd(lines[best] * length[best, None])[2][-1]
    inl = _vp_errors(segs, vp) <= cfg["vp_inlier_deg"]
    return (vp, inl) if inl.sum() >= cfg["min_lines"] else None


def estimate_rectification(
    image: np.ndarray, region: np.ndarray | None = None, cfg: dict | None = None
) -> Rectification | None:
    """region(외벽 bool 마스크) 안의 직선으로 정면화. 직선 부족·퇴화·과도한 회전이면 None.
    출력 크기는 region(없으면 이미지 전체)을 정면화한 범위가 out_max_side에 맞도록 정한다."""
    cfg = cfg or load_config("rectify")
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    segs_h, segs_v = find_axis_lines(gray, region, cfg)
    # 정규화 좌표 (중심 원점, 긴 변 = 1)에서 계산
    s = float(max(h, w))
    T = np.array([[1 / s, 0, -w / 2 / s], [0, 1 / s, -h / 2 / s], [0, 0, 1]])
    c = np.array([w, h, w, h]) / 2
    rng = np.random.default_rng(0)
    fit_h, fit_v = _fit_vp((segs_h - c) / s, cfg, rng), _fit_vp((segs_v - c) / s, cfg, rng)
    if fit_h is None or fit_v is None:
        return None
    vh, vv = fit_h[0], fit_v[0]
    f, estimated = cfg["default_focal_ratio"], False
    lo, hi = cfg["focal_range"]
    if min(abs(vh[2]), abs(vv[2])) > 1e-9:
        f2 = -np.dot(vh[:2] / vh[2], vv[:2] / vv[2])
        if lo**2 <= f2 <= hi**2:
            f, estimated = float(np.sqrt(f2)), True
    K = np.diag([f, f, 1.0])
    Kinv = np.diag([1 / f, 1 / f, 1.0])
    rh, rv = Kinv @ vh, Kinv @ vv
    rh /= np.linalg.norm(rh) * np.sign(rh[0] or 1)  # 오른쪽 = +x
    rv -= rv.dot(rh) * rh
    if np.linalg.norm(rv) < 1e-6:
        return None
    rv /= np.linalg.norm(rv) * np.sign(rv[1] or 1)  # 아래 = +y
    R = np.column_stack([rh, rv, np.cross(rh, rv)])
    if np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))) > cfg["max_rotation_deg"]:
        return None
    Hn = K @ R.T @ Kinv

    # region 윤곽(없으면 이미지 네 귀퉁이)을 정면화해 출력 범위 결정
    if region is not None and region.any():
        cnts, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pts = cv2.convexHull(np.vstack(cnts)).reshape(-1, 2).astype(np.float64)
    else:
        pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    ph = (Hn @ T @ np.column_stack([pts, np.ones(len(pts))]).T).T
    if np.any(ph[:, 2] <= 1e-9):  # 카메라 뒤로 넘어감
        return None
    xy = ph[:, :2] / ph[:, 2:]
    (x0, y0), (x1, y1) = xy.min(0), xy.max(0)
    k = cfg["out_max_side"] / max(x1 - x0, y1 - y0)
    fit = np.array([[k, 0, -k * x0], [0, k, -k * y0], [0, 0, 1]])
    H = fit @ Hn @ T
    size = (max(1, int(np.ceil(k * (x1 - x0)))), max(1, int(np.ceil(k * (y1 - y0)))))
    return Rectification(H / H[2, 2], size, f * s, estimated, segs_h[fit_h[1]], segs_v[fit_v[1]])


def draw_lines(image: np.ndarray, rect: Rectification | None, cfg: dict | None = None,
               region: np.ndarray | None = None) -> np.ndarray:
    """후보 선분(회색) + 소실점 inlier (수평 파랑, 수직 빨강)."""
    cfg = cfg or load_config("rectify")
    out = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for segs, color in ((np.vstack(find_axis_lines(gray, region, cfg)), (160, 160, 160)),
                        (rect.lines_h if rect else [], (255, 80, 0)), (rect.lines_v if rect else [], (0, 0, 255))):
        for x1, y1, x2, y2 in np.asarray(segs).reshape(-1, 4).astype(int):
            cv2.line(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    return out


def main(image: Path = typer.Option(..., exists=True, dir_okay=False), out: Path = typer.Option(None)) -> None:
    img = cv2.imread(str(image))
    rect = estimate_rectification(img)
    if rect is None:
        raise typer.Exit(f"rectification failed (not enough lines): {image}")
    out = out or image.with_name(f"{image.stem}_rectified.jpg")
    cv2.imwrite(str(out), rect.warp(img))
    print(f"f={rect.focal:.0f}px ({'estimated' if rect.focal_estimated else 'default'}), "
          f"lines h={len(rect.lines_h)} v={len(rect.lines_v)} → {out}")


if __name__ == "__main__":
    typer.run(main)
