"""포인트 클라우드 → 외벽 평면 (facade 후보, Y-up 정렬된 좌표에서).

이상치 제거 → 반복 RANSAC [R13] (detect_planes: Open3D segment_plane 등으로 바꿀 수 있게 이 함수만 의존) →
수직 평면만 facade 후보 → 같은 면 병합 → 보이는 면 1개면 그대로, 2~4개면 공통 직교 격자로 스냅.
격자에 안 맞는 면 (날개·출입구 등)은 실패로 보지 않고 extra로 남겨 보고한다.
임계값은 촬영 거리(scene_scale) 비율 (config/recon.yaml planes).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from facade_cv.recon3d.geometry import UP, CameraView, FacadeFrame


@dataclass
class Plane:
    normal: np.ndarray  # 단위벡터, n·x + d = 0
    d: float
    inliers: np.ndarray  # 포인트 index


@dataclass
class FacadeResult:
    facades: list[FacadeFrame]  # 격자에 맞는 외벽 (≤ max_facades, 법선은 밖을 향함)
    extra: list[dict]  # 보고용: 박스 면이 아닌 수직 평면 (parallel_duplicate는 같은 방향 외벽의 가로·높이 범위에는 보탬) {"normal", "inliers"(수), "reason", "angle_off_deg", "points"(index)}
    planes: list[Plane]  # RANSAC이 찾은 모든 평면 (디버그 색칠)
    kept: np.ndarray  # 이상치 제거 후 남은 포인트 마스크
    scene: float  # scene_scale (촬영 거리)
    ground: float
    top: float
    warnings: list[str] = field(default_factory=list)


def remove_outliers(points: np.ndarray, k: int, std_ratio: float) -> np.ndarray:
    """통계적 이상치 제거 (Open3D remove_statistical_outlier와 같은 규칙). 남길 포인트 마스크."""
    if len(points) <= k:
        return np.ones(len(points), bool)
    dist, _ = cKDTree(points).query(points, k + 1)
    mean = dist[:, 1:].mean(1)
    return mean <= mean.mean() + std_ratio * mean.std()


def scene_scale(points: np.ndarray, cameras: list[CameraView]) -> float:
    """촬영 거리: 포인트에서 가장 가까운 카메라 중심까지 거리의 중앙값 (config/recon.yaml 맨 위 주석)."""
    if not cameras or not len(points):
        return 1.0
    return float(np.median(cKDTree([c.center for c in cameras]).query(points)[0])) or 1.0


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    """최소제곱 평면 (SVD)."""
    c = points.mean(0)
    n = np.linalg.svd(points - c)[2][-1]
    return n, -float(n @ c)


def detect_planes(points: np.ndarray, dist: float, iters: int, min_inliers: int, max_planes: int,
                  seed: int = 0) -> list[Plane]:
    """반복 RANSAC: 가장 큰 평면을 찾아 inlier를 빼고 반복. min_inliers 미만이면 종료.
    ponytail: 점수는 최대 20000개 표본으로 계산 — 포인트 수십만이면 Open3D segment_plane으로 바꿀 자리."""
    rng = np.random.default_rng(seed)
    remaining = np.arange(len(points))
    planes: list[Plane] = []
    while len(planes) < max_planes and len(remaining) >= max(min_inliers, 3):
        P = points[remaining]
        S = P[rng.choice(len(P), min(len(P), 20000), replace=False)]
        best, best_score = None, -1
        for start in range(0, iters, 200):
            m = min(200, iters - start)
            tri = P[rng.integers(0, len(P), (m, 3))]
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            norm = np.linalg.norm(n, axis=1)
            ok = norm > 1e-12
            if not ok.any():
                continue
            n = n[ok] / norm[ok, None]
            d = -np.einsum("ij,ij->i", n, tri[ok, 0])
            score = (np.abs(S @ n.T + d) < dist).sum(0)
            i = int(np.argmax(score))
            if score[i] > best_score:
                best, best_score = (n[i], d[i]), score[i]
        if best is None:
            break
        inl = np.abs(P @ best[0] + best[1]) < dist
        if inl.sum() < min_inliers:
            break
        n, d = fit_plane(P[inl])  # 정제: inlier로 다시 맞추고 다시 inlier
        inl = np.abs(P @ n + d) < dist
        if inl.sum() < min_inliers:
            break
        planes.append(Plane(n, d, remaining[inl]))
        remaining = remaining[~inl]
    return planes


def _horizontal(n: np.ndarray) -> np.ndarray:
    h = np.array([n[0], 0.0, n[2]])
    return h / np.linalg.norm(h)


def _angle(n: np.ndarray) -> float:
    return float(np.arctan2(n[2], n[0]))


def merge_planes(planes: list[Plane], points: np.ndarray, angle_deg: float, offset: float) -> list[Plane]:
    """방향(부호 무시)이 angle_deg 이내이고 평면 거리 차가 offset 미만이면 병합 후 다시 맞춘다."""
    merged: list[Plane] = []
    for p in sorted(planes, key=lambda p: -len(p.inliers)):
        for i, q in enumerate(merged):
            s = 1.0 if p.normal @ q.normal >= 0 else -1.0
            if s * p.normal @ q.normal >= np.cos(np.radians(angle_deg)) and abs(s * p.d - q.d) < offset:
                idx = np.concatenate([q.inliers, p.inliers])
                n, d = fit_plane(points[idx])
                merged[i] = Plane(n, d, idx)
                break
        else:
            merged.append(p)
    return merged


def snap_orthogonal(normals: list[np.ndarray], weights: list[float], tol_deg: float) -> tuple[float, list[int | None], list[float]]:
    """수평 법선들의 공통 직교 격자 각 α와 각 법선의 격자 번호 k (θ ≈ α + k·90°, tol 밖이면 None), 격자에서 벗어난 각(도).
    가장 큰(가중치) 평면이 격자를 정하고, tol 안에 든 평면들만으로 α를 다시 평균한다 (4θ 원형평균 / 4).
    전체를 평균하면 직각이 아닌 면 하나가 α를 끌어당겨 큰 면까지 틀어진다."""
    th = np.array([_angle(n) for n in normals])
    w = np.asarray(weights, float)

    def offsets(alpha: float) -> np.ndarray:
        return (th - alpha + np.pi / 4) % (np.pi / 2) - np.pi / 4  # [-45°, 45°)

    alpha = float(th[np.argmax(w)])
    on = np.abs(offsets(alpha)) <= np.radians(tol_deg)
    alpha = float(np.angle(np.sum(w[on] * np.exp(4j * (th[on] - alpha)))) / 4 + alpha)
    off = offsets(alpha)
    ks = [int(np.round((t - alpha - o) / (np.pi / 2))) % 4 if ok else None for t, o, ok in zip(th, off, on)]
    return alpha, ks, [float(np.degrees(o)) for o in off]


def outward_sign(p: Plane, points: np.ndarray, cameras: list[CameraView], visibility: list[np.ndarray] | None,
                 samples: int = 200) -> float:
    """평면 inlier를 실제로 관측한 카메라들이 있는 쪽 = 밖. +1이면 법선 그대로, -1이면 뒤집는다.
    visibility(SfM track)가 있으면 그걸로, 없으면 투영으로 (가림을 몰라 건물 너머 반대편 벽도 '보인다'고 센다).
    건물 중심에서 멀어지는 쪽으로 정하면 평행한 두 면 (외벽과 1층 안쪽 출입구 벽)이 서로 반대로 뒤집힌다."""
    idx = p.inliers[np.linspace(0, len(p.inliers) - 1, min(samples, len(p.inliers))).astype(int)]
    if visibility is not None:
        seen = np.bincount(np.concatenate([visibility[i] for i in idx] + [np.empty(0, int)]),
                           minlength=len(cameras)) / len(idx)
    else:
        seen = np.array([cam.project(points[idx])[1].mean() for cam in cameras])
    votes = 0.0
    for cam, s in zip(cameras, seen):
        if s >= 0.1:  # 정면에 가깝게 본 카메라일수록 크게 (옆 벽을 따라 걷다 비스듬히 본 카메라는 작게)
            votes += s * abs(p.normal @ cam.R[2]) * np.sign(p.normal @ cam.center + p.d)
    if votes == 0:  # 아무 카메라에도 안 보이면 카메라 평균 쪽
        votes = np.sign(p.normal @ np.mean([c.center for c in cameras], 0) + p.d)
    return 1.0 if votes >= 0 else -1.0


def extract_facades(points: np.ndarray, cameras: list[CameraView], cfg: dict,
                    visibility: list[np.ndarray] | None = None) -> FacadeResult:
    """points: Y-up 정렬된 (N,3). cameras: 같은 좌표계 — 법선 방향(그 면을 관측한 카메라 쪽 = 밖)을 정한다.
    visibility: 포인트별 관측 cameras index (SfmResult.visibility)."""
    kept = remove_outliers(points, cfg["outlier_neighbors"], cfg["outlier_std_ratio"])
    idx_kept = np.flatnonzero(kept)
    P = points[kept]
    scene = scene_scale(P, cameras)
    raw = detect_planes(P, scene * cfg["distance_ratio"], cfg["ransac_iters"],
                        max(int(len(P) * cfg["min_inlier_ratio"]), 10), cfg["max_planes"])
    planes = [Plane(p.normal, p.d, idx_kept[p.inliers]) for p in raw]  # 원래 points index로

    sin_tol = np.sin(np.radians(cfg["vertical_tol_deg"]))
    vertical = [p for p in planes if abs(p.normal @ UP) < sin_tol]
    vertical = merge_planes(vertical, points, cfg["merge_angle_deg"], scene * cfg["merge_offset_ratio"])
    vertical.sort(key=lambda p: -len(p.inliers))
    warnings: list[str] = []
    if not vertical:
        return FacadeResult([], [], planes, kept, scene, 0.0, 0.0, ["no_vertical_plane"])

    normals = [_horizontal(outward_sign(p, points, cameras, visibility) * p.normal) for p in vertical]

    extra: list[dict] = []
    # chosen: (평면 위치·법선을 정하는 평면, 법선, 가로·높이 범위에 쓸 포인트 index 목록)
    if len(vertical) == 1:
        chosen = [(vertical[0], normals[0], [vertical[0].inliers])]
    else:
        alpha, ks, offs = snap_orthogonal(normals, [len(p.inliers) for p in vertical], cfg["orthogonal_tol_deg"])
        chosen, used = [], {}
        for p, n, k, off in zip(vertical, normals, ks, offs):  # 큰 평면부터
            rec = {"normal": n.round(4).tolist(), "inliers": len(p.inliers), "angle_off_deg": round(off, 1),
                   "points": p.inliers}
            if k is None:
                extra.append(rec | {"reason": "off_grid_angle"})
            elif k in used:  # 같은 방향의 더 작은 면 (출입구·돌출부, 긴 벽의 조금 휜 구간) → 그 외벽의 범위에만 보탠다
                chosen[used[k]][2].append(p.inliers)
                extra.append(rec | {"reason": "parallel_duplicate"})
            elif len(chosen) >= cfg["max_facades"]:
                extra.append(rec | {"reason": "max_facades"})
            else:
                used[k] = len(chosen)
                a = alpha + k * np.pi / 2
                chosen.append((p, np.array([np.cos(a), 0.0, np.sin(a)]), [p.inliers]))
        if extra:
            warnings.append(f"extra_planes: {len(extra)} vertical planes not used for the box (see recon report)")

    ys = np.concatenate([points[np.concatenate(ext), 1] for _, _, ext in chosen])
    ground, top = (float(np.percentile(ys, q)) for q in (cfg["ground_percentile"], cfg["top_percentile"]))
    facades = []
    for p, n, ext in chosen:
        X = points[p.inliers]
        d = -float(np.median(X @ n))  # 스냅한 법선으로 거리 다시
        f = FacadeFrame("", n, np.zeros(3), 1.0, top - ground, p.inliers)
        on_plane = X - np.outer(X @ n + d, n)
        E = points[np.concatenate(ext)]
        u = E @ f.u_axis
        u0, u1 = np.percentile(u, cfg["extent_percentile"])
        base = on_plane.mean(0) - (on_plane.mean(0) @ f.u_axis) * f.u_axis
        f.origin = base + u0 * f.u_axis
        f.origin[1] = ground
        f.width = float(u1 - u0)
        f.relief = wall_relief(E @ f.u_axis, E @ n + d, cfg["relief_bins"])
        facades.append(f)
    for i, f in enumerate(_order_around(facades), 1):
        f.id = f"F{i}"
    return FacadeResult(sorted(facades, key=lambda f: f.id), extra, planes, kept, scene, ground, top, warnings)


def wall_relief(s: np.ndarray, depth: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray] | None:
    """가로 위치 s별 벽 깊이 중앙값 (s 구간 bins개, 점 10개 미만 구간은 이웃에서 보간, 3구간 이동평균)."""
    if len(s) < 10 * bins:
        return None
    edges = np.linspace(*np.percentile(s, [1, 99]), bins + 1)
    c = (edges[:-1] + edges[1:]) / 2
    k = np.clip(np.digitize(s, edges) - 1, 0, bins - 1)
    m = np.array([np.median(depth[k == i]) if (k == i).sum() >= 10 else np.nan for i in range(bins)])
    ok = ~np.isnan(m)
    if ok.sum() < 3:
        return None
    m = np.interp(c, c[ok], m[ok])
    m = np.convolve(np.pad(m, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    return c, m


def _order_around(facades: list[FacadeFrame]) -> list[FacadeFrame]:
    """가장 큰 면 = F1, 나머지는 법선 각 atan2(z, x)가 커지는 순서 (건물을 한 바퀴 도는 순서)."""
    if not facades:
        return []
    first = max(facades, key=lambda f: len(f.inliers))
    a0 = _angle(first.normal)
    return sorted(facades, key=lambda f: (_angle(f.normal) - a0) % (2 * np.pi))
