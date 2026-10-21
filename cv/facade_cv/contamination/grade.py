"""창문 오염 심각도 지수 CSI (0~1) → 청소 우선순위 level 1~5 (config/grading.yaml). 식·근거·예시: docs/scoring.md

  CSI = Σ_j (A_j × k_s(j) × k_t(j)) / (A_g × k_s_max × k_t_max)

- A_g: 유리 면적 = 정면화 창문 crop − 창틀 마진(frame_margin) − 가림(occlusion, detect.postprocess)
- A_j: 오염 영역 j 중 유리 안 면적 (가림·창틀 부분은 빼고 센다) → area_ratio = A_j / A_g
- k_s: 심각도 1~3 = 영역 평균색과 깨끗한 유리 평균색의 CIELAB ΔE*ab [R22] (불투명도 대리 지표 [R5])
- k_t: 종류 가중치 [R6], k_t_max = 그 최댓값
구조는 외벽 열화 심각도 지수 [R2], 지수 → 등급 구간은 PCI [R4]. 수치는 모두 보정 전 초기값 (calibrate.py로 보정).

python -m facade_cv.contamination.grade 0.03   →  3 가급적 청소
"""

from collections.abc import Mapping
from dataclasses import dataclass

import cv2
import numpy as np
import typer

from facade_cv.config import load_config


@dataclass
class Region:
    type: str  # dust | stain | grime | heavy
    area_ratio: float  # A_j / A_g
    severity: int  # k_s 1~3
    delta_e: float | None = None  # 깨끗한 유리 대비 ΔE*ab (측정했을 때)
    contribution: float = 0.0  # 이 영역의 CSI 항 = area_ratio × k_s × k_t / (k_s_max × k_t_max)


@dataclass
class WindowGrade:
    csi: float
    level: int
    label: str
    regions: list[Region]
    type_ratios: dict[str, float]  # 종류별 Σ A_j / A_g
    glass_ratio: float = 1.0  # A_g / crop 면적
    occluded_ratio: float = 0.0  # 창틀 마진 안쪽에서 가려진 면적 / crop 면적


def k_t_max(cfg: dict) -> float:
    return max(cfg["weights"].values())


def compute_csi(regions: list[Region], cfg: dict | None = None) -> float:
    """Σ area_ratio × k_s × k_t / (k_s_max × k_t_max). 각 Region.contribution도 채운다. 영역이 겹치지 않으면 0~1."""
    cfg = cfg or load_config("grading")
    denom = cfg["severity"]["max"] * k_t_max(cfg)
    for r in regions:
        r.contribution = r.area_ratio * r.severity * cfg["weights"][r.type] / denom
    return sum(r.contribution for r in regions)


def cleaning_level(csi: float, cfg: dict | None = None) -> tuple[int, str]:
    """csi가 min_csi 이상인 가장 높은 level의 (level, label). 경계값은 상위 level."""
    levels = [lv for lv in (cfg or load_config("grading"))["levels"] if csi >= lv["min_csi"]]
    if not levels:
        raise ValueError(f"csi {csi} below every level min_csi")
    top = max(levels, key=lambda lv: lv["min_csi"])
    return top["level"], top["label"]


def severity_level(delta_e: float, cfg: dict | None = None) -> int:
    """ΔE*ab → k_s (1 + 넘은 경계 수)."""
    sc = (cfg or load_config("grading"))["severity"]
    return 1 + sum(delta_e >= t for t in sc["delta_e"])


def glass_mask(shape: tuple[int, int], frame_margin: float, occlusion: np.ndarray | None = None) -> np.ndarray:
    """정면화 crop (H,W) → 유리 마스크: 각 변 frame_margin 비율을 창틀로 빼고, 가림 마스크를 뺀다."""
    h, w = shape
    my, mx = round(h * frame_margin), round(w * frame_margin)
    g = np.zeros(shape, bool)
    g[my:h - my, mx:w - mx] = True
    return g & ~occlusion if occlusion is not None else g


def lab(bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 → CIELAB float (L 0~100, a·b 원래 범위) [R22]. OpenCV 8bit LAB는 스케일이 달라 float로 변환."""
    return cv2.cvtColor(bgr.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)


def delta_e(lab_img: np.ndarray, region: np.ndarray, clean: np.ndarray) -> float:
    """CIE76 ΔE*ab = 영역 평균 LAB와 깨끗한 유리 평균 LAB의 유클리드 거리 [R22]."""
    return float(np.linalg.norm(lab_img[region].mean(0) - lab_img[clean].mean(0)))


def grade_window(
    crop: np.ndarray, regions: list[tuple[str, np.ndarray]], occlusion: np.ndarray | None = None, cfg: dict | None = None
) -> WindowGrade:
    """정면화 crop(BGR) + 오염 영역 [(종류, bool 마스크)] + 가림 마스크 → CSI와 계산 내역.

    영역끼리 겹치면 k_t가 큰 종류가 그 픽셀을 가져간다 (한 픽셀은 한 번만 센다 → CSI ≤ 1).
    가림·창틀에 걸친 부분은 영역 면적에서 빠진다. 유리 안에 남은 면적이 0인 영역은 버린다.
    """
    cfg = cfg or load_config("grading")
    sc = cfg["severity"]
    h, w = crop.shape[:2]
    frame = glass_mask((h, w), cfg["frame_margin"])
    glass = frame & ~occlusion if occlusion is not None else frame
    a_g = int(glass.sum())
    if a_g == 0:
        raise ValueError("glass area is 0 (fully occluded or frame_margin too large)")
    claimed = np.zeros((h, w), bool)
    order = sorted(range(len(regions)), key=lambda i: -cfg["weights"][regions[i][0]])
    kept: list[tuple[int, str, np.ndarray]] = []
    for i in order:
        typ, m = regions[i]
        m = m & glass & ~claimed
        claimed |= m
        if m.any():
            kept.append((i, typ, m))
    kept.sort()  # 입력 순서로 되돌림
    clean = glass & ~claimed
    img = lab(crop)
    use_clean = clean.sum() >= sc["clean_min_ratio"] * a_g
    out = []
    for _, typ, m in kept:
        de = delta_e(img, m, clean) if use_clean else None
        out.append(Region(typ, float(m.sum() / a_g), severity_level(de, cfg) if use_clean else sc["max"],
                          None if de is None else round(de, 2)))
    csi = compute_csi(out, cfg)
    level, label = cleaning_level(csi, cfg)
    ratios = {t: sum(r.area_ratio for r in out if r.type == t) for t in cfg["weights"]}
    occ = float((frame & occlusion).sum() / (h * w)) if occlusion is not None else 0.0
    return WindowGrade(csi, level, label, out, ratios, a_g / (h * w), occ)


def csi_params(cfg: dict | None = None) -> dict:
    """result.json meta.csi_params (schema.CsiParams) — 계산에 쓴 config 스냅샷."""
    cfg = cfg or load_config("grading")
    return {"weights": dict(cfg["weights"]), "severity_max": cfg["severity"]["max"],
            "severity_delta_e": list(cfg["severity"]["delta_e"]), "frame_margin": cfg["frame_margin"],
            "level_min_csi": [lv["min_csi"] for lv in sorted(cfg["levels"], key=lambda lv: lv["level"])],
            "status": cfg["status"]}


def explain(regions: list[Mapping], cfg: dict | None = None) -> str:
    """손 계산용 표 (docs/scoring.md 예시와 같은 형식). regions: [{type, area_ratio, severity}]."""
    cfg = cfg or load_config("grading")
    rs = [Region(r["type"], r["area_ratio"], r["severity"]) for r in regions]
    csi = compute_csi(rs, cfg)
    lines = [f"분모 k_s_max × k_t_max = {cfg['severity']['max']} × {k_t_max(cfg)}"]
    lines += [f"  {r.type:6s} A/A_g={r.area_ratio:.4f} × k_s={r.severity} × k_t={cfg['weights'][r.type]} → {r.contribution:.5f}"
              for r in rs]
    level, label = cleaning_level(csi, cfg)
    return "\n".join(lines + [f"CSI = {csi:.5f} → level {level} {label}"])


def main(csi: float) -> None:
    level, label = cleaning_level(csi)
    print(level, label)


if __name__ == "__main__":
    typer.run(main)
