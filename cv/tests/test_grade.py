"""CSI (docs/scoring.md): 손 계산 예시 1·2와 같은 값, 가림 제외, 0~1 정규화, level 경계."""

import numpy as np
import pytest

from facade_cv.config import load_config
from facade_cv.contamination.grade import (
    Region,
    cleaning_level,
    compute_csi,
    csi_params,
    glass_mask,
    grade_window,
    severity_level,
)

# 예시 공통: 정면화 crop 300×200 (W×H), frame_margin 0.06 → 창틀 18px(좌우)·12px(위아래) → 유리 264×176 = 46,464px
W, H = 300, 200
GLASS = (200, 185, 160)  # BGR
# 깨끗한 유리 대비 ΔE*ab ≈ 3.3 (k_s 1), 8.5 (k_s 2), 21.3 (k_s 3)
SEV_COLOR = {1: (190, 176, 152), 2: (175, 162, 140), 3: (140, 128, 110)}


def _crop_and_masks(rects):
    """rects: [(종류, k_s, x0, y0, x1, y1)] → (crop, [(종류, mask)])."""
    crop = np.full((H, W, 3), GLASS, np.uint8)
    masks = []
    for typ, sev, x0, y0, x1, y1 in rects:
        m = np.zeros((H, W), bool)
        m[y0:y1, x0:x1] = True
        crop[m] = SEV_COLOR[sev]
        masks.append((typ, m))
    return crop, masks


@pytest.mark.parametrize("csi, level", [(0.0, 1), (0.00899, 1), (0.009, 2), (0.022, 3), (0.044, 4), (0.089, 5), (1.0, 5)])
def test_level_boundaries(csi, level):
    """경계값은 상위 level (config/grading.yaml 초기값)."""
    assert cleaning_level(csi)[0] == level


def test_config_is_initial_and_ordered():
    cfg = load_config("grading")
    assert cfg["status"] == "보정 전 초기값"
    mins = [lv["min_csi"] for lv in cfg["levels"]]
    assert [lv["level"] for lv in cfg["levels"]] == [1, 2, 3, 4, 5] and mins[0] == 0 and mins == sorted(mins)
    assert csi_params(cfg)["level_min_csi"] == mins


def test_severity_bands():
    assert [severity_level(d) for d in (0, 5.99, 6.0, 14.99, 15.0, 40)] == [1, 1, 2, 2, 3, 3]


def test_example1_hand_calculation():
    """docs/scoring.md 예시 1 — 가림 없음.
    dust 80×50=4,000px k_s1 / stain 50×30=1,500px k_s2 / grime 30×20=600px k_s3, A_g = 46,464
    CSI = (4000×1×0.5 + 1500×2×0.8 + 600×3×1.0) / (46464 × 3 × 1.5) = 6200 / 209088 = 0.029653 → level 3"""
    crop, masks = _crop_and_masks([("dust", 1, 30, 30, 110, 80), ("stain", 2, 150, 30, 200, 60),
                                   ("grime", 3, 150, 120, 180, 140)])
    g = grade_window(crop, masks)
    assert g.glass_ratio == pytest.approx(46464 / (W * H)) and g.occluded_ratio == 0
    assert [r.severity for r in g.regions] == [1, 2, 3]
    assert [r.area_ratio for r in g.regions] == pytest.approx([4000 / 46464, 1500 / 46464, 600 / 46464])
    hand = (4000 / 46464 * 1 * 0.5 + 1500 / 46464 * 2 * 0.8 + 600 / 46464 * 3 * 1.0) / 4.5
    assert g.csi == pytest.approx(hand) == pytest.approx(6200 / 209088) and round(g.csi, 6) == 0.029653
    assert (g.level, g.label) == (3, "가급적 청소")
    assert sum(r.contribution for r in g.regions) == pytest.approx(g.csi)


def test_example2_occlusion_excluded():
    """docs/scoring.md 예시 2 — 나뭇가지 가림 101×64 = 6,464px → A_g = 40,000.
    heavy 40×30=1,200px 중 400px이 가지 밑 → 800px만, k_s3 / dust 100×60=6,000px k_s1
    CSI = (800/40000×3×1.5 + 6000/40000×1×0.5) / 4.5 = (0.09 + 0.075)/4.5 = 0.036667 → level 3"""
    crop, masks = _crop_and_masks([("heavy", 3, 81, 74, 121, 104), ("dust", 1, 150, 100, 250, 160)])
    occ = np.zeros((H, W), bool)
    occ[20:84, 20:121] = True
    crop[occ] = (40, 90, 60)  # 가지 — 어두워서 그대로 두면 심한 오염처럼 보인다
    g = grade_window(crop, masks, occ)
    assert g.glass_ratio == pytest.approx(40000 / (W * H)) and g.occluded_ratio == pytest.approx(6464 / (W * H))
    assert [r.area_ratio for r in g.regions] == pytest.approx([800 / 40000, 6000 / 40000])
    assert [r.severity for r in g.regions] == [3, 1]
    assert round(g.csi, 6) == 0.036667 and g.level == 3

    # 가림을 모르면(가지를 heavy 오염으로 잘못 잡고, 분모도 그대로) level 5로 과대평가 — 문서의 비교 수치
    naive = [Region("heavy", (1200 + 6464) / 46464, 3), Region("dust", 6000 / 46464, 1)]
    assert round(compute_csi(naive), 6) == 0.179293 and cleaning_level(compute_csi(naive))[0] == 5


def test_csi_bounds_and_overlap():
    """유리 전체가 가장 무거운 종류·최고 심각도면 1, 겹친 영역은 한 번만 센다 (≤ 1)."""
    crop, _ = _crop_and_masks([])
    full = glass_mask((H, W), load_config("grading")["frame_margin"])
    crop[full] = SEV_COLOR[3]
    crop[:20, :20] = GLASS  # 기준 유리는 창틀 쪽 일부만 남아 clean 부족 → k_s = max
    g = grade_window(crop, [("heavy", full), ("dust", full.copy())])
    assert g.csi == pytest.approx(1.0) and len(g.regions) == 1 and g.regions[0].type == "heavy"
    assert grade_window(np.full((H, W, 3), GLASS, np.uint8), []).csi == 0.0


def test_fully_occluded_raises():
    with pytest.raises(ValueError, match="glass area"):
        grade_window(np.zeros((H, W, 3), np.uint8), [], np.ones((H, W), bool))
