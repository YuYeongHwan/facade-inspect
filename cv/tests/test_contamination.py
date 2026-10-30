"""T7 오염 영역 검출 (analyze.py) — 크기·색·모양을 아는 합성 얼룩으로 면적·ΔE·유형·반사 제외 확인."""

import re
from pathlib import Path

import cv2
import numpy as np
import pytest

from facade_cv.config import load_config
from facade_cv.contamination.analyze import analyze_crop, window_fields
from facade_cv.contamination.grade import lab

W, H = 300, 200
GLASS = (120, 105, 90)  # BGR, 어두운 청회색 유리
CFG, GRADING = load_config("contamination"), load_config("grading")


def glass(seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(np.full((H, W, 3), GLASS, np.float32) + rng.normal(0, 1.5, (H, W, 3)), 0, 255).astype(np.uint8)


def de(bgr) -> float:
    a = lab(np.uint8([[bgr]]))[0, 0]
    return float(np.linalg.norm(a - lab(np.uint8([[GLASS]]))[0, 0]))


def only(img):
    a = analyze_crop(img, None, CFG, GRADING)
    return a, window_fields(a, "single", True, CFG, GRADING)


def test_clean_glass_level1():
    a, f = only(glass())
    assert f["contamination"] == [] and a.level == 1 and f["csi"] == 0


def test_heavy_blob_area_and_delta_e():
    img = glass()
    color = (40, 70, 110)  # 갈색 (새똥·흙탕물)
    cv2.circle(img, (150, 100), 20, color, -1)
    a, f = only(img)
    assert len(f["contamination"]) == 1
    r = f["contamination"][0]
    assert r["type"] == "heavy" and r["severity"] == 3
    assert r["area_px"] == pytest.approx(np.pi * 20 ** 2, rel=0.1)
    assert r["delta_e"] == pytest.approx(de(color), abs=2)
    assert f["weighted_coverage_pct"] == pytest.approx(f["csi"] * 4.5 * 100, abs=0.01)


def test_vertical_streak_is_stain():
    img = glass()
    color = (70, 62, 55)  # ΔE ≈ 18: 검출 임계(config candidate.*_delta_e)보다 분명히 진한 자국
    assert de(color) >= 1.5 * CFG["candidate"]["local_delta_e"]
    cv2.rectangle(img, (100, 40), (104, 150), color, -1)  # 세로 빗물 자국 5×111
    _, f = only(img)
    assert [r["type"] for r in f["contamination"]] == ["stain"]
    assert f["contamination"][0]["area_px"] == pytest.approx(5 * 111, rel=0.2)


def test_ring_is_stain():
    img = glass()
    cv2.circle(img, (150, 100), 22, (85, 80, 75), 4)
    _, f = only(img)
    assert [r["type"] for r in f["contamination"]] == ["stain"]


def test_bottom_band_is_grime():
    img = glass()
    # 유리 아래쪽 (창틀 마진 12px 위) 어두운 띠, 유리 폭의 ~70% (폭 전체 균일 띠는 창틀 막대와 구분 못 함 — scoring.md 한계)
    cv2.rectangle(img, (40, 165), (225, 186), (70, 65, 60), -1)
    _, f = only(img)
    assert [r["type"] for r in f["contamination"]] == ["grime"]


def test_scattered_dots_are_one_dust_region():
    img = glass(1)
    dots = np.zeros((H, W), np.uint8)
    rng = np.random.default_rng(3)
    for x, y in zip(rng.integers(60, 240, 40), rng.integers(50, 150, 40)):
        cv2.circle(dots, (int(x), int(y)), 2, 1, -1)
    color = (185, 175, 160)  # ΔE ≈ 27 — 반지름 2 점은 흐림 후 대비가 크게 줄어 임계의 2배 이상이어야 잡힌다
    assert de(color) >= 2 * CFG["candidate"]["local_delta_e"]
    img[dots > 0] = color  # 밝은 먼지 점 40개 (반지름 2)
    _, f = only(img)
    types = [r["type"] for r in f["contamination"]]
    assert types and set(types) == {"dust"}
    assert len(types) <= 3  # 흩어진 점이 한두 영역으로 묶인다
    assert sum(r["area_px"] for r in f["contamination"]) >= 0.5 * dots.sum()


def test_glare_excluded():
    """매우 밝고 무채색인 반사광 블롭은 오염으로 세지 않는다."""
    img = glass()
    cv2.ellipse(img, (150, 90), (45, 25), 20, 0, 360, (250, 250, 250), -1)
    _, f = only(img)
    assert f["contamination"] == []


def test_occlusion_removed_from_glass():
    img = glass()
    cv2.circle(img, (150, 100), 20, (40, 70, 110), -1)
    occ = np.zeros((H, W), bool)
    occ[:, 140:] = True  # 오른쪽 절반 가림 → 얼룩 절반만, A_g도 줄어듦
    a, f = only(img)
    b = analyze_crop(img, occ, CFG, GRADING)
    fb = window_fields(b, "single", True, CFG, GRADING)
    assert fb["glass_area_px"] < f["glass_area_px"]
    assert fb["contamination"][0]["area_px"] < f["contamination"][0]["area_px"]
    assert fb["occluded_ratio"] > 0.4


def test_type_colors_match_palette():
    ts = (Path(__file__).resolve().parents[2] / "web/lib/palette.ts").read_text(encoding="utf-8")
    block = ts[ts.index("TYPE_COLORS"):].split("};")[0]
    assert dict(re.findall(r'(\w+): "(#[0-9a-f]{6})"', block)) == CFG["type_colors"]


def test_mullion_bars_excluded():
    """정면화 창문의 창살(유리 끝에서 끝까지 곧은 막대)은 두께와 상관없이 오염이 아니다."""
    img = glass()
    img[:, 140:160] = (235, 235, 230)  # 세로 창살 (두께 20)
    img[95:103, :] = (235, 235, 230)  # 가로 창살
    _, f = only(img)
    assert f["contamination"] == []


@pytest.mark.parametrize("p", [0.1, 0.2, 0.3])
def test_white_blob_kept_as_contamination(p):
    """흰 덩어리(새똥: 밝고 무채색이지만 윤곽이 날카로움)는 반사광으로 지우지 않는다 (glare.keep_sharp)."""
    img = glass()
    circ = np.zeros((H, W), np.uint8)
    cv2.circle(circ, (150, 100), round(176 * p / 2), 1, -1)  # 지름 = 유리 짧은 변(176) × p
    img[circ > 0] = (225, 230, 232)
    a, f = only(img)
    det = np.logical_or.reduce([r.mask for r in a.regions]) if a.regions else np.zeros((H, W), bool)
    assert (det & (circ > 0)).sum() / circ.sum() >= 0.8
    assert {r["type"] for r in f["contamination"]} == {"heavy"}


def test_white_blob_off_switch():
    img = glass()
    cv2.circle(img, (150, 100), 26, (225, 230, 232), -1)
    cfg = {**CFG, "glare": {**CFG["glare"], "keep_sharp": {**CFG["glare"]["keep_sharp"], "enabled": False}}}
    assert analyze_crop(img, None, cfg, GRADING).regions == []


def test_soft_gradient_band_not_a_bar():
    """노을·하늘처럼 경계가 부드러운 가로 띠는 창살(막대)로 지우지 않는다."""
    from facade_cv.contamination.analyze import bar_structures
    from facade_cv.contamination.grade import glass_mask
    img = glass().astype(np.float32)
    band = np.exp(-((np.arange(H) - 100) / 15.0) ** 2)[:, None, None]  # 부드러운 가로 띠
    img = np.clip(img + band * np.array([0, 60, 110]), 0, 255).astype(np.uint8)
    g = glass_mask((H, W), 0.06)
    L = lab(img)
    diff = (np.linalg.norm(L - L[g].mean(0), axis=2) >= 10) & g
    assert diff.sum() > 0 and not bar_structures(diff, g, L, CFG).any()
    sharp = glass()
    sharp[90:110] = (60, 165, 230)  # 같은 색의 날카로운 띠 = 막대
    L = lab(sharp)
    diff = (np.linalg.norm(L - L[g].mean(0), axis=2) >= 10) & g
    assert bar_structures(diff, g, L, CFG)[95:105].any()


def test_polygon_label_iou():
    """evaluate --labels 형식: 정규화 polygon 정답 → 면적 IoU."""
    from facade_cv.contamination.evaluate import eval_public
    img = glass()
    cv2.rectangle(img, (120, 80), (179, 119), (40, 70, 110), -1)  # 60×40 갈색 덩어리
    poly = [{"class": "heavy", "polygon": [[120 / W, 80 / H], [180 / W, 80 / H], [180 / W, 120 / H], [120 / W, 120 / H]]}]
    row, _ = eval_public(img, [], "heavy", CFG, GRADING, poly)
    assert row["iou"] >= 0.8


def test_reflected_tree_is_suspect_not_contamination():
    """유리에 비친 나뭇가지(넓은 영역 + 긴 직선 많음)는 반사 의심으로 CSI에서 빠진다. 같은 크기의 흐릿한 얼룩은 오염으로 남는다."""
    img = glass()
    rng = np.random.default_rng(5)
    for _ in range(25):  # 나뭇가지: 길고 곧은 어두운 선
        x, y = int(rng.integers(40, 260)), int(rng.integers(30, 170))
        ang = rng.uniform(0, np.pi)
        L = int(rng.integers(40, 90))
        cv2.line(img, (x, y), (int(x + L * np.cos(ang)), int(y + L * np.sin(ang))), (50, 55, 45), 2)
    a, f = only(img)
    assert a.suspects and sum(r["area_ratio"] for r in f["reflection_suspects"]) > 0.05
    assert f["csi"] < 0.009  # 반사 의심은 CSI에 넣지 않음 → level 1

    smear = glass()
    blob = np.zeros((H, W), np.float32)
    cv2.ellipse(blob, (150, 100), (80, 50), 0, 0, 360, 1, -1)
    blob = cv2.GaussianBlur(blob, (0, 0), 6)[..., None]
    smear = np.clip(smear + blob * np.array([60, 60, 55]), 0, 255).astype(np.uint8)  # 넓고 흐릿한 먼지막
    b, _ = only(smear)
    assert not b.suspects and b.regions


def _render_views(n=8, seed=0, jitter=3.0):
    """고정 얼룩(갈색 원, 유리 좌표 (100,100))이 있는 창문을 n개 시점으로 촬영한 합성 프레임.
    프레임마다 비친 나무(초록 원)가 유리 위에서 자리를 옮긴다 (시차). 관측 코너에는 ±jitter px 검출 오차."""
    rng = np.random.default_rng(seed)
    frames, views = {}, []
    for i in range(n):
        tex = glass(i)
        cv2.rectangle(tex, (0, 0), (W - 1, H - 1), (60, 60, 65), 12)  # 창틀 (ECC 정렬 기준)
        cv2.circle(tex, (100, 100), 16, (40, 70, 110), -1)  # 고정 오염
        cv2.circle(tex, (int(60 + 160 * i / (n - 1)), 60), 22, (40, 120, 40), -1)  # 움직이는 반사
        quad = np.float32([[170, 140], [470, 140], [470, 340], [170, 340]]) + rng.uniform(-15, 15, (4, 2)).astype(np.float32)
        M = cv2.getPerspectiveTransform(np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]]), quad)
        frame = np.full((480, 640, 3), (90, 95, 100), np.uint8)
        cv2.warpPerspective(tex, M, (640, 480), dst=frame, borderMode=cv2.BORDER_TRANSPARENT)
        frames[f"f{i}"] = frame
        views.append({"frame": f"f{i}", "corners": (quad + rng.uniform(-jitter, jitter, (4, 2))).tolist()})
    return frames, views


def _det_at(a, u, v, r=0.04) -> bool:
    """정규화 좌표 (u, v) 근처에 오염 픽셀이 있나."""
    det = np.logical_or.reduce([x.mask for x in a.regions]) if a.regions else np.zeros(a.glass.shape, bool)
    h, w = det.shape
    return det[int((v - r) * h):int((v + r) * h), int((u - r) * w):int((u + r) * w)].any()


def test_multiview_keeps_fixed_stain_drops_moving_reflection():
    from facade_cv.contamination.analyze import rectify_quad
    from facade_cv.contamination.multiview import stack_views
    frames, views = _render_views()
    mv = stack_views(frames.__getitem__, views, CFG, GRADING["frame_margin"])
    assert mv.ok and mv.n_used >= 6
    a = analyze_crop(mv.median, mv.occlusion, CFG, GRADING, mv.unstable)
    assert _det_at(a, 100 / W, 100 / H)  # 고정 얼룩은 남는다
    for i in range(8):  # 반사가 지나간 자리엔 오염이 없다
        assert not _det_at(a, (60 + 160 * i / 7) / W, 60 / H, 0.02)
    single, _ = rectify_quad(frames["f0"], views[0]["corners"], 640)
    s = analyze_crop(single, None, CFG, GRADING)
    assert _det_at(s, 60 / W, 60 / H)  # 단일 이미지는 반사를 오염으로 잡는다 (대조군)


def test_multiview_fallbacks():
    from facade_cv.contamination.multiview import stack_views
    frames, views = _render_views(n=8)
    assert stack_views(frames.__getitem__, views[:2], CFG, GRADING["frame_margin"]).reason == "few_views"
    rng = np.random.default_rng(1)
    bad = [{"frame": v["frame"], "corners": (np.float32([[20, 20], [200, 30], [190, 200], [30, 210]])
                                              + rng.uniform(-40, 40, (4, 2))).tolist()} for v in views]  # 엉뚱한 자리
    mv = stack_views(frames.__getitem__, bad, CFG, GRADING["frame_margin"])
    assert not mv.ok and mv.reason in ("ecc", "misaligned")


def test_calib_sheet_is_blind():
    from facade_cv.config import load_config
    from facade_cv.contamination.evaluate import sheet_html
    h = sheet_html(["img_001.jpg", "img_002.jpg"], load_config("calibrate"), GRADING)
    assert "images/img_002.jpg" in h and h.count('data-i="img_001.jpg"') == 3  # 평가자 3열
    assert "csi" not in h.lower() and "image,rating,time" in h  # 예측 없음, calibrate가 읽는 csv 형식
