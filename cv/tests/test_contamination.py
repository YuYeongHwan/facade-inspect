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


def test_calib_sheet_script_parses(tmp_path):
    """평가지 <script>가 문법 오류 없이 파싱된다 (파이썬 문자열의 \\n이 JS 문자열 안에서 실제 줄바꿈이 되면 SyntaxError)."""
    import shutil
    import subprocess

    from facade_cv.contamination.evaluate import sheet_html
    node = shutil.which("node")
    if not node:
        pytest.skip("node 없음")
    h = sheet_html(["img_001.jpg"], load_config("calibrate"), GRADING)
    js = tmp_path / "sheet.js"
    js.write_text(h.split("<script>")[1].split("</script>")[0], encoding="utf-8")
    r = subprocess.run([node, "--check", str(js)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_whole_image_facade():
    from facade_cv.contamination.analyze import whole_image_facade
    img = np.zeros((300, 600, 3), np.uint8)
    f, tex, ppu, wins, det = whole_image_facade(img, "a.jpg", 10.0, 200)
    assert tex.shape[:2] == (100, 200) and f.height == 10.0 and abs(f.width - 20.0) < 1e-6  # 비율 유지, 축소
    (w,) = wins
    assert w.id == "F1-R01-C01" and list(w.rect) == [0.0, 0.0, f.width, 10.0]  # 사진 전체 = 창 1개
    assert w.observations[0].obs.corners.tolist() == [[0, 0], [599, 0], [599, 299], [0, 299]]  # 원본 픽셀 코너
    assert det["windows"][0]["corners"] == w.observations[0].obs.corners.tolist()


# ── T7 2차: film·곧은 막대·반사 의심 예외·ROI·깨끗한 기준 사진 ──

def _set(cfg, **kv):
    from facade_cv.contamination.evaluate import set_key
    for k, v in kv.items():
        cfg = set_key(cfg, k.replace("__", "."), v)
    return cfg


def _scene(h=300, w=400, seed=0):
    """정렬용 질감 있는 배경 (사각형 여러 개 + 완만한 기울기)."""
    rng = np.random.default_rng(seed)
    img = np.dstack([np.tile(np.linspace(60, 160, w, dtype=np.float32), (h, 1))] * 3)
    for _ in range(25):
        x, y = rng.integers(0, w - 40), rng.integers(0, h - 40)
        img[y:y + rng.integers(10, 40), x:x + rng.integers(10, 40)] = rng.integers(20, 230, 3)
    return cv2.GaussianBlur(img, (0, 0), 1.0).clip(0, 255).astype(np.uint8)


def test_reference_pair_truth_and_detection():
    """깨끗한 사진을 살짝 움직인(호모그래피) 오염 사진에 정렬 → 차이 = 정답 마스크가 알려진 얼룩과 겹친다 (IoU ≥ 0.8)."""
    from facade_cv.contamination.analyze import align_reference, analyze_crop, reference_truth
    clean = _scene()
    H = np.array([[1.0, 0.01, 5.0], [-0.008, 1.0, -4.0], [0, 0, 1]])
    dirty = cv2.warpPerspective(clean, H, clean.shape[1::-1], borderMode=cv2.BORDER_REPLICATE)
    known = np.zeros(clean.shape[:2], np.uint8)
    for c in ((120, 100), (280, 200), (330, 80)):
        cv2.circle(known, c, 18, 1, -1)
    k = known.astype(bool)
    before = dirty.copy()
    dirty[k] = (0.45 * dirty[k] + 0.55 * np.array([150, 160, 165])).astype(np.uint8)  # 회갈색 먼지
    k &= np.linalg.norm(lab(dirty) - lab(before), axis=2) >= GRADING["severity"]["delta_e"][0]  # 실제로 보이는 변화만 정답
    al = align_reference(dirty, clean, CFG)
    assert al is not None
    gt = reference_truth(dirty, al, CFG, GRADING)
    assert (gt & k).sum() / (gt | k).sum() >= 0.8
    a = analyze_crop(dirty, None, CFG, GRADING, reference=al)
    det = np.zeros(k.shape, bool)
    for r in a.regions:
        det |= r.mask
    kk = k & a.glass
    assert (det & kk).sum() / (det | kk).sum() >= 0.6


def test_roi_window_polygon():
    from facade_cv.contamination.analyze import roi_window
    corners, occ = roi_window([[0.5, 0.1], [0.9, 0.5], [0.5, 0.9], [0.1, 0.5]], 101, 101)  # 마름모
    assert corners.tolist() == [[10, 10], [90, 10], [90, 90], [10, 90]]
    assert len(occ) == 4  # 외접 사각형 네 모서리 조각
    corners, occ = roi_window([0.2, 0.2, 0.6, 0.8], 101, 101)
    assert corners.tolist() == [[20, 20], [60, 20], [60, 80], [20, 80]] and occ == []


def test_film_veil_adds_coverage_region():
    """유리 전체가 고르게 뿌연 막(밝기 들림·대비 저하) — 배경 곡면에 흡수돼 국소 후보로는 안 잡히는 경우 (img_053 유형) —
    이 film 영역(dust)으로 잡히고 덮인 비율이 면적이 된다. 먼지막 없는 원본보다 훨씬 넓다."""
    from facade_cv.contamination.analyze import analyze_crop
    img = _scene(seed=3)
    img[..., 1:] //= 4  # 색이 진한 장면 (어두운 채널 사전 [R34]은 무채색 밝은 면을 안개로 본다 — 한계)
    hazy = (0.4 * img + 0.6 * 215).astype(np.uint8)
    cfg = _set(CFG, film__enabled=True, film__metric="veil", film__min=0.6)

    def cover(im):
        a = analyze_crop(im, None, cfg, GRADING)
        film = [r for r in a.regions if r.features.get("film")]
        assert all(r.type == "dust" for r in film) and len(film) <= 1
        return film[0].mask.sum() / a.glass.sum() if film else 0.0

    c_hazy, c_clean = cover(hazy), cover(img)
    assert c_hazy > 0.6 and c_hazy > 2 * c_clean


def test_straight_cover():
    from facade_cv.contamination.analyze import _straight_cover
    segs = np.array([[50, 0, 51, 60], [52, 50, 50, 100], [90, 0, 90, 100]], np.float32)  # 막대 띠 안 두 선분 + 띠 밖
    assert _straight_cover(segs, 48, 54, 0, 100, 1) == pytest.approx(1.0)
    assert _straight_cover(segs[:1], 48, 54, 0, 100, 1) == pytest.approx(0.6)
    assert _straight_cover(segs[2:], 48, 54, 0, 100, 1) == 0.0


def test_suspect_rules_and_white_exception():
    """색 번짐(채도 상승) → 반사 의심, 큰 국소 영역 → 반사 의심, 날카로운 흰 덩어리 → 예외로 오염 유지."""
    from facade_cv.contamination.analyze import analyze_crop
    base = np.full((240, 320, 3), 120, np.uint8)
    green = base.copy()
    cv2.circle(green, (160, 120), 30, (110, 170, 100), -1)
    green = cv2.GaussianBlur(green, (0, 0), 6)
    cfg = _set(CFG, reflection__min_chroma_gain=8)
    assert analyze_crop(green, None, CFG, GRADING).regions and not analyze_crop(green, None, cfg, GRADING).regions
    from facade_cv.contamination.analyze import suspect_reason
    m = np.zeros((100, 100), bool)
    m[:, :40] = True
    f = {"chroma_gain": 2.0}
    assert suspect_reason(m, f, 10000, CFG) is None  # 기본값 = 둘 다 끔
    assert suspect_reason(m, f, 10000, _set(CFG, candidate__max_region_ratio=0.3)) == "large"
    assert suspect_reason(m, f, 10000, _set(CFG, candidate__max_region_ratio=0.5)) is None
    assert suspect_reason(m, {"chroma_gain": 9.0}, 10000, _set(CFG, reflection__min_chroma_gain=8)) == "chroma"
    white = base.copy()
    cv2.circle(white, (160, 120), 40, (212, 212, 212), -1)  # 넓은 흰 덩어리 (새똥 근접, 반사광 L* 기준 아래)
    cfg = _set(CFG, candidate__max_region_ratio=0.05)  # 큰 영역 규칙에 걸리게
    assert not analyze_crop(white, None, cfg, GRADING).regions
    cfg = _set(cfg, glare__keep_sharp__reflection_exception=True)
    assert analyze_crop(white, None, cfg, GRADING).regions
    assert not analyze_crop(white, None, _set(cfg, glare__keep_sharp__exception_max_area=0.05), GRADING).regions
