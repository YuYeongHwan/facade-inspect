"""목업 result.json + 이미지 + building.glb 생성 (프론트/백엔드 병렬 개발용).

실행: uv run --project cv python fixtures/make_mock.py
"""

import random
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import trimesh
from facade_cv.config import load_config
from facade_cv.contamination.grade import cleaning_level, csi_params, grade_window, k_t_max
from facade_cv.schema import InspectionResult

ROOT = Path(__file__).resolve().parent
GRADING = load_config("grading")

W, H, D = 12.0, 24.0, 8.0  # 박스 건물 x, y(높이), z
WIN_W, WIN_H = 2.0, 2.5
ROW_CENTERS = [7.0, 16.0]
CROP_W, CROP_H = 256, 320
TEX_PX_PER_M = 20
TYPES = ["dust", "stain", "grime", "heavy"]
COLORS = {"dust": (80, 200, 230), "stain": (40, 130, 240), "grime": (40, 70, 110), "heavy": (30, 30, 200)}  # BGR
GLASS_BGR = (200, 185, 160)
BRANCH_BGR = (40, 90, 60)  # 창 앞 나뭇가지 (가림)
# level별 목표 CSI (구간 중앙 부근). 실제 CSI·level은 그린 crop을 grade_window로 측정해 다시 계산
TARGET = {1: 0.004, 2: 0.015, 3: 0.033, 4: 0.065, 5: 0.15}
TARGET_DE = {1: 4.0, 2: 10.0, 3: 24.0}  # 심각도별로 그릴 색차 (config severity.delta_e 경계 6/15 사이)
OCCLUDED = 4  # 가림(나뭇가지)을 그릴 창문 수
LEVEL_MIX = {1: 6, 2: 5, 3: 4, 4: 3, 5: 2}  # 창문 20개

# (id, origin, u_axis, width, cols) — v_axis는 모두 +Y, normal = u × v
FACADES = [
    ("F1", (0, 0, D), (1, 0, 0), W, 3),
    ("F2", (W, 0, D), (0, 0, -1), D, 2),
    ("F3", (W, 0, 0), (-1, 0, 0), W, 3),
    ("F4", (0, 0, 0), (0, 0, 1), D, 2),
]


def ellipse_poly(rng: random.Random, area: float) -> np.ndarray:
    """정규화 (u,v) 16각형, 면적 ≈ area. 창틀 마진 안쪽에 둔다."""
    aspect = rng.uniform(0.6, 1.6)
    rx = np.sqrt(area / np.pi * aspect)
    ry = area / np.pi / rx
    lo = GRADING["frame_margin"] + 0.02
    rx, ry = min(rx, 0.5 - lo), min(ry, 0.5 - lo)
    cx, cy = rng.uniform(lo + rx, 1 - lo - rx), rng.uniform(lo + ry, 1 - lo - ry)
    t = np.linspace(0, 2 * np.pi, 16, endpoint=False) + rng.uniform(0, 1)
    jitter = np.array([rng.uniform(0.85, 1.15) for _ in t])
    return np.clip(np.stack([cx + rx * jitter * np.cos(t), cy + ry * jitter * np.sin(t)], 1), 0, 1)


def to_px(poly: np.ndarray) -> np.ndarray:
    return np.round(poly * [CROP_W - 1, CROP_H - 1]).astype(np.int32)


def _lab(bgr) -> np.ndarray:
    return cv2.cvtColor(np.array([[bgr]], np.float32) / 255, cv2.COLOR_BGR2LAB)[0, 0]


def make_window(rng: random.Random, target: int, occluded: bool, out: Path) -> dict:
    regions = []  # (종류, 그릴 심각도, polygon)
    denom = GRADING["severity"]["max"] * k_t_max(GRADING)
    if target != 1 or rng.random() < 0.5:
        types = rng.sample(TYPES, rng.randint(1, 3))
        for typ in types:
            sev = rng.randint(1, 3)
            share = TARGET[target] / len(types)  # 이 영역의 CSI 기여 목표 → 면적 = share × 분모 / (k_s × k_t)
            regions.append((typ, sev, ellipse_poly(rng, share * denom / (sev * GRADING["weights"][typ]))))

    # crop: 유리 + 창틀 + 오염 얼룩 (심각도별 색차가 TARGET_DE가 되도록 혼합 비율) + (일부) 나뭇가지
    crop = np.full((CROP_H, CROP_W, 3), GLASS_BGR, np.uint8)
    border = round(GRADING["frame_margin"] * CROP_W)
    cv2.rectangle(crop, (0, 0), (CROP_W - 1, CROP_H - 1), (90, 90, 90), border)
    masks = []
    for typ, sev, poly in regions:
        m = np.zeros((CROP_H, CROP_W), np.uint8)
        cv2.fillPoly(m, [to_px(poly)], 1)
        masks.append((typ, m.astype(bool)))
        alpha = min(0.95, TARGET_DE[sev] / float(np.linalg.norm(_lab(COLORS[typ]) - _lab(GLASS_BGR))))
        layer = crop.copy()
        cv2.fillPoly(layer, [to_px(poly)], COLORS[typ])
        crop = np.where(m[..., None].astype(bool), cv2.addWeighted(layer, alpha, crop, 1 - alpha, 0), crop)
    occlusion = None
    if occluded:  # 왼쪽 위에서 들어오는 굵은 가지 (G5 occlusion mask 역할)
        occ = np.zeros((CROP_H, CROP_W), np.uint8)
        y0 = rng.randint(CROP_H // 5, CROP_H // 2)
        cv2.line(occ, (0, y0), (rng.randint(CROP_W // 2, CROP_W * 3 // 4), y0 + rng.randint(-60, 60)), 1, rng.randint(30, 45))
        crop[occ.astype(bool)] = BRANCH_BGR
        occlusion = occ.astype(bool)
    crop = cv2.GaussianBlur(crop, (3, 3), 0)

    g = grade_window(crop, masks, occlusion, GRADING)
    csi = round(g.csi, 5)
    level, label = cleaning_level(csi, GRADING)
    by_type = {}
    contamination = []
    for (typ, _, poly), r in zip(regions, g.regions, strict=True):  # 영역은 창틀 안이라 버려지는 것 없음
        contamination.append({"type": typ, "area_ratio": round(r.area_ratio, 5), "severity": r.severity,
                              "delta_e": r.delta_e, "polygon": np.round(poly, 4).tolist()})
        by_type[typ] = by_type.get(typ, 0) + r.area_ratio
    type_ratios = {t: round(by_type.get(t, 0.0), 5) for t in TYPES}

    # heatmap: 영역별 k_s × k_t / 분모 → JET, 알파 = 강도
    intensity = np.zeros((CROP_H, CROP_W), np.float32)
    for (typ, m), r in zip(masks, g.regions):
        intensity[m] = np.maximum(intensity[m], r.severity * GRADING["weights"][typ] / denom)
    if occlusion is not None:
        intensity[occlusion] = 0
    intensity = cv2.GaussianBlur(intensity, (31, 31), 0)
    heat = cv2.applyColorMap((intensity * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = np.dstack([heat, (intensity * 200).astype(np.uint8)])

    # overlay: crop 위에 유형별 색 폴리곤
    overlay = crop.copy()
    for typ, _, poly in regions:
        layer = overlay.copy()
        cv2.fillPoly(layer, [to_px(poly)], COLORS[typ])
        overlay = cv2.addWeighted(layer, 0.4, overlay, 0.6, 0)
        cv2.polylines(overlay, [to_px(poly)], True, COLORS[typ], 2)

    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "crop.jpg"), crop)
    cv2.imwrite(str(out / "heatmap.png"), heat)
    cv2.imwrite(str(out / "overlay.png"), overlay)
    if occlusion is not None:  # 가림 영역은 빗금 윤곽으로 (오염 계산에서 뺀 부분)
        cnts, _ = cv2.findContours(occlusion.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (255, 255, 255), 2)
    return {
        "cleaning_level": level,
        "cleaning_label": label,
        "csi": csi,
        "glass_ratio": round(g.glass_ratio, 5),
        "occluded_ratio": round(g.occluded_ratio, 5),
        "contamination": contamination,
        "type_ratios": type_ratios,
    }


def main() -> None:
    rng = random.Random(42)
    targets = [lv for lv, n in LEVEL_MIX.items() for _ in range(n)]
    rng.shuffle(targets)
    occluded = set(rng.sample(range(len(targets)), OCCLUDED))

    facades, windows = [], []
    for fid, origin, u_axis, width, cols in FACADES:
        o, u, v = np.array(origin, float), np.array(u_axis, float), np.array([0.0, 1.0, 0.0])
        facades.append({
            "id": fid, "normal": np.cross(u, v).tolist(), "origin": o.tolist(), "u_axis": u.tolist(),
            "v_axis": v.tolist(), "width": width, "height": H, "texture": f"facades/{fid}.jpg",
        })
        # 외벽 텍스처: 이미지 y는 아래로, v는 위로
        tex = np.full((int(H * TEX_PX_PER_M), int(width * TEX_PX_PER_M), 3), (170, 175, 180), np.uint8)
        for row, vc in enumerate(ROW_CENTERS, 1):
            for col in range(1, cols + 1):
                uc = width * (col - 0.5) / cols
                rect = [uc - WIN_W / 2, vc - WIN_H / 2, uc + WIN_W / 2, vc + WIN_H / 2]
                u0, v0, u1, v1 = rect
                corners = [o + u * a + v * b for a, b in [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]]
                wid = f"{fid}-R{row:02d}-C{col:02d}"
                windows.append({
                    "id": wid, "facade_id": fid, "row": row, "col": col, "facade_rect": rect,
                    "corners_3d": [c.tolist() for c in corners],
                    **make_window(rng, targets[len(windows)], len(windows) in occluded, ROOT / "windows" / wid),
                    "images": {k: f"windows/{wid}/{f}" for k, f in
                               [("crop", "crop.jpg"), ("heatmap", "heatmap.png"), ("overlay", "overlay.png")]},
                    "num_views": rng.randint(6, 20),
                })
                px = lambda a, b: (int(a * TEX_PX_PER_M), int((H - b) * TEX_PX_PER_M))  # noqa: E731
                cv2.rectangle(tex, px(u0, v1), px(u1, v0), (110, 90, 70), -1)
        (ROOT / "facades").mkdir(exist_ok=True)
        cv2.imwrite(str(ROOT / "facades" / f"{fid}.jpg"), tex)

    box = trimesh.creation.box(extents=(W, H, D))
    box.apply_translation((W / 2, H / 2, D / 2))
    box.export(ROOT / "building.glb")

    result = InspectionResult.model_validate({
        "inspection_id": "insp_mock",
        "meta": {"video": "input.mp4", "fps": 30, "frames_used": 240, "scale_known": False,
                 "created_at": datetime(2026, 10, 2, tzinfo=timezone.utc), "csi_params": csi_params(GRADING)},
        "building": {"mesh": "building.glb", "facades": facades},
        "windows": windows,
        "summary": {"total": len(windows),
                    "level_counts": {str(lv): sum(w["cleaning_level"] == lv for w in windows) for lv in range(1, 6)}},
    })
    (ROOT / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(f"wrote {ROOT / 'result.json'}: {result.summary.level_counts}")


if __name__ == "__main__":
    main()
