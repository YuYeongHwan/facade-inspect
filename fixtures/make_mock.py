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
from facade_cv.contamination.grade import cleaning_level, compute_score
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
# level별 목표 score (구간 중앙 부근). 실제 level은 래스터 면적으로 다시 계산
TARGET = {1: 0.008, 2: 0.035, 3: 0.075, 4: 0.15, 5: 0.32}
LEVEL_MIX = {1: 6, 2: 5, 3: 4, 4: 3, 5: 2}  # 창문 20개

# (id, origin, u_axis, width, cols) — v_axis는 모두 +Y, normal = u × v
FACADES = [
    ("F1", (0, 0, D), (1, 0, 0), W, 3),
    ("F2", (W, 0, D), (0, 0, -1), D, 2),
    ("F3", (W, 0, 0), (-1, 0, 0), W, 3),
    ("F4", (0, 0, 0), (0, 0, 1), D, 2),
]


def ellipse_poly(rng: random.Random, area: float) -> np.ndarray:
    """정규화 (u,v) 16각형, 면적 ≈ area."""
    aspect = rng.uniform(0.6, 1.6)
    rx = np.sqrt(area / np.pi * aspect)
    ry = area / np.pi / rx
    rx, ry = min(rx, 0.45), min(ry, 0.45)
    cx, cy = rng.uniform(rx, 1 - rx), rng.uniform(ry, 1 - ry)
    t = np.linspace(0, 2 * np.pi, 16, endpoint=False) + rng.uniform(0, 1)
    jitter = np.array([rng.uniform(0.85, 1.15) for _ in t])
    return np.clip(np.stack([cx + rx * jitter * np.cos(t), cy + ry * jitter * np.sin(t)], 1), 0, 1)


def to_px(poly: np.ndarray) -> np.ndarray:
    return np.round(poly * [CROP_W - 1, CROP_H - 1]).astype(np.int32)


def make_window(rng: random.Random, target: int, out: Path) -> dict:
    regions = []
    if target != 1 or rng.random() < 0.5:
        types = rng.sample(TYPES, rng.randint(1, 3))
        for typ in types:
            share = TARGET[target] / len(types)
            regions.append((typ, ellipse_poly(rng, share / GRADING["weights"][typ])))

    masks = {t: np.zeros((CROP_H, CROP_W), np.uint8) for t in TYPES}
    contamination = []
    for typ, poly in regions:
        m = np.zeros_like(masks[typ])
        cv2.fillPoly(m, [to_px(poly)], 1)
        masks[typ] |= m
        contamination.append(
            {"type": typ, "area_ratio": round(float(m.mean()), 4), "polygon": np.round(poly, 4).tolist()}
        )
    type_ratios = {t: round(float(masks[t].mean()), 4) for t in TYPES}
    score = round(compute_score(type_ratios, GRADING), 4)
    level, label = cleaning_level(score, GRADING)

    # crop: 유리(밝은 청회색) + 프레임 + 오염 얼룩
    crop = np.full((CROP_H, CROP_W, 3), (200, 185, 160), np.uint8)
    cv2.rectangle(crop, (0, 0), (CROP_W - 1, CROP_H - 1), (90, 90, 90), 10)
    for typ, poly in regions:
        layer = crop.copy()
        cv2.fillPoly(layer, [to_px(poly)], COLORS[typ])
        crop = cv2.addWeighted(layer, 0.45, crop, 0.55, 0)
    crop = cv2.GaussianBlur(crop, (5, 5), 0)

    # heatmap: 오염 강도(가중치) → JET, 알파 = 강도
    intensity = sum(masks[t].astype(np.float32) * GRADING["weights"][t] for t in TYPES)
    intensity = cv2.GaussianBlur(np.clip(intensity / 1.5, 0, 1), (31, 31), 0)
    heat = cv2.applyColorMap((intensity * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = np.dstack([heat, (intensity * 200).astype(np.uint8)])

    # overlay: crop 위에 유형별 색 폴리곤
    overlay = crop.copy()
    for typ, poly in regions:
        layer = overlay.copy()
        cv2.fillPoly(layer, [to_px(poly)], COLORS[typ])
        overlay = cv2.addWeighted(layer, 0.4, overlay, 0.6, 0)
        cv2.polylines(overlay, [to_px(poly)], True, COLORS[typ], 2)

    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "crop.jpg"), crop)
    cv2.imwrite(str(out / "heatmap.png"), heat)
    cv2.imwrite(str(out / "overlay.png"), overlay)
    return {
        "cleaning_level": level,
        "cleaning_label": label,
        "score": score,
        "contamination": contamination,
        "type_ratios": type_ratios,
    }


def main() -> None:
    rng = random.Random(42)
    targets = [lv for lv, n in LEVEL_MIX.items() for _ in range(n)]
    rng.shuffle(targets)

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
                    **make_window(rng, targets[len(windows)], ROOT / "windows" / wid),
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
                 "created_at": datetime(2026, 10, 2, tzinfo=timezone.utc)},
        "building": {"mesh": "building.glb", "facades": facades},
        "windows": windows,
        "summary": {"total": len(windows),
                    "level_counts": {str(lv): sum(w["cleaning_level"] == lv for w in windows) for lv in range(1, 6)}},
    })
    (ROOT / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(f"wrote {ROOT / 'result.json'}: {result.summary.level_counts}")


if __name__ == "__main__":
    main()
