"""building.glb (박스 + 외벽 텍스처 사각형) + facades/<id>.jpg + windows/<id>/crop.jpg + result.json.

오염 분석(T7) 전이라 창문은 모두 미분석: cleaning_level/csi/type_ratios/glass_ratio/occluded_ratio/heatmap/overlay = null.
"""

from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image

from facade_cv.recon3d.geometry import FacadeFrame
from facade_cv.recon3d.ortho import rect_to_px
from facade_cv.recon3d.windows3d import Window3D
from facade_cv.schema import UNANALYZED_LABEL, InspectionResult

BODY_COLOR = (150, 150, 150, 255)
LIFT = 2e-4  # 텍스처 사각형을 몸체 밖으로 띄우는 거리 (외벽 크기 대비) — z-fighting 방지


def write_glb(path: Path, body: trimesh.Trimesh, facades: list[FacadeFrame], textures: dict[str, np.ndarray]) -> None:
    body = body.copy()
    body.visual = trimesh.visual.ColorVisuals(body, face_colors=BODY_COLOR)
    scene = trimesh.Scene()
    scene.add_geometry(body, node_name="body")
    for f in facades:
        quad = f.corners() + f.normal * LIFT * max(f.width, f.height)
        uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)  # 왼쪽 아래 = 텍스처 아래쪽 (v 위로)
        img = Image.fromarray(cv2.cvtColor(textures[f.id], cv2.COLOR_BGR2RGB))
        mesh = trimesh.Trimesh(quad, [[0, 1, 2], [0, 2, 3]], process=False,
                               visual=trimesh.visual.TextureVisuals(uv=uv, image=img))
        scene.add_geometry(mesh, node_name=f.id)
    scene.export(path)


def export_result(out: Path, inspection_id: str, meta: dict, body: trimesh.Trimesh, facades: list[FacadeFrame],
                  textures: dict[str, tuple[np.ndarray, float]], windows: list[Window3D], status: dict,
                  jpeg_quality: int = 92) -> InspectionResult:
    """out/ 아래에 모두 쓰고 검증한 InspectionResult를 반환. meta: video, fps, frames_used."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "facades").mkdir(exist_ok=True)
    q = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    for f in facades:
        cv2.imwrite(str(out / "facades" / f"{f.id}.jpg"), textures[f.id][0], q)
    write_glb(out / "building.glb", body, facades, {k: v[0] for k, v in textures.items()})

    by_id = {f.id: f for f in facades}
    wins = []
    for w in windows:
        f = by_id[w.facade_id]
        tex, ppu = textures[f.id][:2]
        x0, y0, x1, y1 = rect_to_px(tuple(w.rect), f.height, ppu)
        crop = tex[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)]
        d = out / "windows" / w.id
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "crop.jpg"), crop if crop.size else np.zeros((8, 8, 3), np.uint8), q)
        wins.append({
            "id": w.id, "facade_id": f.id, "row": w.row, "col": w.col,
            "facade_rect": [round(float(x), 5) for x in w.rect],
            "corners_3d": [[round(float(x), 5) for x in c] for c in w.corners_3d(f)],
            "cleaning_level": None, "cleaning_label": UNANALYZED_LABEL, "csi": None, "glass_ratio": None,
            "occluded_ratio": None, "contamination": [],
            "type_ratios": None, "images": {"crop": f"windows/{w.id}/crop.jpg", "heatmap": None, "overlay": None},
            "num_views": w.num_views,
        })
    result = InspectionResult.model_validate({
        "inspection_id": inspection_id,
        "meta": {**meta, "scale_known": False, "created_at": datetime.now(timezone.utc), "recon_status": status},
        "building": {"mesh": "building.glb", "facades": [f.to_schema(f"facades/{f.id}.jpg") for f in facades]},
        "windows": wins,
        "summary": {"total": len(wins), "level_counts": {"unanalyzed": len(wins)}},
    })
    (out / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result
