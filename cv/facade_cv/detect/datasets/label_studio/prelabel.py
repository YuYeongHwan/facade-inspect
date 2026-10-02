"""학습된 YOLO-seg 예측 → Label Studio import JSON (predictions). 라벨러는 예측 polygon을 고치기만 하면 된다.

python -m facade_cv.detect.datasets.label_studio.prelabel --weights models/window-seg.pt \
    --images data/own/raw --out data/own/prelabel.json --url-prefix "/data/local-files/?d=raw/"
"""

import json
from pathlib import Path

import numpy as np
import typer

from facade_cv.config import load_config
from facade_cv.detect.datasets.label_studio.export_to_yolo import FROM_NAME
from facade_cv.detect.datasets.yolo import IMAGE_EXTS, simplify_polygon


def to_ls_task(
    image_url: str, width: int, height: int, polygons: list[tuple[str, np.ndarray, float]], model_version: str
) -> dict:
    """polygons: (label, 정규화 polygon (N,2), score) → Label Studio task (predictions 포함)."""
    result = [
        {
            "id": f"p{i}", "from_name": FROM_NAME, "to_name": "image", "type": "polygonlabels",
            "original_width": width, "original_height": height, "image_rotation": 0, "score": round(score, 4),
            "value": {"points": np.round(np.asarray(pts) * 100, 3).tolist(), "closed": True, "polygonlabels": [label]},
        }
        for i, (label, pts, score) in enumerate(polygons)
    ]
    score = float(np.mean([s for *_, s in polygons])) if polygons else 0.0
    return {"data": {"image": image_url},
            "predictions": [{"model_version": model_version, "score": round(score, 4), "result": result}]}


def prelabel(
    weights: str | Path, images_dir: str | Path, out_json: str | Path, url_prefix: str, cfg: dict | None = None
) -> int:
    """예측한 이미지 수를 반환."""
    from ultralytics import YOLO

    from facade_cv.device import get_device

    cfg = cfg or load_config("dataset")
    pc, names = cfg["prelabel"], cfg["names"]
    model = YOLO(str(weights))
    tasks = []
    results = model.predict(str(images_dir), conf=pc["conf"], imgsz=pc["imgsz"], device=get_device(),
                            retina_masks=True, stream=True, verbose=False)
    for res in results:
        path = Path(res.path)
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        h, w = res.orig_shape
        polygons = []
        if res.masks is not None:
            for xy, c, s in zip(res.masks.xy, res.boxes.cls.tolist(), res.boxes.conf.tolist()):
                label = model.names[int(c)]
                poly = simplify_polygon(xy, cfg["polygon_eps"]) if len(xy) >= 3 else xy
                if label in names and len(poly) >= 3:
                    polygons.append((label, poly / [w, h], float(s)))
        tasks.append(to_ls_task(url_prefix + path.name, w, h, polygons, Path(weights).stem))
    Path(out_json).write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")
    return len(tasks)


def main(
    weights: Path = typer.Option(..., exists=True, dir_okay=False),
    images: Path = typer.Option(..., exists=True, file_okay=False),
    out: Path = typer.Option(...),
    url_prefix: str = typer.Option("/data/local-files/?d=", help="Label Studio에서 이미지 파일명 앞에 붙일 URL"),
) -> None:
    print(f"{prelabel(weights, images, out, url_prefix)} tasks → {out}")


if __name__ == "__main__":
    typer.run(main)
