"""TTPLA [R20] (Apache-2.0, 항공 송전탑·전선) → YOLO-seg. cable → wire, tower_wooden → pole, 철탑·void 버림 (근거: data/DATASETS.md)

python -m facade_cv.detect.datasets.ttpla_convert --src data/external/ttpla --out data/yolo/ttpla
--src 아래를 재귀 탐색해 labelme JSON(shapes[].label/points, imageWidth/Height)과 같은 이름의 이미지를 짝짓는다.
매핑된 인스턴스가 하나도 없는 이미지는 버린다. 이미지는 긴 변 config ttpla.max_side로 줄여 저장 (라벨은 정규화라 그대로).
"""

import json
from pathlib import Path

import cv2
import numpy as np
import typer

from facade_cv.config import class_names, load_config
from facade_cv.detect.datasets.yolo import IMAGE_EXTS, source_missing, write_label

GUIDE = """TTPLA (가입 불필요, Apache-2.0 [R20]): https://github.com/r3ab/ttpla_dataset README의 Google Drive 링크에서
  이미지+labelme JSON zip을 받아 data/external/ttpla/ 에 압축 해제 (1,100장, 3840×2160)"""


def convert_shapes(doc: dict, cfg: dict, names: list[str]) -> list[tuple[int, np.ndarray]]:
    """labelme JSON → [(class id, 정규화 polygon)]. 매핑 없는 라벨·활성 클래스셋에 없는 클래스·작은 polygon은 버림."""
    w, h = doc["imageWidth"], doc["imageHeight"]
    out = []
    for shape in doc["shapes"]:
        target = cfg["ttpla"]["map"].get(shape["label"])
        pts = np.asarray(shape["points"], np.float32)
        if target not in names or len(pts) < 3:
            continue
        if cv2.contourArea(pts) < cfg["min_area_ratio"][target] * w * h:
            continue
        out.append((names.index(target), pts / [w, h]))
    return out


def convert_ttpla(src: str | Path, out: str | Path, cfg: dict | None = None, names: list[str] | None = None) -> int:
    """변환한 이미지 수를 반환. 데이터가 없으면 안내만 출력하고 0."""
    cfg = cfg or load_config("dataset")
    names = names or class_names()
    src, out = Path(src), Path(out)
    if source_missing(src, GUIDE):
        return 0
    n = 0
    for js in sorted(src.rglob("*.json")):
        doc = json.loads(js.read_text(encoding="utf-8"))
        if "shapes" not in doc:
            continue  # COCO 등 다른 JSON
        img_path = next((p for p in js.parent.glob(f"{js.stem}.*") if p.suffix.lower() in IMAGE_EXTS), None)
        instances = convert_shapes(doc, cfg, names)
        if img_path is None or not instances:
            continue
        img = cv2.imread(str(img_path))
        scale = cfg["ttpla"]["max_side"] / max(img.shape[:2])
        if scale < 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        (out / "images").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "images" / f"{js.stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        write_label(out / "labels" / f"{js.stem}.txt", instances)
        n += 1
    return n


def main(
    src: Path = typer.Option(Path("data/external/ttpla")),
    out: Path = typer.Option(Path("data/yolo/ttpla")),
    classes: str = typer.Option(None, help="v1 | v2 (config/classes.yaml, 기본 active)"),
) -> None:
    print(f"{convert_ttpla(src, out, names=class_names(classes))} images → {out}")


if __name__ == "__main__":
    typer.run(main)
