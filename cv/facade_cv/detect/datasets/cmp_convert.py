"""CMP Facade [R1] (base+extended, CC BY-SA 3.0) → YOLO-seg (0=window, 1=facade). 변환 규칙 표: data/DATASETS.md

python -m facade_cv.detect.datasets.cmp_convert --src data/cmp --out data/yolo/cmp
--src 아래를 재귀 탐색해 같은 이름의 .jpg/.png/.xml 묶음을 변환. 출력: <out>/images/*.jpg, <out>/labels/*.txt
"""

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import typer
from PIL import Image

from facade_cv.config import load_config
from facade_cv.detect.datasets.yolo import mask_to_polygons, simplify_polygon, write_label


def read_xml_rects(path: Path, labelname: str) -> list[tuple[float, float, float, float]]:
    """CMP XML (루트 없는 <object> 나열) → labelname 객체의 정규화 사각형 (x0, y0, x1, y1), x=가로.

    주의: CMP XML의 <x>는 세로(행), <y>는 가로(열)다. 라벨 PNG와 대조해 확인함 (반대로 읽으면 IoU 0.09 vs 0.91).
    """
    root = ET.fromstring(f"<root>{path.read_text(encoding='utf-8')}</root>")
    rects = []
    for obj in root.iter("object"):
        if (obj.findtext("labelname") or "").strip() != labelname:
            continue
        rows = [float(v.text) for v in obj.iter("x")]
        cols = [float(v.text) for v in obj.iter("y")]
        rects.append((min(cols), min(rows), max(cols), max(rows)))
    return rects


def convert_label(label: np.ndarray, window_rects: list, cfg: dict) -> list[tuple[int, np.ndarray]]:
    """CMP 팔레트 index 라벨 (H,W) + XML 창문 사각형 → [(class id, 정규화 polygon)]."""
    cmp, names, eps = cfg["cmp"], cfg["names"], cfg["polygon_eps"]
    index = {int(k): v for k, v in cmp["index"].items()}
    h, w = label.shape
    min_area = {k: v * h * w for k, v in cfg["min_area_ratio"].items()}
    instances = []

    window_px = np.isin(label, [i for i, n in index.items() if n in cmp["window"]])
    for x0, y0, x1, y1 in window_rects:
        c0, r0 = int(np.floor(x0 * w)), int(np.floor(y0 * h))
        c1, r1 = int(np.ceil(x1 * w)), int(np.ceil(y1 * h))
        m = np.zeros_like(window_px)
        m[r0:r1, c0:c1] = window_px[r0:r1, c0:c1]
        polys = mask_to_polygons(m, eps, 0)
        if not polys or cv2.countNonZero(m.astype(np.uint8)) < min_area["window"]:
            continue
        if len(polys) > 1:  # 사각형 안이 여러 조각이면(가림 등) 볼록 껍질로 하나로 — 사각형 밖으로는 안 나간다
            polys = [simplify_polygon(cv2.convexHull(np.vstack(polys).astype(np.float32))[:, 0], eps)]
        instances.append((names.index("window"), polys[0] / [w, h]))

    facade_ids = [i for i, n in index.items() if n in cmp["facade"]]
    for poly in mask_to_polygons(np.isin(label, facade_ids), eps, min_area["facade"]):
        instances.append((names.index("facade"), poly / [w, h]))
    return instances


def convert_cmp(src: str | Path, out: str | Path, cfg: dict | None = None) -> int:
    """변환한 이미지 수를 반환."""
    cfg = cfg or load_config("dataset")
    src, out = Path(src), Path(out)
    known = {int(k) for k in cfg["cmp"]["index"]}
    unknown: set[int] = set()
    n = 0
    for png in sorted(src.rglob("*.png")):
        jpg, xml = png.with_suffix(".jpg"), png.with_suffix(".xml")
        if not jpg.exists():
            continue
        im = Image.open(png)
        if im.mode != "P":  # cv2.imread는 팔레트를 RGB로 풀어버리므로 PIL로 index를 읽는다
            raise ValueError(f"{png}: palette(P) PNG expected, got {im.mode}")
        label = np.array(im)
        unknown |= set(np.unique(label).tolist()) - known
        if not xml.exists():
            raise FileNotFoundError(f"{xml}: CMP XML needed for window rectangles")
        rects = read_xml_rects(xml, cfg["cmp"]["window_xml_label"])
        write_label(out / "labels" / f"{png.stem}.txt", convert_label(label, rects, cfg))
        (out / "images").mkdir(parents=True, exist_ok=True)
        shutil.copy2(jpg, out / "images" / jpg.name)
        n += 1
    if unknown:
        print(f"warning: unknown label index {sorted(unknown)} ignored")
    return n


def main(src: Path = typer.Option(..., exists=True, file_okay=False), out: Path = typer.Option(...)) -> None:
    print(f"{convert_cmp(src, out)} images → {out}")


if __name__ == "__main__":
    typer.run(main)
