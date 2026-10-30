"""박스 라벨 YOLO export → SAM 마스크 → YOLO-seg polygon. 원본이 박스뿐인 공개 데이터(예: Street Trees [R29] → vegetation).

python -m facade_cv.detect.datasets.box_to_seg_sam --src data/public/roboflow --out data/yolo --debug
--src/<키>/ 아래 Roboflow YOLO export (data.yaml + train|valid|test/{images,labels}, 상위 폴더 하나 더 있어도 됨).
키·라이선스·클래스 매핑·SAM 모델·품질 필터는 config/dataset.yaml box_to_seg. 출력 <out>/<키>/images, labels (merge --ext <키>=<out>/<키>).
- 박스 줄 → 박스를 프롬프트로 SAM 마스크 → 박스 안으로 자름 → 면적 필터(min_fill~max_fill) → 큰 조각들 → polygon
- 원본 polygon 줄(사람이 그린 거친 다각형)은 SAM 없이 그대로
- Roboflow 증강 사본(`<원본>.rf.<해시>`)은 원본당 1장만 (valid/test = 증강 없는 원본 우선)
--debug: 샘플 이미지에 박스(빨강)·SAM polygon(초록)·원본 polygon(하늘색)을 그려 --debug-dir에 저장
"""

import random
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import typer
import yaml

from facade_cv.config import class_names, load_config
from facade_cv.detect.datasets.yolo import list_images, mask_to_polygons, source_missing, write_label

GUIDE = """Roboflow Universe (계정 필요, 데이터셋 페이지 → Download Dataset → 형식 "YOLOv11" → zip):
  config/dataset.yaml box_to_seg.datasets의 url마다 받아 data/public/roboflow/<키>/ 에 압축 해제"""

# (H,W,3) BGR 이미지, [(x0,y0,x1,y1)] 픽셀 박스 → 박스마다 (H,W) bool 마스크
Segmenter = Callable[[np.ndarray, list[tuple[float, float, float, float]]], list[np.ndarray]]


def sam_segmenter(model: str) -> Segmenter:
    from ultralytics import SAM

    from facade_cv.device import get_device

    weights = Path(__file__).resolve().parents[3] / "weights" / model
    sam, device = SAM(str(weights) if weights.exists() else model), get_device()

    def run(img: np.ndarray, boxes: list) -> list[np.ndarray]:
        r = sam(img, bboxes=[list(b) for b in boxes], device=device, verbose=False)[0]
        return [m.astype(bool) for m in r.masks.data.cpu().numpy()]

    return run


def mask_to_instance(mask: np.ndarray, box: tuple, cfg: dict, eps: float) -> tuple[list[np.ndarray], str]:
    """SAM 마스크 하나 → (polygon 목록 (픽셀), 이유). 이유: kept | too_small | too_full | no_polygon.
    마스크는 박스 안으로 자른다. 면적 비율 = 마스크 / 박스. 조각은 가장 큰 것 + 박스 면적 min_piece 이상만."""
    h, w = mask.shape
    x0, y0, x1, y1 = (int(round(v)) for v in (max(box[0], 0), max(box[1], 0), min(box[2], w), min(box[3], h)))
    box_area = max((x1 - x0) * (y1 - y0), 1)
    m = np.zeros((h, w), np.uint8)
    m[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    fill = m.sum() / box_area
    if fill < cfg["min_fill"]:
        return [], "too_small"
    if fill > cfg["max_fill"]:
        return [], "too_full"
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i, a in enumerate(areas) if a == areas.max() or a >= cfg["min_piece"] * box_area]
    polys = mask_to_polygons(np.isin(lab, keep), eps, 0)
    return (polys, "kept") if polys else ([], "no_polygon")


def parse_label(text: str, w: int, h: int) -> tuple[list[tuple[int, tuple]], list[tuple[int, np.ndarray]]]:
    """YOLO 라벨 → (박스 [(원본 cls, 픽셀 xyxy)], polygon [(원본 cls, 정규화 (N,2))])."""
    boxes, polys = [], []
    for line in text.splitlines():
        v = line.split()
        if len(v) == 5:
            cx, cy, bw, bh = (float(x) for x in v[1:])
            boxes.append((int(v[0]), ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)))
        elif len(v) >= 7:
            polys.append((int(v[0]), np.array(v[1:], float).reshape(-1, 2)))
    return boxes, polys


def originals(root: Path) -> list[tuple[Path, Path]]:
    """증강 사본을 원본 하나로: 원본 이름(`.rf.` 앞)마다 (이미지, 라벨) 1쌍. valid/test(증강 없음) → train 순."""
    pick: dict[str, tuple[Path, Path]] = {}
    for split in ["valid", "test", "train"]:
        if not (root / split / "images").is_dir():
            continue
        for img in list_images(root / split / "images"):
            pick.setdefault(re.sub(r"\.rf\.[0-9a-f]+$", "", img.stem), (img, root / split / "labels" / f"{img.stem}.txt"))
    return [pick[k] for k in sorted(pick)]


def convert_dataset(root: Path, out: Path, spec: dict, cfg: dict, eps: float, names: list[str], segment: Segmenter,
                    debug_dir: Path | None = None, debug_n: int = 12) -> Counter:
    """변환 통계 Counter 반환 (images, sam_kept/too_small/..., polygon_passthrough, no_label 등)."""
    data = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    src_names = list(data["names"].values()) if isinstance(data["names"], dict) else list(data["names"])
    target = {i: names.index(spec["map"][n]) for i, n in enumerate(src_names) if spec["map"].get(n) in names}
    st, drawn = Counter(), []
    pairs = originals(root)
    st["copies_skipped"] = sum(len(list_images(p / "images")) for p in root.iterdir() if (p / "images").is_dir()) - len(pairs)
    for img_path, lab_path in pairs:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        boxes, polys = parse_label(lab_path.read_text(encoding="utf-8") if lab_path.exists() else "", w, h)
        boxes = [(target[c], b) for c, b in boxes if c in target]
        polys = [(target[c], p) for c, p in polys if c in target]
        if not boxes and not polys:
            st["no_label"] += 1
            continue
        inst = list(polys)
        st["polygon_passthrough"] += len(polys)
        sam_polys = []
        if boxes:
            for (cls, box), mask in zip(boxes, segment(img, [b for _, b in boxes])):
                got, why = mask_to_instance(mask, box, cfg, eps)
                st[f"sam_{why}"] += 1
                st["sam_pieces"] += len(got)
                inst += [(cls, p / [w, h]) for p in got]
                sam_polys += got
        if not inst:
            st["image_all_dropped"] += 1
            continue
        name = f"{img_path.parent.parent.name}_{img_path.name}"
        (out / "images").mkdir(parents=True, exist_ok=True)
        (out / "images" / name).write_bytes(img_path.read_bytes())
        write_label(out / "labels" / f"{Path(name).stem}.txt", inst)
        st["images"] += 1
        if debug_dir is not None:
            drawn.append((img_path, [b for _, b in boxes], sam_polys, [p * [w, h] for _, p in polys]))
    if debug_dir is not None:
        save_debug(drawn, debug_dir, out.name, debug_n)
    return st


def save_debug(items: list, debug_dir: Path, key: str, n: int) -> None:
    """박스 있는 이미지 위주로 n장 (seed 0)."""
    rng = random.Random(0)
    with_box = [it for it in items if it[1]]
    pick = rng.sample(with_box, min(n, len(with_box)))
    pick += rng.sample([it for it in items if not it[1]], min(n - len(pick), len(items) - len(with_box)))
    debug_dir.mkdir(parents=True, exist_ok=True)
    stem = key.removeprefix("veg_")
    for i, (img_path, boxes, sam_polys, src_polys) in enumerate(pick, 1):
        im = cv2.imread(str(img_path))
        for x0, y0, x1, y1 in boxes:
            cv2.rectangle(im, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)
        ov = im.copy()
        for p in sam_polys:
            cv2.fillPoly(ov, [p.astype(np.int32)], (0, 200, 0))
        im = cv2.addWeighted(ov, 0.35, im, 0.65, 0)
        for p in sam_polys:
            cv2.polylines(im, [p.astype(np.int32)], True, (0, 255, 0), 2)
        for p in src_polys:
            cv2.polylines(im, [p.astype(np.int32)], True, (255, 200, 0), 2)
        cv2.imwrite(str(debug_dir / f"{stem}_{i:02d}.jpg"), im)


def convert_box_to_seg(src: str | Path, out: str | Path, cfg: dict | None = None, names: list[str] | None = None,
                       segment: Segmenter | None = None, debug_dir: Path | None = None) -> dict[str, Counter]:
    """{데이터셋 키: 통계}. 폴더가 없는 데이터셋은 안내 후 빈 통계. segment 없으면 config sam_model로 SAM."""
    full = cfg or load_config("dataset")
    bc, names = full["box_to_seg"], names or class_names()
    licenses = full["roboflow"]["licenses"]
    src, out = Path(src), Path(out)
    stats = {}
    for key, spec in bc["datasets"].items():
        if spec["license"] not in licenses:
            raise ValueError(f"{key}: license {spec['license']!r} not in {licenses} — CC BY 계열만 사용")
        d = src / key
        if source_missing(d, f"{GUIDE}\n  {key}: {spec['url']} ({spec['license']})"):
            stats[key] = Counter()
            continue
        segment = segment or sam_segmenter(bc["sam_model"])
        root = next(p.parent for p in d.rglob("data.yaml"))
        stats[key] = convert_dataset(root, out / key, spec, bc, full["polygon_eps"], names, segment, debug_dir)
    return stats


def main(
    src: Path = typer.Option(Path("data/public/roboflow")),
    out: Path = typer.Option(Path("data/yolo")),
    classes: str = typer.Option(None, help="v1 | v2 (config/classes.yaml, 기본 active)"),
    debug: bool = typer.Option(False, help="샘플 12장에 박스·polygon을 그려 --debug-dir에 저장"),
    debug_dir: Path = typer.Option(Path("runs/convert_check")),
) -> None:
    for key, st in convert_box_to_seg(src, out, names=class_names(classes), debug_dir=debug_dir if debug else None).items():
        sam = {k: v for k, v in sorted(st.items()) if k.startswith("sam_")}
        print(f"{key}: {st['images']} images → {out / key}")
        print(f"  SAM 박스 {sum(v for k, v in sam.items() if k != 'sam_pieces')}개: {sam}")
        print(f"  원본 polygon 그대로 {st['polygon_passthrough']}, 라벨 없음 {st['no_label']}, "
              f"전부 버려진 이미지 {st['image_all_dropped']}, 증강 사본 제외 {st['copies_skipped']}")


if __name__ == "__main__":
    typer.run(main)
