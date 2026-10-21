"""ADE20K SceneParsing [R19] (ADEChallengeData2016, 비상업 연구·교육용) → YOLO-seg. 변환 규칙: data/DATASETS.md

python -m facade_cv.detect.datasets.ade20k_convert --src data/external/ade20k --out data/yolo/ade20k
--src 아래 ADEChallengeData2016/{objectInfo150.csv, sceneCategories.txt, images/*, annotations/*} (zip 압축 해제 그대로).
실외 장면(config ade20k.outdoor_scenes) + sky 비율 ≥ min_sky_ratio + building 비율 ≥ min_facade_ratio + window 있음만 사용.
라벨 PNG 픽셀값 = objectInfo150.csv Idx (1~150, 0 = 기타). 클래스 이름은 Name 열 첫 이름 (windowpane;window → windowpane).
"""

import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
import typer

from facade_cv.config import class_names, load_config
from facade_cv.detect.datasets.yolo import mask_to_polygons, source_missing, write_label

GUIDE = """ADE20K SceneParsing (가입 불필요, 923MB zip, 해제 ~2.3GB. 이미지는 비상업 연구·교육용만 [R19]):
  mkdir -p data/external/ade20k && cd data/external/ade20k
  curl -LO http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip && unzip -q ADEChallengeData2016.zip"""


def read_object_info(path: Path) -> dict[str, int]:
    """objectInfo150.csv → {첫 이름: 라벨 index}."""
    with open(path, newline="", encoding="utf-8") as f:
        return {r["Name"].split(";")[0].strip(): int(r["Idx"]) for r in csv.DictReader(f)}


def read_scenes(path: Path) -> dict[str, str]:
    """sceneCategories.txt (`ADE_train_00000001 airport_terminal`) → {stem: 장면}."""
    return dict(line.split()[:2] for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def convert_label(label: np.ndarray, index: dict[str, int], cfg: dict, names: list[str]) -> list[tuple[int, np.ndarray]] | None:
    """라벨 (H,W) → [(class id, 정규화 polygon)]. 선별 조건(sky·building 비율, require)을 못 넘으면 None.
    인스턴스 = 대상 클래스 마스크의 연결 성분."""
    ac, eps = cfg["ade20k"], cfg["polygon_eps"]
    h, w = label.shape
    if (label == index["sky"]).mean() < ac["min_sky_ratio"]:
        return None
    if (label == index["building"]).mean() < ac["min_facade_ratio"]:
        return None
    instances = []
    for target in names:
        src_ids = [index[s] for s, t in ac["map"].items() if t == target]
        if not src_ids:
            continue
        # ponytail: 연결 성분 = 인스턴스 — 맞붙은 창문은 하나로 합쳐진다. 정확히 하려면 ADE20K full의 instance 마스크로
        for poly in mask_to_polygons(np.isin(label, src_ids), eps, cfg["min_area_ratio"][target] * h * w):
            instances.append((names.index(target), poly / [w, h]))
    have = {names[c] for c, _ in instances}
    if not set(ac["require"]) <= have:
        return None
    return instances


def convert_ade20k(src: str | Path, out: str | Path, cfg: dict | None = None, names: list[str] | None = None) -> int:
    """변환한 이미지 수를 반환. 데이터가 없으면 안내만 출력하고 0."""
    cfg = cfg or load_config("dataset")
    names = names or class_names()
    src, out = Path(src), Path(out)
    if source_missing(src, GUIDE):
        return 0
    root = next(p.parent for p in src.rglob("objectInfo150.csv"))
    index = read_object_info(root / "objectInfo150.csv")
    missing = ({"sky", "building"} | set(cfg["ade20k"]["map"])) - set(index)
    if missing:
        raise ValueError(f"config ade20k.map 이름 {sorted(missing)}이 objectInfo150.csv에 없음")
    scenes = read_scenes(root / "sceneCategories.txt")
    outdoor = set(cfg["ade20k"]["outdoor_scenes"])
    jpgs = {p.stem: p for p in (root / "images").rglob("*.jpg")}
    n = 0
    for png in sorted((root / "annotations").rglob("*.png")):
        if scenes.get(png.stem) not in outdoor:
            continue
        jpg = jpgs.get(png.stem)
        if jpg is None:
            continue
        instances = convert_label(cv2.imread(str(png), cv2.IMREAD_UNCHANGED), index, cfg, names)
        if instances is None:
            continue
        write_label(out / "labels" / f"{png.stem}.txt", instances)
        (out / "images").mkdir(parents=True, exist_ok=True)
        shutil.copy2(jpg, out / "images" / jpg.name)
        n += 1
    return n


def main(
    src: Path = typer.Option(Path("data/external/ade20k")),
    out: Path = typer.Option(Path("data/yolo/ade20k")),
    classes: str = typer.Option(None, help="v1 | v2 (config/classes.yaml, 기본 active)"),
) -> None:
    print(f"{convert_ade20k(src, out, names=class_names(classes))} images → {out}")


if __name__ == "__main__":
    typer.run(main)
