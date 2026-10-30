"""Roboflow Universe [R21] YOLO-seg export → 클래스 재매핑. config/dataset.yaml roboflow.datasets에 등록된 것만, 라이선스 화이트리스트만.

python -m facade_cv.detect.datasets.roboflow_convert --src data/public/roboflow --out data/yolo/roboflow
--src/<키>/ 에 Roboflow "YOLOv8"(또는 YOLO11) 형식 export를 압축 해제 (data.yaml + train|valid|test/{images,labels}).
출력: <out>/<키>/images, labels (merge --ext roboflow_<키>=<out>/<키>). 출력 파일명 = <split>_<원본 이름> (split 간 이름 충돌 방지).
semantic 데이터셋도 YOLO-seg export는 polygon이라 같은 방식 (연결 성분이 인스턴스).
"""

from pathlib import Path

import numpy as np
import typer
import yaml

from facade_cv.config import class_names, load_config
from facade_cv.detect.datasets.yolo import list_images, source_missing, write_label

GUIDE = """Roboflow Universe (계정 필요, 데이터셋 페이지 → Download Dataset → 형식 "YOLOv8" (segmentation) → zip):
  config/dataset.yaml roboflow.datasets의 url마다 받아 data/public/roboflow/<키>/ 에 압축 해제
  (라이선스가 CC BY 계열인지 페이지에서 다시 확인하고 config license 칸과 맞출 것)"""


def remap_label(text: str, src_names: list[str], mapping: dict, names: list[str]) -> list[tuple[int, np.ndarray]]:
    """YOLO-seg 라벨 텍스트 (원본 클래스 id) → [(우리 class id, polygon)]. 매핑 없는 클래스, bbox 줄(점 2개)은 버림."""
    out = []
    for line in text.splitlines():
        v = line.split()
        if len(v) < 7:  # cls + 3점 미만 = 검출 bbox 형식이거나 깨진 줄
            continue
        target = mapping.get(src_names[int(v[0])])
        if target in names:
            out.append((names.index(target), np.array(v[1:], float).reshape(-1, 2)))
    return out


def convert_dataset(src: Path, out: Path, spec: dict, licenses: list[str], names: list[str]) -> int:
    if spec["license"] not in licenses:
        raise ValueError(f"{src.name}: license {spec['license']!r} not in {licenses} — CC BY 계열만 사용")
    data = yaml.safe_load((src / "data.yaml").read_text(encoding="utf-8"))
    src_names = list(data["names"].values()) if isinstance(data["names"], dict) else list(data["names"])
    unknown = set(src_names) - set(spec["map"])
    if unknown:
        print(f"{src.name}: 매핑 없는 클래스 버림 {sorted(unknown)}")
    n = 0
    for split_dir in sorted(p for p in src.iterdir() if (p / "images").is_dir()):
        for img in list_images(split_dir / "images"):
            lab = split_dir / "labels" / f"{img.stem}.txt"
            inst = remap_label(lab.read_text(encoding="utf-8"), src_names, spec["map"], names) if lab.exists() else []
            if not inst:
                continue
            name = f"{split_dir.name}_{img.name}"
            (out / "images").mkdir(parents=True, exist_ok=True)
            (out / "images" / name).write_bytes(img.read_bytes())
            write_label(out / "labels" / f"{Path(name).stem}.txt", inst)
            n += 1
    return n


def convert_roboflow(
    src: str | Path, out: str | Path, cfg: dict | None = None, names: list[str] | None = None
) -> dict[str, int]:
    """{데이터셋 키: 변환 이미지 수}. 폴더가 없는 데이터셋은 안내 후 0."""
    cfg = cfg or load_config("dataset")
    names = names or class_names()
    rc, src, out = cfg["roboflow"], Path(src), Path(out)
    counts = {}
    for key, spec in rc["datasets"].items():
        d = src / key
        if source_missing(d, f"{GUIDE}\n  {key}: {spec['url']} ({spec['license']})"):
            counts[key] = 0
            continue
        root = next(p.parent for p in d.rglob("data.yaml"))  # zip 안 상위 폴더가 있든 없든
        counts[key] = convert_dataset(root, out / key, spec, rc["licenses"], names)
    return counts


def main(
    src: Path = typer.Option(Path("data/public/roboflow")),
    out: Path = typer.Option(Path("data/yolo/roboflow")),
    classes: str = typer.Option(None, help="v1 | v2 (config/classes.yaml, 기본 active)"),
) -> None:
    for k, n in convert_roboflow(src, out, names=class_names(classes)).items():
        print(f"{k}: {n} images → {out / k}")


if __name__ == "__main__":
    typer.run(main)
