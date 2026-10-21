"""Label Studio JSON export → YOLO-seg.

python -m facade_cv.detect.datasets.label_studio.export_to_yolo --export export.json --images data/own/raw --out data/yolo/own
Export 형식은 "JSON" (JSON-MIN 아님). 이미지 파일은 --images에서 파일명으로 찾는다.
"""

import json
import re
import shutil
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import typer

from facade_cv.config import class_names
from facade_cv.detect.datasets.yolo import write_label

FROM_NAME = "label"  # labeling_config.xml의 PolygonLabels name
UPLOAD_PREFIX = re.compile(r"^[0-9a-f]{8}-")  # LS 업로드 시 파일명 앞에 붙는 해시


def find_image(url: str, images_dir: Path) -> Path | None:
    """/data/upload/1/8f3a2b1c-a.jpg, /data/local-files/?d=x/a.jpg, s3://.../a.jpg → images_dir/a.jpg"""
    name = Path(unquote(url.split("?d=")[-1])).name
    for cand in (name, UPLOAD_PREFIX.sub("", name)):
        if (images_dir / cand).is_file():
            return images_dir / cand
    return None


def task_instances(task: dict, names: list[str]) -> list[tuple[int, np.ndarray]] | None:
    """최신 비취소 annotation의 polygon. annotation이 없으면 None (미라벨 task)."""
    anns = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
    if not anns:
        return None
    ann = max(anns, key=lambda a: a.get("updated_at") or a.get("created_at") or "")
    instances = []
    for r in ann["result"]:
        if r.get("type") != "polygonlabels" or r.get("from_name", FROM_NAME) != FROM_NAME:
            continue
        label = r["value"]["polygonlabels"][0]
        if label not in names:
            print(f"warning: task {task.get('id')}: unknown label {label!r} skipped")
            continue
        instances.append((names.index(label), np.array(r["value"]["points"], float) / 100))
    return instances


def export_to_yolo(
    export_json: str | Path, images_dir: str | Path, out: str | Path, names: list[str] | None = None
) -> int:
    """변환한 이미지 수를 반환. 이미지를 못 찾은 task는 경고 후 건너뜀. names 기본: config/classes.yaml active."""
    names = names or class_names()
    images_dir, out = Path(images_dir), Path(out)
    tasks = json.loads(Path(export_json).read_text(encoding="utf-8"))
    n, missing = 0, []
    for task in tasks:
        instances = task_instances(task, names)
        if instances is None:
            continue
        img = find_image(task["data"]["image"], images_dir)
        if img is None:
            missing.append(task["data"]["image"])
            continue
        write_label(out / "labels" / f"{img.stem}.txt", instances)
        (out / "images").mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, out / "images" / img.name)
        n += 1
    if missing:
        print(f"warning: {len(missing)} images not found in {images_dir}: {missing[:5]}")
    return n


def main(
    export: Path = typer.Option(..., exists=True, dir_okay=False),
    images: Path = typer.Option(..., exists=True, file_okay=False),
    out: Path = typer.Option(...),
    classes: str = typer.Option(None, help="v1 | v2 (config/classes.yaml, 기본 active)"),
) -> None:
    print(f"{export_to_yolo(export, images, out, class_names(classes))} images → {out}")


if __name__ == "__main__":
    typer.run(main)
