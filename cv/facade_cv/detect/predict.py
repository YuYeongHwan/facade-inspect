"""학습된 YOLO-seg 가중치로 이미지 폴더 예측 → 오버레이 이미지 저장 (로컬 CPU/MPS 가능).

python -m facade_cv.detect.predict --images runs/<id>/frames  (--weights 기본: config/detect.yaml)
--images는 io.extract 출력 폴더(<flightID>__frame_*.jpg + frames.json)를 그대로 받는다 — 이미지가 아닌 파일은 무시.
출력: --out (기본 <images>/../pred)/<원본 stem>.jpg — 마스크 오버레이 + 클래스별 개수
"""

from pathlib import Path

import cv2
import typer

from facade_cv.detect.datasets.yolo import list_images
from facade_cv.detect.segmenter import WindowSegmenter, draw_detections, put_label


def predict_folder(
    weights: str | Path | None,
    images_dir: str | Path,
    out_dir: str | Path | None = None,
    cfg: dict | None = None,
    device: str | None = None,
) -> list[dict]:
    """이미지마다 오버레이를 저장하고 [{"image", <클래스명>: 개수, ...}] 반환. weights=None이면 config/detect.yaml."""
    images_dir = Path(images_dir)
    images = list_images(images_dir)
    if not images:
        raise FileNotFoundError(f"no images in {images_dir}")
    out_dir = Path(out_dir) if out_dir else images_dir.parent / "pred"
    out_dir.mkdir(parents=True, exist_ok=True)

    seg = WindowSegmenter(weights, cfg, device)
    rows = []
    for path in images:
        img = cv2.imread(str(path))
        dets = seg.predict(img)
        counts = {name: sum(d.cls == name for d in dets) for name in seg.model.names.values()}
        vis = draw_detections(img, dets)
        label = "  ".join(f"{k} {v}" for k, v in counts.items())
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(vis, (0, 0), (tw + 20, th + base + 20), (0, 0, 0), -1)
        put_label(vis, label, (10, th + 10), 0.8)
        cv2.imwrite(str(out_dir / f"{path.stem}.jpg"), vis)
        rows.append({"image": path.name, **counts})
    return rows


def main(
    weights: Path = typer.Option(None, exists=True, dir_okay=False, help="기본: config/detect.yaml weights"),
    images: Path = typer.Option(..., exists=True, file_okay=False, help="이미지 폴더 (io.extract 출력 폴더 그대로 가능)"),
    out: Path = typer.Option(None, help="오버레이 저장 폴더 (기본: <images>/../pred)"),
    device: str = typer.Option(None, help="cpu / mps / 0 (기본: config/detect.yaml, auto = cuda > mps > cpu)"),
) -> None:
    rows = predict_folder(weights, images, out, device=device)
    totals = {k: sum(r[k] for r in rows) for k in rows[0] if k != "image"}
    print(f"{len(rows)} images → {out or images.parent / 'pred'} | total {totals}")


if __name__ == "__main__":
    typer.run(main)
