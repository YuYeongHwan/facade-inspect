"""자체 시험셋에서 분할 모델 비교 (예: v1 vs v2): 창문 mask mAP, 클래스별 지표, 장애물 위 창문 오검출 수, 예시 그리드.

python -m facade_cv.eval.compare --data data/yolo/merged/data.yaml \
    --weights cv/weights/seg_window_facade_v1.pt --weights cv/weights/seg_window_facade_v2.pt --out runs/eval
- 평가 대상은 data.yaml의 test split (merge_datasets가 자체 데이터만 넣는다)
- 모델마다 라벨을 그 모델이 아는 클래스만 남긴 임시 복사본으로 Ultralytics val → 2클래스 v1도 같은 이미지로 비교
- 오검출: conf 이상 창문 예측 중 GT 창문과 IoU < iou_thr 이고, GT 장애물(door·vegetation·wire·pole) 마스크와
  (예측 면적 대비) overlap_thr 이상 겹치는 것 → 장애물 클래스별 개수. raw(모델 출력)와 post(detect.postprocess 후) 둘 다
출력: <out>/<날짜_시각>/metrics.json, table.md, grid.jpg (오검출 차이가 큰 순). table.md는 cv/weights/MODEL_CARD.md에도 append
"""

import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import typer
import yaml

from facade_cv.config import CONFIG_DIR, load_config
from facade_cv.detect.datasets.yolo import list_images

MODEL_CARD = CONFIG_DIR.parent / "cv" / "weights" / "MODEL_CARD.md"


def read_gt(label: Path, names: list[str], shape: tuple[int, int]) -> dict[str, list[np.ndarray]]:
    """YOLO-seg 라벨 → {클래스 이름: [bool 마스크]}."""
    h, w = shape
    out: dict[str, list[np.ndarray]] = {}
    if not label.exists():
        return out
    for line in label.read_text().splitlines():
        v = line.split()
        if len(v) < 7:
            continue
        m = np.zeros(shape, np.uint8)
        cv2.fillPoly(m, [np.round(np.array(v[1:], float).reshape(-1, 2) * [w, h]).astype(np.int32)], 1)
        out.setdefault(names[int(v[0])], []).append(m.astype(bool))
    return out


def obstacle_false_positives(
    pred: list[np.ndarray], gt: dict[str, list[np.ndarray]], obstacles: list[str], iou_thr: float, overlap_thr: float
) -> dict[str, int]:
    """창문 예측 마스크 중 GT 창문과 안 맞고(IoU < iou_thr) GT 장애물과 overlap_thr 이상 겹치는 것 → {장애물: 개수}.
    여러 장애물에 걸치면 가장 많이 겹친 클래스 하나로 센다."""
    counts = dict.fromkeys(obstacles, 0)
    union = {c: np.logical_or.reduce(gt[c]) for c in obstacles if gt.get(c)}
    for p in pred:
        area = p.sum()
        if not area:
            continue
        ious = [(p & g).sum() / (p | g).sum() for g in gt.get("window", [])]
        if max(ious, default=0.0) >= iou_thr:
            continue
        ov = {c: (p & u).sum() / area for c, u in union.items()}
        hit = [c for c, r in ov.items() if r >= overlap_thr]
        if hit:
            counts[max(hit, key=ov.get)] += 1
    return counts


def _quad_masks(windows, shape) -> list[np.ndarray]:
    out = []
    for w in windows:
        m = np.zeros(shape, np.uint8)
        cv2.fillConvexPoly(m, np.round(w.corners).astype(np.int32), 1)
        out.append(m.astype(bool))
    return out


def subset_dataset(data: dict, root: Path, model_names: list[str], dst: Path) -> Path:
    """test split을 모델이 아는 클래스만 남기고 모델 id로 바꾼 임시 데이터셋 → data yaml 경로."""
    names = list(data["names"].values()) if isinstance(data["names"], dict) else list(data["names"])
    remap = {i: model_names.index(n) for i, n in enumerate(names) if n in model_names}
    (dst / "images" / "test").mkdir(parents=True, exist_ok=True)
    (dst / "labels" / "test").mkdir(parents=True, exist_ok=True)
    for img in list_images(root / data["test"]):
        (dst / "images" / "test" / img.name).symlink_to(img.resolve())
        lab = root / "labels" / "test" / f"{img.stem}.txt"
        lines = [line.split() for line in lab.read_text().splitlines() if line.split()] if lab.exists() else []
        kept = [" ".join([str(remap[int(v[0])]), *v[1:]]) for v in lines if int(v[0]) in remap]
        (dst / "labels" / "test" / f"{img.stem}.txt").write_text("".join(f"{k}\n" for k in kept))
    y = {"path": str(dst.resolve()), "train": "images/test", "val": "images/test", "test": "images/test",
         "names": dict(enumerate(model_names))}
    (dst / "data.yaml").write_text(yaml.safe_dump(y, allow_unicode=True, sort_keys=False))
    return dst / "data.yaml"


def class_metrics(model, data_yaml: Path, imgsz: int, device) -> dict[str, dict[str, float]]:
    m = model.val(data=str(data_yaml), split="test", imgsz=imgsz, batch=8, device=device, plots=False, verbose=False)
    out = {}
    for i, c in enumerate(m.ap_class_index):
        p, r, ap50, ap = m.seg.class_result(i)
        out[model.names[int(c)]] = {"mask_P": float(p), "mask_R": float(r), "mask_mAP50": float(ap50),
                                    "mask_mAP50-95": float(ap)}
    return out


def table_md(results: dict, obstacles: list[str], n_images: int) -> str:
    head = "| 모델 | window mAP50 | window mAP50-95 | " + " | ".join(f"오검출 {c} (raw/post)" for c in obstacles) + " | 합계 (raw/post) |"
    rows = [head, "|" + "---|" * (len(obstacles) + 4)]
    for name, r in results.items():
        w = r["classes"].get("window", {})
        fp = " | ".join(f"{r['fp_raw'][c]}/{r['fp_post'][c]}" for c in obstacles)
        rows.append(f"| {name} | {w.get('mask_mAP50', float('nan')):.4f} | {w.get('mask_mAP50-95', float('nan')):.4f} | {fp} | "
                    f"{sum(r['fp_raw'].values())}/{sum(r['fp_post'].values())} |")
    per_class = ["", "클래스별 mask mAP50 / mAP50-95:", "", "| 모델 | 클래스 | P | R | mAP50 | mAP50-95 |", "|---|---|---|---|---|---|"]
    for name, r in results.items():
        per_class += [f"| {name} | {c} | {v['mask_P']:.3f} | {v['mask_R']:.3f} | {v['mask_mAP50']:.4f} | {v['mask_mAP50-95']:.4f} |"
                      for c, v in r["classes"].items()]
    return f"자체 test {n_images}장.\n\n" + "\n".join(rows + per_class) + "\n"


def compare(data_yaml: str | Path, weights: list[str | Path], out: str | Path, grid_n: int = 16,
            device: str | None = None) -> dict:
    from facade_cv.detect.postprocess import postprocess
    from facade_cv.detect.segmenter import WindowSegmenter, draw_detections, put_label

    ec, dcfg = load_config("eval"), load_config("detect")
    obstacles = load_config("classes")["obstacles"]
    data_yaml = Path(data_yaml)
    data = yaml.safe_load(data_yaml.read_text())
    if "test" not in data:
        raise ValueError(f"{data_yaml}: test split 없음 — 자체 데이터로 merge했는지 확인")
    root = Path(data.get("path") or data_yaml.parent)
    names = list(data["names"].values()) if isinstance(data["names"], dict) else list(data["names"])
    images = list_images(root / data["test"])
    run_dir = Path(out) / time.strftime("%Y%m%d_%H%M")
    run_dir.mkdir(parents=True, exist_ok=True)

    results, per_image, vis = {}, {}, {}
    for wpath in weights:
        name = Path(wpath).stem
        seg = WindowSegmenter(wpath, dcfg, device)
        model_names = list(seg.model.names.values())
        sub = subset_dataset(data, root, model_names, run_dir / "tmp" / name)
        classes = class_metrics(seg.model, sub, dcfg["imgsz"], seg.device)
        fp_raw, fp_post, rows = dict.fromkeys(obstacles, 0), dict.fromkeys(obstacles, 0), []
        for img_path in images:
            img = cv2.imread(str(img_path))
            gt = read_gt(root / "labels" / "test" / f"{img_path.stem}.txt", names, img.shape[:2])
            dets = seg.predict(img, conf=dcfg["conf"] * dcfg["fill_conf_ratio"])
            raw = [d.mask for d in dets if d.cls == "window" and d.conf >= ec["conf"]]
            post = _quad_masks(postprocess(img, dets, dcfg).windows, img.shape[:2])
            a = obstacle_false_positives(raw, gt, obstacles, ec["iou_thr"], ec["overlap_thr"])
            b = obstacle_false_positives(post, gt, obstacles, ec["iou_thr"], ec["overlap_thr"])
            for c in obstacles:
                fp_raw[c] += a[c]
                fp_post[c] += b[c]
            rows.append({"image": img_path.name, "fp_raw": sum(a.values()), "fp_post": sum(b.values())})
            vis.setdefault(img_path.name, {})[name] = draw_detections(img, [d for d in dets if d.conf >= ec["conf"]])
            put_label(vis[img_path.name][name], f"{name}  fp {sum(a.values())}", (10, 40), 1.2)
        results[name] = {"classes": classes, "fp_raw": fp_raw, "fp_post": fp_post}
        per_image[name] = rows
    shutil.rmtree(run_dir / "tmp")

    # 그리드: 모델 간 오검출 차이가 큰 이미지부터, 한 행 = 한 이미지의 모델별 예측
    diff = {r["image"]: [] for r in next(iter(per_image.values()))}
    for rows in per_image.values():
        for r in rows:
            diff[r["image"]].append(r["fp_raw"])
    order = sorted(diff, key=lambda k: -(max(diff[k]) - min(diff[k])))[:grid_n]
    if order:
        tw = 640
        tiles = [np.hstack([cv2.resize(v, (tw, int(v.shape[0] * tw / v.shape[1]))) for v in vis[k].values()]) for k in order]
        width = max(t.shape[1] for t in tiles)
        cv2.imwrite(str(run_dir / "grid.jpg"), np.vstack([cv2.copyMakeBorder(t, 0, 4, 0, width - t.shape[1], cv2.BORDER_CONSTANT)
                                                          for t in tiles]))

    md = table_md(results, obstacles, len(images))
    (run_dir / "metrics.json").write_text(json.dumps({"data": str(data_yaml), "images": len(images), "eval": ec,
                                                      "results": results, "per_image": per_image},
                                                     ensure_ascii=False, indent=1), encoding="utf-8")
    (run_dir / "table.md").write_text(md, encoding="utf-8")
    MODEL_CARD.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_CARD.open("a", encoding="utf-8") as f:
        f.write(f"\n## 비교 평가 — {time.strftime('%Y-%m-%d %H:%M')} ({', '.join(results)})\n\n{md}"
                f"\n결과: `{run_dir}` (eval.compare, config/eval.yaml)\n")
    print(md)
    return results


def main(
    data: Path = typer.Option(..., exists=True, dir_okay=False),
    weights: list[Path] = typer.Option(..., exists=True, dir_okay=False),
    out: Path = typer.Option(Path("runs/eval")),
    grid_n: int = 16,
    device: str = typer.Option(None),
) -> None:
    compare(data, weights, out, grid_n, device)


if __name__ == "__main__":
    typer.run(main)
