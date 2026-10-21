"""창문 검출 + 기하 후처리 (segmenter → postprocess). 결과는 <out>/detections.json.

python -m facade_cv.detect.run --frames runs/<id>/frames --debug   (또는 --image X.jpg)
--out 기본: <입력 폴더>/../detect, --debug 이미지: <out>/../debug/detect/<stem>_{stages,rectified,lines}.jpg
--weights 생략 시 config/detect.yaml weights
"""

import json
from pathlib import Path

import cv2
import numpy as np
import typer

from facade_cv.config import load_config
from facade_cv.detect.datasets.yolo import list_images
from facade_cv.detect.postprocess import PostResult, Window, postprocess
from facade_cv.detect.segmenter import Detection, WindowSegmenter, draw_detections, put_label
from facade_cv.rectify.homography import draw_lines

COLORS = {"ok": (0, 200, 0), "flagged": (0, 140, 255), "fill": (255, 255, 0), "removed": (0, 0, 255),
          "outside": (255, 0, 255), "empty": (0, 0, 255), "facade": (0, 230, 230), "obstacle": (0, 0, 0),
          "occlusion": (255, 255, 255)}  # BGR
SHORT = {"off_grid": "off", "odd_width": "W?", "odd_height": "H?", "fill_candidate": "fill"}


def window_style(w: Window) -> tuple[tuple[int, int, int], str]:
    if w.reason:
        key = "outside" if w.reason == "outside_facade" else "obstacle" if w.reason.startswith("on_") else "removed"
        return COLORS[key], w.reason
    cell = "-" if w.row is None else f"r{w.row}c{w.col}"
    flags = " ".join(SHORT.get(f, f) for f in w.flags)
    key = "fill" if "fill_candidate" in w.flags else "flagged" if w.flags else "ok"
    return COLORS[key], f"{cell} {flags}".strip()


def _draw_quads(img: np.ndarray, items: list[tuple[np.ndarray, Window]], scale: float) -> None:
    th = max(2, round(2 * scale))
    for corners, w in items:
        color, label = window_style(w)
        cv2.polylines(img, [np.round(corners).astype(np.int32)], True, color, th, cv2.LINE_AA)
        put_label(img, label, corners[0] + [3, 16 * scale], 0.5 * scale, color)


def draw_legend(img: np.ndarray, lines: list[str], scale: float) -> None:
    lh = int(24 * scale)
    w = int(max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, 1)[0][0] for t in lines)) + 20
    cv2.rectangle(img, (0, 0), (w, lh * len(lines) + 10), (0, 0, 0), -1)
    for i, t in enumerate(lines):
        put_label(img, t, (10, lh * (i + 1)), 0.55 * scale)


def draw_stages(img: np.ndarray, dets: list[Detection], res: PostResult, cfg: dict) -> np.ndarray:
    """원본 위: 마스크(옅게) + 외벽 필터 영역(노랑) + 단계별 색 사각형 + row/col 라벨 + 범례."""
    scale = max(img.shape[:2]) / 1080
    out = draw_detections(img, [d for d in dets if d.conf >= cfg["conf"]], alpha=0.25)
    if res.facade_mask is not None:
        cnts, _ = cv2.findContours(res.facade_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, COLORS["facade"], max(2, round(2 * scale)))
    _draw_quads(out, [(w.corners, w) for w in res.removed + res.windows + res.fill_candidates], scale)
    for w in res.windows:  # 가림 영역 (흰 윤곽)
        cv2.polylines(out, [np.round(p).astype(np.int32) for p in w.occlusion], True, COLORS["occlusion"], 1, cv2.LINE_AA)
    c = res.to_dict()["counts"]
    reasons = {r: sum(w.reason == r for w in res.removed) for r in ("small", "not_rect", "outside_facade")}
    on_obstacle = sum(bool(w.reason and w.reason.startswith("on_")) for w in res.removed)
    draw_legend(out, [
        f"raw {c['raw']} -> final {c['final']} (+fill {c['fill_candidates']}) | removed: quad {reasons['small'] + reasons['not_rect']}"
        f" (small {reasons['small']}, not_rect {reasons['not_rect']}), on_obstacle {on_obstacle},"
        f" outside_facade {reasons['outside_facade']}",
        f"flagged {c['flagged']}  fill_candidate {c['fill_candidates']}  empty_cells {c['empty_cells']}"
        f"  rectified {'yes' if res.rect else 'NO'}",
        "green ok | orange flagged (off=off_grid W?/H?=odd size) | cyan fill | red removed(quad) | magenta outside"
        " | black on_obstacle | white occlusion | yellow facade",
        *res.warnings,
    ], scale)
    return out


def draw_rectified(img: np.ndarray, res: PostResult) -> np.ndarray | None:
    """정면화 이미지 + 창문 사각형/라벨 + 빈칸(빨간 X)."""
    if res.rect is None:
        return None
    out = res.rect.warp(img)
    scale = max(out.shape[:2]) / 1080
    items = [(res.rect.apply(w.corners), w) for w in res.windows + res.fill_candidates]
    _draw_quads(out, items, scale)
    grid = [(q.mean(0), w) for q, w in items if w.row is not None]
    for r, k in res.empty_cells:
        x = np.median([p[0] for p, w in grid if w.col == k])
        y = np.median([p[1] for p, w in grid if w.row == r])
        s = int(20 * scale)
        cv2.line(out, (int(x - s), int(y - s)), (int(x + s), int(y + s)), COLORS["empty"], 3)
        cv2.line(out, (int(x - s), int(y + s)), (int(x + s), int(y - s)), COLORS["empty"], 3)
        put_label(out, f"r{r}c{k} empty", (x + s, y), 0.5 * scale, COLORS["empty"])
    return out


def run_detect(
    images: list[Path], out_dir: str | Path, weights: str | Path | None = None, debug: bool = False,
    cfg: dict | None = None, device: str | None = None, segmenter: WindowSegmenter | None = None,
) -> dict:
    """images마다 검출 + 후처리 → out_dir/detections.json 내용을 반환."""
    cfg = cfg or load_config("detect")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir.parent / "debug" / "detect"
    seg = segmenter or WindowSegmenter(weights, cfg, device)
    results = []
    for path in images:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"cannot read image: {path}")
        dets = seg.predict(img, conf=cfg["conf"] * cfg["fill_conf_ratio"])
        res = postprocess(img, dets, cfg)
        results.append({"image": path.name, **res.to_dict()})
        if debug:
            debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_dir / f"{path.stem}_stages.jpg"), draw_stages(img, dets, res, cfg))
            cv2.imwrite(str(debug_dir / f"{path.stem}_lines.jpg"), draw_lines(img, res.rect, region=res.line_region))
            rectified = draw_rectified(img, res)
            if rectified is not None:
                cv2.imwrite(str(debug_dir / f"{path.stem}_rectified.jpg"), rectified)
    manifest = {"weights": str(seg.weights), "images": results}
    (out_dir / "detections.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main(
    image: Path = typer.Option(None, exists=True, dir_okay=False),
    frames: Path = typer.Option(None, exists=True, file_okay=False, help="이미지 폴더 (io.extract 출력 그대로)"),
    weights: Path = typer.Option(None, exists=True, dir_okay=False, help="기본: config/detect.yaml weights"),
    out: Path = typer.Option(None, help="기본: <입력 폴더>/../detect"),
    debug: bool = False,
    device: str = typer.Option(None, help="cpu / mps / 0 (기본: config/detect.yaml)"),
) -> None:
    if (image is None) == (frames is None):
        raise typer.BadParameter("--image 또는 --frames 중 하나만")
    images = [image] if image else list_images(frames)
    if not images:
        raise typer.BadParameter(f"no images in {frames}")
    out = out or (image.parent if image else frames).parent / "detect"
    m = run_detect(images, out, weights, debug, device=device)
    for r in m["images"]:
        c = r["counts"]
        print(f"{r['image']}: raw {c['raw']} → final {c['final']} + fill {c['fill_candidates']} (removed {c['removed']}, flagged {c['flagged']}, "
              f"empty {c['empty_cells']}) rectified={r['rectified']}"
              + "".join(f"\n  ! {w}" for w in r["warnings"]))
    print(f"→ {out / 'detections.json'}")


if __name__ == "__main__":
    typer.run(main)
