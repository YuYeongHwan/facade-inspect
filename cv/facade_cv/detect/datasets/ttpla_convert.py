"""TTPLA [R20] (Apache-2.0, 항공 송전탑·전선) → YOLO-seg 타일. cable → wire, tower_wooden → pole, 철탑·void 버림 (근거: data/DATASETS.md)

python -m facade_cv.detect.datasets.ttpla_convert --src data/public/ttpla --out data/yolo/ttpla [--debug]
--src 아래를 재귀 탐색해 labelme JSON(shapes[].label/points, imageWidth/Height)과 같은 이름의 이미지를 짝짓는다.
- 라벨 이름은 대소문자 무시 (Cable → wire). void polygon은 이미지에서 config void_fill 색으로 칠한다
- 축소하지 않고 원본에서 tile×tile 타일로 자른다 (전선 굵기 유지). polygon은 타일 경계에서 잘라 남기고, 잘린 조각이
  min_area_ratio(타일 면적 대비)보다 작으면 버린다. wire·pole이 하나도 없는 타일은 버린다
- 선택: 최대 max_tiles장. pole 있는 타일 우선, wire만 있는 타일 wire_only_ratio. 영상(`<영상>_<프레임>`)마다 최대 per_video장,
  서로 다른 프레임 중 이미 고른 프레임과 가장 먼 것부터 (연속 프레임 회피). 영상 순서는 round-robin
출력: <out>/images/<stem>_t<k>.jpg, labels/<stem>_t<k>.txt. --debug면 샘플 8장에 polygon을 그려 runs/convert_check/ttpla_*.jpg
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import typer

from facade_cv.config import class_names, load_config
from facade_cv.detect.datasets.yolo import IMAGE_EXTS, source_missing, write_label

GUIDE = """TTPLA (가입 불필요, Apache-2.0 [R20]): https://github.com/r3ab/ttpla_dataset README의 Google Drive 링크에서
  이미지+labelme JSON zip(data_original_size_v1, 4.2GB)을 받아 data/public/ttpla/ 에 압축 해제 (1,242장, 3840×2160)"""

VOID = "void"


def map_shapes(doc: dict, cfg: dict, names: list[str]) -> tuple[list[tuple[int, np.ndarray]], list[np.ndarray]]:
    """labelme JSON → ([(class id, 픽셀 polygon)], [void 픽셀 polygon]). 라벨 대소문자 무시, 점 3개 미만·활성 셋 밖은 버림."""
    mapping = {k.lower(): v for k, v in cfg["ttpla"]["map"].items()}
    inst, void = [], []
    for shape in doc["shapes"]:
        label = shape["label"].strip().lower()
        pts = np.asarray(shape["points"], np.float64)
        if len(pts) < 3:
            continue
        if label == VOID:
            void.append(pts)
        elif mapping.get(label) in names:
            inst.append((names.index(mapping[label]), pts))
    return inst, void


def fill_void(img: np.ndarray, void: list[np.ndarray], color: list[int]) -> np.ndarray:
    if void:
        cv2.fillPoly(img, [np.round(p).astype(np.int32) for p in void], tuple(int(c) for c in color))
    return img


def tile_origins(w: int, h: int, t: int) -> list[tuple[int, int, int, int]]:
    """(x0, y0, tw, th) 타일 — 각 축 ceil(L/t)칸을 같은 간격으로 (마지막 칸이 끝에 맞게 겹침). 3840×2160, 1280 → 3×2."""
    def starts(length: int) -> list[int]:
        size = min(t, length)
        n = -(-length // size)
        return [0] if n == 1 else [round(i * (length - size) / (n - 1)) for i in range(n)]
    return [(x, y, min(t, w), min(t, h)) for y in starts(h) for x in starts(w)]


def clip_to_tile(pts: np.ndarray, tile: tuple[int, int, int, int], min_area: float) -> list[np.ndarray]:
    """픽셀 polygon → 타일 안에 잘린 polygon들 (타일 좌표, 픽셀). 래스터화 → 외곽선 → 1px approxPolyDP.
    ponytail: 래스터 왕복이라 꼭짓점이 1px 단위로 흔들린다 — 정확한 기하 클리핑이 필요하면 shapely로."""
    x0, y0, tw, th = tile
    local = np.round(pts - [x0, y0]).astype(np.int32)
    if local[:, 0].max() < 0 or local[:, 1].max() < 0 or local[:, 0].min() >= tw or local[:, 1].min() >= th:
        return []
    m = np.zeros((th, tw), np.uint8)
    cv2.fillPoly(m, [local], 1)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        p = cv2.approxPolyDP(c, 1.0, True)[:, 0].astype(float)
        if len(p) >= 3:
            out.append(p)
    return out


@dataclass
class Tile:
    stem: str
    video: str
    frame: int
    k: int
    box: tuple[int, int, int, int]
    instances: list[tuple[int, np.ndarray]] = field(default_factory=list)  # (class id, 타일 픽셀 polygon)


def video_frame(stem: str) -> tuple[str, int]:
    """`34_00221` → ("34", 221). 형식이 다르면 (stem, 0)."""
    video, _, frame = stem.partition("_")
    return (video, int(frame)) if frame.isdigit() else (stem, 0)


def make_tiles(stem: str, doc: dict, cfg: dict, names: list[str]) -> list[Tile]:
    """wire·pole이 있는 타일만 (라벨만으로 계산, 이미지는 읽지 않음)."""
    inst, _ = map_shapes(doc, cfg, names)
    video, frame = video_frame(stem)
    tiles = []
    for k, box in enumerate(tile_origins(doc["imageWidth"], doc["imageHeight"], cfg["ttpla"]["tile"])):
        tile = Tile(stem, video, frame, k, box)
        area = box[2] * box[3]
        for cls, pts in inst:
            for p in clip_to_tile(pts, box, cfg["min_area_ratio"][names[cls]] * area):
                tile.instances.append((cls, p))
        if tile.instances:
            tiles.append(tile)
    return tiles


def select_tiles(tiles: list[Tile], cfg: dict, pole_id: int | None, seed: int = 0) -> list[Tile]:
    """pole 있는 타일 (1 − wire_only_ratio)·max_tiles장, wire만 있는 타일 나머지. 한쪽이 모자라면 다른 쪽으로 채움.
    영상마다 per_video장, 프레임은 서로 다르고 이미 고른 프레임과의 최소 간격이 가장 큰 것부터. 영상은 round-robin."""
    tc = cfg["ttpla"]
    total, per_video = tc["max_tiles"], tc["per_video"]
    has_pole = [t for t in tiles if any(c == pole_id for c, _ in t.instances)]
    wire_only = [t for t in tiles if not any(c == pole_id for c, _ in t.instances)]
    n_wire = round(total * tc["wire_only_ratio"])
    rng = random.Random(seed)
    chosen: list[Tile] = []
    frames: dict[str, list[int]] = {}

    def fill(pool: list[Tile], quota: int) -> None:
        by_video: dict[str, list[Tile]] = {}
        for t in pool:
            by_video.setdefault(t.video, []).append(t)
        order = sorted(by_video)
        rng.shuffle(order)
        got, progress = 0, True
        while got < quota and progress:
            progress = False
            for v in order:
                used = frames.setdefault(v, [])
                if got >= quota or len(used) >= per_video:
                    continue
                cand = [t for t in by_video[v] if t.frame not in used]
                if not cand:
                    continue
                best = max(cand, key=lambda t: (min((abs(t.frame - f) for f in used), default=0),
                                                len(t.instances), -t.frame, -t.k))
                chosen.append(best)
                used.append(best.frame)
                got += 1
                progress = True

    fill(has_pole, total - n_wire)
    fill(wire_only, total - len(chosen))
    fill(has_pole, total - len(chosen))  # wire만 있는 타일이 모자랐으면 pole로 채움
    return chosen


def wire_width(poly: np.ndarray) -> float:
    """가는 띠 polygon의 평균 굵기 ≈ 면적 / (둘레/2)."""
    c = poly.astype(np.float32)
    return 2 * cv2.contourArea(c) / max(cv2.arcLength(c, True), 1e-6)


def tile_stats(chosen: list[Tile], names: list[str]) -> dict:
    counts = {n: sum(c == i for t in chosen for c, _ in t.instances) for i, n in enumerate(names)}
    wire = names.index("wire") if "wire" in names else None
    widths = [wire_width(p) for t in chosen for c, p in t.instances if c == wire]
    return {"tiles": len(chosen), "instances": {n: v for n, v in counts.items() if v},
            "videos": len({t.video for t in chosen}),
            "wire_width_px": round(float(np.mean(widths)), 2) if widths else None}


def convert_ttpla(src: str | Path, out: str | Path, cfg: dict | None = None, names: list[str] | None = None,
                  debug_dir: Path | None = None) -> dict:
    """tile_stats 반환 (tiles = 저장한 타일 수). 데이터가 없으면 안내만 출력하고 tiles 0."""
    cfg = cfg or load_config("dataset")
    names = names or class_names()
    src, out = Path(src), Path(out)
    if source_missing(src, GUIDE):
        return tile_stats([], names)
    docs, tiles = {}, []
    for js in sorted(src.rglob("*.json")):
        doc = json.loads(js.read_text(encoding="utf-8"))
        if "shapes" not in doc:
            continue  # COCO 등 다른 JSON
        img_path = next((p for p in js.parent.glob(f"{js.stem}.*") if p.suffix.lower() in IMAGE_EXTS), None)
        if img_path is None:
            continue
        docs[js.stem] = (doc, img_path)
        tiles += make_tiles(js.stem, doc, cfg, names)
    chosen = select_tiles(tiles, cfg, names.index("pole") if "pole" in names else None)
    by_stem: dict[str, list[Tile]] = {}
    for t in chosen:
        by_stem.setdefault(t.stem, []).append(t)
    if chosen:
        (out / "images").mkdir(parents=True, exist_ok=True)
    for stem, ts in by_stem.items():
        doc, img_path = docs[stem]
        img = fill_void(cv2.imread(str(img_path)), map_shapes(doc, cfg, names)[1], cfg["ttpla"]["void_fill"])
        for t in ts:
            x0, y0, tw, th = t.box
            name = f"{stem}_t{t.k}"
            cv2.imwrite(str(out / "images" / f"{name}.jpg"), img[y0:y0 + th, x0:x0 + tw], [cv2.IMWRITE_JPEG_QUALITY, 95])
            write_label(out / "labels" / f"{name}.txt", [(c, p / [tw, th]) for c, p in t.instances])
    if debug_dir is not None and chosen:
        save_debug(chosen, out, names, debug_dir)
    return {**tile_stats(chosen, names), "candidates": len(tiles)}


def save_debug(chosen: list[Tile], out: Path, names: list[str], debug_dir: Path, n: int = 8) -> None:
    """pole 있는 타일과 wire만 있는 타일을 반씩 (seed 0). wire 노랑, pole 자홍."""
    colors = {"wire": (0, 255, 255), "pole": (255, 0, 255)}
    pole = names.index("pole") if "pole" in names else None
    rng = random.Random(0)
    with_pole = [t for t in chosen if any(c == pole for c, _ in t.instances)]
    rest = [t for t in chosen if t not in with_pole]
    pick = rng.sample(with_pole, min(n // 2, len(with_pole)))
    pick += rng.sample(rest, min(n - len(pick), len(rest)))
    debug_dir.mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(pick, 1):
        im = cv2.imread(str(out / "images" / f"{t.stem}_t{t.k}.jpg"))
        for c, p in t.instances:
            cv2.polylines(im, [p.astype(np.int32)], True, colors.get(names[c], (0, 255, 0)), 2)
        cv2.imwrite(str(debug_dir / f"ttpla_{i:02d}.jpg"), im)


def main(
    src: Path = typer.Option(Path("data/public/ttpla")),
    out: Path = typer.Option(Path("data/yolo/ttpla")),
    classes: str = typer.Option(None, help="v1 | v2 (config/classes.yaml, 기본 active)"),
    debug: bool = typer.Option(False, help="샘플 8장에 polygon을 그려 --debug-dir에 저장"),
    debug_dir: Path = typer.Option(Path("runs/convert_check")),
) -> None:
    st = convert_ttpla(src, out, names=class_names(classes), debug_dir=debug_dir if debug else None)
    print(f"{st['tiles']} tiles (후보 {st.get('candidates', 0)}, 영상 {st['videos']}개) → {out}")
    print(f"  인스턴스 {st['instances']}, 전선 평균 굵기 {st['wire_width_px']} px")


if __name__ == "__main__":
    typer.run(main)
