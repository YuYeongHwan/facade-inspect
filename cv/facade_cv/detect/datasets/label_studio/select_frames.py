"""라벨링할 프레임 선택: 장애물(나무·담쟁이·전선·출입문)이 많을 법한 프레임 위주, 시간·유사도로 분산 → 이미지 + selection.csv.

python -m facade_cv.detect.datasets.label_studio.select_frames --video data/raw/videos/b04_shcool/b04_ground_01.MOV \
    --out data/own/b04_obstacle   (--weights 기본: config/detect.yaml)
출력: <out>/raw/<flightID>__frame_<index>.jpg (io.extract와 같은 이름 → merge가 시간 구간으로 분할), <out>/selection.csv

점수 (config/dataset.yaml select): v1 모델은 장애물 클래스를 모르므로 대리 점수를 쓴다
  score = (1 - window∪facade 피복률) + 후처리에서 제거된 창문 수 × removed_weight
  - 피복률이 낮다 = 화면에 건물이 아닌 것(나무·전선·하늘·땅)이 많다. 제거된 창문 = 장애물 위 오검출 후보
  - window/facade 검출이 하나도 없으면 0 (건물 없는 장면은 라벨링 가치 낮음)
분산: 영상을 bins개 시간 구간으로 나눠 구간마다 count/bins장, 구간 안에서는 점수순으로 고르되 이미 고른 프레임과
dHash 해밍거리가 dhash_min_dist 미만이면 건너뜀. 모자라면 전체에서 점수순으로 채운다.
batch: 시간순 round-robin (라벨러마다 영상 전 구간이 고르게).
"""

import csv
from pathlib import Path

import cv2
import numpy as np
import typer

from facade_cv.config import load_config
from facade_cv.io.extract import GROUP_SEP, _samples, default_flight_id


def dhash(img: np.ndarray) -> int:
    """64bit difference hash (9×8 회색조, 가로 이웃 밝기 비교)."""
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).ravel()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def select(cands: list[dict], sc: dict) -> list[dict]:
    """cands: [{index, t, score, dhash, ...}] → 고른 것 (시간순, bin·batch·reason·dhash_min 추가)."""
    if not cands:
        return []
    t_max = max(c["t"] for c in cands) + 1e-6
    for c in cands:
        c["bin"] = int(c["t"] / t_max * sc["bins"])
    quota = [sc["count"] // sc["bins"] + (b < sc["count"] % sc["bins"]) for b in range(sc["bins"])]
    picked: list[dict] = []

    def try_pick(c: dict, reason: str) -> bool:
        d = min((hamming(c["dhash"], p["dhash"]) for p in picked), default=64)
        if d < sc["dhash_min_dist"]:
            return False
        picked.append(c | {"dhash_min": d, "reason": reason})
        return True

    for b in range(sc["bins"]):
        n = 0
        for c in sorted((c for c in cands if c["bin"] == b), key=lambda c: -c["score"]):
            if n >= quota[b]:
                break
            n += try_pick(c, f"bin {b} top score")
    chosen = {p["index"] for p in picked}
    for c in sorted((c for c in cands if c["index"] not in chosen), key=lambda c: -c["score"]):
        if len(picked) >= sc["count"]:
            break
        try_pick(c, "fill (구간 할당량 부족)")
    picked.sort(key=lambda p: p["t"])
    for i, p in enumerate(picked):
        p["batch"] = i % sc["batches"] + 1
    return picked


def score_frame(frame: np.ndarray, seg, dcfg: dict, sc: dict) -> dict:
    from facade_cv.detect.postprocess import postprocess

    dets = seg.predict(frame)
    strong = [d for d in dets if d.conf >= dcfg["conf"] and d.cls in ("window", "facade")]
    if not strong:
        return {"coverage": 0.0, "removed": 0, "score": 0.0}
    coverage = float(np.logical_or.reduce([d.mask for d in strong]).mean())
    removed = len(postprocess(frame, dets, dcfg).removed)
    # ponytail: 장애물 대리 점수 — v2 모델이 생기면 장애물 마스크 면적으로 바꾼다
    return {"coverage": round(coverage, 4), "removed": removed,
            "score": round((1 - coverage) + removed * sc["removed_weight"], 4)}


def select_frames(video: str | Path, out: str | Path, weights: str | Path | None = None, flight_id: str | None = None,
                  cfg: dict | None = None) -> list[dict]:
    from facade_cv.detect.segmenter import WindowSegmenter

    cfg = cfg or load_config("dataset")
    sc, dcfg = cfg["select"], load_config("detect")
    out, flight_id = Path(out), flight_id or default_flight_id(video)
    seg = WindowSegmenter(weights, dcfg)
    cands = []
    for idx, t, frame in _samples(Path(video), sc["fps"], sc["max_side"]):
        cands.append({"index": idx, "t": round(t, 2), "dhash": dhash(frame), **score_frame(frame, seg, dcfg, sc)})
    picked = select(cands, sc)
    keep = {p["index"]: p for p in picked}
    (out / "raw").mkdir(parents=True, exist_ok=True)
    for idx, _, frame in _samples(Path(video), sc["fps"], sc["max_side"]):  # 다시 읽어 고른 것만 저장 (메모리 절약)
        if idx in keep:
            keep[idx]["image"] = f"{flight_id}{GROUP_SEP}frame_{idx:06d}.jpg"
            cv2.imwrite(str(out / "raw" / keep[idx]["image"]), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cols = ["image", "index", "t", "score", "coverage", "removed", "dhash_min", "bin", "batch", "reason"]
    with open(out / "selection.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(picked)
    return picked


def main(
    video: Path = typer.Option(..., exists=True, dir_okay=False),
    out: Path = typer.Option(...),
    weights: Path = typer.Option(None, exists=True, dir_okay=False),
    flight_id: str = typer.Option(None, help="파일명 접두사 (기본: 영상 파일명)"),
) -> None:
    picked = select_frames(video, out, weights, flight_id)
    batches = {b: sum(p["batch"] == b for p in picked) for b in sorted({p["batch"] for p in picked})}
    print(f"{len(picked)} frames → {out / 'raw'}, batches {batches}, {out / 'selection.csv'}")


if __name__ == "__main__":
    typer.run(main)
