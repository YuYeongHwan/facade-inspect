"""영상 프레임 추출 (Post-flight: COLMAP 입력용 키프레임) + Live 순회.

python -m facade_cv.io.extract --video X --out runs/<id>/frames --fps 3 --debug
"""

import json
import time
from bisect import bisect_left, bisect_right
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import typer

from facade_cv.config import load_config

DEBUG_COLORS = {None: (0, 200, 0), "blur": (0, 0, 220), "exposure": (0, 140, 255), "duplicate": (128, 128, 128)}
RESCUED_COLOR = (220, 200, 0)


def iterate_video(source: str | Path | int, realtime: bool = False) -> Iterator[tuple[float, np.ndarray]]:
    """(timestamp 초, BGR frame) 순회. realtime=True면 벽시계에 맞춰 재생하고 늦은 프레임은 디코딩 없이 버린다."""
    cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start = time.monotonic()
    idx = 0
    try:
        while cap.grab():
            t = idx / fps
            idx += 1
            if realtime:
                lag = time.monotonic() - start - t
                if lag > 1 / fps:
                    continue
                if lag < 0:
                    time.sleep(-lag)
            ok, frame = cap.retrieve()
            if not ok:
                break
            yield t, frame
    finally:
        cap.release()


def _samples(video_path: Path, target_fps: float, max_side: int) -> Iterator[tuple[int, float, np.ndarray]]:
    """target_fps 간격으로 (원본 index, timestamp, max_side로 축소된 frame)."""
    next_t = 0.0
    for idx, (t, frame) in enumerate(iterate_video(video_path)):
        if t + 1e-6 < next_t:
            continue
        next_t += 1 / target_fps
        scale = max_side / max(frame.shape[:2])
        if scale < 1:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        yield idx, t, frame


def _resize_side(img: np.ndarray, side: int) -> np.ndarray:
    scale = side / max(img.shape[:2])
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img


def _motion(
    ref: tuple, cur: tuple, shape: tuple[int, int], min_matches: int, matcher: cv2.DescriptorMatcher
) -> float | None:
    """1 - (cur를 호모그래피로 ref에 투영했을 때 겹치는 면적 비율). 매칭/inlier 부족이면 None(불명)."""
    (kp1, d1), (kp2, d2) = ref, cur
    if d1 is None or d2 is None or len(kp1) < min_matches or len(kp2) < min_matches:
        return None
    pairs = matcher.knnMatch(d2, d1, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < min_matches:
        return None
    src = np.float32([kp2[m.queryIdx].pt for m in good])
    dst = np.float32([kp1[m.trainIdx].pt for m in good])
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None or inliers.sum() < min_matches:
        return None
    h, w = shape
    rect = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    warped = cv2.convexHull(cv2.perspectiveTransform(rect[None], H)[0])
    area, _ = cv2.intersectConvexConvex(rect, warped)
    return float(np.clip(1 - area / (w * h), 0, 1))


def _save_debug_grid(path: Path, thumbs: list[np.ndarray], records: list[dict], cfg: dict) -> None:
    cells = []
    for thumb, r in zip(thumbs, records):
        color = RESCUED_COLOR if r["rescued"] else DEBUG_COLORS[r["reason"]]
        cell = cv2.copyMakeBorder(thumb, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=color)
        label = f"{r['timestamp']:.1f}s {'rescued' if r['rescued'] else r['reason'] or 'OK'}"
        cv2.putText(cell, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(cell, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cells.append(cell)
    h, w = max(c.shape[0] for c in cells), max(c.shape[1] for c in cells)
    cells = [cv2.copyMakeBorder(c, 0, h - c.shape[0], 0, w - c.shape[1], cv2.BORDER_CONSTANT) for c in cells]
    cols = cfg["columns"]
    cells += [np.zeros_like(cells[0])] * (-len(cells) % cols)
    rows = [np.hstack(cells[i : i + cols]) for i in range(0, len(cells), cols)]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows))


def extract_frames(
    video_path: str | Path,
    out_dir: str | Path,
    target_fps: float = 3,
    max_side: int = 1920,
    debug: bool = False,
    cfg: dict | None = None,
) -> dict:
    """키프레임을 out_dir/frame_<index>.jpg로 저장하고 out_dir/frames.json 내용을 반환.

    필터 순서: 노출 → 블러 → 중복(ORB 겹침). 겹침이 max_motion을 넘거나 불명이면 사이 프레임
    (중복 후보 → 블러 제거 프레임 중 가장 덜 흐린 것 순)을 끼워 넣는다. 이동량 불명 프레임은 저장한다.
    디버그 그리드는 out_dir/../debug/frames_grid.jpg
    """
    cfg = cfg or load_config("extract")
    video_path, out_dir = Path(video_path), Path(out_dir)
    exp, blur_cfg, dd = cfg["exposure"], cfg["blur"], cfg["dedup"]

    # 1차: 품질 점수. ponytail: 영상을 두 번 디코딩 — 프레임을 메모리에 쌓지 않으려는 선택
    records = []
    for idx, t, frame in _samples(video_path, target_fps, max_side):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        over, under = float((gray >= exp["over_value"]).mean()), float((gray <= exp["under_value"]).mean())
        records.append({
            "index": idx, "timestamp": round(t, 4),
            "blur": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
            "over_ratio": round(over, 4), "under_ratio": round(under, 4),
            "blur_cut": None, "motion": None, "selected": False, "rescued": False,
            "reason": "exposure" if max(over, under) > exp["max_ratio"] else None,
            "file": None,
        })
    if not records:
        raise ValueError(f"no frames decoded: {video_path}")

    # 블러: 절대 임계값 미만 또는 ±window_sec 안의 percentile 미만이면 제거
    ok = [r for r in records if r["reason"] is None]
    ts = [r["timestamp"] for r in ok]
    win = blur_cfg["window_sec"]
    for r in ok:
        lo, hi = bisect_left(ts, r["timestamp"] - win), bisect_right(ts, r["timestamp"] + win)
        local = float(np.percentile([o["blur"] for o in ok[lo:hi]], blur_cfg["percentile"]))
        r["blur_cut"] = round(max(blur_cfg["threshold"], local), 2)
        if r["blur"] < r["blur_cut"]:
            r["reason"] = "blur"

    # 2차: 중복 필터 + 저장
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("frame_*.jpg"):
        old.unlink()
    orb = cv2.ORB_create(dd["orb_features"])
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    thumbs: list[np.ndarray] = []

    def select(r: dict, frame: np.ndarray) -> None:
        r["selected"], r["reason"] = True, None
        r["file"] = f"frame_{r['index']:06d}.jpg"
        cv2.imwrite(str(out_dir / r["file"]), frame, [cv2.IMWRITE_JPEG_QUALITY, cfg["jpeg_quality"]])

    def features(frame: np.ndarray) -> tuple:
        small = _resize_side(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), dd["analysis_side"])
        return orb.detectAndCompute(small, None), small.shape

    def motion_to(feats: tuple, shape: tuple[int, int]) -> float | None:
        return _motion(last, feats, shape, dd["min_matches"], matcher)

    last = None  # 직전 선택 프레임의 ORB 특징
    pending = None  # 직전 중복 후보 (record, frame) — 겹침 부족 시 1순위로 끼워 넣기
    rescue = None  # pending 이후 블러 제거 프레임 중 가장 덜 흐린 것 (record, frame) — 2순위
    for r, (_, _, frame) in zip(records, _samples(video_path, target_fps, max_side)):
        if debug:
            thumbs.append(_resize_side(frame, cfg["debug"]["thumb_width"]))
        if r["reason"] == "blur" and last is not None and (rescue is None or r["blur"] > rescue[0]["blur"]):
            rescue = (r, frame)
        if r["reason"] is not None:
            continue
        feats, shape = features(frame)
        if last is None:
            select(r, frame)
            r["motion"], last = 0.0, feats
            continue
        motion = motion_to(feats, shape)
        if motion is None or motion > dd["max_motion"]:
            # pending은 항상 rescue보다 앞선 시점이므로 이 순서로 넣으면 시간순이 유지된다
            for bridge in (pending, rescue):
                if bridge is None:
                    continue
                br, bframe = bridge
                bfeats, _ = features(bframe)
                if br["reason"] == "blur":
                    m = motion_to(bfeats, shape)
                    br["motion"], br["rescued"] = None if m is None else round(m, 4), True
                select(br, bframe)
                last = bfeats
                motion = motion_to(feats, shape)
                if motion is not None and motion <= dd["max_motion"]:
                    break
            pending = rescue = None
        r["motion"] = None if motion is None else round(motion, 4)
        if motion is None or motion >= dd["min_motion"]:
            select(r, frame)
            last, pending, rescue = feats, None, None
        else:
            r["reason"] = "duplicate"
            pending, rescue = (r, frame), None

    manifest = {
        "video": str(video_path), "target_fps": target_fps, "max_side": max_side,
        "num_selected": sum(r["selected"] for r in records),
        "frames": records,
    }
    (out_dir / "frames.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if debug:
        _save_debug_grid(out_dir.parent / "debug" / "frames_grid.jpg", thumbs, records, cfg["debug"])
    return manifest


def main(
    video: Path = typer.Option(..., exists=True, dir_okay=False),
    out: Path = typer.Option(...),
    fps: float = 3,
    max_side: int = 1920,
    debug: bool = False,
) -> None:
    m = extract_frames(video, out, fps, max_side, debug)
    print(f"{m['num_selected']}/{len(m['frames'])} frames selected → {out}")


if __name__ == "__main__":
    typer.run(main)
