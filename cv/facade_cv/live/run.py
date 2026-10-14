"""실시간 창문 검출: 소스 → WindowSegmenter(imgsz는 config/live.yaml, 비행 후 분석보다 작게) + postprocess (프레임 단위, T5 추적 전) → 화면 오버레이 + JSON 라인.

python -m facade_cv.live.run --source webcam:0 --debug
python -m facade_cv.live.run --source file:data/raw/videos/b04_shcool/b04_ground_01.MOV
python -m facade_cv.live.run --source rtsp://<파이IP>:8554/cam --record

출력 runs/<id>/live/
  detections.jsonl   처리한 프레임마다 한 줄 (LiveFrame). stdout에도 같은 줄, 사람용 로그는 stderr
  summary.json       처리 fps, 소스 통계
  raw.mp4            --record: 수신한 프레임 (원본 해상도)
  overlay.mp4        --record: 오버레이. 둘 다 record_fps 고정, timestamp에 맞춰 반복/건너뛰어 raw와 시간이 같다
  snap_<n>.jpg       화면에서 s 키
runs/<id>/debug/live/<n>_{overlay,stages}.jpg   --debug, debug_every_sec 간격
처리가 소스를 못 따라가면 최신 프레임만 처리하고 사이 프레임은 건너뛴다 (source.dropped).
창문 id는 프레임 안 임시 번호 (추적 전이라 프레임마다 바뀐다). 화면: q/Esc 종료, s 캡처
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TextIO

import cv2
import numpy as np
import typer
from pydantic import BaseModel, ConfigDict

from facade_cv.config import CONFIG_DIR, load_config
from facade_cv.detect.postprocess import PostResult, postprocess
from facade_cv.detect.run import draw_legend, draw_stages, window_style
from facade_cv.detect.segmenter import WindowSegmenter, put_label
from facade_cv.io.source import FrameSource, RateMeter, open_source

RUNS = CONFIG_DIR.parent / "runs"
Point = tuple[float, float]


class LiveWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: int  # 프레임 안 임시 번호
    corners: tuple[Point, Point, Point, Point]  # TL TR BR BL, 처리 해상도 픽셀
    conf: float
    row: int | None
    col: int | None
    flags: list[str]


class LiveFrame(BaseModel):
    """detections.jsonl 한 줄."""

    model_config = ConfigDict(extra="forbid")
    frame: int  # 처리 순번
    t: float  # 소스 timestamp (file: 영상 시각, 그 외: 수신 시각)
    size: tuple[int, int]  # 처리 해상도 (w, h)
    proc_ms: float
    proc_fps: float
    source: dict[str, float]  # SourceStats
    counts: dict[str, int]
    rectified: bool
    warnings: list[str]
    windows: list[LiveWindow]


def to_live_frame(n: int, t: float, img: np.ndarray, res: PostResult, proc_ms: float, proc_fps: float,
                  src: FrameSource) -> LiveFrame:
    d = res.to_dict()
    return LiveFrame(
        frame=n, t=round(t, 3), size=(img.shape[1], img.shape[0]), proc_ms=round(proc_ms, 1),
        proc_fps=round(proc_fps, 1), source=src.stats.to_dict(), counts=d["counts"], rectified=d["rectified"],
        warnings=d["warnings"], windows=[{"id": i, **w} for i, w in enumerate(d["windows"])],
    )


def draw_overlay(img: np.ndarray, res: PostResult, lf: LiveFrame) -> np.ndarray:
    """창문 사각형 + '#임시ID rRcC' + 상단 상태 (처리/수신 fps, 지연, 건너뜀, 끊김)."""
    out = img.copy()
    scale = max(img.shape[:2]) / 1080
    for w in res.windows:
        color, label = window_style(w)
        cv2.polylines(out, [np.round(w.corners).astype(np.int32)], True, color, max(2, round(2 * scale)), cv2.LINE_AA)
    for lw, w in zip(lf.windows, res.windows):
        put_label(out, f"#{lw.id} {window_style(w)[1]}", w.corners[0] + [3, 16 * scale], 0.5 * scale,
                  window_style(w)[0])
    s = lf.source
    draw_legend(out, [
        f"proc {lf.proc_fps:.1f} fps ({lf.proc_ms:.0f} ms) | source {s['recv_fps']:.1f} fps | lag {s['latency_ms']:.0f} ms",
        f"windows {lf.counts['final']} | skipped {s['dropped']:.0f} | disconnects {s['disconnects']:.0f}"
        f" reconnects {s['reconnects']:.0f}",
    ], scale)
    return out


class Recorder:
    """raw.mp4(수신한 모든 프레임) + overlay.mp4. 둘 다 고정 fps로, 첫 raw 프레임 시각 기준 timestamp에 맞춰
    늦게 온 프레임은 직전 프레임을 반복하고 빨리 온 프레임은 건너뛴다 (웹캠 공칭 fps가 실제와 달라도 길이 유지)."""

    def __init__(self, live_dir: Path, fps: float):
        self.live_dir, self.fps = live_dir, fps
        self.writers: dict[str, cv2.VideoWriter] = {}
        self.written: dict[str, int] = {}
        self.t0: float | None = None

    def write(self, name: str, t: float, frame: np.ndarray) -> None:
        if name not in self.writers:
            h, w = frame.shape[:2]
            self.writers[name] = cv2.VideoWriter(str(self.live_dir / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 self.fps, (w, h))
            self.written[name] = 0
        if self.t0 is None:
            self.t0 = t
        target = int((t - self.t0) * self.fps) + 1
        for _ in range(target - self.written[name]):
            self.writers[name].write(frame)
        self.written[name] = max(self.written[name], target)

    def close(self) -> None:
        for w in self.writers.values():
            w.release()


def run_live(
    source: str, run_dir: str | Path, segmenter: WindowSegmenter | None = None, realtime: bool = True,
    record: bool = False, debug: bool = False, show: bool = True, max_frames: int | None = None,
    max_seconds: float | None = None, out: TextIO | None = sys.stdout, cfg: dict | None = None,
    detect_cfg: dict | None = None, weights: str | Path | None = None, device: str | None = None,
    source_kw: dict | None = None,
) -> dict:
    """소스가 끝나거나 max_frames/max_seconds, q/Esc, Ctrl-C까지 실행하고 summary를 반환.
    segmenter: predict(frame, conf) → list[Detection] 와 weights 속성을 가진 객체 (테스트용 주입)."""
    cfg = cfg or load_config("live")
    dcfg = {**(detect_cfg or load_config("detect")), "imgsz": cfg["live"]["imgsz"]}
    lcfg = cfg["live"]
    run_dir = Path(run_dir)
    live_dir, debug_dir = run_dir / "live", run_dir / "debug" / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    if debug:
        debug_dir.mkdir(parents=True, exist_ok=True)
    seg = segmenter or WindowSegmenter(weights, dcfg, device)
    print(f"[live] device={getattr(seg, 'device', '?')} imgsz={dcfg['imgsz']} max_side={lcfg['max_side']} "
          f"weights={seg.weights}", file=sys.stderr, flush=True)

    rec = Recorder(live_dir, lcfg["record_fps"]) if record else None
    src = open_source(source, realtime=realtime, cfg=cfg["source"], **(source_kw or {}))
    if rec:
        src.on_receive = lambda t, f: rec.write("raw", t, f)
    rate = RateMeter(cfg["source"]["stats_window_sec"])
    n, proc_total, last_debug, start = 0, 0.0, -np.inf, time.monotonic()
    device_used = None
    snaps: list[str] = []
    try:
        with src, open(live_dir / "detections.jsonl", "w", encoding="utf-8") as jsonl:
            for t, frame in src:
                t1 = time.monotonic()
                scale = lcfg["max_side"] / max(frame.shape[:2])
                img = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
                dets = seg.predict(img, conf=dcfg["conf"] * dcfg["fill_conf_ratio"])
                res = postprocess(img, dets, dcfg)
                if n == 0:  # 요청한 device가 아니라 Ultralytics predictor가 실제로 올린 device
                    predictor = getattr(getattr(seg, "model", None), "predictor", None)
                    device_used = str(getattr(predictor, "device", getattr(seg, "device", "?")))
                    print(f"[live] inference device={device_used}", file=sys.stderr, flush=True)
                proc_ms = (time.monotonic() - t1) * 1000
                proc_total += proc_ms
                lf = to_live_frame(n, t, img, res, proc_ms, rate.tick(), src)
                line = lf.model_dump_json()
                jsonl.write(line + "\n")
                if out is not None:
                    print(line, file=out, flush=True)
                overlay = draw_overlay(img, res, lf)
                if rec:
                    rec.write("overlay", t, overlay)
                if debug and time.monotonic() - last_debug >= lcfg["debug_every_sec"]:
                    last_debug = time.monotonic()
                    cv2.imwrite(str(debug_dir / f"{n:06d}_overlay.jpg"), overlay)
                    cv2.imwrite(str(debug_dir / f"{n:06d}_stages.jpg"), draw_stages(img, dets, res, dcfg))
                n += 1
                if show:
                    cv2.imshow(lcfg["window_name"], overlay)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("s"):
                        snaps.append(str(live_dir / f"snap_{n:06d}.jpg"))
                        cv2.imwrite(snaps[-1], overlay)
                    if key in (ord("q"), 27):
                        break
                if (max_frames and n >= max_frames) or (max_seconds and time.monotonic() - start >= max_seconds):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if rec:
            rec.close()
        if show:
            cv2.destroyAllWindows()
    elapsed = time.monotonic() - start
    summary = {
        "source": source, "frames": n, "elapsed_sec": round(elapsed, 1),
        "proc_fps_avg": round(n / elapsed, 2) if elapsed else 0.0,
        "proc_ms_avg": round(proc_total / n, 1) if n else 0.0,
        "source_fps": src.fps, "source_stats": src.stats.to_dict(), "weights": str(seg.weights),
        "device": device_used, "imgsz": dcfg["imgsz"], "max_side": lcfg["max_side"],
        "snapshots": snaps,
    }
    (live_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(
    source: str = typer.Option(..., help="webcam:0 | file:<path> | rtsp://<파이IP>:8554/cam | tcp://... | udp://..."),
    id: str = typer.Option(None, help="runs/<id> (기본 live_<날짜_시각>)"),
    record: bool = typer.Option(False, help="runs/<id>/live/raw.mp4, overlay.mp4 저장"),
    debug: bool = typer.Option(False, help="runs/<id>/debug/live/에 overlay·stages 주기 저장"),
    show: bool = typer.Option(True, help="오버레이 창 (q/Esc 종료, s 캡처)"),
    realtime: bool = typer.Option(True, help="file: 실시간 속도 재생 (끄면 모든 프레임 처리)"),
    max_seconds: float = typer.Option(None),
    max_frames: int = typer.Option(None),
    weights: Path = typer.Option(None, exists=True, dir_okay=False, help="기본: config/detect.yaml weights"),
    device: str = typer.Option(None, help="cpu / mps / 0 (기본: config/detect.yaml)"),
    quiet: bool = typer.Option(False, help="stdout JSON 라인 끄기 (파일에는 기록)"),
) -> None:
    run_dir = RUNS / (id or f"live_{datetime.now():%Y%m%d_%H%M%S}")
    s = run_live(source, run_dir, realtime=realtime, record=record, debug=debug, show=show,
                 max_frames=max_frames, max_seconds=max_seconds, out=None if quiet else sys.stdout,
                 weights=weights, device=device)
    print(f"{s['frames']} frames, proc {s['proc_fps_avg']} fps ({s['proc_ms_avg']} ms/frame), "
          f"source {s['source_stats']} → {run_dir / 'live'}", file=sys.stderr)


if __name__ == "__main__":
    typer.run(main)
