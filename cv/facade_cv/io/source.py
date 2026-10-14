"""영상 소스 → (timestamp, BGR frame) 순회. 소스 URI만 바꾸면 같은 코드가 돈다 (config/live.yaml source).

  webcam:0                OpenCV 장치 번호. 맥에서는 아이폰 연속성 카메라도 장치 하나로 잡힌다 (--probe로 번호 확인)
  file:<경로>             녹화 영상. realtime=True면 벽시계 속도로 재생하고 늦은 프레임은 디코딩 없이 버린다
  rtsp:// tcp:// udp://   라즈베리파이 스트림 (OpenCV FFMPEG 백엔드, edge/raspberrypi/stream.sh)

python -m facade_cv.io.source webcam:0 --seconds 10     # 수신 통계
python -m facade_cv.io.source --probe                  # 웹캠 장치 번호 목록

- timestamp: file은 영상 시각, 그 외는 수신 시각 (소스를 연 뒤 초)
- latest=True: 수신 스레드가 계속 읽고 소비자는 항상 최신 프레임만 받는다 (밀린 프레임은 dropped).
  기본값은 실시간 소스(webcam·스트림·realtime file)면 True
- 끊기면 (read 실패·타임아웃) 자동 재연결. file은 끝나면 종료
- 통계 (stats): 수신 fps, 지연 = 수신(read 반환) → 소비자 전달 (file realtime은 영상 시각 대비 늦은 정도).
  네트워크·인코더 지연은 재지 못한다
"""

import os
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import typer

from facade_cv.config import load_config

STREAM_SCHEMES = ("rtsp://", "tcp://", "udp://")


def parse_uri(uri: str) -> tuple[str, int | str]:
    """→ (kind, target). kind: webcam | file | stream."""
    if uri.startswith("webcam:"):
        return "webcam", int(uri.removeprefix("webcam:"))
    if uri.startswith("file:"):
        return "file", uri.removeprefix("file:")
    if uri.startswith(STREAM_SCHEMES):
        return "stream", uri
    raise ValueError(f"unknown source {uri!r}: webcam:N | file:<path> | rtsp:// | tcp:// | udp://")


class RateMeter:
    """최근 window초 동안의 이벤트 빈도 (fps)."""

    def __init__(self, window: float):
        self.window, self.times = window, deque()

    def tick(self, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        self.times.append(now)
        while now - self.times[0] > self.window:
            self.times.popleft()
        return self.rate

    @property
    def rate(self) -> float:
        span = self.times[-1] - self.times[0] if len(self.times) > 1 else 0
        return (len(self.times) - 1) / span if span > 0 else 0.0


@dataclass
class SourceStats:
    received: int = 0  # 디코딩한 프레임
    delivered: int = 0  # 소비자에게 넘긴 프레임
    dropped: int = 0  # latest 모드에서 덮어쓴 것 + file realtime에서 늦어 버린 것
    disconnects: int = 0
    reconnects: int = 0
    recv_fps: float = 0.0
    latency_ms: float = 0.0  # 마지막 전달 프레임

    def to_dict(self) -> dict:
        return {k: round(v, 1) if isinstance(v, float) else v for k, v in asdict(self).items()}


def _open_capture(kind: str, target: int | str, cfg: dict) -> cv2.VideoCapture:
    if kind != "stream":
        return cv2.VideoCapture(target)
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", cfg["ffmpeg_options"])
    ms = int(cfg["timeout_sec"] * 1000)
    return cv2.VideoCapture(target, cv2.CAP_FFMPEG,
                            [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, ms, cv2.CAP_PROP_READ_TIMEOUT_MSEC, ms])


class FrameSource:
    """for t, frame in FrameSource(...). with 문으로 쓰면 중간에 멈춰도 수신 스레드·장치를 정리한다.

    opener: () → VideoCapture 호환 객체 (테스트용 주입). on_receive(t, frame): 디코딩한 모든 프레임마다
    수신 스레드에서 호출 (원본 녹화용 — latest 모드에서 건너뛴 프레임도 포함).
    """

    def __init__(self, uri: str, realtime: bool = False, latest: bool | None = None, cfg: dict | None = None,
                 opener: Callable[[], cv2.VideoCapture] | None = None,
                 on_receive: Callable[[float, np.ndarray], None] | None = None):
        self.uri = uri
        self.kind, self.target = parse_uri(uri)
        self.cfg = cfg or load_config("live")["source"]
        self.realtime = realtime and self.kind == "file"
        self.latest = (self.kind != "file" or self.realtime) if latest is None else latest
        self.opener = opener or (lambda: _open_capture(self.kind, self.target, self.cfg))
        self.on_receive = on_receive
        self.fps: float | None = None  # 소스 공칭 fps (연결 후)
        self.stats = SourceStats()
        self._stop = threading.Event()
        self._fails = 0  # 프레임 없이 연속 실패한 횟수
        self._ever_connected = False
        self._recv = RateMeter(self.cfg["stats_window_sec"])

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._stop.set()

    def _fail(self) -> None:
        self._fails += 1
        if self.cfg["max_retries"] is not None and self._fails > self.cfg["max_retries"]:
            raise ConnectionError(f"{self.uri}: no frames after {self._fails} attempts")
        print(f"[source] {self.uri}: no frames ({self._fails}x), reconnecting", file=sys.stderr, flush=True)

    def _connect(self) -> cv2.VideoCapture | None:
        while not self._stop.is_set():
            if self._fails:
                self._stop.wait(min(self.cfg["reconnect_delay"] * 2 ** (self._fails - 1),
                                    self.cfg["reconnect_max_delay"]))
            cap = self.opener()
            if cap.isOpened():
                self.fps = cap.get(cv2.CAP_PROP_FPS) or self.cfg["default_fps"]
                self.stats.reconnects += self._ever_connected
                self._ever_connected = True
                return cap
            cap.release()
            if self.kind == "file":
                raise FileNotFoundError(f"cannot open video: {self.target}")
            self._fail()
        return None

    def _read(self) -> Iterator[tuple[float, np.ndarray, float]]:
        """(timestamp, frame, 수신 시각 monotonic). 끊기면 재연결."""
        cap = self._connect()
        t0 = time.monotonic()
        idx = 0
        try:
            while cap is not None and not self._stop.is_set():
                ok = cap.grab()
                now = time.monotonic()
                t = now - t0
                if ok and self.kind == "file":
                    t, idx = idx / self.fps, idx + 1
                    if self.realtime:
                        due = t0 + t  # 이 프레임이 화면에 나와야 할 시각 → 지연 기준
                        if now - due > 1 / self.fps:
                            self.stats.dropped += 1
                            continue
                        self._stop.wait(max(due - now, 0))
                        now = due
                frame = cap.retrieve()[1] if ok else None
                if frame is None:
                    cap.release()
                    if self.kind == "file":
                        return
                    self.stats.disconnects += 1
                    self._fail()
                    cap = self._connect()
                    continue
                self._fails = 0
                self.stats.received += 1
                self.stats.recv_fps = self._recv.tick(now)
                if self.on_receive:
                    self.on_receive(t, frame)
                yield t, frame, now
        finally:
            if cap is not None:
                cap.release()

    def _deliver(self, t: float, frame: np.ndarray, recv: float) -> tuple[float, np.ndarray]:
        self.stats.delivered += 1
        self.stats.latency_ms = (time.monotonic() - recv) * 1000
        return t, frame

    def __iter__(self) -> Iterator[tuple[float, np.ndarray]]:
        if not self.latest:
            try:
                for item in self._read():
                    yield self._deliver(*item)
            finally:
                self.close()
            return

        slot: list[tuple[float, np.ndarray, float]] = []
        cond = threading.Condition()
        state: dict = {"done": False, "error": None}

        def reader() -> None:
            try:
                for item in self._read():
                    with cond:
                        self.stats.dropped += bool(slot)
                        slot[:] = [item]
                        cond.notify()
            except BaseException as e:  # noqa: BLE001 — 소비자 스레드에서 다시 던진다
                state["error"] = e
            finally:
                with cond:
                    state["done"] = True
                    cond.notify()

        th = threading.Thread(target=reader, name=f"source:{self.uri}", daemon=True)
        th.start()
        try:
            while True:
                with cond:
                    cond.wait_for(lambda: slot or state["done"])
                    if not slot:
                        break
                    item = slot.pop()
                yield self._deliver(*item)
            if state["error"]:
                raise state["error"]
        finally:
            self.close()
            th.join(self.cfg["timeout_sec"] + 1)


def open_source(uri: str, realtime: bool = False, latest: bool | None = None, cfg: dict | None = None,
                **kw) -> FrameSource:
    """uri → (timestamp, frame) 순회 객체. 통계는 .stats"""
    return FrameSource(uri, realtime, latest, cfg, **kw)


def probe_webcams(max_index: int) -> list[tuple[int, int, int]]:
    """열리는 웹캠 (index, width, height)."""
    found = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        if ok:
            found.append((i, frame.shape[1], frame.shape[0]))
    return found


def main(
    uri: str = typer.Argument(None, help="webcam:0 | file:<path> | rtsp://... | tcp://... | udp://..."),
    seconds: float = typer.Option(10, help="수신 시간"),
    realtime: bool = typer.Option(True, help="file: 벽시계 속도로 재생"),
    probe: bool = typer.Option(False, help="웹캠 장치 번호 목록"),
) -> None:
    cfg = load_config("live")["source"]
    if probe:
        for i, w, h in probe_webcams(cfg["probe_max_index"]):
            print(f"webcam:{i}  {w}x{h}")
        return
    if uri is None:
        raise typer.BadParameter("uri 또는 --probe")
    start = last = time.monotonic()
    with open_source(uri, realtime=realtime, cfg=cfg) as src:
        for _ in src:
            now = time.monotonic()
            if now - last >= 1:
                print(src.stats.to_dict(), flush=True)
                last = now
            if now - start >= seconds:
                break
    print("final", src.stats.to_dict())


if __name__ == "__main__":
    typer.run(main)
