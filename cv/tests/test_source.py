import time

import cv2
import numpy as np
import pytest

from facade_cv.config import load_config
from facade_cv.io.source import RateMeter, open_source, parse_uri


def _cfg(**kw):
    return load_config("live")["source"] | {"reconnect_delay": 0.01, "reconnect_max_delay": 0.02} | kw


class FakeCap:
    """VideoCapture 흉내: frames개를 내보낸 뒤 read 실패 (= 끊김). opened=False면 열기 실패."""

    def __init__(self, frames=0, opened=True, start=0):
        self.queue = [np.full((4, 4, 3), start + i, np.uint8) for i in range(frames)]
        self.opened = opened

    def isOpened(self):
        return self.opened

    def get(self, prop):
        return 30.0

    def grab(self):
        return bool(self.queue)

    def retrieve(self):
        return True, self.queue.pop(0)

    def release(self):
        pass


def _opener(caps):
    it = iter(caps)
    return lambda: next(it, FakeCap(opened=False))


@pytest.mark.parametrize("uri, kind", [("webcam:1", "webcam"), ("file:a.mp4", "file"), ("rtsp://10.0.0.2:8554/cam", "stream"),
                                       ("tcp://10.0.0.2:8888", "stream"), ("udp://@:5000", "stream")])
def test_parse_uri(uri, kind):
    assert parse_uri(uri)[0] == kind


def test_parse_uri_unknown():
    with pytest.raises(ValueError):
        parse_uri("http://x")


def test_reconnect_after_disconnect():
    """5프레임 → 끊김 → 열기 실패 2번 → 다시 5프레임 → 끊김 → max_retries 초과로 종료."""
    caps = [FakeCap(5), FakeCap(opened=False), FakeCap(opened=False), FakeCap(5, start=100)]
    src = open_source("rtsp://fake/cam", latest=False, cfg=_cfg(max_retries=3), opener=_opener(caps))
    got = []
    with pytest.raises(ConnectionError):
        for _, f in src:
            got.append(int(f[0, 0, 0]))
    assert got == [0, 1, 2, 3, 4, 100, 101, 102, 103, 104]
    assert src.stats.disconnects == 2 and src.stats.reconnects == 1
    assert src.stats.received == src.stats.delivered == 10


def test_latest_drops_stale_frames():
    """소비자가 느리면 최신 프레임만 받고 나머지는 dropped. 순서는 유지."""
    src = open_source("rtsp://fake/cam", cfg=_cfg(max_retries=0), opener=_opener([FakeCap(200)]))
    assert src.latest
    got = []
    with pytest.raises(ConnectionError):
        for _, f in src:
            got.append(int(f[0, 0, 0]))
            time.sleep(0.005)
    assert got == sorted(got) and got[-1] == 199
    assert src.stats.received == 200
    assert src.stats.delivered == len(got) < 200
    assert src.stats.dropped == 200 - len(got)


def _video(path, n=30, fps=30):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(n):
        vw.write(np.full((48, 64, 3), i * 8 % 256, np.uint8))
    vw.release()
    return path


def test_file_all_frames(tmp_path):
    src = open_source(f"file:{_video(tmp_path / 'v.mp4')}")
    ts = [t for t, _ in src]
    assert len(ts) == 30 and ts[1] == pytest.approx(1 / 30)
    assert src.stats.dropped == 0 and src.stats.disconnects == 0


def test_file_realtime_paced(tmp_path):
    src = open_source(f"file:{_video(tmp_path / 'v.mp4', n=15)}", realtime=True)
    assert src.latest
    start = time.monotonic()
    n = sum(1 for _ in src)
    assert time.monotonic() - start >= 14 / 30 * 0.9
    assert n + src.stats.dropped == 15


def test_file_missing():
    with pytest.raises(FileNotFoundError):
        list(open_source("file:/nonexistent.mp4"))


def test_rate_meter():
    r = RateMeter(1.0)
    for i in range(11):
        r.tick(i * 0.1)
    assert r.rate == pytest.approx(10)
