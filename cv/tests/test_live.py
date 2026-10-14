import io
from pathlib import Path

import cv2
import numpy as np
import pytest

from facade_cv.config import load_config
from facade_cv.detect.segmenter import Detection
from facade_cv.live.run import LiveFrame, Recorder, run_live

W, H, N = 640, 480, 20
# 2행 × 3열 창문 (x0, y0, x1, y1)
RECTS = [(100 + c * 160, 80 + r * 180, 200 + c * 160, 200 + r * 180) for r in range(2) for c in range(3)]


def _video(path: Path) -> Path:
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    for i in range(N):
        f = np.full((H, W, 3), 170, np.uint8)
        for x0, y0, x1, y1 in RECTS:
            cv2.rectangle(f, (x0, y0), (x1, y1), (90, 70, 60), -1)
        cv2.putText(f, str(i), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        vw.write(f)
    vw.release()
    return path


class FakeSegmenter:
    """정답 사각형을 그대로 검출로 돌려주는 모델 대역 (가중치 없이 live 파이프라인만 검사)."""

    weights = Path("fake.pt")

    def predict(self, frame, conf=None):
        h, w = frame.shape[:2]
        s = w / W
        dets = []
        for x0, y0, x1, y1 in RECTS:
            m = np.zeros((h, w), bool)
            m[int(y0 * s):int(y1 * s), int(x0 * s):int(x1 * s)] = True
            poly = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) * s
            dets.append(Detection("window", m, poly, (x0 * s, y0 * s, x1 * s, y1 * s), 0.9))
        return dets


def test_live_file_jsonl_and_record(tmp_path):
    video = _video(tmp_path / "in.mp4")
    out = io.StringIO()
    s = run_live(f"file:{video}", tmp_path / "run", FakeSegmenter(), realtime=False, record=True, debug=True,
                 show=False, out=out)
    lines = out.getvalue().splitlines()
    assert s["frames"] == len(lines) == N  # realtime=False → 건너뛰지 않고 끝까지
    frames = [LiveFrame.model_validate_json(line) for line in lines]
    assert [f.frame for f in frames] == list(range(N))
    assert all(f.counts["final"] == 6 and len(f.windows) == 6 for f in frames)
    assert {(w.row, w.col) for w in frames[0].windows} == {(r, c) for r in (0, 1) for c in (0, 1, 2)}  # postprocess row/col은 0부터
    assert frames[0].size == (W, H) and frames[-1].source["delivered"] == N

    live = tmp_path / "run" / "live"
    assert (live / "detections.jsonl").read_text().splitlines() == lines
    assert (live / "summary.json").is_file()
    for name in ("raw.mp4", "overlay.mp4"):
        cap = cv2.VideoCapture(str(live / name))
        assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == N, name
    assert list((tmp_path / "run" / "debug" / "live").glob("*_overlay.jpg"))


def test_live_resizes_to_max_side(tmp_path):
    cfg = load_config("live")
    cfg["live"]["max_side"] = 320
    out = io.StringIO()
    run_live(f"file:{_video(tmp_path / 'in.mp4')}", tmp_path / "run", FakeSegmenter(), realtime=False, show=False,
             out=out, cfg=cfg, max_frames=3)
    frames = [LiveFrame.model_validate_json(x) for x in out.getvalue().splitlines()]
    assert len(frames) == 3 and frames[0].size == (320, 240)
    assert max(c for w in frames[0].windows for p in w.corners for c in p) <= 320


@pytest.mark.parametrize("src_fps", [15, 30, 60])
def test_recorder_length_follows_timestamps(tmp_path, src_fps):
    """웹캠 공칭 fps가 실제와 달라도 녹화 길이 = 실제 경과 시간 (2초 → 30fps × 2 = 60프레임)."""
    rec = Recorder(tmp_path, fps=30)
    for i in range(src_fps * 2):
        rec.write("raw", 5.0 + i / src_fps, np.zeros((48, 64, 3), np.uint8))
    rec.close()
    n = cv2.VideoCapture(str(tmp_path / "raw.mp4")).get(cv2.CAP_PROP_FRAME_COUNT)
    assert abs(n - 60) <= 1


def test_live_imgsz_overrides_detect_only_for_live(tmp_path, monkeypatch):
    seen = {}

    class Spy(FakeSegmenter):
        def __init__(self, cfg):
            seen["imgsz"] = cfg["imgsz"]

    monkeypatch.setattr("facade_cv.live.run.WindowSegmenter", lambda weights, cfg, device: Spy(cfg))
    s = run_live(f"file:{_video(tmp_path / 'in.mp4')}", tmp_path / "run", realtime=False, show=False, out=None,
                 max_frames=1)
    assert seen["imgsz"] == s["imgsz"] == load_config("live")["live"]["imgsz"] == 640
    assert load_config("detect")["imgsz"] == 1024
