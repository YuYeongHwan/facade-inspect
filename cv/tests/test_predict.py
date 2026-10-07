import subprocess
import sys

import cv2
import numpy as np
import pytest
import yaml

from facade_cv.config import load_config
from facade_cv.detect.predict import predict_folder


def _img(path, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((96, 128, 3), 128, np.uint8)
    for _ in range(6):
        x, y = rng.integers(0, 100), rng.integers(0, 70)
        cv2.rectangle(img, (int(x), int(y)), (int(x) + 20, int(y) + 20), rng.integers(0, 255, 3).tolist(), -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


@pytest.fixture(scope="module")
def tiny_weights(tmp_path_factory):
    """window/facade 2클래스 YOLO-seg를 CPU에서 1에폭 학습한 가중치 (다운로드 없음)."""
    from ultralytics import YOLO

    root = tmp_path_factory.mktemp("tiny")
    for split in ("train", "val"):
        for i in range(2):
            _img(root / "images" / split / f"{i}.jpg", i)
            (root / "labels" / split).mkdir(parents=True, exist_ok=True)
            (root / "labels" / split / f"{i}.txt").write_text("0 0.1 0.1 0.3 0.1 0.3 0.4 0.1 0.4\n1 0 0 1 0 1 1 0 1\n")
    (root / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(root), "train": "images/train", "val": "images/val", "names": {0: "window", 1: "facade"}}))
    model = YOLO("yolo11n-seg.yaml")
    model.train(data=str(root / "data.yaml"), epochs=1, imgsz=64, batch=2, device="cpu", workers=0, plots=False,
                val=False, project=str(root / "runs"), name="t", verbose=False, amp=False)
    return model.trainer.last


@pytest.fixture
def extract_out(tmp_path):
    """io.extract 출력 폴더 모양: <flightID>__frame_*.jpg + frames.json."""
    frames = tmp_path / "run" / "frames"
    for i in range(3):
        _img(frames / f"F1__frame_{i:06d}.jpg", i)
    (frames / "frames.json").write_text("{}")
    return frames


@pytest.mark.slow
def test_predict_extract_folder(tiny_weights, extract_out):
    cfg = load_config("detect")
    cfg["conf"] = 0.0  # 1에폭 모델이라 검출이 거의 없으므로 오버레이 경로를 타도록
    rows = predict_folder(tiny_weights, extract_out, cfg=cfg, device="cpu")
    out = extract_out.parent / "pred"  # 기본 출력: <images>/../pred
    assert sorted(p.name for p in out.iterdir()) == [f"F1__frame_{i:06d}.jpg" for i in range(3)]
    assert [r["image"] for r in rows] == [f"F1__frame_{i:06d}.jpg" for i in range(3)]
    assert all(set(r) == {"image", "window", "facade"} for r in rows)
    assert cv2.imread(str(out / "F1__frame_000000.jpg")).shape == (96, 128, 3)  # 원본 크기 유지


@pytest.mark.slow
def test_predict_cli(tiny_weights, extract_out, tmp_path):
    r = subprocess.run([sys.executable, "-m", "facade_cv.detect.predict", "--weights", str(tiny_weights),
                        "--images", str(extract_out), "--out", str(tmp_path / "o"), "--device", "cpu"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "3 images" in r.stdout and len(list((tmp_path / "o").glob("*.jpg"))) == 3


def test_predict_empty_folder(tmp_path):
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames" / "frames.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="no images"):
        predict_folder(tmp_path / "w.pt", tmp_path / "frames")


def test_config_weights_path():
    from facade_cv.config import CONFIG_DIR

    assert load_config("detect")["weights"] == "cv/weights/seg_window_facade_v1.pt"
    assert (CONFIG_DIR.parent / "cv").is_dir()  # 레포 루트 기준 상대경로
