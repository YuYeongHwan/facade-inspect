"""eval.compare: 장애물 위 오검출 집계, 모델 클래스 맞춤 시험셋, 결과 표 (모델 실행 없음)."""

import numpy as np
import yaml

from facade_cv.eval.compare import obstacle_false_positives, read_gt, subset_dataset, table_md

OBS = ["vegetation", "wire", "pole", "door"]


def _box(x0, y0, x1, y1, shape=(100, 100)):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


def test_obstacle_false_positives():
    gt = {"window": [_box(10, 10, 30, 30)], "door": [_box(50, 50, 70, 90)], "vegetation": [_box(0, 60, 40, 100)]}
    pred = [
        _box(10, 10, 30, 30),   # 정검출
        _box(52, 52, 68, 88),   # 문 위 → door
        _box(5, 65, 35, 95),    # 나무 위 → vegetation
        _box(80, 0, 95, 20),    # 아무것도 아닌 곳 (일반 오검출, 장애물 아님)
        _box(30, 55, 60, 75),   # 문·나무에 조금씩 걸침 (각 < 0.5) → 안 셈
    ]
    assert obstacle_false_positives(pred, gt, OBS, 0.5, 0.5) == {"vegetation": 1, "wire": 0, "pole": 0, "door": 1}


def test_subset_dataset_and_read_gt(tmp_path):
    """v2 6클래스 test 라벨 → v1(2클래스) 모델용: window/facade만 남고 id는 모델 순서."""
    root = tmp_path / "merged"
    (root / "images" / "test").mkdir(parents=True)
    (root / "labels" / "test").mkdir(parents=True)
    import cv2

    cv2.imwrite(str(root / "images" / "test" / "own__v__frame_000001.jpg"), np.zeros((10, 10, 3), np.uint8))
    (root / "labels" / "test" / "own__v__frame_000001.txt").write_text(
        "5 0 0 1 0 1 1\n0 0 0 0.5 0 0.5 0.5\n1 0 0 1 0 1 1\n")
    data = {"path": str(root), "test": "images/test",
            "names": {0: "window", 1: "facade", 2: "vegetation", 3: "wire", 4: "pole", 5: "door"}}
    y = subset_dataset(data, root, ["window", "facade"], tmp_path / "sub")
    lab = (tmp_path / "sub" / "labels" / "test" / "own__v__frame_000001.txt").read_text().split("\n")
    assert [line.split()[0] for line in lab if line] == ["0", "1"]
    assert yaml.safe_load(y.read_text())["names"] == {0: "window", 1: "facade"}
    gt = read_gt(root / "labels" / "test" / "own__v__frame_000001.txt", list(data["names"].values()), (10, 10))
    assert set(gt) == {"window", "facade", "door"} and gt["window"][0].sum() in range(20, 40)


def test_table_md():
    m = {"mask_P": 0.9, "mask_R": 0.8, "mask_mAP50": 0.7, "mask_mAP50-95": 0.5}
    fp = dict.fromkeys(OBS, 0)
    res = {"v1": {"classes": {"window": m}, "fp_raw": fp | {"door": 3}, "fp_post": fp | {"door": 3}},
           "v2": {"classes": {"window": m, "door": m}, "fp_raw": fp | {"door": 1}, "fp_post": fp}}
    md = table_md(res, OBS, 12)
    assert "자체 test 12장" in md and "| v1 | 0.7000 | 0.5000 | 0/0 | 0/0 | 0/0 | 3/3 | 3/3 |" in md
    assert "| v2 | door | 0.900 |" in md
