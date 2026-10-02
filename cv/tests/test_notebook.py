"""notebooks/train_seg.ipynb 정적 검증 + Colab 의존 없는 셀(parameters, helpers)을 로컬에서 실행해 검증."""

import ast
import re
from pathlib import Path

import cv2
import nbformat
import numpy as np
import pytest
import yaml

from facade_cv.config import CONFIG_DIR, load_config
from facade_cv.detect.datasets.merge_datasets import merge

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "train_seg.ipynb"


@pytest.fixture(scope="module")
def nb():
    return nbformat.read(NOTEBOOK, as_version=4)


def _tagged(nb, tag):
    (cell,) = [c for c in nb.cells if tag in c.metadata.get("tags", [])]
    return cell.source


@pytest.fixture(scope="module")
def ns(nb):
    """parameters + helpers 셀을 실행한 네임스페이스."""
    env: dict = {}
    exec(_tagged(nb, "parameters"), env)
    exec(_tagged(nb, "helpers"), env)
    return env


def test_notebook_valid(nb):
    nbformat.validate(nb)
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":  # IPython 셸 명령(!, %)은 제외하고 파이썬 문법 검사
            src = "\n".join(line for line in cell.source.splitlines() if not line.lstrip().startswith(("!", "%")))
            ast.parse(src, filename=f"cell {i}")


def test_params_match_repo_config(ns):
    assert ns["AUGMENT"] == yaml.safe_load((CONFIG_DIR / "augment.yaml").read_text())
    assert ns["CLASSES"] == load_config("dataset")["names"]
    assert ns["CLASSES"][ns["WINDOW_ID"]] == "window"
    assert ns["STAGE1"]["model"] == "yolo11s-seg.pt" and ns["STAGE1"]["imgsz"] == 1024
    assert ns["STAGE1"]["epochs"] == 100 and ns["STAGE1"]["patience"] > 0
    assert ns["OWN_OVERSAMPLE"] == 3
    assert ns["STAGE1"]["batch"] == ns["STAGE2"]["batch"] == 0.7  # AutoBatch (GPU 메모리 70%)
    assert isinstance(ns["EVAL_BATCH"], int)  # val은 소수 batch 미지원
    assert ns["RESUME"] is False and ns["RUN_NAME"] is None
    assert ns["RUNS_DIR"].startswith(ns["DRIVE_ROOT"])  # 학습 결과는 드라이브에


def _yolo_dir(root, stems):
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    for s in stems:
        cv2.imwrite(str(root / "images" / f"{s}.jpg"), np.zeros((32, 32, 3), np.uint8))
        (root / "labels" / f"{s}.txt").write_text("0 0.1 0.1 0.4 0.1 0.4 0.4\n1 0 0 1 0 1 1\n")
    return root


@pytest.fixture
def merged(tmp_path):
    cmp = _yolo_dir(tmp_path / "cmp", [f"cmp_b{i:04d}" for i in range(20)])
    own = _yolo_dir(tmp_path / "own", [f"flight{g}__frame_{i:03d}" for g in "ABCDE" for i in range(2)])
    merge(cmp, [own], tmp_path / "merged")
    return tmp_path / "merged"


def test_make_stage_yamls(ns, merged):
    y = ns["make_stage_yamls"](merged, 3, ns["CLASSES"])
    c = y["counts"]
    lines = {k: (merged / "lists" / f"{k}.txt").read_text().split() for k in c}

    assert all(ns["is_cmp"](p) for k in ("stage1_train", "stage1_val") for p in lines[k])
    own_train = [p for p in lines["stage2_train"] if not ns["is_cmp"](p)]
    assert len(own_train) == 3 * len(set(own_train)) > 0  # ×3 오버샘플링
    assert c["stage2_train"] == c["stage1_train"] + len(own_train)
    assert lines["own_test"] and not any(ns["is_cmp"](p) for p in lines["own_test"])
    assert any(not ns["is_cmp"](p) for p in lines["stage2_val"])
    for p in (p for v in lines.values() for p in v):
        assert ns["label_of"](p).is_file()

    data = yaml.safe_load(y["stage2"].read_text())
    assert data["names"] == {0: "window", 1: "facade"} and data["test"] == "lists/own_test.txt"


def test_ultralytics_keeps_oversampled_duplicates(ns, merged):
    """Ultralytics가 목록의 중복 경로를 그대로 샘플링하는지 (중복 제거하면 오버샘플링이 무효)."""
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset

    y = ns["make_stage_yamls"](merged, 3, ns["CLASSES"])
    ds = YOLODataset(img_path=str(merged / "lists" / "stage2_train.txt"), imgsz=64, augment=False,
                     hyp=get_cfg(), data={"names": dict(enumerate(ns["CLASSES"])), "channels": 3}, task="segment")
    assert len(ds) == y["counts"]["stage2_train"]
    assert len(ds.labels) == len(ds.im_files) and all(len(lb["segments"]) == 2 for lb in ds.labels)


def test_make_stage_yamls_requires_own_test(ns, tmp_path):
    cmp = _yolo_dir(tmp_path / "cmp", [f"cmp_b{i:04d}" for i in range(5)])
    merge(cmp, [], tmp_path / "merged")
    with pytest.raises(ValueError, match="test"):
        ns["make_stage_yamls"](tmp_path / "merged", 3, ns["CLASSES"])


def test_build_augment_is_valid_ultralytics_cfg(ns):
    import albumentations as A
    from ultralytics.cfg import get_cfg

    args = ns["build_augment"](ns["AUGMENT"])
    assert {type(t) for t in args["augmentations"]} == {A.RandomBrightnessContrast, A.MotionBlur}
    get_cfg(overrides={**ns["STAGE2"], **args})


def test_count_metrics(ns):
    m = ns["count_metrics"]([10, 20, 0, 5], [10, 18, 1, 8], 0.1)
    assert m["MAE"] == pytest.approx((0 + 2 + 1 + 3) / 4)
    assert m["bias"] == pytest.approx((0 - 2 + 1 + 3) / 4)
    assert m["exact"] == 0.25
    assert m["within_10%"] == 0.5  # 10/10, 18/20 (10%) 통과, 0→1 (100%), 5→8 (60%) 실패
    assert (m["gt_total"], m["pred_total"]) == (35, 37)


def test_gt_count_and_label_of(ns, tmp_path):
    lab = tmp_path / "ds" / "labels" / "test" / "a.txt"
    lab.parent.mkdir(parents=True)
    lab.write_text("0 0 0 1 0 1 1\n1 0 0 1 0 1 1\n0 0 0 1 0 1 1\n")
    assert ns["label_of"](tmp_path / "ds" / "images" / "test" / "a.jpg") == lab
    assert ns["gt_count"](lab, 0) == 2 and ns["gt_count"](tmp_path / "missing.txt", 0) == 0


def test_next_version(ns, tmp_path):
    assert ns["next_version"](tmp_path) == 1
    for n in ("seg_window_facade_v1.pt", "seg_window_facade_v3.pt", "seg_window_facade_v2_old.pt", "other_v9.pt"):
        (tmp_path / n).touch()
    assert ns["next_version"](tmp_path) == 4


def test_model_card_entry(ns):
    ev = {"mask_P": 0.9, "mask_R": 0.8, "mask_mAP50": 0.75, "mask_mAP50-95": 0.5}
    cnt = ns["count_metrics"]([10, 20], [10, 18], 0.1)
    entry = ns["model_card_entry"]({
        "name": "seg_window_facade_v2", "date": "2026-10-02", "file": "seg_window_facade_v2.pt", "dataset": "merged.zip",
        "counts": {"own_test": 7}, "stage1": ns["STAGE1"], "stage2": ns["STAGE2"], "oversample": 3,
        "augment": ns["AUGMENT"], "ultralytics": "8.4.171", "gpu": "T4", "seed": 0, "run": "r",
        "eval": {"stage1": ev, "stage2": ev}, "count": {"stage1": cnt, "stage2": cnt}, "count_conf": 0.4, "count_tol": 0.1,
    })
    assert entry.startswith("\n## seg_window_facade_v2 — 2026-10-02")
    assert "| stage2 | 0.7500 | 0.5000 | 1.00 | 5.0% | 100.0% |" in entry
    assert "자체 test셋 7장" in entry


def _ckpt(run_dir, epoch):
    import torch

    (run_dir / "weights").mkdir(parents=True)
    torch.save({"epoch": epoch}, run_dir / "weights" / "last.pt")


def test_stage_state(ns, tmp_path):
    assert ns["stage_state"](tmp_path / "r_stage1") == "new"
    _ckpt(tmp_path / "r_stage1", 7)
    assert ns["stage_state"](tmp_path / "r_stage1") == "partial"
    _ckpt(tmp_path / "r_stage2", -1)  # Ultralytics strip_optimizer가 학습 종료 시 epoch=-1로 저장
    assert ns["stage_state"](tmp_path / "r_stage2") == "done"


def test_train_stage_refuses_to_clobber_without_resume(ns, tmp_path):
    _ckpt(tmp_path / "r_stage1", 3)
    with pytest.raises(FileExistsError, match="RESUME"):
        ns["train_stage"]("stage1", "x.pt", tmp_path / "d.yaml", {}, str(tmp_path), "r", False)


def test_train_stage_skips_done_stage(ns, tmp_path):
    _ckpt(tmp_path / "r_stage1", -1)
    (tmp_path / "r_stage1" / "weights" / "best.pt").touch()
    best = ns["train_stage"]("stage1", "x.pt", tmp_path / "d.yaml", {}, str(tmp_path), "r", True)
    assert best == tmp_path / "r_stage1" / "weights" / "best.pt"


def test_resolve_run_name(ns, tmp_path):
    import os

    assert ns["resolve_run_name"](tmp_path, "mine", True) == "mine"
    assert re.fullmatch(r"\d{8}_\d{4}", ns["resolve_run_name"](tmp_path, None, False))
    with pytest.raises(FileNotFoundError):
        ns["resolve_run_name"](tmp_path, None, True)
    _ckpt(tmp_path / "older_run_stage2", 3)
    _ckpt(tmp_path / "newer_run_stage1", 3)
    (tmp_path / "newer_run_eval" / "stage1").mkdir(parents=True)  # 평가 디렉토리는 후보 아님
    os.utime(tmp_path / "older_run_stage2" / "weights" / "last.pt", (1, 1))
    assert ns["resolve_run_name"](tmp_path, None, True) == "newer_run"
