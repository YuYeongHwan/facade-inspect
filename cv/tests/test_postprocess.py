import cv2
import numpy as np
import pytest
from facade_scene import COLS, GAP_X, GAP_Y, ROWS, WIN_H, WIN_W, make_scene, window_det

from facade_cv.config import load_config
from facade_cv.detect.postprocess import mask_to_quad, order_corners, postprocess
from facade_cv.detect.segmenter import Detection

MISSING = (1, 2)


@pytest.fixture(scope="module")
def scene():
    return make_scene()


def _rect(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


def _cell(r, k):
    """격자 칸 (r, k) 창문의 정면 좌표 사각형."""
    x0, y0 = GAP_X + k * (WIN_W + GAP_X), GAP_Y + r * (WIN_H + GAP_Y)
    return _rect(x0, y0, x0 + WIN_W, y0 + WIN_H)


def _by_cell(res):
    return {(w.row, w.col): w for w in res.windows}


def test_order_corners():
    pts = np.array([[10, 90], [90, 85], [12, 5], [95, 10]])  # 뒤섞인 기울어진 사각형
    assert order_corners(pts).tolist() == [[12, 5], [95, 10], [90, 85], [10, 90]]


def test_mask_to_quad():
    cfg = load_config("detect")
    m = np.zeros((200, 200), np.uint8)
    cv2.fillConvexPoly(m, np.array([[40, 30], [160, 40], [150, 170], [30, 160]]), 1)
    quad, area, rect = mask_to_quad(m, cfg)
    assert np.abs(quad - [[40, 30], [160, 40], [150, 170], [30, 160]]).max() <= 2  # approxPolyDP 4점
    assert rect > 0.95
    tri = np.zeros((200, 200), np.uint8)
    cv2.fillConvexPoly(tri, np.array([[20, 180], [180, 180], [100, 20]]), 1)
    quad, _, rect = mask_to_quad(tri, cfg)  # 4점이 안 나옴 → minAreaRect
    assert quad.shape == (4, 2) and rect < cfg["min_rectangularity"]


def test_grid_rows_cols_and_missing(scene):
    """원근 변환된 3x4 격자, (1,2) 검출 누락 → 나머지 11개의 row/col이 정답, 빈칸 (1,2)."""
    img, wins, facade, _ = scene
    dets = [d for rc, d in wins.items() if rc != MISSING] + [facade]
    res = postprocess(img, dets)
    assert res.rect is not None and not res.warnings and not res.removed
    assert {rc: w.flags for rc, w in _by_cell(res).items()} == {rc: [] for rc in wins if rc != MISSING}
    for (r, k), w in _by_cell(res).items():  # 정답 칸의 창문인지 (모서리 일치)
        assert np.abs(w.corners - wins[r, k].polygon).max() < 3
    assert res.empty_cells == [MISSING]


def test_fill_candidate(scene):
    """누락 칸에 저신뢰(conf ≥ conf × fill_conf_ratio) 검출이 있으면 fill_candidate. 격자 밖 저신뢰 검출은 버린다."""
    img, wins, facade, H_true = scene
    cfg = load_config("detect")
    weak = Detection(**{**wins[MISSING].__dict__, "conf": cfg["conf"] * 0.6})
    stray = window_det(_cell(0, 0) + [WIN_W + GAP_X / 2 - 20, 0], H_true, conf=cfg["conf"] * 0.6)  # 열 사이
    dets = [d for rc, d in wins.items() if rc != MISSING] + [weak, stray, facade]
    res = postprocess(img, dets, cfg)
    assert MISSING not in _by_cell(res) and len(res.windows) == ROWS * COLS - 1  # 창문 수에 넣지 않는다
    [fill] = res.fill_candidates
    assert (fill.row, fill.col) == MISSING and fill.conf == weak.conf and res.empty_cells == []
    d = res.to_dict()
    assert d["counts"] == {"raw": 11, "final": 11, "removed": 0, "flagged": 0, "fill_candidates": 1, "empty_cells": 0}
    assert len(d["windows"]) == 11 and [(f["row"], f["col"]) for f in d["fill_candidates"]] == [MISSING]


def test_flags_not_removed(scene):
    """격자 밖 작은 검출(간판)은 off_grid, 같은 열보다 폭이 큰 검출은 odd_width — 둘 다 제거하지 않는다."""
    img, wins, facade, H_true = scene
    sign = window_det(_rect(GAP_X + 2.5 * (WIN_W + GAP_X), 15, GAP_X + 2.5 * (WIN_W + GAP_X) + 60, 55), H_true)
    wide = window_det(_cell(2, 1) + [[-30, 0], [30, 0], [30, 0], [-30, 0]], H_true)
    dets = [d for rc, d in wins.items() if rc != (2, 1)] + [wide, sign, facade]
    res = postprocess(img, dets)
    assert not res.removed and len(res.windows) == ROWS * COLS + 1
    flagged = {tuple(w.flags): w for w in res.windows if w.flags}
    assert set(flagged) == {("off_grid",), ("odd_width",)}
    assert flagged["off_grid",].row is None
    assert (flagged["odd_width",].row, flagged["odd_width",].col) == (2, 1)


def test_facade_filter_removes_outside(scene):
    """실제 모델처럼 외벽 마스크에 창문 자리가 뚫려 있어도 격자 창문은 안, 벽 밖 창문만 제거."""
    img, wins, facade, H_true = scene
    holed = facade.mask.copy()
    for d in wins.values():
        holed &= ~d.mask
    facade = Detection("facade", holed, facade.polygon, facade.bbox, facade.conf)
    outside = window_det(_rect(-300, 200, -200, 340), H_true)  # 벽 왼쪽 밖
    small = window_det(_rect(300, 10, 310, 20), H_true)  # min_area_px 미만
    res = postprocess(img, list(wins.values()) + [outside, small, facade])
    assert sorted(w.reason for w in res.removed) == ["outside_facade", "small"]
    assert len(res.windows) == ROWS * COLS and not res.warnings


@pytest.mark.parametrize("bad", ["none", "tiny", "elsewhere"])
def test_facade_filter_skipped_when_facade_fails(scene, bad):
    """외벽 검출 실패(없음 / 창문 bbox 합집합보다 작음 / 창문 대부분이 밖) → 필터를 건너뛰고 경고."""
    img, wins, facade, H_true = scene
    outside = window_det(_rect(-300, 200, -200, 340), H_true)
    fac = {
        "none": [],
        "tiny": [Detection("facade", np.pad(np.ones((50, 50), bool), ((0, img.shape[0] - 50), (0, img.shape[1] - 50))),
                           np.zeros((0, 2), np.float32), (0, 0, 50, 50), 0.9)],
        "elsewhere": [Detection("facade", np.pad(np.ones((img.shape[0], 300), bool), ((0, 0), (img.shape[1] - 300, 0))),
                                np.zeros((0, 2), np.float32), (img.shape[1] - 300, 0, img.shape[1], img.shape[0]),
                                0.9)],  # 옆 건물만 잡힌 경우 (b02)
    }[bad]
    weak_facade = Detection(**{**facade.__dict__, "conf": 0.15})  # 저신뢰 외벽은 필터에 쓰지 않는다
    res = postprocess(img, list(wins.values()) + [outside, weak_facade] + fac)
    assert not res.removed and len(res.windows) == ROWS * COLS + 1
    assert any(w.startswith("facade_filter_skipped") for w in res.warnings)
    assert res.facade_mask is None


def test_grid_fallback_without_rectification():
    """정면화 실패 시 원본 좌표로 클러스터링 (정면 이미지면 그대로 맞아야 한다)."""
    img, wins, facade, _ = make_scene(0, 0)
    cfg = load_config("rectify")
    cfg["min_lines"] = 10_000  # 강제 실패
    res = postprocess(img, list(wins.values()) + [facade], rect_cfg=cfg)
    assert res.rect is None and any(w.startswith("rectification_failed") for w in res.warnings)
    assert {(w.row, w.col) for w in res.windows} == set(wins) and res.empty_cells == []


def test_sparse_grid_skips_empty_cells(scene):
    """계단식 배치(점유율 6/12)에 min_occupancy를 그보다 높이면 빈칸·보완 후보 없이 경고만."""
    img, wins, facade, _ = scene
    stairs = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)]
    weak = Detection(**{**wins[0, 3].__dict__, "conf": 0.15})
    dets = [wins[rc] for rc in stairs] + [weak, facade]
    assert postprocess(img, dets).empty_cells != []  # 기본 0.4면 격자로 본다
    cfg = load_config("detect")
    cfg["grid"]["min_occupancy"] = 0.6
    res = postprocess(img, dets, cfg)
    assert res.empty_cells == [] and res.fill_candidates == [] and len(res.windows) == len(stairs)
    assert any(w.startswith("grid_unreliable") for w in res.warnings)
    assert {(w.row, w.col) for w in res.windows} == set(stairs)
