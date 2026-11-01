"""CSI 보정 도구: 가중 카파, 평가 저장·이어하기, 합성 데이터에서 알려진 기준값·k_t 복원."""

import json

import numpy as np
import pytest

from facade_cv.config import load_config
from facade_cv.contamination.calibrate import (
    agreement_table,
    append_rating,
    consensus,
    cross_validate,
    csi_values,
    done_images,
    features,
    fit,
    levels_of,
    load_regions,
    qwk,
    rating_order,
    read_ratings,
    write_doc,
)

K0 = np.array([0.5, 0.8, 1.0, 1.5])  # config 초기값 (dust, stain, grime, heavy)
CUTS0 = [0.009, 0.022, 0.044, 0.089]


def test_qwk_known_values():
    assert qwk([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 1.0
    assert qwk([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == pytest.approx(-1.0)
    # 손 계산: O = 대각 2개 + (1,2), E = 행·열 합 곱 / 3 → 1 − Σw·O/Σw·E
    a, b = [1, 2, 2], [1, 2, 3]
    o = np.zeros((5, 5)); o[0, 0] = o[1, 1] = o[1, 2] = 1  # noqa: E702
    e = np.outer(o.sum(1), o.sum(0)) / 3
    w = (np.subtract.outer(np.arange(5), np.arange(5)) ** 2) / 16
    assert qwk(a, b) == pytest.approx(1 - (w * o).sum() / (w * e).sum())
    assert qwk([1, 2, 3], [2, 3, 4]) < qwk([1, 2, 3], [1, 2, 4])  # 차이가 클수록 낮다


def test_rating_store_resume_and_undo(tmp_path):
    p = tmp_path / "kim.csv"
    imgs = [f"w{i}.jpg" for i in range(6)]
    order = rating_order(imgs, "kim", set())
    assert sorted(order) == imgs and order == rating_order(imgs, "kim", set())  # 평가자별 고정
    assert order != rating_order(imgs, "lee", set())
    append_rating(p, order[0], 3)
    append_rating(p, order[1], 5)
    append_rating(p, order[1], None)  # 되돌리기
    assert done_images(p) == {order[0]}
    assert rating_order(imgs, "kim", done_images(p)) == order[1:]  # 이어하기
    append_rating(p, order[1], 4)
    assert read_ratings(tmp_path) == {"kim": {order[0]: 3, order[1]: 4}}
    (tmp_path / "ratings_all.csv").write_text("\ufeffimage,kim,lee\nw0.jpg,1,2\n", encoding="utf-8")  # 합본 형식은 무시
    assert read_ratings(tmp_path) == {"kim": {order[0]: 3, order[1]: 4}}


def test_agreement_and_consensus():
    r = {"a": {"x": 1, "y": 3, "z": 5, "w": 2}, "b": {"x": 1, "y": 3, "z": 5, "w": 2}, "c": {"x": 3, "y": 1, "z": 5, "w": 4}}
    names, m, loo = agreement_table(r)
    assert names == ["a", "b", "c"] and m[0, 1] == 1.0 and m[0, 2] < 1
    assert loo["c"] < loo["a"]
    assert consensus(r) == {"x": 1, "y": 3, "z": 5, "w": 2}
    assert consensus({"a": {"x": 2}, "b": {"x": 3}}) == {"x": 3}  # 짝수 명이면 높은 쪽


K_TRUE, CUTS_TRUE = (0.2, 0.6, 0.9, 1.0), (0.02, 0.05, 0.10, 0.20)  # 초기값과 다른 "사람 기준"


def _synthetic(n=300, seed=1, k_true=K_TRUE, cuts_true=CUTS_TRUE):
    rng = np.random.default_rng(seed)
    regions = []
    for _ in range(n):
        regions.append([{"type": ["dust", "stain", "grime", "heavy"][rng.integers(4)],
                         "area_ratio": 10 ** rng.uniform(-2.5, -0.3), "severity": int(rng.integers(1, 4))}
                        for _ in range(rng.integers(1, 3))])
    x = features(regions)
    return x, levels_of(csi_values(x, np.array(k_true), 3), cuts_true)


def test_fit_recovers_known_thresholds_and_weights():
    """사람 평가 = 알려진 k_t·기준값으로 매긴 level (잡음 없음) → 초기값(config)에서 출발해 복원."""
    x, y = _synthetic()
    cfg = load_config("calibrate")["fit"]
    assert (levels_of(csi_values(x, K0, 3), CUTS0) == y).mean() < 0.6  # 초기값으로는 절반도 안 맞는다
    r = fit(x, y, K0, CUTS0, 3, cfg)
    assert r["qwk"] > 0.99 and (levels_of(csi_values(x, r["k_rel"], 3), r["cuts"]) == y).mean() > 0.95
    # 기준값·k_t는 이웃 데이터 사이 간격만큼만 정해진다 → 근사 복원 + 새 데이터에서 같은 판정인지로 확인
    np.testing.assert_allclose(r["cuts"], CUTS_TRUE, rtol=0.15)
    np.testing.assert_allclose(r["k_rel"], K_TRUE, atol=0.1)
    xt, yt = _synthetic(n=500, seed=2)
    assert (levels_of(csi_values(xt, r["k_rel"], 3), r["cuts"]) == yt).mean() > 0.9
    assert min(cross_validate(x, y, K0, CUTS0, 3, cfg)) > 0.95


def test_fit_keeps_absent_type_weight():
    """평가 데이터에 없는 종류의 k_t는 초기값 그대로 (근거 없이 바꾸지 않는다)."""
    x, y = _synthetic(n=120)
    x[:, 1] = 0  # stain 없음
    r = fit(x, y, K0, CUTS0, 3, load_config("calibrate")["fit"])
    assert r["k_rel"][1] * r["k_rel"].max() == pytest.approx(0.8 / 1.5 * r["k_rel"].max())


def test_fit_cuts_only_keeps_weights():
    x, y = _synthetic(n=200)
    r = fit(x, y, K0, CUTS0, 3, load_config("calibrate")["fit"], cuts_only=True)
    assert r["k_rel"] == pytest.approx(K0 / K0.max())
    assert list(r["cuts"]) == sorted(r["cuts"]) and r["qwk"] > 0.9


def test_load_regions_from_result_and_doc_block(tmp_path):
    res = {"windows": [
        {"images": {"crop": "windows/A/crop.jpg"}, "csi": 0.1, "contamination": [{"type": "dust", "area_ratio": 0.2, "severity": 2}]},
        {"images": {"crop": "windows/B/crop.jpg"}, "csi": None, "contamination": []},  # 미분석 제외
    ]}
    (tmp_path / "result.json").write_text(json.dumps(res))
    reg = load_regions([tmp_path / "result.json"])
    assert list(reg) == ["windows/A/crop.jpg"] and features(list(reg.values())).tolist() == [[0.4, 0, 0, 0]]

    doc = tmp_path / "scoring.md"
    doc.write_text("# CSI\n\n<!-- calibration:start -->\n(아직 없음)\n<!-- calibration:end -->\n\n## 끝\n")
    write_doc(doc, "표 1")
    write_doc(doc, "표 2")  # 다시 실행하면 교체
    s = doc.read_text()
    assert "표 2" in s and "표 1" not in s and "(아직 없음)" not in s and s.endswith("## 끝\n")
