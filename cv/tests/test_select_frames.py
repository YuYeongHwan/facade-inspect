"""라벨링 프레임 선택: 시간 구간 분산, dHash 유사 프레임 배제, batch 균등, prelabel batch 분할 (모델 없이 점수 주입)."""

import csv

import numpy as np

from facade_cv.config import load_config
from facade_cv.detect.datasets.label_studio.prelabel import split_batches
from facade_cv.detect.datasets.label_studio.select_frames import dhash, hamming, select


def _cands(n=200, dur=100.0, dup_every=1):
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        h = int(rng.integers(0, 2**63)) if i % dup_every == 0 else out[-1]["dhash"]  # dup_every>1: 직전과 같은 해시
        out.append({"index": i * 15, "t": i * dur / n, "score": float(rng.random()), "dhash": h})
    return out


def _sc(**kw):
    return load_config("dataset")["select"] | {"count": 30, "bins": 5, "batches": 3} | kw


def test_select_spreads_over_time_and_batches():
    cands = _cands()
    picked = select(cands, _sc())  # cands에 bin이 붙는다
    assert len(picked) == 30
    assert [sum(p["bin"] == b for p in picked) for b in range(5)] == [6] * 5  # 구간마다 count/bins
    assert [p["t"] for p in picked] == sorted(p["t"] for p in picked)
    assert [sum(p["batch"] == b for p in picked) for b in (1, 2, 3)] == [10, 10, 10]
    for b in range(5):  # 구간 안에서는 점수 높은 것부터
        inside = [c["score"] for c in cands if c["bin"] == b]
        chosen = sorted((p["score"] for p in picked if p["bin"] == b), reverse=True)
        assert chosen == sorted(inside, reverse=True)[:6]


def test_select_skips_near_duplicates():
    """같은 해시(정지 화면)가 4장씩 이어지면 그중 하나만 — 모자라면 fill로 다른 구간에서 채운다."""
    picked = select(_cands(n=200, dup_every=4), _sc())
    hashes = [p["dhash"] for p in picked]
    assert len(set(hashes)) == len(hashes)
    assert all(p["dhash_min"] >= _sc()["dhash_min_dist"] for p in picked)


def test_dhash_similarity():
    img = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (48, 1))[..., None].repeat(3, 2)
    assert hamming(dhash(img), dhash(np.clip(img.astype(int) + 3, 0, 255).astype(np.uint8))) <= 2  # 밝기만 조금
    assert hamming(dhash(img), dhash(img[:, ::-1].copy())) > 30  # 좌우 반전 = 다른 장면


def test_split_batches(tmp_path):
    p = tmp_path / "selection.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, ["image", "batch"])
        w.writeheader()
        w.writerows([{"image": "a.jpg", "batch": 1}, {"image": "b.jpg", "batch": 2}, {"image": "c.jpg", "batch": 1}])
    tasks = [{"n": n} for n in "abcd"]
    out = split_batches(tasks, ["a.jpg", "b.jpg", "c.jpg", "d.jpg"], p)
    assert out == {1: [{"n": "a"}, {"n": "c"}], 2: [{"n": "b"}]}  # csv에 없는 d는 버림
