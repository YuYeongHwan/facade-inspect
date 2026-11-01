"""CLIP 제로샷 점수 (clip_score.py)와 사전 등록 비교 도구 (evaluate compare)."""

import copy

import numpy as np
import pytest

from facade_cv.config import load_config
from facade_cv.contamination import clip_score
from facade_cv.contamination.evaluate import adopt, bootstrap_ci, combined, cuts_for, zstats
from facade_cv.schema import Window

CFG = load_config("contamination")
IMG = np.full((64, 64, 3), 120, np.uint8)


def test_clip_none_when_disabled_or_missing(monkeypatch):
    off = copy.deepcopy(CFG)
    off["clip"]["enabled"] = False
    assert clip_score.clip_dirty([IMG, IMG], off) == [None, None]
    monkeypatch.setattr(clip_score, "_model", lambda *a: None)  # open_clip 미설치와 같은 경로
    assert clip_score.clip_dirty([IMG], CFG) == [None]


def test_schema_clip_dirty_optional_ratio():
    f = Window.model_fields["clip_dirty"]
    assert f.default is None and not f.is_required()
    from pydantic import TypeAdapter

    ta = TypeAdapter(f.annotation)
    assert ta.validate_python(0.42) == 0.42 and ta.validate_python(None) is None
    with pytest.raises(Exception):
        ta.validate_python(1.5)


def test_prereg_helpers():
    rng = np.random.default_rng(0)
    y = np.repeat([1, 2, 3, 4, 5], 6)
    s = {"M0": y * 0.01 + rng.random(30) * 0.02, "M1": rng.random(30)}
    z = zstats(s)
    m2 = combined(s, z, 1.0)
    assert np.allclose(m2, (s["M1"] - s["M1"].mean()) / s["M1"].std())
    lo, cuts = cuts_for(s["M0"], y, load_config("calibrate")["fit"])
    assert lo == s["M0"].min() and len(cuts) == 4 and all(a < b for a, b in zip(cuts, cuts[1:]))
    a, b = bootstrap_ci(s["M0"], y, 200, 0)
    assert a < b and (a, b) == bootstrap_ci(s["M0"], y, 200, 0)
    # 채택 규칙: 최고 Spearman이고 구간 하한 > M0 Spearman이어야 채택
    assert adopt({"M0": 0.1, "M1": 0.5, "M2": 0.4}, {"M0": (-.3, .4), "M1": (0.2, .8), "M2": (.1, .7)}) == "M1"
    assert adopt({"M0": 0.1, "M1": 0.5, "M2": 0.4}, {"M0": (-.3, .4), "M1": (0.05, .8), "M2": (.1, .7)}) == "M0"
    assert adopt({"M0": 0.6, "M1": 0.5, "M2": 0.4}, {"M0": (.3, .8), "M1": (.2, .8), "M2": (.1, .7)}) == "M0"


@pytest.mark.slow
def test_clip_real_model_range():
    pytest.importorskip("open_clip")
    s = clip_score.clip_dirty([IMG], CFG)[0]
    assert s is not None and 0 <= s <= 1
