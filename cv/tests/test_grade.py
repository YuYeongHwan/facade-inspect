import pytest

from facade_cv.config import load_config
from facade_cv.contamination.grade import cleaning_level, compute_score


@pytest.mark.parametrize(
    "score, level",
    [(0.0, 1), (0.0199, 1), (0.02, 2), (0.0499, 2), (0.05, 3), (0.0999, 3), (0.10, 4), (0.1999, 4), (0.20, 5), (1.0, 5)],
)
def test_level_boundaries(score, level):
    assert cleaning_level(score)[0] == level


def test_config_levels():
    levels = load_config("grading")["levels"]
    assert [lv["level"] for lv in levels] == [1, 2, 3, 4, 5]
    mins = [lv["min_score"] for lv in levels]
    assert mins[0] == 0 and mins == sorted(mins)
    assert cleaning_level(0.25) == (5, "우선 청소")


def test_custom_config():
    cfg = {"weights": {"dust": 1.0}, "levels": [{"level": 1, "label": "a", "min_score": 0}, {"level": 2, "label": "b", "min_score": 0.5}]}
    assert compute_score({"dust": 0.6}, cfg) == 0.6
    assert cleaning_level(0.6, cfg) == (2, "b")


def test_score_weights():
    assert compute_score({"dust": 0.02, "stain": 0.0, "grime": 0.03, "heavy": 0.01}) == pytest.approx(0.01 + 0.03 + 0.015)
