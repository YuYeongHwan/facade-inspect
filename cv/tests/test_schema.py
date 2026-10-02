import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from facade_cv.schema import InspectionResult, load_result

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def test_fixture_result_valid():
    r = load_result(FIXTURES / "result.json")
    assert len(r.building.facades) == 4
    assert r.summary.total == len(r.windows) == 20
    assert all(r.summary.grade_counts[g] > 0 for g in "ABCD")
    assert any(w.contamination for w in r.windows)


def test_fixture_files_exist():
    r = load_result(FIXTURES / "result.json")
    paths = [r.building.mesh, *(f.texture for f in r.building.facades)]
    paths += [p for w in r.windows for p in (w.images.crop, w.images.heatmap, w.images.overlay)]
    missing = [p for p in paths if not (FIXTURES / p).is_file()]
    assert not missing


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["windows"][0].update(facade_id="F9"),
        lambda d: d["windows"][0].update(id="F1-R09-C09"),
        lambda d: d["windows"][0].update(grade="E"),
        lambda d: d["windows"][0]["contamination"].append({"type": "mold", "area_ratio": 0.1, "polygon": [[0, 0], [1, 0], [1, 1]]}),
        lambda d: d["windows"][0]["contamination"].append({"type": "dust", "area_ratio": 0.1, "polygon": [[0, 0], [1.5, 0], [1, 1]]}),
        lambda d: d["summary"].update(total=99),
        lambda d: d["summary"]["grade_counts"].update(A=0),
        lambda d: d.update(extra_field=1),
    ],
)
def test_invalid_rejected(mutate):
    data = json.loads((FIXTURES / "result.json").read_text())
    mutate(data)
    with pytest.raises(ValidationError):
        InspectionResult.model_validate(data)
