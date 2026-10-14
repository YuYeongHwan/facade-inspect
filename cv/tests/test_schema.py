import json
import shutil
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from facade_cv.config import load_config
from facade_cv.schema import InspectionResult, load_result

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "fixtures"
WEB = ROOT / "web"


def test_fixture_result_valid():
    r = load_result(FIXTURES / "result.json")
    assert len(r.building.facades) == 4
    assert r.summary.total == len(r.windows) == 20
    assert all(r.summary.level_counts[str(lv)] > 0 for lv in range(1, 6))
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
        lambda d: d["windows"][0].update(cleaning_level=6),
        lambda d: d["windows"][0].update(cleaning_level=0),
        lambda d: d["windows"][0].update(cleaning_label=""),
        lambda d: d["windows"][0].update(grade="A"),
        lambda d: d["windows"][0].update(cleaning_level=None),  # 미분석인데 score·type_ratios가 남아 있음
        lambda d: d["windows"][0].update(score=None),  # 분석됐는데 score 없음
        lambda d: d["windows"][0]["contamination"].append({"type": "mold", "area_ratio": 0.1, "polygon": [[0, 0], [1, 0], [1, 1]]}),
        lambda d: d["windows"][0]["contamination"].append({"type": "dust", "area_ratio": 0.1, "polygon": [[0, 0], [1.5, 0], [1, 1]]}),
        lambda d: d["summary"].update(total=99),
        lambda d: d["summary"]["level_counts"].update({"1": 0}),
        lambda d: d.update(extra_field=1),
    ],
)
def test_invalid_rejected(mutate):
    data = json.loads((FIXTURES / "result.json").read_text())
    mutate(data)
    with pytest.raises(ValidationError):
        InspectionResult.model_validate(data)


@pytest.mark.skipif(not (shutil.which("node") and (WEB / "node_modules/zod").exists()), reason="web 의존성 없음")
def test_fixture_passes_zod():
    """같은 result.json이 web zod 스키마도 통과하고, palette.ts LEVELS가 grading.yaml과 같은지."""
    out = subprocess.run(
        ["node", "scripts/validate-result.mjs", str(FIXTURES / "result.json")],
        cwd=WEB, capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    web_levels = json.loads(out.stdout)
    cfg = {lv["level"]: lv for lv in load_config("grading")["levels"]}
    assert {int(k): v for k, v in web_levels.items()} == {
        k: {"label": lv["label"], "min": lv["min_score"], "color": lv["color"]} for k, lv in cfg.items()
    }


def test_unanalyzed_window_valid():
    data = json.loads((FIXTURES / "result.json").read_text())
    w = data["windows"][0]
    w.update(cleaning_level=None, cleaning_label="미분석", score=None, type_ratios=None, contamination=[])
    w["images"].update(heatmap=None, overlay=None)
    lv = str(load_result(FIXTURES / "result.json").windows[0].cleaning_level)
    data["summary"]["level_counts"][lv] -= 1
    data["summary"]["level_counts"]["unanalyzed"] = 1
    r = InspectionResult.model_validate(data)
    assert r.windows[0].cleaning_level is None and r.summary.level_counts["unanalyzed"] == 1
