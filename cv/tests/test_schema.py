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


def _first_dirty(d):
    return next(w for w in d["windows"] if w["contamination"])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["windows"][0].update(facade_id="F9"),
        lambda d: d["windows"][0].update(id="F1-R09-C09"),
        lambda d: d["windows"][0].update(cleaning_level=6),
        lambda d: d["windows"][0].update(cleaning_level=0),
        lambda d: d["windows"][0].update(cleaning_label=""),
        lambda d: d["windows"][0].update(grade="A"),
        lambda d: d["windows"][0].update(cleaning_level=None),  # 미분석인데 csi·type_ratios가 남아 있음
        lambda d: d["windows"][0].update(csi=None),  # 분석됐는데 csi 없음
        lambda d: d["windows"][0].update(occluded_ratio=None),
        lambda d: d["windows"][0].update(score=0.1),  # 옛 필드 이름
        lambda d: d["windows"][0].update(csi=1.2),  # 0~1 밖
        lambda d: _first_dirty(d).update(csi=_first_dirty(d)["csi"] + 0.01),  # Σ 영역 항과 불일치
        lambda d: _first_dirty(d)["contamination"][0].update(severity=4),
        lambda d: _first_dirty(d)["contamination"][0].update(severity=_first_dirty(d)["contamination"][0]["severity"] % 3 + 1),
        lambda d: d["meta"].update(csi_params=None),  # 분석 결과가 있는데 계산 파라미터 없음
        lambda d: d["windows"][0]["contamination"].append(
            {"type": "mold", "area_ratio": 0.1, "severity": 1, "delta_e": 1.0, "polygon": [[0, 0], [1, 0], [1, 1]]}),
        lambda d: d["windows"][0]["contamination"].append(
            {"type": "dust", "area_ratio": 0.1, "severity": 1, "delta_e": 1.0, "polygon": [[0, 0], [1.5, 0], [1, 1]]}),
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
        k: {"label": lv["label"], "min": lv["min_csi"], "color": lv["color"]} for k, lv in cfg.items()
    }


def test_unanalyzed_window_valid():
    data = json.loads((FIXTURES / "result.json").read_text())
    w = data["windows"][0]
    w.update(cleaning_level=None, cleaning_label="미분석", csi=None, type_ratios=None, glass_ratio=None, occluded_ratio=None,
             contamination=[])
    w["images"].update(heatmap=None, overlay=None)
    lv = str(load_result(FIXTURES / "result.json").windows[0].cleaning_level)
    data["summary"]["level_counts"][lv] -= 1
    data["summary"]["level_counts"]["unanalyzed"] = 1
    r = InspectionResult.model_validate(data)
    assert r.windows[0].cleaning_level is None and r.summary.level_counts["unanalyzed"] == 1


def test_fixture_csi_matches_config_and_occlusion():
    """목업은 현재 config로 계산됐고, 가림이 있는 창문이 있다 (상세 패널 시연용)."""
    from facade_cv.contamination.grade import csi_params

    r = load_result(FIXTURES / "result.json")
    assert r.meta.csi_params.model_dump() == csi_params()
    assert sum(w.occluded_ratio > 0 for w in r.windows) >= 2
    assert all(0 <= w.csi <= 1 for w in r.windows)
