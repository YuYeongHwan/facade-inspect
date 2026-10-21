"""v2 공개 데이터 변환기(ADE20K·TTPLA·Roboflow·CMP door), 균형 조정, 자체 데이터 시간 구간 분할 — 작은 가짜 데이터."""

import json
import re
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from test_datasets import _cmp_label, _read_label, _write_cmp, _yolo_dir

from facade_cv.config import class_names, load_config
from facade_cv.detect.datasets.ade20k_convert import convert_ade20k
from facade_cv.detect.datasets.cmp_convert import convert_cmp
from facade_cv.detect.datasets.merge_datasets import balance, merge
from facade_cv.detect.datasets.roboflow_convert import convert_roboflow
from facade_cv.detect.datasets.ttpla_convert import convert_ttpla

V2 = class_names("v2")
WINDOW, FACADE, VEG, WIRE, POLE, DOOR = range(6)


def test_class_sets():
    assert class_names("v1") == ["window", "facade"] and V2[:2] == class_names("v1")  # v1 id 유지
    assert set(load_config("classes")["obstacles"]) == set(V2[2:])


# ── ADE20K ──
ADE_IDX = {"wall": 1, "building": 2, "sky": 3, "tree": 5, "windowpane": 9, "grass": 10, "door": 15, "plant": 18,
           "flower": 67, "pole": 94}


def _ade(tmp_path, items):
    """items: {stem: (장면, 라벨 (H,W))}."""
    root = tmp_path / "ade20k" / "ADEChallengeData2016"
    (root / "images" / "training").mkdir(parents=True)
    (root / "annotations" / "training").mkdir(parents=True)
    names = {"wall": "wall", "building": "building;edifice", "sky": "sky", "tree": "tree",
             "windowpane": "windowpane;window", "door": "door;double;door", "pole": "pole", "grass": "grass",
             "plant": "plant;flora;plant;life", "flower": "flower"}
    (root / "objectInfo150.csv").write_text(
        "Idx,Ratio,Train,Val,Name\n" + "".join(f'{i},0.1,1,1,"{names[n]}"\n' for n, i in ADE_IDX.items()))
    (root / "sceneCategories.txt").write_text("".join(f"{k} {scene}\n" for k, (scene, _) in items.items()))
    for k, (_, lab) in items.items():
        cv2.imwrite(str(root / "annotations" / "training" / f"{k}.png"), lab)
        cv2.imwrite(str(root / "images" / "training" / f"{k}.jpg"), np.zeros((*lab.shape, 3), np.uint8))
    return tmp_path / "ade20k"


def _street(sky=True):
    lab = np.zeros((100, 100), np.uint8)
    lab[:20] = ADE_IDX["sky"] if sky else 0
    lab[20:80, 10:70] = ADE_IDX["building"]
    lab[30:40, 20:30] = ADE_IDX["windowpane"]
    lab[30:40, 40:50] = ADE_IDX["windowpane"]
    lab[60:80, 30:40] = ADE_IDX["door"]
    lab[20:80, 70:90] = ADE_IDX["wall"]  # 담장 → facade 아님
    lab[40:90, 90:98] = ADE_IDX["tree"]
    return lab


def test_ade20k_outdoor_filter_and_mapping(tmp_path):
    src = _ade(tmp_path, {
        "ADE_train_1": ("street", _street()),
        "ADE_train_2": ("bedroom", _street()),            # 실내 장면 → 버림
        "ADE_train_3": ("street", _street(sky=False)),    # sky 없음 → 버림
    })
    assert convert_ade20k(src, tmp_path / "out", names=V2) == 1
    inst = _read_label(tmp_path / "out" / "labels" / "ADE_train_1.txt")
    by = {c: [p for k, p in inst if k == c] for c in range(6)}
    assert len(by[WINDOW]) == 2 and len(by[DOOR]) == 1 and len(by[VEG]) == 1 and len(by[FACADE]) >= 1
    assert max(p[:, 0].max() for p in by[FACADE]) <= 0.7  # wall(70~90열)은 facade로 안 들어감
    assert not (tmp_path / "out" / "images" / "ADE_train_2.jpg").exists()


def test_ade20k_v1_keeps_two_classes(tmp_path):
    src = _ade(tmp_path, {"ADE_train_1": ("street", _street())})
    convert_ade20k(src, tmp_path / "out", names=class_names("v1"))
    assert {c for c, _ in _read_label(tmp_path / "out" / "labels" / "ADE_train_1.txt")} == {WINDOW, FACADE}


@pytest.mark.parametrize("fn", [convert_ade20k, convert_ttpla])
def test_missing_source_prints_guide(tmp_path, capsys, fn):
    assert fn(tmp_path / "nope", tmp_path / "out") == 0
    assert "skip" in capsys.readouterr().out and not (tmp_path / "out").exists()


# ── TTPLA ──
def _labelme(d, stem, shapes, w=400, h=200):
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / f"{stem}.jpg"), np.zeros((h, w, 3), np.uint8))
    (d / f"{stem}.json").write_text(json.dumps({
        "imageWidth": w, "imageHeight": h, "imagePath": f"{stem}.jpg",
        "shapes": [{"label": lab, "points": pts, "shape_type": "polygon"} for lab, pts in shapes]}))


def test_ttpla_mapping_and_resize(tmp_path):
    cfg = load_config("dataset")
    cfg["ttpla"]["max_side"] = 100
    cable = [[0, 50], [400, 52], [400, 56], [0, 54]]
    wooden = [[100, 20], [110, 20], [110, 200], [100, 200]]
    lattice = [[200, 0], [260, 0], [260, 200], [200, 200]]
    _labelme(tmp_path / "ttpla", "a", [("cable", cable), ("tower_wooden", wooden), ("tower_lattice", lattice)])
    _labelme(tmp_path / "ttpla", "b", [("tower_lattice", lattice), ("void", lattice)])  # 매핑 없음 → 이미지 버림
    assert convert_ttpla(tmp_path / "ttpla", tmp_path / "out", cfg, V2) == 1
    inst = _read_label(tmp_path / "out" / "labels" / "a.txt")
    assert sorted(c for c, _ in inst) == [WIRE, POLE]
    np.testing.assert_allclose(next(p for c, p in inst if c == POLE)[0], [100 / 400, 20 / 200], atol=1e-6)
    assert cv2.imread(str(tmp_path / "out" / "images" / "a.jpg")).shape[:2] == (50, 100)


# ── Roboflow ──
def _roboflow(d, names, lines):
    (d / "train" / "images").mkdir(parents=True)
    (d / "train" / "labels").mkdir(parents=True)
    (d / "data.yaml").write_text(yaml.safe_dump({"names": names, "nc": len(names)}))
    cv2.imwrite(str(d / "train" / "images" / "x.jpg"), np.zeros((8, 8, 3), np.uint8))
    (d / "train" / "labels" / "x.txt").write_text("".join(f"{line}\n" for line in lines))


def _rf_cfg(license="CC BY 4.0"):
    cfg = load_config("dataset")
    cfg["roboflow"]["datasets"] = {"ds": {"url": "u", "license": license, "map": {"Window": "window", "tree": "vegetation"}}}
    return cfg


def test_roboflow_remap(tmp_path):
    _roboflow(tmp_path / "rf" / "ds", ["Window", "car", "tree"],
              ["0 0.1 0.1 0.2 0.1 0.2 0.2", "1 0 0 1 0 1 1", "2 0.5 0.5 0.6 0.5 0.6 0.6", "0 0.5 0.5 0.1 0.1"])  # 마지막: bbox
    assert convert_roboflow(tmp_path / "rf", tmp_path / "out", _rf_cfg(), V2) == {"ds": 1}
    assert [c for c, _ in _read_label(tmp_path / "out" / "ds" / "labels" / "train_x.txt")] == [WINDOW, VEG]


def test_roboflow_rejects_license(tmp_path):
    _roboflow(tmp_path / "rf" / "ds", ["Window"], ["0 0.1 0.1 0.2 0.1 0.2 0.2"])
    with pytest.raises(ValueError, match="license"):
        convert_roboflow(tmp_path / "rf", tmp_path / "out", _rf_cfg("CC BY-NC 4.0"), V2)


# ── CMP door ──
def test_cmp_door_v2_only(tmp_path):
    _write_cmp(tmp_path / "src", "a", _cmp_label())
    convert_cmp(tmp_path / "src", tmp_path / "v2", names=V2)
    convert_cmp(tmp_path / "src", tmp_path / "v1", names=class_names("v1"))
    v2 = _read_label(tmp_path / "v2" / "labels" / "a.txt")
    doors = [p for c, p in v2 if c == DOOR]
    assert len(doors) == 1 and np.allclose(doors[0].min(0) * [100, 80], [45, 50], atol=1)
    assert {c for c, _ in _read_label(tmp_path / "v1" / "labels" / "a.txt")} == {WINDOW, FACADE}


# ── 균형 ──
def test_balance_caps_and_rare_first():
    bc = {"max_total": 6, "max_images": {"cmp": None, "ade20k": 3, "roboflow": 4}}
    pub = {"cmp": [Path(f"c{i}") for i in range(2)], "ade20k": [Path(f"a{i}") for i in range(5)],
           "roboflow_x": [Path(f"r{i}") for i in range(3)], "roboflow_y": [Path(f"s{i}") for i in range(3)]}
    labels = {p: [WINDOW] for v in pub.values() for p in v}
    labels[Path("a4")] = [WINDOW, DOOR]  # 희귀 클래스 → 먼저 남는다
    out = balance(pub, labels, bc, 0)
    assert out["cmp"] == pub["cmp"]  # 상한 없음 (null) → 전체
    n_ade, n_rf = len(out["ade20k"]), len(out["roboflow_x"]) + len(out["roboflow_y"])
    assert (n_ade, n_rf) == (2, 2)  # 남은 4장을 3:4 비율로 (1.71, 2.29 → 최대 나머지 2, 2)
    assert Path("a4") in out["ade20k"]


# ── merge: 공개 test 금지, 접두사, 출처, 시간 구간 분할 ──
def test_merge_rejects_public_test(tmp_path):
    cfg = load_config("dataset")
    cfg["split"]["public"]["test"] = 0.1
    ext = _yolo_dir(tmp_path / "ade", [f"ADE_{i}" for i in range(5)])
    with pytest.raises(ValueError, match="자체"):
        merge(None, [], tmp_path / "m", cfg=cfg, ext={"ade20k": ext})


def test_merge_public_prefix_sources_stats(tmp_path):
    cmp = _yolo_dir(tmp_path / "cmp", [f"cmp_b{i}" for i in range(4)])
    ade = _yolo_dir(tmp_path / "ade", [f"ADE_{i}" for i in range(4)])
    own = _yolo_dir(tmp_path / "own", [f"vid__frame_{i * 700:06d}" for i in range(4)])
    stats = merge(cmp, [own], tmp_path / "m", ext={"ade20k": ade})
    assert stats["test"].keys() == {"own"}  # 시험셋은 자체만
    names = {p.name.split("__")[0] for p in (tmp_path / "m" / "images").rglob("*.jpg")}
    assert names == {"pub_cmp", "pub_ade20k", "own"}
    src = json.loads((tmp_path / "m" / "sources.json").read_text())
    assert src["cmp"]["ref"] == "R1" and src["ade20k"]["ref"] == "R19" and src["ade20k"]["images"] == 4
    st = json.loads((tmp_path / "m" / "stats.json").read_text())
    assert sum(st["classes"][s]["window"]["instances"] for s in st["classes"]) == 12


def test_merge_own_time_segments_with_guard(tmp_path):
    """영상 하나(vid)만 있어도 구간(segment_frames) 단위로 분할, 다른 split 경계 ±guard_frames는 버림."""
    cfg = load_config("dataset")
    sp = cfg["split"] | {"own_group_by": "segment", "segment_frames": 100, "guard_frames": 10}
    cfg["split"] = sp
    idx = list(range(0, 1000, 5))  # 10구간 × 20장
    own = _yolo_dir(tmp_path / "own", [f"vid__frame_{i:06d}" for i in idx])
    stats = merge(None, [own], tmp_path / "m", cfg=cfg)
    split_of = {}
    for s in ("train", "val", "test"):
        for p in (tmp_path / "m" / "images" / s).glob("*.jpg"):
            split_of[int(re.search(r"frame_(\d+)", p.name).group(1))] = s
    seg_split = {}
    for f, s in split_of.items():
        assert seg_split.setdefault(f // 100, s) == s  # 한 구간은 한 split
    assert stats["val"]["own"] and stats["test"]["own"]
    for f, s in split_of.items():  # 경계 근처에 다른 split 프레임이 없다
        for g, t in split_of.items():
            if t != s:
                assert abs(f - g) > 10
    assert json.loads((tmp_path / "m" / "stats.json").read_text())["own_guard_dropped"] > 0
