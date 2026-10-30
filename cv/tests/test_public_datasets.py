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
from facade_cv.detect.datasets.ttpla_convert import Tile, clip_to_tile, convert_ttpla, select_tiles, tile_origins, wire_width

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


@pytest.mark.parametrize("fn", [convert_ade20k, lambda s, o: convert_ttpla(s, o)["tiles"]])
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


def _ttpla_cfg(**kw):
    cfg = load_config("dataset")
    cfg["ttpla"] |= {"tile": 100, "max_tiles": 10, "per_video": 4, "wire_only_ratio": 0.3} | kw
    return cfg


def test_ttpla_mapping_case_void_and_tiles(tmp_path):
    """Cable(대문자) → wire, tower_wooden → pole, 철탑 버림. 300×200 → 100px 타일 3×2. void는 회색으로 칠함."""
    cable = [[0, 50], [300, 52], [300, 56], [0, 54]]          # 가로로 타일 3개를 지나는 전선
    wooden = [[120, 20], [130, 20], [130, 90], [120, 90]]     # 가운데 위 타일 안 기둥
    lattice = [[200, 120], [260, 120], [260, 190], [200, 190]]
    void = [[10, 110], [60, 110], [60, 160], [10, 160]]       # 왼쪽 아래 타일
    _labelme(tmp_path / "ttpla", "7_00010", [("Cable", cable), ("tower_wooden", wooden), ("tower_lattice", lattice),
                                            ("void", void)], w=300, h=200)
    _labelme(tmp_path / "ttpla", "7_00020", [("tower_lattice", lattice)], w=300, h=200)  # 매핑 없음 → 타일 없음
    st = convert_ttpla(tmp_path / "ttpla", tmp_path / "out", _ttpla_cfg(), V2)
    assert st["candidates"] == 3  # 전선이 지나는 위쪽 3칸만 후보 (아래쪽 3칸은 wire·pole 없음)
    # 한 프레임에서는 타일 1장 (같은 순간의 타일끼리 중복 회피) → pole 있고 인스턴스 많은 가운데 타일
    assert st["tiles"] == 1 and st["instances"] == {"wire": 1, "pole": 1}
    assert [p.stem for p in (tmp_path / "out" / "labels").glob("*.txt")] == ["7_00010_t1"]
    mid = _read_label(tmp_path / "out" / "labels" / "7_00010_t1.txt")
    assert sorted(c for c, _ in mid) == [WIRE, POLE]
    pole = next(p for c, p in mid if c == POLE)
    np.testing.assert_allclose(pole.min(0), [0.20, 0.20], atol=0.011)  # 타일 좌표 (120−100)/100
    assert cv2.imread(str(tmp_path / "out" / "images" / "7_00010_t1.jpg")).shape[:2] == (100, 100)  # 축소 없음


def test_ttpla_void_filled_gray(tmp_path):
    void = [[10, 10], [60, 10], [60, 60], [10, 60]]
    cable = [[0, 80], [100, 82], [100, 86], [0, 84]]
    _labelme(tmp_path / "ttpla", "1_00001", [("cable", cable), ("void", void)], w=100, h=100)
    convert_ttpla(tmp_path / "ttpla", tmp_path / "out", _ttpla_cfg(), V2)
    img = cv2.imread(str(tmp_path / "out" / "images" / "1_00001_t0.jpg"))
    assert np.abs(img[35, 35].astype(int) - 114).max() <= 2 and img[5, 95].max() <= 2  # void 안 회색, 밖은 원래(검정)


def test_tile_origins_cover_with_overlap():
    tiles = tile_origins(3840, 2160, 1280)
    assert [(x, y) for x, y, _, _ in tiles] == [(0, 0), (1280, 0), (2560, 0), (0, 880), (1280, 880), (2560, 880)]
    assert tile_origins(800, 600, 1280) == [(0, 0, 800, 600)]  # 타일보다 작은 이미지는 통째로


def test_clip_polygon_at_tile_border():
    """타일 경계를 넘는 polygon은 잘라서 남기고, 잘린 조각이 min_area보다 작으면 버린다."""
    rect = np.array([[80, 40], [140, 40], [140, 60], [80, 60]], float)  # x 80~140 — 타일 [0,100) 경계에서 잘림
    inside = clip_to_tile(rect, (0, 0, 100, 100), min_area=10)
    assert len(inside) == 1 and inside[0][:, 0].min() == 80 and inside[0][:, 0].max() == 99
    right = clip_to_tile(rect, (100, 0, 100, 100), min_area=10)
    assert len(right) == 1 and right[0][:, 0].min() == 0 and right[0][:, 0].max() == 40  # 타일 좌표로
    assert clip_to_tile(rect, (0, 0, 100, 100), min_area=10_000) == []  # 너무 작으면 버림
    assert clip_to_tile(rect, (300, 300, 100, 100), min_area=0) == []  # 타일 밖
    # 타일 밖으로 나갔다 들어오는 U자 → 조각 2개
    u = np.array([[10, 10], [30, 10], [30, 150], [70, 150], [70, 10], [90, 10], [90, 180], [10, 180]], float)
    assert len(clip_to_tile(u, (0, 0, 100, 100), min_area=10)) == 2


def test_wire_width_of_strip():
    assert wire_width(np.array([[0, 0], [400, 0], [400, 4], [0, 4]], float)) == pytest.approx(4, rel=0.02)


def test_select_tiles_quota_per_video_and_spread():
    """pole 우선 (1−0.3), wire만 0.3, 영상당 4장, 같은 프레임 두 번 안 고르고 프레임 간격을 벌림."""
    poly = np.array([[0, 0], [10, 0], [10, 10]], float)
    tiles = []
    for v in range(3):  # 영상 3개 × 프레임 0..9 × 타일 2개 (k0 = pole+wire, k1 = wire만)
        for f in range(10):
            tiles.append(Tile(f"{v}_{f:05d}", str(v), f, 0, (0, 0, 1, 1), [(POLE, poly), (WIRE, poly)]))
            tiles.append(Tile(f"{v}_{f:05d}", str(v), f, 1, (0, 0, 1, 1), [(WIRE, poly)]))
    cfg = _ttpla_cfg(max_tiles=10)
    got = select_tiles(tiles, cfg, POLE)
    assert len(got) == 10
    assert sum(any(c == POLE for c, _ in t.instances) for t in got) == 7  # 10 × (1 − 0.3)
    for v in "012":
        frames = [t.frame for t in got if t.video == v]
        assert len(frames) <= 4 and len(set(frames)) == len(frames)
        if len(frames) >= 2:
            assert min(abs(a - b) for i, a in enumerate(frames) for b in frames[i + 1:]) >= 2  # 연속 프레임 아님
    # wire만 있는 타일이 없으면 pole로 채운다
    assert len(select_tiles([t for t in tiles if t.k == 0], cfg, POLE)) == 10

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
