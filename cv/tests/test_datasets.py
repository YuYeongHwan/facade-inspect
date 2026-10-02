import json

import cv2
import numpy as np
import pytest
import yaml
from PIL import Image

from facade_cv.config import load_config
from facade_cv.detect.augment import train_augment_args
from facade_cv.detect.datasets.cmp_convert import convert_cmp
from facade_cv.detect.datasets.label_studio.export_to_yolo import export_to_yolo
from facade_cv.detect.datasets.label_studio.prelabel import to_ls_task
from facade_cv.detect.datasets.merge_datasets import merge

W, H = 100, 80
WINDOW, FACADE = 0, 1


def _read_label(path):
    out = []
    for line in path.read_text().splitlines():
        v = line.split()
        out.append((int(v[0]), np.array(v[1:], float).reshape(-1, 2)))
    return out


def _px(poly):
    return (poly * [W, H]).astype(np.float32)


def _inside(poly, x, y):
    return cv2.pointPolygonTest(_px(poly), (x, y), False) > 0


BLIND = 8
CMP_WINDOWS = [(20, 20, 40, 35), (60, 20, 80, 35)]  # XML 창문 사각형 (col0, row0, col1, row1) 픽셀


def _cmp_label():
    """CMP index 라벨: 1 background, 2 facade, 3 window, 4 door, 5 cornice."""
    lab = np.full((H, W), 1, np.uint8)
    lab[10:70, 10:90] = 2
    lab[5:10, 10:90] = 5  # cornice → facade로 합쳐짐
    for c0, r0, c1, r1 in CMP_WINDOWS:
        lab[r0:r1, c0:c1] = 3
    lab[50:70, 45:55] = 4  # door: facade 아래쪽 경계에 닿음 → facade에서 빠짐
    lab[60, 20] = 3  # XML 사각형 밖 창문 픽셀 → 무시
    return lab


def _write_cmp(dirpath, stem, lab, windows=CMP_WINDOWS):
    """CMP 형식: 팔레트 PNG + jpg + XML. XML의 <x>는 세로(행), <y>는 가로(열), 0~1 정규화."""
    dirpath.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(lab, mode="P")
    im.putpalette([0, 0, 0, 0, 0, 170, 0, 0, 255, 0, 85, 255, 0, 170, 255, 0, 255, 255] + [128] * 3 * 250)
    im.save(dirpath / f"{stem}.png")
    cv2.imwrite(str(dirpath / f"{stem}.jpg"), np.zeros((H, W, 3), np.uint8))
    objs = [
        f"<object><points><x>{r0 / H}</x><x>{r1 / H}</x><y>{c0 / W}</y><y>{c1 / W}</y></points>"
        f"<label>3</label><labelname>window</labelname><flag>1</flag></object>"
        for c0, r0, c1, r1 in windows
    ]
    objs.append("<object><points><x>0.1</x><x>0.9</x><y>0.1</y><y>0.9</y></points>"
                "<label>2</label><labelname>facade</labelname><flag>1</flag></object>")
    (dirpath / f"{stem}.xml").write_text("".join(objs))


def _boxes(polys):
    return sorted((p.min(0) * [W, H]).round().tolist() + (p.max(0) * [W, H]).round().tolist() for p in polys)


def _convert_one(tmp_path, lab, windows, cfg=None):
    _write_cmp(tmp_path / "src", "a", lab, windows)
    convert_cmp(tmp_path / "src", tmp_path / "out", cfg)
    inst = _read_label(tmp_path / "out" / "labels" / "a.txt")
    return [p for c, p in inst if c == WINDOW], [p for c, p in inst if c == FACADE]


def test_cmp_convert(tmp_path):
    _write_cmp(tmp_path / "src" / "base", "cmp_b0001", _cmp_label())
    _write_cmp(tmp_path / "src" / "extended", "cmp_x0001", np.full((H, W), 1, np.uint8), [])  # 배경뿐
    assert convert_cmp(tmp_path / "src", tmp_path / "out") == 2

    inst = _read_label(tmp_path / "out" / "labels" / "cmp_b0001.txt")
    windows = [p for c, p in inst if c == WINDOW]
    facades = [p for c, p in inst if c == FACADE]
    assert len(windows) == 2 and len(facades) == 1

    assert _boxes(windows) == [[20, 20, 39, 34], [60, 20, 79, 34]]  # 1px 창문은 사각형 밖이라 무시
    assert all(len(p) == 4 for p in windows)  # 사각형은 4점으로 단순화

    facade = facades[0]
    assert (facade.min(0) * [W, H]).round().tolist() == [10, 5]  # cornice 포함
    assert _inside(facade, 30, 27)  # 창문 구멍은 채워짐
    assert not _inside(facade, 50, 65)  # door 제외
    assert not _inside(facade, 5, 40)  # background 제외

    assert (tmp_path / "out" / "labels" / "cmp_x0001.txt").read_text() == ""
    assert (tmp_path / "out" / "images" / "cmp_b0001.jpg").is_file()


def test_cmp_class_mapping_from_config(tmp_path):
    cfg = load_config("dataset")
    cfg["cmp"]["facade"] = ["facade"]  # cornice를 벽 계열에서 빼면 facade 위쪽 경계가 내려간다
    _write_cmp(tmp_path / "src", "a", _cmp_label())
    convert_cmp(tmp_path / "src", tmp_path / "out", cfg)
    facade = next(p for c, p in _read_label(tmp_path / "out" / "labels" / "a.txt") if c == FACADE)
    assert (facade.min(0) * [W, H]).round().tolist() == [10, 10]


def test_cmp_touching_windows_split_by_xml(tmp_path):
    """붙어 있는 두 창문은 픽셀상 연결요소 하나지만 XML 사각형 기준으로 2개."""
    lab = np.full((H, W), 2, np.uint8)
    lab[20:35, 20:60] = 3
    windows, _ = _convert_one(tmp_path, lab, [(20, 20, 40, 35), (40, 20, 60, 35)])
    assert _boxes(windows) == [[20, 20, 39, 34], [40, 20, 59, 34]]


def test_cmp_h_shaped_window_becomes_rect(tmp_path):
    """덧문(blind)이 창문 픽셀을 H자로 가르면 사각형 ∩ (window ∪ blind)로 사각형 복원.
    사각형 밖 blind(옆 덧문)는 창문에 안 들어가고 facade로 남는다."""
    lab = np.full((H, W), 2, np.uint8)
    lab[30:60, 20:50] = BLIND
    lab[30:60, 20:26] = 3
    lab[30:60, 44:50] = 3
    lab[42:48, 20:50] = 3  # H
    lab[30:60, 52:60] = BLIND  # 사각형 밖 덧문
    windows, facades = _convert_one(tmp_path, lab, [(20, 30, 50, 60)])
    assert len(windows) == 1 and len(windows[0]) == 4
    assert _boxes(windows) == [[20, 30, 49, 59]]
    assert len(facades) == 1 and _inside(facades[0], 56, 45)


def test_cmp_h_shaped_without_blind_in_window_classes(tmp_path):
    """config에서 blind를 빼면 H자 그대로 (사각형이 아님) — 매핑이 config를 따르는지 확인."""
    lab = np.full((H, W), 2, np.uint8)
    lab[30:60, 20:50] = BLIND
    lab[30:60, 20:26] = 3
    lab[30:60, 44:50] = 3
    lab[42:48, 20:50] = 3
    cfg = load_config("dataset")
    cfg["cmp"]["window"] = ["window"]
    windows, _ = _convert_one(tmp_path, lab, [(20, 30, 50, 60)], cfg)
    assert len(windows) == 1 and len(windows[0]) > 4
    assert not _inside(windows[0], 35, 35)  # 덧문 부분은 창문 밖


def test_cmp_requires_xml(tmp_path):
    _write_cmp(tmp_path / "src", "a", _cmp_label())
    (tmp_path / "src" / "a.xml").unlink()
    with pytest.raises(FileNotFoundError, match="XML"):
        convert_cmp(tmp_path / "src", tmp_path / "out")


def test_cmp_rejects_rgb_png(tmp_path):
    (tmp_path / "src").mkdir()
    cv2.imwrite(str(tmp_path / "src" / "a.png"), np.zeros((H, W, 3), np.uint8))
    cv2.imwrite(str(tmp_path / "src" / "a.jpg"), np.zeros((H, W, 3), np.uint8))
    with pytest.raises(ValueError, match="palette"):
        convert_cmp(tmp_path / "src", tmp_path / "out")


def test_prelabel_roundtrip_through_label_studio_export(tmp_path):
    """prelabel 출력을 라벨러가 그대로 승인했다고 보고 export → YOLO 변환하면 polygon이 보존돼야 한다."""
    win = np.array([[0.1, 0.2], [0.3, 0.2], [0.3, 0.4], [0.1, 0.4]])
    fac = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    task = to_ls_task("/data/upload/1/8f3a2b1c-img_a.jpg", W, H, [("window", win, 0.9), ("facade", fac, 0.8)], "m")
    pred = task["predictions"][0]
    assert pred["score"] == pytest.approx(0.85)

    export = [
        {"id": 1, "data": task["data"], "annotations": [{"result": pred["result"], "updated_at": "2026-10-02"}]},
        {"id": 2, "data": {"image": "/data/local-files/?d=raw/img_b.jpg"}, "annotations": [
            {"result": [], "updated_at": "2026-10-03"},  # 최신 = 객체 없음
            {"result": pred["result"], "updated_at": "2026-10-01"},
        ]},
        {"id": 3, "data": {"image": "/data/upload/1/img_c.jpg"}, "annotations": [{"result": [], "was_cancelled": True}]},
        {"id": 4, "data": {"image": "/data/upload/1/img_d.jpg"}, "annotations": []},  # 미라벨
    ]
    (tmp_path / "raw").mkdir()
    for n in "abcd":
        cv2.imwrite(str(tmp_path / "raw" / f"img_{n}.jpg"), np.zeros((H, W, 3), np.uint8))
    (tmp_path / "export.json").write_text(json.dumps(export))

    assert export_to_yolo(tmp_path / "export.json", tmp_path / "raw", tmp_path / "out") == 2
    inst = _read_label(tmp_path / "out" / "labels" / "img_a.txt")
    assert [c for c, _ in inst] == [WINDOW, FACADE]
    np.testing.assert_allclose(inst[0][1], win, atol=1e-5)
    np.testing.assert_allclose(inst[1][1], fac, atol=1e-5)
    assert (tmp_path / "out" / "labels" / "img_b.txt").read_text() == ""
    assert sorted(p.name for p in (tmp_path / "out" / "images").iterdir()) == ["img_a.jpg", "img_b.jpg"]


def _yolo_dir(root, stems):
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    for s in stems:
        cv2.imwrite(str(root / "images" / f"{s}.jpg"), np.zeros((8, 8, 3), np.uint8))
        (root / "labels" / f"{s}.txt").write_text("0 0.1 0.1 0.2 0.1 0.2 0.2\n")
    return root


def test_merge_splits(tmp_path):
    cmp = _yolo_dir(tmp_path / "cmp", [f"cmp_b{i:04d}" for i in range(20)])
    own = _yolo_dir(tmp_path / "own", [f"flight{g}__frame_{i:03d}" for g in "ABC" for i in range(4)])
    stats = merge(cmp, [own], tmp_path / "merged")
    out = tmp_path / "merged"

    assert stats["val"].get("own", 0) >= 1 and stats["test"].get("own", 0) >= 1
    assert sum(d.get("cmp", 0) for d in stats.values()) == 20
    assert sum(d.get("own", 0) for d in stats.values()) == 12
    assert stats["test"].get("cmp", 0) == 0  # config: cmp test 비율 0

    # 같은 비행(그룹)의 프레임은 한 split에만
    split_of = {}
    for s in ("train", "val", "test"):
        for p in (out / "images" / s).glob("own__*.jpg"):
            split_of.setdefault(p.name.split("__")[1], set()).add(s)
    assert len(split_of) == 3 and all(len(v) == 1 for v in split_of.values())

    for s in ("train", "val", "test"):
        imgs = {p.stem for p in (out / "images" / s).iterdir()}
        assert imgs == {p.stem for p in (out / "labels" / s).iterdir()}

    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["names"] == {0: "window", 1: "facade"}
    assert (out / data["val"]).is_dir()

    with pytest.raises(FileExistsError):
        merge(cmp, [own], out)
    merge(cmp, [own], out, overwrite=True)


def test_merge_requires_two_own_groups(tmp_path):
    own = _yolo_dir(tmp_path / "own", ["flightA__0", "flightA__1"])
    with pytest.raises(ValueError, match="2 groups"):
        merge(None, [own], tmp_path / "merged")


def test_train_augment_args_are_valid_ultralytics_cfg():
    import albumentations as A
    from ultralytics.cfg import get_cfg

    args = train_augment_args()
    assert {type(t) for t in args["augmentations"]} == {A.RandomBrightnessContrast, A.MotionBlur}
    cfg = get_cfg(overrides=args)  # 알 수 없는 키/잘못된 타입이면 예외
    assert cfg.perspective > 0 and cfg.flipud == 0


def test_merge_rejects_ungrouped_own_names(tmp_path):
    own = _yolo_dir(tmp_path / "own", ["flightA__0", "flightB__0", "IMG_0001", "IMG_0002"])
    with pytest.raises(ValueError, match="IMG_0001"):
        merge(None, [own], tmp_path / "merged")
    assert not (tmp_path / "merged").exists()  # 검증 실패 시 아무것도 안 만든다

    stats = merge(None, [own], tmp_path / "merged", allow_ungrouped=True)  # 각자 한 그룹 → 4그룹
    assert sum(d.get("own", 0) for d in stats.values()) == 4


def test_merge_validates_before_overwrite(tmp_path):
    good = _yolo_dir(tmp_path / "good", ["flightA__0", "flightB__0", "flightC__0"])
    bad = _yolo_dir(tmp_path / "bad", ["IMG_0001", "IMG_0002"])
    merge(None, [good], tmp_path / "merged")
    with pytest.raises(ValueError):
        merge(None, [bad], tmp_path / "merged", overwrite=True)
    assert len(list((tmp_path / "merged" / "images").rglob("*.jpg"))) == 3  # 기존 결과 보존


def test_cmp_min_area_per_class(tmp_path):
    cfg = load_config("dataset")
    cfg["min_area_ratio"] = {"window": 0.001, "facade": 0.01}  # 8px / 80px (100×80 이미지)
    lab = np.full((H, W), 1, np.uint8)
    lab[10:70, 10:60] = 2
    lab[20:24, 20:24] = 3  # 16px 창문 → 유지
    lab[40:42, 20:22] = 3  # 4px 창문 → 제거
    lab[10:16, 80:86] = 2  # 36px 외벽 조각 → 제거 (창문 기준이면 남았을 크기)
    windows, facades = _convert_one(tmp_path, lab, [(20, 20, 24, 24), (20, 40, 22, 42)], cfg)
    assert _boxes(windows) == [[20, 20, 23, 23]]
    assert len(facades) == 1


def test_merge_cmp_only_omits_empty_test(tmp_path):
    """자체 데이터 없이 CMP만 (STAGE1_ONLY 학습용): test 비율 0 → data.yaml에 test 키 없음."""
    cmp = _yolo_dir(tmp_path / "cmp", [f"cmp_b{i:04d}" for i in range(20)])
    stats = merge(cmp, [], tmp_path / "merged")
    assert stats["test"] == {} and sum(stats["train"].values()) + sum(stats["val"].values()) == 20
    data = yaml.safe_load((tmp_path / "merged" / "data.yaml").read_text())
    assert "test" not in data and data["train"] == "images/train" and data["val"] == "images/val"
