"""박스 → SAM 마스크 → polygon: 품질 필터(면적 비율·조각), 원본 polygon 통과, 증강 사본 1장 — 가짜 마스크·가짜 SAM."""

import cv2
import numpy as np
import yaml

from facade_cv.config import class_names, load_config
from facade_cv.detect.datasets.box_to_seg_sam import convert_box_to_seg, mask_to_instance, originals, parse_label

CFG = {"min_fill": 0.25, "max_fill": 0.95, "min_piece": 0.05}
BOX = (10, 10, 110, 110)  # 100×100 = 10,000px
VEG = class_names("v2").index("vegetation")


def _mask(*rects):
    m = np.zeros((128, 128), bool)
    for x0, y0, x1, y1 in rects:
        m[y0:y1, x0:x1] = True
    return m


def test_fill_ratio_filter():
    assert mask_to_instance(_mask((10, 10, 30, 30)), BOX, CFG, 0.005) == ([], "too_small")  # 4%
    assert mask_to_instance(_mask((10, 10, 110, 110)), BOX, CFG, 0.005) == ([], "too_full")  # 박스 그대로 100%
    polys, why = mask_to_instance(_mask((20, 20, 90, 100)), BOX, CFG, 0.005)  # 56%
    assert why == "kept" and len(polys) == 1
    assert cv2.contourArea(polys[0].astype(np.float32)) == np.float32(69 * 79)  # 외곽선 = 픽셀 중심, 70×80 사각형


def test_mask_clipped_to_box():
    """박스 밖으로 번진 마스크는 박스 안만 센다 → 박스 밖 큰 영역이 있어도 박스 안 50%면 통과, polygon도 박스 안."""
    polys, why = mask_to_instance(_mask((10, 10, 60, 110), (0, 0, 128, 5)), BOX, CFG, 0.005)
    assert why == "kept" and len(polys) == 1 and polys[0][:, 1].min() >= 10


def test_pieces_largest_plus_big_enough():
    """가장 큰 조각 + 박스 면적 5% 이상 조각만 — 4% 조각은 버리고 6% 조각은 따로 polygon."""
    m = _mask((10, 10, 60, 110), (80, 10, 100, 40), (80, 60, 100, 80))  # 50% / 6% / 4%
    polys, why = mask_to_instance(m, BOX, CFG, 0.005)
    assert why == "kept" and len(polys) == 2
    assert sorted(round(cv2.contourArea(p.astype(np.float32))) for p in polys) == [19 * 29, 49 * 99]


def test_parse_label_boxes_and_polygons():
    boxes, polys = parse_label("0 0.5 0.5 0.2 0.4\n0 0.1 0.1 0.3 0.1 0.2 0.3\n\n", 200, 100)
    assert boxes == [(0, (80.0, 30.0, 120.0, 70.0))]
    assert len(polys) == 1 and polys[0][1].shape == (3, 2)


def _export(root):
    """Roboflow 형식: 원본 a는 train 증강 3장, b는 valid 1장(polygon), c는 라벨 없음."""
    root.mkdir(parents=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"names": ["Tree"]}))
    img = np.full((64, 64, 3), 128, np.uint8)
    files = [("train", "a_jpg.rf.0a1"), ("train", "a_jpg.rf.0b2"), ("train", "a_jpg.rf.0c3"),
             ("valid", "b_jpg.rf.0d4"), ("train", "c_jpg.rf.0e5")]
    labels = {"a": "0 0.5 0.5 0.5 0.5\n", "b": "0 0.1 0.1 0.6 0.1 0.4 0.6\n", "c": ""}
    for split, stem in files:
        for d in ("images", "labels"):
            (root / split / d).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(root / split / "images" / f"{stem}.jpg"), img)
        (root / split / "labels" / f"{stem}.txt").write_text(labels[stem[0]])
    return root


def test_converter_end_to_end_with_fake_sam(tmp_path):
    src = tmp_path / "roboflow"
    _export(src / "veg_street_trees" / "Street Trees Detection.v4i.yolov11")  # zip 상위 폴더가 한 겹 더
    assert [p[0].stem[0] for p in originals(src / "veg_street_trees" / "Street Trees Detection.v4i.yolov11")] == ["a", "b", "c"]

    calls = []

    def fake_sam(img, boxes):  # 박스 가운데 절반 크기 사각형 = 박스 면적 25%보다 크게
        calls.append(len(boxes))
        out = []
        for x0, y0, x1, y1 in boxes:
            m = np.zeros(img.shape[:2], bool)
            m[int(y0) + 2:int(y1) - 2, int(x0) + 4:int(x1) - 4] = True
            out.append(m)
        return out

    st = convert_box_to_seg(src, tmp_path / "out", names=class_names("v2"), segment=fake_sam,
                            debug_dir=tmp_path / "dbg")["veg_street_trees"]
    assert calls == [1]  # SAM은 박스 있는 원본 a 한 번만 (증강 사본·polygon 이미지에는 안 부름)
    assert st["images"] == 2 and st["sam_kept"] == 1 and st["polygon_passthrough"] == 1
    assert st["no_label"] == 1 and st["copies_skipped"] == 2
    labels = sorted((tmp_path / "out" / "veg_street_trees" / "labels").glob("*.txt"))
    assert [p.name for p in labels] == ["train_a_jpg.rf.0a1.txt", "valid_b_jpg.rf.0d4.txt"]
    for p in labels:
        rows = [line.split() for line in p.read_text().splitlines()]
        assert all(int(r[0]) == VEG and len(r) >= 7 for r in rows)  # polygon만, 클래스 vegetation(2)
    assert len(list((tmp_path / "dbg").glob("street_trees_*.jpg"))) == 2


def test_config_registered_for_merge():
    cfg = load_config("dataset")
    assert "veg_street_trees" in cfg["box_to_seg"]["datasets"] and "veg_street_trees" in cfg["sources"]
    assert cfg["balance"]["max_images"]["veg_street_trees"] == 0  # 별도 상한 키, 검토 후 미사용 (data/DATASETS.md)
