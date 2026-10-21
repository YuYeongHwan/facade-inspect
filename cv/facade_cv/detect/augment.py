"""config/augment.yaml → Ultralytics [R8] train 인자 (+ albumentations [R17]).

    YOLO("yolo11s-seg.pt").train(data="data/yolo/merged/data.yaml", **train_augment_args())
"""

from facade_cv.config import load_config


def train_augment_args(cfg: dict | None = None) -> dict:
    """Ultralytics 하이퍼파라미터 + `augmentations`(albumentations 객체 리스트)."""
    import albumentations as A

    args = dict(cfg or load_config("augment"))
    specs = [dict(s) for s in args.pop("albumentations", [])]
    args["augmentations"] = [getattr(A, s.pop("name"))(**s) for s in specs]
    return args
