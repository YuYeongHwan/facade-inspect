"""YOLO11-seg 창문/외벽 분할 래퍼. 가중치·추론 설정은 config/detect.yaml."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from facade_cv.config import CONFIG_DIR, load_config
from facade_cv.device import get_device

CLASS_COLORS = {"window": (220, 60, 40), "facade": (200, 200, 0)}  # BGR


@dataclass
class Detection:
    cls: str  # "window" | "facade"
    mask: np.ndarray  # bool (H, W), 원본 이미지 크기
    polygon: np.ndarray  # (N, 2) float32 픽셀
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1
    conf: float


def split_classes(dets: list[Detection]) -> tuple[list[Detection], list[Detection]]:
    """(windows, facades)."""
    return [d for d in dets if d.cls == "window"], [d for d in dets if d.cls == "facade"]


class WindowSegmenter:
    def __init__(self, weights: str | Path | None = None, cfg: dict | None = None, device: str | None = None):
        from ultralytics import YOLO

        self.cfg = cfg or load_config("detect")
        self.weights = Path(weights) if weights else CONFIG_DIR.parent / self.cfg["weights"]
        if not self.weights.is_file():
            raise FileNotFoundError(f"weights not found: {self.weights}")
        self.device = device or (get_device() if self.cfg["device"] == "auto" else self.cfg["device"])
        self.model = YOLO(str(self.weights))

    def predict(self, frame: np.ndarray, conf: float | None = None) -> list[Detection]:
        """BGR frame → 검출 목록. conf 기본은 config conf (격자 보완 후보용으로 낮춰 부를 수 있다)."""
        c = self.cfg
        res = self.model.predict(frame, imgsz=c["imgsz"], conf=c["conf"] if conf is None else conf, iou=c["iou"],
                                 max_det=c["max_det"], device=self.device, retina_masks=True, verbose=False)[0]
        if res.masks is None:
            return []
        masks = res.masks.data.cpu().numpy() > 0.5
        return [
            Detection(self.model.names[int(k)], m, np.asarray(xy, np.float32), tuple(float(v) for v in box), float(s))
            for m, xy, k, s, box in zip(masks, res.masks.xy, res.boxes.cls.tolist(), res.boxes.conf.tolist(),
                                        res.boxes.xyxy.tolist())
        ]


def draw_detections(img: np.ndarray, dets: list[Detection], alpha: float = 0.5) -> np.ndarray:
    """클래스별 색 마스크 오버레이 (외벽을 먼저 칠해 창문이 위에 오도록)."""
    out = img.copy()
    for d in sorted(dets, key=lambda d: d.cls != "facade"):
        color = np.array(CLASS_COLORS.get(d.cls, (0, 255, 0)), np.float32)
        out[d.mask] = (out[d.mask] * (1 - alpha) + color * alpha).astype(np.uint8)
    return out


def put_label(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.6,
              color: tuple[int, int, int] = (255, 255, 255)) -> None:
    """검은 테두리 글자 (어느 배경에서나 읽히게). OpenCV 5는 굵기에 따라 글자 폭이 달라져 테두리를 이동 복사로 그린다."""
    x, y = int(org[0]), int(org[1])
    for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1), (0, 2), (0, -2), (2, 0), (-2, 0)):
        cv2.putText(img, text, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
