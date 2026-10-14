"""degrade 모드: SfM 실패·평면 0개일 때 가상 외벽 1개 + T4의 2D row/col.

창문이 가장 많이 검출된 프레임 하나를 소실점으로 정면화해 텍스처로 쓰고 (실패하면 원본 그대로),
창문 사각형을 그 이미지 좌표에서 평면 좌표로 옮긴다. 높이 = degrade.facade_height (상대 단위).
"""

from pathlib import Path

import cv2
import numpy as np

from facade_cv.detect.postprocess import Window, window_hull
from facade_cv.rectify.homography import estimate_rectification
from facade_cv.recon3d.geometry import FacadeFrame
from facade_cv.recon3d.windows3d import LiftedObservation, Window3D, WindowObservation, dedupe_cells


def degrade_facade(det: dict, frames_dir: Path, cfg: dict) -> tuple[FacadeFrame, np.ndarray, float, list[Window3D], str]:
    """(가상 facade, BGR 텍스처, ppu, 창문, 기준 프레임 이름)."""
    if not det["images"]:
        raise ValueError("degrade: no detection images")
    best = max(det["images"], key=lambda r: r["counts"]["final"])
    img = cv2.imread(str(frames_dir / best["image"]))
    if img is None:
        raise FileNotFoundError(frames_dir / best["image"])
    wins = [Window(np.asarray(w["corners"], float), w["conf"], w["row"], w["col"]) for w in best["windows"]]
    rect = estimate_rectification(img, window_hull(wins, img.shape[:2]))
    tex = rect.warp(img) if rect else img
    to_tex = rect.apply if rect else (lambda p: np.asarray(p, float))
    h_px, w_px = tex.shape[:2]
    ppu = h_px / cfg["facade_height"]
    facade = FacadeFrame("F1", np.array([0.0, 0.0, 1.0]), np.zeros(3), w_px / ppu, cfg["facade_height"])

    gridded = [w for w in wins if w.row is not None]
    max_row = max((w.row for w in gridded), default=0)
    windows = []
    for w in gridded:
        q = to_tex(w.corners)
        x0, y0 = np.clip(q.min(0), 0, [w_px, h_px])
        x1, y1 = np.clip(q.max(0), 0, [w_px, h_px])
        if x1 <= x0 or y1 <= y0:
            continue
        r = np.array([x0 / ppu, (h_px - y1) / ppu, x1 / ppu, (h_px - y0) / ppu])
        o = LiftedObservation(WindowObservation(best["image"], w.corners, w.conf), "F1", r)
        windows.append(Window3D("F1", r, [o], row=max_row - w.row + 1, col=w.col + 1))  # T4 row 0 = 위
    dedupe_cells(windows)  # T4 격자는 한 칸에 둘이 들어갈 수 있다
    return facade, tex, ppu, windows, best["image"]
