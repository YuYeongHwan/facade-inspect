"""테스트용 합성 외벽: 회색 벽 + 3x4 창문 격자를 카메라 회전(K R K⁻¹)으로 원근 변환. 마스크는 Detection으로 직접 만든다."""

import cv2
import numpy as np

from facade_cv.detect.segmenter import Detection

W, H = 1280, 960
ROWS, COLS = 3, 4
WIN_W, WIN_H, GAP_X, GAP_Y = 110, 140, 70, 90
FOCAL = 0.8 * W


def rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    y, p = np.radians([yaw_deg, pitch_deg])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    return ry @ rx


def _det(cls: str, corners: np.ndarray, conf: float) -> Detection:
    mask = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(mask, np.round(corners).astype(np.int32), 1)
    (x0, y0), (x1, y1) = corners.min(0), corners.max(0)
    return Detection(cls, mask.astype(bool), corners.astype(np.float32), (x0, y0, x1, y1), conf)


def make_scene(yaw: float = 20, pitch: float = -10):
    """(이미지, {(row, col): window Detection}, facade Detection, H_true 정면 좌표 → 이미지).
    정면 좌표는 벽 왼쪽 위가 원점인 픽셀. 이미지엔 12개 창문을 모두 그린다."""
    K = np.array([[FOCAL, 0, W / 2], [0, FOCAL, H / 2], [0, 0, 1]])
    Hrot = K @ rotation(yaw, pitch) @ np.linalg.inv(K)
    wall_w = COLS * WIN_W + (COLS + 1) * GAP_X
    wall_h = ROWS * WIN_H + (ROWS + 1) * GAP_Y
    # 벽 중심이 이미지 중심에 오도록 정면 좌표를 옮긴다
    c = np.linalg.inv(Hrot) @ [W / 2, H / 2, 1]
    shift = np.array([[1, 0, c[0] / c[2] - wall_w / 2], [0, 1, c[1] / c[2] - wall_h / 2], [0, 0, 1]])
    H_true = Hrot @ shift

    pad = 200
    canvas = np.full((wall_h + 2 * pad, wall_w + 2 * pad, 3), 215, np.uint8)  # 하늘
    cv2.rectangle(canvas, (pad, pad), (pad + wall_w, pad + wall_h), (150, 150, 150), -1)
    for y in range(pad + 30, pad + wall_h, 60):  # 벽 줄눈
        cv2.line(canvas, (pad, y), (pad + wall_w, y), (125, 125, 125), 2)
    rects = {}
    for r in range(ROWS):
        for k in range(COLS):
            x0, y0 = GAP_X + k * (WIN_W + GAP_X), GAP_Y + r * (WIN_H + GAP_Y)
            rects[r, k] = np.array([[x0, y0], [x0 + WIN_W, y0], [x0 + WIN_W, y0 + WIN_H], [x0, y0 + WIN_H]], float)
            cv2.rectangle(canvas, (pad + x0, pad + y0), (pad + x0 + WIN_W, pad + y0 + WIN_H), (235, 235, 235), -1)
            cv2.rectangle(canvas, (pad + x0 + 8, pad + y0 + 8), (pad + x0 + WIN_W - 8, pad + y0 + WIN_H - 8),
                          (60, 50, 40), -1)
            cv2.line(canvas, (pad + x0 + WIN_W // 2, pad + y0), (pad + x0 + WIN_W // 2, pad + y0 + WIN_H),
                     (235, 235, 235), 6)
    to_canvas = np.array([[1, 0, pad], [0, 1, pad], [0, 0, 1]], float)
    img = cv2.warpPerspective(canvas, H_true @ np.linalg.inv(to_canvas), (W, H), borderValue=(215, 215, 215))

    def proj(pts):
        return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H_true).reshape(-1, 2)

    windows = {rc: _det("window", proj(q), 0.9) for rc, q in rects.items()}
    wall = np.array([[0, 0], [wall_w, 0], [wall_w, wall_h], [0, wall_h]], float)
    return img, windows, _det("facade", proj(wall), 0.9), H_true


def window_det(corners_front: np.ndarray, H_true: np.ndarray, conf: float = 0.9) -> Detection:
    """정면 좌표 사각형 → 이미지 Detection (격자 밖 검출 등 추가용)."""
    return _det("window", cv2.perspectiveTransform(corners_front.reshape(-1, 1, 2), H_true).reshape(-1, 2), conf)
