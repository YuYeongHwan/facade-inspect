"""외벽 정사영 텍스처: facade 평면 texel마다 각 프레임으로 투영 (plane-induced 사상 — 핀홀이면 호모그래피와 같고,
카메라 왜곡·어안도 그대로 처리) → texel별로 가장 정면으로 본 상위 K개 뷰의 median (반사·지나가는 가림 감소).

이미지 좌표: 텍스처 x = u·ppu, y = (height - v)·ppu (위가 지붕). ppu = 단위당 픽셀.
ponytail: 가림(occlusion) 검사 없음 — 박스 가정. 나무·돌출부가 외벽을 가리면 그 색이 섞일 수 있다.
"""

import warnings
from collections.abc import Callable

import cv2
import numpy as np

from facade_cv.recon3d.geometry import CameraView, FacadeFrame

STEP = 4  # 투영은 STEP 픽셀 격자에서 계산하고 보간 (왜곡은 부드럽다)


def texture_size(f: FacadeFrame, max_side: int) -> tuple[int, int, float]:
    ppu = max_side / max(f.width, f.height)
    return max(1, round(f.width * ppu)), max(1, round(f.height * ppu)), ppu


def _texel_points(f: FacadeFrame, w: int, h: int, ppu: float) -> np.ndarray:
    """(h/STEP+1, w/STEP+1, 3) 성긴 격자의 월드 좌표."""
    xs = np.linspace(0, w, int(np.ceil(w / STEP)) + 1)
    ys = np.linspace(0, h, int(np.ceil(h / STEP)) + 1)
    X, Y = np.meshgrid(xs, ys)
    uv = np.stack([X.ravel() / ppu, f.height - Y.ravel() / ppu], 1)
    return f.to_world(uv).reshape(len(ys), len(xs), 3)


def build_textures(facades: list[FacadeFrame], cameras: list[CameraView], load: Callable[[str], np.ndarray],
                   cfg: dict) -> dict[str, tuple[np.ndarray, float, np.ndarray]]:
    """facade id → (BGR 텍스처, ppu, texel별 사용한 뷰 수). 이미지는 뷰마다 한 번만 읽는다."""
    K, min_cos = cfg["top_k"], cfg["min_cos"]
    state = {}
    for f in facades:
        w, h, ppu = texture_size(f, cfg["max_side_px"])
        state[f.id] = {"w": w, "h": h, "ppu": ppu, "pts": _texel_points(f, w, h, ppu),
                       "cos": np.full((K, h, w), -1.0, np.float32), "col": np.zeros((K, h, w, 3), np.uint8)}
    for cam in cameras:
        todo = [f for f in facades if (cam.center - f.origin) @ f.normal > 0]  # 외벽 앞쪽에서 본 뷰만
        if not todo:
            continue
        img = None
        for f in todo:
            s = state[f.id]
            pts = s["pts"]
            px, ok = cam.project(pts.reshape(-1, 3))
            if not ok.any():
                continue
            ray = cam.center - pts.reshape(-1, 3)
            cos = (ray @ f.normal) / np.linalg.norm(ray, axis=1)
            cos = np.where(ok, cos, -1.0).reshape(pts.shape[:2]).astype(np.float32)
            if cos.max() < min_cos:
                continue
            img = load(cam.name) if img is None else img
            size = (s["w"], s["h"])
            mx = cv2.resize(np.nan_to_num(px[:, 0], nan=-1).reshape(pts.shape[:2]).astype(np.float32), size)
            my = cv2.resize(np.nan_to_num(px[:, 1], nan=-1).reshape(pts.shape[:2]).astype(np.float32), size)
            c = cv2.resize(cos, size, interpolation=cv2.INTER_NEAREST)
            col = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            slot = np.argmin(s["cos"], axis=0)
            cur = np.take_along_axis(s["cos"], slot[None], 0)[0]
            rr, cc = np.nonzero((c > cur) & (c >= min_cos))
            k = slot[rr, cc]
            s["cos"][k, rr, cc] = c[rr, cc]
            s["col"][k, rr, cc] = col[rr, cc]
    out = {}
    for f in facades:
        s = state[f.id]
        valid = s["cos"] >= min_cos
        col = np.where(valid[..., None], s["col"].astype(np.float32), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # 아무 뷰도 없는 texel의 All-NaN slice
            med = np.nanmedian(col, axis=0)
        tex = np.nan_to_num(med, nan=40).clip(0, 255).astype(np.uint8)  # 아무 뷰도 못 본 곳 = 어두운 회색
        out[f.id] = (tex, s["ppu"], valid.sum(0).astype(np.uint8))
    return out


def rect_to_px(rect: tuple[float, float, float, float], height: float, ppu: float) -> tuple[int, int, int, int]:
    """facade_rect [u0, v0, u1, v1] → 텍스처 픽셀 (x0, y0, x1, y1)."""
    u0, v0, u1, v1 = rect
    return round(u0 * ppu), round((height - v1) * ppu), round(u1 * ppu), round((height - v0) * ppu)
