"""Ultralytics YOLO-seg 라벨 공용 함수. 라벨 한 줄 = `cls x1 y1 x2 y2 ...` (0~1 정규화 polygon)."""

from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def simplify_polygon(pts: np.ndarray, eps: float) -> np.ndarray:
    """approxPolyDP, epsilon = eps × 윤곽 길이. (N,2) → (M,2)."""
    c = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    return cv2.approxPolyDP(c, eps * cv2.arcLength(c, True), True)[:, 0].astype(float)


def mask_to_polygons(mask: np.ndarray, eps: float, min_area: float) -> list[np.ndarray]:
    """binary mask → 연결요소별 외곽 polygon (픽셀 좌표). 구멍은 채운다 — YOLO polygon은 구멍을 표현 못 함."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = [simplify_polygon(c[:, 0], eps) for c in contours if cv2.contourArea(c) >= min_area]
    return [p for p in polys if len(p) >= 3]


def write_label(path: Path, instances: list[tuple[int, np.ndarray]]) -> None:
    """instances: (class id, 정규화 polygon (N,2)). 빈 리스트면 빈 파일(배경 이미지)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{cls} " + " ".join(f"{v:.6f}" for v in np.clip(p, 0, 1).ravel()) for cls, p in instances]
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def list_images(images_dir: Path) -> list[Path]:
    return sorted(p for p in Path(images_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def source_missing(src: Path, guide: str) -> bool:
    """공개 데이터 폴더가 없거나 비었으면 다운로드 안내를 출력하고 True (변환기는 건너뛴다 — 에러 아님)."""
    if Path(src).is_dir() and any(Path(src).iterdir()):
        return False
    print(f"skip: {src} 없음 — 사람이 받아 둬야 한다.\n{guide}")
    return True
