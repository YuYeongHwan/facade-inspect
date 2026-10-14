"""중력(위) 방향 추정 → Y-up 정렬.

1차: 카메라 업벡터(이미지 -y) 평균. 사람이 건물을 올려다보며 찍으면 업벡터가 한쪽으로 기우므로,
2차(refine_up): 수직에 가까운 외벽 평면의 법선과 수직이 되도록 보정한다 (면 2개 이상이면 두 법선의 외적 방향).
"""

import numpy as np

from facade_cv.recon3d.geometry import UP, CameraView, rotation_between


def estimate_up(cameras: list[CameraView]) -> np.ndarray:
    up = np.mean([c.up for c in cameras], axis=0)
    return up / np.linalg.norm(up)


def refine_up(up: np.ndarray, normals: list[np.ndarray], weights: list[float], iters: int = 10) -> np.ndarray:
    """up을 각 법선에 수직인 평면으로 번갈아 투영 (가중치 큰 순). 법선이 없으면 그대로."""
    order = np.argsort(weights)[::-1]
    for _ in range(iters if normals else 0):
        for i in order:
            n = normals[i] / np.linalg.norm(normals[i])
            up = up - (up @ n) * n
            up /= np.linalg.norm(up)
    return up


def align_rotation(up: np.ndarray) -> np.ndarray:
    """up → +Y 회전 행렬 (월드 X' = R @ X)."""
    return rotation_between(up, UP)


def path_straightness(cameras: list[CameraView]) -> float:
    """정렬 후 카메라 중심의 수평(xz) 분포: 둘째 표준편차 / 첫째. 0에 가까우면 직선 경로."""
    if len(cameras) < 3:
        return 0.0
    xz = np.array([c.center[[0, 2]] for c in cameras])
    s = np.linalg.svd(xz - xz.mean(0), compute_uv=False)
    return float(s[1] / s[0]) if s[0] > 0 else 0.0
