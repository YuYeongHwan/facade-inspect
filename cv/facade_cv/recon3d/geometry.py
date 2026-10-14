"""recon3d 공용 기하: 카메라 뷰, 외벽 평면 좌표계. pycolmap에 묶이지 않게 numpy만 쓴다 (테스트는 합성 카메라)."""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

UP = np.array([0.0, 1.0, 0.0])  # 정렬 후 월드 Y-up


@dataclass
class CameraView:
    """한 프레임의 카메라. X_cam = R @ X_world + t (COLMAP cam_from_world, x 오른쪽·y 아래·z 앞)."""

    name: str
    R: np.ndarray
    t: np.ndarray
    width: int
    height: int
    img_from_cam: Callable[[np.ndarray], np.ndarray]  # (N,3) 카메라 좌표 → (N,2) 픽셀 (왜곡 포함)
    ray_from_img: Callable[[np.ndarray], np.ndarray]  # (N,2) 픽셀 → (N,3) 카메라 좌표 광선

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    @property
    def fov_tan(self) -> np.ndarray:
        """이미지 테두리 광선의 |x/z|, |y/z| 최대 (왜곡 없는 화각)."""
        if not hasattr(self, "_fov_tan"):
            w, h = self.width - 1, self.height - 1
            edge = np.array([[0, 0], [w / 2, 0], [w, 0], [w, h / 2], [w, h], [w / 2, h], [0, h], [0, h / 2]], float)
            d = self.ray_from_img(edge)
            self._fov_tan = np.abs(d[:, :2] / d[:, 2:3]).max(0)
        return self._fov_tan

    @property
    def up(self) -> np.ndarray:
        """월드 좌표의 카메라 위쪽 (이미지 -y)."""
        return -self.R[1]

    def project(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N,3) 월드 → (픽셀 (N,2), 보이는지 (N,)) — 카메라 앞 + 화각 안 + 이미지 안.
        화각 검사를 왜곡 적용 전에 한다: 왜곡 다항식은 보정 범위 밖에서 접혀 화각 밖 점이 이미지 안으로 돌아온다."""
        Xc = X @ self.R.T + self.t
        with np.errstate(divide="ignore", invalid="ignore"):
            tan = np.abs(Xc[:, :2] / Xc[:, 2:3])
        front = (Xc[:, 2] > 1e-6) & (tan <= self.fov_tan * 1.02).all(1)
        px = np.full((len(X), 2), np.nan)
        if front.any():
            px[front] = self.img_from_cam(Xc[front])
        ok = front & (px[:, 0] >= 0) & (px[:, 0] <= self.width - 1) & (px[:, 1] >= 0) & (px[:, 1] <= self.height - 1)
        return px, ok

    def rays(self, px: np.ndarray) -> np.ndarray:
        """(N,2) 픽셀 → (N,3) 월드 방향 (단위벡터)."""
        d = self.ray_from_img(np.asarray(px, float)) @ self.R
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def rotated(self, Rw: np.ndarray) -> "CameraView":
        """월드를 X' = Rw @ X로 돌린 뒤의 같은 카메라."""
        return CameraView(self.name, self.R @ Rw.T, self.t, self.width, self.height, self.img_from_cam,
                          self.ray_from_img)

    @classmethod
    def pinhole(cls, name: str, R: np.ndarray, t: np.ndarray, f: float, width: int, height: int) -> "CameraView":
        cx, cy = width / 2, height / 2

        def img_from_cam(Xc):
            return np.stack([f * Xc[:, 0] / Xc[:, 2] + cx, f * Xc[:, 1] / Xc[:, 2] + cy], 1)

        def ray_from_img(px):
            d = np.stack([(px[:, 0] - cx) / f, (px[:, 1] - cy) / f, np.ones(len(px))], 1)
            return d / np.linalg.norm(d, axis=1, keepdims=True)

        return cls(name, np.asarray(R, float), np.asarray(t, float), width, height, img_from_cam, ray_from_img)

    @classmethod
    def look_at(cls, name: str, eye, target, f: float = 800, width: int = 1280, height: int = 720,
                up=UP) -> "CameraView":
        """합성용: eye에서 target을 보는 핀홀 카메라."""
        eye, target = np.asarray(eye, float), np.asarray(target, float)
        z = target - eye
        z /= np.linalg.norm(z)
        x = np.cross(z, up)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z])
        return cls.pinhole(name, R, -R @ eye, f, width, height)


@dataclass
class FacadeFrame:
    """외벽 평면 좌표계 (schema.Facade와 같은 규칙).

    normal: 건물 밖을 향함. u_axis = UP × normal (밖에서 봤을 때 오른쪽), v_axis = UP.
    origin: 외벽 왼쪽 아래 (지면). 평면 위 점 = origin + u·u_axis + v·v_axis, 0 ≤ u ≤ width, 0 ≤ v ≤ height
    """

    id: str
    normal: np.ndarray
    origin: np.ndarray
    width: float
    height: float
    inliers: np.ndarray = field(default_factory=lambda: np.empty(0, int), repr=False)  # planes: 포인트 index

    @property
    def u_axis(self) -> np.ndarray:
        u = np.cross(UP, self.normal)
        return u / np.linalg.norm(u)

    @property
    def v_axis(self) -> np.ndarray:
        return UP.copy()

    @property
    def offset(self) -> float:
        """평면 n·x + d = 0 의 d."""
        return -float(self.normal @ self.origin)

    def to_uv(self, X: np.ndarray) -> np.ndarray:
        """(N,3) → (N,2) 평면 좌표 (평면에 수직 투영)."""
        D = np.asarray(X) - self.origin
        return np.stack([D @ self.u_axis, D @ self.v_axis], 1)

    def to_world(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, float)
        return self.origin + uv[:, :1] * self.u_axis + uv[:, 1:2] * self.v_axis

    def corners(self) -> np.ndarray:
        """왼쪽 아래, 오른쪽 아래, 오른쪽 위, 왼쪽 위 (밖에서 봤을 때)."""
        return self.to_world([[0, 0], [self.width, 0], [self.width, self.height], [0, self.height]])

    def intersect(self, origins: np.ndarray, dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """광선 o + s·d 와 평면의 교점 (N,3), 광선 거리 s (N,) — 평행·뒤쪽이면 s = nan."""
        denom = dirs @ self.normal
        with np.errstate(divide="ignore", invalid="ignore"):
            s = -(origins @ self.normal + self.offset) / denom
        s = np.where((np.abs(denom) > 1e-9) & (s > 0), s, np.nan)
        return origins + s[:, None] * dirs, s

    def to_schema(self, texture: str) -> dict:
        r = lambda a: [round(float(x), 5) for x in a]  # noqa: E731
        return {"id": self.id, "normal": r(self.normal), "origin": r(self.origin), "u_axis": r(self.u_axis),
                "v_axis": r(self.v_axis), "width": round(self.width, 5), "height": round(self.height, 5),
                "texture": texture}


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a를 b로 보내는 최소 회전 (Rodrigues)."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        if c > 0:
            return np.eye(3)
        p = np.eye(3)[np.argmin(np.abs(a))]  # 180°: a에 수직인 축으로
        k = np.cross(a, p)
        k /= np.linalg.norm(k)
        return 2 * np.outer(k, k) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)
