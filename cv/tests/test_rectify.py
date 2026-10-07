import cv2
import numpy as np
import pytest
from facade_scene import FOCAL, H, W, make_scene

from facade_cv.rectify.homography import estimate_rectification


@pytest.mark.parametrize(("yaw", "pitch"), [(20, -10), (-30, 15), (0, 0)])
def test_rectification_recovers_front_view(yaw, pitch):
    img, _, facade, H_true = make_scene(yaw, pitch)
    rect = estimate_rectification(img, facade.mask)
    assert rect is not None
    # 정면 좌표 → 정면화 좌표가 닮음변환이면 정사각형이 축 정렬 정사각형으로
    sq = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)
    q = cv2.perspectiveTransform(sq.reshape(-1, 1, 2), rect.H @ H_true).reshape(-1, 2)
    edges = np.diff(np.vstack([q, q[:1]]), axis=0)
    ang = np.degrees(np.arctan2(edges[:, 1], edges[:, 0]))
    assert np.allclose(ang, [0, 90, 180, -90], atol=1.0) or np.allclose(ang, [0, 90, -180, -90], atol=1.0)
    lengths = np.linalg.norm(edges, axis=1)
    assert lengths.max() / lengths.min() < 1.03
    if yaw and pitch:  # 두 소실점이 모두 유한해야 초점거리를 추정할 수 있다
        assert rect.focal_estimated and rect.focal == pytest.approx(FOCAL, rel=0.05)


def test_rectification_fails_without_lines():
    assert estimate_rectification(np.full((H, W, 3), 128, np.uint8)) is None
    img, *_ = make_scene()
    tiny = np.zeros((H, W), bool)
    tiny[:20, :20] = True  # 직선이 없는 영역만 허용
    assert estimate_rectification(img, tiny) is None
