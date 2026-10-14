"""recon3d 기하 검증 — 합성 박스 건물 (12 × 10 × 8, 상대 단위), 보이는 면 1/2/4개."""

import json
import shutil
import subprocess

import numpy as np
import pytest

from facade_cv.config import CONFIG_DIR, load_config
from facade_cv.recon3d.export import export_result
from facade_cv.recon3d.geometry import UP, CameraView, FacadeFrame, rotation_between
from facade_cv.recon3d.mesh import build_mesh
from facade_cv.recon3d.ortho import build_textures
from facade_cv.recon3d.planes import extract_facades, snap_orthogonal
from facade_cv.recon3d.run import align_scene
from facade_cv.recon3d.sfm import SfmResult
from facade_cv.recon3d.up_axis import path_straightness
from facade_cv.recon3d.windows3d import WindowObservation, build_windows, lift_observations, vertical_tilt
from facade_cv.schema import InspectionResult

W, H, D = 12.0, 10.0, 8.0
YAW = np.radians(20)  # 건물을 수평으로 돌려 축 정렬이 우연히 맞지 않게
ROT = np.array([[np.cos(YAW), 0, np.sin(YAW)], [0, 1, 0], [-np.sin(YAW), 0, np.cos(YAW)]])
# 면 k: (밖 법선, 밖에서 봤을 때 왼쪽 아래 코너, 오른쪽 방향, 폭) — 회전 전
SIDES = [
    ((0, 0, 1), (0, 0, D), (1, 0, 0), W),
    ((1, 0, 0), (W, 0, D), (0, 0, -1), D),
    ((0, 0, -1), (W, 0, 0), (-1, 0, 0), W),
    ((-1, 0, 0), (0, 0, 0), (0, 0, 1), D),
]
CENTER = np.array([W / 2, H / 2, D / 2])


def true_facade(k: int) -> FacadeFrame:
    n, o, _, w = SIDES[k]
    return FacadeFrame(f"T{k}", ROT @ np.array(n, float), ROT @ np.array(o, float), w, H)


def box_points(visible: list[int], n: int = 4000, noise: float = 0.01, seed: int = 0) -> np.ndarray:
    """보이는 면 위 점 + 지면 + 나무 덩어리 + 이상치."""
    return labeled_points(visible, n, noise, seed)[0]


def labeled_points(visible: list[int], n: int = 4000, noise: float = 0.01, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(points, label) — label = 면 번호 k, 외벽이 아니면 -1."""
    rng = np.random.default_rng(seed)
    pts = []
    for k in visible:
        f = true_facade(k)
        uv = rng.uniform([0, 0], [f.width, H], (n, 2))
        pts.append(f.to_world(uv) + rng.normal(0, noise, (n, 3)))
    ground = rng.uniform([-6, 0, -6], [W + 6, 0, D + 6], (n // 2, 3)) @ ROT.T
    tree = rng.normal(ROT @ np.array([W / 2, 1.5, D + 4]), 0.6, (n // 8, 3))
    outliers = rng.uniform(-20, 30, (n // 20, 3))
    rest = np.vstack([ground, tree, outliers])
    labels = np.concatenate([np.full(n, k) for k in visible] + [np.full(len(rest), -1)])
    return np.vstack(pts + [rest]), labels


def visibility(pts: np.ndarray, labels: np.ndarray, cams: list[CameraView]) -> list[np.ndarray]:
    """SfM track 흉내: 이미지 안에 투영되고, 외벽 점이면 그 외벽 바깥쪽 카메라만 (건물 너머는 가려짐)."""
    vis: list[list[int]] = [[] for _ in pts]
    for ci, cam in enumerate(cams):
        ok = cam.project(pts)[1]
        for k in np.unique(labels[labels >= 0]):
            f = true_facade(int(k))
            if (cam.center - f.origin) @ f.normal <= 0:
                ok &= labels != k
        for i in np.flatnonzero(ok):
            vis[i].append(ci)
    return [np.array(v, int) for v in vis]


def scene(visible: list[int], extra: np.ndarray | None = None):
    """(points, cameras, visibility). extra: 추가 점 (가림 없이 투영으로만 관측)."""
    pts, lab = labeled_points(visible)
    if extra is not None:
        pts, lab = np.vstack([pts, extra]), np.concatenate([lab, np.full(len(extra), -1)])
    cams = cameras_for(visible)
    return pts, cams, visibility(pts, lab, cams)


def cameras_for(visible: list[int], per_side: int = 12, pitch_up: float = 3.0) -> list[CameraView]:
    """보이는 면마다 10 떨어진 곳을 따라 걸으며 건물 위쪽을 올려다보는 카메라."""
    cams = []
    for k in visible:
        f = true_facade(k)
        for i, a in enumerate(np.linspace(0.1, 0.9, per_side)):
            base = f.origin + f.u_axis * f.width * a
            eye = base + f.normal * 10 + UP * 1.5 + f.u_axis * (i % 3 - 1) * 0.5
            cams.append(CameraView.look_at(f"{k}_{i:02d}.jpg", eye, base + UP * (1.5 + pitch_up), f=500,
                                           width=640, height=480))
    return cams


def _cfg(section: str) -> dict:
    return load_config("recon")[section]


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


@pytest.mark.parametrize("visible, kind", [([0], "slab"), ([0, 1], "L"), ([0, 1, 2, 3], "box")])
def test_planes_and_mesh(visible, kind):
    pts, cams, vis = scene(visible)
    fr = extract_facades(pts, cams, _cfg("planes"), vis)
    assert len(fr.facades) == len(visible)
    truth = [true_facade(k) for k in visible]
    for f in fr.facades:  # 각 검출 면이 정답 면 하나와 법선 2° 이내 · 평면 거리 0.1 이내
        t = min(truth, key=lambda t: _angle_deg(f.normal, t.normal))
        assert _angle_deg(f.normal, t.normal) < 2
        assert abs(f.offset - t.offset) < 0.1
    assert abs(fr.ground) < 0.15 and abs(fr.top - H) < 0.3
    if len(visible) > 1:  # 직교 정규화
        for a in fr.facades:
            for b in fr.facades:
                assert min(abs(_angle_deg(a.normal, b.normal) - x) for x in (0, 90, 180)) < 1e-6

    body, facades, got = build_mesh(fr.facades, _cfg("mesh"))
    assert got == kind
    for f in facades:  # 모든 면이 정답 범위를 덮고 (조금 넘쳐도 됨) 높이가 맞다
        t = min(truth, key=lambda t: _angle_deg(f.normal, t.normal))
        assert abs(f.width - t.width) < 0.5 and abs(f.height - H) < 0.4
        assert np.allclose(f.u_axis, np.cross(UP, f.normal)) and np.allclose(f.v_axis, UP)
        assert np.linalg.norm((f.origin - t.origin) * [1, 0, 1]) < 0.5
    if kind == "box":
        ext = sorted(body.bounding_box_oriented.primitive.extents)
        assert np.allclose(ext, sorted([W, H, D]), atol=0.5)
        assert body.is_watertight


def test_extra_plane_reported_not_failed():
    """정면에 45° 날개(작은 수직 벽)가 붙어도 박스는 4면 그대로, 날개는 extra로 보고."""
    rng = np.random.default_rng(1)
    wing_n = ROT @ np.array([np.sin(np.radians(45)), 0, np.cos(np.radians(45))])
    wing_o = ROT @ np.array([W + 3, 0, D + 3])
    wing = FacadeFrame("w", wing_n, wing_o, 4, H)
    pts, cams, vis = scene([0, 1, 2, 3], wing.to_world(rng.uniform([0, 0], [4, H], (2500, 2))))
    fr = extract_facades(pts, cams, _cfg("planes"), vis)
    assert len(fr.facades) == 4
    assert any(e["reason"] == "off_grid_angle" for e in fr.extra)
    assert any("extra_planes" in w for w in fr.warnings)


def test_snap_orthogonal():
    a = np.radians([10, 101, 189, 280, 55])
    normals = [np.array([np.cos(t), 0, np.sin(t)]) for t in a]
    alpha, ks, offs = snap_orthogonal(normals, [5, 4, 3, 2, 1], 25)
    assert abs(np.degrees(alpha) - 10) < 2
    assert ks[:4] == [0, 1, 2, 3] and ks[4] is None


def test_align_scene_recovers_gravity():
    """장면 전체를 기울이고 카메라가 올려다봐도 (업벡터 평균이 치우침) 외벽 법선으로 Y-up을 되찾는다."""
    visible = [0, 1]
    tilt = rotation_between(UP, np.array([0.3, 1.0, -0.2]))
    cams = [c.rotated(tilt) for c in cameras_for(visible, pitch_up=6.0)]
    pts, _, vis = scene(visible)
    sfm = SfmResult(cams, pts @ tilt.T, np.zeros((0, 3), np.uint8), len(cams), [], 1, visibility=vis)
    points, cameras, R = align_scene(sfm, _cfg("planes"))
    assert _angle_deg(R @ tilt @ UP, UP) < 1.0
    fr = extract_facades(points, cameras, _cfg("planes"), sfm.visibility)
    assert len(fr.facades) == 2


def test_path_straightness():
    straight = [CameraView.look_at(str(i), [i, 1.5, 10], [i, 1.5, 0]) for i in range(10)]
    arc = [CameraView.look_at(str(i), [10 * np.cos(t), 1.5, 10 * np.sin(t)], [0, 1.5, 0])
           for i, t in enumerate(np.linspace(0, np.pi, 10))]
    assert path_straightness(straight) < 0.05 and path_straightness(arc) > 0.3  # 반원 ≈ 0.48


def window_grid(f: FacadeFrame, rows: int = 3, win=(1.6, 1.8)) -> list[np.ndarray]:
    """facade 위 rows × cols 창문 [u0, v0, u1, v1], 아래층부터."""
    cols = int(f.width // 3)
    out = []
    for r in range(rows):
        for c in range(cols):
            uc, vc = (c + 0.5) * f.width / cols, 2.0 + r * 3.0
            out.append(np.array([uc - win[0] / 2, vc - win[1] / 2, uc + win[0] / 2, vc + win[1] / 2]))
    return out


def observe(cams: list[CameraView], facades: list[FacadeFrame], noise: float = 1.0, seed: int = 0):
    """카메라마다 4코너가 모두 보이는 창문을 TL TR BR BL 픽셀로 (+ 잡음)."""
    rng = np.random.default_rng(seed)
    obs = []
    for cam in cams:
        for f in facades:
            if (cam.center - f.origin) @ f.normal <= 0:
                continue
            for u0, v0, u1, v1 in window_grid(f):
                X = f.to_world([[u0, v1], [u1, v1], [u1, v0], [u0, v0]])
                px, ok = cam.project(X)
                if ok.all():
                    obs.append(WindowObservation(cam.name, px + rng.normal(0, noise, px.shape), 0.9))
    return obs


@pytest.mark.parametrize("visible", [[0], [0, 1], [0, 1, 2, 3]])
def test_windows3d(visible):
    facades = [true_facade(k) for k in visible]
    for i, f in enumerate(facades, 1):
        f.id = f"F{i}"
    cams = cameras_for(visible)
    windows, stats = build_windows(observe(cams, facades), cams, facades, _cfg("windows3d"))
    truth = {f.id: window_grid(f) for f in facades}
    assert stats["dropped_same_cell"] == 0
    assert len(windows) == sum(len(v) for v in truth.values())
    for w in windows:
        f = next(f for f in facades if f.id == w.facade_id)
        t = min(truth[f.id], key=lambda t: np.abs(t - w.rect).sum())
        assert np.abs(t - w.rect).max() < 0.08  # 사각형 오차 (창문 1.6 × 1.8)
        r_true = int(round((t[1] + t[3]) / 2 - 2.0) / 3) + 1  # 아래층 = 1
        c_true = int((t[0] + t[2]) / 2 // (f.width / int(f.width // 3))) + 1
        assert (w.row, w.col) == (r_true, c_true)
        assert w.id == f"{f.id}-R{r_true:02d}-C{c_true:02d}"
        assert w.num_views >= 2
        assert np.allclose(w.corners_3d(f)[0], f.to_world([[w.rect[0], w.rect[1]]])[0])


def test_windows3d_track_ids_group_without_clustering():
    """T5 tracklet 입력: track_id가 있으면 거리와 상관없이 같은 id끼리 묶는다."""
    f = true_facade(0)
    f.id = "F1"
    cams = cameras_for([0])
    obs = observe(cams, [f])
    for o in obs:
        o.track_id = 7  # 모두 같은 tracklet이라고 하면 창문 하나
    windows, _ = build_windows(obs, cams, [f], _cfg("windows3d"))
    assert len(windows) == 1


def render(cam: CameraView, f: FacadeFrame, pattern) -> np.ndarray:
    """합성 이미지: 픽셀 광선이 facade에 맞으면 pattern(u, v) 색, 아니면 검정."""
    ys, xs = np.mgrid[0:cam.height, 0:cam.width]
    px = np.stack([xs.ravel(), ys.ravel()], 1).astype(float)
    X, s = f.intersect(np.repeat(cam.center[None], len(px), 0), cam.rays(px))
    uv = f.to_uv(np.nan_to_num(X))
    inside = ~np.isnan(s) & (uv[:, 0] >= 0) & (uv[:, 0] <= f.width) & (uv[:, 1] >= 0) & (uv[:, 1] <= f.height)
    img = np.zeros((len(px), 3), np.uint8)
    img[inside] = pattern(uv[inside])
    return img.reshape(cam.height, cam.width, 3)


def test_ortho_texture_recovers_pattern():
    f = true_facade(0)
    f.id = "F1"

    def pattern(uv):  # 2 × 2 단위 체커보드
        on = ((uv[:, 0] // 2 + uv[:, 1] // 2) % 2).astype(bool)
        return np.where(on[:, None], [230, 230, 230], [30, 30, 30]).astype(np.uint8)

    cams = [CameraView.look_at(f"{i}.jpg", f.origin + f.u_axis * x + f.normal * 14 + UP * 5, f.origin + f.u_axis * x + UP * 5,
                               f=420, width=480, height=360) for i, x in enumerate([3, 6, 9])]
    images = {c.name: render(c, f, pattern) for c in cams}
    cfg = _cfg("ortho") | {"max_side_px": 240}
    tex, ppu, nviews = build_textures([f], cams, images.__getitem__, cfg)["F1"]
    hh, ww = tex.shape[:2]
    ys, xs = np.mgrid[0:hh, 0:ww]
    uv = np.stack([(xs.ravel() + 0.5) / ppu, f.height - (ys.ravel() + 0.5) / ppu], 1)
    expect = pattern(uv).reshape(hh, ww, 3).astype(int)
    seen = nviews > 0
    assert seen.mean() > 0.6
    # 체커 경계 1~2 texel은 보간으로 섞이니 중앙값 오차로 본다
    assert np.median(np.abs(tex.astype(int) - expect)[seen]) < 5


@pytest.mark.skipif(not shutil.which("node"), reason="node 필요")
def test_export_unanalyzed_passes_pydantic_and_zod(tmp_path):
    visible = [0, 1]
    pts, cams, vis = scene(visible)
    fr = extract_facades(pts, cams, _cfg("planes"), vis)
    body, facades, _ = build_mesh(fr.facades, _cfg("mesh"))
    cams = cameras_for(visible)
    windows, _ = build_windows(observe(cams, facades), cams, facades, _cfg("windows3d"))
    textures = {f.id: (np.full((100, int(100 * f.width / f.height), 3), 128, np.uint8), 100 / f.height) for f in facades}
    status = {"mode": "3d", "num_facades": 2, "registered_ratio": 1.0, "warnings": []}
    r = export_result(tmp_path, "insp_syn", {"video": "v.mp4", "fps": 30, "frames_used": len(cams)}, body, facades,
                      textures, windows, status)
    assert r.summary.level_counts["unanalyzed"] == r.summary.total == len(windows) > 0
    assert all(w.cleaning_level is None and w.cleaning_label == "미분석" for w in r.windows)
    for p in ["building.glb", *(f.texture for f in r.building.facades), *(w.images.crop for w in r.windows)]:
        assert (tmp_path / p).is_file()
    InspectionResult.model_validate_json((tmp_path / "result.json").read_text())
    web = CONFIG_DIR.parent / "web"
    out = subprocess.run(["node", "scripts/validate-result.mjs", str(tmp_path / "result.json")], cwd=web,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    loaded = json.loads((tmp_path / "result.json").read_text())
    assert loaded["meta"]["recon_status"]["mode"] == "3d"


def test_snap_largest_plane_sets_grid():
    """직각이 아닌 (61°) 둘째 면이 있어도 큰 면은 원래 방향 그대로, 둘째 면은 격자 밖."""
    a = np.radians([15, 15 + 61])
    alpha, ks, offs = snap_orthogonal([np.array([np.cos(t), 0, np.sin(t)]) for t in a], [947, 843], 25)
    assert abs(np.degrees(alpha) - 15) < 1e-6
    assert ks[0] == 0 and ks[1] is None and abs(abs(offs[1]) - 29) < 1e-6


def test_parallel_recess_faces_same_way():
    """외벽 + 1층 안쪽으로 들어간 출입구 벽 (평행, 2 뒤): 둘 다 카메라 쪽을 향하고, 박스 반대편 면으로 오인하지 않는다."""
    rng = np.random.default_rng(2)
    front = true_facade(0)
    recess = FacadeFrame("r", front.normal, front.origin - front.normal * 2 + front.u_axis * 3, 5, 3)
    pts, cams, vis = scene([0], recess.to_world(rng.uniform([0, 0], [5, 3], (2500, 2))))
    fr = extract_facades(pts, cams, _cfg("planes"), vis)
    assert len(fr.facades) == 1 and _angle_deg(fr.facades[0].normal, front.normal) < 2
    assert [e["reason"] for e in fr.extra] == ["parallel_duplicate"]


def test_vertical_tilt_from_windows():
    """업벡터가 면 안에서 3° 틀리면 창문 세로 변이 -3° 기운 것으로 나온다."""
    f = true_facade(0)
    f.id = "F1"
    cams = cameras_for([0])
    obs = observe(cams, [f], noise=0.5)
    th = np.radians(3)
    tilted_up = np.cos(th) * UP + np.sin(th) * f.u_axis  # 틀린 업벡터
    R = rotation_between(tilted_up, UP)
    g = FacadeFrame("F1", R @ f.normal, R @ f.origin, f.width, f.height)
    lifted = lift_observations(obs, {c.name: c.rotated(R) for c in cams}, [g], 0.1)
    assert abs(np.degrees(vertical_tilt(lifted)) + 3) < 0.3


def test_degrade_unique_ids_bottom_row_first(tmp_path):
    """degrade: T4 격자 (row 0 = 위)를 아래층 = 1로 뒤집고, 한 칸에 둘이면 하나만 남긴다."""
    import cv2

    from facade_cv.recon3d.degrade import degrade_facade

    img = np.full((480, 640, 3), 180, np.uint8)
    cv2.imwrite(str(tmp_path / "a.jpg"), img)
    win = lambda x, y, r, c: {"corners": [[x, y], [x + 60, y], [x + 60, y + 80], [x, y + 80]], "conf": 0.9,  # noqa: E731
                              "row": r, "col": c, "flags": []}
    windows = [win(100, 60, 0, 0), win(300, 60, 0, 1), win(100, 300, 1, 0), win(104, 304, 1, 0)]  # 마지막은 중복 칸
    det = {"images": [{"image": "a.jpg", "counts": {"final": 4}, "windows": windows}]}
    facade, tex, ppu, ws, ref = degrade_facade(det, tmp_path, _cfg("degrade"))
    assert ref == "a.jpg" and sorted(w.id for w in ws) == ["F1-R01-C01", "F1-R02-C01", "F1-R02-C02"]
    low = next(w for w in ws if w.id == "F1-R01-C01")
    assert low.rect[1] < next(w for w in ws if w.id == "F1-R02-C01").rect[1]  # 아래층 v가 더 작다


def test_project_rejects_points_outside_fov_despite_distortion_foldover():
    """k1 < 0 왜곡: 화각 훨씬 밖의 점이 다항식이 접혀 이미지 안으로 투영돼도 '안 보임'이어야 한다."""
    f, w, h, k1 = 500.0, 640, 480, -0.3

    def img_from_cam(Xc):
        x, y = Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2]
        r = 1 + k1 * (x * x + y * y)
        return np.stack([f * x * r + w / 2, f * y * r + h / 2], 1)

    def ray_from_img(px):  # 화각 안에서는 반복으로 역변환
        x, y = (px[:, 0] - w / 2) / f, (px[:, 1] - h / 2) / f
        xu, yu = x.copy(), y.copy()
        for _ in range(50):
            r = 1 + k1 * (xu * xu + yu * yu)
            xu, yu = x / r, y / r
        d = np.stack([xu, yu, np.ones_like(xu)], 1)
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    cam = CameraView("c", np.eye(3), np.zeros(3), w, h, img_from_cam, ray_from_img)
    far = np.array([[2.0, 0.0, 1.0]])  # tan 2.0: x_d = 2.0·(1 - 0.3·4) = -0.4 → x = 120px, 이미지 안으로 접힘
    px = img_from_cam(far)
    assert 0 <= px[0, 0] <= w  # 접힌 투영이 실제로 이미지 안
    assert not cam.project(far)[1][0]
    assert cam.project(np.array([[0.2, 0.1, 1.0]]))[1][0]


def test_relief_follows_bowed_wall():
    """직선 경로 촬영의 휜 벽: 실제 벽이 가운데서 0.8 앞으로 휨. 평평한 평면과 교차하면 보는 각마다 u가 달라 창문이
    갈라지고, relief(점에서 잰 깊이 프로필)를 쓰면 같은 창문으로 모이고 평면 좌표도 정답과 맞는다."""
    from dataclasses import replace

    from facade_cv.recon3d.planes import wall_relief

    flat = true_facade(0)
    flat.id = "F1"
    bow = lambda u: 0.8 * (1 - (2 * u / flat.width - 1) ** 2)  # noqa: E731
    rng = np.random.default_rng(3)
    uv = rng.uniform([0, 0], [flat.width, flat.height], (8000, 2))
    wall = flat.to_world(uv) + bow(uv[:, :1]) * flat.normal
    curved = replace(flat, relief=wall_relief(wall @ flat.u_axis, (wall - flat.origin) @ flat.normal, 24))
    err = np.abs(curved.depth_at(wall) - bow(uv[:, 0]))
    assert np.median(err) < 0.02 and err.max() < 0.15  # 양 끝은 마지막 구간 값으로 고정돼 조금 크다

    # 관측: 휜 실제 벽 위 창문을 투영
    cams = cameras_for([0], per_side=16)
    obs = []
    for cam in cams:
        for u0, v0, u1, v1 in window_grid(flat):
            q = np.array([[u0, v1], [u1, v1], [u1, v0], [u0, v0]])
            X = flat.to_world(q) + bow(q[:, :1]) * flat.normal
            px, ok = cam.project(X)
            if ok.all():
                obs.append(WindowObservation(cam.name, px + rng.normal(0, 0.3, px.shape), 0.9))
    cfg = _cfg("windows3d")
    truth = window_grid(flat)

    def spread(ws):
        return max(np.abs(np.array([x.rect for x in w.observations]) - w.rect).max() for w in ws)

    plain, _ = build_windows(obs, cams, [flat], cfg)
    relief, _ = build_windows(obs, cams, [curved], cfg)
    assert len(relief) == len(truth)
    assert spread(relief) < 0.1 < spread(plain)
    for w in relief:
        assert min(np.abs(t - w.rect).max() for t in truth) < 0.1


def test_grid_labels_not_chained_by_strays():
    """두 층 (v 2, 5) 사이에 오검출이 2.9, 3.6, 4.2에 떠 있어도 층이 하나로 이어지지 않는다."""
    from facade_cv.recon3d.windows3d import grid_labels

    v = np.array([2.0, 2.1, 1.9, 2.05, 5.0, 5.1, 4.9, 5.05, 2.9, 3.6, 4.2])
    lab = grid_labels(v, 1.8, _cfg("windows3d"))
    assert len(set(lab[:4])) == 1 and len(set(lab[4:8])) == 1 and lab[0] < lab[4]
