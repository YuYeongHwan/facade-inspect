import json

import cv2
import numpy as np
import pytest

from facade_cv.config import load_config
from facade_cv.io.extract import _motion, default_flight_id, extract_frames, iterate_video

FPS, W, H, SPEED = 30, 640, 480, 20
N = 90
BLUR_IDX = {15, 33, 51, 69}  # 10fps 샘플링(3프레임 간격)에 걸리는 index
DARK_IDX = 42


def _scene(width, seed=0):
    """무작위 사각형이 깔린 텍스처 장면."""
    rng = np.random.default_rng(seed)
    canvas = np.full((H, width, 3), 128, np.uint8)
    for _ in range(width // 4):
        x, y = rng.integers(0, width), rng.integers(0, H)
        w, h = rng.integers(10, 60, 2)
        cv2.rectangle(canvas, (int(x), int(y)), (int(x + w), int(y + h)), rng.integers(30, 220, 3).tolist(), -1)
    return canvas


def _pan_video(path, n, edit=lambda i, f: f, seed=0):
    """카메라가 장면을 가로로 SPEED px/frame으로 훑는 영상. edit(i, frame)으로 프레임 변형."""
    canvas = _scene(W + SPEED * n, seed)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(n):
        vw.write(edit(i, canvas[:, i * SPEED : i * SPEED + W].copy()))
    vw.release()
    return path


def _cfg(**blur):
    cfg = load_config("extract")
    cfg["blur"].update(blur)
    return cfg


@pytest.fixture
def pan_video(tmp_path):
    def edit(i, f):
        if i in BLUR_IDX:
            f = cv2.GaussianBlur(f, (31, 31), 0)
        if i == DARK_IDX:
            f[:] = 0
        return f

    return _pan_video(tmp_path / "pan.mp4", N, edit)


def test_iterate_video(pan_video):
    ts = [t for t, _ in iterate_video(pan_video)]
    assert len(ts) == N
    assert ts[1] == pytest.approx(1 / FPS)


def test_extract_filters(pan_video, tmp_path):
    out = tmp_path / "run" / "frames"
    m = extract_frames(pan_video, out, target_fps=10, debug=True)
    by_idx = {r["index"]: r for r in m["frames"]}

    assert len(m["frames"]) == N // 3
    assert all(by_idx[i]["reason"] == "blur" and not by_idx[i]["selected"] for i in BLUR_IDX)
    assert by_idx[DARK_IDX]["reason"] == "exposure"
    selected = [r for r in m["frames"] if r["selected"]]

    assert sorted(p.name for p in out.glob("*.jpg")) == sorted(r["file"] for r in selected)
    assert all(r["file"] == f"pan__frame_{r['index']:06d}.jpg" for r in selected)  # 기본 flight_id = 영상 파일명
    assert m["flight_id"] == "pan"
    assert json.loads((out / "frames.json").read_text())["num_selected"] == len(selected)
    assert (tmp_path / "run" / "debug" / "frames_grid.jpg").is_file()


def test_dedup_keeps_overlap(pan_video, tmp_path):
    cfg = _cfg(percentile=None)  # 절대 임계값만 — 중복 필터만 본다
    m = extract_frames(pan_video, tmp_path / "frames", target_fps=10, cfg=cfg)
    selected = [r for r in m["frames"] if r["selected"]]
    # 샘플 간 이동 ~9% → 2~3 샘플마다 선택, 선택 프레임 간 겹침 70~80%
    assert 5 <= len(selected) <= 15
    assert any(r["reason"] == "duplicate" for r in m["frames"])
    assert all(r["motion"] <= cfg["dedup"]["max_motion"] for r in selected)


def test_static_video_dedup(tmp_path):
    frame = _scene(W)
    vw = cv2.VideoWriter(str(tmp_path / "static.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for _ in range(30):
        vw.write(frame)
    vw.release()
    m = extract_frames(tmp_path / "static.mp4", tmp_path / "frames", target_fps=10)
    assert m["num_selected"] == 1


def test_blur_window_is_local(tmp_path):
    """앞 3초는 전체가 절대 임계값 미만(약한 블러). 전역 하위 20% 컷이면 그 구간 대부분이 잘리지만
    ±2초 윈도우 컷은 구간 안에서만 비교하므로 대부분 살아남아야 한다."""
    soft_until = 90  # 원본 frame index (3초)
    path = _pan_video(tmp_path / "soft.mp4", 300, lambda i, f: cv2.GaussianBlur(f, (5, 5), 0) if i < soft_until else f)
    m = extract_frames(path, tmp_path / "frames", target_fps=10)
    soft = [r for r in m["frames"] if r["index"] < soft_until]

    assert all(r["blur"] < 30 for r in soft)  # 절대 조건은 전부 참 → 상대 조건이 판정
    global_cut = np.percentile([r["blur"] for r in m["frames"]], 20)
    assert sum(r["blur"] < global_cut for r in soft) / len(soft) > 0.5  # 전역 컷이었다면 절반 넘게 제거
    assert 0 < sum(r["reason"] == "blur" for r in soft) / len(soft) < 0.35


def test_blur_requires_both_conditions(pan_video, tmp_path):
    m = extract_frames(pan_video, tmp_path / "frames", target_fps=10)
    # 선명한 프레임은 로컬 하위 20%여도 절대 임계값 이상이므로 유지 (OR였다면 제거됨)
    sharp = [r for r in m["frames"] if r["reason"] != "exposure" and r["index"] not in BLUR_IDX]
    assert any(r["blur"] < np.percentile([o["blur"] for o in sharp], 20) for r in sharp)
    assert not any(r["reason"] == "blur" for r in sharp)
    assert all(r["reason"] == "blur" for r in m["frames"] if r["index"] in BLUR_IDX)


@pytest.mark.parametrize(
    ("blur", "expect_removed"),
    [
        ({"threshold": 1e9, "percentile": None}, "all"),  # 절대 조건만: 전부 미만 → 전부 제거
        ({"threshold": 1e9}, "some"),  # 절대는 전부 참이어도 로컬 하위 20%만 제거
        ({"threshold": None}, "some"),  # 상대 조건만
        ({"threshold": 0}, "none"),  # 절대 조건이 거짓이면 로컬 하위여도 유지
    ],
)
def test_blur_condition_toggles(pan_video, tmp_path, blur, expect_removed):
    m = extract_frames(pan_video, tmp_path / "frames", target_fps=10, cfg=_cfg(**blur))
    candidates = [r for r in m["frames"] if r["reason"] != "exposure"]
    removed = sum(r["reason"] == "blur" for r in candidates)
    expected = {"all": len(candidates), "none": 0}.get(expect_removed)
    assert removed == expected if expected is not None else 0 < removed < len(candidates)


def test_blur_gap_rescues_least_blurry(tmp_path):
    """연속 블러 구간(이동 ~45%)이 max_motion을 깨면 그 구간에서 가장 덜 흐린 프레임을 살린다."""
    kernels = {30: 41, 33: 31, 36: 15, 39: 51, 42: 41}  # index → 블러 커널 (압축 후 점수 순서는 커널과 다를 수 있음)

    def edit(i, f):
        return cv2.GaussianBlur(f, (kernels[i], kernels[i]), 0) if i in kernels else f

    path = _pan_video(tmp_path / "gap.mp4", N, edit)
    m = extract_frames(path, tmp_path / "frames", target_fps=10, cfg=_cfg(percentile=None))
    by_idx = {r["index"]: r for r in m["frames"]}

    assert all(by_idx[i]["blur"] < 30 for i in kernels)  # 전부 절대 임계값으로 제거되는 블러
    best = max(kernels, key=lambda i: by_idx[i]["blur"])
    assert [r["index"] for r in m["frames"] if r["rescued"]] == [best]
    assert by_idx[best]["selected"] and by_idx[best]["reason"] is None
    assert all(not by_idx[i]["selected"] for i in kernels if i != best)


def test_rescue_without_gap_is_noop(pan_video, tmp_path):
    """단발 블러 프레임(이동 9%)은 겹침을 깨지 않으므로 살리지 않는다."""
    m = extract_frames(pan_video, tmp_path / "frames", target_fps=10)
    assert not any(r["rescued"] for r in m["frames"])


def test_motion_unknown_when_few_matches():
    orb, matcher = cv2.ORB_create(1000), cv2.BFMatcher(cv2.NORM_HAMMING)
    blank = np.full((H, W), 128, np.uint8)
    textured = cv2.cvtColor(_scene(W), cv2.COLOR_BGR2GRAY)
    f_blank, f_tex = orb.detectAndCompute(blank, None), orb.detectAndCompute(textured, None)
    assert f_blank[1] is None  # 특징점 없음
    assert _motion(f_tex, f_blank, (H, W), 15, matcher) is None
    other = cv2.cvtColor(_scene(W, seed=7), cv2.COLOR_BGR2GRAY)
    assert _motion(f_tex, orb.detectAndCompute(other, None), (H, W), 15, matcher) is None
    assert _motion(f_tex, f_tex, (H, W), 15, matcher) == pytest.approx(0, abs=1e-3)


def test_scene_cut_saves_unknown_motion(tmp_path):
    """장면 전환으로 매칭이 안 되면 이동량 불명 → 중복으로 버리지 않고 저장."""
    cut = 45
    other = _scene(W + SPEED * N, seed=7)

    def edit(i, f):
        return other[:, i * SPEED : i * SPEED + W].copy() if i >= cut else f

    path = _pan_video(tmp_path / "cut.mp4", N, edit)
    m = extract_frames(path, tmp_path / "frames", target_fps=10)
    first_after = next(r for r in m["frames"] if r["index"] >= cut)
    assert first_after["selected"] and first_after["motion"] is None
    assert any(r["reason"] == "duplicate" for r in m["frames"] if r["index"] > cut)  # 이후엔 다시 정상 비교


def test_flight_id_naming(pan_video, tmp_path):
    out = tmp_path / "frames"
    m = extract_frames(pan_video, out, target_fps=10, flight_id="2026-10-02_bldgA")
    names = {p.name for p in out.glob("*.jpg")}
    assert names and all(n.startswith("2026-10-02_bldgA__frame_") for n in names)
    # 재실행 시 flight_id가 바뀌어도 이전 프레임은 지워진다
    extract_frames(pan_video, out, target_fps=10, flight_id="other")
    assert all(p.name.startswith("other__") for p in out.glob("*.jpg"))
    for bad in ("a__b", "a/b", "a b"):
        with pytest.raises(ValueError, match="flight_id"):
            extract_frames(pan_video, out, target_fps=10, flight_id=bad)


@pytest.mark.parametrize(
    ("video", "expected"),
    [("DJI_0001.MP4", "DJI_0001"), ("runs/x/my flight__2.mp4", "my_flight_2"), ("__.mp4", "flight")],
)
def test_default_flight_id(video, expected):
    assert default_flight_id(video) == expected


def test_extract_cli_flight_id(pan_video, tmp_path):
    import subprocess
    import sys

    out = tmp_path / "frames"
    subprocess.run([sys.executable, "-m", "facade_cv.io.extract", "--video", str(pan_video), "--out", str(out),
                    "--fps", "10", "--flight-id", "F1"], check=True, capture_output=True)
    assert all(p.name.startswith("F1__frame_") for p in out.glob("*.jpg"))


@pytest.mark.skipif(not __import__("shutil").which("ffmpeg"), reason="ffmpeg 필요")
def test_rotation_metadata_applied(pan_video, tmp_path):
    """아이폰 세로 촬영 = 가로 프레임 + 회전 메타데이터. 추출 결과가 눕지 않아야 한다."""
    import subprocess

    rotated = tmp_path / "rot.mov"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-display_rotation", "90", "-i", str(pan_video), "-c", "copy",
                    str(rotated)], check=True)
    _, frame = next(iterate_video(rotated))
    assert frame.shape[:2] == (W, H)
    m = extract_frames(rotated, tmp_path / "frames", target_fps=10)
    assert cv2.imread(str(tmp_path / "frames" / m["frames"][0]["file"])).shape[:2] == (W, H)
