"""3D 복원 전체: T1 프레임 + T4 검출 → SfM → Y-up 정렬 → 외벽 평면 → 메시·정사영 텍스처·창문 → result.json.

python -m facade_cv.recon3d.run --frames runs/<id>/frames --detections runs/<id>/detect --out runs/<id> --debug

출력 runs/<id>/
  result.json, building.glb, facades/<F>.jpg, windows/<id>/crop.jpg
  recon/colmap/      COLMAP database·sparse 모델
  recon/report.json  등록률·미등록 프레임·평면·extra 평면·창문 통계·경고·단계별 시간
  recon/window_obs.json  창문별 원본 프레임 관측 (프레임·4코너·가림) — 오염 분석(T7) 입력
--reuse-sfm: recon/colmap의 SfM 모델을 다시 읽어 SfM(가장 오래 걸림)을 건너뛴다
  debug/recon/       --debug: points_planes.ply (평면별 색), cameras.ply (궤적, 파랑→빨강 = 시간순),
                     top_view.jpg (위에서 본 포인트·평면·외벽·카메라 궤적), <F>_windows.jpg (텍스처 위 창문 ID)
SfM 실패·등록률 부족·외벽 평면 0개면 degrade 모드 (가상 평면 1개 + T4 2D row/col). meta.recon_status에 기록.
"""

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh
import typer

from facade_cv.config import CONFIG_DIR, load_config
from facade_cv.recon3d.degrade import degrade_facade
from facade_cv.recon3d.export import export_result
from facade_cv.recon3d.geometry import UP, CameraView, FacadeFrame
from facade_cv.recon3d.mesh import build_mesh
from facade_cv.recon3d.ortho import build_textures, rect_to_px
from facade_cv.recon3d.planes import FacadeResult, detect_planes, extract_facades, remove_outliers, scene_scale
from facade_cv.recon3d.sfm import SfmResult, load_sfm, run_sfm
from facade_cv.recon3d.up_axis import align_rotation, estimate_up, path_straightness, refine_up
from facade_cv.recon3d.windows3d import (
    Window3D,
    build_windows,
    lift_observations,
    observations_from_detections,
    vertical_tilt,
)

FACADE_COLORS = [(60, 76, 231), (113, 204, 46), (219, 152, 52), (182, 89, 155)]  # BGR
EXTRA_COLOR = (0, 165, 255)


def align_scene(sfm: SfmResult, pcfg: dict) -> tuple[np.ndarray, list[CameraView], np.ndarray]:
    """카메라 업벡터 평균 → 대략 수직 평면 법선으로 보정 → Y-up. (points, cameras, 월드 회전)."""
    R = align_rotation(estimate_up(sfm.cameras))
    pts = sfm.points @ R.T
    P = pts[remove_outliers(pts, pcfg["outlier_neighbors"], pcfg["outlier_std_ratio"])]
    scene = scene_scale(P, [c.rotated(R) for c in sfm.cameras])
    planes = detect_planes(P, scene * pcfg["distance_ratio"], pcfg["ransac_iters"],
                           max(int(len(P) * pcfg["min_inlier_ratio"]), 10), pcfg["max_planes"])
    tol = np.sin(np.radians(pcfg["up_refine_tol_deg"]))
    rough = [p for p in planes if abs(p.normal @ UP) < tol]
    up = refine_up(UP.copy(), [p.normal for p in rough], [len(p.inliers) for p in rough], pcfg["up_refine_iters"])
    R = align_rotation(up) @ R
    return sfm.points @ R.T, [c.rotated(R) for c in sfm.cameras], R


def _video_meta(frames: Path, num_used: int) -> dict:
    """frames.json(io.extract)의 원본 영상 경로·fps."""
    m = json.loads((frames / "frames.json").read_text()) if (frames / "frames.json").is_file() else {}
    video = m.get("video", str(frames))
    path = Path(video) if Path(video).is_absolute() else CONFIG_DIR.parent / video
    fps = cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FPS) if path.is_file() else 0
    return {"video": video, "fps": round(fps or m.get("target_fps") or 1.0, 3), "frames_used": num_used}


def draw_top_view(points: np.ndarray, fr: FacadeResult | None, facades: list[FacadeFrame],
                  cameras: list[CameraView], size: int = 1400) -> np.ndarray:
    """위에서 본 (x, z): 포인트(회색, 외벽 inlier는 외벽 색, extra 평면은 주황) + 최종 외벽 선분·ID + 카메라 궤적."""
    cams = np.array([c.center for c in cameras]) if cameras else np.empty((0, 3))
    ref = points[fr.kept] if fr is not None and len(points) else points
    xz = np.vstack([ref[:, [0, 2]], cams[:, [0, 2]]]) if len(ref) else cams[:, [0, 2]]
    lo, hi = np.percentile(xz, [1, 99], axis=0)
    pad = max(hi - lo) * 0.08  # 긴 건물도 짧은 축에 여백 (외벽 법선 화살표·ID가 잘리지 않게)
    lo, hi = lo - pad, hi + pad
    s = (size - 40) / max(hi - lo)
    img = np.full((int((hi - lo)[1] * s) + 40, int((hi - lo)[0] * s) + 40, 3), 20, np.uint8)

    def px(p: np.ndarray) -> np.ndarray:
        return np.round((np.atleast_2d(p)[:, [0, 2]] - lo) * s + 20).astype(np.int32)

    def dots(idx: np.ndarray, color: tuple[int, int, int]) -> None:
        q = px(points[idx])
        ok = (q[:, 0] >= 0) & (q[:, 1] >= 0) & (q[:, 0] < img.shape[1]) & (q[:, 1] < img.shape[0])
        img[q[ok, 1], q[ok, 0]] = color

    if fr is not None:
        dots(np.flatnonzero(fr.kept), (90, 90, 90))
        for e in fr.extra:
            dots(e["points"], EXTRA_COLOR)
        for i, f in enumerate(facades):
            dots(f.inliers, FACADE_COLORS[i % 4])
    if len(cams) > 1:
        cv2.polylines(img, [px(cams)], False, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(img, tuple(px(cams[0])[0]), 6, (255, 128, 0), -1)
    for i, f in enumerate(facades):
        a, b = px(f.origin)[0], px(f.origin + f.u_axis * f.width)[0]
        cv2.line(img, tuple(a), tuple(b), FACADE_COLORS[i % 4], 4, cv2.LINE_AA)
        m = (a + b) // 2
        tip = px(f.origin + f.u_axis * f.width / 2 + f.normal * f.width * 0.08)[0]
        cv2.arrowedLine(img, tuple(m), tuple(tip), FACADE_COLORS[i % 4], 2, cv2.LINE_AA)
        cv2.putText(img, f.id, tuple(tip + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    legend = ["top view (x right, z down) | grey: points | colors: facades F1..F4 (arrow = outward)",
              "orange: extra vertical planes (not in box) | white: camera path (blue dot = start)"]
    for i, t in enumerate(legend):
        cv2.putText(img, t, (10, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    return img


def draw_facade_windows(tex: np.ndarray, f: FacadeFrame, ppu: float, windows: list[Window3D]) -> np.ndarray:
    out = tex.copy()
    for w in windows:
        if w.facade_id != f.id:
            continue
        x0, y0, x1, y1 = rect_to_px(tuple(w.rect), f.height, ppu)
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(out, f"R{w.row}C{w.col} ({w.num_views})", (x0 + 3, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA)
    return out


def write_debug(debug: Path, points: np.ndarray, colors: np.ndarray, fr: FacadeResult | None,
                facades: list[FacadeFrame], cameras: list[CameraView], textures: dict, windows: list[Window3D]) -> None:
    debug.mkdir(parents=True, exist_ok=True)
    if len(points):
        c = (colors // 3).astype(np.uint8)  # 평면이 아닌 포인트는 어둡게
        if fr is not None:
            for e in fr.extra:
                c[e["points"]] = EXTRA_COLOR[::-1]
            for i, f in enumerate(facades):
                c[f.inliers] = FACADE_COLORS[i % 4][::-1]
        trimesh.PointCloud(points, np.hstack([c, np.full((len(c), 1), 255, np.uint8)])).export(debug / "points_planes.ply")
    if cameras:
        centers = np.array([cm.center for cm in cameras])
        t = np.linspace(0, 1, len(centers))[:, None]
        rgb = (np.hstack([t, np.zeros_like(t), 1 - t, np.ones_like(t)]) * 255).astype(np.uint8)
        trimesh.PointCloud(centers, rgb).export(debug / "cameras.ply")
    if len(points) or cameras:
        cv2.imwrite(str(debug / "top_view.jpg"), draw_top_view(points, fr, facades, cameras))
    for f in facades:
        tex, ppu = textures[f.id][:2]
        cv2.imwrite(str(debug / f"{f.id}_windows.jpg"), draw_facade_windows(tex, f, ppu, windows))


def write_window_obs(out: Path, frames: Path, windows: list[Window3D], det: dict) -> None:
    """recon/window_obs.json — 창문 ID별 원본 프레임 관측 (프레임 이름, 4코너, 가림 polygon). 오염 분석(T7)이
    원본 해상도·다시점으로 창문을 다시 정렬할 때 쓴다. frames는 out 기준 상대경로."""
    occ = {(img["image"], tuple(np.round(w["corners"], 1).ravel())): w.get("occlusion", [])
           for img in det["images"] for w in img["windows"]}
    obs = {w.id: [{"frame": o.obs.frame, "corners": np.round(o.obs.corners, 2).tolist(),
                   "occlusion": occ.get((o.obs.frame, tuple(np.round(o.obs.corners, 1).ravel())), [])}
                  for o in w.observations] for w in windows}
    (out / "recon").mkdir(parents=True, exist_ok=True)
    (out / "recon" / "window_obs.json").write_text(json.dumps(
        {"frames": os.path.relpath(Path(frames).resolve(), out.resolve()), "windows": obs}, ensure_ascii=False),
        encoding="utf-8")


def run_recon(frames: str | Path, detections: str | Path, out: str | Path, debug: bool = False,
              cfg: dict | None = None, reuse_sfm: bool = False) -> dict:
    """report (recon/report.json 내용)를 반환. reuse_sfm: out/recon/colmap의 기존 SfM 모델을 다시 읽는다."""
    cfg = cfg or load_config("recon")
    frames, out = Path(frames), Path(out)
    det_path = Path(detections)
    det = json.loads((det_path / "detections.json" if det_path.is_dir() else det_path).read_text())
    work = out / "recon"
    work.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    t = time.monotonic()

    def lap(name: str) -> None:
        nonlocal t
        timings[name] = round(time.monotonic() - t, 1)
        t = time.monotonic()

    warnings: list[str] = []
    report: dict = {"frames": str(frames), "detections": str(det_path)}
    sfm: SfmResult | None = None
    try:
        sfm = load_sfm(frames, work / "colmap") if reuse_sfm else run_sfm(frames, work / "colmap", cfg["sfm"])
        report["sfm"] = sfm.report()
        if sfm.num_models > 1 and sfm.unregistered:
            warnings.append(f"sfm_split: {sfm.num_models} separate models, used the largest "
                            f"({len(sfm.unregistered)} frames only in the others)")
    except Exception as e:  # noqa: BLE001 — SfM 실패는 degrade로 넘긴다
        warnings.append(f"sfm_failed: {type(e).__name__}: {e}")
    lap("sfm")

    sc = cfg["sfm"]
    degrade_reason = None
    if sfm is None:
        degrade_reason = "sfm_failed"
    elif len(sfm.cameras) < sc["min_registered"] or sfm.registered_ratio < sc["min_registered_ratio"]:
        degrade_reason = f"low_registration: {len(sfm.cameras)}/{sfm.num_input}"
    elif sfm.registered_ratio < 0.8:
        warnings.append(f"partial_registration: {len(sfm.cameras)}/{sfm.num_input} frames registered")

    points, colors, cameras, fr = np.empty((0, 3)), np.empty((0, 3), np.uint8), [], None
    if degrade_reason is None:
        points, cameras, _ = align_scene(sfm, cfg["planes"])
        colors = sfm.colors
        fr = extract_facades(points, cameras, cfg["planes"], sfm.visibility)
        if len(fr.facades) == 1:  # 면 안 회전(roll)은 법선으로 못 정한다 → 창문 세로 변으로 보정
            f = fr.facades[0]
            lifted = lift_observations(observations_from_detections(det), {c.name: c for c in cameras}, [f],
                                       cfg["windows3d"]["max_inside_margin"])
            tilt = vertical_tilt(lifted)
            report["window_tilt_deg"] = None if tilt is None else round(float(np.degrees(tilt)), 2)
            if tilt is not None and abs(np.degrees(tilt)) > cfg["planes"]["min_tilt_fix_deg"]:
                R = align_rotation(np.sin(tilt) * f.u_axis + np.cos(tilt) * f.v_axis)
                points, cameras = points @ R.T, [c.rotated(R) for c in cameras]
                fr = extract_facades(points, cameras, cfg["planes"], sfm.visibility)
        warnings += fr.warnings
        report["planes"] = {"scene_scale": round(fr.scene, 4),
                            "relief": {f.id: None if f.relief is None else round(float(np.ptp(f.relief[1])), 4)
                                       for f in fr.facades}, "num_points": len(points),
                            "num_kept": int(fr.kept.sum()), "num_ransac_planes": len(fr.planes),
                            "facades": [{"id": f.id, "normal": f.normal.round(4).tolist(), "inliers": len(f.inliers)}
                                        for f in fr.facades],
                            "extra": [{k: v for k, v in e.items() if k != "points"} for e in fr.extra],
                            "ground": round(fr.ground, 4), "top": round(fr.top, 4)}
        if not fr.facades:
            degrade_reason = "no_facade_plane"
        straight = path_straightness(cameras)
        report["path_straightness"] = round(straight, 3)
        if straight < cfg["path"]["straight_ratio"]:
            warnings.append(f"camera_path_straight: {straight:.2f} — depth/plane may be unreliable, walk an arc")
    lap("planes")

    if degrade_reason is None:
        wc = cfg["windows3d"]
        obs = observations_from_detections(det)
        # 민무늬 위층 외벽은 SfM 점이 적어 높이(포인트 백분위)가 낮게 잡힌다 → 창문이 올라간 높이까지 늘린다
        lifted = lift_observations(obs, {c.name: c for c in cameras}, fr.facades, wc["max_inside_margin"])
        if lifted:
            top = float(np.percentile([x.rect[3] for x in lifted], 95)) * (1 + wc["top_margin_ratio"])
            if top > max(f.height for f in fr.facades):
                report["height_from_windows"] = round(top, 4)
                for f in fr.facades:
                    f.height = top
        body, facades, kind = build_mesh(fr.facades, cfg["mesh"])
        lap("mesh")
        textures = build_textures(facades, cameras, lambda n: cv2.imread(str(frames / n)), cfg["ortho"])
        lap("ortho")
        windows, wstats = build_windows(obs, cameras, facades, wc)
        lap("windows3d")
    else:
        warnings.append(f"degraded: {degrade_reason}")
        facade, tex, ppu, windows, ref = degrade_facade(det, frames, cfg["degrade"])
        body, facades, kind = build_mesh([facade], cfg["mesh"])
        textures = {facade.id: (tex, ppu)}
        wstats = {"windows": len(windows), "reference_frame": ref}
        lap("degrade")
    report.update({"mode": "3d" if degrade_reason is None else "degraded", "mesh_kind": kind,
                   "facades": [{"id": f.id, "width": round(f.width, 4), "height": round(f.height, 4)} for f in facades],
                   "windows": wstats})

    status = {"mode": report["mode"], "num_facades": len(facades),
              "registered_ratio": None if sfm is None else round(sfm.registered_ratio, 4), "warnings": warnings}
    meta = _video_meta(frames, len(cameras) if cameras else 1)
    result = export_result(out, out.name, meta, body, facades, textures, windows, status, cfg["ortho"]["jpeg_quality"])
    write_window_obs(out, frames, windows, det)
    lap("export")
    if debug:
        write_debug(out / "debug" / "recon", points, colors, fr, facades, cameras, textures, windows)
        lap("debug")
    report.update({"warnings": warnings, "timings_sec": timings, "total_sec": round(sum(timings.values()), 1),
                   "num_windows": result.summary.total})
    (work / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main(
    frames: Path = typer.Option(..., exists=True, file_okay=False, help="T1 io.extract 출력 폴더"),
    detections: Path = typer.Option(..., exists=True, help="T4 detect.run 출력 폴더 (또는 detections.json)"),
    out: Path = typer.Option(..., help="runs/<id> — inspection_id = 폴더 이름"),
    debug: bool = False,
    camera_model: str = typer.Option(None, help="config sfm.camera_model 덮어쓰기 (SIMPLE_RADIAL, OPENCV_FISHEYE …)"),
    single_camera: bool = typer.Option(None, help="config sfm.single_camera 덮어쓰기 (손떨림 보정 영상은 --no-single-camera)"),
    reuse_sfm: bool = typer.Option(False, help="<out>/recon/colmap의 기존 SfM 모델 재사용 (SfM 생략)"),
) -> None:
    cfg = load_config("recon")
    if camera_model:
        cfg["sfm"]["camera_model"] = camera_model
    if single_camera is not None:
        cfg["sfm"]["single_camera"] = single_camera
    r = run_recon(frames, detections, out, debug, cfg, reuse_sfm)
    sfm = r.get("sfm", {})
    print(f"mode={r['mode']} mesh={r['mesh_kind']} facades={len(r['facades'])} windows={r['num_windows']} "
          f"registered={sfm.get('num_registered')}/{sfm.get('num_input')} total={r['total_sec']}s → {out}")
    for w in r["warnings"]:
        print(f"  ! {w}")


if __name__ == "__main__":
    typer.run(main)
