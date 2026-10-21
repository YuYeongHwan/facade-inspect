"""pycolmap SfM [R9]: T1 프레임 → SIFT [R16] → sequential(+루프 검출) / exhaustive 매칭 → incremental mapping.

python -m facade_cv.recon3d.sfm --frames runs/<id>/frames --out runs/<id>/recon
카메라 모델·공유 여부는 config/recon.yaml sfm. 등록 실패 프레임 비율을 리포트한다.
pycolmap은 함수 안에서 import한다 (torch와 OpenMP 런타임 충돌 — facade_cv/device.py 참고).
"""

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import typer

from facade_cv.config import load_config
from facade_cv.detect.datasets.yolo import list_images
from facade_cv.recon3d.geometry import CameraView


@dataclass
class SfmResult:
    cameras: list[CameraView]  # 등록된 프레임 (이름순)
    points: np.ndarray  # (N,3)
    colors: np.ndarray  # (N,3) uint8 RGB
    num_input: int
    unregistered: list[str]
    num_models: int  # COLMAP이 만든 분리 모델 수 (가장 큰 것만 씀)
    camera_info: dict = field(default_factory=dict)
    visibility: list[np.ndarray] | None = None  # 포인트마다 그 점을 실제로 관측한 cameras index (SfM track)

    @property
    def registered_ratio(self) -> float:
        return len(self.cameras) / self.num_input if self.num_input else 0.0

    def report(self) -> dict:
        return {"num_input": self.num_input, "num_registered": len(self.cameras),
                "registered_ratio": round(self.registered_ratio, 3), "num_points": len(self.points),
                "num_models": self.num_models, "unregistered": self.unregistered, "camera": self.camera_info}


def _view(image) -> CameraView:  # pycolmap.Image
    cam = image.camera
    M = image.cam_from_world().matrix()
    return CameraView(image.name, M[:, :3].copy(), M[:, 3].copy(), cam.width, cam.height,
                      lambda Xc, c=cam: np.asarray(c.img_from_cam(Xc)),
                      lambda px, c=cam: np.asarray(c.cam_ray_from_img(px)))


def from_reconstruction(rec, num_input: int, names: list[str], num_models: int) -> SfmResult:
    images = sorted((im for im in rec.images.values() if im.has_pose), key=lambda im: im.name)
    pts = list(rec.points3D.values())
    order = {im.image_id: i for i, im in enumerate(images)}
    reg = {im.name for im in images}
    cams = {c.camera_id: c for c in rec.cameras.values()}
    focal = [float(c.mean_focal_length()) for c in cams.values()]
    info = {"model": next(iter(cams.values())).model_name if cams else None, "num_cameras": len(cams),
            "focal_px": {"min": round(min(focal), 1), "max": round(max(focal), 1)} if focal else None}
    return SfmResult(
        [_view(im) for im in images],
        np.array([p.xyz for p in pts]).reshape(-1, 3),
        np.array([p.color for p in pts], np.uint8).reshape(-1, 3),
        num_input, [n for n in names if n not in reg], num_models, info,
        [np.array([order[e.image_id] for e in p.track.elements if e.image_id in order], int) for p in pts],
    )


def run_sfm(frames: str | Path, work: str | Path, cfg: dict | None = None) -> SfmResult:
    """work/database.db, work/sparse/<k>/ 를 만든다 (기존 것은 지움). 모델이 없으면 등록 0인 결과."""
    import pycolmap

    cfg = cfg or load_config("recon")["sfm"]
    frames, work = Path(frames), Path(work)
    names = [p.name for p in list_images(frames)]
    if work.exists():
        shutil.rmtree(work)
    (work / "sparse").mkdir(parents=True)  # sparse/seed<k>/<model>/
    db = work / "database.db"

    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = cfg["camera_model"]
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.sift.max_num_features = cfg["max_num_features"]
    mode = pycolmap.CameraMode.SINGLE if cfg["single_camera"] else pycolmap.CameraMode.PER_IMAGE
    pycolmap.extract_features(db, frames, names, camera_mode=mode, reader_options=reader,
                              extraction_options=extraction)

    if cfg["matcher"] == "exhaustive":
        pycolmap.match_exhaustive(db)
    else:
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = cfg["sequential_overlap"]
        if cfg["loop_detection"]:
            if not cfg["vocab_tree_path"]:
                raise ValueError("sfm.loop_detection needs sfm.vocab_tree_path")
            pairing.loop_detection = True
            pairing.vocab_tree_path = str(cfg["vocab_tree_path"])
        pycolmap.match_sequential(db, pairing_options=pairing)

    best, best_n, trials = None, 0, []
    for seed in range(cfg["mapping_trials"]):
        pycolmap.set_random_seed(seed)
        opts = pycolmap.IncrementalPipelineOptions()
        opts.random_seed = seed
        opts.mapper.abs_pose_min_num_inliers = cfg["abs_pose_min_num_inliers"]
        opts.mapper.abs_pose_min_inlier_ratio = cfg["abs_pose_min_inlier_ratio"]
        out = work / "sparse" / f"seed{seed}"
        out.mkdir()
        models = pycolmap.incremental_mapping(db, frames, out, opts)
        top = max(models.values(), key=lambda r: r.num_reg_images()) if models else None
        n = top.num_reg_images() if top else 0
        trials.append({"seed": seed, "registered": n, "models": len(models)})
        if n > best_n:
            best, best_n, num_models = top, n, len(models)
        if best_n >= cfg["mapping_target_ratio"] * len(names):
            break
    if best is None:
        return SfmResult([], np.empty((0, 3)), np.empty((0, 3), np.uint8), len(names), names, 0,
                         {"trials": trials})
    r = from_reconstruction(best, len(names), names, num_models)
    r.camera_info["trials"] = trials
    return r


def main(frames: Path = typer.Option(..., exists=True, file_okay=False), out: Path = typer.Option(...),
         camera_model: str = typer.Option(None), single_camera: bool = typer.Option(None)) -> None:
    cfg = load_config("recon")["sfm"]
    if camera_model:
        cfg["camera_model"] = camera_model
    if single_camera is not None:
        cfg["single_camera"] = single_camera
    r = run_sfm(frames, out, cfg)
    print(json.dumps(r.report(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    typer.run(main)
