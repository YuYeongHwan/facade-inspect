"""CSI 보정: 사람 평가(1~5) 수집 → 평가자 간 일치도 → 기준값 4개·k_t를 사람 평가에 맞게 최적화 (docs/scoring.md "보정").

1) 평가 (평가자마다, 이미지 순서는 평가자별 무작위 고정. 매 입력마다 csv에 저장 → 다시 실행하면 이어서)
   python -m facade_cv.contamination.calibrate rate --rater kim --images data/raw/photos/dirty_windows
   python -m facade_cv.contamination.calibrate rate --rater kim --images runs/insp_001 --glob "windows/*/crop.jpg"
2) 일치도 — 평가자 쌍별 quadratic weighted kappa [R24] 표
   python -m facade_cv.contamination.calibrate agreement
3) 보정 — 합의 등급(평가자 중앙값)에 대해 CSI 기준값 4개와 k_t를 가중 카파 최대화로 맞추고 k-fold 교차 검증.
   CSI 계산에 영역 표가 필요하다: 오염 분석이 끝난 result.json (rate --images <그 폴더> --glob "windows/*/crop.jpg")
   python -m facade_cv.contamination.calibrate fit --regions runs/insp_001/result.json --doc docs/scoring.md
   --doc: docs/scoring.md의 보정 결과 절(표 + docs/img/calibration_confusion.png)을 갱신. config 반영은 출력된 yaml을 사람이 옮긴다
"""

import csv
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import typer

from facade_cv.config import CONFIG_DIR, load_config
from facade_cv.detect.datasets.yolo import IMAGE_EXTS

ROOT = CONFIG_DIR.parent
TYPES = ("dust", "stain", "grime", "heavy")
MIN_RELIABLE = 100  # 보정 결과 표에 과적합 경고를 붙이는 이미지 수 기준
DOC_START, DOC_END = "<!-- calibration:start -->", "<!-- calibration:end -->"
app = typer.Typer(no_args_is_help=True)


# ── 일치도 ──
def qwk(a, b, n: int = 5) -> float:
    """quadratic weighted kappa [R24]: 1 − Σ w·O / Σ w·E, w_ij = (i−j)²/(n−1)², 등급 1~n.
    기대 불일치가 0이면(둘 다 한 등급만) 완전 일치일 때 1, 아니면 0."""
    a, b = np.asarray(a, int) - 1, np.asarray(b, int) - 1
    o = np.zeros((n, n))
    np.add.at(o, (a, b), 1)
    e = np.outer(o.sum(1), o.sum(0)) / max(len(a), 1)
    w = (np.subtract.outer(np.arange(n), np.arange(n)) ** 2) / (n - 1) ** 2
    de = (w * e).sum()
    if de == 0:
        return 1.0 if (w * o).sum() == 0 else 0.0
    return float(1 - (w * o).sum() / de)


def read_ratings(ratings_dir: str | Path) -> dict[str, dict[str, int]]:
    """{평가자: {이미지: 등급}}. 같은 이미지를 다시 평가했으면 마지막 값, 빈 등급 줄은 되돌리기."""
    out = {}
    for p in sorted(Path(ratings_dir).glob("*.csv")):
        rs: dict[str, int] = {}
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["rating"]:
                    rs[r["image"]] = int(r["rating"])
                else:  # 되돌리기
                    rs.pop(r["image"], None)
        out[p.stem] = rs
    return out


def consensus(ratings: dict[str, dict[str, int]]) -> dict[str, int]:
    """이미지별 평가자 중앙값 (짝수 명이면 높은 쪽 — 청소 쪽으로 보수적)."""
    images = sorted({k for r in ratings.values() for k in r})
    return {k: int(np.ceil(np.median([r[k] for r in ratings.values() if k in r]))) for k in images}


def agreement_table(ratings: dict[str, dict[str, int]]) -> tuple[list[str], np.ndarray, dict[str, float]]:
    """(평가자 목록, 쌍별 QWK 행렬 (공통 이미지 기준), 평가자별 vs 나머지 합의 QWK)."""
    names = sorted(ratings)
    m = np.eye(len(names))
    for i, a in enumerate(names):
        for j, b in enumerate(names[i + 1:], i + 1):
            common = sorted(set(ratings[a]) & set(ratings[b]))
            m[i, j] = m[j, i] = qwk([ratings[a][k] for k in common], [ratings[b][k] for k in common]) if common else np.nan
    loo = {}
    for a in names:
        rest = consensus({b: ratings[b] for b in names if b != a}) if len(names) > 1 else {}
        common = sorted(set(ratings[a]) & set(rest))
        loo[a] = qwk([ratings[a][k] for k in common], [rest[k] for k in common]) if common else float("nan")
    return names, m, loo


# ── CSI 특징 / 예측 ──
def load_regions(paths: list[Path]) -> dict[str, list[dict]]:
    """{이미지 키: [{type, area_ratio, severity}]}.
    result.json → 키 = images.crop (rate --images <result 폴더> --glob "windows/*/crop.jpg"와 같은 상대경로), 분석된 창문만.
    그 밖의 json은 이미 {이미지: [영역]} 형식으로 본다."""
    out: dict[str, list[dict]] = {}
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        if "windows" in d:
            out |= {w["images"]["crop"]: w["contamination"] for w in d["windows"] if w["csi"] is not None}
        else:
            out |= d
    return out


def features(regions: list[list[dict]]) -> np.ndarray:
    """(N, 4): 이미지마다 종류별 Σ area_ratio × k_s — CSI = X @ k_t / (k_s_max × k_t_max)."""
    x = np.zeros((len(regions), len(TYPES)))
    for i, rs in enumerate(regions):
        for r in rs:
            x[i, TYPES.index(r["type"])] += r["area_ratio"] * r["severity"]
    return x


def csi_values(x: np.ndarray, k: np.ndarray, k_s_max: int) -> np.ndarray:
    return x @ k / (k_s_max * k.max())


def levels_of(csi: np.ndarray, cuts) -> np.ndarray:
    """기준값 4개 (level 2~5 하한) → level 1~5. 경계값은 상위 level."""
    return 1 + np.searchsorted(np.asarray(cuts), csi, side="right")


def fit_cuts(csi: np.ndarray, y: np.ndarray, init, cfg: dict) -> tuple[list[float], float]:
    """기준값 4개 좌표 하강: 하나씩 후보 컷(정렬 CSI 사이 중간점) 중 QWK 최대로. 기준값은 0 < c1 < c2 < c3 < c4로
    엄격히 증가 (같아지면 빈 level이 생긴다). (기준값, QWK)."""
    v = np.unique(csi)
    cand = (v[:-1] + v[1:]) / 2 if len(v) > 1 else v
    if len(cand) > cfg["max_cut_candidates"]:
        cand = np.unique(np.quantile(cand, np.linspace(0, 1, cfg["max_cut_candidates"])))
    cand = np.concatenate([[0.0], cand, [max(csi.max(), 0) * 1.01 + 1e-9]])
    cuts = sorted(float(c) for c in init)
    best = qwk(y, levels_of(csi, cuts))
    for _ in range(cfg["passes"]):
        improved = False
        for i in range(4):
            lo = cuts[i - 1] if i else 0.0
            hi = cuts[i + 1] if i < 3 else np.inf
            for c in cand[(cand > lo) & (cand < hi)]:
                trial = cuts[:i] + [float(c)] + cuts[i + 1:]
                s = qwk(y, levels_of(csi, trial))
                if s > best + 1e-12:
                    best, cuts, improved = s, trial, True
        if not improved:
            break
    return cuts, best


def fit(x: np.ndarray, y: np.ndarray, k0: np.ndarray, cuts0, k_s_max: int, cfg: dict) -> dict:
    """k_t(상대값, 최댓값 1)와 기준값을 번갈아 좌표 하강. 바꿔도 QWK가 오르지 않으면 초기값 유지
    (평가 데이터에 없는 종류의 k_t는 그대로 남는다)."""
    grid = np.arange(cfg["kt_grid"][0], cfg["kt_grid"][1] + 1e-9, cfg["kt_grid"][2])
    k = k0 / k0.max()
    cuts, best = fit_cuts(csi_values(x, k, k_s_max), y, cuts0, cfg)
    for _ in range(cfg["passes"]):
        improved = False
        for t in range(len(TYPES)):
            if not x[:, t].any():
                continue
            for g in grid:
                trial = k.copy()
                trial[t] = g
                trial = trial / trial.max()
                c, s = fit_cuts(csi_values(x, trial, k_s_max), y, cuts, cfg)
                if s > best + 1e-9:
                    best, k, cuts, improved = s, trial, c, True
        if not improved:
            break
    return {"k_rel": k, "cuts": cuts, "qwk": best}


def cross_validate(x: np.ndarray, y: np.ndarray, k0, cuts0, k_s_max: int, cfg: dict) -> list[float]:
    """k-fold: train에서 fit, test에서 QWK."""
    idx = np.random.default_rng(cfg["seed"]).permutation(len(y))
    folds = np.array_split(idx, min(cfg["folds"], len(y)))
    out = []
    for f in folds:
        tr = np.setdiff1d(idx, f)
        r = fit(x[tr], y[tr], k0, cuts0, k_s_max, cfg)
        out.append(qwk(y[f], levels_of(csi_values(x[f], r["k_rel"], k_s_max), r["cuts"])))
    return out


def confusion(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    m = np.zeros((5, 5), int)
    np.add.at(m, (y - 1, p - 1), 1)
    return m


# ── 평가 화면 ──
def rating_order(images: list[str], rater: str, done: set[str]) -> list[str]:
    """평가자별 고정 무작위 순서 (이름이 시드) 중 아직 안 한 것."""
    order = sorted(images)
    random.Random(rater).shuffle(order)
    return [k for k in order if k not in done]


def append_rating(path: Path, image: str, rating: int | None) -> None:
    """한 줄 추가 (즉시 저장 = 중간 저장). rating None = 되돌리기 (빈 값 — read_ratings가 그 이미지 평가를 지운다)."""
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["image", "rating", "time"])
        w.writerow([image, "" if rating is None else rating, time.strftime("%Y-%m-%dT%H:%M:%S")])


def done_images(path: Path) -> set[str]:
    """평가 끝난 이미지 (되돌린 것 제외)."""
    if not path.exists():
        return set()
    state: dict[str, bool] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            state[r["image"]] = bool(r["rating"])
    return {k for k, v in state.items() if v}


def _panel(height: int, cfg: dict, grading: dict, progress: str) -> np.ndarray:
    """기준표 패널 (한글은 PIL로 그린다 — cv2.putText는 한글 미지원)."""
    from PIL import Image, ImageDraw, ImageFont

    try:
        font, small = ImageFont.truetype(cfg["font"], 22), ImageFont.truetype(cfg["font"], 17)
    except OSError:
        font = small = ImageFont.load_default()
    img = Image.new("RGB", (600, height), (24, 24, 24))
    d = ImageDraw.Draw(img)
    y = 16
    d.text((16, y), f"이 창문의 청소 우선순위는?  {progress}", font=font, fill=(255, 255, 255))
    y += 44
    for lv in grading["levels"]:
        r, g, b = (int(lv["color"][i:i + 2], 16) for i in (1, 3, 5))
        d.rectangle((16, y + 4, 34, y + 22), fill=(r, g, b))
        d.text((44, y), f"{lv['level']}  {lv['label']}", font=font, fill=(255, 255, 255))
        d.text((44, y + 28), cfg["rubric"][lv["level"]], font=small, fill=(190, 190, 190))
        y += 68
    y += 8
    for n in cfg["notes"]:
        d.text((16, y), f"· {n}", font=small, fill=(160, 200, 255))
        y += 26
    return np.array(img)[..., ::-1]


def run_rating(images_dir: Path, pattern: str, rater: str, cfg: dict) -> int:
    import cv2

    grading = load_config("grading")
    images = [p.relative_to(images_dir).as_posix() for p in images_dir.glob(pattern) if p.suffix.lower() in IMAGE_EXTS]
    if not images:
        raise FileNotFoundError(f"no images: {images_dir}/{pattern}")
    out = ROOT / cfg["ratings_dir"] / f"{rater}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    todo = rating_order(images, rater, done_images(out))
    history: list[str] = []
    i, n_done = 0, len(images) - len(todo)
    win = f"CSI rating - {rater}"
    while i < len(todo):
        img = cv2.imread(str(images_dir / todo[i]))
        s = cfg["view_max_side"] / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        h = max(img.shape[0], 620)
        canvas = np.zeros((h, img.shape[1], 3), np.uint8)
        canvas[: img.shape[0]] = img
        cv2.imshow(win, np.hstack([canvas, _panel(h, cfg, grading, f"{n_done + 1}/{len(images)}")]))
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        if ord("1") <= key <= ord("5"):
            append_rating(out, todo[i], key - ord("0"))
            history.append(todo[i])
            i, n_done = i + 1, n_done + 1
        elif key == ord("s"):  # 건너뛰기 (다음 실행 때 다시 나온다)
            i += 1
        elif key == ord("b") and history:  # 되돌리기
            prev = history.pop()
            append_rating(out, prev, None)
            todo.insert(i, prev)
            n_done -= 1
    cv2.destroyAllWindows()
    print(f"{rater}: {n_done}/{len(images)} 평가 → {out}")
    return n_done


# ── CLI ──
@app.command()
def rate(
    rater: str = typer.Option(..., help="평가자 이름 (영문, csv 파일명)"),
    images: Path = typer.Option(..., exists=True, file_okay=False, help="창문 이미지 폴더 (휴대폰 사진도 가능)"),
    glob: str = typer.Option("**/*", help='예: "windows/*/crop.jpg" (result 폴더)'),
) -> None:
    if not re.fullmatch(r"[\w-]+", rater):
        raise typer.BadParameter("rater: 영문·숫자·_- 만")
    run_rating(images, glob, rater, load_config("calibrate"))


@app.command()
def agreement(ratings_dir: Path = typer.Option(None)) -> None:
    ratings = read_ratings(ratings_dir or ROOT / load_config("calibrate")["ratings_dir"])
    print(agreement_md(ratings))


def agreement_md(ratings: dict[str, dict[str, int]]) -> str:
    names, m, loo = agreement_table(ratings)
    head = "| | " + " | ".join(names) + " | 나머지 합의 대비 |\n|---|" + "---|" * (len(names) + 1)
    rows = [f"| {a} ({len(ratings[a])}장) | " + " | ".join("—" if i == j else f"{m[i, j]:.3f}" for j in range(len(names)))
            + f" | {loo[a]:.3f} |" for i, a in enumerate(names)]
    off = m[~np.eye(len(names), dtype=bool)]
    mean = f"평균 쌍별 QWK {np.nanmean(off):.3f}" if off.size else "평가자 1명 — 쌍별 일치도 없음"
    return "\n".join([head, *rows, "", mean])


@app.command("fit")
def fit_cmd(
    regions: list[Path] = typer.Option(..., exists=True, dir_okay=False, help="오염 분석 result.json (여러 개 가능)"),
    ratings_dir: Path = typer.Option(None),
    doc: Path = typer.Option(None, help="docs/scoring.md — 보정 결과 절 갱신 + 혼동 행렬 그림"),
) -> None:
    cfg, grading = load_config("calibrate"), load_config("grading")
    ratings = read_ratings(ratings_dir or ROOT / cfg["ratings_dir"])
    cons = consensus(ratings)
    reg = load_regions(regions)
    keys = sorted(set(cons) & set(reg))
    if len(keys) < 10:
        raise typer.BadParameter(f"평가와 영역 표가 둘 다 있는 이미지 {len(keys)}장 — 10장 이상 필요")
    x, y = features([reg[k] for k in keys]), np.array([cons[k] for k in keys])
    k0 = np.array([grading["weights"][t] for t in TYPES], float)
    cuts0 = [lv["min_csi"] for lv in sorted(grading["levels"], key=lambda lv: lv["level"])][1:]
    ks = grading["severity"]["max"]
    p0 = levels_of(csi_values(x, k0, ks), cuts0)
    r = fit(x, y, k0, cuts0, ks, cfg["fit"])
    p1 = levels_of(csi_values(x, r["k_rel"], ks), r["cuts"])
    cv = cross_validate(x, y, k0, cuts0, ks, cfg["fit"])
    weights = {t: round(float(v * k0.max()), 3) for t, v in zip(TYPES, r["k_rel"])}
    md = result_md(ratings, len(keys), qwk(y, p0), r["qwk"], cv, weights, r["cuts"], grading)
    print(md)
    print("\n# config/grading.yaml 반영용 (사람이 검토 후 옮긴다)")
    print("weights: " + json.dumps(weights))
    print("level min_csi: " + json.dumps([0.0] + [round(c, 4) for c in r["cuts"]]))
    if doc:
        fig = doc.parent / "img" / "calibration_confusion.png"
        save_confusion_png(fig, confusion(y, p0), confusion(y, p1))
        write_doc(doc, md + f"\n\n![혼동 행렬](img/{fig.name})\n")
        print(f"→ {doc}, {fig}")


def result_md(ratings, n, q0, q1, cv, weights, cuts, grading) -> str:
    return "\n".join([
        f"보정 실행: {time.strftime('%Y-%m-%d %H:%M')} · 평가자 {len(ratings)}명 · 평가+영역 표가 있는 이미지 {n}장",
        "", "**평가자 간 일치도** (quadratic weighted kappa [R24], 해석 기준 [R25])", "", agreement_md(ratings), "",
        "**CSI ↔ 합의 등급 일치도**", "",
        "| 설정 | QWK |", "|---|---|",
        f"| 보정 전 초기값 (config) | {q0:.3f} |",
        f"| 보정값 (전체 데이터에 맞춤) | {q1:.3f} |",
        f"| 보정값 {len(cv)}-fold 교차 검증 (held-out) | {np.mean(cv):.3f} ± {np.std(cv):.3f} |",
        *(["", f"⚠️ 이미지 {n}장 — {MIN_RELIABLE}장 미만이라 보정값이 과적합됐을 수 있다. 교차 검증 값을 기준으로 판단할 것"]
          if n < MIN_RELIABLE else []),
        "", "**보정값**", "",
        "| 항목 | 초기값 | 보정값 |", "|---|---|---|",
        *[f"| k_t {t} | {grading['weights'][t]} | {weights[t]} |" for t in TYPES],
        *[f"| level {i + 2} 하한 (min_csi) | {lv} | {c:.4f} |" for i, (lv, c) in
          enumerate(zip([lv["min_csi"] for lv in grading["levels"]][1:], cuts))],
    ])


def save_confusion_png(path: Path, before: np.ndarray, after: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, m, title in zip(axes, (before, after), ("initial (config)", "calibrated")):
        ax.imshow(m, cmap="Blues")
        for i in range(5):
            for j in range(5):
                ax.text(j, i, m[i, j], ha="center", va="center", color="white" if m[i, j] > m.max() / 2 else "black")
        ax.set(xticks=range(5), yticks=range(5), xticklabels=range(1, 6), yticklabels=range(1, 6),
               xlabel="CSI level", ylabel="human consensus level", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_doc(doc: Path, body: str) -> None:
    """docs/scoring.md의 DOC_START~DOC_END 사이를 body로 교체 (표식이 없으면 끝에 붙인다)."""
    s = doc.read_text(encoding="utf-8")
    block = f"{DOC_START}\n{body}\n{DOC_END}"
    if DOC_START in s and DOC_END in s:
        s = s[: s.index(DOC_START)] + block + s[s.index(DOC_END) + len(DOC_END):]
    else:
        s = s.rstrip() + "\n\n" + block + "\n"
    doc.write_text(s, encoding="utf-8")


if __name__ == "__main__":
    app()
