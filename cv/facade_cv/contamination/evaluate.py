"""T7 오염 분석 검증 — data/eval/t7_dev.csv (단일 이미지 모드). 데이터: data/DATASETS.md "오염 창문 평가 데이터"

python -m facade_cv.contamination.evaluate t7 [--out runs/t7_dev]
python -m facade_cv.contamination.evaluate calib [--out runs/csi_calib]   # 보정용 평가지 (아래)

- 공개 근접 사진 (window check 계열 box, dirty window polygon): 이미지 전체를 유리로 보고 분석
  - dirt / stain / 심한 오염: dirt·stain 박스 중 오염이 잡힌 박스 비율 (박스 면적의 eval.hit_min 이상)
  - glare / wet_glass: 그 박스 면적 중 오염으로 잘못 잡은 비율
- 깨끗한 창 (CMP 정면 외벽, b04 프레임): T4로 창문 검출 → 창별 정면화 → 분석 → level ≤ 2 비율
결과: <out>/NN_<분류>.jpg (왼쪽 정답 박스, 오른쪽 검출), summary.csv, summary.md

heavy(심한 오염) 분류의 박스는 사진 전체를 덮어 "박스 적중"이 '5% 이상 무엇이든 잡았나'일 뿐이다 → 검출률로 쓰지 않고 참고로만 표시.

--labels <json>: 오염 부분 polygon 정답 (직접 촬영 사진용, 없으면 생략). 형식 —
  {"<csv path 열과 같은 경로>": [{"class": "dirt|stain|heavy|...", "polygon": [[u, v], ...]}, ...], ...}
  (u, v)는 이미지 폭·높이 대비 0~1. 있으면 그 이미지에 면적 IoU = |검출 ∩ 정답| / |검출 ∪ 정답| (유리 안)을 계산한다.

calib: data/eval/csi_calib.csv 60장 → 평가자 3명용 블라인드 평가지. 이 결과로 contamination.yaml 임계값을 고치지 않는다
(보정은 grading.yaml k_t·level 경계만, calibrate fit). 평가 단위 = 분석한 이미지 그대로 (공개 사진은 전체,
외벽 사진은 T4로 검출한 창 중 분석 가능한 가장 큰 창 1개를 정면화). 출력 <out>/
  images/img_NNN.jpg  익명·순서 섞음 · regions.json  {img_NNN.jpg: 오염 영역} (calibrate fit --regions)
  pred.csv  예측 CSI·level (평가자에게 주지 않음) · key.csv  익명 이름 ↔ 원본 경로·split (평가자에게 주지 않음)
  sheet.html  이미지·기준표(config/calibrate.yaml)·평가자 3열 → 평가자별 csv 다운로드 (data/ratings/<이름>.csv 형식)
"""

import csv
import html
import json
import random
from pathlib import Path

import cv2
import numpy as np
import typer
import yaml

from facade_cv.config import CONFIG_DIR, load_config
from facade_cv.contamination.analyze import _bgr, analyze_crop, draw_overlay, rectify_quad, window_fields

ROOT = CONFIG_DIR.parent
BOX_COLORS = {"dirt": (0, 140, 255), "stain": (255, 190, 60), "glare": (255, 255, 255), "wet_glass": (255, 80, 200)}
DIRT = ("dirt", "stain")
NOT_DIRT = ("glare", "wet_glass")


def category(note: str) -> str:
    head = note.split(";")[0]
    for key, cat in (("dirt", "dirt"), ("stain", "stain"), ("심한", "heavy"), ("glare", "glare"),
                     ("wet_glass", "wet_glass"), ("깨끗", "clean")):
        if head.startswith(key):
            return cat
    raise ValueError(f"unknown t7_dev category: {note!r}")


def yolo_labels(img_path: Path) -> list[tuple[str, np.ndarray]]:
    """YOLO 라벨 (box 또는 polygon) → [(클래스 이름, 정규화 bbox [x0 y0 x1 y1])]. 라벨 없으면 []."""
    lbl = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
    if not lbl.exists():
        return []
    names = yaml.safe_load((img_path.parents[2] / "data.yaml").read_text())["names"]
    out = []
    for line in lbl.read_text().split("\n"):
        v = [float(x) for x in line.split()]
        if not v:
            continue
        if len(v) == 5:
            _, cx, cy, w, h = v
            box = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        else:
            p = np.array(v[1:]).reshape(-1, 2)
            box = np.r_[p.min(0), p.max(0)]
        out.append((names[int(v[0])], np.clip(box, 0, 1)))
    return out


def box_mask(box: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    x0, y0, x1, y1 = np.round(box * [w, h, w, h]).astype(int)
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


def polygon_mask(polys: list[dict], shape: tuple[int, int]) -> np.ndarray:
    """정규화 polygon 정답 목록 → bool 마스크."""
    h, w = shape
    m = np.zeros(shape, np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.round(np.asarray(p["polygon"], float) * [w, h]).astype(np.int32)], 1)
    return m.astype(bool)


def eval_public(img: np.ndarray, labels, cat: str, cfg: dict, grading: dict,
                polys: list[dict] | None = None) -> tuple[dict, np.ndarray]:
    a = analyze_crop(img, None, cfg, grading)
    f = window_fields(a, "single", True, cfg, grading)
    shape = a.crop.shape[:2]
    det = np.zeros(shape, bool)
    for r in a.regions:
        det |= r.mask
    row = {"level": a.level, "csi": round(a.csi, 4), "wcov_pct": f["weighted_coverage_pct"],
           "det_ratio": round(det.sum() / a.glass.sum(), 3), "regions": len(a.regions),
           "types": "/".join(sorted({r.type for r in a.regions}))}
    dirt_boxes = [box_mask(b, shape) & a.glass for c, b in labels if c in DIRT]
    dirt_boxes = [m for m in dirt_boxes if m.any()]
    if cat in ("dirt", "stain", "heavy") and dirt_boxes:
        hits = [(det & m).sum() / m.sum() >= cfg["eval"]["hit_min"] for m in dirt_boxes]
        row["box_hit"] = round(float(np.mean(hits)), 3)
        union = np.logical_or.reduce(dirt_boxes)
        row["det_in_box"] = round(float((det & union).sum() / max(det.sum(), 1)), 3)
    if polys:
        gt = polygon_mask(polys, shape) & a.glass
        row["iou"] = round(float((det & gt).sum() / max((det | gt).sum(), 1)), 3)
    neg = [box_mask(b, shape) & a.glass for c, b in labels if c in NOT_DIRT and c == cat]
    if cat in NOT_DIRT and neg:
        union = np.logical_or.reduce(neg)
        row["false_in_box"] = round(float((det & union).sum() / max(union.sum(), 1)), 3)

    left = a.crop.copy()
    for c, b in labels:
        h, w = shape
        x0, y0, x1, y1 = np.round(b * [w, h, w, h]).astype(int)
        cv2.rectangle(left, (x0, y0), (x1, y1), BOX_COLORS.get(c, (0, 255, 0)), 2)
        cv2.putText(left, c, (x0 + 3, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, BOX_COLORS.get(c, (0, 255, 0)), 1)
    return row, np.hstack([left, draw_overlay(a, cfg)])


def eval_facade(img: np.ndarray, seg, cfg: dict, grading: dict) -> tuple[dict, np.ndarray]:
    """깨끗한 창: T4 창문 검출 → 창별 분석. 시각화 = 창 사각형을 level 색으로."""
    from facade_cv.detect.postprocess import postprocess

    dcfg = seg.cfg
    res = postprocess(img, seg.predict(img, conf=dcfg["conf"] * dcfg["fill_conf_ratio"]), dcfg)
    colors = {lv["level"]: _bgr(lv["color"]) for lv in grading["levels"]}
    vis, levels, small = img.copy(), [], 0
    for w in res.windows:
        crop, _ = rectify_quad(img, w.corners, cfg["rectify"]["max_side"])
        if min(crop.shape[:2]) < cfg["rectify"]["min_side"]:  # 해상도 부족 → 미분석 (회색)
            small += 1
            cv2.polylines(vis, [np.round(w.corners).astype(np.int32)], True, (128, 128, 128), 1)
            continue
        a = analyze_crop(crop, None, cfg, grading)
        levels.append(a.level)
        q = np.round(w.corners).astype(np.int32)
        cv2.polylines(vis, [q], True, colors[a.level], 3)
        cv2.putText(vis, str(a.level), tuple(q[0] + [4, 22]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[a.level], 2)
    lv = np.array(levels)
    return {"windows": len(lv), "too_small": small, "level": int(lv.max()) if len(lv) else None,
            "clean_ok": round(float((lv <= 2).mean()), 3) if len(lv) else None,
            "level_hist": "/".join(str(int((lv == k).sum())) for k in range(1, 6))}, vis


def run_t7(csv_path: Path, out: Path, cfg: dict | None = None, grading: dict | None = None,
           labels_json: Path | None = None) -> list[dict]:
    cfg = cfg or load_config("contamination")
    polys = json.loads(Path(labels_json).read_text(encoding="utf-8")) if labels_json else {}
    grading = grading or load_config("grading")
    out.mkdir(parents=True, exist_ok=True)
    seg = None
    rows = []
    for i, r in enumerate(csv.DictReader(open(csv_path, encoding="utf-8")), 1):
        p = ROOT / r["path"]
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(p)
        cat = category(r["note"])
        if r["source"] in ("cmp", "b04"):
            if seg is None:
                from facade_cv.detect.segmenter import WindowSegmenter
                seg = WindowSegmenter()
            row, vis = eval_facade(img, seg, cfg, grading)
        else:
            row, vis = eval_public(img, yolo_labels(p), cat, cfg, grading, polys.get(r["path"]))
        name = f"{i:02d}_{cat}"
        cv2.imwrite(str(out / f"{name}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
        rows.append({"id": name, "category": cat, "source": r["source"], **row, "path": r["path"]})
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, keys)
        wr.writeheader()
        wr.writerows(rows)
    (out / "summary.md").write_text(summary_md(rows), encoding="utf-8")
    return rows


def _mean(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return f"{np.mean(v):.2f}" if v else "—"


def summary_md(rows: list[dict]) -> str:
    """분류별 집계 + 이미지별 표."""
    cats = ["dirt", "stain", "heavy", "glare", "wet_glass", "clean"]
    lines = ["| 분류 | 장수 | 박스 적중률 (dirt/stain 박스) | 검출 중 박스 안 비율 | 박스 안 오검출률 (glare/wet) | 평균 level | 깨끗한 창 level≤2 |",
             "|---|---|---|---|---|---|---|"]
    for c in cats:
        rs = [r for r in rows if r["category"] == c]
        hit, inbox = _mean(rs, "box_hit"), _mean(rs, "det_in_box")
        if c == "heavy" and hit != "—":  # 정답 박스가 사진 전체 → 검출률 아님
            hit, inbox = f"({hit})*", f"({inbox})*"
        lines.append(f"| {c} | {len(rs)} | {hit} | {inbox} | "
                     f"{_mean(rs, 'false_in_box')} | {_mean(rs, 'level')} | {_mean(rs, 'clean_ok')} |")
    lines.append("")
    lines.append("\\* heavy: 정답 박스가 사진 전체라 검출률로 쓰지 않음 (박스의 5% 이상을 무엇이든 잡았는지만 — 참고값)")
    if any(r.get("iou") is not None for r in rows):
        lines.append(f"\n면적 IoU (polygon 정답 있는 {sum(r.get('iou') is not None for r in rows)}장): {_mean(rows, 'iou')}")
    lines += ["", "| id | level | CSI | 가중 면적률% | 유리 중 검출 | 유형 | 박스 적중 | 박스 안 오검출 | 창(level 1/2/3/4/5) | 해상도 부족 창 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        g = lambda k: "—" if r.get(k) is None else r[k]  # noqa: E731
        lines.append(f"| {r['id']} | {g('level')} | {g('csi')} | {g('wcov_pct')} | {g('det_ratio')} | {g('types')} | "
                     f"{g('box_hit')} | {g('false_in_box')} | {g('level_hist')} | {g('too_small')} |")
    return "\n".join(lines) + "\n"


app = typer.Typer(add_completion=False)


@app.callback()
def _main() -> None:  # 하위 명령 이름을 항상 받게 (t7, 이후 calib)
    pass


@app.command()
def t7(csv_path: Path = typer.Option(ROOT / "data/eval/t7_dev.csv", "--csv", exists=True),
       out: Path = typer.Option(ROOT / "runs/t7_dev"),
       labels: Path = typer.Option(None, exists=True, dir_okay=False, help="오염 polygon 정답 json (모듈 설명의 형식)")) -> None:
    """t7_dev 30장 평가 → 오버레이·summary.csv·summary.md."""
    rows = run_t7(csv_path, out, labels_json=labels)
    print(summary_md(rows).split("\n\n| id")[0])
    print(f"→ {out}")


def calib_item(img: np.ndarray, source: str, seg, cfg: dict, grading: dict):
    """평가 단위 이미지와 분석 결과. 외벽 사진이면 분석 가능한 가장 큰 창 (없으면 None)."""
    if source not in ("cmp", "b04"):
        return analyze_crop(img, None, cfg, grading)
    from facade_cv.detect.postprocess import postprocess

    res = postprocess(img, seg.predict(img, conf=seg.cfg["conf"] * seg.cfg["fill_conf_ratio"]), seg.cfg)
    crops = [rectify_quad(img, w.corners, cfg["rectify"]["max_side"])[0] for w in res.windows]
    crops = [c for c in crops if min(c.shape[:2]) >= cfg["rectify"]["min_side"]]
    return analyze_crop(max(crops, key=lambda c: c.size), None, cfg, grading) if crops else None


def run_calib(csv_path: Path, out: Path, seed: int = 0) -> list[dict]:
    cfg, grading, ccfg = load_config("contamination"), load_config("grading"), load_config("calibrate")
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    random.Random(seed).shuffle(rows)
    (out / "images").mkdir(parents=True, exist_ok=True)
    seg, items, regions = None, [], {}
    for r in rows:
        if r["source"] in ("cmp", "b04") and seg is None:
            from facade_cv.detect.segmenter import WindowSegmenter
            seg = WindowSegmenter()
        a = calib_item(cv2.imread(str(ROOT / r["path"])), r["source"], seg, cfg, grading)
        if a is None:
            print(f"건너뜀 (분석 가능한 창 없음): {r['path']}")
            continue
        name = f"img_{len(items) + 1:03d}.jpg"
        cv2.imwrite(str(out / "images" / name), a.crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        f = window_fields(a, "single", True, cfg, grading)
        regions[name] = f["contamination"]
        items.append({"image": name, "csi": f["csi"], "level": f["cleaning_level"], "mode": "single",
                      "path": r["path"], "source": r["source"], "split": r["split"]})
    (out / "regions.json").write_text(json.dumps(regions, ensure_ascii=False), encoding="utf-8")
    for fn, keys in (("pred.csv", ["image", "csi", "level", "mode"]), ("key.csv", ["image", "path", "source", "split"])):
        with open(out / fn, "w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(items)
    (out / "sheet.html").write_text(sheet_html([i["image"] for i in items], ccfg, grading), encoding="utf-8")
    return items


def sheet_html(images: list[str], ccfg: dict, grading: dict, raters: int = 3) -> str:
    """정답·예측이 없는 평가지. 평가 결과는 평가자별 csv (image,rating,time) 다운로드 — calibrate fit이 바로 읽는다."""
    labels = {lv["level"]: lv["label"] for lv in grading["levels"]}
    rubric = "".join(f"<tr><th>{k}</th><td>{html.escape(labels[int(k)])}</td><td>{html.escape(v)}</td></tr>"
                     for k, v in ccfg["rubric"].items())
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in ccfg["notes"] if "키 1~5" not in n)
    heads = "".join(f'<th><input class="rn" data-r="{j}" placeholder="평가자{j + 1} 이름(영문)"></th>' for j in range(raters))
    body = "".join(
        f'<tr><td><a href="images/{im}" target="_blank"><img src="images/{im}" loading="lazy"></a><div>{im}</div></td>'
        + "".join(f'<td><select data-r="{j}" data-i="{im}"><option value="">-</option>'
                  + "".join(f"<option>{k}</option>" for k in range(1, 6)) + "</select></td>" for j in range(raters))
        + "</tr>" for im in images)
    return f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>창문 오염 평가지</title>
<style>body{{font-family:system-ui,sans-serif;margin:16px;background:#fff;color:#111}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:6px;vertical-align:top}}img{{max-width:420px;max-height:320px;display:block}}
select{{font-size:20px}}.rn{{width:9em}}#bar{{position:sticky;top:0;background:#fff;padding:8px 0}}</style>
<h1>창문 오염 평가지 ({len(images)}장)</h1>
<p>각 사진의 <b>유리</b>만 보고 청소 우선순위 1~5를 고르세요. 서로 의논하지 말고 각자 자기 열만 채웁니다.</p>
<table><tr><th>등급</th><th>라벨</th><th>기준</th></tr>{rubric}</table><ul>{notes}</ul>
<div id="bar">{"".join(f'<button data-r="{j}">평가자{j + 1} csv 받기</button> ' for j in range(raters))}
<span id="cnt"></span> — 입력은 이 브라우저에 자동 저장됩니다. 받은 csv를 data/ratings/&lt;이름&gt;.csv로 두세요.</div>
<table><tr><th>사진</th>{heads}</tr>{body}</table>
<script>
const K="t7-calib-sheet";let st={{}};try{{st=JSON.parse(localStorage.getItem(K)||"{{}}")}}catch(e){{}}
const save=()=>{{try{{localStorage.setItem(K,JSON.stringify(st))}}catch(e){{}};count()}};
const sel=[...document.querySelectorAll("select")],rn=[...document.querySelectorAll(".rn")];
sel.forEach(s=>{{s.value=(st[s.dataset.r]||{{}})[s.dataset.i]||"";s.onchange=()=>{{(st[s.dataset.r]??={{}})[s.dataset.i]=s.value;save()}}}});
rn.forEach(i=>{{i.value=(st.names||{{}})[i.dataset.r]||"";i.oninput=()=>{{(st.names??={{}})[i.dataset.r]=i.value;save()}}}});
function count(){{document.getElementById("cnt").textContent=rn.map(i=>`${{i.value||"평가자"+(+i.dataset.r+1)}} ${{sel.filter(s=>s.dataset.r==i.dataset.r&&s.value).length}}/{len(images)}`).join(" · ")}}
document.querySelectorAll("button").forEach(b=>b.onclick=()=>{{const r=b.dataset.r,n=rn[r].value.trim();
if(!/^[\\w-]+$/.test(n)){{rn[r].focus();rn[r].style.outline="2px solid red";return}}
const t=new Date().toISOString().slice(0,19),rows=sel.filter(s=>s.dataset.r==r&&s.value).map(s=>`${{s.dataset.i}},${{s.value}},${{t}}`);
const a=document.createElement("a");a.href=URL.createObjectURL(new Blob(["image,rating,time\\n"+rows.join("\\n")+"\\n"],{{type:"text/csv"}}));
a.download=n+".csv";a.click()}});count();
</script></html>
"""


@app.command()
def calib(csv_path: Path = typer.Option(ROOT / "data/eval/csi_calib.csv", "--csv", exists=True),
          out: Path = typer.Option(ROOT / "runs/csi_calib"), seed: int = 0) -> None:
    """csi_calib 60장 → 블라인드 평가지·regions.json·pred.csv·key.csv. 결과로 임계값을 튜닝하지 않는다."""
    items = run_calib(csv_path, out, seed)
    print(f"{len(items)}장 → {out}/sheet.html (pred.csv·key.csv는 평가자에게 주지 않음)")


if __name__ == "__main__":
    app()
