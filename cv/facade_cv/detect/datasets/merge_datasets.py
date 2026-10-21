"""공개(CMP·ADE20K·TTPLA·Roboflow) + 자체 YOLO-seg 데이터 → 장수 균형 + train/val/test 분할 + data.yaml.

python -m facade_cv.detect.datasets.merge_datasets --cmp data/yolo/cmp --ext ade20k=data/yolo/ade20k \
    --ext ttpla=data/yolo/ttpla --ext roboflow_building_facade=data/yolo/roboflow/building_facade \
    --own data/yolo/own --out data/yolo/merged
입력은 각각 <dir>/images, <dir>/labels (변환기 출력). --ext/--own은 여러 번 지정 가능.
- 공개 데이터: config balance 상한 (소스별 max_images — roboflow_*는 roboflow로 합산 — 와 전체 max_total). 상한 안에서는
  희귀 클래스를 가진 이미지부터 남긴다. test 비율은 0이어야 한다 → 시험셋은 자체 촬영만.
  출력 파일명 pub_<소스>__<원본> (노트북이 접두사로 공개/자체를 가른다)
- 자체 데이터: 그룹 단위 분할 (split.own_group_by — flight: 영상 하나 / segment: 영상 안 시간 구간), val/test에 반드시
  1그룹 이상. segment면 다른 split인 이웃 구간 경계 ±guard_frames 이미지는 버린다.
  패턴에 안 맞는 파일명이 있으면 에러 — --allow-ungrouped면 그 이미지는 혼자 한 그룹.
출력: data.yaml, stats.json (split×소스별 이미지 수, 클래스별 이미지·인스턴스 수), sources.json (소스 → 참고문헌·라이선스·장수)
"""

import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path

import typer
import yaml

from facade_cv.config import class_names, load_config
from facade_cv.detect.datasets.yolo import list_images

SPLITS = ("train", "val", "test")
PUBLIC_PREFIX = "pub_"


def _split_counts(n: int, ratios: dict, ensure_eval: bool) -> dict[str, int]:
    val, test = round(n * ratios["val"]), round(n * ratios["test"])
    if ensure_eval:
        val, test = max(val, 1), max(test, 1)
    return {"train": n - val - test, "val": val, "test": test}


def _assign(groups: list[list[Path]], ratios: dict, rng: random.Random, ensure_eval: bool) -> dict[str, list[list[Path]]]:
    groups = groups[:]
    rng.shuffle(groups)
    counts, out, i = _split_counts(len(groups), ratios, ensure_eval), {}, 0
    for s in ("val", "test", "train"):  # 반올림 손실은 train이 흡수
        out[s] = groups[i : i + counts[s]]
        i += counts[s]
    return out


def own_group_key(stem: str, sp: dict) -> tuple[str, int | None] | None:
    """(flight, 구간 번호 또는 None). 그룹 패턴에 안 맞으면 None. segment인데 프레임 번호가 없으면 영상 하나가 한 그룹."""
    m = re.match(sp["own_group_pattern"], stem)
    if not m:
        return None
    if sp.get("own_group_by", "flight") == "flight":
        return m.group(1), None
    f = re.search(sp["own_frame_pattern"], stem)
    return m.group(1), int(f.group(1)) // sp["segment_frames"] if f else None


def own_groups(images: list[Path], sp: dict, allow_ungrouped: bool = False) -> dict[tuple, list[Path]]:
    keys = {p: own_group_key(p.stem, sp) for p in images}
    ungrouped = [p for p, k in keys.items() if k is None]
    if ungrouped and not allow_ungrouped:
        raise ValueError(
            f"{len(ungrouped)} own images don't match {sp['own_group_pattern']!r} (<flightID>__frame_<index>), e.g. "
            f"{[p.name for p in ungrouped[:3]]}. 같은 비행 프레임이 train/test로 갈려 성능이 부풀려진다. "
            "파일명을 고치거나 --allow-ungrouped로 우회"
        )
    groups: dict[tuple, list[Path]] = {}
    for img in images:
        groups.setdefault(keys[img] or (f"\0{img.stem}", None), []).append(img)  # \0: 그룹명과 충돌 방지
    return groups


def guard_drop(split_of: dict[tuple, str], img: Path, key: tuple, sp: dict) -> bool:
    """segment 분할에서 다른 split인 이웃 구간과의 경계 ±guard_frames 안이면 True (버림)."""
    flight, seg = key
    f = re.search(sp["own_frame_pattern"], img.stem) if seg is not None else None
    if not f:
        return False
    idx, n, g, me = int(f.group(1)), sp["segment_frames"], sp["guard_frames"], split_of[key]
    near_prev = idx - seg * n < g and split_of.get((flight, seg - 1), me) != me
    near_next = (seg + 1) * n - idx <= g and split_of.get((flight, seg + 1), me) != me
    return near_prev or near_next


def label_classes(label: Path) -> list[int]:
    if not label.exists():
        return []
    return [int(line.split()[0]) for line in label.read_text().splitlines() if line.split()]


def cap_key(source: str, caps: dict) -> str:
    """roboflow_building_facade → roboflow (caps에 소스 이름이 없으면 첫 _ 앞으로 합산)."""
    return source if source in caps else source.split("_", 1)[0]


def balance(public: dict[str, list[Path]], labels: dict[Path, list[int]], bc: dict, seed: int) -> dict[str, list[Path]]:
    """소스별·전체 상한 적용. 상한 안에서는 이미지가 가진 클래스 중 가장 희귀한(전체 인스턴스 빈도 최소) 클래스 빈도
    오름차순으로 남긴다 (같으면 seed 무작위)."""
    caps, rng = bc["max_images"], random.Random(seed)
    pools: dict[str, list[Path]] = {}
    for src, imgs in public.items():
        pools.setdefault(cap_key(src, caps), []).extend(imgs)
    freq = Counter(c for imgs in pools.values() for p in imgs for c in labels[p])
    limit = {k: len(v) if caps.get(k) is None else min(len(v), caps[k]) for k, v in pools.items()}
    fixed = sum(limit[k] for k in pools if caps.get(k) is None)
    rest = sum(limit[k] for k in pools if caps.get(k) is not None)
    room = max(bc["max_total"] - fixed, 0)
    if rest > room:  # 전체 상한 초과 → 상한 있는 소스를 비율대로 줄임 (최대 나머지 방식으로 합 = room)
        capped = [k for k in pools if caps.get(k) is not None]
        quota = {k: limit[k] * room / rest for k in capped}
        limit |= {k: int(q) for k, q in quota.items()}
        for k in sorted(capped, key=lambda k: int(quota[k]) - quota[k])[: room - sum(limit[k] for k in capped)]:
            limit[k] += 1

    def rarity(p: Path) -> tuple[int, float]:
        return min((freq[c] for c in labels[p]), default=1 << 30), rng.random()

    keep = {p for k, imgs in pools.items() for p in sorted(imgs, key=rarity)[: limit[k]]}
    return {src: [p for p in imgs if p in keep] for src, imgs in public.items()}


def merge(
    cmp: str | Path | None,
    own: list[str | Path],
    out: str | Path,
    overwrite: bool = False,
    allow_ungrouped: bool = False,
    cfg: dict | None = None,
    ext: dict[str, str | Path] | None = None,
    names: list[str] | None = None,
) -> dict[str, dict[str, int]]:
    """{split: {source: 이미지 수}} 반환. ext: {소스 이름: 디렉토리} (cmp 외 공개 데이터)."""
    cfg = cfg or load_config("dataset")
    names = names or class_names()
    sp, bc, out = cfg["split"], cfg["balance"], Path(out)
    roots = {**({"cmp": Path(cmp)} if cmp else {}), **{k: Path(v) for k, v in (ext or {}).items()}}
    for src in roots:
        ratios = sp["cmp"] if src == "cmp" else sp["public"]
        if ratios["test"]:
            raise ValueError(f"split.{'cmp' if src == 'cmp' else 'public'}.test must be 0 — 시험셋은 자체 촬영 데이터만")
    own_images = [(Path(d), p) for d in own for p in list_images(Path(d) / "images")]
    groups = own_groups([p for _, p in own_images], sp, allow_ungrouped) if own else {}
    if own and len(groups) < 2:
        raise ValueError(f"own data needs >= 2 groups for val/test, got {len(groups)}")
    for sub in ("images", "labels"):
        if (out / sub).exists() and any((out / sub).iterdir()):
            if not overwrite:
                raise FileExistsError(f"{out / sub} is not empty (use overwrite=True / --overwrite)")
            shutil.rmtree(out / sub)

    rng = random.Random(sp["seed"])
    public = {src: list_images(root / "images") for src, root in roots.items()}
    labels = {p: label_classes(roots[src] / "labels" / f"{p.stem}.txt") for src, imgs in public.items() for p in imgs}
    public = balance(public, labels, bc, sp["seed"])
    plan: dict[str, list[tuple[str, str, Path, Path]]] = {s: [] for s in SPLITS}  # (source, prefix, src root, image)
    for src, imgs in public.items():
        for s, gs in _assign([[p] for p in imgs], sp["cmp"] if src == "cmp" else sp["public"], rng, False).items():
            plan[s] += [(src, f"{PUBLIC_PREFIX}{src}__", roots[src], g[0]) for g in gs]
    dropped = 0
    if own:
        root_of = {p: d for d, p in own_images}
        keys = list(groups)
        assigned = _assign([[k] for k in keys], sp["own"], rng, True)
        split_of = {g[0]: s for s, gs in assigned.items() for g in gs}
        for key, s in split_of.items():
            for img in groups[key]:
                if guard_drop(split_of, img, key, sp):
                    dropped += 1
                    continue
                plan[s].append(("own", f"{root_of[img].name}__", root_of[img], img))

    stats: dict[str, dict[str, int]] = {s: {} for s in SPLITS}
    cls_stats = {s: {n: {"images": 0, "instances": 0} for n in names} for s in SPLITS}
    for s, items in plan.items():
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)
        for src, prefix, root, img in items:
            name = prefix + img.name
            shutil.copy2(img, out / "images" / s / name)
            label = root / "labels" / f"{img.stem}.txt"
            if label.exists():  # 라벨 없는 이미지는 Ultralytics가 배경으로 취급
                shutil.copy2(label, out / "labels" / s / f"{Path(name).stem}.txt")
            stats[s][src] = stats[s].get(src, 0) + 1
            cls = Counter(label_classes(label))
            for c, k in cls.items():
                cls_stats[s][names[c]]["images"] += 1
                cls_stats[s][names[c]]["instances"] += k

    # test는 비어 있으면(자체 데이터 없이 공개만) 키를 빼서 빈 디렉토리를 가리키지 않게 한다
    splits = [s for s in SPLITS if s != "test" or stats["test"]]
    data = {"path": str(out.resolve()), **{s: f"images/{s}" for s in splits}, "names": dict(enumerate(names))}
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (out / "stats.json").write_text(json.dumps({"images": stats, "classes": cls_stats, "own_guard_dropped": dropped},
                                               ensure_ascii=False, indent=1), encoding="utf-8")
    meta = cfg.get("sources", {})
    sources = {src: {**meta.get(src, {"ref": "?", "license": "?"}), "images": sum(st.get(src, 0) for st in stats.values())}
               for src in [*public, *(["own"] if own else [])]}
    (out / "sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=1), encoding="utf-8")
    return stats


def parse_ext(items: list[str]) -> dict[str, Path]:
    out = {}
    for it in items:
        name, sep, path = it.partition("=")
        if not sep or not name or "__" in name:
            raise typer.BadParameter(f"--ext {it!r}: <이름>=<디렉토리> 형식, 이름에 __ 금지")
        out[name] = Path(path)
    return out


def main(
    out: Path = typer.Option(...),
    cmp: Path = typer.Option(None, exists=True, file_okay=False),
    ext: list[str] = typer.Option([], help="<소스 이름>=<디렉토리> (예: ade20k=data/yolo/ade20k). 없는 디렉토리는 건너뜀"),
    own: list[Path] = typer.Option([], exists=True, file_okay=False),
    classes: str = typer.Option(None, help="v1 | v2 (config/classes.yaml, 기본 active) — data.yaml names"),
    overwrite: bool = False,
    allow_ungrouped: bool = typer.Option(False, help="<flightID>__frame_<index> 형식이 아닌 자체 이미지를 각자 한 그룹으로 허용"),
) -> None:
    sources = parse_ext(ext)
    for k, d in list(sources.items()):
        if not (d / "images").is_dir():
            print(f"skip --ext {k}: {d / 'images'} 없음 (변환 전?)")
            del sources[k]
    for s, counts in merge(cmp, own, out, overwrite, allow_ungrouped, ext=sources, names=class_names(classes)).items():
        print(f"{s}: {counts}")
    print(f"→ {out / 'data.yaml'}, stats.json, sources.json")


if __name__ == "__main__":
    typer.run(main)
