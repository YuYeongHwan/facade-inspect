"""CMP + 자체 YOLO-seg 데이터 → train/val/test 분할 + data.yaml.

python -m facade_cv.detect.datasets.merge_datasets --cmp data/yolo/cmp --own data/yolo/own --out data/yolo/merged
입력은 각각 <dir>/images, <dir>/labels (cmp_convert / export_to_yolo 출력). --own은 여러 번 지정 가능.
자체 데이터는 그룹(config split.own_group_pattern, 기본 <flightID>__<frame>) 단위로 나누고 val/test에 반드시
1그룹 이상 넣는다. 패턴에 안 맞는 파일명이 있으면 에러 — --allow-ungrouped면 그 이미지는 혼자 한 그룹.
"""

import random
import re
import shutil
from pathlib import Path

import typer
import yaml

from facade_cv.config import load_config
from facade_cv.detect.datasets.yolo import list_images

SPLITS = ("train", "val", "test")


def _split_counts(n: int, ratios: dict, ensure_eval: bool) -> dict[str, int]:
    val, test = round(n * ratios["val"]), round(n * ratios["test"])
    if ensure_eval:
        val, test = max(val, 1), max(test, 1)
    return {"train": n - val - test, "val": val, "test": test}


def _assign(groups: list[list[Path]], ratios: dict, rng: random.Random, ensure_eval: bool) -> dict[str, list[Path]]:
    groups = groups[:]
    rng.shuffle(groups)
    counts, out, i = _split_counts(len(groups), ratios, ensure_eval), {}, 0
    for s in ("val", "test", "train"):  # 반올림 손실은 train이 흡수
        out[s] = [p for g in groups[i : i + counts[s]] for p in g]
        i += counts[s]
    return out


def own_groups(images: list[Path], pattern: str, allow_ungrouped: bool = False) -> list[list[Path]]:
    ungrouped = [p for p in images if not re.match(pattern, p.stem)]
    if ungrouped and not allow_ungrouped:
        raise ValueError(
            f"{len(ungrouped)} own images don't match {pattern!r} (<flightID>__<frame>), e.g. "
            f"{[p.name for p in ungrouped[:3]]}. 같은 비행 프레임이 train/test로 갈려 성능이 부풀려진다. "
            "파일명을 고치거나 --allow-ungrouped로 우회"
        )
    groups: dict[str, list[Path]] = {}
    for img in images:
        m = re.match(pattern, img.stem)
        groups.setdefault(m.group(1) if m else f"\0{img.stem}", []).append(img)  # \0: 그룹명과 충돌 방지
    return list(groups.values())


def merge(
    cmp: str | Path | None,
    own: list[str | Path],
    out: str | Path,
    overwrite: bool = False,
    allow_ungrouped: bool = False,
    cfg: dict | None = None,
) -> dict[str, dict[str, int]]:
    """{split: {source: 이미지 수}} 반환."""
    cfg = cfg or load_config("dataset")
    sp, out = cfg["split"], Path(out)
    own_images = [(Path(d), p) for d in own for p in list_images(Path(d) / "images")]
    groups = own_groups([p for _, p in own_images], sp["own_group_pattern"], allow_ungrouped) if own else []
    if own and len(groups) < 2:
        raise ValueError(f"own data needs >= 2 groups for val/test, got {len(groups)}")
    for sub in ("images", "labels"):
        if (out / sub).exists() and any((out / sub).iterdir()):
            if not overwrite:
                raise FileExistsError(f"{out / sub} is not empty (use overwrite=True / --overwrite)")
            shutil.rmtree(out / sub)

    rng = random.Random(sp["seed"])
    plan: dict[str, list[tuple[str, Path, Path]]] = {s: [] for s in SPLITS}  # (prefix, src root, image)
    if cmp:
        cmp = Path(cmp)
        for s, imgs in _assign([[p] for p in list_images(cmp / "images")], sp["cmp"], rng, False).items():
            plan[s] += [("cmp__", cmp, p) for p in imgs]
    if own:
        root_of = {p: d for d, p in own_images}
        for s, imgs in _assign(groups, sp["own"], rng, True).items():
            plan[s] += [(f"{root_of[p].name}__", root_of[p], p) for p in imgs]

    stats: dict[str, dict[str, int]] = {s: {} for s in SPLITS}
    for s, items in plan.items():
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)
        for prefix, root, img in items:
            name = prefix + img.name
            shutil.copy2(img, out / "images" / s / name)
            label = root / "labels" / f"{img.stem}.txt"
            if label.exists():  # 라벨 없는 이미지는 Ultralytics가 배경으로 취급
                shutil.copy2(label, out / "labels" / s / f"{Path(name).stem}.txt")
            src = "cmp" if prefix == "cmp__" else "own"
            stats[s][src] = stats[s].get(src, 0) + 1

    data = {"path": str(out.resolve()), **{s: f"images/{s}" for s in SPLITS},
            "names": dict(enumerate(cfg["names"]))}
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return stats


def main(
    out: Path = typer.Option(...),
    cmp: Path = typer.Option(None, exists=True, file_okay=False),
    own: list[Path] = typer.Option([], exists=True, file_okay=False),
    overwrite: bool = False,
    allow_ungrouped: bool = typer.Option(False, help="<flightID>__<frame> 형식이 아닌 자체 이미지를 각자 한 그룹으로 허용"),
) -> None:
    for s, counts in merge(cmp, own, out, overwrite, allow_ungrouped).items():
        print(f"{s}: {counts}")
    print(f"→ {out / 'data.yaml'}")


if __name__ == "__main__":
    typer.run(main)
