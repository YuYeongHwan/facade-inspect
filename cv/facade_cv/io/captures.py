"""촬영 기록(data/raw/videos/captures.csv) → building 단위 영상 목록 (건물 단위 train/val/test 분할용).

python -m facade_cv.io.captures --videos data/raw/videos
"""

import csv
from pathlib import Path

import typer

VIDEO_EXTS = {".mov", ".mp4", ".m4v", ".avi"}


def _rows(csv_path: Path) -> list[dict]:
    """BOM·CRLF, 엑셀이 행 전체를 따옴표로 감싼 형식("a,b,c")도 읽는다."""
    lines = csv_path.read_text(encoding="utf-8-sig").splitlines()
    unwrapped = [r[0] if len(r) == 1 else line for line, r in zip(lines, csv.reader(lines))]
    return list(csv.DictReader(unwrapped))


def load_captures(videos_dir: str | Path, csv_name: str = "captures.csv") -> dict[str, list[dict]]:
    """{building: [csv 행 + "path"(실제 영상 경로), ...]} — building/파일명 순 정렬.

    csv의 file은 파일명만 적으므로 videos_dir 아래 하위 폴더까지 대소문자 무시로 찾는다 (.mov ↔ .MOV).
    """
    videos_dir = Path(videos_dir)
    found: dict[str, list[Path]] = {}
    for p in videos_dir.rglob("*"):
        if p.suffix.lower() in VIDEO_EXTS:
            found.setdefault(p.name.lower(), []).append(p)

    by_building: dict[str, list[dict]] = {}
    for row in _rows(videos_dir / csv_name):
        row = {k.strip(): (v or "").strip() for k, v in row.items()}
        paths = found.get(row["file"].lower(), [])
        if len(paths) != 1:
            raise FileNotFoundError(f"{row['file']}: {len(paths)} matches under {videos_dir}")
        by_building.setdefault(row["building"], []).append({**row, "path": paths[0]})
    return {b: sorted(v, key=lambda r: r["file"]) for b, v in sorted(by_building.items())}


def main(videos: Path = typer.Option(Path("data/raw/videos"), exists=True, file_okay=False)) -> None:
    for building, rows in load_captures(videos).items():
        print(f"{building}: {len(rows)}")
        for r in rows:
            print(f"  {r['path']}  [{r['type']}] {r['note']}")


if __name__ == "__main__":
    typer.run(main)
