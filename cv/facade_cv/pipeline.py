"""영상 → runs/<id>/result.json 전체 파이프라인 (미구현)."""

from pathlib import Path

import typer

app = typer.Typer()


@app.command()
def run(video: Path, out_dir: Path = Path("runs/insp_001"), debug: bool = False) -> None:
    raise NotImplementedError("detect → track → recon3d → rectify → contamination")


if __name__ == "__main__":
    app()
