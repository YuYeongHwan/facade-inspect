from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def load_config(name: str) -> dict:
    """config/<name>.yaml 로드."""
    return yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


def class_names(version: str | None = None) -> list[str]:
    """config/classes.yaml의 클래스 이름 (YOLO id 순서). version=None이면 active."""
    cfg = load_config("classes")
    return list(cfg["sets"][version or cfg["active"]])
