"""오염 score → 청소 우선순위 level 1~5 (config/grading.yaml).

python -m facade_cv.contamination.grade 0.07   →  3 가급적 청소
"""

from collections.abc import Mapping

import typer

from facade_cv.config import load_config


def compute_score(type_ratios: Mapping[str, float], cfg: dict | None = None) -> float:
    """score = Σ(type_ratio × weight)."""
    weights = (cfg or load_config("grading"))["weights"]
    return sum(type_ratios[t] * w for t, w in weights.items())


def cleaning_level(score: float, cfg: dict | None = None) -> tuple[int, str]:
    """score가 min_score 이상인 가장 높은 level의 (level, label)."""
    levels = [lv for lv in (cfg or load_config("grading"))["levels"] if score >= lv["min_score"]]
    if not levels:
        raise ValueError(f"score {score} below every level min_score")
    top = max(levels, key=lambda lv: lv["min_score"])
    return top["level"], top["label"]


def main(score: float) -> None:
    level, label = cleaning_level(score)
    print(level, label)


if __name__ == "__main__":
    typer.run(main)
