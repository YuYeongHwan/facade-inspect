"""CLIP 제로샷 오염 점수 p(dirty) [R35][R36] — 창문 크롭 1장 → 0~1. config/contamination.yaml clip (고정 프롬프트 6개)

p(dirty) = softmax(온도 × [cos(이미지, 더러운 창 프롬프트 평균), cos(이미지, 깨끗한 창 프롬프트 평균)])[0]
CSI·level과 별개인 참고 점수 (result.json clip_dirty). 선택 의존성: `uv sync --extra clip` — 없으면 None.
한계: 물방울·반사도 "더럽다"로 볼 수 있고, 위치·면적은 알려주지 않는다 (docs/scoring.md "CLIP 제로샷")

python -m facade_cv.contamination.clip_score <이미지...>
"""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import typer

from facade_cv.config import load_config
from facade_cv.device import get_device


@lru_cache(maxsize=2)
def _model(name: str, pretrained: str, dirty: tuple[str, ...], clean: tuple[str, ...]):
    """(model, preprocess, 클래스 텍스트 임베딩 (2, d), device) 또는 open_clip 없으면 None."""
    try:
        import open_clip
        import torch
    except ImportError:
        return None
    dev = get_device()
    model, _, pre = open_clip.create_model_and_transforms(name, pretrained=pretrained, device=dev)
    model.eval()
    tok = open_clip.get_tokenizer(name)
    with torch.no_grad():
        cls = []
        for prompts in (dirty, clean):
            e = model.encode_text(tok(list(prompts)).to(dev))
            m = (e / e.norm(dim=-1, keepdim=True)).mean(0)  # 프롬프트 앙상블 [R35]
            cls.append(m / m.norm())
    return model, pre, torch.stack(cls), dev


def clip_dirty(images: list[np.ndarray], cfg: dict | None = None) -> list[float | None]:
    """BGR 이미지들 → p(dirty) 목록. clip.enabled=false거나 open_clip이 없으면 전부 None."""
    c = (cfg or load_config("contamination"))["clip"]
    m = _model(c["model"], c["pretrained"], tuple(c["dirty"]), tuple(c["clean"])) if c["enabled"] else None
    if m is None:
        return [None] * len(images)
    import torch
    from PIL import Image

    model, pre, text, dev = m
    out = []
    with torch.no_grad():
        for img in images:
            x = pre(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(dev)
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            out.append(round(float((c["temperature"] * f @ text.T).softmax(-1)[0, 0]), 4))
    return out


def main(paths: list[Path] = typer.Argument(..., exists=True, dir_okay=False)) -> None:
    for p, s in zip(paths, clip_dirty([cv2.imread(str(p)) for p in paths])):
        print(f"{s if s is None else f'{s:.3f}'}  {p}")


if __name__ == "__main__":
    typer.run(main)
