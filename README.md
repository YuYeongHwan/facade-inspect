# Facade Inspect

드론 영상 기반 건물 창문 오염도 분석 웹 시스템. 설계 원칙·데이터 계약은 [CLAUDE.md](CLAUDE.md) 참조.

## 구조

```
cv/                 파이썬 패키지 facade_cv (uv, Python 3.11)
  facade_cv/
    schema.py       result.json 데이터 계약 (pydantic v2) — server/web의 기준
    device.py       torch 디바이스 선택 (cuda > mps > cpu)
    pipeline.py     영상 → result.json 전체 파이프라인 (미구현)
    io/ detect/ track/ rectify/ contamination/ recon3d/   (빈 패키지)
  tests/            pytest
  notebooks/        Colab 학습 노트북
config/             grading.yaml(등급 가중치·임계값), detect.yaml, recon.yaml
fixtures/           목업 run 디렉토리 (result.json, building.glb, facades/, windows/)
  make_mock.py      목업 재생성 스크립트
server/             FastAPI (미구현)
web/                Next.js (미구현)
data/ models/ runs/ (gitignore)
```

## 설치 / 테스트

```bash
cd cv
uv sync                      # open3d 포함: uv sync --extra open3d (macOS는 먼저 brew install libusb)
uv run pytest                # 무거운 테스트 제외: uv run pytest -m "not slow"
```

## CLI

```bash
uv run --project cv python -m facade_cv.schema fixtures/result.json   # result.json 검증
uv run --project cv python -m facade_cv.device                        # 사용할 디바이스 출력
uv run --project cv python fixtures/make_mock.py                      # 목업 재생성
```

## 목업 (fixtures/)

박스 건물(12×24×8, 상대 단위) 외벽 4면(F1~F4), 창문 20개(등급 A8/B6/C4/D2).
오염 polygon은 래스터 면적으로 `type_ratios`·`score`를 계산하고 `config/grading.yaml`로 등급을 매긴다.
`result.json` 안의 경로는 모두 `fixtures/` 기준 상대경로다.
