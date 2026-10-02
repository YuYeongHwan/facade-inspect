# Facade Inspect

드론 영상 기반 건물 창문 오염도 분석 웹 시스템. 설계 원칙·데이터 계약은 [CLAUDE.md](CLAUDE.md) 참조.

## 구조

```
cv/                 파이썬 패키지 facade_cv (uv, Python 3.11)
  facade_cv/
    schema.py       result.json 데이터 계약 (pydantic v2) — server/web의 기준
    device.py       torch 디바이스 선택 (cuda > mps > cpu)
    config.py       config/*.yaml 로더
    pipeline.py     영상 → result.json 전체 파이프라인 (미구현)
    io/extract.py   프레임 추출 (노출/블러/ORB 중복 필터) + Live용 iterate_video
    detect/
      augment.py    config/augment.yaml → Ultralytics train 인자
      datasets/     창문/외벽 분할 데이터셋 도구 (cmp_convert, label_studio/, merge_datasets)
    track/ rectify/ contamination/ recon3d/   (빈 패키지)
  tests/            pytest
  notebooks/        Colab 학습 노트북
config/             grading.yaml(등급), extract.yaml(프레임 필터), dataset.yaml(데이터셋), augment.yaml(학습 증강),
                    detect.yaml, recon.yaml
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
uv run --project cv python -m facade_cv.io.extract --video X --out runs/<id>/frames --fps 3 --debug [--flight-id ID]
```

## 프레임 추출 (facade_cv.io.extract)

target_fps로 샘플링 → 노출 필터(과/저노출 픽셀 비율) → 블러 필터(Laplacian variance가 절대 임계값 `blur.threshold`
미만이면서 ±`blur.window_sec`초 안에서 하위 `blur.percentile`% 미만일 때만 제거, null인 조건은 끔) → 중복 필터(ORB 호모그래피로 직전 선택
프레임과의 겹침 계산, 겹침 70~80% 유지: `dedup.min_motion`/`max_motion`).
겹침이 max_motion을 넘으면 직전 중복 후보, 그래도 부족하면 그 구간 블러 제거 프레임 중 가장 덜 흐린 것을 살린다(`rescued`).
매칭 수 부족으로 이동량을 알 수 없으면(`motion: null`, 장면 전환·무늬 없는 벽) 버리지 않고 저장한다.
결과는 `frames/<flightID>__frame_<index>.jpg` + `frames/frames.json`
(flightID는 `--flight-id`, 기본은 영상 파일명 — 데이터셋 분할의 그룹 키가 된다. 프레임별 timestamp, blur, blur_cut, 노출 비율, motion, selected, rescued, reason). `--debug`면 `debug/frames_grid.jpg`
(초록=선택, 하늘=rescued, 빨강=블러, 주황=노출, 회색=중복).

## 창문/외벽 분할 데이터셋 (facade_cv.detect.datasets)

클래스: `0=window`, `1=facade` (Ultralytics YOLO-seg polygon 포맷, `config/dataset.yaml`의 `names`).
각 도구의 출력은 `<dir>/images/*`, `<dir>/labels/*.txt`이고, `merge_datasets`가 이것들을 합쳐 분할한다.

### 1. CMP Facade 변환

CMP Facade Database (Tyleček & Šára, GCPR 2013, http://cmp.felk.cvut.cz/~tylecr1/facade/) — 연구 목적 사용, 인용 필요.

```bash
mkdir -p data/cmp && cd data/cmp
curl -LO https://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_base.zip       # 378장, 36MB
curl -LO https://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_extended.zip   # 228장, 20MB
unzip -q CMP_facade_DB_base.zip && unzip -q CMP_facade_DB_extended.zip && cd ../..
uv run --project cv python -m facade_cv.detect.datasets.cmp_convert --src data/cmp --out data/yolo/cmp
```

- 라벨 PNG는 팔레트(P) 모드, 픽셀값 = `label_names.txt`의 label_id(1~12). PIL로 index를 읽는다 (cv2.imread 금지)
- window → **XML 창문 사각형 하나 ∩ (window ∪ blind 픽셀) = 인스턴스 하나**. 붙은 창문은 사각형 단위로 분리되고,
  덧문으로 갈라진 H자 창문은 사각형으로 복원된다. 사각형 밖 blind와 사각형 밖 window 픽셀(전체의 ~1.9%)은 창문이 아니다.
  XML의 `<x>`는 세로, `<y>`는 가로 좌표다 (라벨 PNG와 대조해 확인)
- base 378장 기준 XML 창문 수 대비 90% 미만인 이미지 21장: 대부분 `min_area_ratio`(0.0005)보다 작은 먼 창문,
  나머지는 PNG에서 balcony/deco에 완전히 가려진 창문. `min_area_ratio: 0.0001`이면 7장
- facade, molding, cornice, pillar, sill, deco, blind → facade 하나로 합침 (door/shop/balcony/background 제외).
  YOLO polygon은 구멍을 표현 못 하므로 외곽을 채운 polygon이고, 창문은 그 위에 겹친다 (`overlap_mask`로 작은 인스턴스가 우선)
- 매핑은 `config/dataset.yaml`의 `cmp.window` / `cmp.facade`에서 변경

### 2. Label Studio (자체 드론 데이터)

1. 프로젝트 생성 → Labeling Interface에 `cv/facade_cv/detect/datasets/label_studio/labeling_config.xml` 붙여넣기
2. (선택) 사전 라벨링 — 학습된 모델이 있으면 예측을 import해 라벨러는 수정만 한다:
   ```bash
   uv run --project cv python -m facade_cv.detect.datasets.label_studio.prelabel \
       --weights models/window-seg.pt --images data/own/raw --out data/own/prelabel.json \
       --url-prefix "/data/local-files/?d=raw/"   # Label Studio가 이미지를 찾는 URL. 업로드 방식이면 맞게 변경
   ```
   Label Studio에서 `prelabel.json`을 Import (이미지는 local files storage 등으로 연결)
3. 라벨링 후 Export → **JSON** 형식으로 받아 변환:
   ```bash
   uv run --project cv python -m facade_cv.detect.datasets.label_studio.export_to_yolo \
       --export export.json --images data/own/raw --out data/yolo/own
   ```
   이미지는 파일명으로 찾는다 (업로드 시 붙는 `8f3a2b1c-` 접두사는 무시). 취소된 annotation과 미라벨 task는 건너뛰고,
   annotation이 여러 개면 가장 최근 것을 쓴다.

### 3. 병합·분할

```bash
uv run --project cv python -m facade_cv.detect.datasets.merge_datasets \
    --cmp data/yolo/cmp --own data/yolo/own --out data/yolo/merged   # --own 여러 번 가능, 재실행은 --overwrite
```

- CMP: train 0.9 / val 0.1 / test 0 — test는 자체 데이터로만 평가
- 자체 데이터: train 0.6 / val 0.2 / test 0.2, val·test에 최소 1그룹씩 보장 (그룹이 2개 미만이면 에러)
- **그룹 단위 분할**: 같은 비행의 연속 프레임이 train/test로 나뉘면 성능이 부풀려진다.
  자체 이미지 파일명은 `<flightID>__<frame>.jpg`여야 하고 (`split.own_group_pattern`, `io.extract` 출력 그대로),
  아니면 에러로 중단한다. `--allow-ungrouped`로만 우회 (그 이미지는 혼자 한 그룹)
- 출력 파일명은 `cmp__<원본>` / `<own 디렉토리명>__<원본>`, `data.yaml` 생성

### 4. 학습 증강 정책 (config/augment.yaml)

```python
from ultralytics import YOLO
from facade_cv.detect.augment import train_augment_args
YOLO("yolo11s-seg.pt").train(data="data/yolo/merged/data.yaml", **train_augment_args())
```

| 증강 | 설정 | 근거 |
|---|---|---|
| 밝기 | `hsv_v: 0.4` (Ultralytics) | 역광·그늘, 시간대별 조도 |
| 대비 | `RandomBrightnessContrast(contrast_limit=0.3, p=0.5)` (albumentations) | 흐린 날·헤이즈. 밝기는 hsv_v가 담당하므로 brightness_limit=0 |
| 원근 | `perspective: 0.0005`, `degrees: 5`, `shear: 2` (Ultralytics) | CMP는 정면 정사영 → 드론 사선 촬영과의 도메인 차이 보완 |
| 모션블러 | `MotionBlur(blur_limit=[3, 11], p=0.3)` (albumentations) | 드론 이동·진동 |
| 기타 | `fliplr: 0.5`, `flipud: 0`, `mosaic: 1.0`, `close_mosaic: 10` | 건물은 위아래 고정 |

대비·모션블러는 Ultralytics 하이퍼파라미터에 없어서 `augmentations` 인자(albumentations 객체)로 넘긴다.
여기에는 픽셀 전용 변환만 둔다 — 기하 변환은 Ultralytics가 polygon과 함께 처리해야 라벨이 어긋나지 않는다.

## 목업 (fixtures/)

박스 건물(12×24×8, 상대 단위) 외벽 4면(F1~F4), 창문 20개(등급 A8/B6/C4/D2).
오염 polygon은 래스터 면적으로 `type_ratios`·`score`를 계산하고 `config/grading.yaml`로 등급을 매긴다.
`result.json` 안의 경로는 모두 `fixtures/` 기준 상대경로다.
