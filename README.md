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
    io/captures.py  data/raw/videos/captures.csv → building 단위 영상 목록 (건물 단위 split용)
    detect/
      augment.py    config/augment.yaml → Ultralytics train 인자
      segmenter.py  WindowSegmenter — YOLO-seg 래퍼 (config/detect.yaml weights) → Detection(cls, mask, polygon, bbox, conf)
      predict.py    이미지 폴더 예측 → 마스크 오버레이 (io.extract 출력 폴더 그대로 입력)
      postprocess.py 마스크 → 4점 사각형 → 외벽 필터 → 격자 정규화 (row/col/flags)
      run.py        검출 + 후처리 CLI → detect/detections.json (+ --debug 단계별 시각화)
      datasets/     창문/외벽 분할 데이터셋 도구 (cmp_convert, label_studio/, merge_datasets)
    rectify/homography.py  소실점 두 개로 외벽 정면화 호모그래피 (detect 후처리·T6 공용)
    contamination/grade.py  score → 청소 우선순위 level 1~5 (config/grading.yaml)
    track/ recon3d/   (빈 패키지)
  tests/            pytest
  notebooks/        Colab 학습 노트북 (train_seg.ipynb — 사용법은 notebooks/README.md)
config/             grading.yaml(청소 우선순위 5단계), extract.yaml(프레임 필터), dataset.yaml(데이터셋), augment.yaml(학습 증강),
                    detect.yaml(추론·후처리), rectify.yaml(정면화), recon.yaml
fixtures/           목업 run 디렉토리 (result.json, building.glb, facades/, windows/)
  make_mock.py      목업 재생성 스크립트
server/             FastAPI (미구현)
web/                Next.js 대시보드 — 검사 결과 3D·창문 상세, 검출 뷰어 (실행법은 web/README.md)
data/ runs/ cv/weights/ (gitignore — 가중치·MODEL_CARD.md는 cv/weights/)
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
uv run --project cv python -m facade_cv.io.captures --videos data/raw/videos                # building별 영상 목록
uv run --project cv python -m facade_cv.detect.predict --images runs/<id>/frames   # 가중치 기본: config/detect.yaml weights
uv run --project cv python -m facade_cv.detect.run --frames runs/<id>/frames --debug  # 또는 --image X.jpg
uv run --project cv python -m facade_cv.rectify.homography --image X.jpg           # 이미지 전체 정면화
uv run --project cv python -m facade_cv.contamination.grade 0.07                  # score → "3 가급적 청소"
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
아이폰 회전 메타데이터(display matrix)는 OpenCV(FFmpeg 백엔드)가 디코딩 시 적용하므로 세로 촬영 영상도 눕지 않는다.
카메라가 거의 정지한 영상은 이동량이 min_motion 미만이라 대부분 duplicate로 빠지는 것이 정상이다 (COLMAP엔 시차가 필요).

### 촬영 기록 (data/raw/videos/captures.csv)

`file,building,type,date,weather,height_m,note`. 영상은 `data/raw/videos/<building폴더>/` 아래 어디든 두면 되고,
`load_captures(videos_dir)`가 파일명을 대소문자 무시로 찾아 `{building: [행 + path]}`를 돌려준다.
엑셀이 저장한 형식(BOM, 행 전체가 따옴표로 감싸짐)도 읽는다.

## 창문 검출 후처리 (facade_cv.detect.run)

YOLO-seg는 conf × `fill_conf_ratio`까지 낮춰 한 번 돌리고, conf 미만 검출은 격자 빈칸 보완 후보로만 쓴다.

1. **quad**: 마스크 윤곽 → approxPolyDP 4점(볼록), 안 되면 minAreaRect. 꼭짓점 TL,TR,BR,BL.
   `min_area_px` 미만(`small`), 직사각형도 `min_rectangularity` 미만(`not_rect`)은 제거
2. **외벽 필터**: 외벽 마스크(창문 자리 구멍을 메우고 `facade_filter.margin_ratio`만큼 팽창) 밖 창문 제거(`outside_facade`).
   외벽이 없거나, 창문 bbox 합집합보다 작거나, 창문 `max_outside` 비율 이상이 밖이면 외벽 검출 실패로 보고 건너뛴다(경고)
3. **격자**: 외벽(실패 시 창문 볼록 껍질) 안 LSD 직선 → 수평·수직 소실점 → 정면화 (`config/rectify.yaml`, 실패 시 원본 좌표).
   창문 중심 y/x를 1D 클러스터링해 row/col. 제거하지 않고 flag만: `off_grid`(행·열 모두 혼자), `odd_width`/`odd_height`
   (같은 열 폭/행 높이 중앙값과 `size_tol` 이상 차이). 빈칸에 저신뢰 검출이 있으면 `fill_candidates`(창문 수·`windows`와 별도), 없으면 `empty_cells`.
   점유율이 `grid.min_occupancy` 미만(코너 건물처럼 여러 면이 섞임)이면 빈칸·보완 후보는 만들지 않는다(경고 `grid_unreliable`)

출력 `<frames>/../detect/detections.json` — 이미지마다 counts(final은 보완 후보 제외), windows(corners, conf, row, col, flags),
fill_candidates, removed(reason), empty_cells, warnings. `--debug`: `debug/detect/<stem>_stages.jpg`(초록 OK, 주황 flag, 하늘 fill, 빨강 quad 제거,
자홍 외벽 밖, 노랑 외벽 영역), `_rectified.jpg`(정면화 + row/col + 빈칸 X), `_lines.jpg`(직선 후보 회색, 소실점 inlier 파랑/빨강).
row/col은 프레임 단위 임시값 — 최종 창문 ID는 3D 외벽 평면에서 정한다.

## 창문/외벽 분할 데이터셋 (facade_cv.detect.datasets)

클래스: `0=window`, `1=facade` (Ultralytics YOLO-seg polygon 포맷, `config/dataset.yaml`의 `names`).
각 도구의 출력은 `<dir>/images/*`, `<dir>/labels/*.txt`이고, `merge_datasets`가 이것들을 합쳐 분할한다.

### 1. CMP Facade 변환

CMP Facade Database (Tyleček & Šára, GCPR 2013, http://cmp.felk.cvut.cz/~tylecr1/facade/) — 연구 목적 사용, 인용 필요.

```bash
mkdir -p data/cmp && cd data/cmp
curl -LO https://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_base.zip       # 378장, 36MB
curl -LO https://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_extended.zip   # 228장, 20MB
unzip -qo CMP_facade_DB_base.zip && unzip -qo CMP_facade_DB_extended.zip && cd ../..   # -o: 공통 label_names.txt 덮어쓰기
uv run --project cv python -m facade_cv.detect.datasets.cmp_convert --src data/cmp --out data/yolo/cmp
```

- 라벨 PNG는 팔레트(P) 모드, 픽셀값 = `label_names.txt`의 label_id(1~12). PIL로 index를 읽는다 (cv2.imread 금지)
- window → **XML 창문 사각형 하나 ∩ (window ∪ blind 픽셀) = 인스턴스 하나**. 붙은 창문은 사각형 단위로 분리되고,
  덧문으로 갈라진 H자 창문은 사각형으로 복원된다. 사각형 밖 blind와 사각형 밖 window 픽셀(전체의 ~1.9%)은 창문이 아니다.
  XML의 `<x>`는 세로, `<y>`는 가로 좌표다 (라벨 PNG와 대조해 확인)
- 작은 인스턴스 제거는 클래스별 `min_area_ratio` (window 0.0001 — 먼 창문도 살림, facade 0.0005)
- 검증 (XML 창문 사각형 수 대비 변환된 창문 수, 창문 없는 이미지 제외):
  base 378장 — 90% 미만 7장, 창문 12161/12222 (99.5%) / extended 228장 — 90% 미만 9장, 7737/7823 (98.9%).
  빠지는 창문은 PNG에서 balcony/deco/shop에 완전히 가려진 경우 (예: cmp_x0173은 창문 10개가 shop 라벨 아래)
- 매핑은 `config/dataset.yaml`의 `cmp.window` / `cmp.facade`에서 변경

### 2. Label Studio (자체 드론 데이터)

1. 프로젝트 생성 → Labeling Interface에 `cv/facade_cv/detect/datasets/label_studio/labeling_config.xml` 붙여넣기
2. (선택) 사전 라벨링 — 학습된 모델이 있으면 예측을 import해 라벨러는 수정만 한다:
   ```bash
   uv run --project cv python -m facade_cv.detect.datasets.label_studio.prelabel \
       --weights cv/weights/seg_window_facade_v1.pt --images data/own/raw --out data/own/prelabel.json \
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

Colab 학습은 `cv/notebooks/train_seg.ipynb` ([사용법](cv/notebooks/README.md)). 로컬에서 돌릴 때:

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

박스 건물(12×24×8, 상대 단위) 외벽 4면(F1~F4), 창문 20개(청소 우선순위 level 1~5 = 6/5/4/3/2).
오염 polygon은 래스터 면적으로 `type_ratios`·`score`를 계산하고 `contamination/grade.py`(`config/grading.yaml`)로 level을 매긴다.
`cv/tests/test_schema.py`가 같은 result.json을 pydantic과 web zod(`web/scripts/validate-result.mjs`) 양쪽으로 검증한다.
`result.json` 안의 경로는 모두 `fixtures/` 기준 상대경로다.
