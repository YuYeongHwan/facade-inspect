# notebooks

## train_seg.ipynb — 창문/외벽/장애물 분할 학습 v2 (Colab)

YOLO11s-seg 2단계, 클래스 v2 (`config/classes.yaml`: window, facade, vegetation, wire, pole, door):
Stage1 공개 데이터(CMP·ADE20K·TTPLA·Roboflow, ≤3,000장, imgsz 640) → Stage2 공개 표본 1,000장 + 자체 데이터(train ×3) 미세조정 (imgsz 1024).
자체 test셋으로 클래스별 mask mAP50 / mAP50-95와 창문 개수 정확도를 내고, `seg_window_facade_v2.pt`와 모델 카드(학습 데이터 출처 [R번호])를 드라이브에 남긴다.
자체 데이터가 아직 없으면 `STAGE1_ONLY = True`로 공개 데이터만 학습할 수 있다.

### 학습 시간 (무료 Colab T4, 세션 끊김 방지)

v1 실측 CMP ~550장 × 100에폭 @1024 = 2.1시간 → **이미지·에폭당 0.137초 @1024**, imgsz에는 픽셀 수(제곱)로 비례한다고 본다.
3번 셀이 실제 장수로 같은 식을 계산해 출력하고 Stage당 `MAX_STAGE_HOURS`(3시간)를 넘으면 경고한다.

| Stage | 데이터 | imgsz | epochs (patience) | 예상 |
|---|---|---|---|---|
| 1 | 공개 ≤3,000장 | 640 | 35 (10) | 3,000 × 35 × 0.137 × 0.39 ≈ **1.6시간** |
| 2 | 공개 표본 1,000 + 자체 train ~90 × 3 | 1024 | 15 (5) | 1,270 × 15 × 0.137 ≈ **0.7시간** |

val·AutoBatch 시간은 빠져 있어 실제는 10~20% 더 걸릴 수 있다. 조기종료(patience)되면 더 짧다.
넘을 것 같으면 `STAGE1["epochs"]` 또는 `config/dataset.yaml balance` 상한을 줄인다.

### 1. 로컬: 데이터셋 zip 만들기 (레포 루트에서)

공개 데이터는 `data/external/<이름>/`에 받아 둔다 (루트 README "0. 공개 데이터"). 없는 것은 변환기가 안내만 출력하고 건너뛴다.

```bash
# 변환 (각각 data/yolo/<이름>/)
uv run --project cv python -m facade_cv.detect.datasets.cmp_convert --src data/cmp --out data/yolo/cmp
uv run --project cv python -m facade_cv.detect.datasets.ade20k_convert     # data/external/ade20k → data/yolo/ade20k
uv run --project cv python -m facade_cv.detect.datasets.ttpla_convert      # data/external/ttpla → data/yolo/ttpla
uv run --project cv python -m facade_cv.detect.datasets.roboflow_convert   # data/external/roboflow/<키> → data/yolo/roboflow/<키>
# 자체 (Label Studio JSON export, 라벨러 2명 export를 각각 변환해 같은 폴더로)
uv run --project cv python -m facade_cv.detect.datasets.label_studio.export_to_yolo \
    --export export_batch1.json --images data/own/b04_obstacle/raw --out data/yolo/own   # batch2도 같은 --out

# 병합 (균형 상한·분할) → zip
uv run --project cv python -m facade_cv.detect.datasets.merge_datasets --out data/yolo/merged_v2 \
    --cmp data/yolo/cmp --ext ade20k=data/yolo/ade20k --ext ttpla=data/yolo/ttpla \
    --ext roboflow_building_facade=data/yolo/roboflow/building_facade \
    --ext roboflow_facade_test=data/yolo/roboflow/facade_test \
    --ext roboflow_window_segment=data/yolo/roboflow/window_segment \
    --own data/yolo/own
cat data/yolo/merged_v2/stats.json   # 소스·클래스별 장수 확인
(cd data/yolo && zip -qr merged_v2.zip merged_v2)
```

`--own` 없이 만들면 자체 test가 없어 `STAGE1_ONLY = True`로만 돌릴 수 있다 (`STAGE1_ONLY = False`면 3번 셀에서 멈춤).
나중에 자체 데이터가 생기면 Stage1을 다시 돌리지 않고 `STAGE1_WEIGHTS = ".../runs/<그 실행>_stage1/weights/best.pt"`
(또는 같은 `RUN_NAME` + `RESUME = True`)로 Stage2만 돌린다.

### 2. Google Drive 배치

```
MyDrive/facade-inspect/
  datasets/merged_v2.zip  ← 업로드 (노트북 DATASET_ZIP을 이 이름으로)
  weights/                ← 노트북이 seg_window_facade_v2.pt, MODEL_CARD.md 생성 → 로컬 cv/weights/로 내려받는다
  runs/                   ← 학습 결과(Ultralytics project): <RUN_NAME>_stage1/, <RUN_NAME>_stage2/,
                            <RUN_NAME>_eval/, <RUN_NAME>_pred_grid.jpg — 드라이브라 세션이 끊겨도 남는다
```

경로가 다르면 노트북 맨 위 하이퍼파라미터 셀의 `DRIVE_ROOT`, `DATASET_ZIP`을 고친다.

### 3. Colab 실행

1. Colab → 파일 → 노트북 업로드 → `cv/notebooks/train_seg.ipynb` (또는 GitHub에서 열기)
2. 런타임 → 런타임 유형 변경 → **GPU** (T4)
3. 맨 위 하이퍼파라미터 셀 확인 (`DATASET_ZIP = f"{DRIVE_ROOT}/datasets/merged_v2.zip"`) 후 **런타임 → 모두 실행**. 드라이브 마운트 권한 팝업 승인

| 섹션 | 내용 |
|---|---|
| 0 | 하이퍼파라미터 — 경로, `RESUME`/`RUN_NAME`, `STAGE1_ONLY`, Stage1/2 설정, `STAGE2_PUBLIC_MAX`, 시간 예산, 증강, 평가. **여기만 고친다** |
| 1 | 드라이브 마운트, `pip install ultralytics albumentations` (버전은 로컬 `uv.lock`과 같게 고정) |
| 2 | 헬퍼 함수 |
| 3 | zip 압축 해제, Stage별 이미지 목록·data yaml, 출처(`sources.json`), **Stage별 예상 학습 시간** |
| 4–5 | Stage1, Stage2 학습 (조기종료 `patience`). `RESUME`이면 끝난 Stage는 건너뛰고 중단된 Stage는 이어서 |
| 6 | 클래스별 mask P/R/mAP50/mAP50-95 — 자체 test셋 Stage1 vs Stage2 (`STAGE1_ONLY`: 공개 val, Stage1만) |
| 7 | 창문 개수 정확도 (6과 같은 대상) |
| 8 | 최종 모델 예측 오버레이 16장 그리드 → `runs/<RUN_NAME>_pred_grid.jpg` |
| 9 | 최종 best.pt → 드라이브 `weights/seg_window_facade_v2.pt` (이미 있으면 멈춤 — 덮어쓰지 않음), `weights/MODEL_CARD.md`에 학습 데이터 출처 [R번호]·라이선스·장수, 설정, 지표 append |

### 세션이 끊겼을 때 — `RESUME = True`

Ultralytics가 **매 에폭** `runs/<RUN_NAME>_stage*/weights/last.pt`를 드라이브에 덮어쓴다. 새 세션에서
하이퍼파라미터 셀의 **`RESUME = True`** 로 바꾸고 **모두 실행**하면:

- `RUN_NAME = None`이면 `runs/`에서 가장 최근 실행을 찾는다 (특정 실행은 `RUN_NAME = "20261002_1052"`처럼 지정)
- Stage별로 `last.pt`를 보고 판단 — 끝남(종료 시 `epoch=-1`로 저장) → 건너뜀, 중단 → `last.pt`부터 `train(resume=True)`, 없음 → 새로 학습
- Stage1이 끝나고 Stage2 중 끊겼으면 Stage1은 건너뛰고 Stage2만 이어간다
- 이미 저장(9번 셀)까지 끝난 실행을 다시 돌리면 모델을 중복 저장하지 않고 멈춘다

`RESUME = False`인데 같은 `RUN_NAME`에 결과가 있으면 덮어쓰지 않고 에러로 멈춘다.

### batch

기본 `batch=0.7` — GPU 메모리의 70%를 쓰는 배치 크기를 Ultralytics AutoBatch가 학습 시작 시 계산한다
(단일 CUDA GPU 전용, CPU/MPS는 16으로 대체). 고정하려면 정수로. 평가는 소수 batch를 지원하지 않아 `EVAL_BATCH`(16)를 따로 쓴다.

### 설계 메모

- **Stage1 imgsz 640**: 공개 데이터가 5배로 늘어 1024로는 T4 3시간을 넘는다. 공개 데이터로 클래스 개념을 먼저 배우고,
  자체 데이터 해상도(1024)는 Stage2에서 맞춘다
- **Stage2 공개 표본**: 공개 전체를 다시 돌리면 길고 자체 비중이 작아진다 → seed 고정 무작위 1,000장 + 자체 ×3
- **오버샘플링**: Stage2 train 목록에 자체 이미지 경로를 3번 쓴다 (Ultralytics는 목록 중복을 제거하지 않음 — 테스트가 확인)
- **val**: Stage1은 공개 val, Stage2는 공개 val + 자체 val. **test**: 자체 test만 (merge가 공개 test를 0으로 강제)
- **공개/자체 구분**: 파일명 접두사 `pub_<소스>__` (구 zip의 `cmp__`도 공개로 본다)
- **증강** `AUGMENT`는 `config/augment.yaml`과 같아야 한다 (테스트가 확인)
- `MAX_DET=500`: CMP에 창문 300개 넘는 이미지가 있어 Ultralytics 기본값(300)보다 크게

### 로컬 검증

```bash
cd cv && uv run pytest tests/test_notebook.py
```

nbformat 스키마 검증, 모든 코드 셀 파이썬 문법 검사, 하이퍼파라미터·헬퍼 셀 실행
(Stage 목록 생성, 공개 표본·오버샘플링, 예상 시간 식과 3시간 예산, 개수 지표, 모델 카드 포맷·출처, 재개 상태 판정)을 확인한다.
학습 셀은 실행하지 않는다.

## 학습된 가중치로 예측 (로컬)

드라이브 `facade-inspect/weights/`의 `seg_window_facade_v{N}.pt`와 `MODEL_CARD.md`를 레포의 `cv/weights/`에 받는다
(gitignore). 기본 가중치는 `config/detect.yaml`의 `weights` — 새 버전을 쓰려면 여기를 고친다.

```bash
# 영상 → 프레임 → 예측 오버레이. --images에 io.extract 출력 폴더를 그대로 (frames.json 등 이미지 아닌 파일은 무시)
uv run --project cv python -m facade_cv.io.extract --video input.mp4 --out runs/insp_001/frames
uv run --project cv python -m facade_cv.detect.predict \
    --images runs/insp_001/frames   # → runs/insp_001/pred/*.jpg (다른 가중치: --weights cv/weights/seg_window_facade_v1.pt)
```

디바이스는 cuda > mps > cpu 자동 (`--device cpu`로 지정 가능), 추론 설정은 `config/detect.yaml`
(imgsz는 학습과 같은 1024). 오버레이는 클래스별 색 마스크 + 왼쪽 위에 `window N  facade M`.
