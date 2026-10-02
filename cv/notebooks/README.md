# notebooks

## train_seg.ipynb — 창문/외벽 분할 학습 (Colab)

YOLO11s-seg 2단계 학습: Stage1 CMP만 → Stage2 CMP + 자체 데이터(자체 train ×3) 파인튜닝.
자체 test셋으로 window mask mAP50 / mAP50-95와 창문 개수 정확도를 내고, 가중치와 모델 카드를 드라이브에 남긴다.

### 1. 로컬: 데이터셋 zip 만들기

루트 README의 "창문/외벽 분할 데이터셋" 순서대로 만든 merge 결과를 zip한다. 자체 데이터가 없으면 노트북이 멈춘다
(test셋 = 자체 데이터).

```bash
uv run --project cv python -m facade_cv.detect.datasets.merge_datasets \
    --cmp data/yolo/cmp --own data/yolo/own --out data/yolo/merged
cd data/yolo && zip -qr merged.zip merged
```

### 2. Google Drive 배치

```
MyDrive/facade-inspect/
  datasets/merged.zip     ← 업로드
  models/                 ← 노트북이 seg_window_facade_v{N}.pt, MODEL_CARD.md 생성
  runs/                   ← 학습 결과(Ultralytics project): <RUN_NAME>_stage1/, <RUN_NAME>_stage2/,
                            <RUN_NAME>_eval/, <RUN_NAME>_pred_grid.jpg — 드라이브라 세션이 끊겨도 남는다
```

경로가 다르면 노트북 맨 위 하이퍼파라미터 셀의 `DRIVE_ROOT`, `DATASET_ZIP`을 고친다.

### 3. Colab 실행

1. Colab → 파일 → 노트북 업로드 → `cv/notebooks/train_seg.ipynb` (또는 GitHub에서 열기)
2. 런타임 → 런타임 유형 변경 → **GPU** (T4 이상)
3. 맨 위 하이퍼파라미터 셀 확인 후 **런타임 → 모두 실행**. 드라이브 마운트 권한 팝업 승인

| 섹션 | 내용 |
|---|---|
| 0 | 하이퍼파라미터 — 경로, Stage1/2 학습 설정, 증강, 평가 설정. **여기만 고친다** |
| 1 | 드라이브 마운트, `pip install ultralytics albumentations` (버전은 로컬 `uv.lock`과 같게 고정) |
| 2 | 헬퍼 함수 |
| 3 | zip 압축 해제(`/content/data`), Stage별 이미지 목록·data yaml 생성 |
| 4–5 | Stage1, Stage2 학습 (조기종료 `patience`). `RESUME`이면 끝난 Stage는 건너뛰고 중단된 Stage는 이어서 |
| 6 | 자체 test셋 window mask P/R/mAP50/mAP50-95, Stage1 vs Stage2 |
| 7 | 창문 개수 정확도: 이미지당 \|검출−GT\| (MAE, MAPE, ±10% 이내 비율, 오차 큰 이미지 10장) |
| 8 | 예측 오버레이 16장 그리드 (자체 test, 부족하면 자체 val로 채움) → `runs/<RUN_NAME>_pred_grid.jpg` |
| 9 | Stage2 best.pt → `models/seg_window_facade_v{N}.pt` (N 자동 증가), `models/MODEL_CARD.md`에 설정·지표 append. 같은 실행을 두 번 저장하지 않음 |

### 세션이 끊겼을 때 — `RESUME = True`

Ultralytics가 에폭마다 `runs/<RUN_NAME>_stage*/weights/last.pt`를 드라이브에 저장한다. 새 세션에서
하이퍼파라미터 셀의 **`RESUME = True`** 로 바꾸고 **모두 실행**하면:

- `RUN_NAME = None`이면 `runs/`에서 가장 최근 실행을 찾는다 (특정 실행은 `RUN_NAME = "20261002_1052"`처럼 지정)
- Stage별로 `last.pt`를 보고 판단 — 끝남(Ultralytics가 종료 시 `epoch=-1`로 저장) → 건너뜀,
  중단 → `last.pt`부터 `train(resume=True)` (설정·증강·배치 크기는 체크포인트 값 그대로), 없음 → 새로 학습
- Stage1이 끝나고 Stage2 중 끊겼으면 Stage1은 건너뛰고 Stage2만 이어간다
- 이미 저장(9번 셀)까지 끝난 실행을 다시 돌리면 모델을 중복 저장하지 않고 멈춘다

`RESUME = False`인데 같은 `RUN_NAME`에 결과가 있으면 덮어쓰지 않고 에러로 멈춘다.
다른 실행에서 학습한 Stage1을 재사용하려면 `STAGE1_WEIGHTS = ".../runs/<실행>_stage1/weights/best.pt"`.

### batch

기본 `batch=0.7` — GPU 메모리의 70%를 쓰는 배치 크기를 Ultralytics AutoBatch가 학습 시작 시 계산한다
(단일 CUDA GPU 전용, CPU/MPS는 16으로 대체). 고정하려면 정수로 (예: T4·imgsz 1024면 `batch=8`부터, OOM이면 줄임).
평가는 소수 batch를 지원하지 않아 `EVAL_BATCH`(16)를 따로 쓴다.

### 설계 메모

- **오버샘플링**: Stage2 train 목록(`lists/stage2_train.txt`)에 자체 이미지 경로를 3번 쓴다. Ultralytics는 목록의 중복을
  제거하지 않으므로 에폭당 3배 샘플링된다 (`cv/tests/test_notebook.py`가 실제 `YOLODataset`으로 확인)
- **val**: Stage1은 CMP val, Stage2는 CMP val + 자체 val (조기종료·best 선택 기준). 자체 val만 쓰면 장수가 적어 흔들린다
- **test**: 두 Stage 모두 자체 test셋으로 평가 — Stage2 파인튜닝 효과를 같은 기준으로 비교
- **증강** `AUGMENT`는 `config/augment.yaml`과 같아야 한다 (테스트가 확인). 바꿀 땐 둘 다 고친다
- `MAX_DET=500`: CMP에 창문 300개 넘는 이미지가 있어 Ultralytics 기본값(300)보다 크게

### 로컬 검증

```bash
cd cv && uv run pytest tests/test_notebook.py
```

nbformat 스키마 검증, 모든 코드 셀 파이썬 문법 검사, 하이퍼파라미터·헬퍼 셀 실행
(Stage 목록 생성, 오버샘플링, 개수 지표, 버전 번호, 모델 카드 포맷, 재개 상태 판정·실행 이름 결정)을 확인한다.
학습 셀은 실행하지 않는다.
