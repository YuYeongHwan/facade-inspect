# notebooks

## train_seg.ipynb — 창문/외벽 분할 학습 (Colab)

YOLO11s-seg 2단계 학습: Stage1 CMP만 → Stage2 CMP + 자체 데이터(자체 train ×3) 파인튜닝.
자체 test셋으로 window mask mAP50 / mAP50-95와 창문 개수 정확도를 내고, 가중치와 모델 카드를 드라이브에 남긴다.
자체 데이터가 아직 없으면 `STAGE1_ONLY = True`로 CMP만 학습할 수 있다 (아래 1-A).

### 1. 로컬: 데이터셋 zip 만들기 (레포 루트에서)

#### 1-A. CMP만 (자체 데이터 없음 → `STAGE1_ONLY = True`)

```bash
# CMP Facade 다운로드 (base 378장 + extended 228장)
mkdir -p data/cmp
curl -L -o data/cmp/CMP_facade_DB_base.zip     https://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_base.zip
curl -L -o data/cmp/CMP_facade_DB_extended.zip https://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_extended.zip
unzip -qo data/cmp/CMP_facade_DB_base.zip -d data/cmp       # -o: 두 zip에 같은 label_names.txt가 있어
unzip -qo data/cmp/CMP_facade_DB_extended.zip -d data/cmp   #     없으면 덮어쓰기 질문에서 멈춤

# YOLO-seg 변환 → --own 없이 병합 (CMP train 0.9 / val 0.1, test 없음) → zip
uv run --project cv python -m facade_cv.detect.datasets.cmp_convert --src data/cmp --out data/yolo/cmp
uv run --project cv python -m facade_cv.detect.datasets.merge_datasets --cmp data/yolo/cmp --out data/yolo/cmp_only
(cd data/yolo && zip -qr cmp_only.zip cmp_only)
```

드라이브의 `datasets/cmp_only.zip`에 올리고 노트북 첫 셀에서
`DATASET_ZIP = f"{DRIVE_ROOT}/datasets/cmp_only.zip"`, `STAGE1_ONLY = True`.
→ Stage1 학습 → **CMP val**로 평가(mAP·개수 정확도·오버레이) → `seg_window_facade_v{N}.pt` + 모델 카드 저장.
Stage2와 자체 test 평가는 건너뛴다. (확인: 606장 → train 545 / val 61)

#### 1-B. CMP + 자체 데이터 (기본, 2단계 학습)

1-A의 CMP 변환까지 한 뒤, 루트 README의 Label Studio 순서로 `data/yolo/own`을 만들고:

```bash
uv run --project cv python -m facade_cv.detect.datasets.merge_datasets \
    --cmp data/yolo/cmp --own data/yolo/own --out data/yolo/merged
(cd data/yolo && zip -qr merged.zip merged)
```

자체 데이터 test셋이 비어 있는 zip으로 `STAGE1_ONLY = False`면 노트북이 3번 셀에서 멈춘다.

#### 나중에 자체 데이터가 생기면

CMP만으로 학습한 Stage1을 그대로 쓰고 Stage2만 돌릴 수 있다: 1-B zip을 올리고 `STAGE1_ONLY = False`,
`STAGE1_WEIGHTS = ".../runs/<그 실행>_stage1/weights/best.pt"` (또는 같은 `RUN_NAME` + `RESUME = True` —
Stage1은 끝난 것으로 보고 건너뛴다). 최종 Stage가 달라 모델은 새 버전으로 따로 저장된다.

### 2. Google Drive 배치

```
MyDrive/facade-inspect/
  datasets/merged.zip     ← 업로드
  weights/                ← 노트북이 seg_window_facade_v{N}.pt, MODEL_CARD.md 생성 → 로컬 cv/weights/로 내려받는다
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
| 0 | 하이퍼파라미터 — 경로, `RESUME`/`RUN_NAME`, `STAGE1_ONLY`, Stage1/2 학습 설정, 증강, 평가 설정. **여기만 고친다** |
| 1 | 드라이브 마운트, `pip install ultralytics albumentations` (버전은 로컬 `uv.lock`과 같게 고정) |
| 2 | 헬퍼 함수 |
| 3 | zip 압축 해제(`/content/data/<zip 이름>`), Stage별 이미지 목록·data yaml 생성, 평가 대상 결정 |
| 4–5 | Stage1, Stage2 학습 (조기종료 `patience`). `RESUME`이면 끝난 Stage는 건너뛰고 중단된 Stage는 이어서. `STAGE1_ONLY`면 5는 건너뜀 |
| 6 | window mask P/R/mAP50/mAP50-95 — 자체 test셋 Stage1 vs Stage2 (`STAGE1_ONLY`: CMP val, Stage1만) |
| 7 | 창문 개수 정확도 (6과 같은 대상): 이미지당 \|검출−GT\| (MAE, MAPE, ±10% 이내 비율, 오차 큰 이미지 10장) |
| 8 | 최종 모델 예측 오버레이 16장 그리드 (평가 대상, 부족하면 자체 val로 채움) → `runs/<RUN_NAME>_pred_grid.jpg` |
| 9 | 최종 best.pt(Stage2, `STAGE1_ONLY`면 Stage1) → 드라이브 `weights/seg_window_facade_v{N}.pt` (N 자동 증가), `weights/MODEL_CARD.md`에 설정·지표·평가 대상 append. 같은 실행·같은 최종 Stage는 두 번 저장하지 않음 |

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

## 학습된 가중치로 예측 (로컬)

드라이브 `facade-inspect/weights/`의 `seg_window_facade_v{N}.pt`와 `MODEL_CARD.md`를 레포의 `cv/weights/`에 받는다
(gitignore). 기본 가중치는 `config/detect.yaml`의 `weights` — 새 버전을 쓰려면 여기를 고친다.

```bash
# 영상 → 프레임 → 예측 오버레이. --images에 io.extract 출력 폴더를 그대로 (frames.json 등 이미지 아닌 파일은 무시)
uv run --project cv python -m facade_cv.io.extract --video input.mp4 --out runs/insp_001/frames
uv run --project cv python -m facade_cv.detect.predict \
    --images runs/insp_001/frames   # → runs/insp_001/pred/*.jpg (다른 가중치: --weights cv/weights/seg_window_facade_v2.pt)
```

디바이스는 cuda > mps > cpu 자동 (`--device cpu`로 지정 가능), 추론 설정은 `config/detect.yaml`
(imgsz는 학습과 같은 1024). 오버레이는 클래스별 색 마스크 + 왼쪽 위에 `window N  facade M`.
