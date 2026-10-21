# Facade Inspect

드론 영상 기반 건물 창문 오염도 분석 웹 시스템. 설계 원칙·데이터 계약은 [CLAUDE.md](CLAUDE.md) 참조.
참고 자료(논문·데이터·도구) 출처는 [docs/REFERENCES.md](docs/REFERENCES.md) — 코드·문서의 `[R번호]`가 이 목록을 가리킨다.
데이터셋 출처·라이선스·사용 장수·클래스 변환 규칙은 [data/DATASETS.md](data/DATASETS.md).

## 구조

```
cv/                 파이썬 패키지 facade_cv (uv, Python 3.11)
  facade_cv/
    schema.py       result.json 데이터 계약 (pydantic v2) — server/web의 기준
    device.py       torch 디바이스 선택 (cuda > mps > cpu)
    config.py       config/*.yaml 로더
    pipeline.py     영상 → result.json 전체 파이프라인 (미구현)
    io/extract.py   프레임 추출 (노출/블러/ORB 중복 필터)
    io/source.py    실시간 소스 webcam:N / file: / rtsp:// tcp:// udp:// → (timestamp, frame), 재연결·최신 프레임·통계
    io/captures.py  data/raw/videos/captures.csv → building 단위 영상 목록 (건물 단위 split용)
    detect/
      augment.py    config/augment.yaml → Ultralytics train 인자
      segmenter.py  WindowSegmenter — YOLO-seg 래퍼 (config/detect.yaml weights) → Detection(cls, mask, polygon, bbox, conf)
      predict.py    이미지 폴더 예측 → 마스크 오버레이 (io.extract 출력 폴더 그대로 입력)
      postprocess.py 마스크 → 4점 사각형 → 외벽 필터 → 격자 정규화 (row/col/flags)
      run.py        검출 + 후처리 CLI → detect/detections.json (+ --debug 단계별 시각화)
      datasets/     분할 데이터셋 도구 (cmp/ade20k/ttpla/roboflow_convert, label_studio/, merge_datasets)
    eval/compare.py 자체 시험셋 모델 비교 (v1 vs v2: mAP, 장애물 위 오검출, 예시 그리드)
    rectify/homography.py  소실점 두 개로 외벽 정면화 호모그래피 (detect 후처리·T6 공용)
    contamination/grade.py  오염 심각도 지수 CSI(0~1) → 청소 우선순위 level 1~5 (config/grading.yaml, docs/scoring.md)
    contamination/calibrate.py  사람 평가(1~5) 수집 → 가중 카파 일치도 → CSI 기준값·k_t 보정
    live/run.py     실시간 창문 검출 (소스 → segmenter + postprocess, 프레임 단위) → 오버레이 + JSON 라인
    recon3d/        3D 복원: sfm(pycolmap) → up_axis(Y-up) → planes(RANSAC 외벽) → mesh(판/L자/박스) →
                    ortho(정사영 텍스처) → windows3d(창문 ID) → export(glb·result.json). degrade(가상 평면 1개)
    track/          (빈 패키지)
  tests/            pytest
  notebooks/        Colab 학습 노트북 (train_seg.ipynb — 사용법은 notebooks/README.md)
config/             grading.yaml(청소 우선순위 5단계), extract.yaml(프레임 필터), dataset.yaml(데이터셋), augment.yaml(학습 증강),
                    detect.yaml(추론·후처리), rectify.yaml(정면화), live.yaml(실시간 소스·처리), recon.yaml(3D 복원)
fixtures/           목업 run 디렉토리 (result.json, building.glb, facades/, windows/)
  make_mock.py      목업 재생성 스크립트
edge/raspberrypi/   드론 탑재 송출·녹화 스크립트 (미검증 — setup.md)
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
uv run --project cv python -m facade_cv.contamination.grade 0.03                  # CSI → "3 가급적 청소"
uv run --project cv python -m facade_cv.recon3d.run --frames runs/<id>/frames --detections runs/<id>/detect --out runs/<id> --debug
uv run --project cv python -m facade_cv.io.source --probe                         # 웹캠 장치 번호 (아이폰 연속성 카메라 포함)
uv run --project cv python -m facade_cv.live.run --source webcam:0 --debug        # 실시간 검출 (아래 "실시간")
```

## 실시간 (facade_cv.io.source, facade_cv.live.run)

소스 URI만 바꾸면 같은 코드가 돈다. 설정은 `config/live.yaml`.

| 소스 | 예 | 비고 |
|---|---|---|
| `webcam:N` | `webcam:0` | 노트북 카메라, 맥 연속성 카메라(아이폰). 번호는 `io.source --probe`. 맥은 터미널 앱에 카메라 권한 필요 |
| `file:<경로>` | `file:data/raw/videos/b04_shcool/b04_ground_01.MOV` | 기본 실시간 속도 재생 (`--no-realtime`이면 모든 프레임 처리) |
| `rtsp://` `tcp://` `udp://` | `rtsp://<파이IP>:8554/cam` | 라즈베리파이 (edge/raspberrypi). OpenCV FFMPEG 백엔드 |

- 끊기면 자동 재연결 (프레임 없이 연속 실패하면 대기 시간을 2배씩, `max_retries: null` = 무한)
- 실시간 소스는 수신 스레드가 계속 읽고 처리는 항상 최신 프레임만 → 처리가 느려도 지연이 쌓이지 않고 사이 프레임은 건너뛴다
- 통계: 수신 fps, 지연(수신 → 처리 시작, 네트워크 지연 제외), 건너뜀, 끊김/재연결
- `live.run`: 프레임을 `max_side`(1280)로 줄여 segmenter(`imgsz` 640 — 비행 후 분석 `detect.yaml`은 1024 그대로) + postprocess → 화면 오버레이(창문 사각형 `#임시ID rRcC`,
  처리/수신 fps·지연·건너뜀) + JSON 라인 (stdout과 `runs/<id>/live/detections.jsonl`, 스키마 `LiveFrame`).
  `--record`: `raw.mp4`(수신한 모든 프레임) + `overlay.mp4`. `--debug`: `runs/<id>/debug/live/`에 5초마다 오버레이·단계 시각화.
  화면에서 `q` 종료, `s` 캡처. 추적(T5)이 없어 창문 id는 프레임마다 바뀐다

## 3D 복원 (facade_cv.recon3d)

T1 프레임 + T4 검출 → `runs/<id>/result.json`, `building.glb`, `facades/<F>.jpg`, `windows/<id>/crop.jpg`.
오염 분석(T7) 전이라 창문은 모두 **미분석** (cleaning_level null, web 회색 패널). 설정 `config/recon.yaml`.

1. **sfm**: SIFT → sequential 매칭 (`sequential_overlap` 20, 루프 검출은 `loop_detection` + vocab tree 파일) →
   incremental mapping. 매핑은 실행마다 달라 시드를 바꿔 최대 `mapping_trials`번, 가장 많이 등록된 모델.
   카메라 모델 기본 OPENCV (아이폰), 드론 FPV 광각은 OPENCV_FISHEYE. torch와 같은 프로세스에 올리지 않는다
   (OpenMP 런타임 충돌 — `facade_cv/device.py`)
2. **up_axis**: 카메라 업벡터 평균 → 대략 수직 평면 법선으로 보정 → Y-up. 면이 1개면 면 안 회전을 법선으로
   못 정하므로 창문 세로 변 기울기 중앙값으로 보정
3. **planes**: 통계적 이상치 제거 → numpy 반복 RANSAC (`detect_planes`만 바꾸면 Open3D로 교체) → 수직 평면 →
   병합 → 2~4면이면 가장 큰 면 기준 직교 격자로 스냅. 법선(밖) 방향은 그 면을 실제로 관측한 카메라(SfM track) 쪽.
   격자에 안 맞는 면·같은 방향의 작은 면은 실패로 보지 않고 extra로 보고 (같은 방향 면은 외벽 범위에 보탬).
   임계값은 촬영 거리(포인트→가장 가까운 카메라 중앙값) 비율
4. **mesh**: 1면 → 두께 있는 판, 2면 직교 → L자, 평행 2면·3~4면 → 박스. 외벽 좌표계 origin/u/v/width/height
5. **ortho**: texel마다 각 프레임으로 투영 (왜곡·어안 포함, 화각 밖은 왜곡 전에 제외) → 가장 정면인 K개 median
6. **windows3d**: `WindowObservation`(프레임, 4코너, track_id) → 광선-평면 교차 → 평면 좌표 클러스터링
   (T5 tracklet이 생기면 track_id로 묶음) → 중앙값 사각형 → row(아래층 = 1)/col → `F{n}-R{rr}-C{cc}`
7. **degrade**: SfM 실패·등록 부족·외벽 0개면 창문이 가장 많은 프레임을 정면화한 가상 평면 1개 + T4 row/col.
   `meta.recon_status`에 mode·면 수·등록률·경고 (직선 촬영 경로, extra 평면 등)

`--debug`: `runs/<id>/debug/recon/` points_planes.ply (외벽별 색, extra 주황), cameras.ply (궤적),
top_view.jpg (위에서 본 포인트·외벽·궤적), `<F>_windows.jpg` (텍스처 위 창문 ID). 리포트 `runs/<id>/recon/report.json`.

## 드론 탑재 (edge/raspberrypi) — 미검증

Raspberry Pi Zero 2 W + Camera Module 3 → `stream.sh` (rpicam-vid H.264 → SD에 TS 조각 녹화 + 파이 안 MediaMTX)
→ 노트북은 RTSP 또는 TCP 중계(`tcp_relay.sh`)로 수신. systemd로 부팅 시 자동 시작.
SD 녹화는 모드와 관계없이 카메라 시작부터 계속되고, Wi-Fi·노트북 연결이 끊겨도 끊기지 않는다. **하드웨어 없이 작성했고 실제 장비에서 검증하지 않았다.**
설치·핫스팟·전원(BEC)·2.4 GHz 공존 점검은 [edge/raspberrypi/setup.md](edge/raspberrypi/setup.md).

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
1-2. **장애물** (v2 가중치, `config/detect.yaml obstacles`): 창 마스크가 vegetation·wire·pole·door 마스크와 `remove_overlap`
   이상 겹치면 그 장애물 위 오검출로 보고 제거(`on_door` 등). 남은 창의 사각형 안 vegetation·wire·pole 영역은
   `occlusion`(polygon, 픽셀) + `occluded_ratio`로 결과에 넣는다 — 오염 계산에서 면적을 뺄 부분.
   v1 가중치는 장애물 클래스가 없어 이 단계가 아무것도 바꾸지 않는다 (출력 형식도 기존과 같음)
2. **외벽 필터**: 외벽 마스크(창문 자리 구멍을 메우고 `facade_filter.margin_ratio`만큼 팽창) 밖 창문 제거(`outside_facade`).
   외벽이 없거나, 창문 bbox 합집합보다 작거나, 창문 `max_outside` 비율 이상이 밖이면 외벽 검출 실패로 보고 건너뛴다(경고)
3. **격자**: 외벽(실패 시 창문 볼록 껍질) 안 LSD 직선 → 수평·수직 소실점 → 정면화 (`config/rectify.yaml`, 실패 시 원본 좌표).
   창문 중심 y/x를 1D 클러스터링해 row/col. 제거하지 않고 flag만: `off_grid`(행·열 모두 혼자), `odd_width`/`odd_height`
   (같은 열 폭/행 높이 중앙값과 `size_tol` 이상 차이). 빈칸에 저신뢰 검출이 있으면 `fill_candidates`(창문 수·`windows`와 별도), 없으면 `empty_cells`.
   점유율이 `grid.min_occupancy` 미만(코너 건물처럼 여러 면이 섞임)이면 빈칸·보완 후보는 만들지 않는다(경고 `grid_unreliable`)

출력 `<frames>/../detect/detections.json` — 이미지마다 counts(final은 보완 후보 제외), windows(corners, conf, row, col, flags),
fill_candidates, removed(reason), empty_cells, warnings, obstacles(검출된 장애물 클래스). 가림이 있는 창에만 occluded_ratio·occlusion.
`--debug`: `debug/detect/<stem>_stages.jpg`(초록 OK, 주황 flag, 하늘 fill, 빨강 quad 제거, 검정 장애물 위 제거, 흰 윤곽 가림,
자홍 외벽 밖, 노랑 외벽 영역), `_rectified.jpg`(정면화 + row/col + 빈칸 X), `_lines.jpg`(직선 후보 회색, 소실점 inlier 파랑/빨강).
row/col은 프레임 단위 임시값 — 최종 창문 ID는 3D 외벽 평면에서 정한다.

## 창문/외벽 분할 데이터셋 (facade_cv.detect.datasets)

클래스 (`config/classes.yaml`, Ultralytics YOLO-seg polygon 포맷): **v2** `0=window, 1=facade, 2=vegetation, 3=wire, 4=pole, 5=door`
(기본, active) / v1 `0=window, 1=facade`. v2는 v1 id를 그대로 둬서 v1 라벨·가중치와 호환. 모든 변환·병합 CLI에 `--classes v1|v2`.
각 도구의 출력은 `<dir>/images/*`, `<dir>/labels/*.txt`이고, `merge_datasets`가 이것들을 합쳐 분할한다.
출처·라이선스·장수·변환 규칙은 [data/DATASETS.md](data/DATASETS.md), 서지는 [docs/REFERENCES.md](docs/REFERENCES.md).

### 0. 공개 데이터 (v2) — 사람이 `data/external/<이름>/`에 받아 둔다

변환기는 폴더가 없으면 다운로드 안내만 출력하고 건너뛴다.

| 데이터 | 받는 곳 | 가입 | 용량 | 라이선스 | 변환 |
|---|---|---|---|---|---|
| ADE20K [R19] | `curl -LO http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip` → `data/external/ade20k/`에 unzip | 불필요 | 923MB (해제 ~2.3GB) | 이미지 **비상업 연구·교육용만** (MIT CSAIL terms), 주석 BSD-3 | `python -m facade_cv.detect.datasets.ade20k_convert` |
| TTPLA [R20] | https://github.com/r3ab/ttpla_dataset README의 Google Drive zip → `data/external/ttpla/` | 불필요 (Google 계정으로 받기) | 수 GB [확인 필요] | Apache-2.0 | `... .ttpla_convert` |
| Roboflow [R21] | 데이터셋 페이지 → Download → "YOLOv8" 형식 zip → `data/external/roboflow/<키>/` (키·URL: `config/dataset.yaml roboflow.datasets`) | **필요** (무료 계정) | 수백 MB | CC BY 4.0 (화이트리스트 외 거부) | `... .roboflow_convert` |

- ADE20K: 실외 장면(`sceneCategories.txt`) + sky ≥5% + building ≥10% + 창문 있음만. facade = building만 (wall 제외)
- TTPLA: cable → wire, tower_wooden → pole (철탑 제외), 긴 변 1280px로 축소
- Roboflow: 데이터셋별 클래스 매핑(`map`)에 없는 클래스는 버림. 실제 export 클래스명을 보고 `map`을 맞출 것

### 1. CMP Facade 변환

CMP Facade Database [R1] (Tyleček & Šára, GCPR 2013, http://cmp.felk.cvut.cz/~tylecr1/facade/) — CC BY-SA 3.0, 인용 필요.

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

### 2-1. v2 라벨링 준비 (b04 장애물 프레임 150장, 3명 × 50장)

```bash
# v1 모델 대리 점수(건물 피복률 낮음 + 후처리 제거 창문)로 장애물이 많을 법한 프레임, 15개 시간 구간 × 10장, dHash로 유사 프레임 배제
uv run --project cv python -m facade_cv.detect.datasets.label_studio.select_frames \
    --video data/raw/videos/b04_shcool/b04_ground_01.MOV --out data/own/b04_obstacle
# → data/own/b04_obstacle/raw/b04_ground_01__frame_<index>.jpg, selection.csv (점수·구간·batch·선택 사유)
uv run --project cv python -m facade_cv.detect.datasets.label_studio.prelabel --weights cv/weights/seg_window_facade_v1.pt \
    --images data/own/b04_obstacle/raw --out data/own/b04_obstacle/prelabel.json \
    --selection data/own/b04_obstacle/selection.csv --url-prefix "/data/local-files/?d=raw/"
# → prelabel_batch1.json, _batch2.json, _batch3.json (시간순 round-robin — 라벨러마다 영상 전 구간)
```

라벨링 규칙: [docs/labeling_guide.md](docs/labeling_guide.md). Labeling Interface는 6클래스 `labeling_config.xml`.

### 3. 병합·분할

```bash
uv run --project cv python -m facade_cv.detect.datasets.merge_datasets \
    --cmp data/yolo/cmp --ext ade20k=data/yolo/ade20k --ext ttpla=data/yolo/ttpla \
    --ext roboflow_building_facade=data/yolo/roboflow/building_facade \
    --own data/yolo/own --out data/yolo/merged   # --ext/--own 여러 번 가능 (없는 --ext 디렉토리는 건너뜀), 재실행은 --overwrite
```

- 공개 데이터(CMP 포함): train 0.9 / val 0.1 / **test 0 강제** — 시험셋은 자체 촬영만. 출력 파일명 `pub_<소스>__<원본>`
- 장수 상한 `balance`: 공개 전체 ≤3,000 (ADE20K ≤1,500, TTPLA ≤200, Roboflow 합 ≤1,200, CMP 전체), 상한 안에서는 희귀 클래스 우선
- 출력: `data.yaml`, `stats.json`(split·소스·클래스별 이미지/인스턴스 수), `sources.json`(소스 → [R번호]·라이선스·장수, 모델 카드용)
- 자체 데이터 그룹 `split.own_group_by`: `flight`(영상 하나) / `segment`(기본 — 프레임 600개 구간, 다른 split 경계 ±60프레임 버림)
- 자체 데이터: train 0.6 / val 0.2 / test 0.2, val·test에 최소 1그룹씩 보장 (그룹이 2개 미만이면 에러)
- **그룹 단위 분할**: 같은 비행의 연속 프레임이 train/test로 나뉘면 성능이 부풀려진다.
  자체 이미지 파일명은 `<flightID>__<frame>.jpg`여야 하고 (`split.own_group_pattern`, `io.extract` 출력 그대로),
  아니면 에러로 중단한다. `--allow-ungrouped`로만 우회 (그 이미지는 혼자 한 그룹)
- 출력 파일명은 `pub_<소스>__<원본>` / `<own 디렉토리명>__<원본>` (구 zip의 `cmp__`도 노트북이 공개 데이터로 인식)

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

## 모델 비교 평가 (facade_cv.eval.compare)

```bash
uv run --project cv python -m facade_cv.eval.compare --data data/yolo/merged/data.yaml \
    --weights cv/weights/seg_window_facade_v1.pt --weights cv/weights/seg_window_facade_v2.pt --out runs/eval
```

- 대상: data.yaml의 test split (= 자체 촬영만). 모델마다 라벨을 그 모델이 아는 클래스만 남긴 임시 복사본으로 Ultralytics val
  → 2클래스 v1과 6클래스 v2를 같은 이미지로 비교 (창문 mask mAP50·mAP50-95, 클래스별 P/R/mAP)
- 장애물 위 오검출: conf ≥ 0.4 창문 예측 중 GT 창문과 IoU < 0.5이고 GT door/vegetation/wire/pole과 예측 면적의 50% 이상 겹친 것
  (`config/eval.yaml`). raw(모델 출력)와 post(후처리 후)를 장애물 클래스별로
- 출력 `runs/eval/<날짜_시각>/metrics.json, table.md, grid.jpg`(모델별 예측을 나란히, 오검출 차이 큰 이미지 순). 표는 `cv/weights/MODEL_CARD.md`에도 append

## 오염 심각도 지수 CSI · 청소 우선순위 (facade_cv.contamination)

CSI = Σ_j (A_j × k_s(j) × k_t(j)) / (A_g × k_s_max × k_t_max), 0~1. 식·기호·근거·손 계산 예시·한계는 [docs/scoring.md](docs/scoring.md).

- A_g 유리 면적 = 창문 crop − 창틀 마진 − 가림(G5 occlusion), A_j 오염 영역 중 유리 안 면적
- k_s 심각도 1~3 = 깨끗한 유리 대비 CIELAB ΔE*ab (< 6, 6~15, ≥ 15), k_t 종류 가중치 (dust 0.5, stain 0.8, grime 1.0, heavy 1.5)
- level 경계 CSI 0.009 / 0.022 / 0.044 / 0.089 (경계값은 상위 level)
- **수치는 모두 보정 전 초기값** (`config/grading.yaml` status). result.json `meta.csi_params`에 계산에 쓴 값을 함께 저장하고,
  pydantic·zod가 창문마다 csi = Σ 영역 항, level = csi 구간인지 검사한다
- 웹 상세 패널: 영역별 종류(k_t)·A_j/A_g·심각도(ΔE)·기여, 유리 면적·창틀 마진·가림 비율, 합계와 단계

보정 (사람 평가 → 가중 카파 최대화, `config/calibrate.yaml`):

```bash
uv run --project cv python -m facade_cv.contamination.calibrate rate --rater kim --images runs/insp_001 --glob "windows/*/crop.jpg"
uv run --project cv python -m facade_cv.contamination.calibrate agreement
uv run --project cv python -m facade_cv.contamination.calibrate fit --regions runs/insp_001/result.json --doc docs/scoring.md
```

## 목업 (fixtures/)

박스 건물(12×24×8, 상대 단위) 외벽 4면(F1~F4), 창문 20개(청소 우선순위 level 1~5 = 6/4/4/4/2, 그중 4개는 창 앞 나뭇가지 가림).
crop에 오염 영역을 심각도별 색차로 그린 뒤 `contamination/grade.py` `grade_window`로 실제 측정해 CSI·level을 매긴다
(유리 면적 = 창틀 마진·가림 제외, 영역별 ΔE → 심각도). 계산에 쓴 값은 `meta.csi_params`.
`cv/tests/test_schema.py`가 같은 result.json을 pydantic과 web zod(`web/scripts/validate-result.mjs`) 양쪽으로 검증한다.
`result.json` 안의 경로는 모두 `fixtures/` 기준 상대경로다.
