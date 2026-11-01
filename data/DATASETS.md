# 데이터셋

`data/`는 gitignore (이 파일만 추적). 서지·라이선스 원문은 [docs/REFERENCES.md](../docs/REFERENCES.md)의 [R번호].
학습에 쓰는 이미지가 늘거나 바뀌면 이 표를 갱신한다.

## 목록

| 데이터셋 | 출처 | 라이선스 | 원본 장수 | 사용 장수 (용도) | 위치 |
|---|---|---|---|---|---|
| CMP Facade Database — base | Tyleček & Šára, GCPR 2013 [R1] <br> https://cmp.felk.cvut.cz/~tylecr1/facade/ | CC BY-SA 3.0 | 378 | 378 (분할 학습) | `data/cmp/base` |
| CMP Facade Database — extended | 〃 [R1] | CC BY-SA 3.0 | 228 | 228 (분할 학습) | `data/cmp/extended` |
| ADE20K SceneParsing (ADEChallengeData2016) | Zhou et al., CVPR 2017 / IJCV 2019 [R19] <br> http://sceneparsing.csail.mit.edu/ | 이미지 **비상업 연구·교육용만**, 주석 BSD-3 | 22,210 (train 20,210 + val 2,000) | **0 — 제외** (아래 "ADE20K 제외"). 2021 전체판은 배포 사이트 오류로 받을 수 없음 | `data/public/ade20k/ADEChallengeData2016` (받아 둠, 미사용) |
| TTPLA | Abdelfattah, Wang, Wang, ACCV 2020 [R20] <br> https://github.com/r3ab/ttpla_dataset | Apache-2.0 | **1,242** (3840×2160, 영상 85개, labelme polygon) | **타일 200장** (1280×1280, 영상 79개) — wire 1,633·pole 175 인스턴스 | `data/public/ttpla/data_original_size_v1` |
| Roboflow building-facade-segmentation-instance v4 | Roboflow Universe `building-facade` [R21-a] | CC BY 4.0 | 2,226 | Roboflow 합계 ≤1,200 (v2) — 다운로드 후 갱신 | `data/public/roboflow/building_facade` |
| Roboflow Facade_test | Roboflow Universe `ir-2znee` [R21-b] | CC BY 4.0 | 382 | 〃 | `data/public/roboflow/facade_test` |
| Roboflow window-segment-main | Roboflow Universe `mzworkspace-segj7` [R21-c] | CC BY 4.0 | [확인 필요] | 〃 | `data/public/roboflow/window_segment` |
| Roboflow Street Trees Detection v4 | Roboflow Universe `treedetection-vf07h` [R29] <br> https://universe.roboflow.com/treedetection-vf07h/street-trees-detection | CC BY 4.0 | 1,477 (= 원본 615 × 증강, 영상 8개 프레임) | **0 — 미사용** (아래 "공개 vegetation 후보 검토"). 시험 변환 503장 (SAM 자동 polygon, 원본은 박스) — 결과 삭제 | `data/public/roboflow/veg_street_trees` |
| Roboflow dirty window v1 | Roboflow Universe `project-novx2` [R26] <br> https://universe.roboflow.com/project-novx2/dirty-window | CC BY 4.0 | 366 (train 254 / valid 112), 장면 366 | t7_dev 2 + csi_calib 25 (**평가만, 학습 미사용**) | `data/public/dirty_windows/dirty-window` |
| Roboflow window check v3 | Roboflow Universe `dirty-window` [R27] <br> https://universe.roboflow.com/dirty-window/window-check | CC BY 4.0 | 865 (train 798 / valid 67), 고유 장면 ≈53 | t7_dev 16 + csi_calib 16 (평가만) | `data/public/dirty_windows/window-check` |
| Roboflow window check2 v2 | Roboflow Universe `dirty-window` [R28] <br> https://universe.roboflow.com/dirty-window/window-check2 | CC BY 4.0 | 263 (train 243 / valid 20), 고유 장면 ≈45 | t7_dev 7 + csi_calib 4 (평가만) | `data/public/dirty_windows/window-check2` |
| Wikimedia Commons 공개 오염 유리 사진 | Commons 개별 업로더 [R33] <br> `data/own/dirty_windows/web/SOURCES.csv` | 사진마다 다름 (CC0 / CC BY / CC BY-SA) | 16 (heavy 5, dust 4, grime 3, stain 1, wet_glass 2, clean 1) | 16 (**사진 데모 `runs/demo_web`·T7 정성 검증만, 학습 미사용**) | `data/own/dirty_windows/web` |
| 오염 창문 근접 사진 (자체 촬영, **예정**) | 직접 촬영 10~20장 | 자체 소유 | — | T7 검증·CSI 보정 (예정) | `data/own/dirty_windows/` |
| b01_songsan (자체 촬영) | 2026-10-07, 지상 정지 촬영, 맑음 | 자체 소유 | 영상 1개 | 0 (학습 미사용) | `data/raw/videos/b01_songsan` |
| b02_smilen (자체 촬영) | 2026-10-07, 지상, 올려다보는 각도 | 자체 소유 | 영상 1개 | 0 (학습 미사용) | `data/raw/videos/b02_smilen` |
| b03_shopglass (자체 촬영) | 2026-10-07, 근접, 반사 심함·오염 있음 | 자체 소유 | 영상 1개 | 0 (학습 미사용) | `data/raw/videos/b03_shopglass` |
| b04_shcool (자체 촬영) | 2026-10-14, 1층 한 바퀴 이동 촬영 (3840×2160, 30fps, 325초) | 자체 소유 | 영상 1개 | v2 라벨링 150장 (장애물 위주, 라벨링 전) / 프레임 80·266장 (3D 복원 테스트), 실시간 검출 테스트 | `data/raw/videos/b04_shcool`, 라벨링용 `data/own/b04_obstacle` |

- CMP 분할 (`data/yolo/cmp_only`, `config/dataset.yaml` split.cmp, seed 0): train 545 / val 61 / test 0
- 자체 촬영 메타데이터(날짜·날씨·각도·비고): `data/raw/videos/captures.csv`. 촬영 장비: 아이폰 (`config/recon.yaml` sfm.camera_model 주석)
- 자체 영상은 아직 Label Studio 라벨링 전 → 현재 가중치 `seg_window_facade_v1.pt`는 **CMP만으로 학습**
- b04 3D 테스트 프레임: `runs/recon_b04_wall` 80장 (흰 외벽 구간), `runs/recon_b04_full` 266장 (2 fps, 전체)
- CC BY-SA 3.0: CMP에서 파생한 YOLO 라벨(`data/yolo/cmp*`)을 배포할 때는 출처 표시 + 같은 라이선스 필요
- 자체 영상에는 행인·차량이 찍혀 있다. 외부 공개 시 얼굴·번호판 처리 필요

## v2 공개 데이터 장수 (merge_datasets balance)

`config/dataset.yaml balance`: 공개 전체 ≤ **3,000장** (CMP 전체 606장 + ADE20K ≤1,500 + TTPLA ≤200 + Roboflow 합 ≤1,200).
**ADE20K는 제외** — 상한 조정은 검토 중 (아래 표의 ADE20K 몫이 비어 실제 합은 ≤2,006).
전체가 넘으면 상한 있는 소스를 비율대로 줄인다. 상한 안에서는 가진 클래스 중 가장 희귀한 클래스(전체 인스턴스 빈도)가
드문 이미지부터 남긴다 (door·pole·wire가 있는 이미지 우선). 이유: 무료 Colab T4에서 Stage1(imgsz 640, 35에폭)이
3시간 안에 끝나야 한다 — v1 실측(550장×100에폭@1024 = 2.1h) 기준 3,000장 ≈ 1.6h. 실제 장수는 merge 출력 `stats.json`/`sources.json`.

## 분할 규칙 (v2)

- **시험셋 = 자체 촬영만**: 공개 데이터는 train 0.9 / val 0.1 / test 0 (`split.cmp`, `split.public` — test가 0이 아니면 merge가 에러)
- 자체 데이터는 그룹 단위 (`split.own_group_by`): 건물(영상)이 여럿이면 `flight`(영상 하나 = 그룹),
  b04 하나뿐인 지금은 `segment` — 프레임 index // 600 (30fps × 20초) 구간이 한 그룹. 서로 다른 split에 속한
  이웃 구간 경계 ±60프레임(2초) 안의 이미지는 버린다 (이웃 프레임이 train과 test에 섞이지 않게)
- train 0.6 / val 0.2 / test 0.2 (그룹 수 기준), val·test 최소 1그룹

## 클래스 변환 규칙 → YOLO-seg v2 (`config/classes.yaml`)

`0=window, 1=facade, 2=vegetation, 3=wire, 4=pole, 5=door` (v1 = 앞의 2개, id 동일). 변환기마다 `--classes v1|v2`.
활성 클래스셋에 없는 클래스는 버린다 → 같은 원본에서 v1/v2 라벨을 모두 만들 수 있다.

| 데이터셋 | window (0) | facade (1) | vegetation (2) | wire (3) | pole (4) | door (5) | 제외 |
|---|---|---|---|---|---|---|---|
| CMP [R1] | 아래 v1 규칙 | 아래 v1 규칙 | — | — | — | door 픽셀 연결 성분마다 1 | background, shop, balcony |
| ADE20K [R19] | windowpane | **building만** | tree, grass, plant, flower | — | pole | door | wall 포함 나머지 전부 |
| TTPLA [R20] | — | — | — | cable | tower_wooden | — | tower_lattice, tower_tucohy, void |
| Roboflow [R21] | 데이터셋별 `config roboflow.datasets.*.map` (window/windows) | building_facade의 facade | building_facade의 vegetation | — | — | facade_test의 door | 매핑 없는 클래스 (street, car, fence 등) |
| 자체 (Label Studio) | window | facade | vegetation | wire | pole | door | — (`docs/labeling_guide.md`) |

ADE20K·Roboflow semantic 원본의 인스턴스 = 클래스 마스크의 **연결 성분** (붙은 창문은 하나로 합쳐짐 — 한계).

### ADE20K 선별 기준과 제외 이유
- **실외 장면만**: `sceneCategories.txt`의 장면이 `config ade20k.outdoor_scenes`(street, building_facade, skyscraper, house 등
  36개)에 있는 이미지. ADE20K는 실내가 대부분이라 그대로 쓰면 실내 창·문·벽이 대량으로 섞여 드론 외벽 시점과 동떨어진다
- **sky 픽셀 ≥ 5%** (`min_sky_ratio`): 장면 이름이 실외여도 건물 안을 찍은 사진·근접 사진을 한 번 더 거른다
- **building 픽셀 ≥ 10%** (`min_facade_ratio`) 그리고 **window 인스턴스 ≥ 1** (`require`): 건물 외벽이 주인공인 장면만
- **facade = building만, wall 제외**: ADE20K의 wall은 실외 장면에서도 담장·옹벽·실내 벽이 섞여 있다.
  wall을 facade로 넣으면 담장 앞 창 없는 면을 외벽으로 배워 외벽 필터(후처리 b)가 무력해진다

### ADE20K 제외 — 2016판 사용 불가, 2021 전체판 받을 수 없음
받은 파일: `ADEChallengeData2016` (SceneParsing, 150클래스 semantic PNG, 22,210장). 2026-10-22 확인 결과 v2에 쓰지 않는다
(`config/dataset.yaml ade20k.enabled: false`. 변환 코드 `ade20k_convert.py`와 테스트는 지우지 않고 남겨 둠).

**2016판 (ADEChallengeData2016): 외벽 창문·문이 따로 라벨링되지 않음 → 사용하면 창문 학습을 해친다 → 제외.**
근거: 실외 필터 후 창문 포함 41장, building_facade 장면 245장 중 windowpane 있는 것 1장. 세부:
- **외벽의 창문·문이 `building`에 포함되어 있다.** SceneParsing은 물체 단위 150클래스만 남기고 부품(part)을 버린 판이라
  건물 외벽의 창·문은 building 픽셀이다. building ≥ 10%인 이미지 중 windowpane 픽셀이 있는 비율:
  building_facade 1/245, skyscraper 3/271, apartment_building_outdoor 0/71, street 56/2,115
  (예: `ADE_train_00004594` — 창 12개·문 3개 외벽 전체가 building 하나)
- 실외 선별(장면 3,635 → sky ≥5% 2,350 → building ≥10% 2,011) 후 window 인스턴스가 있는 이미지는 **41장**뿐이고 대부분 1층 상점 유리.
  window 조건을 빼고 2,011장을 쓰면 "창·문이 있는 외벽 = facade"를 배워 CMP의 window/door 학습과 정면으로 충돌한다
- instance 구분도 없다 → 연결 성분이 인스턴스: 붙은 건물들이 facade 하나(2,011장 중 이미지 20% 넘는 facade 1,591개), 가로수 줄이
  vegetation 하나, 작은 조각은 버림(vegetation 9,025개). 파일도 변환기 기대와 다르다 (`objectInfo150.txt` 탭 구분, 이름 `,` 구분)

**2021 전체판 (ADE20K_2021_17_01, instance·part 주석 — building의 부품 window·door 포함): 받을 수 없음.**
배포 사이트 https://ade20k.csail.mit.edu/request_data 서버 오류 — PHP가 실행되지 않고(소스가 그대로 보임) 보안 코드 이미지가 깨져
신청서를 제출할 수 없다. 2026-10-22 Chrome·Safari 모두에서 확인.

**대안: vegetation은 Roboflow 데이터셋으로 대체** (후보·라이선스 확인 후 config roboflow.datasets에 등록). window·facade·door는
CMP + Roboflow 창문 데이터셋, pole은 TTPLA(+ Roboflow 후보)로 채운다. ADE20K는 config에서 비활성화만 했다 — 사이트가 복구되면
2021판용 reader를 새로 붙인다.

### 공개 vegetation 후보 검토 → 결론: vegetation은 b04 자체 라벨로 확보

| 후보 | 시점·라벨 | 판정 | 사유 |
|---|---|---|---|
| ADE20K 2016 (ADEChallengeData2016) [R19] | 지상, semantic | 제외 | 외벽 창문·문이 building에 포함돼 라벨 없음 → 창문 학습을 해친다 (위 "ADE20K 제외") |
| ADE20K 2021 전체판 [R19] | 지상, instance·part | 받을 수 없음 | 배포 사이트(request_data) 서버 오류로 가입 불가 (2026-10-22) |
| Roboflow Blue Halo vegetation-segmentation / tree-segmentation | 항공 | 제외 | 항공 시점 — 드론이 외벽을 보는 시점과 다르다 (사람이 썸네일로 확인) |
| Roboflow Street Trees Detection v4 [R29] | 거리, **박스** (SAM으로 polygon 생성) | 제외 | 박스 범위 문제 (아래) |

**Street Trees 제외 사유** (SAM 변환 샘플 12장 `runs/convert_check/street_trees_*.jpg`를 사람이 확인, 2026-10-22):
1. 박스가 줄기·하단 수관만 감싼다 → **상단 수관 대부분이 배경으로 학습**된다
2. 줄기만 칠한 마스크(가늘고 긴 세로 막대)가 **pole 클래스와 충돌**한다
3. SAM이 박스 안의 **건물 벽·흙더미를 칠한 오류**가 있다

설정: `config/dataset.yaml balance.max_images.veg_street_trees: 0` (비활성). 변환 도구 `box_to_seg_sam.py`(박스 → SAM 2.1 [R30] → polygon,
면적 필터 25~95%·조각 5%)와 테스트는 다른 박스 데이터용으로 남겨 둔다. 시험 변환 기록: 원본 615장(1,477장 중 증강 사본 862 제외),
SAM 박스 565개 중 유지 496 / 25% 미만 69, 최종 503장·인스턴스 931. 변환 결과 `data/yolo/veg_street_trees`는 삭제, 원본 export는 `data/public`에 유지.

**결론: v2의 vegetation 정답은 b04 라벨뿐이다** — 라벨러가 가로수·화단·덤불·잔디·담쟁이를 빠짐없이 칠한다 (`docs/labeling_guide.md` 맨 위 강조).
공개 데이터(CMP·TTPLA) 안의 나무는 라벨이 없어 배경으로 학습되는 한계가 남는다.

### TTPLA 사용 범위 결정
- cable → wire: 도심 전선과 같은 물체 (항공 시점이지만 가는 선 모양은 같다)
- tower_wooden → pole: 목재 전신주는 도심 전봇대와 형태(수직 단일 기둥)가 같다
- tower_lattice·tower_tucohy 제외: 송전 철탑은 건물 외벽 앞에 서 있지 않고, 넣으면 pole이 격자 철골·건물 장식
  기둥으로 일반화될 위험이 있다. void(불명확 영역) 제외
- 라벨 이름은 대소문자 무시 (원본에 `Cable` 28개 → wire)
- **void 영역은 이미지에서 회색(114,114,114)으로 칠한다** — void 안 라벨 없는 전선이 배경으로 학습되지 않게
- **축소 대신 타일**: 원본 3840×2160에서 1280×1280 타일 6개 (가로 3칸, 세로 2칸이 400px 겹침). polygon은 타일 경계에서 잘라 유지,
  잘린 조각이 min_area_ratio(타일 면적 대비 wire 0.00002 ≈ 33px², pole 0.0002)보다 작으면 버림. wire·pole 없는 타일은 버림.
  전선 평균 굵기: 원본 10.7px(중앙값 8.0) → 타일 8.0px. 예전 방식(긴 변 1280으로 축소, ×1/3)이면 3.6px
- **선택 (200장)**: pole 있는 타일 140(70%) + wire만 있는 타일 60(30%, config `wire_only_ratio`). 같은 영상(파일명 `<영상>_<프레임>`)에서
  최대 4장, 한 프레임에서는 1장, 이미 고른 프레임과 가장 먼 프레임부터 (영상 내 프레임 간격 최소 2, 중앙값 255). 영상 순서 round-robin
- 결과 (2026-10-22): 후보 타일 6,341 → 200장, 영상 79개 (4장 27개·3장 14개·2장 12개·1장 26개). wire 1,633·pole 175 인스턴스.
  샘플 `runs/convert_check/ttpla_01~08.jpg` (wire 노랑, pole 자홍)
- **한계**: 200장 중 57장에 라벨 없는 철탑(tower_lattice 39장, tower_tucohy 25장)이 배경으로 남는다. tower_tucohy(콘크리트·강관 전주)는
  도심 전봇대와 비슷해 pole과 충돌할 수 있다. TTPLA 자체 라벨에서 빠진 작은 전주도 있다 (샘플 8). 항공 시점이라 장수 상한을 200장으로 낮게

## 클래스 변환 규칙 v1 → YOLO-seg (0=window, 1=facade)

설정: `config/dataset.yaml`. 변환 코드: `cv/facade_cv/detect/datasets/cmp_convert.py`, `label_studio/export_to_yolo.py`

| 데이터셋 | 원본 클래스 | → window (0) | → facade (1) | 제외 |
|---|---|---|---|---|
| CMP [R1] | 12개 (PNG 팔레트 index 1~12, `label_names.txt`) | XML 사각형(`window`) 하나 ∩ 픽셀 클래스 {window, blind} → 인스턴스 1개. 붙은 창문 분리, 덧문으로 갈라진 창문 복원 | {facade, molding, cornice, pillar, sill, deco, blind(창문 사각형 밖)} → 구멍을 채운 외곽 polygon 하나 | background, door, shop, balcony |
| 자체 촬영 (b01~b04) | Label Studio polygon: window, facade (`labeling_config.xml`) | window 그대로 | facade 그대로 | — |

공통: 이미지 면적 대비 window < 0.0001, facade < 0.0005인 인스턴스는 버림. polygon 단순화 approxPolyDP eps 0.005 (윤곽 길이 대비).
CMP에서 창문이 빠지는 경우: PNG에서 balcony/deco/shop에 완전히 가려진 창문 (예: `cmp_x0173`).

## 오염 창문 평가 데이터 (T7 검증·CSI 보정, 학습 미사용)

[R26]~[R28] Roboflow 3개. 형식 YOLOv11 (`{train,valid}/{images,labels}`, test split 없음). 라벨 없는 이미지 0장.
**학습에는 쓰지 않는다** — 오염 분석(T7)이 맞게 동작하는지 보는 개발 세트와, CSI 기준값을 사람 평가에 맞추는 보정 세트로만 쓴다.

| | dirty window [R26] | window check [R27] | window check2 [R28] |
|---|---|---|---|
| 장수 (train / valid) | 254 / 112 | 798 / 67 | 243 / 20 |
| 고유 장면 | 366 | ≈53 | ≈45 |
| 클래스 (인스턴스 수) | dirty window 4,492 | stain 733, glare 506, dirt 252, wet_glass 197 | glare 463, stain 144, dirt 63, wet_glass 7 |
| 라벨 형식 | polygon (중앙값 34점, 장당 ≈12개) | box | box |
| 해상도 | 원본 그대로 (93종, 810×1080·960×1280 최다) | 전부 640×640 (stretch) | 전부 640×640 (stretch) |
| 시점 (육안, 대략) | 실내에서 본 창 1~2개 ≈90%, 유리 근접 ≈10%, 외벽 원거리 0 | 유리 표면 근접 ≈85%, 실내에서 본 창 ≈15%, 외벽 0 | 커튼월 외관(원거리) ≈65%, 근접 ≈35% |

- **dirty window의 polygon은 오염 얼룩이 아니라 "더러운 창(유리+창틀+창턱) 영역 전체"**다. 창턱·우편함 같은
  유리 밖 표면도 칠해져 있다 → 오염 위치·면적 정답으로 쓸 수 없고, 창 단위 오염 정도(사람 평가)용 이미지로만 쓴다
- window check 계열의 box는 오염 종류별 영역이지만 이미지 전체를 덮는 box가 많다 (dirt+stain 합집합 면적 > 0.95인 장면 ≈25개).
  glare(반사)·wet_glass(물기)는 오염이 아니다 → T7이 이것을 오염으로 잡지 않는지 확인하는 용도
- **중복 주의**: window check 계열은 업로드 전에 이미 증강한 사본(`aug_<원본>_<번호>`, `glare_aug_...`)이 있고,
  Roboflow export가 원본마다 3배 증강(`.rf.<해시>`)을 다시 만든다. 두 데이터셋 사이에도 같은 사진이 있다 (예: "WASH ME").
  장면은 파일명 정규화(`.rf.`·`aug_`·`glare_`·끝 번호 제거) + 256비트 dHash(좌우 반전 포함, 해밍 ≤ 20) + 육안으로 묶었고,
  장면마다 1장만 고른다 (valid = 증강 없는 원본 우선). train과 valid에 같은 장면이 있다 — 원래 split은 쓰지 않는다
- 선택 목록 (이미지 복사 없음, 경로만):
  - `data/eval/t7_dev.csv` 30장 — dirt 5, stain 5, 넓은 면적 심한 오염 5, glare 5, wet_glass 5, 깨끗한 창 5 (CMP 3, b04 2)
  - `data/eval/csi_calib.csv` 60장 (t7_dev와 겹치지 않음) — 공개 45 + 깨끗/약간 15 (CMP 8, b04 7), split fit 40 / holdout 20
    - `est_grade` 0~4 = 이미지 고를 때 쓴 **임시 추정** 오염 정도 (0 깨끗/오염 없음 ~ 4 매우 넓음). window check 계열은
      dirt+stain box 합집합 면적 구간, dirty window는 한 사람의 육안(1 약함·2 중간·4 심함), CMP·b04는 0.
      고르게 분포시키려고 붙인 값일 뿐이다 → **사람 3명의 평가로 대체 예정, 보정 정답으로 사용 금지**
      (보정 정답은 `contamination/calibrate.py rate`로 모은 평가의 합의값)
- 직접 촬영 오염 창문 사진 10~20장 **예정**: `data/own/dirty_windows/` (공개 데이터에 부족한 국소 dirt·새똥·물때·외부 시점)

### 공개 오염 유리 사진 16장 (사진 데모) [R33]
- 수집 2026-11-01, Wikimedia Commons. 분류(heavy=새똥, dust, grime, stain, wet_glass·clean=음성 대조)는 수집자가 붙인 것이며 정답 라벨이 아니다 (`manifest.json`)
- web_01~10은 원본, web_11~16은 원본이 429로 막혀 Commons 표준 1920px 썸네일로 받음 (`download.sh`)
- 차 앞유리·유리 지붕 근접 사진이 많아 창문 검출이 안 된다 → `analyze photos --whole-image`로 사진 전체를 유리로 분석.
  차체·배경도 유리로 계산되므로 CSI는 참고값이다
- CC BY-SA 사진의 crop·오버레이를 배포하면 같은 라이선스로 공개해야 한다. 웹 결과 화면에 "공개 이미지 예시 · 출처"를 표시한다
