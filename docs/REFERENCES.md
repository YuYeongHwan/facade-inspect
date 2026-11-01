# 참고 자료 (References)

프로젝트에서 사용한 외부 자료(데이터·식·코드·수치·문서) 목록. 코드 주석과 문서에서는 `[R번호]`로 인용한다.
새 자료는 번호를 이어서 붙이고, 기존 번호는 바꾸지 않는다 (삭제 시에도 번호는 비워 둔다).
서지 정보를 원문에서 확인하지 못한 항목은 **[확인 필요]**로 표시한다.

> **청소 우선순위 수치에 대한 주의**: CSI(창문 오염 심각도 지수, [docs/scoring.md](scoring.md))의 `config/grading.yaml`
> 종류 가중치 k_t(0.5/0.8/1.0/1.5), 심각도 ΔE 경계(6/15), level 경계(0.009/0.022/0.044/0.089), 창틀 마진(0.06)은
> 아래 문헌의 **구조**(면적 × 상태 등급 × 유형 가중치 / 최댓값, 지수 → 등급 구간)를 빌린 **보정 전 초기값**이다.
> 문헌에서 가져온 수치가 아니며 실측으로 검증되지 않았다 — 사람 평가로 보정한다 (`contamination/calibrate.py`).

## 데이터셋

### [R1] CMP Facade Database
- 저자: Radim Tyleček, Radim Šára
- 연도: 2013
- 제목: Spatial Pattern Templates for Recognition of Objects with Regular Structure
- 학회: Proc. German Conference on Pattern Recognition (GCPR), Saarbrücken
- 링크: https://cmp.felk.cvut.cz/~tylecr1/facade/
- 라이선스: CC BY-SA 3.0 (출처 표시, 동일 조건 변경 허락)
- 사용 부분: 창문/외벽 분할 학습 데이터 606장 (base 378 + extended 228). 12개 클래스 → window/facade 2개 클래스로 변환
  (`cv/facade_cv/detect/datasets/cmp_convert.py`, `config/dataset.yaml` cmp). 세부는 [data/DATASETS.md](../data/DATASETS.md)

### [R19] ADE20K (SceneParsing / ADEChallengeData2016)
- 저자: Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, Antonio Torralba
- 연도·제목:
  - [R19-a] 2017, Scene Parsing through ADE20K Dataset. IEEE CVPR, 633–641
  - [R19-b] 2019, Semantic Understanding of Scenes through the ADE20K Dataset. IJCV, 127(3), 302–321 (Tete Xiao 외 공저 추가)
- 링크: https://groups.csail.mit.edu/vision/datasets/ADE20K/ , http://sceneparsing.csail.mit.edu/
  (데이터: http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip, 923MB)
- 라이선스: 이미지는 **비상업 연구·교육 목적만** (MIT CSAIL terms, 이미지 저작권은 원 소유자), 주석·코드 BSD-3.
  공개 배포·상업 이용 시 별도 허가 필요
- 사용 부분: v2 학습 데이터 — 실외 장면만 골라 windowpane→window, building→facade, tree/grass/plant/flower→vegetation,
  pole→pole, door→door (`cv/facade_cv/detect/datasets/ade20k_convert.py`, `config/dataset.yaml` ade20k). 세부는 data/DATASETS.md
  **v2에서는 제외** (2016판은 외벽 창문·문이 building에 포함). 2021 전체판(instance·part 주석)은 배포 사이트(https://ade20k.csail.mit.edu/request_data) 서버 오류로 받을 수 없음 (2026-10-22)

### [R20] TTPLA
- 저자: Rabab Abdelfattah, Xiaofeng Wang, Song Wang
- 연도: 2020
- 제목: TTPLA: An Aerial-Image Dataset for Detection and Segmentation of Transmission Towers and Power Lines
- 학회: Asian Conference on Computer Vision (ACCV). arXiv:2010.10032
- 링크: https://github.com/r3ab/ttpla_dataset (이미지는 README의 Google Drive 링크)
- 라이선스: Apache-2.0 (GitHub 저장소 LICENSE)
- 사용 부분: v2 학습 데이터 — cable→wire, tower_wooden→pole, 철탑(tower_lattice/tower_tucohy)·void 제외
  (`cv/facade_cv/detect/datasets/ttpla_convert.py`, `config/dataset.yaml` ttpla). 제외 근거는 data/DATASETS.md

### [R21] Roboflow Universe 창문 데이터셋
Roboflow Universe(https://universe.roboflow.com)에 사용자가 올린 데이터셋. 저자·장수·클래스는 각 페이지 기준이며
자동 조회가 막혀(HTTP 403) **[확인 필요: 다운로드 시 페이지에서 라이선스·클래스·버전 재확인]**. CC BY 계열만 사용 (출처 표시 필요)
- [R21-a] Building Facade (workspace `building-facade`), building-facade-segmentation-instance, v4 (2,226장), CC BY 4.0
  — https://universe.roboflow.com/building-facade/building-facade-segmentation-instance (facade, window, vegetation 등)
- [R21-b] ir (workspace `ir-2znee`), Facade_test (382장), CC BY 4.0 — https://universe.roboflow.com/ir-2znee/facade_test (window, door)
- [R21-c] mzworkspace (workspace `mzworkspace-segj7`), window-segment-main, CC BY 4.0
  — https://universe.roboflow.com/mzworkspace-segj7/window-segment-main (window)
- 사용 부분: v2 학습 데이터 창문 보강 (`cv/facade_cv/detect/datasets/roboflow_convert.py`, `config/dataset.yaml` roboflow)

### [R26]~[R28] Roboflow Universe 오염 창문 데이터셋
Roboflow Universe 사용자 업로드 데이터셋 (저자 이름 없음 — "Provided by a Roboflow user"). 서지는 각 export의
`README.roboflow.txt`·`data.yaml` 기준. 라이선스 CC BY 4.0 (출처 표시 필요). **학습에는 쓰지 않고** T7 오염 분석
검증·CSI 보정 평가 이미지로만 쓴다 (`data/eval/t7_dev.csv`, `data/eval/csi_calib.csv`). 세부는 data/DATASETS.md
- [R26] dirty window — workspace `project-novx2`, v1 (2026-02-09), export 2026-05-25, 366장, YOLOv11 polygon,
  클래스 `dirty window` 1개, 전처리·증강 없음 — https://universe.roboflow.com/project-novx2/dirty-window
- [R27] window check — workspace `dirty-window`, v3 (2026-05-23), export 2026-10-21, 865장, YOLOv11 box,
  클래스 dirt·glare·stain·wet_glass, 640×640 stretch 리사이즈, 원본당 3배 증강(좌우 반전, 밝기 ±25%, 블러 0~2px,
  소금·후추 잡음 5%) — https://universe.roboflow.com/dirty-window/window-check
- [R28] window check2 — workspace `dirty-window`, v2 (2026-05-23), export 2026-10-21, 263장, YOLOv11 box,
  클래스·전처리·증강은 [R27]과 같음 — https://universe.roboflow.com/dirty-window/window-check2

### [R29] Street Trees Detection (Roboflow Universe)
- 제공: Roboflow 사용자 (workspace `treedetection-vf07h`, 저자 이름 없음). 버전 v4 (2024-06-05), export 2026-06-30
- 링크: https://universe.roboflow.com/treedetection-vf07h/street-trees-detection
- 라이선스: CC BY 4.0 (출처 표시 필요)
- 내용: 거리 시점 나무 1,477장 (원본 615장 × 증강), YOLOv11, 클래스 `Tree` 1개 — 박스(일부 원본은 거친 polygon)
- 사용 부분: v2 vegetation 후보 — **검토 후 미사용** (박스가 줄기·하단 수관만, data/DATASETS.md). 변환 도구는 SAM([R30])으로 박스에서 polygon 자동 생성
  (`cv/facade_cv/detect/datasets/box_to_seg_sam.py`, `config/dataset.yaml` box_to_seg). 세부는 data/DATASETS.md

### [R30] SAM 2 (Segment Anything Model 2)
- 저자: Nikhila Ravi 외 (Meta AI)
- 연도: 2024
- 제목: SAM 2: Segment Anything in Images and Videos. arXiv:2408.00714
- 코드·가중치: https://github.com/facebookresearch/sam2 (Apache-2.0), Ultralytics 래퍼 `SAM("sam2.1_b.pt")` ([R8])
- 사용 부분: 박스 프롬프트 → 마스크 (`box_to_seg_sam.py`. [R29]에 적용했으나 결과 미사용)

### [R33] Wikimedia Commons 공개 오염 유리 사진 16장
- 제공: Wikimedia Commons 개별 업로더 (아래 표). 수집 2026-11-01. 원본 페이지·작성자·라이선스는 `data/own/dirty_windows/web/SOURCES.csv`·`manifest.json`
- 라이선스: CC0 / CC BY 2.0·3.0·4.0 / CC BY-SA 2.0·3.0·4.0 (사진마다 다름). CC BY-SA 사진을 가공(crop·오버레이)해 배포하면 같은 라이선스로 공개해야 한다
- 사용 부분: **학습 미사용**. 사진 데모(`analyze photos --whole-image` → `runs/demo_web`)와 T7 정성 검증만. 웹·발표에서는 "공개 이미지 예시"로 표기하고 출처를 함께 밝힌다
  (`web/lib/photoSources.ts`). web_11~16은 원본 대신 Commons 1920px 썸네일

| 파일 | 분류 | 작성자 | 라이선스 | 원본 페이지 |
|---|---|---|---|---|
| web_01.jpg | heavy | Sergey Ashmarin | CC BY-SA 3.0 | https://commons.wikimedia.org/wiki/File:Car_with_bird_droppings_-_Rome,_Italy_-_panoramio.jpg |
| web_02.jpg | heavy | Atirador | CC BY 4.0 | https://commons.wikimedia.org/wiki/File:Shitty_Mazda.jpg |
| web_03.jpg | heavy | Guillaume Vachey from Chalon sur Saone, France | CC0 | https://commons.wikimedia.org/wiki/File:Renault_Twingo_(46901672205).jpg |
| web_04.jpg | heavy | XoMEoX | CC BY 2.0 | https://commons.wikimedia.org/wiki/File:Bird%27s_Art_(14971642117).jpg |
| web_05.jpg | heavy | W.carter | CC BY-SA 4.0 | https://commons.wikimedia.org/wiki/File:Three_gulls_on_glass_ceiling,_Kampenhof_bus_terminal,_Uddevalla.jpg |
| web_06.jpg | stain | Daniel Christensen | CC BY-SA 3.0 | https://commons.wikimedia.org/wiki/File:Wells_Fargo_Center_dirty_windows.jpg |
| web_07.jpg | dust | Nenad Stojkovic | CC BY 2.0 | https://commons.wikimedia.org/wiki/File:Dirty_back_car_window_-_49757773438.jpg |
| web_08.jpg | grime | Dejan Krsmanovic | CC BY 2.0 | https://commons.wikimedia.org/wiki/File:Old_Dark_and_Dirty_Window_(40343422273).jpg |
| web_09.jpg | dust | Alexreavis | CC BY-SA 3.0 | https://commons.wikimedia.org/wiki/File:Dusty_Window_View.JPG |
| web_10.jpg | dust | Brylie | CC BY-SA 3.0 | https://commons.wikimedia.org/wiki/File:Smoke_through_a_dusty_window.jpg |
| web_11.jpg | dust | UncleVinny | CC BY 3.0 | https://commons.wikimedia.org/wiki/File:Eiffel_Tower,_dirty_bus_window_2_-_panoramio.jpg |
| web_12.jpg | grime | rachel_bernadette | CC BY-SA 2.0 | https://commons.wikimedia.org/wiki/File:Though_the_dirty_window_-_Flickr_-_rachel_bernadette.jpg |
| web_13.jpg | grime | Conall from Downpatrick, Northern Ireland | CC BY 2.0 | https://commons.wikimedia.org/wiki/File:The_back_window_of_the_palm_house_(52712743837).jpg |
| web_14.jpg | wet_glass | Wuzur | CC BY-SA 3.0 | https://commons.wikimedia.org/wiki/File:Houses_and_cars_seen_through_a_rain-streaked_car_window.jpg |
| web_15.jpg | wet_glass | Ashwini Sukhdeve | CC BY-SA 3.0 | https://commons.wikimedia.org/wiki/File:The_road_to_Lavasa,_seen_through_rain-streaked_glass.jpg |
| web_16.jpg | clean | W.carter | CC BY-SA 4.0 | https://commons.wikimedia.org/wiki/File:East_part_of_glass_ceiling_in_Kampenhof_bus_terminal,_Uddevalla_1.jpg |

## 오염 점수·청소 우선순위

### [R2] Gaspar & de Brito — 외벽 열화 심각도 지수 (Severity of degradation)
- 저자: Pedro Lima Gaspar, Jorge de Brito
- 연도: 2008 (같은 저자의 2008년 논문 두 편)
  - [R2-a] Service life estimation of cement-rendered facades. Building Research & Information, 36(1), 44–55.
    https://doi.org/10.1080/09613210701434164
  - [R2-b] Quantifying environmental effects on cement-rendered facades: A comparison between different degradation indicators.
    Building and Environment, 43(11), 1818–1828. https://doi.org/10.1016/j.buildenv.2007.10.022
- 식: S_w = Σ(A_n · k_n · k_a,n) / (A · k)
  — A_n 결함 n의 면적, k_n 결함 상태 등급(0~4), k_a,n 결함 유형별 상대 가중치, A 전체 면적, k 최고 등급(4).
  [R2-a]는 같은 구조의 Overall Degradation Level(ODL, 결함 면적의 가중 합)로 수명을 추정한다
- **[확인 필요]** 두 편 중 어느 논문이 심각도 지수 식을 처음 정의했는지, k_a,n 원 값. 원문은 도서관에서 확인 예정
- 사용 부분: CSI 식의 구조 그대로 — A_n → 오염 영역 면적 A_j, k_n(상태 등급) → 심각도 k_s(1~3), k_a,n → 종류 가중치 k_t,
  분모 A·k → 유리 면적 A_g × k_s_max × k_t_max (0~1 정규화). (`cv/facade_cv/contamination/grade.py` `compute_csi`,
  `config/grading.yaml`, docs/scoring.md). 등급 수(3)·가중치 수치는 이 논문 값이 아니다

### [R3] (비움 — [R2-a]로 통합)

### [R4] ASTM D6433 — Pavement Condition Index (PCI)
- 저자: ASTM International (Committee E17)
- 연도: 2024 (D6433-24, 최신판. 1999년 초판)
- 제목: Standard Practice for Roads and Parking Lots Pavement Condition Index Surveys
- 발행: ASTM International, Book of Standards Vol. 04.03
- 링크: https://store.astm.org/d6433-24.html
- 사용 부분: 육안 조사 결함(유형·심각도·면적)을 하나의 지수로 합친 뒤 **구간별 등급(라벨)으로 나눠 관리자가 보수 우선순위를
  정하게 하는** 방식. CSI → 청소 우선순위 5단계 구간 (`config/grading.yaml` levels min_csi). PCI의 deduct 곡선·수치는 쓰지 않음

### [R5] Smestad et al. — 오염(soiling)에 의한 투과율 손실
- 저자: Greg P. Smestad, Thomas A. Germer, Hameed Alrashidi, Eduardo F. Fernández 외 (총 18명), Leonardo Micheli
- 연도: 2020
- 제목: Modelling photovoltaic soiling losses through optical characterization
- 저널: Scientific Reports, 10, Article 58
- 링크: https://doi.org/10.1038/s41598-019-56868-z
- 사용 부분: 유리 표면 오염의 투과율 손실이 오염 면적(입자 피복률)에 따라 커지고, 같은 피복률이라도 오염층의 광학적 성질에
  따라 손실이 달라진다는 근거 → CSI의 면적 항 **A_j/A_g**와 심각도 **k_s**(영역의 불투명도를 깨끗한 유리 대비 색차로 근사)
  (`grade.py` `grade_window`, `config/grading.yaml` severity). ΔE를 불투명도 대리로 쓰는 것과 경계값은 자체 설정

### [R6] Willers et al. — 먼지 종류별 유리 투과율 영향
- 저자: G. Willers, N. Sakarapunthip, K. Ilse, S. Chuangchote, R. Gottschalg
- 연도: 2025
- 제목: Impact of Different Types of Dust on Solar Glass Transmittance and PV Module Performance
- 저널: Progress in Photovoltaics: Research and Applications, 33(8), 844–853
- 링크: https://doi.org/10.1002/pip.3930
- 사용 부분: 같은 피복률이라도 오염 종류에 따라 투과율 손실 기울기가 다르다는 결과 → CSI의 종류 가중치 **k_t**를 두는 근거
  (`config/grading.yaml` weights). 가중치 수치는 이 논문 값이 아니다 — 사람 평가로 보정 (`calibrate.py`)

### [R7] Lee et al. — 유리창 청소 장치용 영상 기반 먼지 검출
- 저자: Jiseok Lee, Hobyeong Chae, KyungMin Kim, Hwa Soo Kim, TaeWon Seo
- 연도: 2022
- 제목: Detection method for transparent window cleaning device, image processing approach
- 저널: Scientific Reports, 12, Article 3229
- 링크: https://doi.org/10.1038/s41598-022-07235-y
- 사용 부분: 관련 연구 — 커튼월 유리의 먼지 밀도를 영상으로 추정해 청소 로봇에 쓰는 선행 사례. 밝기·배경에 민감한 문제
  지적은 "유리에 고정된 패턴만 오염"(다시점 정렬) 설계 원칙의 배경. 코드에서 직접 쓴 식 없음.
  인용 위치: docs/scoring.md §5 "오염 영역 검출과 유형 분류", contamination/analyze.py 모듈 설명

## 모델·라이브러리·도구

### [R8] Ultralytics YOLO11
- 저자: Glenn Jocher, Jing Qiu (Ultralytics)
- 연도: 2024
- 제목: Ultralytics YOLO11 (version 11.0.0, 소프트웨어 — 논문 없음)
- 링크: https://github.com/ultralytics/ultralytics , https://docs.ultralytics.com/models/yolo11/
- 라이선스: AGPL-3.0
- 사용 부분: `yolo11s-seg.pt` 사전학습 가중치에서 창문/외벽 인스턴스 분할 미세조정 (`cv/notebooks/train_seg.ipynb`),
  추론 래퍼 (`cv/facade_cv/detect/segmenter.py`), 학습 증강 하이퍼파라미터 (`config/augment.yaml`, `detect/augment.py`)

### [R9] COLMAP — Structure-from-Motion
- 저자: Johannes L. Schönberger, Jan-Michael Frahm
- 연도: 2016
- 제목: Structure-from-Motion Revisited
- 학회: IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 링크: https://colmap.github.io/ , https://github.com/colmap/colmap
- 라이선스: BSD-3-Clause (pycolmap 포함)
- 사용 부분: 프레임 → 특징 추출·매칭 → incremental mapping으로 카메라 포즈·sparse 포인트 복원
  (`cv/facade_cv/recon3d/sfm.py`, `config/recon.yaml` sfm). 카메라 좌표계 규약 (`recon3d/geometry.py`)

### [R10] ArduPilot 문서
- 저자: ArduPilot Dev Team
- 연도: 온라인 문서 (2026-10-21 열람)
- 제목: ArduPilot Documentation
- 링크: https://ardupilot.org/
- 사용 부분: 자동 비행 설계 (`flight/`, `docs/hardware/`)에서 참고
  - [R10-a] Communicating with Raspberry Pi via MAVLink — https://ardupilot.org/dev/docs/raspberry-pi-via-mavlink.html
  - [R10-b] Benewake TFmini / TFmini Plus lidar — https://ardupilot.org/plane/docs/common-benewake-tfmini-lidar.html
  - [R10-c] Guided Mode (Copter) — https://ardupilot.org/copter/docs/ac2_guidedmode.html

### [R11] Hartley & Zisserman — 다시점 기하
- 저자: Richard Hartley, Andrew Zisserman
- 연도: 2004 (2nd ed.)
- 제목: Multiple View Geometry in Computer Vision — 8장 "More Single View Geometry"의 소실점·소실선 절
  **[확인 필요: 절 번호]**
- 발행: Cambridge University Press
- 사용 부분: 직교 소실점 두 개로 카메라 회전·초점거리 추정 (v_h·v_v = −f², 주점 = 이미지 중심) → 정면화 호모그래피
  H = K Rᵀ K⁻¹ (`cv/facade_cv/rectify/homography.py`). 평면 유도 호모그래피 (`recon3d/ortho.py`)

### [R12] LSD 선분 검출기
- 저자: Rafael Grompone von Gioi, Jérémie Jakubowicz, Jean-Michel Morel, Gregory Randall
- 연도: 2010
- 제목: LSD: A Fast Line Segment Detector with a False Detection Control
- 저널: IEEE Transactions on Pattern Analysis and Machine Intelligence, 32(4), 722–732
- 사용 부분: 정면화용 수평/수직 선분 검출 (`cv2.createLineSegmentDetector`, `rectify/homography.py` `find_axis_lines`)

### [R13] RANSAC
- 저자: Martin A. Fischler, Robert C. Bolles
- 연도: 1981
- 제목: Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography
- 저널: Communications of the ACM, 24(6), 381–395
- 사용 부분: 소실점 추정 (`rectify/homography.py`), 외벽 평면 검출 (`recon3d/planes.py`), 프레임 간 호모그래피
  (`io/extract.py`, `cv2.findHomography(..., cv2.RANSAC)`)

### [R14] ORB 특징
- 저자: Ethan Rublee, Vincent Rabaud, Kurt Konolige, Gary Bradski
- 연도: 2011
- 제목: ORB: An efficient alternative to SIFT or SURF
- 학회: IEEE International Conference on Computer Vision (ICCV)
- 사용 부분: 프레임 추출 시 직전 프레임과의 겹침(중복) 판정 (`io/extract.py`, `config/extract.yaml` dedup)

### [R15] Laplacian 분산 초점(블러) 척도
- 저자: J. L. Pech-Pacheco, G. Cristóbal, J. Chamorro-Martínez, J. Fernández-Valdivia
- 연도: 2000
- 제목: Diatom autofocusing in brightfield microscopy: a comparative study
- 학회: Proc. 15th International Conference on Pattern Recognition (ICPR), vol. 3, 314–317 **[확인 필요: 페이지]**
- 사용 부분: 흐린 프레임 제거 점수 = Laplacian 분산 (`io/extract.py`, `config/extract.yaml` blur)

### [R16] SIFT
- 저자: David G. Lowe
- 연도: 2004
- 제목: Distinctive Image Features from Scale-Invariant Keypoints
- 저널: International Journal of Computer Vision, 60(2), 91–110
- 사용 부분: COLMAP 특징 추출 (`recon3d/sfm.py`, `config/recon.yaml` max_num_features)

### [R17] Albumentations
- 저자: Alexander Buslaev, Vladimir I. Iglovikov, Eugene Khvedchenya, Alex Parinov, Mikhail Druzhinin, Alexandr A. Kalinin
- 연도: 2020
- 제목: Albumentations: Fast and Flexible Image Augmentations
- 저널: Information, 11(2), 125
- 링크: https://doi.org/10.3390/info11020125
- 사용 부분: Ultralytics에 없는 픽셀 증강 (대비, 모션블러) (`config/augment.yaml` albumentations, `detect/augment.py`)

### [R18] RadioLink CrossRace 비행 컨트롤러 공식 문서
- 저자: RadioLink
- 연도: 온라인 문서 (2026-10-21 열람)
- 제목: CrossRace 제품 페이지 / CrossRace User Manuals
- 사용 부분: 자동 비행 설계 (`flight/`, `docs/hardware/`)에서 FC 하드웨어 참고
  - [R18-a] https://www.radiolink.com/crossrace — APM(ArduPilot) 기반, TELEM 포트 구성 (TELEM2 텔레메트리, TELEM3 GPS 등)
  - [R18-b] https://www.radiolink.com/crossrace_manual — Mavlink UART 2개 (TELEM1·2 사용자 지정 시리얼), Guided 모드 (3.7.8절)

### [R31] ECC 영상 정렬 (Enhanced Correlation Coefficient)
- 저자: Georgios D. Evangelidis, Emmanouil Z. Psarakis
- 연도: 2008
- 제목: Parametric Image Alignment Using Enhanced Correlation Coefficient Maximization. IEEE TPAMI, 30(10), 1858–1865.
  https://doi.org/10.1109/TPAMI.2008.113
- 구현: OpenCV `cv2.findTransformECC` (MOTION_HOMOGRAPHY, inputMask)
- 사용 부분: 다시점 오염 분석에서 같은 창문의 정면화 관측들을 호모그래피로 정밀 정렬 — 상관은 창틀 띠에서만
  (`cv/facade_cv/contamination/multiview.py` `ecc_align`, `config/contamination.yaml` multiview, docs/scoring.md §5 다시점 모드)

### [R32] LSD 직선 검출 (Line Segment Detector)
- 저자: Rafael Grompone von Gioi, Jérémie Jakubowicz, Jean-Michel Morel, Gregory Randall
- 연도: 2010
- 제목: LSD: A Fast Line Segment Detector with a False Detection Control. IEEE TPAMI, 32(4), 722–732.
  https://doi.org/10.1109/TPAMI.2008.300
- 구현: OpenCV `cv2.createLineSegmentDetector`
- 사용 부분: 단일 이미지 반사 의심 규칙의 긴 직선 밀도 (`analyze.py` `scene_structure`, `config/contamination.yaml` reflection)

### [R34] 어두운 채널 사전 (Dark Channel Prior) 안개 추정
- 저자: Kaiming He, Jian Sun, Xiaoou Tang
- 연도: 2009 (CVPR), 확장판 2011
- 제목: Single Image Haze Removal Using Dark Channel Prior. CVPR 2009 / IEEE TPAMI 33(12), 2341–2353 (2011)
- 링크: https://doi.org/10.1109/TPAMI.2010.168
- 사용 부분: T7 2차 먼지막(film) 시도 — haze = 1 − t, t = 1 − ω·dark(I/A), 패치 15 px·ω 0.95·t0 0.1은 논문 기본값
  (`analyze.film_map`, `config/contamination.yaml` film). **효과 없어 꺼 둠** (docs/scoring.md "T7 2차")

### [R35] CLIP (Contrastive Language-Image Pre-training)
- 저자: Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever
- 연도: 2021
- 제목: Learning Transferable Visual Models From Natural Language Supervision. ICML 2021 (PMLR 139, 8748–8763)
- 링크: https://arxiv.org/abs/2103.00020
- 사용 부분: 제로샷 분류 방식 — 클래스마다 프롬프트 여러 개의 텍스트 임베딩 평균(prompt ensembling, 논문 3.1.4),
  이미지·텍스트 코사인 유사도 × 온도 100의 softmax. 창문 크롭의 p(dirty) = "더러운 창" 대 "깨끗한 창" 확률
  (`contamination/clip_score.py`, `config/contamination.yaml` clip). 프롬프트 6개는 자체 작성 (데이터를 보기 전 고정)

### [R36] OpenCLIP / LAION-2B — ViT-B-32 laion2b_s34b_b79k 가중치
- 저자: Gabriel Ilharco, Mitchell Wortsman, Ross Wightman 외 (OpenCLIP); Christoph Schuhmann 외 (LAION-5B); Mehdi Cherti 외 (재현 실험)
- 연도: 2021 (코드), 2022 (LAION-5B, NeurIPS Datasets and Benchmarks), 2023 (Cherti et al., CVPR)
- 제목: OpenCLIP (https://github.com/mlfoundations/open_clip, MIT 라이선스) · LAION-5B: An open large-scale dataset for training next generation
  image-text models · Reproducible scaling laws for contrastive language-image learning
- 링크: https://doi.org/10.5281/zenodo.5143773 · https://arxiv.org/abs/2210.08402 · https://arxiv.org/abs/2212.07143
- 사용 부분: `open_clip_torch` 패키지 (선택 의존성 `uv sync --extra clip`)와 사전학습 가중치 ViT-B-32 `laion2b_s34b_b79k`
  (LAION-2B 영어 부분, 34B 샘플 학습, ~600 MB, Hugging Face에서 받음). 미세조정 없음

## 색차·평가 일치도 (CSI 심각도·보정)

### [R22] CIE 15:2004 — Colorimetry (CIELAB, ΔE*ab)
- 저자: Commission Internationale de l'Éclairage (CIE)
- 연도: 2004 (3rd ed.)
- 제목: CIE 15:2004 Colorimetry
- 발행: CIE Central Bureau, Vienna. ISBN 978-3-901906-33-6 **[확인 필요: ISBN]**
- 사용 부분: CIELAB 색공간과 CIE 1976 색차 ΔE*ab = √(ΔL*² + Δa*² + Δb*²) — 오염 영역 평균색과 깨끗한 유리 평균색의 차
  (`grade.py` `lab`, `delta_e`). 변환은 OpenCV `cvtColor(BGR2LAB)` (D65, sRGB 가정)

### [R23] Mahy, Van Eycken, Oosterlinck — 지각 가능 최소 색차
- 저자: Marc Mahy, Luc Van Eycken, André Oosterlinck
- 연도: 1994
- 제목: Evaluation of Uniform Color Spaces Developed after the Adoption of CIELAB and CIELUV
- 저널: Color Research & Application, 19(2), 105–121 **[확인 필요: 페이지]**
- 사용 부분: CIELAB에서 사람이 겨우 구분하는 색차(JND) ≈ 2.3 — 심각도 경계 ΔE 6(≈ 2.5 JND)을 정한 기준점
  (`config/grading.yaml` severity.delta_e). 6·15라는 값 자체는 자체 초기값

### [R24] Cohen — 가중 카파 (weighted kappa)
- 저자: Jacob Cohen
- 연도: 1968
- 제목: Weighted kappa: Nominal scale agreement with provision for scaled disagreement or partial credit
- 저널: Psychological Bulletin, 70(4), 213–220. https://doi.org/10.1037/h0026256
- 사용 부분: 순서 등급(1~5) 일치도 — 이차 가중치 w_ij = (i−j)²/(n−1)² (quadratic weighted kappa).
  평가자 간 일치도, CSI level ↔ 사람 합의 등급 일치도, 보정 목적함수 (`contamination/calibrate.py` `qwk`)

### [R25] Landis & Koch — 카파 해석 기준
- 저자: J. Richard Landis, Gary G. Koch
- 연도: 1977
- 제목: The Measurement of Observer Agreement for Categorical Data
- 저널: Biometrics, 33(1), 159–174. https://doi.org/10.2307/2529310
- 사용 부분: 카파 해석 구간 (0.41–0.60 moderate, 0.61–0.80 substantial, 0.81– almost perfect) — 보정 결과 판독 (docs/scoring.md)
