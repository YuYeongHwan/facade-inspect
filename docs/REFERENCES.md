# 참고 자료 (References)

프로젝트에서 사용한 외부 자료(데이터·식·코드·수치·문서) 목록. 코드 주석과 문서에서는 `[R번호]`로 인용한다.
새 자료는 번호를 이어서 붙이고, 기존 번호는 바꾸지 않는다 (삭제 시에도 번호는 비워 둔다).
서지 정보를 원문에서 확인하지 못한 항목은 **[확인 필요]**로 표시한다.

> **청소 우선순위 수치에 대한 주의**: `config/grading.yaml`의 가중치(0.5/0.8/1.0/1.5)와 level 경계(0.02/0.05/0.10/0.20)는
> 아래 문헌의 **구조**(면적 × 심각도 가중 합, 지수 → 등급 구간)를 빌린 **자체 설정값**이다. 문헌에서 가져온 수치가 아니며
> 실측 데이터로 검증되지 않았다.

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
- 사용 부분: `score = Σ(type_ratio × weight)`의 구조 — 오염 유형별 **면적 비율 × 유형 가중치**의 합
  (`cv/facade_cv/contamination/grade.py` `compute_score`, `config/grading.yaml` weights). 유형 가중치 수치는 이 논문 값이 아니다

### [R3] (비움 — [R2-a]로 통합)

### [R4] ASTM D6433 — Pavement Condition Index (PCI)
- 저자: ASTM International (Committee E17)
- 연도: 2024 (D6433-24, 최신판. 1999년 초판)
- 제목: Standard Practice for Roads and Parking Lots Pavement Condition Index Surveys
- 발행: ASTM International, Book of Standards Vol. 04.03
- 링크: https://store.astm.org/d6433-24.html
- 사용 부분: 육안 조사 결함(유형·심각도·면적)을 하나의 지수로 합친 뒤 **구간별 등급(라벨)으로 나눠 관리자가 보수 우선순위를
  정하게 하는** 방식. 청소 우선순위 5단계 구간 설계에 참고 (`config/grading.yaml` levels). PCI의 deduct 곡선·수치는 쓰지 않음

### [R5] Smestad et al. — 오염(soiling)에 의한 투과율 손실
- 저자: Greg P. Smestad, Thomas A. Germer, Hameed Alrashidi, Eduardo F. Fernández 외 (총 18명), Leonardo Micheli
- 연도: 2020
- 제목: Modelling photovoltaic soiling losses through optical characterization
- 저널: Scientific Reports, 10, Article 58
- 링크: https://doi.org/10.1038/s41598-019-56868-z
- 사용 부분: 유리 표면 오염의 투과율 손실이 오염 면적(입자 피복률)에 따라 커진다는 근거 → 오염 정도의 기본 척도로
  **면적 비율(area_ratio)**을 사용 (`schema.py` contamination.area_ratio, `grade.py`)

### [R6] Willers et al. — 먼지 종류별 유리 투과율 영향
- 저자: G. Willers, N. Sakarapunthip, K. Ilse, S. Chuangchote, R. Gottschalg
- 연도: 2025
- 제목: Impact of Different Types of Dust on Solar Glass Transmittance and PV Module Performance
- 저널: Progress in Photovoltaics: Research and Applications, 33(8), 844–853
- 링크: https://doi.org/10.1002/pip.3930
- 사용 부분: 같은 피복률이라도 오염 종류에 따라 투과율 손실 기울기가 다르다는 결과 → 오염 **유형별 가중치**를 두는 근거
  (`config/grading.yaml` weights). 가중치 수치는 이 논문 값이 아니다

### [R7] Lee et al. — 유리창 청소 장치용 영상 기반 먼지 검출
- 저자: Jiseok Lee, Hobyeong Chae, KyungMin Kim, Hwa Soo Kim, TaeWon Seo
- 연도: 2022
- 제목: Detection method for transparent window cleaning device, image processing approach
- 저널: Scientific Reports, 12, Article 3229
- 링크: https://doi.org/10.1038/s41598-022-07235-y
- 사용 부분: 관련 연구 — 커튼월 유리의 먼지 밀도를 영상으로 추정해 청소 로봇에 쓰는 선행 사례. 밝기·배경에 민감한 문제
  지적은 "유리에 고정된 패턴만 오염"(다시점 정렬) 설계 원칙의 배경. 코드에서 직접 쓴 식 없음

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
