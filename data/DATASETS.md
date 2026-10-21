# 데이터셋

`data/`는 gitignore (이 파일만 추적). 서지·라이선스 원문은 [docs/REFERENCES.md](../docs/REFERENCES.md)의 [R번호].
학습에 쓰는 이미지가 늘거나 바뀌면 이 표를 갱신한다.

## 목록

| 데이터셋 | 출처 | 라이선스 | 원본 장수 | 사용 장수 (용도) | 위치 |
|---|---|---|---|---|---|
| CMP Facade Database — base | Tyleček & Šára, GCPR 2013 [R1] <br> https://cmp.felk.cvut.cz/~tylecr1/facade/ | CC BY-SA 3.0 | 378 | 378 (분할 학습) | `data/cmp/base` |
| CMP Facade Database — extended | 〃 [R1] | CC BY-SA 3.0 | 228 | 228 (분할 학습) | `data/cmp/extended` |
| b01_songsan (자체 촬영) | 2026-10-07, 지상 정지 촬영, 맑음 | 자체 소유 | 영상 1개 | 0 (학습 미사용) | `data/raw/videos/b01_songsan` |
| b02_smilen (자체 촬영) | 2026-10-07, 지상, 올려다보는 각도 | 자체 소유 | 영상 1개 | 0 (학습 미사용) | `data/raw/videos/b02_smilen` |
| b03_shopglass (자체 촬영) | 2026-10-07, 근접, 반사 심함·오염 있음 | 자체 소유 | 영상 1개 | 0 (학습 미사용) | `data/raw/videos/b03_shopglass` |
| b04_shcool (자체 촬영) | 2026-10-14, 1층 한 바퀴 이동 촬영 | 자체 소유 | 영상 1개 | 0 (학습) / 프레임 80·266장 (3D 복원 테스트), 실시간 검출 테스트 | `data/raw/videos/b04_shcool` |

- CMP 분할 (`data/yolo/cmp_only`, `config/dataset.yaml` split.cmp, seed 0): train 545 / val 61 / test 0
- 자체 촬영 메타데이터(날짜·날씨·각도·비고): `data/raw/videos/captures.csv`. 촬영 장비: 아이폰 (`config/recon.yaml` sfm.camera_model 주석)
- 자체 영상은 아직 Label Studio 라벨링 전 → 현재 가중치 `seg_window_facade_v1.pt`는 **CMP만으로 학습**
- b04 3D 테스트 프레임: `runs/recon_b04_wall` 80장 (흰 외벽 구간), `runs/recon_b04_full` 266장 (2 fps, 전체)
- CC BY-SA 3.0: CMP에서 파생한 YOLO 라벨(`data/yolo/cmp*`)을 배포할 때는 출처 표시 + 같은 라이선스 필요
- 자체 영상에는 행인·차량이 찍혀 있다. 외부 공개 시 얼굴·번호판 처리 필요

## 클래스 변환 규칙 → YOLO-seg (0=window, 1=facade)

설정: `config/dataset.yaml`. 변환 코드: `cv/facade_cv/detect/datasets/cmp_convert.py`, `label_studio/export_to_yolo.py`

| 데이터셋 | 원본 클래스 | → window (0) | → facade (1) | 제외 |
|---|---|---|---|---|
| CMP [R1] | 12개 (PNG 팔레트 index 1~12, `label_names.txt`) | XML 사각형(`window`) 하나 ∩ 픽셀 클래스 {window, blind} → 인스턴스 1개. 붙은 창문 분리, 덧문으로 갈라진 창문 복원 | {facade, molding, cornice, pillar, sill, deco, blind(창문 사각형 밖)} → 구멍을 채운 외곽 polygon 하나 | background, door, shop, balcony |
| 자체 촬영 (b01~b04) | Label Studio polygon: window, facade (`labeling_config.xml`) | window 그대로 | facade 그대로 | — |

공통: 이미지 면적 대비 window < 0.0001, facade < 0.0005인 인스턴스는 버림. polygon 단순화 approxPolyDP eps 0.005 (윤곽 길이 대비).
CMP에서 창문이 빠지는 경우: PNG에서 balcony/deco/shop에 완전히 가려진 창문 (예: `cmp_x0173`).
