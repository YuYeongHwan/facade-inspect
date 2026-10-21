# Facade Inspect — 드론 영상 기반 건물 창문 오염도 분석 웹 시스템

## 목표
건물 촬영 영상을 입력받아
1) 실시간(Live): 프레임마다 창문을 검출해 박스+임시 ID 오버레이
2) 비행 후(Post-flight): 창문별 오염 영역(먼지/얼룩/때/심한오염) 히트맵, 청소 우선순위 level 1~5,
   건물 박스형 3D 모델 위 창문 패널을 level 색으로 표시
웹 대시보드에서 창문 클릭 → 확대 이미지 + 오염 영역 오버레이 + 청소 우선순위 표시

## 핵심 설계 원칙
- 창문/외벽 구분에 색(HSV 등) 임계값 사용 금지. 학습 기반 분할(YOLO11-seg) + 기하 후처리만 사용
- 오염 = 유리 표면에 고정된 패턴. 다시점 정렬 후 "시점이 바뀌어도 위치가 고정된 이상"만 오염으로 판정
  (반사·실내 풍경은 시차로 이동하므로 제외)
- 3D는 사실적 재구성이 아니라 "평면 기반 박스 메시 + 외벽 정사영 텍스처 + 창문 사각 패널"
- 최종 창문 ID는 3D 외벽 평면 위의 (facade, row, col)로 결정. 2D 추적 ID는 임시값

## 레포 구조
- cv/        파이썬 패키지 `facade_cv` (영상처리·학습·3D·오염 분석)
  - facade_cv/io, detect, track, rectify, contamination, recon3d, pipeline.py
  - tests/, notebooks/(Colab 학습)
- server/    FastAPI 앱 (`app/`)
- web/       Next.js 프론트
- fixtures/  목업 result.json 및 샘플 이미지 (프론트/백엔드 병렬 개발용)
- data/, runs/, cv/weights/  (gitignore)

## 환경
- Python 3.11, uv 사용. 타입힌트는 3.11 문법(`str | None`)
- 개발 머신은 Mac(Apple Silicon). torch 디바이스 우선순위: cuda > mps > cpu (facade_cv/device.py 공용 함수 사용)
- 학습은 Colab(GPU)에서 notebooks/ 노트북으로 수행, 가중치는 드라이브 facade-inspect/weights/ → 로컬 cv/weights/에 배치
- 프론트: pnpm, TypeScript strict

## 데이터 계약: runs/<inspection_id>/result.json
모든 모듈은 이 스키마를 기준으로 입출력한다. 스키마는 cv/facade_cv/schema.py (pydantic)에 정의하고
server와 web(TS 타입)은 이를 따른다. 좌표 규칙:
- 이미지 좌표: 픽셀, 원점 좌상단
- 창문 crop 내부 좌표(u,v): 0~1 정규화
- 3D 좌표: 미터 단위(스케일 미상이면 상대 단위, meta.scale_known=false), Y-up

```json
{
  "inspection_id": "insp_001",
  "meta": {"video": "input.mp4", "fps": 30, "frames_used": 240, "scale_known": false, "created_at": "..."},
  "building": {
    "mesh": "building.glb",
    "facades": [
      {"id": "F1", "normal": [0,0,1], "origin": [0,0,0], "u_axis": [1,0,0], "v_axis": [0,1,0],
       "width": 12.0, "height": 24.0, "texture": "facades/F1.jpg"}
    ]
  },
  "windows": [
    {"id": "F1-R03-C02", "facade_id": "F1", "row": 3, "col": 2,
     "facade_rect": [u0, v0, u1, v1],
     "corners_3d": [[x,y,z],[x,y,z],[x,y,z],[x,y,z]],
     "cleaning_level": 3, "cleaning_label": "가급적 청소", "csi": 0.031,
     "glass_ratio": 0.77, "occluded_ratio": 0.05,
     "contamination": [
       {"type": "dust", "area_ratio": 0.021, "severity": 2, "delta_e": 8.4,
        "polygon": [[0.1,0.2],[0.15,0.2],[0.15,0.25]]}
     ],
     "type_ratios": {"dust": 0.021, "stain": 0.0, "grime": 0.033, "heavy": 0.0},
     "images": {"crop": "windows/F1-R03-C02/crop.jpg", "heatmap": "windows/F1-R03-C02/heatmap.png",
                "overlay": "windows/F1-R03-C02/overlay.png"},
     "num_views": 14}
  ],
  "summary": {"total": 20, "level_counts": {"1": 6, "2": 5, "3": 4, "4": 3, "5": 2}}
}
```
오염 유형 enum: dust(먼지), stain(얼룩), grime(때), heavy(심한 오염)
미분석 창문 (오염 분석 T7 전, recon3d 출력): cleaning_level=null, cleaning_label="미분석", csi=null,
glass_ratio=null, occluded_ratio=null, type_ratios=null, images.heatmap/overlay=null, contamination=[].
summary.level_counts에 "unanalyzed" 키. web은 회색 패널
분석된 창문이 있으면 meta.csi_params (계산에 쓴 k_t·심각도 경계·창틀 마진·level 경계 스냅샷) 필수
meta.recon_status (선택): {"mode": "3d"|"degraded", "num_facades", "registered_ratio", "warnings": [...]}

## 청소 우선순위 규칙 (facade_cv/contamination/grade.py, docs/scoring.md)
관리자가 라벨만 보고 청소 여부를 판단하도록 5단계. 지표는 창문 오염 심각도 지수 CSI (0~1):
CSI = Σ_j (A_j × k_s(j) × k_t(j)) / (A_g × k_s_max × k_t_max)
- A_g 유리 면적 = 창문 crop − 창틀 마진 − 가림(occlusion), A_j 오염 영역 중 유리 안 면적 [R2][R5]
- k_s 심각도 1~3 = 깨끗한 유리 대비 CIELAB ΔE*ab (<6 / 6~15 / ≥15) [R5][R22][R23]
- k_t 종류 가중치 dust 0.5, stain 0.8, grime 1.0, heavy 1.5 [R6]. 구조 [R2], 구간 방식 [R4]
모든 수치는 **보정 전 초기값** (contamination/calibrate.py로 사람 평가에 맞춰 보정). 임계값·라벨·색은 config/grading.yaml
(web/lib/palette.ts와 동기, 테스트로 검사)

| level | 라벨 | CSI | 색 |
|---|---|---|---|
| 1 | 청소 불필요 | < 0.009 | #22c55e |
| 2 | 경과 관찰 | 0.009 ~ 0.022 | #84cc16 |
| 3 | 가급적 청소 | 0.022 ~ 0.044 | #eab308 |
| 4 | 청소 필요 | 0.044 ~ 0.089 | #f97316 |
| 5 | 우선 청소 | ≥ 0.089 | #ef4444 |

경계값은 상위 level에 포함 (CSI 0.022 → level 3)

## 작업 규칙
- 각 모듈은 CLI(`python -m facade_cv.<module> ...`)와 함수 API를 둘 다 제공
- 새 기능마다 pytest 테스트 추가, 무거운 모델 테스트는 `@pytest.mark.slow`
- 설정값은 config/*.yaml, 하드코딩 금지
- 디버그 시각화는 runs/<id>/debug/ 에 저장하는 옵션(--debug) 제공
- 외부 자료(데이터·식·코드·수치)를 쓰면 docs/REFERENCES.md에 등록하고 코드 주석과 문서에 [R번호]로 표시
  (데이터셋은 data/DATASETS.md에도 출처·라이선스·사용 장수·클래스 변환 규칙 기록)
- 작업 완료 시: 테스트 통과 확인 → 변경 요약 → README 해당 섹션 갱신