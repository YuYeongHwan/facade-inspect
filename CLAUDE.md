# Facade Inspect — 드론 영상 기반 건물 창문 오염도 분석 웹 시스템

## 목표
건물 촬영 영상을 입력받아
1) 실시간(Live): 프레임마다 창문을 검출해 박스+임시 ID 오버레이
2) 비행 후(Post-flight): 창문별 오염 영역(먼지/얼룩/때/심한오염) 히트맵, A~D 등급,
   건물 박스형 3D 모델 위 창문 패널을 등급 색으로 표시
웹 대시보드에서 창문 클릭 → 확대 이미지 + 오염 영역 오버레이 + 등급 표시

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
- data/, models/, runs/  (gitignore)

## 환경
- Python 3.11, uv 사용. 타입힌트는 3.11 문법(`str | None`)
- 개발 머신은 Mac(Apple Silicon). torch 디바이스 우선순위: cuda > mps > cpu (facade_cv/device.py 공용 함수 사용)
- 학습은 Colab(GPU)에서 notebooks/ 노트북으로 수행, 가중치는 models/에 배치
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
     "grade": "B", "score": 0.054,
     "contamination": [
       {"type": "dust", "area_ratio": 0.021, "polygon": [[0.1,0.2],[0.15,0.2],[0.15,0.25]]}
     ],
     "type_ratios": {"dust": 0.021, "stain": 0.0, "grime": 0.033, "heavy": 0.0},
     "images": {"crop": "windows/F1-R03-C02/crop.jpg", "heatmap": "windows/F1-R03-C02/heatmap.png",
                "overlay": "windows/F1-R03-C02/overlay.png"},
     "num_views": 14}
  ],
  "summary": {"total": 20, "grade_counts": {"A": 8, "B": 6, "C": 4, "D": 2}}
}
```
오염 유형 enum: dust(먼지), stain(얼룩), grime(때), heavy(심한 오염)

## 등급 규칙 (facade_cv/contamination/grade.py)
score = Σ(type_ratio × weight) , weight: dust 0.5, stain 0.8, grime 1.0, heavy 1.5
A < 0.02 ≤ B < 0.08 ≤ C < 0.20 ≤ D  (config/grading.yaml에서 조정 가능)

## 작업 규칙
- 각 모듈은 CLI(`python -m facade_cv.<module> ...`)와 함수 API를 둘 다 제공
- 새 기능마다 pytest 테스트 추가, 무거운 모델 테스트는 `@pytest.mark.slow`
- 설정값은 config/*.yaml, 하드코딩 금지
- 디버그 시각화는 runs/<id>/debug/ 에 저장하는 옵션(--debug) 제공
- 작업 완료 시: 테스트 통과 확인 → 변경 요약 → README 해당 섹션 갱신