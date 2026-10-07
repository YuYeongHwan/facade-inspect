# web — Facade Inspect 대시보드

Next.js 16 (App Router, TypeScript strict, Tailwind v4) + @react-three/fiber/drei + zod. 다크 테마.
백엔드가 생기기 전까지는 `fixtures/`, `runs/`의 결과 파일을 `public/data/`로 복사해 정적 파일로 읽는다.

## 실행

```bash
cd web
pnpm install            # pnpm이 없으면: npx pnpm@latest install (이하 동일)
pnpm sync-data          # fixtures/ + runs/ → public/data/ (데이터가 바뀔 때마다 다시)
pnpm dev                # http://localhost:3000
pnpm build && pnpm start
pnpm typecheck && pnpm lint
```

| 화면 | 데이터 |
|---|---|
| `/` | 동기화된 검사 결과·검출 실행 링크 (임시 첫 화면) |
| `/inspections/[id]` | `public/data/inspections/<id>/result.json` (목업: `/inspections/insp_mock`) |
| `/detections?run=<run>&frame=<image>` | `public/data/detections/<run>/detections.json` + 프레임·정면화 이미지 |

## 데이터

- **스키마**: `lib/types.ts` — `cv/facade_cv/schema.py`(result.json)와 `detect/postprocess.py`(detections.json)를
  zod로 옮긴 것. `.strict()`(extra=forbid), ID·facade_rect·summary 검사까지 같다. **스키마를 바꾸면 둘 다 고친다.**
- **로딩**: `lib/api.ts` 한 곳. 모든 fetch는 zod로 검증하고, 실패하면 화면에 어떤 필드가 틀렸는지 표시한다.
  백엔드로 바꿀 땐 `NEXT_PUBLIC_DATA_BASE`(기본 `/data`)를 API URL로 바꾸고, 경로가 다르면 `*Url` 함수만 고친다.
- **동기화**: `scripts/sync-data.mjs`
  - `fixtures/result.json`, `runs/**/result.json` → `inspections/<inspection_id>/` (result.json이 참조하는 mesh·texture·창문 이미지만)
  - `runs/**/detect/detections.json` → `detections/<run>/` (+ `<run>/frames/<image>`, `<run>/debug/detect/<stem>_rectified.jpg`).
    `<run>`은 runs/ 기준 경로 (예: `t1_check/20261007_b01_ground_01`)
  - 각 폴더에 `index.json` 생성. `public/data/`는 gitignore
  - `inspections/index.json`의 `source`: fixtures/ 출처는 `"mock"`, runs/ 출처는 `"run"`.
    결과 페이지는 `mock`이면 상단에 **목업 데이터 · 오염 분석 미구현** 배지를 항상 띄운다.
    `/detections`는 실제 파이프라인 출력이라 **실제 검출 결과 · 모델 vN** 배지 (N은 detections.json `weights` 파일명에서)

## 컴포넌트

```
app/
  layout.tsx                  헤더 내비, 다크 테마
  page.tsx                    → components/Home.tsx
  inspections/[id]/page.tsx   → InspectionView
  detections/page.tsx         → DetectionsView (useSearchParams라 Suspense로 감쌈)
components/
  ui.tsx                      GradeBadge, ToggleChip, Status(로딩/에러)
  inspection/
    InspectionView.tsx        상태(선택·hover·등급 표시) 보유, 레이아웃 (모바일: 3D 위·리스트 아래)
    SummaryBar.tsx            전체 수, 평균 점수, 등급 분포 막대, 등급별 표시 토글
    BuildingViewer.tsx        R3F: building.glb + 외벽 텍스처 평면 + 창문 패널(등급 색, 법선 방향으로 띄움),
                              hover 툴팁, 클릭 시 창문 정면으로 카메라 이동(CameraRig), 전체 보기
    WindowList.tsx            정렬(ID·층·등급·점수)·검색, 3D와 선택/hover 연동
    WindowDetail.tsx          drawer: crop/히트맵/오염 polygon(SVG) 레이어 토글, polygon hover 유형·면적비,
                              유형별 면적 막대, 등급 배지, 관측 뷰 수 (Esc로 닫기)
  detections/
    DetectionsView.tsx        영상·프레임 선택, 프레임 위 SVG 사각형(확정/플래그/보완 후보/제거됨 토글),
                              row/col 라벨, 정면화 이미지 나란히, 후처리 전→후 개수·경고
lib/
  types.ts  api.ts  palette.ts(등급·오염 유형 색/라벨)  useData.ts(클라이언트 로딩 훅)
```

- 층 = `row` (1부터, 위로 증가). 등급 색: A `#22c55e` B `#eab308` C `#f97316` D `#ef4444`
- 오염 유형 색은 등급 색과 겹치지 않게: 먼지 slate, 얼룩 sky, 때 violet, 심한 오염 pink (`lib/palette.ts`)
- 이미지가 정적·원본 크기 그대로라 `next/image` 대신 `<img>`를 쓴다 (SVG 오버레이 좌표를 원본 픽셀에 맞추기 위함)
