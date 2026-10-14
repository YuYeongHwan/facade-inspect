# 드론 탑재 카메라 — Raspberry Pi Zero 2 W + Camera Module 3

> **미검증.** 하드웨어 없이 작성했다. 처음 설치할 때 각 단계 결과를 확인하고 틀린 곳은 이 문서를 고친다.
> 특히 MediaMTX 설정 키, rpicam-vid 옵션, BEC 배선은 실제 장비에서 확인 전까지 믿지 말 것.

```
Camera Module 3 ─ rpicam-vid (H.264 한 번) ─ ffmpeg -c copy ─┬─ SD: ~/recordings/<시각>/seg_*.ts   (비행 후 분석)
                  └──── facade-stream.service ─────────────────┘└─ localhost MediaMTX ─┬─ Wi-Fi RTSP  rtsp://<파이IP>:8554/cam
                                                                                         └─ MODE=tcp: facade-tcp.service
                                                                                              → Wi-Fi tcp://<파이IP>:8888
노트북: python -m facade_cv.live.run --source rtsp://<파이IP>:8554/cam   (또는 tcp://<파이IP>:8888)
```

| 파일 | 위치 (파이) |
|---|---|
| `stream.sh`, `tcp_relay.sh`, `stream.env`, `mediamtx.yml` | `/opt/facade-inspect/` |
| `facade-stream.service`, `mediamtx.service`, `facade-tcp.service` | `/etc/systemd/system/` |

## 녹화 동작 (MODE와 무관)

- **SD 녹화는 카메라가 시작되는 순간부터 계속 돈다.** 노트북이 연결돼 있는지, Wi-Fi가 살아 있는지와 상관없다
- 녹화 프로세스(`stream.sh`)가 보내는 곳은 SD 카드와 **파이 안의 localhost MediaMTX 뿐**이다. Wi-Fi로 노트북에
  보내는 일은 MediaMTX(RTSP)와 `tcp_relay.sh`(TCP)가 따로 한다. 그래서 Wi-Fi가 끊기거나, 노트북이 접속·종료·재접속을
  반복해도 녹화 프로세스에는 영향이 없다. 느리거나 끊긴 시청자는 MediaMTX가 혼자 정리한다
- 노트북은 Wi-Fi가 돌아오면 다시 붙으면 된다 (`live.run`은 자동 재연결). 그사이 영상은 노트북에는 없지만 SD에는 있다
- localhost publish마저 실패해도 (MediaMTX가 없거나 죽음) ffmpeg는 그 출력만 버리고 녹화를 계속한다 (`onfail=ignore`)
- 녹화가 새 폴더로 끊기는 경우는 녹화 프로세스 자체가 다시 시작될 때뿐이다 (카메라 오류, 재부팅, `systemctl restart`)
- 확인한 것 (맥, 가짜 카메라 = ffmpeg testsrc, MediaMTX 대신 ffmpeg RTSP 수신기): 수신기가 없을 때, 그리고 수신
  중 수신기를 강제 종료했을 때 모두 녹화 조각이 끊김 없이 이어졌다 (14.5초 = 435프레임, 조각을 이으면 연속).
  실제 파이·MediaMTX·Wi-Fi 끊김으로는 확인하지 않았다 — 7장 5번 항목으로 지상 점검

## 1. 준비물

- Raspberry Pi Zero 2 W, Camera Module 3 (일반 화각 또는 Wide)
- **Zero용 카메라 케이블** (22핀 ↔ 15핀). Camera Module 3에 들어 있는 기본 케이블(15핀 ↔ 15핀)은 Zero에 맞지 않는다
- microSD 32 GB 이상, A2/V30급 (연속 기록). 6 Mbps면 시간당 약 2.7 GB
- 5V 3A 이상 BEC (6장)

## 2. OS 설치

1. Raspberry Pi Imager → **Raspberry Pi OS Lite (64-bit)**
2. OS 설정 편집: 호스트 이름 `facade-pi`, 사용자 `pi` (다르게 하면 `facade-stream.service`의 `User`, `stream.env`의
   `REC_DIR`도 바꾼다), Wi-Fi = 노트북 핫스팟 SSID/비밀번호 (5장), 국가 KR, SSH 사용
3. 부팅 후 `ssh pi@facade-pi.local`

## 3. 카메라 확인

최근 Raspberry Pi OS는 카메라를 자동 인식한다 (`/boot/firmware/config.txt`의 `camera_auto_detect=1`).

```bash
rpicam-hello --list-cameras          # imx708 (Camera Module 3)이 보여야 한다
rpicam-vid -t 5000 -n -o test.h264   # 5초 녹화. 안 되면 케이블 방향(접점 면)부터 확인
vcgencmd get_throttled               # 0x0 이어야 한다 (아니면 전원 부족 — 6장)
```

## 4. 설치

```bash
sudo apt update && sudo apt install -y ffmpeg

# MediaMTX: https://github.com/bluenviron/mediamtx/releases 에서 최신 linux arm64 tar.gz 이름 확인
VER=v1.x.y
curl -L "https://github.com/bluenviron/mediamtx/releases/download/$VER/mediamtx_${VER}_linux_arm64.tar.gz" \
  | sudo tar xz -C /usr/local/bin mediamtx

# 이 폴더 파일 (노트북에서 scp edge/raspberrypi/* pi@facade-pi.local:~ 로 먼저 복사)
sudo mkdir -p /opt/facade-inspect
sudo cp stream.sh tcp_relay.sh stream.env mediamtx.yml /opt/facade-inspect/
sudo chmod +x /opt/facade-inspect/stream.sh /opt/facade-inspect/tcp_relay.sh
sudo cp facade-stream.service mediamtx.service facade-tcp.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now mediamtx facade-stream    # 부팅 시 자동 시작
sudo systemctl enable --now facade-tcp                # MODE=tcp일 때만 중계 (rtsp면 바로 끝남)

journalctl -u facade-stream -f    # 로그
ls ~/recordings/                  # 녹화 조각이 SEGMENT_SEC마다 늘어나는지
```

설정 변경은 `/opt/facade-inspect/stream.env` 수정 후 `sudo systemctl restart facade-stream facade-tcp`
(녹화 프로세스가 재시작되므로 새 녹화 폴더가 생긴다).

## 5. 노트북 핫스팟 + 수신

Zero 2 W의 Wi-Fi는 **2.4 GHz 전용**이다. 핫스팟을 2.4 GHz, 고정 채널(1/6/11 중 하나)로 만든다.

- **macOS 인터넷 공유**: 시스템 설정 → 일반 → 공유 → 인터넷 공유. 공유할 연결 = 유선 LAN 또는 iPhone USB,
  대상 = Wi-Fi (Wi-Fi 옵션에서 SSID·채널·WPA2 설정). 맥은 **Wi-Fi 말고 다른 연결이 있어야** 공유를 켤 수 있다.
  현장에 다른 연결이 없으면: iPhone 개인용 핫스팟에 파이와 노트북을 둘 다 붙이거나, 휴대용 공유기를 쓴다
- 파이 IP: `ping facade-pi.local` (안 되면 맥에서 `arp -a`)

```bash
# 수신 확인 (fps·지연·끊김 통계를 1초마다)
uv run --project cv python -m facade_cv.io.source rtsp://facade-pi.local:8554/cam --seconds 10
# 실시간 창문 검출 (+ 노트북에도 녹화)
uv run --project cv python -m facade_cv.live.run --source rtsp://<파이IP>:8554/cam --record
# MODE=tcp 일 때 (파이 안 MediaMTX를 tcp로 중계)
uv run --project cv python -m facade_cv.live.run --source tcp://<파이IP>:8888
# 영상만 빠르게 보기
ffplay -fflags nobuffer -flags low_delay rtsp://<파이IP>:8554/cam
```

끊기면 `live.run`이 자동 재연결한다 (화면 상단 `disconnects`/`reconnects`, `config/live.yaml` source).

## 6. 전원 (5V BEC) 주의

- 드론 배터리 → **전용 5V BEC (3A 이상)** → 파이. 수신기·FC와 같은 5V 레일을 쓰지 않는다
  (파이 부팅·Wi-Fi 송신 순간 전류가 수신기를 브라운아웃시키면 조종 불능·페일세이프)
- GPIO 2번 핀(5V) + 6번 핀(GND)에 연결하면 보드의 폴리퓨즈·역전압 보호를 거치지 않는다:
  **극성을 두 번 확인**하고, 연결 전 멀티미터로 BEC 출력이 5.0~5.25 V인지 본다
- BEC와 USB 전원을 **동시에 연결하지 않는다** (역류)
- 송출+녹화 부하 상태에서 4.9 V 이상, `vcgencmd get_throttled`가 `0x0`인지 확인 (`0x50005` 등은 저전압)
- 전원을 그냥 끊으면 SD가 손상될 수 있다: 착륙 후 `sudo poweroff` → 배터리 분리. 녹화는 TS 조각이라 끊겨도
  마지막 조각만 잃는다. 반복 운용하면 raspi-config의 Overlay File System(읽기 전용 루트)을 검토
- 카메라 리본 케이블은 진동에 빠지지 않게 고정. 지상 대기 중 발열: `vcgencmd measure_temp`

## 7. 조종기 2.4 GHz와 Wi-Fi 공존 — 지상 점검 (프롭 제거 상태에서 시작)

조종기(ELRS 2.4 GHz, FrSky 등)와 Wi-Fi가 같은 대역이다. 비행 전에 다음을 모두 통과해야 한다.

1. **기준값**: 송출을 끈 상태(`sudo systemctl stop facade-stream`)에서 조종기 링크 품질(LQ/RSSI 텔레메트리)을 기록
2. **송출 켠 상태 비교**: 같은 위치에서 LQ/RSSI가 눈에 띄게 떨어지면 → `BITRATE` 낮추기, 핫스팟 채널 변경,
   파이와 수신기 안테나 간격 늘리기(가능한 멀리, 카본 프레임에 가리지 않게)
3. **레인지 체크**: 조종기 레인지 체크(저출력) 모드로 송출을 켠 채 비행 예정 거리 이상 걸어가며 조종 유지 확인
4. **페일세이프**: 송출 중 조종기를 꺼서 드론 페일세이프가 정상 동작하는지
5. **영상 끊김 복구**: 노트북 Wi-Fi를 껐다 켜기 → `live.run`의 `reconnects`가 늘고 영상이 돌아오는지,
   그동안 파이 SD 녹화가 **같은 폴더에서** 조각이 계속 늘어나는지 (`ls ~/recordings/<폴더>`, 새 폴더가 생기면 안 됨).
   `live.run`을 껐다 켜기, 노트북 없이 부팅해 몇 분 뒤 접속하기도 같은 방식으로 확인 (두 MODE 모두)
6. **거리**: 비행 거리에서 `python -m facade_cv.io.source rtsp://...`로 수신 fps·지연이 쓸 만한지
7. **모터 가동**(프롭 제거): 저전압(`get_throttled`)·영상 노이즈·링크 품질 변화
8. **지속 시간**: 배터리 한 팩 동안 연속 송출+녹화, 온도

## 8. 비행 후 녹화 회수

```bash
scp -r pi@facade-pi.local:recordings/<폴더> data/raw/videos/<building>/
cat data/raw/videos/<building>/<폴더>/seg_*.ts > flight.ts    # 타임스탬프가 조각 사이에 이어져 있어 그냥 이으면 된다
ffmpeg -i flight.ts -c copy data/raw/videos/<building>/<날짜>_<building>_drone_01.mp4
```
`data/raw/videos/captures.csv`에 한 줄 추가 후 기존 Post-flight 파이프라인(`io.extract` …)에 넣는다.

## 알려진 한계

- **송출과 녹화는 같은 스트림이다.** rpicam-vid는 인코딩 출력이 하나라서 기본값은 1920×1080 30fps 6 Mbps 하나를
  송출과 SD 녹화에 같이 쓴다. 링크가 약해 `WIDTH/HEIGHT=1280x720`으로 낮추면 녹화도 720p가 된다.
  "720p 송출 + 1080p 녹화" 동시는 이중 인코딩이 필요한데 (picamera2 main+lores 스트림에 인코더 두 개),
  Zero 2 W 하드웨어 H.264 인코더 처리량(대략 1080p30 한 개 분량)을 넘을 수 있어 구현하지 않았다
- MediaMTX 자체가 도중에 죽으면 (Wi-Fi 끊김과는 별개) 녹화는 계속되지만, ffmpeg가 publish 출력을 버린 뒤라
  송출은 `facade-stream` 재시작 전까지 돌아오지 않는다. 재시작하면 새 녹화 폴더가 생긴다
- `MODE=tcp`는 시청자를 한 번에 하나만 받는다 (`tcp_relay.sh`가 접속마다 다시 대기). 여럿이 보려면 `MODE=rtsp`
- 파이에 RTC가 없어 오프라인 부팅이면 녹화 폴더 시각이 틀릴 수 있다
