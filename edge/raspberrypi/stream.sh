#!/usr/bin/env bash
# 드론 탑재 카메라: SD 녹화 + 파이 안 MediaMTX로 publish. 미검증 (하드웨어 없이 작성) — setup.md
#
# Camera Module 3 → rpicam-vid H.264 (하드웨어 인코딩 한 번) → ffmpeg -c copy (재인코딩 없음) tee:
#   1) SD 녹화  $REC_DIR/<시작시각>/seg_0000.ts ...  카메라 시작부터 계속. MPEG-TS 조각 (전원 차단에도 이전 조각은 온전)
#   2) publish  $RTSP_URL = 파이 안 localhost MediaMTX. 노트북은 MediaMTX에서 받는다
#               MODE=rtsp → rtsp://<파이IP>:8554/cam
#               MODE=tcp  → facade-tcp.service(tcp_relay.sh)가 MediaMTX를 tcp://<파이IP>:$TCP_PORT 로 중계
# 노트북 연결·Wi-Fi 상태는 이 프로세스에 닿지 않는다: 보내는 곳이 localhost뿐이고, 느리거나 끊긴 시청자는
# MediaMTX가 따로 처리한다. publish가 실패해도(onfail=ignore) 녹화는 계속된다.
# 설정: stream.env (systemd는 EnvironmentFile, 직접 실행하면 같은 폴더의 stream.env)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
if [[ -z "${MODE:-}" && -f "$here/stream.env" ]]; then
  set -a; source "$here/stream.env"; set +a
fi

# MediaMTX가 늦게 뜨면 publish가 시작 시 실패해 이번 실행 동안 송출이 없다 → 잠깐 기다린다 (녹화는 어차피 시작)
hostport=${RTSP_URL#rtsp://}; hostport=${hostport%%/*}
for ((i = 0; i < MTX_WAIT_SEC * 2; i++)); do
  (exec 3<>"/dev/tcp/${hostport%:*}/${hostport#*:}") 2>/dev/null && break
  sleep 0.5
done

# 파이에 RTC가 없어 오프라인 부팅이면 시각이 틀릴 수 있다 — 폴더 구분용
out_dir="$REC_DIR/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$out_dir"

af=(--autofocus-mode "$AF_MODE")
[[ "$AF_MODE" == manual ]] && af+=(--lens-position "$LENS_POSITION")

echo "MODE=$MODE ${WIDTH}x${HEIGHT}@${FPS} ${BITRATE}bps → $out_dir + $RTSP_URL" >&2
# --inline: 매 I-프레임에 SPS/PPS → 중간 접속·재접속한 시청자도 디코딩 가능
rpicam-vid -t 0 -n --width "$WIDTH" --height "$HEIGHT" --framerate "$FPS" \
    --codec h264 --profile high --level 4.1 --bitrate "$BITRATE" --intra "$INTRA" --inline \
    "${af[@]}" -o - \
  | ffmpeg -hide_banner -loglevel warning -fflags +genpts -f h264 -framerate "$FPS" -i - \
    -c copy -map 0 -f tee \
    "[f=segment:segment_time=$SEGMENT_SEC:segment_format=mpegts]$out_dir/seg_%04d.ts|[f=rtsp:rtsp_transport=tcp:onfail=ignore]$RTSP_URL"
