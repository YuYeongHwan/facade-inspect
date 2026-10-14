#!/usr/bin/env bash
# MODE=tcp: 파이 안 MediaMTX 스트림을 tcp://0.0.0.0:$TCP_PORT 로 중계 (노트북 --source tcp://<파이IP>:$TCP_PORT).
# 미검증 — setup.md. 녹화(stream.sh)와 별개 프로세스: 노트북이 끊기면 이 중계만 끝나고 systemd가 다시 대기시킨다.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
if [[ -z "${MODE:-}" && -f "$here/stream.env" ]]; then
  set -a; source "$here/stream.env"; set +a
fi
if [[ "$MODE" != tcp ]]; then
  echo "MODE=$MODE: tcp 중계 안 함" >&2
  exit 0
fi
exec ffmpeg -hide_banner -loglevel warning -rtsp_transport tcp -i "$RTSP_URL" \
  -c copy -f mpegts "tcp://0.0.0.0:$TCP_PORT?listen=1"
