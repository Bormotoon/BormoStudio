#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  printf 'Usage: night_run.sh CHANNEL_URL DESTINATION [bstudio channel options...]\n' >&2
  exit 2
fi
url=$1
destination=$2
shift 2
exec "${BSTUDIO_BIN:-${UVF_BIN:-bstudio}}" channel "$url" "$destination" --events --report "$destination/night-report.json" "$@"
