#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install it before starting BirdwatchPi." >&2
    exit 1
fi

OUTPUT_DIR="${BIRDWATCH_OUTPUT_DIR:-$APP_DIR/birds}"
WEB_HOST="${BIRDWATCH_WEB_HOST:-127.0.0.1}"
WEB_PORT="${BIRDWATCH_WEB_PORT:-8000}"
DATABASE_PATH="$OUTPUT_DIR/detections.sqlite3"

mkdir -p "$OUTPUT_DIR"
uv run python -c \
    'import sys; from pathlib import Path; from bird_data import initialize_database; initialize_database(Path(sys.argv[1]), Path(sys.argv[2]) / "detections.csv", Path(sys.argv[2]))' \
    "$DATABASE_PATH" "$OUTPUT_DIR"

uv run python birdwatch.py --camera --output-dir "$OUTPUT_DIR" &
detector_pid=$!
uv run python birdwatch_web.py \
    --output-dir "$OUTPUT_DIR" \
    --host "$WEB_HOST" \
    --port "$WEB_PORT" &
web_pid=$!

shutdown() {
    trap - EXIT INT TERM
    kill "$detector_pid" "$web_pid" 2>/dev/null || true
    wait "$detector_pid" 2>/dev/null || true
    wait "$web_pid" 2>/dev/null || true
}

trap shutdown EXIT
trap 'exit 0' INT TERM

echo "BirdwatchPi detector and webpage are running."
echo "Open http://$WEB_HOST:$WEB_PORT"

set +e
wait -n "$detector_pid" "$web_pid"
status=$?
set -e
exit "$status"
