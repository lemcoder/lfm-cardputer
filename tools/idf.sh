#!/usr/bin/env bash
# Run idf.py inside firmware/ with the ESP-IDF environment loaded.
# Usage: tools/idf.sh build | flash monitor | menuconfig | ...
# Override the ESP-IDF location with IDF_PATH.
set -euo pipefail

IDF_PATH="${IDF_PATH:-$HOME/esp/esp-idf-v5.4.2}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ ! -f "$IDF_PATH/export.sh" ]]; then
    echo "ESP-IDF not found at $IDF_PATH (set IDF_PATH)" >&2
    exit 1
fi

# A monitor left open in another terminal holds the serial port and makes
# flash/monitor fail with "port is busy". Close it first.
if [[ " $* " == *" flash "* || " $* " == *" monitor "* ]]; then
    pkill -f esp_idf_monitor 2>/dev/null && sleep 0.5 || true
fi

. "$IDF_PATH/export.sh" > /dev/null
cd "$ROOT/firmware"
exec idf.py "$@"
