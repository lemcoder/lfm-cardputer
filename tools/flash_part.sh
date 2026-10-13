#!/usr/bin/env bash
# Write a file into a flash data partition (see firmware/partitions.csv).
#   tools/flash_part.sh draft  out/dhead.bin    # speculative-decoding draft head
#   tools/flash_part.sh lmhead out/lmhead.bin   # LM head shortlist table
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IDF_PATH="${IDF_PATH:-$HOME/esp/esp-idf-v5.4.2}"
NAME="$1"
FILE="$2"
PORT="${3:-$(ls /dev/cu.usbmodem* | head -1)}"
pkill -f esp_idf_monitor 2>/dev/null && sleep 0.5 || true
. "$IDF_PATH/export.sh" > /dev/null
python "$IDF_PATH/components/partition_table/parttool.py" --port "$PORT" \
    --partition-table-file "$ROOT/firmware/partitions.csv" \
    write_partition --partition-name "$NAME" --input "$FILE"
