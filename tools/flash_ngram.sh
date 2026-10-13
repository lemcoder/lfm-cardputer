#!/usr/bin/env bash
# Write the speculative-decoding n-gram table into the flash "weights" partition.
#   uv run tools/spec_eval.py --export out/ngram.bin --gen-prompts 0   # build it (once)
#   tools/flash_ngram.sh [PORT]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IDF_PATH="${IDF_PATH:-$HOME/esp/esp-idf-v5.4.2}"
PORT="${1:-$(ls /dev/cu.usbmodem* | head -1)}"
pkill -f esp_idf_monitor 2>/dev/null && sleep 0.5 || true
. "$IDF_PATH/export.sh" > /dev/null
python "$IDF_PATH/components/partition_table/parttool.py" --port "$PORT" \
    --partition-table-file "$ROOT/firmware/partitions.csv" \
    write_partition --partition-name weights --input "$ROOT/out/ngram.bin"
