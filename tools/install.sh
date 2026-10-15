#!/usr/bin/env bash
# Install a release on a Cardputer-Adv without building anything:
# firmware + flash tables over esptool, model + tokenizer to the SD card over USB.
#   tools/install.sh               # latest release
#   tools/install.sh v1.0 [PORT]   # a given tag / serial port
# Needs: gh (logged in), uv. A formatted microSD card (FAT32) in the Cardputer.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="${1:-}"
PORT="${2:-$(ls /dev/cu.usbmodem* | head -1)}"
DIR="$ROOT/out/release${TAG:+-$TAG}"

mkdir -p "$DIR"
gh release download ${TAG:+"$TAG"} --repo lemcoder/lfm-cardputer --dir "$DIR" --clobber
(cd "$DIR" && shasum -a 256 -c SHA256SUMS)

pkill -f esp_idf_monitor 2>/dev/null && sleep 0.5 || true
uvx --from esptool==4.8.1 esptool.py --chip esp32s3 --port "$PORT" -b 460800 \
    --before default_reset --after hard_reset write_flash \
    --flash_mode dio --flash_size 8MB --flash_freq 80m \
    0x0 "$DIR/bootloader.bin" \
    0x8000 "$DIR/partition-table.bin" \
    0x10000 "$DIR/lfm_cardputer.bin" \
    0x110000 "$DIR/dhead.bin" \
    0x250000 "$DIR/lmhead.bin"

# ~6 min at ~0.4 MB/s; the firmware waits for the upload while the model is missing.
uv run "$ROOT/tools/upload.py" --port "$PORT" \
    "$DIR/lfm.lfm:lfm.lfm" "$DIR/tok.bin:tok.bin" "$DIR/cfg.txt:cfg.txt"
