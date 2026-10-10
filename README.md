# lfm-cardputer

LFM2.5-230M inference on the M5Stack Cardputer-Adv (ESP32-S3), weights streamed from microSD.
Hand-written, for learning inference engineering.

## Layout

```
lfm/        portable C++20 inference core (ESP-IDF component AND host static lib)
host/       host build: reference engine, tests vs golden dumps
firmware/   ESP-IDF 5.4.x standalone app (.bin) for the Cardputer-Adv; prints tokens on screen
tools/      Python: golden activation dumps, weight converter -> .lfm
docs/       notes, measurements
```

## Setup

### Host

```sh
cmake -S host -B host/build -G Ninja && cmake --build host/build
./host/build/lfm_host
```

### Firmware (ESP-IDF v5.4.2, same as the Cardputer-Adv demo)

```sh
mkdir -p ~/esp && cd ~/esp
git clone -b v5.4.2 --recursive https://github.com/espressif/esp-idf.git esp-idf-v5.4.2
./esp-idf-v5.4.2/install.sh esp32s3
. ~/esp/esp-idf-v5.4.2/export.sh

git submodule update --init   # M5GFX 0.2.15 + M5Unified 0.2.10 (same as ADV demo)
cd firmware
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/cu.usbmodem* flash monitor
```

### Zed

Tasks in `.zed/tasks.json` (cmd-shift-p -> `task: spawn`): `fw: build`, `fw: flash + monitor`,
`fw: monitor`, `fw: menuconfig`, `host: build + run`. They go through `tools/idf.sh`, which sources
ESP-IDF from `~/esp/esp-idf-v5.4.2` (override with `IDF_PATH`), so no shell alias is needed.

Uses clangd. Each of `firmware/`, `host/`, `lfm/` has a `.clangd` pointing at the right
`compile_commands.json` (build the host / firmware once to generate them).
For firmware, Espressif's `esp-clang` (`idf_tools.py install esp-clang`) gives better
Xtensa header resolution than macOS clangd.

## Hardware facts (Cardputer-Adv, verified against M5Cardputer-UserDemo `CardputerADV` branch)

- ESP32-S3FN8, 240 MHz dual core, 8 MB flash, **no PSRAM** -> ~512 KB SRAM, ~300 KB usable.
- microSD over **SPI**: MISO 39, MOSI 14, SCLK 40, CS 12. Bus shared with LoRa cap (NSS 5).
  SDSPI default 20 MHz -> ~2.5 MB/s ceiling. Try 40 MHz, and SDMMC 1-bit via GPIO matrix.

## Model facts (LFM2.5-230M config.json)

- dim 1024, ff 2560 (SwiGLU), 14 layers: conv x8, attn x6 (layers 2,4,6,8,10,12)
- 16 heads, 8 KV heads, head_dim 64, RoPE theta 1e6, vocab 65536, tied embeddings
- KV cache: 6 x 8 x 64 x 2 x 2 B = **12 KB/token** fp16 (6 KB int8) -> context is RAM-bound
- ~163M MACs in layers + 67M in full LM head per token
- **Bytes read per token is the metric.** Q4 ~130 MB, Q2 ~70 MB; tied head = 25% of it

## Milestones

0. [ ] tools: dump golden per-layer activations + logits from HF transformers
1. [ ] host: fp32 single-token forward in C++, matches golden layer-by-layer
2. [ ] host: custom Q4/Q2 block quant; per-layer error, top-1 agreement, perplexity
3. [ ] host: streaming engine, fixed ~280 KB arena, bytes/token counter
4. [ ] device: SD sequential read benchmark (SDSPI 20/40 MHz, SDMMC 1-bit)
5. [ ] device: quantized matvec benchmark, scalar then PIE SIMD (cycles/MAC)
6. [ ] device: full model; then BPE tokenizer; then UI
