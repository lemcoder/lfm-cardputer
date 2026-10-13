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
./host/build/lfm_tests                                  # GoogleTest
./host/build/lfm_tests --gtest_filter='MathRef.Rope*'    # one group
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

**Performance and findings:** [docs/performance.md](docs/performance.md): what limits speed on this
hardware, every optimisation with device measurements (413 s -> 55 s for the golden prompt), quality of
the quantized variants, and what's next.

## Hardware facts (Cardputer-Adv, verified against M5Cardputer-UserDemo `CardputerADV` branch)

- ESP32-S3FN8, 240 MHz dual core, 8 MB flash, **no PSRAM** -> ~512 KB SRAM, ~300 KB usable.
- microSD wired for SPI: MISO 39, MOSI 14, SCLK 40, CS 12. Bus shared with LoRa cap (NSS 5).
- **Measured (firmware/main/sd_bench, 32 KB sequential reads):**
  - SPI 20 MHz: 1.82 MB/s. SPI 40 MHz: fails (GPIO-matrix input timing).
  - **SD 1-bit mode on the same wires** (CLK 40, CMD 14, DAT0 39, DAT3 12): 2.45 MB/s @ 20 MHz,
    **4.85 MB/s @ 40 MHz** (97% of the 1-bit ceiling). Input delay phase 0-3 makes no difference.
  - FATFS `read()` costs ~0% vs raw sectors. `fread()` with `_IONBF` is ~30x slower: don't use it.
  - 4 KB reads ~17% slower than 32 KB; read weights in >= 32 KB chunks.
  - Once a card has been in SPI mode it needs a power cycle to enter SD mode (and vice versa here).
- Free heap after display init: 366 KB, largest block 272 KB.

## Model facts (LFM2.5-230M config.json)

- dim 1024, ff 2560 (SwiGLU), 14 layers: conv x8, attn x6 (layers 2,4,6,8,10,12)
- 16 heads, 8 KV heads, head_dim 64, RoPE theta 1e6, vocab 65536, tied embeddings
- KV cache: 6 x 8 x 64 x 2 x 2 B = **12 KB/token** fp16 (6 KB int8) -> context is RAM-bound
- ~163M MACs in layers + 67M in full LM head per token
- **Bytes read per token is the metric.** Q4 ~130 MB, Q2 ~70 MB; tied head = 25% of it

## Golden reference (tools/golden.py)

`uv run tools/golden.py [--prompt "..." --name x]` runs HF LFM2.5-230M in fp32 and writes
`golden/<name>/`: `manifest.json` (tokens, config, weights path, tensor index) + one raw
little-endian `float32` file per activation (`*.f32`, row-major, leading axis = position).
See the script's docstring for every tensor name (`L2.attn.q_rope`, `L0.conv.Bx`, ...).

Weights: `model.safetensors` (path in manifest) = 8-byte LE header length, JSON header
(`name -> dtype, shape, data_offsets`), then raw data. All 132 tensors are **BF16**
(fp32 = bf16 bits << 16). Linear weights are `[out, in]` row-major, so `y = W x` reads rows.

Semantics the C++ must match:
- RMSNorm: `x * rsqrt(mean(x^2) + 1e-5) * w`, in fp32.
- Conv layer: `in_proj` -> `[B | C | x]`; `Bx = B*x`; causal depthwise conv, kernel 3
  (`w[:,0]*Bx[t-2] + w[:,1]*Bx[t-1] + w[:,2]*Bx[t]`); `out_proj(C * conv)`.
  Decode state per conv layer = last 2 `Bx` rows (2 x 1024 floats).
- Attention: q/k/v proj -> per-head RMSNorm on q and k -> RoPE (rotate_half: pairs are
  `(i, i+32)`, theta 1e6) -> causal GQA (2 q heads per kv head), scale 1/8 -> `out_proj`.
- MLP: `w2(silu(w1 x) * w3 x)`.
- Layer: `h = x + op(operator_norm(x)); out = h + mlp(ffn_norm(h))`.
- Head: `logits = embed_tokens @ embedding_norm(h)` (tied).

## Running the model

```
uv run tools/export.py                 # out/lfm-q8.lfm (245 MB) + out/prompt.txt
uv run tools/export.py --dtype bf16    # out/lfm-bf16.lfm (460 MB), exact weights for host tests
./host/build/lfm_host out/lfm-q8.lfm   # same engine on the Mac
```

On the device: copy `out/lfm-q8.lfm` to the SD card root as `lfm.lfm` and
`out/prompt.txt` as `prompt.txt`, then Zed task **fw: flash + monitor**.
No card reader: flash first, then `uv run tools/upload.py` sends both files
over the USB cable (~0.4 MB/s, ~10 min for the model; `--prompt-only` for
just the prompt). The firmware listens for it for 1.5 s after every boot.
G0 runs the prompt again. Change the prompt with
`uv run tools/export.py --prompt "..."` (rewrites prompt.txt; the model file
is rewritten too, same content).

Engine (`lfm/engine.hpp`): one token at a time, every weight streamed from the
file through one 32 KB buffer. On the device the KV cache is a file too
(`/kv.bin`, 12 KB per token, read back with an online softmax), so RAM is a
fixed ~136 KB and the context is limited to 1024 tokens by config only.

## Milestones

0. [x] tools: dump golden per-layer activations + logits from HF transformers
1. [ ] host: fp32 single-token forward in C++, matches golden layer-by-layer
2. [ ] host: custom Q4/Q2 block quant; per-layer error, top-1 agreement, perplexity
3. [ ] host: streaming engine, fixed ~280 KB arena, bytes/token counter
4. [x] device: SD sequential read benchmark -> SD 1-bit @ 40 MHz, 4.85 MB/s
5. [ ] device: quantized matvec benchmark, scalar then PIE SIMD (cycles/MAC)
6. [ ] device: full model; then BPE tokenizer; then UI
