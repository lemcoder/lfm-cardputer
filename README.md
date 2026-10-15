# lfm-cardputer

LFM2.5-230M running on a stock M5Stack Cardputer-Adv (ESP32-S3, no PSRAM), every
weight streamed from the microSD card. A from-scratch C++ inference engine:
type a message on the Cardputer's keyboard and the model answers on its screen.

| | first working build | now |
|---|---|---|
| "The capital of France is" → " Paris." | 413 s | **21 s** |
| per generated token (planets benchmark) | 63.1 s | **4.39 s** |
| per generated token (chat reply) | — | **~6.1 s** (~2 tokens per weight pass) |

Same unmodified hardware throughout. The card's 1-bit bus at 80 MHz (~9.4 MB/s,
busy ~92%) is the limit: [docs/performance.md](docs/performance.md) explains
where the time goes, [docs/experiments.md](docs/experiments.md) lists every
idea tried (kept or not), [docs/extensions.md](docs/extensions.md) the hardware
changes that would go further, [docs/medium.md](docs/medium.md) the story.

## Install a release (no build needed)

Each [release](https://github.com/lemcoder/lfm-cardputer/releases) has
everything the device needs:

| file | goes to | |
|---|---|---|
| `bootloader.bin`, `partition-table.bin`, `lfm_cardputer.bin` | flash 0x0 / 0x8000 / 0x10000 | firmware |
| `dhead.bin` | flash 0x110000 (`draft`) | speculative-decoding draft head |
| `lmhead.bin` | flash 0x250000 (`lmhead`) | LM head shortlist table |
| `lfm.lfm` | SD card | the model, Q4 (148 MB) |
| `tok.bin` | SD card | tokenizer |
| `cfg.txt` | SD card | `chat=1` |
| `SHA256SUMS` | | checksums |

With a FAT32 microSD card in the Cardputer and the USB cable plugged in:

```sh
tools/install.sh            # latest release; or: tools/install.sh v1.0 /dev/cu.usbmodemXXXX
```

It downloads the release (`gh`), checks the checksums, flashes with esptool
(`uvx`), then copies the SD files over the USB cable (~6 min, no card reader
needed; or copy them to the card root yourself).

On the device: type, Enter sends, fn+` starts a new chat. Words already typed
are pushed through the model while you type, so the reply starts sooner.

The `.lfm` is derived from the LFM2.5-230M weights and covered by their
license.

## Layout

```
lfm/        portable C++20 inference core (ESP-IDF component AND host static lib)
host/       host build: reference engine, tests vs golden dumps
firmware/   ESP-IDF 5.4.x standalone app (.bin) for the Cardputer-Adv; prints tokens on screen
tools/      Python: golden dumps, weight export, tokenizer, draft-head training, bench, upload, install
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

## Hardware facts (Cardputer-Adv, verified against M5Cardputer-UserDemo `CardputerADV` branch)

- ESP32-S3FN8, 240 MHz dual core, 8 MB flash (GD25Q64, run at 120 MHz QIO), **no PSRAM**
  -> ~512 KB SRAM, ~300 KB usable.
- microSD wired for SPI only: MISO 39, MOSI 14, SCLK 40, CS 12; card DAT1/DAT2 unconnected,
  so no 4-bit SD mode. Bus shared with the LoRa cap (NSS 5).
- **SD 1-bit mode works on the same wires** (CLK 40, CMD 14, DAT0 39, DAT3 12 held high).
  Measured: SPI 20 MHz 1.82 MB/s (40 MHz fails); SD 1-bit 40 MHz 4.85 MB/s (the IDF maximum),
  **80 MHz overclock (160 MHz PLL ÷ 2) ~9.4 MB/s in the engine**, 9.9 MB/s streamed raw;
  160 MHz doesn't answer. Details: docs/performance.md, experiments #34–36.
  - Once a card has been in SPI mode it needs a power cycle to enter SD mode.
  - `fread()` with `_IONBF` is ~30x slower than `read()`: don't use it.
- Keyboard: TCA8418 on I2C (SDA 8, SCL 9, address 0x34).

## Model facts (LFM2.5-230M config.json)

- dim 1024, ff 2560 (SwiGLU), 14 layers: conv x8, attn x6 (layers 2,4,6,8,10,12)
- 16 heads, 8 KV heads, head_dim 64, RoPE theta 1e6, vocab 65536, tied embeddings
- KV cache: 6 x 8 x 64 x 2 x 2 B = **12 KB/token** fp16, kept on the SD card (context cap 1024)
- ~163M MACs in layers + 67M in full LM head per token
- **Bytes read per token is the metric.** Q4 layers ~91 MB per weight pass; the tied head
  (25% of the model) is replaced by a low-rank shortlist in flash + ~768 exact rows from SD
- Q4 is the floor: 2-bit breaks the model, 3-bit isn't worth it (experiment #38)

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

## Building it yourself

```sh
uv run tools/lm_basis.py                                       # out/lm_basis.npy (low-rank LM head basis)
uv run tools/export.py --quant q4 --lm-head lowrank --out out/lfm-q4-head-lr256.lfm
uv run tools/export_tokenizer.py                               # out/tok.bin
uv run tools/flash_head.py                                     # out/lmhead.bin
uv run tools/selfgen_corpus.py ...                             # model's own text (see draft_heads.py)
uv run tools/draft_heads.py collect && uv run tools/draft_heads.py train && uv run tools/draft_heads.py export
tools/idf.sh build flash                                       # firmware
tools/flash_part.sh lmhead out/lmhead.bin
tools/flash_part.sh draft out/dhead.bin
printf 'chat=1\n' > out/cfg.txt
uv run tools/upload.py out/lfm-q4-head-lr256.lfm:lfm.lfm out/tok.bin:tok.bin out/cfg.txt:cfg.txt
./host/build/lfm_tests                                         # host tests
```

`/cfg.txt` on the card holds `key=value` knobs (`RunConfig` in `firmware/main/main.cpp`).
`chat=0` runs `/prompt.txt` instead (`uv run tools/export.py --prompt "..."`), which is
what `tools/bench.py` uses; G0 runs it again. `kKernelTest` in `main.cpp` switches the
firmware to the SIMD kernel test.

## Milestones

0. [x] tools: golden per-layer activations + logits from HF transformers
1. [x] host: fp32 forward in C++, matches golden layer by layer
2. [x] host: Q8 / Q4 block quantization; perplexity, top-1 agreement, KL vs BF16
3. [x] host + device: streaming engine, fixed RAM, KV cache on SD
4. [x] device: SD 1-bit mode, overclocked 40 -> 80 MHz, async + aligned reads
5. [x] device: PIE SIMD Q4/Q8 kernels (6-7x vs C), multi-token kernel in assembly
6. [x] device: full model, " Paris." in 413 s -> 55 s
7. [x] device: LM head shortlist in flash, batched prefill (batch 7), prompt cache
8. [x] device: speculative decoding: draft head trained on the model's own text, tree drafts
9. [x] device: patched SD driver, one streamed command per matrix (-10%)
10. [x] device: BPE tokenizer, keyboard chat, prefill while typing
11. [x] release: prebuilt firmware + model, `tools/install.sh`
