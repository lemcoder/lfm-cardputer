# Running LFM2.5-230M on the Cardputer-Adv: findings

Status as of 2026-10-13 (commit `8c72da9`). LFM2.5-230M runs on the
Cardputer-Adv (ESP32-S3, no PSRAM) with every weight streamed from the
microSD card. For the golden prompt "The capital of France is" it answers
" Paris." in **55 s**, down from **413 s** for the first working build.

Everything below was measured on the device unless marked *host*. Each row
of the timeline is one real run of the same prompt: 6 prompt tokens, then
3 generated tokens (" Paris", ".", `<|im_end|>`).

## Where the time goes

A token has to touch every weight once. The model is ~230M parameters and
RAM is ~300 KB, so the weights are read from the SD card **once per token**
(once per *batch* of tokens during the prompt). Throughput is therefore:

    seconds per token ≈ max(bytes per token / SD bandwidth, compute per token)

Most of the work was making both terms small and keeping them overlapped.

## Timeline

| # | Change | Prompt (6 tok) | Per token | Total | SD MB/s | Notes |
|---|---|---|---|---|---|---|
| 1 | First working build: Q8, SD 1-bit @ 40 MHz, synchronous reads | 286.5 s | 63.1 s | 412.7 s | 4.58 | SD wait 85%, compute 15% |
| 2 | Overlap reads with compute (reader task on core 1, double buffer) | 243.8 s | 53.7 s | 351.2 s | 4.58 | SD busy 99% |
| 3 | LM head shortlist: scan a Q4 copy, re-score top 32 with Q8 rows | 236.3 s | 46.1 s | 328.6 s | 4.59 | −34 MB/token, same tokens |
| 4 | Batched prefill (prompt batch 3) | 84.7 s | 46.1 s | 176.9 s | 4.59 | weights read once per batch |
| 5 | SD clock 53.3 MHz (overclock, 160 / 3) | 65.4 s | 35.5 s | 136.4 s | 5.98 | |
| 6 | Sector-aligned 32 KB reads | 63.8 s | 34.7 s | 133.3 s | 6.13 | 92% of the 1-bit limit |
| 7 | SD clock 80 MHz (overclock, 160 / 2) | 47.2 s | 24.0 s | 95.2 s | 8.90 | compute starts to matter |
| 8 | **Q4 weights everywhere** + int8 activations (C) | 48.2 s | 15.0 s | 78.2 s | 8.89 | now compute-bound: SD busy 62% |
| 9 | ESP32-S3 PIE SIMD dot kernels | 36.1 s | 14.8 s | 65.7 s | 8.89 | prompt batch had dropped to 2 |
| 10 | Smaller allocations → prompt batch 4 fits | **25.7 s** | **14.8 s** | **55.3 s** | 8.89 | SD-bound again (98%) |
| 11 | Autonomous run ([experiments.md](experiments.md)): low-rank / flash LM head, self-generated n-gram drafts (k=2), batch 7, cross-stream prefetch, KV layout, … | **11.3 s** | **~11.7 s** (B1), **7.48 s** (B2, drafts hit) | **34.7 s** | 8.9 | prompt cache: 23.4 s on a repeat |
| 12 | Draft head instead of the n-gram table ([experiments.md](experiments.md) #30) | 11.3 s | **~6.1 s** (B1, both drafts hit), **5.61 s** (B2), ~7.4 s (chat) | **23.6 s** | 8.9 | ~1.7 tokens per pass on chat replies |

Rows 1–7 are lossless (same tokens as the PyTorch reference). Row 8 changes
the weights to 4 bits (see *Quality*); the golden prompt still gives " Paris.".

## Storage: the SD card

### The bus is 1 bit wide (hardware)

The microSD slot is wired for SPI only. In the official schematic
(`Sch_M5CardputerAdv_v1.0`, sheet 1) connector **J3 (TF-015)** has only
`CS 2, MOSI 3, VCC 4, CLK 5, GND 6, MISO 7`. Card pins 1 (DAT2) and 8 (DAT1)
aren't in the symbol, and there are exactly 4 ESD diodes (D9–D12) and 4
series resistors (R19–R22) for G12 / G14 / G40 / G39. M5Stack's pin map and
M5Unified's `_pin_table_spi_sd` agree.

So 4-bit SD mode (the 20–25 MB/s figure usually quoted) is impossible.
SD **1-bit** mode works on the same wires (CLK 40, CMD 14, DAT0 39,
DAT3 12 held high) and caps at `clock / 8` bytes per second.

### Measured bus options

| Mode | Clock | Read MB/s | Notes |
|---|---|---|---|
| SPI | 20 MHz | 1.82 | |
| SPI | 40 MHz | — | fails (GPIO-matrix input timing) |
| SD 1-bit | 20 MHz | 2.45 | |
| SD 1-bit | 40 MHz | 4.85 bench / 4.58 in engine | IDF driver's maximum |
| SD 1-bit | 53.3 MHz | 5.03 bench / 6.13 in engine | all 4 sampling phases CRC-clean |
| SD 1-bit | **80 MHz** | 6.76 bench / **8.90 in engine** | phases 1–3 clean, phase 0 fails; using phase 2 |

The bench numbers (`firmware/main/sd_speed.cpp`) include CRC computation in
the read loop, so they're lower than in-engine throughput. Ratios are what
matter.

- **How the overclock works:** the S3's SDMMC clock is 160 MHz / integer
  divider. The IDF driver forces any request ≥ 40 MHz to divider 4.
  `SdCard` wraps `host.set_card_clk` and rewrites the clock generator's
  divider (3 or 2) right after the driver's setup (`firmware/main/sd_card.cpp`).
- **Safety:** every SD data block is CRC-checked by the protocol, so a bad
  clock shows up as read errors, not silent corruption. The speed test also
  compares a CRC of 16 MB of the model file against a 40 MHz read.
- **Duty cycle:** divider 3 has a 2:1 duty cycle. The alternative duty
  setting failed at phase 2, so the default is used.
- **Fragile states:**
  - A card that has seen SPI mode stays in SPI mode until it's power-cycled.
  - A chip reset while the card is mid-transfer can leave it unresponsive
    (`send_op_cond` timeout 0x107); only a power cycle fixes it.

### Read pattern

- **Async reads:** a reader task pinned to core 1 fills one 32 KB buffer
  while core 0 computes on the other (`firmware/main/async_file.cpp`,
  `Source::start_read/wait`).
- **Aligned reads:** reads are fixed 32 KB blocks starting on sector
  boundaries. Tensors are 4 KB-aligned in the file. A row split across two
  blocks is stitched in a small carry buffer. Before this, whole-row chunks
  started mid-sector and cost FATFS an extra single-sector command each:
  worth +2.5%.
- **FATFS fast seek** is enabled. Without it, seeking back in a 130–280 MB
  file walks the cluster chain.
- **The KV cache lives on the card** (`/kv.bin`, 12 KB per token at bf16),
  appended position-major and read back in chunks with an online softmax.
  RAM no longer depends on context length; the context cap is 1024 tokens.

## Compute: the ESP32-S3

### Integer path

PIE, the S3's SIMD unit, is integer-only: 128-bit registers, 16 int8
multiply-accumulates per instruction into a 40-bit accumulator. Floats get
no SIMD. So each matvec input is quantized to int8 in blocks of 32 with a
scale (as in llama.cpp's Q8 × Q8), and weight rows are dotted with integer
math.

### PIE kernels (`lfm/src/math/esp32s3_dot.S`)

- **Per block:** integer sum of 32 weight × activation pairs; C applies the
  scales.
- **Unaligned loads:** Q4 blocks are 18 bytes and Q8 blocks 34, so weight
  loads are unaligned: `EE.LD.128.USAR` + `EE.SRC.Q`.
- **Nibble unpacking:** `EE.ANDQ` (mask 0x0F), `EE.VSR.32` (shift by 4),
  `EE.VSUBS.S8` (−8).

| | cycles per 32-weight block | vs C reference |
|---|---|---|
| Q4 kernel alone | 21 | |
| Q4 kernel + scale step | ~53 | 6.4–6.6× |
| Q8 kernel alone | 16.4 | |
| Q8 kernel + scale step | ~47 | 7.2–7.6× |

The kernels were verified on the device against the C reference: 1280
random rows at every byte misalignment, results bit-identical
(`firmware/main/kernel_test.cpp`, `kKernelTest`).

**Pitfall: `memcpy`.** ESP-IDF compiles with `-fno-builtin-memcpy`, so
every `memcpy` is a real function call. Two of them per block in the scale
step cost ~100 cycles per block, more than the SIMD work itself. Read bytes
directly and use `std::bit_cast`.

### Prompt batching

`Engine::prefill` runs up to `max_batch` tokens through each layer together:
every streamed weight row is applied to all of them before the next row, so
the prompt costs one read of the weights per batch. It needs ~27 KB of RAM
per batch token. Batch 4 fits after these changes:
- per-token scratch slots allocated separately instead of one block;
- the conv state split into 8 KB blocks per layer instead of one 64 KB
  block. The heap had 88 KB free but its largest block was 31 KB.

### Memory budget (device, batch 4)

| | KB |
|---|---|
| Two 32 KB weight buffers (+ slack) | 64 |
| Conv state (8 layers × 2 × 1024 floats) | 64 |
| Activations x / xn / tmp (4 tokens) | 48 |
| Scratch slots (4 × 12 KB) | 48 |
| int8 activations + scales | 12 |
| Norm weights, shared scratch, carry row | ~20 |
| **Engine total** | **~256** |
| Heap free after open | 33 (largest block 15) |

## Quality (host)

`host/eval.cpp` (`lfm_eval`) runs each file teacher-forced over 419 tokens
of mixed text (`tools/eval_text.py`: history, science, a recipe, code, a
story). It compares against BF16 weights with float activations.

| Weights | Perplexity | Top-1 = BF16 | KL(BF16 ‖ m) | MB |
|---|---|---|---|---|
| BF16 | 7.570 | 100% | 0 | 460 |
| Q8 | 7.580 | 98.3% | 0.0007 | 245 |
| Q8 + Q4 embedding | 7.593 | 93.8% | 0.014 | 211 |
| Q8 + Q4 MLP | 8.062 | 89.2% | 0.071 | 190 |
| Q8 + Q4 MLP + Q4 embedding | 8.007 | 85.4% | 0.081 | 156 |
| **Q4 everywhere (device)** | 8.663 | 82.1% | 0.146 | 130 |

Notes:
- **Q8 is effectively lossless.**
- **Q4 costs about +14% perplexity** and changes the top token in ~1 of 6
  positions; the MLP is the most sensitive part.
- **Q4 export:** per-block scale search (smallest squared error) instead of
  llama.cpp's max-magnitude scale. Same format; it improved KL slightly.
- **The device setting wasn't measured here.** The device uses Q4
  everywhere *plus* int8 activations, and the int8 activation effect was not
  measured, by choice. It still produces the golden answer on the host and
  on the device.

## Speculative decoding (host simulation)

The idea: draft k tokens cheaply, verify all k+1 positions in one pass over
the weights (the same reads as one token), and keep drafts that match the
model's own greedy choice. Output is identical to plain greedy decoding.

`tools/spec_eval.py` builds a trigram → next-token table (bigram fallback)
from WikiText-103 and simulates this on the model's own greedy output.
Results in tokens per pass (24 prompts, 1292 generated tokens):

| Draft | k=1 | k=2 | k=3 |
|---|---|---|---|
| Prompt lookup (repeat of earlier context) | 1.21 | 1.31 | 1.34 |
| n-gram table, 1–5.5 MB | 1.17–1.19 | 1.20–1.23 | 1.20–1.24 |
| **Prompt lookup, else 1 MB n-gram** | 1.36 | **1.51** | 1.56 |

- A 1 MB table is as good as 5.5 MB, and it fits the 5.9 MB `weights`
  flash partition.
- **Caveat:** small sample, and greedy text from a 230M model repeats
  itself, which flatters prompt lookup. Expect ~1.3× on varied text.

## USB upload (no card reader)

`tools/upload.py` sends files over the USB cable (USB-Serial-JTAG). The
firmware listens for 1.5 s after boot, writes `name.tmp`, renames it, then
reads the file back and checks size and CRC-32.

- **Throughput is 0.42 MB/s:** 130 MB in ~5.5 min, 283 MB in ~12.5 min.
  - Overlapping USB receive with SD writes (writer task on core 1) didn't
    change it: the device-side stats showed the SD writer idle 90% of the
    time.
  - Neither did bypassing the driver to read the USB FIFO directly.
  - So the limit is on the Mac side, most likely the macOS tty layer. Wi-Fi
    would be ~3–5× faster; a card reader faster still.
- **Bug found the hard way:** renaming a file while it's still open in
  FATFS leaves it 0 bytes long, because the size is written to the old
  directory entry on close. Hence close → rename → verify.

## Ideas that were checked and set aside

- **Weights in RAM:** ~70 KB spare is 0.03% of a token's reads.
- **Weights in internal flash:** ~5.9 MB at ~40 MB/s saves ~3% of reads.
  Better spent on the speculative-decoding table.
- **SD + Wi-Fi in parallel:** measured later (experiments.md #29, code removed): 10.3–10.8 MB/s
  combined vs 9.3, but Wi-Fi needs 73–121 KB of RAM, which costs the prompt
  batch (7 → 3–4) and makes short answers slower overall.
- **64 KB read blocks:** need a contiguous 64 KB block the heap doesn't have.

## Next steps

1. **Speculative decoding.** CPU is idle ~78% of a generated token, so
   verifying 2 drafts per pass is nearly free. Plan: prompt lookup plus a
   1 MB n-gram table in flash; batch-3 verification with one shared LM head
   scan. Expected ~10–11 s/token.
2. **Scale step in assembly.** The C part is ~32 of ~53 cycles per block.
   That would matter once compute is the limit again (batch verification).
3. **Faster uploads (Wi-Fi)** if model variants get swapped often.

## Reproducing

```
uv run tools/export.py --quant q4                      # out/lfm-q4.lfm (device file)
uv run tools/upload.py                                 # to the card over USB
# Zed: fw: flash + monitor
./host/build/lfm_tests                                 # host tests (golden, async, batch, int8)
./host/build/lfm_eval out/eval.txt out/lfm-bf16.lfm out/lfm-q4.lfm   # quality
uv run tools/spec_eval.py                              # speculative decoding simulation
```

In `firmware/main/main.cpp`, `kSdSpeedTest` and `kKernelTest` switch the
firmware to the SD clock test or the SIMD kernel test.
