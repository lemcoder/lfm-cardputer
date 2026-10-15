# Running a 230M-parameter language model on a €60 pocket computer

*How LFM2.5-230M went from 63 seconds per token to about 4 on an ESP32-S3 with
no PSRAM, every weight streamed from a microSD card.*

---

The M5Stack Cardputer-Adv is a credit-card-sized computer with a keyboard, a
small screen and a microSD slot. Inside is an ESP32-S3: two cores at 240 MHz,
about 512 KB of SRAM (roughly 300 KB of it free for us), 8 MB of flash and no
PSRAM.

LFM2.5-230M is a small language model from Liquid AI. Even at 4 bits per
weight its file is about 130 MB, roughly 400 times the free RAM.

The question was simple: can it run on the stock device, with no soldering
and no extra boards, and how fast can we make it?

It runs. You type "Hello" on the Cardputer's keyboard, and the model answers
on its own screen at about 6 seconds per token. On a benchmark prompt it
reaches 4.4 seconds per token. The first working build took 63.

## The one rule that decides everything

The weights don't fit in RAM, so every token has to read every weight from
the SD card once. That gives the whole project one equation:

```
seconds per token ≈ max(bytes read per token / SD bandwidth,
                        compute per token)
```

So there are two levers: read fewer bytes per useful token, or read them
faster, while making sure the CPU never becomes the bottleneck. Every
optimization below pulls one of these levers.

## Step 0: the first working build — 413 seconds for "Paris."

The first version used 8-bit weights, read the SD card at 40 MHz (the
maximum the ESP-IDF driver allows), and did one thing at a time: read a
block, compute on it, read the next. For the prompt "The capital of France
is", it answered " Paris." after **413 seconds**: 63 seconds per generated
token. The CPU waited on the card 85% of the time.

The output matched the PyTorch reference token for token, which mattered
more than speed at that point. Every later step was checked against it.

## The SD card is a one-lane road

The usual figure for SD on an ESP32 is 20–25 MB/s, using 4-bit SD mode. The
Cardputer's slot schematic tells a different story: the slot is wired for
SPI, and the card's DAT1 and DAT2 pins aren't connected to anything. So
4-bit mode is physically impossible.

What does work is **SD 1-bit mode** on the same wires, which is faster than
SPI. Its ceiling is the clock divided by 8, in bytes per second.

Then came the overclock. The S3's SD controller divides a fixed 160 MHz
clock by an integer, and the driver forces anything at or above 40 MHz to
divide by 4. Rewriting the divider register right after the driver sets it
up gives 53 MHz (÷3) and then **80 MHz (÷2)**.

SD data is CRC-checked by the protocol, so a clock that's too fast shows up
as read errors, not as silently corrupted weights. 80 MHz was clean at three
of four sampling phases. 160 MHz (÷1) didn't answer at all; the faster SD
modes need 1.8 V signalling, which this board doesn't have. So 80 MHz,
about 9.4 MB/s in practice, is the ceiling.

Along the way:

- **Read while computing.** A reader task on core 1 fills one 32 KB buffer
  while core 0 computes on the other.
- **Align every read to SD sectors.** Before this, reads that started
  mid-sector cost FATFS an extra command each: +2.5%.
- **Enable FATFS fast seek.** Without it, seeking back in a 130 MB file
  walks the FAT cluster chain.

## Fewer bytes: 4 bits, a smarter LM head, and batching

**Q4 everywhere.** Going from 8-bit to 4-bit weights halved the bytes per
token. It costs about 14% perplexity on the host. The model still answers
" Paris.", and Q4 became the floor: nothing below it.

**The LM head shortlist.** The output layer maps the hidden state onto the
whole 65K-token vocabulary, and was ~38 MB per token by itself. Instead, a
**rank-128 low-rank copy** lives in the S3's internal flash (memory-mapped,
4.7 MB). It picks the 768 most likely tokens, and only those rows are read
exactly from the SD card. It agrees with the full head 99.5% of the time.

**Batched prefill.** During the prompt, every weight row is applied to
several prompt tokens before moving on, so the prompt costs one read of the
weights per *batch*. Getting the batch to 7 tokens was mostly a RAM fight:

- splitting big allocations into small ones, because the heap had 88 KB
  free but its largest block was 31 KB;
- moving the conv state and the KV cache onto the SD card;
- shrinking FATFS buffers from 4 KB to 512 bytes per open file.

## The CPU: SIMD that only does integers

At Q4 the CPU became the bottleneck. The S3 has a SIMD extension (PIE) with
128-bit registers, but it's integer-only. So activations are quantized to
int8 in blocks of 32, llama.cpp-style, and the dot products are written in
assembly.

Q4 blocks are 18 bytes, so every load is unaligned; unpacking the nibbles
takes a mask, a shift and a subtract. The result is 6–7× faster than the C
reference, bit-identical on 1280 random rows at every byte misalignment.

One pitfall cost a day: ESP-IDF compiles with `-fno-builtin-memcpy`, so
every `memcpy` is a real function call. Two of them per block in the scale
step cost more than the SIMD work itself.

## Reading ahead in time: speculative decoding

Once the card is saturated, the only way to get more tokens per second is
more tokens **per read of the weights**. Speculative decoding does exactly
that: guess the next few tokens cheaply, then verify all of them in one pass
over the weights. Correct guesses are free tokens, and the output stays
identical to plain greedy decoding.

**Attempt 1: an n-gram table from WikiText.** In simulation it promised
1.5 tokens per pass. On the device, **0 of 85 drafts were accepted.** The
table was full of WikiText artifacts like `@,@` that the model never writes.
The fix was to build the table from **the model's own generated text**,
which helped, but chat replies were still mostly misses.

**Attempt 2: a draft head.** This one worked. It's a tiny network in the
style of Medusa and Hydra that reads the model's final hidden state and
guesses the next token. It feeds each guess back in to make the next one.
It was trained on 3.2M positions of the Q4 model's own output (WikiText
openings, Alpaca prompts, ChatML conversations) and stored in flash as
0.86 MB at Q4.

Training data mattered more than architecture: trained on WikiText only,
the head was right 51% of the time on WikiText but ~31% on real prompts.

**Attempt 3: trees instead of chains.** Rather than verifying one chain of
guesses, the pass verifies a small tree: the head's top 3 first guesses,
plus a few continuations. This needed tree attention: every node attends to
its own ancestors, and the convolution windows follow the ancestry. After
verification, the accepted path's state is rebuilt.

The result is **~2 tokens per pass on chat**. Exact LM head re-scoring then
follows only the path that can still be accepted, which saved another 4%.

## Patching the SD driver

Profiling showed the card was idle ~10% of the time between read commands.
Each read command has a fixed ~183 µs cost, and IDF's driver adds an extra
status round-trip per read.

The fix was a project copy of IDF's `esp_driver_sdmmc` with a **streaming
API**: one multi-block SD command per weight matrix. The DMA ring takes the
engine's buffers as they're handed back, and when none is free, the host
stops the card clock and waits. Raw throughput went from 9.46 to 9.91 MB/s,
and the benchmark got **10% faster**.

Flash moved from 80 to 120 MHz, which sped up the LM head and draft-head
reads.

## A chat on the device

With speed in a usable range, the Cardputer got a proper interface:

- **A keyboard driver** for the TCA8418 controller on I2C.
- **A BPE tokenizer on the device.** It matches Hugging Face on 4729 of
  4729 test strings; its merge table lives on the SD card behind a 4 KB
  index.
- **Multi-turn ChatML** in a single KV cache.
- **Prefill while typing.** The chat template header is precomputed, and
  while you type, every finished word is already pushed through the model.
  Only the last couple of pieces wait for Enter. A "Hello" turn went from
  127 to 70 seconds.

## What didn't work

Half of the work was ideas that failed. Each was measured, written up, and
the code removed:

- **Wi-Fi + SD in parallel.** Streaming part of the weights from a laptop
  gave 10.3–10.8 MB/s combined, but Wi-Fi needs 73–121 KB of RAM. That cost
  the prompt batch, so it was a net loss.
- **2-bit and 3-bit weights.** At 2 bits, even with GPTQ, perplexity went
  from 53 to 224+: broken. 3-bit was usable (68.6) but not worth it.
- **Lossless compression of the Q4 weights.** A Huffman codec decoded in
  place while the DMA was still writing saved 8.4% of the bytes and 3.7% per
  token. But its 8 KB of tables pushed the prompt batch from 7 to 6, which
  also disabled the tree. Net loss.
- **Dropping layers.** Removing even the most expendable layer took
  perplexity from 8.4 to 14.4.
- **EAGLE-2 dynamic trees and a distilled draft head.** The papers' gains
  come from 50–60 draft tokens per pass; at a budget of 6, both tied the
  simple fixed tree.

Lesson: **on 300 KB of RAM, every kilobyte has a price in prompt batch
size**, and most ideas lose once you pay it.

## The scoreboard

```
                                     per token    "The capital of France is"
first working build (Q8, 40 MHz)     63.1 s       413 s
+ overlap, LM shortlist, batching    46.1 s       177 s
+ SD overclock to 80 MHz             24.0 s        95 s
+ Q4 + SIMD + batch 4                14.8 s        55 s
+ flash LM head, n-gram drafts, ...   7.5 s *      35 s
+ draft head                          5.6 s *      24 s
+ 120 MHz flash, streamed SD reads    5.0 s *      21 s
+ tree drafts, lazy re-score          4.4 s *      21 s
                                     * planets benchmark; chat replies ~6.1 s
```

About **14× faster per token** than the first build, on the same unmodified
hardware.

## The last metric: CPU idle time

Once the SD card was saturated, speed stopped moving. So the last target
was CPU headroom: how much of core 0's time is spent waiting on the card,
free for anything else.

A profiler that charges every moment to a category showed core 0 was busy
82% of the time. With a prompt batch or a draft tree, each weight block was
being unpacked once *per token*.

A multi-token kernel unpacks each Q4 block once and applies it to all
tokens. Its final form is hand-written assembly, with per-token accumulators
in float registers and the same operations in the same order as the C
version, so the output stays bit-identical. Results:

- CPU idle went from **18% to 54.5%**;
- a 7-token block went from 351 to 157 cycles.

The device is now firmly SD-bound with the CPU half idle.

## Where it ends

On a stock Cardputer this is close to the limit: a one-bit bus at its
maximum clock, busy ~92% of the time. The next 2–3× would come from wiring
the card for 4-bit mode or adding a second card on the expansion header.
That changes the project from "run a model on a Cardputer" to a hardware
project, so it's written down as a possible extension and left there.

What surprised me most:

- **The bottleneck was always the bus, never the math.** Every big win
  either read fewer bytes or got more tokens out of each read.
- **Measure on the device.** Simulations promised 1.5 tokens per pass for
  the WikiText drafts; the device accepted 0 of 85.
- **Train the draft model on the target model's own output.** Not on clean
  human text.

The code, every measurement, and the list of failed experiments are in the
repo.
