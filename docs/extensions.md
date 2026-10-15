# Possible extensions: hardware changes

This project runs LFM2.5-230M on a **stock** Cardputer-Adv: nothing added,
nothing soldered. Within that rule the device is at its limit. Each pass
reads ~91 MB of Q4 weights over a 1-bit SD bus at its 80 MHz ceiling,
~9.4 MB/s, with the card busy ~92% of the time (see
[performance.md](performance.md), [experiments.md](experiments.md)).

Every idea below changes the hardware. At that point it stops being "run a
model on a Cardputer" and becomes a hardware project, so none of it is part
of this one. It's written down because the measurements say where the next
factor of two would come from, and how much work it would be.

## Why the SD bus is the lever

| | today |
|---|---|
| a pass (one weight read; ~2.0 tokens on chat with tree drafts) | ~12.3 s |
| of which reading the weights | ~9.5 s |
| of which core-0 compute (layers, LM head, drafts; overlaps the reads) | ~6.2 s |

Speed = clock × data lines. The clock is maxed (80 MHz: the controller's
160 MHz PLL ÷ 2; 160 MHz doesn't answer, #36). The faster SD modes need
1.8 V signalling the board doesn't have. Only the number of data lines can
grow: the slot is wired for SPI, so only DAT0 is connected.

| bus | read speed | pass (est.) |
|---|---|---|
| 1-bit, 80 MHz (today) | ~9.4 MB/s | ~12.3 s (read-bound) |
| 4-bit, 40 MHz | ~19 MB/s | ~6.5 s (compute-bound) |
| 4-bit, 80 MHz | ~35–40 MB/s | ~6.2 s compute; ~3.5–4 s if the layer math is split over both cores (core 1 is free once the DMA does the reading) |

So about **2× with a 4-bit card**, and close to **3×** with the compute also
spread over both cores.

## Option A: wire the internal slot for 4-bit

The microSD contacts DAT1 (contact 8) and DAT2 (contact 1) have no
connection on this board. Two thin wires from those socket pins to free
GPIOs make the built-in slot 4-bit:

| SD contact | GPIO | where |
|---|---|---|
| 8, DAT1 | G3 | EXT header pin 1 |
| 1, DAT2 | G4 | EXT header pin 3 |

- Confirm first with a multimeter: VSS (contact 6) goes to GND and VDD
  (contact 4) to 3.3 V, which tells you which end is contact 1. Contacts 1
  and 8 must connect to nothing.
- Use 30 AWG wire-wrap wire, ~3 cm, fine tip, flux; the socket pins are ~1 mm
  apart, next to CMD and DAT0. Add 10 kΩ pull-ups to 3.3 V if the
  ESP32's internal ones aren't enough.
- Permanent and fiddly; the most compact result.

## Option B: a second card on the EXT header, 4-bit

The ESP32-S3's SD controller has a second slot that can use any GPIOs. The
EXT header has exactly six free pins for it:

| SD line | GPIO | EXT pin |
|---|---|---|
| CLK | G5 (LoRa CS) | 13 |
| CMD | G6 (LoRa BUSY) | 5 |
| DAT0 | G3 | 1 |
| DAT1 | G4 | 3 |
| DAT2 | G13 (UART TX; the console is on USB) | 12 |
| DAT3 | G15 (UART RX) | 14 |
| power | 5VOUT → 3.3 V regulator | 6 |
| GND | | 4 |

Not usable: G40/G14/G39 on the EXT header are the internal slot's own lines,
and G8/G9 are the keyboard's I2C bus. The two slots share one controller,
so only one card transfers at a time: the model lives on the external card.

### B1: off-the-shelf parts (~€15–20)

| Part | Example | Notes |
|---|---|---|
| microSD breakout with all 8 pins | Adafruit 4682 "Micro SD SPI or SDIO Card Breakout – 3V only", or Pololu 2597 "Breakout Board for microSD Card" | 3.3 V only, no level shifters. Pololu: no pull-ups. Neither lists a regulator |
| 3.3 V regulator | small AMS1117-3.3 / AP2112-3.3 module, ≥ 300 mA | the EXT header only has 5 V; cards draw up to ~100–200 mA |
| pull-ups (if the breakout has none) | 5 × 10 kΩ | CMD, DAT0–3 to 3.3 V |
| wiring | 2×7 2.54 mm male header and short wires (< 5 cm) | |

Check that EXT pin 6 (5VOUT) carries 5 V with the Cardputer on, also on
battery. Expect 4-bit at 40 MHz (20 MB/s); 80 MHz on loose wires is
unlikely.

### B2: a custom EXT "cap" PCB (best signal quality)

A ~25 × 20 mm two-layer board that plugs straight into the EXT socket:

- J1: 2×7 2.54 mm male header (pins as in the table above)
- J2: microSD push-push socket (e.g. Molex 104031-0811, Hirose DM3AT)
- U1: AP2112K-3.3 (SOT-23-5) from 5VOUT; 1 µF in, 1 µF + 10 µF out
- 100 nF + 10 µF at the socket's VDD
- 10 kΩ pull-ups on CMD and DAT0–3; 22–33 Ω series resistor on CLK, at J1
- Bottom layer solid ground; signal traces < 15 mm, about equal length,
  CLK away from the data lines

About $10–20 for five boards from a PCB service. Short traces make 80 MHz
4-bit realistic. Before laying it out, check the EXT socket's position and
pin 1 against the case.

## Option C: a second processor with more RAM

A board with PSRAM (an ESP32-S3 module with 8 MB of PSRAM, for instance) on
the EXT header could hold the KV cache, larger prompt batches and read-ahead
buffers, or run part of the model, with the Cardputer as keyboard and
screen. That's a different machine with a Cardputer front end, not this
project.

## Option D: Wi-Fi

Measured (#29): streaming part of the weights from a Mac over Wi-Fi gives
10.3–10.8 MB/s together with the card, but needs 73–121 KB of RAM, which
costs the prompt batch. No soldering, but a server on the network, and a net
loss on this board. Code removed.

## Software for options A and B

- Mount in 4-bit mode: `SDMMC_HOST_FLAG_4BIT`, slot width 4, D1/D2 (and for
  B, all six pins) routed through the GPIO matrix, the same
  `set_card_clk` overclock as today.
- A test mode first: sweep 20/40/53/80 MHz and the sampling phase, compare
  every read with a checksum, measure MB/s (like the 1-bit sweep in #2–4).
- Streaming (#35) works unchanged on a 4-bit bus.
- After that the pass is compute-bound: split the layer matvecs between the
  two cores (core 1 then has only the DMA to wait on).
