# /// script
# requires-python = ">=3.10"
# dependencies = ["pyserial"]
# ///
"""Copy files to the Cardputer's SD card over the USB cable (no card reader).

    uv run tools/upload.py                        # out/lfm-q8.lfm -> lfm.lfm, out/prompt.txt -> prompt.txt
    uv run tools/upload.py --prompt-only          # just prompt.txt (seconds)
    uv run tools/upload.py SRC:NAME ...           # any files, NAME = file name on the card root

Resets the device; the firmware listens for 1.5 s after boot (or until this
tool connects, if the model is missing). Protocol: firmware/main/usb_upload.hpp.
Close any serial monitor first: only one program can hold the port.
"""

import argparse
import glob
import sys
import time
import zlib
from pathlib import Path

import serial

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILES = [(ROOT / "out/lfm-q4.lfm", "lfm.lfm"), (ROOT / "out/prompt.txt", "prompt.txt")]
CHUNK = 64 * 1024


def find_port() -> str:
    ports = sorted(glob.glob("/dev/cu.usbmodem*"))
    if not ports:
        sys.exit("no /dev/cu.usbmodem* port: is the Cardputer plugged in?")
    return ports[0]


def open_port(port: str, timeout: float = 10.0) -> serial.Serial:
    end = time.time() + timeout
    while True:
        try:
            return serial.Serial(port, 115200, timeout=0.05)
        except serial.SerialException:
            if time.time() > end:
                raise
            time.sleep(0.1)


def reset(port: str) -> None:
    s = open_port(port)
    s.dtr = False
    s.rts = True  # USB-Serial-JTAG: RTS pulse resets the chip
    time.sleep(0.1)
    s.rts = False
    s.close()
    time.sleep(0.3)  # USB re-enumerates


class Device:
    def __init__(self, s: serial.Serial):
        self.s = s
        self.buf = b""

    def lines(self):
        """Complete lines received so far (non-blocking)."""
        self.buf += self.s.read(self.s.in_waiting or 1)
        *done, self.buf = self.buf.split(b"\n")
        return [l.decode(errors="replace").strip() for l in done]

    def wait_for(self, prefixes: tuple[str, ...], timeout: float) -> str:
        end = time.time() + timeout
        while time.time() < end:
            for line in self.lines():
                if line.startswith(prefixes):
                    return line
        raise TimeoutError(f"no {prefixes} from device")

    def send_line(self, text: str) -> None:
        self.s.write(text.encode() + b"\n")


def connect(port: str) -> Device:
    print(f"resetting {port} ...")
    reset(port)
    dev = Device(open_port(port))
    end = time.time() + 15
    while time.time() < end:
        dev.send_line("LFMUP HELLO")
        try:
            dev.wait_for(("UP:HELLO",), 0.15)
            return dev
        except TimeoutError:
            pass
    sys.exit("device didn't answer: is the upload-capable firmware flashed? (fw: flash + monitor)")


def send_file(dev: Device, src: Path, name: str) -> None:
    data = src.read_bytes()
    crc = zlib.crc32(data)
    dev.send_line(f"LFMUP FILE {name} {len(data)} {crc}")
    reply = dev.wait_for(("UP:READY", "UP:ERR"), 10)
    if reply.startswith("UP:ERR"):
        sys.exit(f"{name}: {reply}")

    t0 = time.time()
    stat = ""
    for off in range(0, len(data), CHUNK):
        dev.s.write(data[off:off + CHUNK])  # blocks while the device is busy: USB flow control
        for line in dev.lines():
            if line.startswith("UP:ERR"):
                sys.exit(f"\n{name}: {line}")
            if line.startswith("UP:STAT"):
                stat = line
        done = min(off + CHUNK, len(data))
        rate = done / 1e6 / max(time.time() - t0, 1e-6)
        eta = (len(data) - done) / 1e6 / max(rate, 1e-6)
        print(f"\r{name}: {done / 1e6:7.1f}/{len(data) / 1e6:.1f} MB  {rate:4.2f} MB/s  eta {eta:4.0f} s",
              end="", flush=True)
    print(f"\r{name}: sent, device verifying the copy on the card ..." + " " * 30, end="", flush=True)
    reply = dev.wait_for(("UP:OK", "UP:ERR", "UP:STAT"), 300)  # read-back: ~50 s for the model
    if reply.startswith("UP:STAT"):
        stat = reply
        reply = dev.wait_for(("UP:OK", "UP:ERR"), 300)
    if stat:
        print(f"\r{name}: device {stat[len('UP:STAT '):]}" + " " * 20)
    print(f"\r{name}: {len(data) / 1e6:.1f} MB in {time.time() - t0:.0f} s  {reply}" + " " * 20)
    if reply.startswith("UP:ERR"):
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="SRC:NAME pairs (default: model + prompt)")
    ap.add_argument("--prompt-only", action="store_true")
    ap.add_argument("--port")
    args = ap.parse_args()

    if args.files:
        files = [(Path(f.split(":", 1)[0]), f.split(":", 1)[1]) for f in args.files]
    elif args.prompt_only:
        files = DEFAULT_FILES[1:]
    else:
        files = DEFAULT_FILES
    for src, _ in files:
        if not src.exists():
            sys.exit(f"{src} missing (uv run tools/export.py)")

    dev = connect(args.port or find_port())
    for src, name in files:
        send_file(dev, src, name)
    dev.send_line("LFMUP END")
    dev.wait_for(("UP:BYE",), 5)
    print("done; the Cardputer is starting the model")


if __name__ == "__main__":
    main()
