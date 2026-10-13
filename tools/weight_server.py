# /// script
# requires-python = ">=3.10"
# ///
"""Serve byte ranges of the model file to the Cardputer over Wi-Fi.

    uv run tools/weight_server.py                      # out/lfm-q4-head-lr256.lfm on port 7777
    uv run tools/weight_server.py --file F --port P

Protocol (firmware/main/wifi_test.cpp): the device sends requests of
12 bytes, little-endian <uint64 offset, uint32 length>, and may send several
before reading; the server answers each with exactly `length` bytes of the
file, in order. A read past the end is answered with zeros.

The device finds this server through /sdcard/wifi.txt, one value per line:
SSID, password, server IP, port (uv run tools/upload.py out/wifi.txt:wifi.txt).
"""

import argparse
import os
import socket
import struct
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQ = struct.Struct("<QI")


def recv_exact(conn: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        part = conn.recv(n - len(buf))
        if not part:
            return None
        buf += part
    return bytes(buf)


def serve(conn: socket.socket, addr, path: Path) -> None:
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    size = path.stat().st_size
    sent = 0
    t0 = time.time()
    with open(path, "rb") as f:
        fd = f.fileno()
        while True:
            req = recv_exact(conn, REQ.size)
            if req is None:
                break
            offset, length = REQ.unpack(req)
            n = max(0, min(length, size - offset))
            done = 0
            while done < n:
                done += os.sendfile(conn.fileno(), fd, offset + done, n - done)
            if n < length:
                conn.sendall(bytes(length - n))
            sent += length
    s = time.time() - t0
    print(f"{addr[0]}: closed, {sent / 1e6:.1f} MB in {s:.1f} s ({sent / 1e6 / max(s, 1e-9):.2f} MB/s)", flush=True)
    conn.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=Path, default=ROOT / "out/lfm-q4-head-lr256.lfm")
    ap.add_argument("--port", type=int, default=7777)
    args = ap.parse_args()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", args.port))
    srv.listen(4)
    print(f"serving {args.file} ({args.file.stat().st_size / 1e6:.1f} MB) on port {args.port}", flush=True)
    while True:
        conn, addr = srv.accept()
        print(f"{addr[0]}: connected", flush=True)
        threading.Thread(target=serve, args=(conn, addr, args.file), daemon=True).start()


if __name__ == "__main__":
    main()
