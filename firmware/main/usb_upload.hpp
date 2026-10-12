#pragma once
// Receive files over the USB cable (USB-Serial-JTAG, the same port used for
// flashing and logs) and write them to the SD card. Host side: tools/upload.py.
//
// Protocol (text lines from the host start with "LFMUP", replies with "UP:";
// anything else on the line, e.g. log output, is ignored by both sides):
//   host: LFMUP HELLO                      device: UP:HELLO
//   host: LFMUP FILE <name> <size> <crc32> device: UP:READY
//   host: <size raw bytes>                 device: UP:PROG <bytes> ... UP:OK | UP:ERR <why>
//   host: LFMUP END                        device: UP:BYE
// Files are written to <name>.tmp and renamed once the CRC-32 (zlib) matches.

#include <cstdint>

namespace upload {

using StatusFn = void (*)(const char* line);

// Wait up to wait_ms for the host (forever if wait_ms < 0), then receive
// files until the host says END. Returns the number of files written, or 0
// if no host showed up. `status` gets short human-readable progress lines.
int serve(int wait_ms, StatusFn status);

}  // namespace upload
