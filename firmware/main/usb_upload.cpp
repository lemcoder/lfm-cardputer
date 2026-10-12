#include "usb_upload.hpp"

#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "driver/usb_serial_jtag.h"
#include "driver/usb_serial_jtag_vfs.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "lfm/source.hpp"
#include "sd_card.hpp"

namespace upload {
namespace {

constexpr std::size_t kChunk = 32 * 1024;
constexpr uint32_t kProgressEvery = 1024 * 1024;
constexpr TickType_t kDataTimeout = pdMS_TO_TICKS(10'000);

uint32_t g_crc_table[256];

void crc_init() {
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        g_crc_table[i] = c;
    }
}

// zlib-compatible CRC-32: start with 0, feed chunks, compare with zlib.crc32().
uint32_t crc_update(uint32_t crc, const uint8_t* p, std::size_t n) {
    crc = ~crc;
    for (std::size_t i = 0; i < n; ++i) crc = g_crc_table[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
    return ~crc;
}

void send(const char* line) {
    usb_serial_jtag_write_bytes(line, std::strlen(line), pdMS_TO_TICKS(1000));
}

// One '\n'-terminated line (without it). False on timeout.
bool read_line(char* buf, std::size_t cap, int64_t deadline_us) {
    std::size_t n = 0;
    while (deadline_us < 0 || esp_timer_get_time() < deadline_us) {
        char c;
        if (usb_serial_jtag_read_bytes(&c, 1, pdMS_TO_TICKS(50)) != 1) continue;
        if (c == '\r') continue;
        if (c == '\n') {
            buf[n] = '\0';
            return true;
        }
        if (n + 1 < cap) buf[n++] = c;
    }
    return false;
}

void ensure_driver() {
    if (usb_serial_jtag_is_driver_installed()) return;
    usb_serial_jtag_driver_config_t cfg = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    cfg.rx_buffer_size = 16 * 1024;
    cfg.tx_buffer_size = 1024;
    usb_serial_jtag_driver_install(&cfg);
    usb_serial_jtag_vfs_use_driver();  // logs now go through the driver too
}

// Receive `size` bytes from USB into `tmp`. The file is closed on return
// (FileSource's destructor), which is what makes FATFS record its size.
const char* receive_to(const char* tmp, const char* name, uint32_t size, uint32_t want_crc, uint8_t* buf,
                       StatusFn status) {
    lfm::FileSource out(tmp, lfm::FileSource::Mode::ReadWrite);
    if (!out.ok()) return "can't create file";
    send("UP:READY\n");

    const int64_t t0 = esp_timer_get_time();
    uint32_t done = 0, crc = 0, next_progress = kProgressEvery;
    char line[160];
    while (done < size) {
        const uint32_t want = size - done < kChunk ? size - done : kChunk;
        uint32_t fill = 0;
        while (fill < want) {
            const int got = usb_serial_jtag_read_bytes(buf + fill, want - fill, kDataTimeout);
            if (got <= 0) return "timeout waiting for data";
            fill += static_cast<uint32_t>(got);
        }
        crc = crc_update(crc, buf, fill);
        if (!out.write_at(done, buf, fill)) return "SD write failed";
        done += fill;
        if (done >= next_progress || done == size) {
            next_progress += kProgressEvery;
            std::snprintf(line, sizeof(line), "UP:PROG %lu\n", static_cast<unsigned long>(done));
            send(line);
            const double s = (esp_timer_get_time() - t0) / 1e6;
            std::snprintf(line, sizeof(line), "%s %lu/%lu MB %.2f MB/s", name, static_cast<unsigned long>(done >> 20),
                          static_cast<unsigned long>(size >> 20), done / 1e6 / s);
            status(line);
        }
    }
    return crc == want_crc ? nullptr : "CRC mismatch (USB)";
}

// Read the file back from the card and check size and CRC.
const char* verify(const char* path, uint32_t size, uint32_t want_crc, uint8_t* buf, StatusFn status) {
    struct stat st;
    if (stat(path, &st) != 0) return "file missing after rename";
    if (static_cast<uint32_t>(st.st_size) != size) return "wrong size on card";
    status("verifying on card...");
    lfm::FileSource in(path);
    if (!in.ok()) return "can't reopen file";
    uint32_t crc = 0;
    for (uint32_t off = 0; off < size; off += kChunk) {
        const uint32_t n = size - off < kChunk ? size - off : kChunk;
        if (!in.read_at(off, buf, n)) return "read-back failed";
        crc = crc_update(crc, buf, n);
    }
    return crc == want_crc ? nullptr : "CRC mismatch (card)";
}

// Receive one file. Returns an error string, or nullptr on success.
const char* receive_file(const char* name, uint32_t size, uint32_t want_crc, uint8_t* buf, StatusFn status) {
    char path[96], tmp[100];
    std::snprintf(path, sizeof(path), "%s/%s", SdCard::kMountPoint, name);
    std::snprintf(tmp, sizeof(tmp), "%s.tmp", path);

    if (const char* err = receive_to(tmp, name, size, want_crc, buf, status)) return err;

    // Only rename closed files. FATFS rename doesn't overwrite.
    struct stat st;
    if (stat(path, &st) == 0) unlink(path);
    if (rename(tmp, path) != 0) return "rename failed";
    return verify(path, size, want_crc, buf, status);
}

}  // namespace

int serve(int wait_ms, StatusFn status) {
    ensure_driver();
    crc_init();

    // Handshake.
    char line[160];
    const int64_t deadline = wait_ms < 0 ? -1 : esp_timer_get_time() + int64_t{wait_ms} * 1000;
    bool hello = false;
    while (!hello) {
        if (!read_line(line, sizeof(line), deadline)) return 0;
        hello = std::strcmp(line, "LFMUP HELLO") == 0;
    }
    send("UP:HELLO\n");
    status("USB upload: connected");

    auto* buf = static_cast<uint8_t*>(heap_caps_malloc(kChunk, MALLOC_CAP_DMA | MALLOC_CAP_8BIT));
    if (buf == nullptr) {
        send("UP:ERR out of memory\n");
        return 0;
    }

    int files = 0;
    while (read_line(line, sizeof(line), -1)) {
        if (std::strcmp(line, "LFMUP HELLO") == 0) {
            send("UP:HELLO\n");
            continue;
        }
        if (std::strcmp(line, "LFMUP END") == 0) {
            send("UP:BYE\n");
            break;
        }
        char name[64];
        unsigned long size = 0, crc = 0;
        if (std::sscanf(line, "LFMUP FILE %63s %lu %lu", name, &size, &crc) != 3) continue;
        if (std::strchr(name, '/') != nullptr) {  // card root only
            send("UP:ERR bad name\n");
            continue;
        }
        const char* err = receive_file(name, static_cast<uint32_t>(size), static_cast<uint32_t>(crc), buf, status);
        if (err == nullptr) {
            ++files;
            send("UP:OK\n");
        } else {
            std::snprintf(line, sizeof(line), "UP:ERR %s\n", err);
            send(line);
            status(err);
        }
    }
    heap_caps_free(buf);
    return files;
}

}  // namespace upload
