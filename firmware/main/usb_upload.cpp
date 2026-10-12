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
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "lfm/source.hpp"
#include "sd_card.hpp"

namespace upload {
namespace {

// Receiving (USB, core 0) and writing (SD + CRC, core 1) overlap: the
// receiver fills one block while the writer drains another.
constexpr int kBlocks = 3;
constexpr std::size_t kBlockSize = 64 * 1024;
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
    cfg.rx_buffer_size = 32 * 1024;
    cfg.tx_buffer_size = 1024;
    usb_serial_jtag_driver_install(&cfg);
    usb_serial_jtag_vfs_use_driver();  // logs now go through the driver too
}

// ---------------------------------------------------------------- writer task

struct Writer {
    uint8_t* block[kBlocks] = {};
    uint32_t len[kBlocks] = {};
    QueueHandle_t free_q = nullptr;  // block indices the receiver may fill
    QueueHandle_t full_q = nullptr;  // block indices waiting to be written; -1 = stop
    SemaphoreHandle_t stopped = nullptr;
    lfm::FileSource* out = nullptr;
    uint32_t offset = 0;
    uint32_t crc = 0;
    bool failed = false;
    int64_t busy_us = 0;  // time spent writing + CRC
};

void writer_task(void* arg) {
    auto* w = static_cast<Writer*>(arg);
    while (true) {
        int idx;
        xQueueReceive(w->full_q, &idx, portMAX_DELAY);
        if (idx < 0) break;
        const int64_t t0 = esp_timer_get_time();
        w->crc = crc_update(w->crc, w->block[idx], w->len[idx]);
        if (!w->failed && !w->out->write_at(w->offset, w->block[idx], w->len[idx])) w->failed = true;
        w->offset += w->len[idx];
        w->busy_us += esp_timer_get_time() - t0;
        xQueueSend(w->free_q, &idx, portMAX_DELAY);
    }
    xSemaphoreGive(w->stopped);
    vTaskDelete(nullptr);
}

// ---------------------------------------------------------------- one file

// Receive `size` bytes from USB into `tmp`. The file is closed on return
// (FileSource's destructor), which is what makes FATFS record its size.
const char* receive_to(Writer& w, const char* tmp, const char* name, uint32_t size, uint32_t want_crc,
                       StatusFn status) {
    lfm::FileSource out(tmp, lfm::FileSource::Mode::ReadWrite);
    if (!out.ok()) return "can't create file";

    w.out = &out;
    w.offset = 0;
    w.crc = 0;
    w.failed = false;
    w.busy_us = 0;
    xQueueReset(w.free_q);
    xQueueReset(w.full_q);
    for (int i = 0; i < kBlocks; ++i) xQueueSend(w.free_q, &i, 0);
    xTaskCreatePinnedToCore(writer_task, "sd_writer", 4096, &w, 5, nullptr, 1);

    send("UP:READY\n");
    const int64_t t0 = esp_timer_get_time();
    int64_t wait_us = 0;  // receiver blocked on the writer (SD is the bottleneck)
    uint32_t done = 0, next_progress = kProgressEvery;
    const char* err = nullptr;
    char line[160];
    while (done < size) {
        int idx;
        const int64_t tw = esp_timer_get_time();
        xQueueReceive(w.free_q, &idx, portMAX_DELAY);
        wait_us += esp_timer_get_time() - tw;
        if (w.failed) {
            err = "SD write failed";
            break;
        }

        const uint32_t want = size - done < kBlockSize ? size - done : kBlockSize;
        uint32_t fill = 0;
        while (fill < want) {
            const int got = usb_serial_jtag_read_bytes(w.block[idx] + fill, want - fill, kDataTimeout);
            if (got <= 0) break;
            fill += static_cast<uint32_t>(got);
        }
        if (fill < want) {
            err = "timeout waiting for data";
            break;
        }
        w.len[idx] = fill;
        xQueueSend(w.full_q, &idx, portMAX_DELAY);
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

    // Stop the writer (it finishes queued blocks first).
    const int stop = -1;
    xQueueSend(w.full_q, &stop, portMAX_DELAY);
    xSemaphoreTake(w.stopped, portMAX_DELAY);
    w.out = nullptr;

    const double total = (esp_timer_get_time() - t0) / 1e6;
    std::snprintf(line, sizeof(line), "UP:STAT %.1f s, %.2f MB/s, waited on SD %.1f s, SD+CRC busy %.1f s\n", total,
                  done / 1e6 / total, wait_us / 1e6, w.busy_us / 1e6);
    send(line);

    if (err != nullptr) return err;
    if (w.failed) return "SD write failed";
    return w.crc == want_crc ? nullptr : "CRC mismatch (USB)";
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
    for (uint32_t off = 0; off < size; off += kBlockSize) {
        const uint32_t n = size - off < kBlockSize ? size - off : kBlockSize;
        if (!in.read_at(off, buf, n)) return "read-back failed";
        crc = crc_update(crc, buf, n);
    }
    return crc == want_crc ? nullptr : "CRC mismatch (card)";
}

// Receive one file. Returns an error string, or nullptr on success.
const char* receive_file(Writer& w, const char* name, uint32_t size, uint32_t want_crc, StatusFn status) {
    char path[96], tmp[100];
    std::snprintf(path, sizeof(path), "%s/%s", SdCard::kMountPoint, name);
    std::snprintf(tmp, sizeof(tmp), "%s.tmp", path);

    if (const char* err = receive_to(w, tmp, name, size, want_crc, status)) return err;

    // Only rename closed files. FATFS rename doesn't overwrite.
    struct stat st;
    if (stat(path, &st) == 0) unlink(path);
    if (rename(tmp, path) != 0) return "rename failed";
    return verify(path, size, want_crc, w.block[0], status);
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

    // Only allocated once the host is here: the model needs this RAM later.
    Writer w;
    bool ok = true;
    for (auto& b : w.block) {
        b = static_cast<uint8_t*>(heap_caps_malloc(kBlockSize, MALLOC_CAP_DMA | MALLOC_CAP_8BIT));
        ok = ok && b != nullptr;
    }
    w.free_q = xQueueCreate(kBlocks, sizeof(int));
    w.full_q = xQueueCreate(kBlocks + 1, sizeof(int));
    w.stopped = xSemaphoreCreateBinary();
    ok = ok && w.free_q && w.full_q && w.stopped;

    int files = 0;
    if (!ok) {
        send("UP:ERR out of memory\n");
    } else {
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
            const char* err = receive_file(w, name, static_cast<uint32_t>(size), static_cast<uint32_t>(crc), status);
            if (err == nullptr) {
                ++files;
                send("UP:OK\n");
            } else {
                std::snprintf(line, sizeof(line), "UP:ERR %s\n", err);
                send(line);
                status(err);
            }
        }
    }

    for (auto* b : w.block) heap_caps_free(b);
    if (w.free_q) vQueueDelete(w.free_q);
    if (w.full_q) vQueueDelete(w.full_q);
    if (w.stopped) vSemaphoreDelete(w.stopped);
    return files;
}

}  // namespace upload
