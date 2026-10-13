#include "wifi_test.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>

#include "async_file.hpp"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_rom_crc.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "lwip/sockets.h"
#include "net.hpp"

namespace {

constexpr const char* TAG = "wifitest";
constexpr std::size_t kChunk = 32 * 1024;

unsigned heap_free() { return static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_8BIT)); }
// Lowest free heap since heap_caps_monitor_local_minimum_free_size_start().
unsigned heap_min() { return static_cast<unsigned>(heap_caps_get_minimum_free_size(MALLOC_CAP_8BIT)); }

bool send_request(int fd, uint64_t offset, uint32_t len) {
    uint8_t req[12];
    std::memcpy(req, &offset, 8);
    std::memcpy(req + 8, &len, 4);
    return send(fd, req, sizeof(req), 0) == static_cast<ssize_t>(sizeof(req));
}

bool recv_exact(int fd, uint8_t* dst, std::size_t n) {
    while (n > 0) {
        const ssize_t got = recv(fd, dst, n, 0);
        if (got <= 0) return false;
        dst += got;
        n -= static_cast<std::size_t>(got);
    }
    return true;
}

// `total` bytes from `offset` in kChunk requests, `depth` of them in flight.
// Returns MB/s, or -1 on error. CRC of the data in *crc when given.
double fetch(int fd, uint64_t offset, std::size_t total, int depth, uint8_t* buf, uint32_t* crc) {
    const std::size_t n = total / kChunk;
    std::size_t sent = 0;
    const int64_t t0 = esp_timer_get_time();
    for (; sent < n && sent < static_cast<std::size_t>(depth); ++sent) {
        if (!send_request(fd, offset + sent * kChunk, kChunk)) return -1;
    }
    for (std::size_t got = 0; got < n; ++got) {
        if (!recv_exact(fd, buf, kChunk)) return -1;
        if (sent < n) {
            if (!send_request(fd, offset + sent * kChunk, kChunk)) return -1;
            ++sent;
        }
        if (crc != nullptr) *crc = esp_rom_crc32_le(*crc, buf, kChunk);
    }
    return total / 1e6 / ((esp_timer_get_time() - t0) / 1e6);
}

struct NetJob {
    int fd;
    uint8_t* buf;
    std::size_t total;
    double mbps;
    TaskHandle_t notify;
};

void net_task(void* arg) {
    auto* job = static_cast<NetJob*>(arg);
    job->mbps = fetch(job->fd, 0, job->total, 2, job->buf, nullptr);
    xTaskNotifyGive(job->notify);
    vTaskDelete(nullptr);
}

struct SdJob {
    AsyncFile* src;
    uint8_t* buf;
    volatile bool stop;
    std::size_t bytes;
    TaskHandle_t notify;
};

void sd_task(void* arg) {
    auto* job = static_cast<SdJob*>(arg);
    for (uint64_t off = 64u << 20; !job->stop; off = off + kChunk < (128u << 20) ? off + kChunk : 64u << 20) {
        job->src->start_read(off, job->buf, kChunk);
        job->src->wait();
        job->bytes += kChunk;
    }
    xTaskNotifyGive(job->notify);
    vTaskDelete(nullptr);
}

}  // namespace

void wifi_test(PrintFn print) {
    char line[96];
    auto say = [&](const char* fmt, auto... args) {
        std::snprintf(line, sizeof(line), fmt, args...);
        print(line);
        ESP_LOGI(TAG, "%s", line);
    };

    WifiConfig cfg;
    if (!read_wifi_config("/sdcard/wifi.txt", &cfg)) {
        say("no /sdcard/wifi.txt (ssid, password, host, port)");
        return;
    }
    // Everything the test itself needs, before Wi-Fi: the drop in free heap
    // from here on is the Wi-Fi stack's (plus 8 KB of test task stacks).
    auto* buf = static_cast<uint8_t*>(heap_caps_malloc(kChunk, MALLOC_CAP_8BIT));
    auto* sd_buf = static_cast<uint8_t*>(heap_caps_malloc(kChunk, MALLOC_CAP_DMA | MALLOC_CAP_8BIT));
    AsyncFile src("/sdcard/lfm.lfm");
    const unsigned before = heap_free();
    heap_caps_monitor_local_minimum_free_size_start();
    say("heap before wifi: %u (rx static %d dyn %d, wnd %d)", before, CONFIG_ESP_WIFI_STATIC_RX_BUFFER_NUM,
        CONFIG_ESP_WIFI_DYNAMIC_RX_BUFFER_NUM, CONFIG_LWIP_TCP_WND_DEFAULT);
    const int64_t tc = esp_timer_get_time();
    if (!wifi_connect(cfg, 15000)) {
        say("wifi: no IP address (check SSID / password order in wifi.txt)");
        return;
    }
    wifi_ap_record_t ap;
    const int rssi = esp_wifi_sta_get_ap_info(&ap) == ESP_OK ? ap.rssi : 0;
    say("wifi up %.1f s, rssi %d, heap %u (-%u)", (esp_timer_get_time() - tc) / 1e6, rssi, heap_free(), before - heap_free());

    const int fd = net_connect(cfg);
    if (fd < 0) {
        say("no server at %s:%u", cfg.host, cfg.port);
        return;
    }
    // Correctness: the first 1 MB over Wi-Fi and from the card.
    {
        uint32_t net_crc = 0, sd_crc = 0;
        fetch(fd, 0, 1 << 20, 2, buf, &net_crc);
        for (std::size_t off = 0; off < (1u << 20); off += kChunk) {
            src.read_at(off, sd_buf, kChunk);
            sd_crc = esp_rom_crc32_le(sd_crc, sd_buf, kChunk);
        }
        say("crc net %08lx sd %08lx %s", (unsigned long)net_crc, (unsigned long)sd_crc, net_crc == sd_crc ? "OK" : "MISMATCH");
    }

    for (int depth : {1, 2, 4}) {
        const double mbps = fetch(fd, 0, 8u << 20, depth, buf, nullptr);
        say("net alone, depth %d: %.2f MB/s", depth, mbps);
    }

    // The card alone, same loop, for comparison.
    {
        const int64_t t0 = esp_timer_get_time();
        for (uint64_t off = 64u << 20; off < (80u << 20); off += kChunk) {
            src.start_read(off, sd_buf, kChunk);
            src.wait();
        }
        say("sd alone: %.2f MB/s", 16.0 * 1.048576 / ((esp_timer_get_time() - t0) / 1e6));
    }

    // Both at once: Wi-Fi on a core 0 task, the card from a core 1 task (as
    // the engine's reader would), so neither waits on the other's scheduling.
    {
        SdJob sd{&src, sd_buf, false, 0, xTaskGetCurrentTaskHandle()};
        NetJob job{fd, buf, 16u << 20, 0, xTaskGetCurrentTaskHandle()};
        const int64_t t0 = esp_timer_get_time();
        xTaskCreatePinnedToCore(sd_task, "sdjob", 4096, &sd, 10, nullptr, 1);
        xTaskCreatePinnedToCore(net_task, "net", 4096, &job, 5, nullptr, 0);
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);  // net done
        sd.stop = true;
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);  // sd done
        const double s = (esp_timer_get_time() - t0) / 1e6;
        const double sdr = sd.bytes / 1e6 / s;
        say("both: net %.2f + sd %.2f = %.2f MB/s", job.mbps, sdr, job.mbps + sdr);
        say("wifi heap: %u connected, %u at peak", before - heap_free(), before - heap_min());
    }
    close(fd);
    heap_caps_free(buf);
    heap_caps_free(sd_buf);
}
