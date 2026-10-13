// Decode speed of lfm::z_decode on the device, on real chunks
// (/sdcard/ztest.bin from tools/zcodec.py test). Enabled with kZTest.

#include <cstdio>
#include <cstring>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "lfm/zcodec.hpp"

namespace {

constexpr const char* TAG = "ztest";

struct Job {
    const lfm::ZTable* q;
    const lfm::ZTable* s;
    const uint8_t* comp;
    uint32_t comp_len;
    uint8_t* out;
    uint32_t raw_len;
    uint32_t phase;
    bool ok;
    int64_t us;
    TaskHandle_t notify;
};

void decode_job(Job* j) {
    const int64_t t0 = esp_timer_get_time();
    j->ok = lfm::z_decode(*j->q, *j->s, j->comp, j->comp_len, j->out, j->raw_len, j->phase);
    j->us = esp_timer_get_time() - t0;
}

void core1_task(void* arg) {
    auto* j = static_cast<Job*>(arg);
    decode_job(j);
    xTaskNotifyGive(j->notify);
    vTaskDelete(nullptr);
}

}  // namespace

void z_test(void (*print)(const char*)) {
    char line[96];
    FILE* f = std::fopen("/sdcard/ztest.bin", "rb");
    if (f == nullptr) {
        print("no /ztest.bin (uv run tools/zcodec.py test)");
        return;
    }
    char magic[8];
    uint32_t hdr[2];
    std::fread(magic, 1, 8, f);
    std::fread(hdr, 4, 2, f);
    static lfm::ZTable q, s;
    std::fread(q.e, 1, sizeof(q.e), f);
    std::fread(s.e, 1, sizeof(s.e), f);
    const uint32_t chunk = hdr[1];
    auto* raw = static_cast<uint8_t*>(heap_caps_malloc(chunk, MALLOC_CAP_8BIT));
    auto* comp = static_cast<uint8_t*>(heap_caps_malloc(chunk + 16, MALLOC_CAP_8BIT));
    auto* out = static_cast<uint8_t*>(heap_caps_malloc(chunk, MALLOC_CAP_8BIT));
    int64_t us[2] = {0, 0};
    uint64_t bytes = 0, comp_bytes = 0;
    int bad = 0;
    for (uint32_t i = 0; i < hdr[0]; ++i) {
        uint32_t c[3];
        std::fread(c, 4, 3, f);
        std::fread(raw, 1, c[0], f);
        std::fread(comp, 1, c[2] + ((4 - c[2] % 4) % 4), f);
        for (int core = 0; core < 2; ++core) {
            Job j{&q, &s, comp, c[2], out, c[0], c[1], false, 0, xTaskGetCurrentTaskHandle()};
            std::memset(out, 0, c[0]);
            if (core == 0) {
                decode_job(&j);
            } else {
                xTaskCreatePinnedToCore(core1_task, "zdec", 4096, &j, 5, nullptr, 1);
                ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
            }
            us[core] += j.us;
            if (!j.ok || std::memcmp(out, raw, c[0]) != 0) ++bad;
        }
        bytes += c[0];
        comp_bytes += c[2];
    }
    std::fclose(f);
    for (int core = 0; core < 2; ++core) {
        std::snprintf(line, sizeof(line), "z_decode core %d: %.1f MB/s out, %.0f us per 32 KB, %d bad", core,
                      bytes / (us[core] / 1e6) / 1e6, us[core] / (bytes / 32768.0), bad);
        print(line);
        ESP_LOGI(TAG, "%s", line);
    }
    std::snprintf(line, sizeof(line), "%.2f MB -> %.2f MB (%.1f%% smaller)", bytes / 1e6, comp_bytes / 1e6,
                  100.0 * (1.0 - static_cast<double>(comp_bytes) / bytes));
    print(line);
    ESP_LOGI(TAG, "%s", line);
    heap_caps_free(raw);
    heap_caps_free(comp);
    heap_caps_free(out);
}
