#include "sd_speed.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_rom_crc.h"
#include "esp_timer.h"
#include "sd_card.hpp"

namespace {

constexpr const char* TAG = "sdspeed";
constexpr std::size_t kChunk = 32 * 1024;
constexpr std::size_t kBytes = 16 * 1024 * 1024;

struct Case {
    const char* name;
    SdCard::Config cfg;
};

constexpr Case kCases[] = {
    {"53M  p1      ", {3, 1, -1}},  // current setting: reference CRC
    {"80M  p0      ", {2, 0, -1}},
    {"80M  p1      ", {2, 1, -1}},
    {"80M  p2      ", {2, 2, -1}},
    {"80M  p3      ", {2, 3, -1}},
    {"53M  p1 again", {3, 1, -1}},  // card still fine afterwards?
};

}  // namespace

void sd_speed_test(PrintFn print) {
    auto* buf = static_cast<uint8_t*>(heap_caps_malloc(kChunk, MALLOC_CAP_DMA | MALLOC_CAP_8BIT));
    char line[96];
    uint32_t ref_crc = 0;
    bool have_ref = false;

    for (const Case& c : kCases) {
        SdCard card(c.cfg);
        if (!card.ok()) {
            std::snprintf(line, sizeof(line), "%s mount fail %s", c.name, esp_err_to_name(card.err()));
            print(line);
            ESP_LOGW(TAG, "%s", line);
            continue;
        }
        const int fd = open("/sdcard/lfm.lfm", O_RDONLY);
        if (fd < 0) {
            print("no /lfm.lfm");
            break;
        }
        uint32_t crc = 0;
        std::size_t done = 0;
        const int64_t t0 = esp_timer_get_time();
        while (done < kBytes) {
            const ssize_t got = read(fd, buf, kChunk);
            if (got <= 0) break;
            crc = esp_rom_crc32_le(crc, buf, static_cast<uint32_t>(got));
            done += static_cast<std::size_t>(got);
        }
        const double s = (esp_timer_get_time() - t0) / 1e6;
        close(fd);

        const char* verdict;
        if (done < kBytes) {
            verdict = "READ ERR";
        } else if (!have_ref) {
            ref_crc = crc;
            have_ref = true;
            verdict = "ref";
        } else {
            verdict = crc == ref_crc ? "crc ok" : "CRC BAD";
        }
        std::snprintf(line, sizeof(line), "%s %5d kHz %5.2f MB/s %s", c.name, card.freq_khz(), done / 1e6 / s, verdict);
        print(line);
        ESP_LOGI(TAG, "%s", line);
    }
    heap_caps_free(buf);
}
