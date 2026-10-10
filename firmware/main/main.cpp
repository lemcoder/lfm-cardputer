#include <M5Unified.hpp>

#include "esp_log.h"
#include "esp_timer.h"
#include "sd_bench.hpp"

#include "lfm.hpp"

static const char *TAG = "lfm";

// Append one decoded token's text to the screen. Text wraps and scrolls.
[[maybe_unused]] static void show_token(const char *text)
{
    M5.Display.print(text);
}

extern "C" void app_main(void)
{
    M5.begin();
    M5.Display.setRotation(1);  // landscape, 240x135
    M5.Display.setTextScroll(true);
    M5.Display.setTextWrap(true, true);
    M5.Display.setFont(&fonts::Font0);
    M5.Display.setTextSize(1);
    M5.Display.fillScreen(TFT_BLACK);

    M5.Display.setTextColor(TFT_CYAN, TFT_BLACK);
    M5.Display.printf("LFM2.5-230M  d=%d L=%d V=%d\n\n", lfm::kDim, lfm::kNumLayers, lfm::kVocab);
    M5.Display.setTextColor(TFT_WHITE, TFT_BLACK);

    M5.Display.println("SD read benchmark...");
    auto results = sdbench::run();

    M5.Display.println(" MHz  buf   file   raw  MB/s");
    for (const auto& r : results) {
        if (r.megabytes_per_sec == 0.0f) {
            M5.Display.printf(" %3d %3uK  failed\n", r.frequency_khz / 1000, (unsigned)(r.buffer_size / 1024));
            continue;
        }
        M5.Display.printf(" %3d %3uK %6.2f %5.2f\n", r.frequency_khz / 1000, (unsigned)(r.buffer_size / 1024),
                          r.megabytes_per_sec, r.raw_megabytes_per_sec);
    }

    ESP_LOGI(TAG, "free heap: %u, largest block: %u",
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_8BIT),
             (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));

    // TODO: generation loop goes here. show_token() appends text to the screen.
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
