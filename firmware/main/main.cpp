#include <M5Unified.hpp>

#include "esp_log.h"
#include "esp_timer.h"
#include "sd_bench.hpp"

#include "lfm.hpp"

static const char *TAG = "lfm";

// Append one decoded token's text to the screen. Text wraps and scrolls.
static void show_token(const char *text)
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

    ESP_LOGI(TAG, "free heap: %u, largest block: %u",
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_8BIT),
             (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));

    // Placeholder generation loop: replace with the real decode step.
    static const char *fake[] = {"Hello", ",", " world", "!", " I", " run", " on", " a", " Cardputer", ".\n"};
    for (int i = 0;; i = (i + 1) % 10) {
        int64_t t0 = esp_timer_get_time();
        // TODO: token = lfm_decode_step(...); text = tokenizer_decode(token);
        vTaskDelay(pdMS_TO_TICKS(300));
        show_token(fake[i]);
        ESP_LOGI(TAG, "token %d: %.1f ms", i, (esp_timer_get_time() - t0) / 1000.0);
    }
}
