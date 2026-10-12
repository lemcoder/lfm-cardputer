// LFM2.5-230M on the Cardputer-Adv: weights streamed from the SD card.
//
// SD card contents (from tools/export.py):
//   /lfm.lfm       out/lfm-q8.lfm
//   /prompt.txt    out/prompt.txt  (token ids; optional, defaults below)
//
// Screen: prompt + generated text, status line with tokens, s/token and a
// per-layer progress bar. Press G0 (BtnA) to run again.

#include <M5Unified.hpp>

#include <cstdio>
#include <cstdlib>
#include <memory>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "lfm.hpp"
#include "lfm/engine.hpp"
#include "sd_card.hpp"

namespace {

constexpr const char* TAG = "lfm";
constexpr const char* kModelPath = "/sdcard/lfm.lfm";
constexpr const char* kPromptPath = "/sdcard/prompt.txt";
constexpr int32_t kDefaultPrompt[] = {1, 1098, 5706, 803, 4481, 856};  // "The capital of France is"
constexpr int kMaxPrompt = 64;

// Layout (240 x 135, Font0 is 6x8).
constexpr int kW = 240, kH = 135;
constexpr int kTitleH = 10;
constexpr int kStatusH = 12;
constexpr int kTextY = kTitleH + 2;
constexpr int kTextH = kH - kTextY - kStatusH - 1;

auto& lcd = M5.Display;

// ---------------------------------------------------------------- screen

struct Status {
    int tokens = 0;        // generated so far
    float sec_per_tok = 0;
    const char* phase = "";
};
Status g_status;

void draw_status_text() {
    // Outside the scroll region: turn scrolling off so this line can't scroll the text.
    lcd.setTextScroll(false);
    lcd.clearScrollRect();
    lcd.fillRect(0, kH - kStatusH, kW, kStatusH - 3, TFT_BLACK);
    lcd.setTextColor(TFT_DARKGREY, TFT_BLACK);
    char line[48];
    if (g_status.tokens > 0) {
        std::snprintf(line, sizeof(line), "%s  %d tok  %.1f s/tok", g_status.phase, g_status.tokens, g_status.sec_per_tok);
    } else {
        std::snprintf(line, sizeof(line), "%s", g_status.phase);
    }
    lcd.drawString(line, 0, kH - kStatusH);
    lcd.setScrollRect(0, kTextY, kW, kTextH);
    lcd.setTextScroll(true);
}

// Engine progress hook: thin bar under the status text. Also yields so the
// idle task (and its watchdog) gets to run between layers.
void on_progress(void*, int done, int total) {
    const int w = kW * done / total;
    lcd.fillRect(0, kH - 3, w, 3, TFT_DARKGREEN);
    lcd.fillRect(w, kH - 3, kW - w, 3, TFT_BLACK);
    vTaskDelay(1);
}

// Text goes into the scrolling region; the title and status line stay put.
void text_begin() {
    lcd.setScrollRect(0, kTextY, kW, kTextH);
    lcd.fillRect(0, kTextY, kW, kTextH, TFT_BLACK);
    lcd.setCursor(0, kTextY);
}

void show_token(lfm::Engine& eng, int32_t id, uint16_t color) {
    char buf[64];
    const std::size_t n = eng.token_text(id, buf, sizeof(buf) - 1);
    const bool special = n > 4 && buf[0] == '<' && buf[1] == '|';
    for (std::size_t i = 0; i < n; ++i) {
        if (static_cast<unsigned char>(buf[i]) >= 0x80) buf[i] = '?';  // Font0 is ASCII only
    }
    buf[n] = '\0';

    const int x = lcd.getCursorX(), y = lcd.getCursorY();
    lcd.setScrollRect(0, kTextY, kW, kTextH);
    lcd.setCursor(x, y);
    lcd.setTextColor(special ? TFT_DARKGREY : color, TFT_BLACK);
    lcd.print(buf);
    const int nx = lcd.getCursorX(), ny = lcd.getCursorY();
    draw_status_text();
    lcd.setCursor(nx, ny);
}

void set_phase(const char* phase) {
    const int x = lcd.getCursorX(), y = lcd.getCursorY();
    g_status.phase = phase;
    draw_status_text();
    lcd.setCursor(x, y);
}

[[noreturn]] void fatal(const char* what, const char* detail) {
    ESP_LOGE(TAG, "%s: %s", what, detail);
    lcd.setTextColor(TFT_RED, TFT_BLACK);
    lcd.printf("\n%s\n", what);
    lcd.setTextColor(TFT_WHITE, TFT_BLACK);
    lcd.printf("%s\n", detail);
    while (true) vTaskDelay(pdMS_TO_TICKS(1000));
}

// ---------------------------------------------------------------- setup

int read_prompt(int32_t* out, int cap) {
    FILE* f = std::fopen(kPromptPath, "r");
    int n = 0;
    if (f != nullptr) {
        long id;
        while (n < cap && std::fscanf(f, "%ld", &id) == 1) out[n++] = static_cast<int32_t>(id);
        std::fclose(f);
    }
    if (n == 0) {
        for (int32_t id : kDefaultPrompt) out[n++] = id;
    }
    return n;
}

// Largest context that fits, keeping `reserve` bytes free for everything else.
std::unique_ptr<lfm::Engine> open_engine(lfm::Source& src, lfm::EngineConfig cfg, std::size_t reserve, lfm::Status* st) {
    void* hold = std::malloc(reserve);
    std::unique_ptr<lfm::Engine> eng;
    for (int ctx = 64; ctx >= 4; ctx -= 2) {
        cfg.max_ctx = ctx;
        eng = std::make_unique<lfm::Engine>();
        *st = eng->open(src, cfg);
        if (*st != lfm::Status::OutOfMemory) break;
    }
    std::free(hold);
    return eng;
}

}  // namespace

extern "C" void app_main(void) {
    M5.begin();
    lcd.setRotation(1);
    lcd.setFont(&fonts::Font0);
    lcd.setTextSize(1);
    lcd.setTextWrap(true, true);
    lcd.setTextScroll(true);
    lcd.fillScreen(TFT_BLACK);
    lcd.setTextColor(TFT_CYAN, TFT_BLACK);
    lcd.drawString("LFM2.5-230M  Q8  weights from SD", 0, 0);
    lcd.drawFastHLine(0, kTitleH, kW, TFT_DARKGREY);
    text_begin();
    lcd.setTextColor(TFT_WHITE, TFT_BLACK);

    set_phase("mounting SD");
    static SdCard card(40'000);
    if (!card.ok()) fatal("SD mount failed", esp_err_to_name(card.err()));

    static lfm::FileSource src(kModelPath);
    if (!src.ok()) fatal("missing /lfm.lfm", "copy out/lfm-q8.lfm to the SD card as lfm.lfm");

    const lfm::EngineConfig base{.kv_bf16 = true, .io_bytes = 32 * 1024};
    lfm::Status st;
    std::unique_ptr<lfm::Engine> eng = open_engine(src, base, 24 * 1024, &st);
    if (st != lfm::Status::Ok) fatal("engine open failed", lfm::to_string(st));
    eng->set_progress_hook(&on_progress, nullptr);
    ESP_LOGI(TAG, "engine: ctx %d, %u KB; heap free %u, largest %u", eng->max_ctx(),
             (unsigned)(eng->ram_bytes({.max_ctx = eng->max_ctx(), .kv_bf16 = true, .io_bytes = base.io_bytes}) / 1024),
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_8BIT),
             (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));

    static int32_t prompt[kMaxPrompt];
    const int n_prompt = read_prompt(prompt, kMaxPrompt);

    while (true) {
        eng->reset();
        g_status = {};
        text_begin();
        for (int i = 0; i < n_prompt; ++i) show_token(*eng, prompt[i], TFT_LIGHTGREY);

        // Prefill: every prompt token goes through the layers; only the last
        // one needs the LM head.
        int64_t t0 = esp_timer_get_time();
        int32_t next = lfm::Engine::kNoToken;
        char phase[32];
        for (int i = 0; i < n_prompt && i < eng->max_ctx(); ++i) {
            std::snprintf(phase, sizeof(phase), "prefill %d/%d", i + 1, n_prompt);
            set_phase(phase);
            next = eng->forward(prompt[i], i + 1 == n_prompt);
        }
        ESP_LOGI(TAG, "prefill %d tokens: %.1f s", n_prompt, (esp_timer_get_time() - t0) / 1e6);

        // Generate until EOS or the context is full.
        set_phase("generating");
        int64_t t_gen = esp_timer_get_time();
        while (next >= 0) {
            show_token(*eng, next, TFT_WHITE);
            ++g_status.tokens;
            if (next == lfm::kEosId) break;
            next = eng->forward(next, true);
            g_status.sec_per_tok = (esp_timer_get_time() - t_gen) / 1e6f / g_status.tokens;
            ESP_LOGI(TAG, "token %d: %.1f s/tok avg", g_status.tokens, g_status.sec_per_tok);
        }
        if (next == lfm::Engine::kIoError) fatal("SD read error", "check the card");
        set_phase(next == lfm::Engine::kContextFull ? "context full  G0: again" : "done  G0: again");
        ESP_LOGI(TAG, "done: %d tokens, %.1f s total, read %.0f MB", g_status.tokens,
                 (esp_timer_get_time() - t0) / 1e6, src.bytes_read() / 1e6);

        do {
            vTaskDelay(pdMS_TO_TICKS(20));
            M5.update();
        } while (!M5.BtnA.wasPressed());
    }
}
