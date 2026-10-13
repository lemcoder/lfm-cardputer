// LFM2.5-230M on the Cardputer-Adv: weights streamed from the SD card.
//
// SD card contents (from tools/export.py):
//   /lfm.lfm       out/lfm-q8.lfm
//   /prompt.txt    out/prompt.txt  (token ids; optional, defaults below)
//   /kv.bin        created at boot: the KV cache lives on the card, not in RAM
//   /conv.bin      created at boot: conv state history for speculative rollback
//   /cstate.bin    created at boot: the conv layers' state (frees 64 KB of RAM)
//
// Flash partition "weights": the n-gram draft table (tools/flash_ngram.sh).
// Without it, speculative decoding still runs with prompt-lookup drafts.
//
// No card reader? `uv run tools/upload.py` sends the files over the USB cable
// (the firmware listens for it for 1.5 s after boot, or until it arrives if
// /lfm.lfm is missing).
//
// Screen: prompt + generated text, status line with tokens, s/token and a
// per-layer progress bar. Press G0 (BtnA) to run again.

#include <M5Unified.hpp>

#include <cstdio>
#include <cstdlib>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_partition.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "lfm.hpp"
#include "lfm/draft.hpp"
#include "lfm/engine.hpp"
#include "async_file.hpp"
#include "sd_card.hpp"
#include "sd_speed.hpp"
#include "kernel_test.hpp"
#include "usb_upload.hpp"

namespace {

constexpr const char* TAG = "lfm";
constexpr const char* kModelPath = "/sdcard/lfm.lfm";
constexpr const char* kPromptPath = "/sdcard/prompt.txt";
constexpr const char* kKvPath = "/sdcard/kv.bin";
constexpr const char* kConvHistPath = "/sdcard/conv.bin";
constexpr const char* kConvStatePath = "/sdcard/cstate.bin";
constexpr int kMaxCtx = 1024;          // KV file grows 12 KB per token: 12 MB at most
constexpr bool kSdSpeedTest = false;   // run the SD clock experiment instead of the model
constexpr bool kKernelTest = false;    // check the SIMD kernels against the C reference instead
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
    char phase[40] = "";   // copied: callers pass temporary strings
};
Status g_status;

void draw_status_text() {
    // Outside the scroll region: turn scrolling off so this line can't scroll the text.
    lcd.setTextScroll(false);
    lcd.clearScrollRect();
    lcd.fillRect(0, kH - kStatusH, kW, kStatusH - 3, TFT_BLACK);
    lcd.setTextColor(TFT_DARKGREY, TFT_BLACK);
    char line[96];  // wider than the screen; drawString clips
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
    std::snprintf(g_status.phase, sizeof(g_status.phase), "%s", phase);
    draw_status_text();
    lcd.setCursor(x, y);
}

long st_size(const char* path) {
    struct stat st;
    return stat(path, &st) == 0 ? static_cast<long>(st.st_size) : -1;
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

    if (kKernelTest) {
        set_phase("SIMD kernel test");
        vTaskDelay(pdMS_TO_TICKS(3000));  // time to attach a serial log after flashing
        const bool ok = kernel_test([](const char* line) { lcd.println(line); });
        set_phase(ok ? "SIMD kernels OK" : "SIMD kernels MISMATCH");
        while (true) vTaskDelay(pdMS_TO_TICKS(1000));
    }
    if (kSdSpeedTest) {
        set_phase("SD clock test");
        vTaskDelay(pdMS_TO_TICKS(3000));  // time to attach a serial log after flashing
        vTaskDelay(pdMS_TO_TICKS(3000));  // time to attach a serial log after flashing
        sd_speed_test([](const char* line) { lcd.println(line); });
        set_phase("SD clock test done");
        while (true) vTaskDelay(pdMS_TO_TICKS(1000));
    }

    set_phase("mounting SD");
    // 80 MHz (160 / 2, overclock; SD high-speed spec is 50 MHz). Measured on
    // this card with sd_speed.cpp: CRC-clean at sampling phases 1-3, phase 0
    // fails; phase 2 is mid-window. 53.3 MHz (div 3) works at every phase.
    static SdCard card(SdCard::Config{.host_div = 2, .delay_phase = 2});
    if (!card.ok()) fatal("SD mount failed", esp_err_to_name(card.err()));

    struct stat st;
    const bool have_model = stat(kModelPath, &st) == 0;
    if (!have_model) {
        lcd.setTextColor(TFT_YELLOW, TFT_BLACK);
        lcd.println("No /lfm.lfm on the SD card.\nOn the Mac run:\n  uv run tools/upload.py\n");
        lcd.setTextColor(TFT_WHITE, TFT_BLACK);
    }
    set_phase(have_model ? "USB upload? (1.5 s)" : "waiting for USB upload");
    if (upload::serve(have_model ? 1500 : -1, &set_phase) > 0) text_begin();

    static AsyncFile src(kModelPath);  // reads on core 1 while core 0 computes
    if (!src.ok()) fatal("missing /lfm.lfm", "uv run tools/upload.py");

    // Biggest prompt batch that fits, keeping kReserve bytes for everything else.
    constexpr std::size_t kReserve = 24 * 1024;
    lfm::EngineConfig cfg{.max_ctx = kMaxCtx, .kv_bf16 = true, .io_bytes = 32 * 1024, .kv_path = kKvPath,
                          .int8_activations = true, .speculative = true, .conv_history_path = kConvHistPath,
                          .conv_state_path = kConvStatePath};
    static lfm::Engine* eng = nullptr;
    lfm::Status status = lfm::Status::OutOfMemory;
    for (int batch = 6; batch >= 1 && status == lfm::Status::OutOfMemory; --batch) {
        void* reserve = std::malloc(kReserve);
        cfg.max_batch = batch;
        delete eng;
        eng = new lfm::Engine();
        status = eng->open(src, cfg);
        std::free(reserve);
        ESP_LOGI(TAG, "batch %d: %s (largest free block %u)", batch, lfm::to_string(status),
                 (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
    }
    if (status != lfm::Status::Ok) {
        // Narrow it down in the log: header readable? KV file creatable?
        char magic[8] = {};
        const bool hdr = src.read_at(0, magic, sizeof(magic));
        errno = 0;
        const int fd = open(kKvPath, O_RDWR | O_CREAT | O_TRUNC, 0644);
        ESP_LOGE(TAG, "open: %s | header read %s, magic %.7s | kv open fd %d errno %d (%s) | size %ld",
                 lfm::to_string(status), hdr ? "ok" : "FAILED", magic, fd, errno, std::strerror(errno), (long)st_size(kModelPath));
        if (fd >= 0) close(fd);
        fatal("engine open failed", lfm::to_string(status));
    }
    eng->set_progress_hook(&on_progress, nullptr);
    ESP_LOGI(TAG, "engine: ctx %d, prompt batch %d, %u KB RAM; heap free %u, largest %u", eng->max_ctx(),
             cfg.max_batch, (unsigned)(eng->ram_bytes(cfg) / 1024), (unsigned)heap_caps_get_free_size(MALLOC_CAP_8BIT),
             (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));

    static int32_t prompt[kMaxPrompt];
    const int n_prompt = read_prompt(prompt, kMaxPrompt);

    // Speculative decoding drafts: prompt lookup over the context, else the
    // n-gram table memory-mapped from flash (if flashed).
    static int32_t draft_hist[kMaxCtx];
    static lfm::Drafter drafter(draft_hist, kMaxCtx);
    if (const esp_partition_t* part = esp_partition_find_first(
            ESP_PARTITION_TYPE_DATA, static_cast<esp_partition_subtype_t>(0x40), "weights")) {
        const void* table = nullptr;
        esp_partition_mmap_handle_t handle;
        if (esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA, &table, &handle) == ESP_OK) {
            const bool ok = drafter.set_ngram(table, part->size);
            ESP_LOGI(TAG, "n-gram draft table: %s", ok ? "loaded from flash" : "not flashed (lookup only)");
        }
    }
    const int max_draft = cfg.max_batch - 1;

    while (true) {
        eng->reset();
        g_status = {};
        text_begin();
        for (int i = 0; i < n_prompt; ++i) show_token(*eng, prompt[i], TFT_LIGHTGREY);

        // Prefill: the prompt goes through the layers max_batch tokens at a
        // time (one weight read per batch); only the last needs the LM head.
        int64_t t0 = esp_timer_get_time();
        int32_t next = lfm::Engine::kNoToken;
        char phase[32];
        for (int i = 0; i < n_prompt && next >= lfm::Engine::kNoToken;) {
            const int n = n_prompt - i < cfg.max_batch ? n_prompt - i : cfg.max_batch;
            std::snprintf(phase, sizeof(phase), "prefill %d-%d/%d", i + 1, i + n, n_prompt);
            set_phase(phase);
            next = eng->prefill({prompt + i, static_cast<std::size_t>(n)}, i + n == n_prompt);
            i += n;
        }
        ESP_LOGI(TAG, "prefill %d tokens: %.1f s", n_prompt, (esp_timer_get_time() - t0) / 1e6);

        // Generate until EOS or the context is full: speculative steps, each
        // one pass over the weights for the newest token plus its drafts.
        set_phase("generating");
        drafter.reset();
        for (int i = 0; i < n_prompt; ++i) drafter.push(prompt[i]);
        lfm::SpecStats spec;
        int64_t t_gen = esp_timer_get_time();
        int32_t fed = next;
        if (fed >= 0) {
            show_token(*eng, fed, TFT_WHITE);
            ++g_status.tokens;
            drafter.push(fed);
        }
        while (fed >= 0 && fed != lfm::kEosId) {
            int32_t produced[lfm::Engine::kMaxBatch];
            const int got = lfm::speculative_step(*eng, drafter, fed, max_draft, produced, &spec);
            if (got < 0) {
                next = got;
                break;
            }
            for (int i = 0; i < got; ++i) {
                show_token(*eng, produced[i], i + 1 < got ? TFT_GREENYELLOW : TFT_WHITE);  // accepted drafts tinted
                ++g_status.tokens;
                fed = produced[i];
                if (fed == lfm::kEosId) break;
            }
            g_status.sec_per_tok = (esp_timer_get_time() - t_gen) / 1e6f / (g_status.tokens - 1);
            const double total = (esp_timer_get_time() - t0) / 1e6;
            ESP_LOGI(TAG, "token %d: %.1f s/tok avg | pass %d: %.2f tok/pass, drafts %d/%d accepted | "
                     "so far %.1f s: SD busy %.0f%% (%.2f MB/s), compute stalled on SD %.0f%%",
                     g_status.tokens, g_status.sec_per_tok, spec.passes,
                     static_cast<double>(spec.tokens) / spec.passes, spec.accepted, spec.drafted, total,
                     100.0 * src.sd_busy_seconds() / total, src.bytes_read() / 1e6 / src.sd_busy_seconds(),
                     100.0 * src.stall_seconds() / total);
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
