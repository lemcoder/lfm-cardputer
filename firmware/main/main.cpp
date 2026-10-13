// LFM2.5-230M on the Cardputer-Adv: weights streamed from the SD card.
//
// SD card contents (from tools/export.py):
//   /lfm.lfm       out/lfm-q8.lfm
//   /prompt.txt    out/prompt.txt  (token ids; optional, defaults below)
//   /kv.bin        created at boot: the KV cache lives on the card, not in RAM
//   /conv.bin      created at boot: conv state history for speculative rollback
//   /cstate.bin    created at boot: the conv layers' state (frees 64 KB of RAM)
//   /cfg.txt       optional experiment knobs, key=value per line (RunConfig)
//
// Flash partitions (tools/flash_part.sh), both optional:
//   ngram   the speculative-decoding n-gram draft table (else lookup drafts only)
//   lmhead  the LM head shortlist table (else the .lfm file's LM head, from SD)
//
// No card reader? `uv run tools/upload.py` sends the files over the USB cable
// (the firmware listens for it for 1.5 s after boot, or until it arrives if
// /lfm.lfm is missing).
//
// Screen: prompt + generated text, status line with tokens, s/token and a
// per-layer progress bar. Press G0 (BtnA) to run again.

#include <M5Unified.hpp>

#include <algorithm>
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
#include "lfm/tokenizer.hpp"
#include "async_file.hpp"
#include "sd_card.hpp"
#include "sd_speed.hpp"
#include "kernel_test.hpp"
#include "keyboard.hpp"
#include "usb_upload.hpp"
#include "wifi_test.hpp"

namespace {

constexpr const char* TAG = "lfm";
constexpr const char* kModelPath = "/sdcard/lfm.lfm";
constexpr const char* kPromptPath = "/sdcard/prompt.txt";
constexpr const char* kKvPath = "/sdcard/kv.bin";
constexpr const char* kConvHistPath = "/sdcard/conv.bin";
constexpr const char* kConvStatePath = "/sdcard/cstate.bin";
constexpr const char* kRunConfigPath = "/sdcard/cfg.txt";
constexpr const char* kPrefixPath = "/sdcard/prefix.bin";
constexpr const char* kTokenizerPath = "/sdcard/tok.bin";  // tools/export_tokenizer.py
constexpr int kMaxCtx = 1024;          // KV file grows 12 KB per token: 12 MB at most
constexpr bool kSdSpeedTest = false;   // run the SD clock experiment instead of the model
constexpr bool kKernelTest = false;    // check the SIMD kernels against the C reference instead
constexpr bool kWifiTest = false;      // measure weight streaming over Wi-Fi instead (wifi_test.hpp)
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

// Generated tokens also go to the serial console as "TOK <id>" lines, for
// tools/chat.py (which decodes them with the real tokenizer).
void emit_token(int32_t id) {
    std::printf("TOK %ld\n", static_cast<long>(id));
    std::fflush(stdout);
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

// Experiment knobs, read from /sdcard/cfg.txt so a config change needs an
// upload (seconds) instead of a rebuild. tools/bench.py writes it.
struct RunConfig {
    int spec = 1;         // speculative decoding
    int max_draft = 2;    // drafts per pass (2 measured best; -1 = batch - 1)
    int max_batch = 8;    // largest prompt batch to try
    int io_kb = 32;       // weight streaming buffer (x2)
    int gen = 0;          // stop after this many generated tokens (0 = at EOS)
    int ignore_eos = 0;   // keep generating past EOS (with gen > 0)
    int progress = 1;     // LCD progress bar per layer
    int runs = 1;         // run the prompt this many times without waiting for G0
    int flash_head = 1;   // use the LM head table in flash (if flashed)
    int reserve_kb = 24;  // heap kept free when choosing the prompt batch
    int spec_prefill = 0; // drafts in the last prompt batch (no gain on B1/B2: experiment 20)
    int prompt_cache = 1; // reuse the prompt's state when a run repeats the previous prompt
    int chat = 1;         // keyboard chat (needs /tok.bin); 0 = run /prompt.txt (benchmarks)
    int max_reply = 256;  // chat: longest answer, tokens
};

RunConfig read_run_config() {
    RunConfig rc;
    FILE* f = std::fopen(kRunConfigPath, "r");
    if (f == nullptr) return rc;
    char key[32];
    int value;
    while (std::fscanf(f, " %31[^=]=%d", key, &value) == 2) {
        if (!std::strcmp(key, "spec")) rc.spec = value;
        else if (!std::strcmp(key, "max_draft")) rc.max_draft = value;
        else if (!std::strcmp(key, "max_batch")) rc.max_batch = value;
        else if (!std::strcmp(key, "io_kb")) rc.io_kb = value;
        else if (!std::strcmp(key, "gen")) rc.gen = value;
        else if (!std::strcmp(key, "ignore_eos")) rc.ignore_eos = value;
        else if (!std::strcmp(key, "progress")) rc.progress = value;
        else if (!std::strcmp(key, "runs")) rc.runs = value;
        else if (!std::strcmp(key, "flash_head")) rc.flash_head = value;
        else if (!std::strcmp(key, "reserve_kb")) rc.reserve_kb = value;
        else if (!std::strcmp(key, "spec_prefill")) rc.spec_prefill = value;
        else if (!std::strcmp(key, "prompt_cache")) rc.prompt_cache = value;
        else if (!std::strcmp(key, "chat")) rc.chat = value;
        else if (!std::strcmp(key, "max_reply")) rc.max_reply = value;
    }
    std::fclose(f);
    return rc;
}

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

// ---------------------------------------------------------------- chat

// Input line, drawn in the status bar while waiting for the user.
void draw_input(const char* text, std::size_t n) {
    lcd.setTextScroll(false);
    lcd.clearScrollRect();
    lcd.fillRect(0, kH - kStatusH, kW, kStatusH, TFT_BLACK);
    lcd.setTextColor(TFT_CYAN, TFT_BLACK);
    constexpr std::size_t kCols = kW / 6 - 3;  // "> " and the cursor
    const std::size_t from = n > kCols ? n - kCols : 0;  // keep the end visible
    char line[64];
    std::snprintf(line, sizeof(line), "> %.*s_", static_cast<int>(n - from), text + from);
    lcd.drawString(line, 0, kH - kStatusH);
    lcd.setScrollRect(0, kTextY, kW, kTextH);
    lcd.setTextScroll(true);
}

// Chat with the ChatML template the model was trained on:
//   <|startoftext|><|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n{reply}<|im_end|>\n<|im_start|>user\n...
// Turns continue in the same KV cache: each one only prefills the new part.
// The reply's closing <|im_end|> is produced but not run through the model,
// so the next turn starts with it. fn+` (escape) starts a new chat.
struct ChatContext {
    lfm::Engine& eng;
    lfm::Drafter& drafter;
    lfm::Tokenizer& tok;
    Keyboard& kbd;
    int max_batch;
    int max_draft;
    bool spec;
    int max_reply;
};

constexpr int32_t kBos = 1, kImStart = 6, kImEnd = 7;

void run_chat(ChatContext& c) {
    static char input[256];
    static int32_t turn[640];
    bool fresh = true;
    c.eng.reset();
    c.drafter.reset();
    text_begin();
    lcd.setTextColor(TFT_DARKGREY, TFT_BLACK);
    lcd.print("Type a message, Enter to send.\nfn+` starts a new chat.\n");

    while (true) {
        // Read a line.
        std::size_t n = 0;
        draw_input(input, n);
        while (true) {
            const char ch = c.kbd.poll();
            if (ch == 0) {
                vTaskDelay(pdMS_TO_TICKS(10));
                continue;
            }
            if (ch == 27) {  // new chat
                c.eng.reset();
                c.drafter.reset();
                fresh = true;
                n = 0;
                text_begin();
                lcd.setTextColor(TFT_DARKGREY, TFT_BLACK);
                lcd.print("New chat.\n");
            } else if (ch == '\b') {
                if (n > 0) --n;
            } else if (ch == '\n') {
                if (n > 0) break;
            } else if (n + 1 < sizeof(input)) {
                input[n++] = ch;
            }
            draw_input(input, n);
        }
        input[n] = '\0';
        const int64_t t0 = esp_timer_get_time();
        lcd.setTextColor(TFT_CYAN, TFT_BLACK);
        lcd.printf("%s> %s\n", lcd.getCursorX() > 0 ? "\n" : "", input);

        // Tokens of this turn (the template around the text, as HF tokenizes it:
        // special tokens by id, the text between them encoded per segment).
        static char seg[sizeof(input) + 8];
        int m = 0;
        auto add = [&](int32_t id) { if (m < static_cast<int>(sizeof(turn) / sizeof(turn[0]))) turn[m++] = id; };
        auto add_text = [&](const char* text) {
            const int got = c.tok.encode(text, turn + m, static_cast<int>(sizeof(turn) / sizeof(turn[0])) - m);
            if (got < 0) return false;
            m += got;
            return true;
        };
        if (fresh) {
            add(kBos);
        } else {
            add(kImEnd);
            add_text("\n");
        }
        add(kImStart);
        std::snprintf(seg, sizeof(seg), "user\n%s", input);
        const bool ok = add_text(seg);
        add(kImEnd);
        add_text("\n");
        add(kImStart);
        add_text("assistant\n");
        if (!ok) {
            set_phase("tokenizer error (tok.bin?)");
            continue;
        }
        if (c.eng.pos() + m + c.max_reply > c.eng.max_ctx()) {
            lcd.setTextColor(TFT_YELLOW, TFT_BLACK);
            lcd.print("Context full: fn+` for a new chat.\n");
            continue;
        }
        fresh = false;
        ESP_LOGI(TAG, "chat turn: %d tokens at position %d", m, c.eng.pos());

        // Prefill the turn, max_batch tokens per pass.
        g_status = {};
        int32_t next = lfm::Engine::kNoToken;
        char phase[40];
        for (int i = 0; i < m && next >= lfm::Engine::kNoToken;) {
            const int k = m - i < c.max_batch ? m - i : c.max_batch;
            std::snprintf(phase, sizeof(phase), "reading %d-%d/%d", i + 1, i + k, m);
            set_phase(phase);
            next = c.eng.prefill({turn + i, static_cast<std::size_t>(k)}, i + k == m);
            i += k;
        }
        for (int i = 0; i < m; ++i) c.drafter.push(turn[i]);
        if (next < 0) {
            if (next == lfm::Engine::kIoError) fatal("SD read error", "check the card");
            set_phase("prefill failed");
            continue;
        }
        const double prefill_s = (esp_timer_get_time() - t0) / 1e6;
        c.drafter.push(next);

        // Reply.
        set_phase("thinking");
        lcd.setTextColor(TFT_WHITE, TFT_BLACK);
        const int64_t t_gen = esp_timer_get_time();
        lfm::SpecStats spec;
        int32_t fed = next;
        auto show = [&](int32_t id) {
            ++g_status.tokens;
            if (id != kImEnd) show_token(c.eng, id, TFT_WHITE);
            emit_token(id);
        };
        show(fed);
        while (fed != kImEnd && g_status.tokens < c.max_reply) {
            int32_t produced[lfm::Engine::kMaxBatch];
            int got;
            if (c.spec) {
                got = lfm::speculative_step(c.eng, c.drafter, fed, c.max_draft, produced, &spec);
            } else {
                produced[0] = c.eng.forward(fed, true);
                got = produced[0] < 0 ? produced[0] : 1;
                if (got > 0) c.drafter.push(produced[0]);
            }
            if (got < 0) {
                if (got == lfm::Engine::kIoError) fatal("SD read error", "check the card");
                break;
            }
            for (int i = 0; i < got && fed != kImEnd; ++i) {
                fed = produced[i];
                show(fed);
            }
            g_status.sec_per_tok = (esp_timer_get_time() - t_gen) / 1e6f / g_status.tokens;
            set_phase("thinking");
        }
        lcd.print("\n");
        const double total_s = (esp_timer_get_time() - t0) / 1e6;
        ESP_LOGI(TAG, "chat reply: %d tokens, prefill %d in %.1f s, total %.1f s (%.1f s/tok), drafts %d/%d",
                 g_status.tokens, m, prefill_s, total_s, g_status.sec_per_tok, spec.accepted, spec.drafted);
        std::snprintf(phase, sizeof(phase), "%.0f s", total_s);
        set_phase(phase);
    }
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
        flash_scan_bench([](const char* line) { lcd.println(line); });
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

    if (kWifiTest) {
        set_phase("USB upload? (1.5 s)");
        upload::serve(1500, &set_phase);
        set_phase("Wi-Fi test");
        wifi_test([](const char* line) { lcd.println(line); });
        set_phase("Wi-Fi test done");
        while (true) vTaskDelay(pdMS_TO_TICKS(1000));
    }

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

    const RunConfig rc = read_run_config();
    ESP_LOGI(TAG, "run config: spec %d max_draft %d max_batch %d io_kb %d gen %d ignore_eos %d progress %d runs %d",
             rc.spec, rc.max_draft, rc.max_batch, rc.io_kb, rc.gen, rc.ignore_eos, rc.progress, rc.runs);

    // Biggest prompt batch that fits, keeping kReserve bytes for everything else.
    const std::size_t kReserve = static_cast<std::size_t>(rc.reserve_kb) * 1024;
    lfm::EngineConfig cfg{.max_ctx = kMaxCtx, .kv_bf16 = true,
                          .io_bytes = static_cast<std::size_t>(rc.io_kb) * 1024, .kv_path = kKvPath,
                          .int8_activations = true, .speculative = rc.spec != 0, .conv_history_path = kConvHistPath,
                          .conv_state_path = kConvStatePath, .prefix_path = kPrefixPath};
    static lfm::Engine* eng = nullptr;
    lfm::Status status = lfm::Status::OutOfMemory;
    for (int batch = std::min(rc.max_batch, static_cast<int>(lfm::Engine::kMaxBatch)); batch >= 1 && status == lfm::Status::OutOfMemory; --batch) {
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
    if (rc.progress) eng->set_progress_hook(&on_progress, nullptr);
    ESP_LOGI(TAG, "engine: ctx %d, prompt batch %d, %u KB RAM; heap free %u, largest %u", eng->max_ctx(),
             cfg.max_batch, (unsigned)(eng->ram_bytes(cfg) / 1024), (unsigned)heap_caps_get_free_size(MALLOC_CAP_8BIT),
             (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));

    static int32_t prompt[kMaxPrompt];
    const int n_prompt = read_prompt(prompt, kMaxPrompt);

    // Speculative decoding drafts: prompt lookup over the context, else the
    // n-gram table memory-mapped from flash (if flashed).
    static int32_t draft_hist[kMaxCtx];
    static lfm::Drafter drafter(draft_hist, kMaxCtx);
    auto map_partition = [](const char* name, std::size_t* size) -> const void* {
        const esp_partition_t* part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, name);
        const void* ptr = nullptr;
        esp_partition_mmap_handle_t handle;
        if (part == nullptr || esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA, &ptr, &handle) != ESP_OK) {
            return nullptr;
        }
        *size = part->size;
        return ptr;
    };
    std::size_t ngram_size = 0, head_size = 0;
    const void* ngram = map_partition("ngram", &ngram_size);
    const bool ngram_ok = drafter.set_ngram(ngram, ngram_size);
    ESP_LOGI(TAG, "n-gram draft table: %s", ngram_ok ? "loaded from flash" : "not flashed (lookup only)");
    const void* head = map_partition("lmhead", &head_size);
    const bool head_ok = rc.flash_head && eng->set_flash_head(head, head_size);
    ESP_LOGI(TAG, "LM head table: %s", head_ok ? "from flash" : "from the SD card");
    const int max_draft = rc.max_draft >= 0 ? rc.max_draft : cfg.max_batch - 1;

    if (rc.chat) {
        static lfm::FileSource tok_file(kTokenizerPath);
        static lfm::Tokenizer tok;
        static Keyboard kbd;
        const bool tok_ok = tok_file.ok() && tok.open(tok_file);
        const bool kbd_ok = kbd.begin();
        ESP_LOGI(TAG, "chat: tokenizer %s, keyboard %s", tok_ok ? "ok" : "missing", kbd_ok ? "ok" : "not found");
        if (tok_ok && kbd_ok) {
            ChatContext chat{*eng, drafter, tok, kbd, cfg.max_batch, max_draft, rc.spec != 0, rc.max_reply};
            run_chat(chat);
        }
        if (!tok_ok) {
            lcd.setTextColor(TFT_YELLOW, TFT_BLACK);
            lcd.println("No /tok.bin for chat. On the Mac:\n  uv run tools/export_tokenizer.py\n  uv run tools/upload.py out/tok.bin:tok.bin");
        }
    }

    for (int run = 0;; ++run) {
        eng->reset();
        g_status = {};
        text_begin();
        for (int i = 0; i < n_prompt; ++i) show_token(*eng, prompt[i], TFT_LIGHTGREY);

        // Prefill: the prompt goes through the layers max_batch tokens at a
        // time (one weight read per batch); only the last needs the LM head.
        int64_t t0 = esp_timer_get_time();
        const double sd0 = src.sd_busy_seconds(), stall0 = src.stall_seconds();
        const uint64_t read0 = src.bytes_read();
        const double head0 = eng->stats().lm_head_s;
        const long rescore0 = eng->stats().rescore_reads;
        const double scan0 = eng->stats().flash_scan_s;
        const double proj0 = eng->stats().flash_proj_s, topk0 = eng->stats().topk_s;
        int32_t next = lfm::Engine::kNoToken;
        char phase[32];
        drafter.reset();
        for (int i = 0; i < n_prompt; ++i) drafter.push(prompt[i]);
        lfm::SpecStats spec;
        int32_t first[lfm::Engine::kMaxBatch];  // tokens produced by the prompt pass
        int n_first = 0;
        // Prompt cache: the previous run had this prompt, so its K/V rows are
        // still in kv.bin and the conv state after it was snapshotted.
        static int32_t cached_first = -1;
        const bool cached = rc.prompt_cache && run > 0 && cached_first >= 0 && eng->restore_prefix();
        if (cached) {
            first[0] = cached_first;
            n_first = 1;
            next = cached_first;
            drafter.push(next);
        } else if (rc.spec && rc.spec_prefill) {
            // Drafts for the first answer tokens ride in the last prompt batch.
            set_phase("prefill");
            n_first = lfm::speculative_prefill(*eng, drafter, {prompt, static_cast<std::size_t>(n_prompt)},
                                               cfg.max_batch, max_draft, first, &spec);
            next = n_first > 0 ? first[n_first - 1] : n_first;
        } else {
            for (int i = 0; i < n_prompt && next >= lfm::Engine::kNoToken;) {
                const int n = n_prompt - i < cfg.max_batch ? n_prompt - i : cfg.max_batch;
                std::snprintf(phase, sizeof(phase), "prefill %d-%d/%d", i + 1, i + n, n_prompt);
                set_phase(phase);
                next = eng->prefill({prompt + i, static_cast<std::size_t>(n)}, i + n == n_prompt);
                i += n;
            }
            if (next >= 0) {
                first[0] = next;
                n_first = 1;
                drafter.push(next);
                if (rc.prompt_cache && eng->snapshot_prefix()) cached_first = next;
            }
        }
        const double prefill_s = (esp_timer_get_time() - t0) / 1e6;
        ESP_LOGI(TAG, "prefill %d tokens: %.1f s", n_prompt, prefill_s);

        // Generate until EOS or the context is full: speculative steps, each
        // one pass over the weights for the newest token plus its drafts.
        set_phase("generating");
        int64_t t_gen = esp_timer_get_time();
        int32_t fed = next;
        for (int i = 0; i < n_first; ++i) {
            show_token(*eng, first[i], i + 1 < n_first ? TFT_GREENYELLOW : TFT_WHITE);
            emit_token(first[i]);
            ++g_status.tokens;
        }
        auto finished = [&] {
            if (rc.gen > 0 && g_status.tokens >= rc.gen) return true;
            return fed == lfm::kEosId && !(rc.ignore_eos && rc.gen > 0);
        };
        while (fed >= 0 && !finished()) {
            int32_t produced[lfm::Engine::kMaxBatch];
            int got;
            if (rc.spec) {
                got = lfm::speculative_step(*eng, drafter, fed, max_draft, produced, &spec);
            } else {
                produced[0] = eng->forward(fed, true);
                got = produced[0] < 0 ? produced[0] : 1;
                ++spec.passes;
                spec.tokens += 1;
            }
            if (got < 0) {
                next = got;
                break;
            }
            for (int i = 0; i < got; ++i) {
                show_token(*eng, produced[i], i + 1 < got ? TFT_GREENYELLOW : TFT_WHITE);  // accepted drafts tinted
                emit_token(produced[i]);
                ++g_status.tokens;
                fed = produced[i];
                if (finished()) break;
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
        const double total_s = (esp_timer_get_time() - t0) / 1e6;
        const double gen_s = (esp_timer_get_time() - t_gen) / 1e6;
        ESP_LOGI(TAG, "done: %d tokens, %.1f s total, read %.0f MB", g_status.tokens, total_s, src.bytes_read() / 1e6);
        // One machine-readable line per run (tools/bench.py).
        ESP_LOGI(TAG, "RESULT run=%d prompt_tokens=%d prefill_s=%.2f gen_tokens=%d gen_s=%.2f s_per_tok=%.3f "
                 "total_s=%.2f passes=%d drafted=%d accepted=%d batch=%d read_mb=%.1f sd_busy=%.3f stall=%.3f heap_free=%u "
                 "lm_head_s=%.2f rescore_reads=%ld flash_scan_s=%.2f proj_s=%.2f topk_s=%.2f",
                 run, n_prompt, prefill_s, g_status.tokens, gen_s,
                 g_status.tokens > 1 ? gen_s / (g_status.tokens - 1) : 0.0, total_s, spec.passes, spec.drafted,
                 spec.accepted, cfg.max_batch, (src.bytes_read() - read0) / 1e6, (src.sd_busy_seconds() - sd0) / total_s,
                 (src.stall_seconds() - stall0) / total_s,
                 (unsigned)heap_caps_get_free_size(MALLOC_CAP_8BIT), eng->stats().lm_head_s - head0,
                 eng->stats().rescore_reads - rescore0, eng->stats().flash_scan_s - scan0,
                 eng->stats().flash_proj_s - proj0, eng->stats().topk_s - topk0);

        if (run + 1 < rc.runs) continue;
        do {
            vTaskDelay(pdMS_TO_TICKS(20));
            M5.update();
        } while (!M5.BtnA.wasPressed());
    }
}
