#include "kernel_test.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <initializer_list>

#include "esp_cpu.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_random.h"
#include "lfm/math/esp32s3.hpp"
#include "lfm/math/ref.hpp"

namespace m = lfm::math;

extern "C" void lfm_s3_q4_q8_sums(const std::byte* row, const int8_t* xq, int32_t* sums, int nb);
extern "C" void lfm_s3_q8_q8_sums(const std::byte* row, const int8_t* xq, int32_t* sums, int nb);

namespace {

constexpr const char* TAG = "kernel";

struct Kind {
    const char* name;
    std::size_t block_bytes;
    float (*pie)(const std::byte*, const int8_t*, const float*, std::size_t);
    float (*ref)(const std::byte*, const int8_t*, const float*, std::size_t);
};

}  // namespace

bool kernel_test(PrintFn print) {
    constexpr std::size_t kMaxN = 2560;
    auto* raw = static_cast<std::byte*>(heap_caps_aligned_alloc(16, kMaxN / 32 * 34 + 64, MALLOC_CAP_8BIT));
    auto* xq = static_cast<int8_t*>(heap_caps_aligned_alloc(16, kMaxN, MALLOC_CAP_8BIT));
    float xs[kMaxN / 32];
    char line[96];
    bool all_ok = true;

    const Kind kinds[] = {
        {"q4", 18, m::esp32s3::dot_q4_0_q8, m::ref::dot_q4_0_q8},
        {"q8", 34, m::esp32s3::dot_q8_0_q8, m::ref::dot_q8_0_q8},
    };
    for (const Kind& k : kinds) {
        for (std::size_t n : {std::size_t{1024}, kMaxN}) {
            int bad = 0, cases = 0;
            for (int trial = 0; trial < 20; ++trial) {
                for (std::size_t off = 0; off < 16; ++off) {
                    std::byte* row = raw + off;
                    esp_fill_random(row, n / 32 * k.block_bytes);
                    for (std::size_t b = 0; b < n / 32; ++b) {  // sane f16 scales (no inf/NaN)
                        const uint16_t sc = static_cast<uint16_t>(0x2000 + (esp_random() & 0x0FFF));
                        std::memcpy(row + b * k.block_bytes, &sc, 2);
                    }
                    esp_fill_random(xq, n);
                    for (std::size_t b = 0; b < n / 32; ++b) xs[b] = 0.001f * static_cast<float>(1 + (esp_random() & 255));
                    const float a = k.pie(row, xq, xs, n), r = k.ref(row, xq, xs, n);
                    ++cases;
                    if (a != r) {
                        if (bad++ < 3) ESP_LOGE(TAG, "%s n=%u off=%u: pie %.6f ref %.6f", k.name, (unsigned)n, (unsigned)off, a, r);
                    }
                }
            }
            // Timing on the last row.
            const std::byte* row = raw + 2;
            uint32_t t0 = esp_cpu_get_cycle_count();
            for (int i = 0; i < 100; ++i) k.pie(row, xq, xs, n);
            const uint32_t pie_cyc = (esp_cpu_get_cycle_count() - t0) / 100;
            t0 = esp_cpu_get_cycle_count();
            for (int i = 0; i < 100; ++i) k.ref(row, xq, xs, n);
            const uint32_t ref_cyc = (esp_cpu_get_cycle_count() - t0) / 100;

            int32_t sums[kMaxN / 32];
            t0 = esp_cpu_get_cycle_count();
            for (int i = 0; i < 100; ++i) {
                if (k.block_bytes == 18) lfm_s3_q4_q8_sums(row, xq, sums, static_cast<int>(n / 32));
                else lfm_s3_q8_q8_sums(row, xq, sums, static_cast<int>(n / 32));
            }
            const uint32_t asm_cyc = (esp_cpu_get_cycle_count() - t0) / 100;
            ESP_LOGI(TAG, "%s n=%u: asm alone %lu cyc = %.1f per block", k.name, (unsigned)n, (unsigned long)asm_cyc,
                     static_cast<double>(asm_cyc) / (n / 32));
            all_ok = all_ok && bad == 0;
            std::snprintf(line, sizeof(line), "%s n=%4u %s %d/%d  pie %5lu ref %6lu cyc (%.1fx)", k.name, (unsigned)n,
                          bad == 0 ? "OK " : "BAD", cases - bad, cases, (unsigned long)pie_cyc, (unsigned long)ref_cyc,
                          static_cast<double>(ref_cyc) / pie_cyc);
            print(line);
            ESP_LOGI(TAG, "%s", line);
        }
    }
    // Row scan (LM head table): many 4-block rows against one activation.
    {
        constexpr std::size_t kRows = 1000, kCols = 128, kRb = kCols / 32 * 18;
        auto* table = static_cast<std::byte*>(heap_caps_malloc(kRows * kRb + 32, MALLOC_CAP_8BIT));
        esp_fill_random(table, kRows * kRb);
        for (std::size_t b = 0; b < kRows * kCols / 32; ++b) {
            const uint16_t sc = static_cast<uint16_t>(0x2000 + (esp_random() & 0x0FFF));
            std::memcpy(table + b * 18, &sc, 2);
        }
        esp_fill_random(xq, kCols);
        static float pie_out[kRows], ref_out[kRows];
        const uint32_t t0 = esp_cpu_get_cycle_count();
        m::esp32s3::scan_q4_0_q8(table, kRows, kCols, xq, xs, pie_out);
        const uint32_t cyc = esp_cpu_get_cycle_count() - t0;
        m::ref::scan_q4_0_q8(table, kRows, kCols, xq, xs, ref_out);
        int bad = 0;
        for (std::size_t r = 0; r < kRows; ++r) bad += pie_out[r] != ref_out[r];
        all_ok = all_ok && bad == 0;
        std::snprintf(line, sizeof(line), "scan q4 rows=%u %s, %lu cyc/row", (unsigned)kRows, bad ? "BAD" : "OK",
                      (unsigned long)(cyc / kRows));
        print(line);
        ESP_LOGI(TAG, "%s", line);
        heap_caps_free(table);
    }
    // Multi-token Q4 dot: one row against 1..7 token inputs, vs the per-token kernel.
    {
        constexpr std::size_t kStride = 2560;  // the engine's xq_cols_
        auto* xqm = static_cast<int8_t*>(heap_caps_aligned_alloc(16, kStride * 7, MALLOC_CAP_8BIT));
        static float xsm[7 * (kStride / 32)];
        esp_fill_random(xqm, kStride * 7);
        for (float& v : xsm) v = 0.001f + (esp_random() % 1000) * 1e-5f;
        for (std::size_t n : {std::size_t{1024}, std::size_t{2560}}) {
            esp_fill_random(raw, n / 32 * 18);
            for (std::size_t b = 0; b < n / 32; ++b) {
                const uint16_t sc = static_cast<uint16_t>(0x2000 + (esp_random() & 0x0FFF));
                std::memcpy(raw + b * 18, &sc, 2);
            }
            for (int tok : {1, 3, 7}) {
                float single[7], multi[7];
                uint32_t t0 = esp_cpu_get_cycle_count();
                for (int rep = 0; rep < 20; ++rep) {
                    for (int t = 0; t < tok; ++t) {
                        single[t] = m::esp32s3::dot_q4_0_q8(raw, xqm + t * kStride, xsm + t * (kStride / 32), n);
                    }
                }
                const uint32_t single_cyc = (esp_cpu_get_cycle_count() - t0) / 20;
                t0 = esp_cpu_get_cycle_count();
                for (int rep = 0; rep < 20; ++rep) {
                    m::esp32s3::dot_q4_0_q8_multi(raw, xqm, kStride, xsm, kStride / 32, tok, n, multi);
                }
                const uint32_t multi_cyc = (esp_cpu_get_cycle_count() - t0) / 20;
                int bad = 0;
                for (int t = 0; t < tok; ++t) bad += single[t] != multi[t];
                all_ok = all_ok && bad == 0;
                std::snprintf(line, sizeof(line), "q4 multi n=%u tok=%d %s: %lu vs %lu cyc/block", (unsigned)n, tok,
                              bad ? "BAD" : "OK", (unsigned long)(multi_cyc / (n / 32)),
                              (unsigned long)(single_cyc / (n / 32)));
                print(line);
                ESP_LOGI(TAG, "%s", line);
            }
        }
        heap_caps_free(xqm);
    }
    heap_caps_free(raw);
    heap_caps_free(xq);
    return all_ok;
}

#include "esp_partition.h"
#include "esp_timer.h"
#include "lfm/format.hpp"

// Flash table scan: memory bandwidth vs compute (one position).
void flash_scan_bench(PrintFn print) {
    const esp_partition_t* part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "lmhead");
    const void* ptr = nullptr;
    esp_partition_mmap_handle_t h;
    if (part == nullptr || esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA, &ptr, &h) != ESP_OK) return;
    lfm::FlashHeadHeader fh;
    std::memcpy(&fh, ptr, sizeof(fh));
    const auto* table = static_cast<const std::byte*>(ptr) + fh.table_offset;
    const std::size_t rb = fh.rank / 32 * 18, bytes = static_cast<std::size_t>(fh.vocab) * rb;
    char line[96];

    int64_t t0 = esp_timer_get_time();
    uint32_t acc = 0;
    const auto* w = reinterpret_cast<const uint32_t*>(table);
    for (std::size_t i = 0; i < bytes / 4; i += 8) acc += w[i];  // one load per 32-byte cache line
    const double read_s = (esp_timer_get_time() - t0) / 1e6;

    auto* xq = static_cast<int8_t*>(heap_caps_aligned_alloc(16, fh.rank, MALLOC_CAP_8BIT));
    float xs[16];
    for (std::size_t i = 0; i < fh.rank; ++i) xq[i] = static_cast<int8_t>(i);
    for (float& f : xs) f = 0.01f;
    t0 = esp_timer_get_time();
    float best = -1e30f;
    for (uint32_t v = 0; v < fh.vocab; ++v) best = std::max(best, m::esp32s3::dot_q4_0_q8(table + v * rb, xq, xs, fh.rank));
    const double scan_s = (esp_timer_get_time() - t0) / 1e6;
    std::snprintf(line, sizeof(line), "flash: read %.2f MB in %.3f s (%.1f MB/s), scan 1 pos %.3f s [%u %.0f]",
                  bytes / 1e6, read_s, bytes / 1e6 / read_s, scan_s, (unsigned)acc, best);
    print(line);
    ESP_LOGI(TAG, "%s", line);
    heap_caps_free(xq);
}
