#include "kernel_test.hpp"

#include <cstdio>
#include <cstring>

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
    heap_caps_free(raw);
    heap_caps_free(xq);
    return all_ok;
}
