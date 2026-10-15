#include "lfm/math/ops.hpp"

#if defined(LFM_MATH_ESP32S3)

#include <bit>
#include <cassert>

extern "C" {
void lfm_s3_q4_q8_sums(const std::byte* row, const int8_t* xq, int32_t* sums, int nb);
void lfm_s3_q8_q8_sums(const std::byte* row, const int8_t* xq, int32_t* sums, int nb);
void lfm_s3_q4_q8_rows(const std::byte* blocks, const int8_t* xq, int32_t* sums, int rows, int nb);
void lfm_s3_q4_q8_multi(const std::byte* row, const int8_t* xq, int32_t* sums, int nb, int n_tok, int xq_stride);
}

namespace lfm::math::esp32s3 {
namespace {

constexpr std::size_t kMaxBlocks = 128;  // rows up to 4096 wide

// No memcpy in here: ESP-IDF builds with -fno-builtin-memcpy, so each one
// would be a real call (it was ~100 cycles per block).

// f16 -> f32 inline: normal numbers by rebiasing the exponent; zero,
// subnormal, inf and NaN (rare for weight scales) through the reference.
inline float f16_fast(uint32_t h) {
    const uint32_t exp = (h >> 10) & 0x1Fu;
    if (exp == 0 || exp == 0x1F) [[unlikely]] return ref::f16_to_f32(static_cast<uint16_t>(h));
    return std::bit_cast<float>(((h & 0x8000u) << 16) | (((h & 0x7FFFu) << 13) + 0x38000000u));
}

// sum_b w_scale[b] * x_scale[b] * sums[b], same order as the reference.
float combine(const std::byte* row, std::size_t block_bytes, const float* xscale, const int32_t* sums, std::size_t nb) {
    const auto* p = reinterpret_cast<const uint8_t*>(row);
    float sum = 0.0f;
    for (std::size_t b = 0; b < nb; ++b, p += block_bytes) {
        const uint32_t scale_bits = p[0] | (static_cast<uint32_t>(p[1]) << 8);  // little-endian, any alignment
        sum += f16_fast(scale_bits) * xscale[b] * static_cast<float>(sums[b]);
    }
    return sum;
}

}  // namespace

float dot_q8_0_q8(const std::byte* row, const int8_t* xq, const float* xscale, std::size_t n) {
    const std::size_t nb = n / 32;
    assert(nb <= kMaxBlocks && (reinterpret_cast<uintptr_t>(xq) & 15) == 0);
    int32_t sums[kMaxBlocks];
    lfm_s3_q8_q8_sums(row, xq, sums, static_cast<int>(nb));
    return combine(row, 34, xscale, sums, nb);
}

float dot_q4_0_q8(const std::byte* row, const int8_t* xq, const float* xscale, std::size_t n) {
    const std::size_t nb = n / 32;
    assert(nb <= kMaxBlocks && (reinterpret_cast<uintptr_t>(xq) & 15) == 0);
    int32_t sums[kMaxBlocks];
    lfm_s3_q4_q8_sums(row, xq, sums, static_cast<int>(nb));
    return combine(row, 18, xscale, sums, nb);
}

// Each weight block unpacked once for all tokens (sums[block][token]), each
// scale converted once; per token the same arithmetic in the same order as
// dot_q4_0_q8, so the results are identical.
void dot_q4_0_q8_multi(const std::byte* row, const int8_t* xq, std::size_t xq_stride, const float* xscale,
                       std::size_t xs_stride, int n_tok, std::size_t n, float* out) {
    constexpr int kMaxTok = 8;
    const std::size_t nb = n / 32;
    assert(nb <= kMaxBlocks && n_tok >= 1 && n_tok <= kMaxTok && (reinterpret_cast<uintptr_t>(xq) & 15) == 0 &&
           xq_stride % 16 == 0);
    int32_t sums[kMaxBlocks * kMaxTok];
    lfm_s3_q4_q8_multi(row, xq, sums, static_cast<int>(nb), n_tok, static_cast<int>(xq_stride));
    // Block scales once, then each token's sum in a register (the same
    // expression, d * xs * sum, in the same block order as dot_q4_0_q8).
    float d[kMaxBlocks];
    const auto* p = reinterpret_cast<const uint8_t*>(row);
    for (std::size_t b = 0; b < nb; ++b, p += 18) d[b] = f16_fast(p[0] | (static_cast<uint32_t>(p[1]) << 8));
    for (int t = 0; t < n_tok; ++t) {
        const float* xs = xscale + static_cast<std::size_t>(t) * xs_stride;
        const int32_t* s = sums + t;
        float acc = 0.0f;
        for (std::size_t b = 0; b < nb; ++b, s += n_tok) acc += d[b] * xs[b] * static_cast<float>(*s);
        out[t] = acc;
    }
}

void scan_q4_0_q8(const std::byte* rows, std::size_t n_rows, std::size_t n, const int8_t* xq, const float* xscale,
                  float* out) {
    const std::size_t nb = n / 32;
    assert(nb <= 8 && (reinterpret_cast<uintptr_t>(xq) & 15) == 0);
    constexpr std::size_t kChunk = 64;  // rows per kernel call
    int32_t sums[kChunk * 8];
    for (std::size_t r0 = 0; r0 < n_rows; r0 += kChunk) {
        const std::size_t cnt = n_rows - r0 < kChunk ? n_rows - r0 : kChunk;
        const std::byte* base = rows + r0 * nb * 18;
        lfm_s3_q4_q8_rows(base, xq, sums, static_cast<int>(cnt), static_cast<int>(nb));
        for (std::size_t r = 0; r < cnt; ++r) out[r0 + r] = combine(base + r * nb * 18, 18, xscale, sums + r * nb, nb);
    }
}

}  // namespace lfm::math::esp32s3
#endif
