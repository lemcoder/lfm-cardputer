#include "lfm/math/ops.hpp"

#if defined(LFM_MATH_ESP32S3)

#include <bit>
#include <cassert>

extern "C" {
void lfm_s3_q4_q8_sums(const std::byte* row, const int8_t* xq, int32_t* sums, int nb);
void lfm_s3_q8_q8_sums(const std::byte* row, const int8_t* xq, int32_t* sums, int nb);
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

}  // namespace lfm::math::esp32s3
#endif
