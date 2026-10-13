#pragma once

// The math API the model code calls: lfm::math::matvec(...), etc.
//
// Each op forwards to the fastest backend that implements it, falling back to
// ref::. Backends are picked at compile time (see lfm/CMakeLists.txt):
//   LFM_MATH_NEON     arm64 host (Apple Silicon)   -> lfm/math/neon.hpp
//   LFM_MATH_ESP32S3  ESP32-S3 PIE SIMD            -> lfm/math/esp32s3.hpp
// A backend header announces each op it provides with a macro, e.g.
//   #define LFM_NEON_HAS_DOT 1
// so a backend can be filled in one op at a time.

#include "lfm/math/ref.hpp"

#if defined(LFM_MATH_NEON)
#include "lfm/math/neon.hpp"
#endif
#if defined(LFM_MATH_ESP32S3)
#include "lfm/math/esp32s3.hpp"
#endif

namespace lfm::math {

inline float dot(std::span<const float> a, std::span<const float> b) {
#if defined(LFM_NEON_HAS_DOT)
    return neon::dot(a, b);
#elif defined(LFM_ESP32S3_HAS_DOT)
    return esp32s3::dot(a, b);
#else
    return ref::dot(a, b);
#endif
}

inline void matvec(std::span<float> out, MatRef w, std::span<const float> x) {
#if defined(LFM_NEON_HAS_MATVEC)
    neon::matvec(out, w, x);
#elif defined(LFM_ESP32S3_HAS_MATVEC)
    esp32s3::matvec(out, w, x);
#else
    ref::matvec(out, w, x);
#endif
}

inline float dot_q8_0_q8(const std::byte* row, const int8_t* xq, const float* xscale, std::size_t n) {
#if defined(LFM_ESP32S3_HAS_DOT_Q8_Q8)
    return esp32s3::dot_q8_0_q8(row, xq, xscale, n);
#else
    return ref::dot_q8_0_q8(row, xq, xscale, n);
#endif
}

inline float dot_q4_0_q8(const std::byte* row, const int8_t* xq, const float* xscale, std::size_t n) {
#if defined(LFM_ESP32S3_HAS_DOT_Q4_Q8)
    return esp32s3::dot_q4_0_q8(row, xq, xscale, n);
#else
    return ref::dot_q4_0_q8(row, xq, xscale, n);
#endif
}

// Ops below are cheap (O(n), not O(n^2)); they only get a backend if profiling
// says so. Add the same #if pattern when you vectorize one.
using ref::add;
using ref::argmax;
using ref::bf16_to_f32;
using ref::dot_bf16;
using ref::dot_q4_0;
using ref::dot_q8_0;
using ref::f16_to_f32;
using ref::f32_to_bf16;
using ref::mul;
using ref::quantize_q8;
using ref::rmsnorm;
using ref::rope;
using ref::silu;
using ref::softmax;
using ref::swiglu;

}  // namespace lfm::math
