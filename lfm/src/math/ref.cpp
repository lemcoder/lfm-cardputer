#include "lfm/math/ref.hpp"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <span>

namespace {
inline float sigmoid(float v) { return 1 / (1 + std::exp(-v)); }

} // namespace

namespace lfm::math::ref {

float dot(std::span<const float> a, std::span<const float> b) {
  assert(a.size() == b.size());

  float sum = 0;
  for (std::size_t i = 0; i < a.size(); i++) {
    sum += a[i] * b[i];
  }

  return sum;
}

void matvec(std::span<float> out, MatRef w, std::span<const float> x) {
  assert(out.size() >= w.rows);
  assert(x.size() >= w.cols);

  for (std::size_t i = 0; i < w.rows; i++) {
    std::size_t row_offset = i * w.cols;
    std::span<const float> matrix_row(w.data + row_offset, w.cols);
    out[i] = dot(matrix_row, x);
  }
}

void rmsnorm(std::span<float> out, std::span<const float> x,
             std::span<const float> weight, float eps) {
  // 1. Calculate the size of the input vector
  std::size_t d = x.size();
  assert(weight.size() == d);
  assert(out.size() >= d);

  // 2. Compute the sum of squares
  float sum_squares = 0;
  for (std::size_t i = 0; i < d; ++i) {
    sum_squares += x[i] * x[i];
  }

  // 3. Calculate the root mean square (RMS) inverse scale factor
  float rms_inv = 1.0f / std::sqrt((sum_squares / static_cast<float>(d)) + eps);

  // 4. Normalize and apply learnable weights
  for (std::size_t i = 0; i < d; ++i) {
    out[i] = (x[i] * rms_inv) * weight[i];
  }
}

void add(std::span<float> acc, std::span<const float> x) {
  assert(acc.size() == x.size());

  for (std::size_t i = 0; i < x.size(); i++) {
    acc[i] += x[i];
  }
}

void mul(std::span<float> out, std::span<const float> a,
         std::span<const float> b) {
  assert(out.size() == a.size() && a.size() == b.size());

  for (std::size_t i = 0; i < a.size(); i++) {
    out[i] = a[i] * b[i];
  }
}

inline float silu(float g) { return g * sigmoid(g); }

void silu(std::span<float> x) {
  for (std::size_t i = 0; i < x.size(); i++) {
    x[i] = silu(x[i]);
  }
}

void swiglu(std::span<float> out, std::span<const float> gate,
            std::span<const float> up) {
  for (std::size_t i = 0; i < gate.size(); i++) {
    out[i] = silu(gate[i]) * up[i];
  }
}

void softmax(std::span<float> x) {
  float max = x[0];
  for (std::size_t i = 0; i < x.size(); i++) {
    max = std::max(x[i], max);
  }

  float sum = 0;
  for (std::size_t i = 0; i < x.size(); i++) {
    x[i] = std::exp(x[i] - max);
    sum += x[i];
  }

  for (std::size_t i = 0; i < x.size(); i++) {
    x[i] /= sum;
  }
}

void rope(std::span<float> x, std::size_t head_dim, std::size_t pos,
          float theta) {
  const std::size_t half = head_dim / 2;
  const std::size_t n_heads = x.size() / head_dim;

  for (std::size_t h = 0; h < n_heads; h++) {
    const std::size_t base = h * head_dim;

    for (std::size_t k = 0; k < half; k++) {
      // Pair k rotates at freq = theta^(-2k / head_dim): pair 0 fastest.
      const float exponent =
          2.0f * static_cast<float>(k) / static_cast<float>(head_dim);
      const float freq = 1.0f / std::pow(theta, exponent);
      const float angle = static_cast<float>(pos) * freq;
      const float cos_a = std::cos(angle);
      const float sin_a = std::sin(angle);

      // Read both before writing: the second line needs the original a.
      const float a = x[base + k];
      const float b = x[base + k + half];

      x[base + k] = a * cos_a - b * sin_a;
      x[base + k + half] = b * cos_a + a * sin_a;
    }
  }
}

void bf16_to_f32(std::span<float> out, std::span<const uint16_t> in) {
  for (std::size_t i = 0; i < in.size(); i++) {
    uint32_t wide = in[i];
    out[i] = std::bit_cast<float>(wide << 16);
  }
}

std::size_t argmax(std::span<const float> x) {
  assert(!x.empty());

  std::size_t best = 0;
  for (std::size_t i = 1; i < x.size(); i++) {
    if (x[i] > x[best]) {
      best = i;
    }
  }

  return best;
}

float f16_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x3FFu;
    if (exp == 0) {
        // Zero or subnormal: value = mant * 2^-24.
        const float v = static_cast<float>(mant) * 5.9604644775390625e-8f;
        return sign ? -v : v;
    }
    const uint32_t bits = exp == 0x1F ? sign | 0x7F800000u | (mant << 13)       // inf / NaN
                                      : sign | ((exp + 112) << 23) | (mant << 13);  // rebias 15 -> 127
    return std::bit_cast<float>(bits);
}

uint16_t f32_to_bf16(float f) {
    const uint32_t bits = std::bit_cast<uint32_t>(f);
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) return static_cast<uint16_t>((bits >> 16) | 0x40);  // quiet NaN
    const uint32_t rounding = 0x7FFFu + ((bits >> 16) & 1u);  // to nearest, ties to even
    return static_cast<uint16_t>((bits + rounding) >> 16);
}

float dot_bf16(const std::byte* row, std::span<const float> x) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < x.size(); ++i) {
        uint16_t h;
        std::memcpy(&h, row + 2 * i, 2);
        sum += std::bit_cast<float>(static_cast<uint32_t>(h) << 16) * x[i];
    }
    return sum;
}

float dot_q8_0(const std::byte* row, std::span<const float> x) {
    assert(x.size() % 32 == 0);
    float sum = 0.0f;
    for (std::size_t b = 0; b < x.size() / 32; ++b) {
        const std::byte* block = row + b * 34;
        uint16_t scale_bits;
        std::memcpy(&scale_bits, block, 2);
        const auto* q = reinterpret_cast<const int8_t*>(block + 2);  // byte types may alias anything
        const float* xb = x.data() + b * 32;
        float acc = 0.0f;
        for (std::size_t i = 0; i < 32; ++i) {
            acc += static_cast<float>(q[i]) * xb[i];
        }
        sum += f16_to_f32(scale_bits) * acc;
    }
    return sum;
}

float dot_q4_0(const std::byte* row, std::span<const float> x) {
    assert(x.size() % 32 == 0);
    float sum = 0.0f;
    for (std::size_t b = 0; b < x.size() / 32; ++b) {
        const std::byte* block = row + b * 18;
        uint16_t scale_bits;
        std::memcpy(&scale_bits, block, 2);
        const auto* q = reinterpret_cast<const uint8_t*>(block + 2);
        const float* xb = x.data() + b * 32;
        float acc = 0.0f;
        for (std::size_t j = 0; j < 16; ++j) {
            acc += static_cast<float>(static_cast<int>(q[j] & 0x0F) - 8) * xb[j];
            acc += static_cast<float>(static_cast<int>(q[j] >> 4) - 8) * xb[j + 16];
        }
        sum += f16_to_f32(scale_bits) * acc;
    }
    return sum;
}

void quantize_q8(std::span<const float> x, int8_t* q, float* scale) {
    assert(x.size() % 32 == 0);
    for (std::size_t b = 0; b < x.size() / 32; ++b) {
        const float* xb = x.data() + b * 32;
        float amax = 0.0f;
        for (std::size_t i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(xb[i]));
        const float d = amax / 127.0f;
        const float inv = d > 0.0f ? 1.0f / d : 0.0f;
        scale[b] = d;
        for (std::size_t i = 0; i < 32; ++i) q[b * 32 + i] = static_cast<int8_t>(std::lrintf(xb[i] * inv));
    }
}

float dot_q8_0_q8(const std::byte* row, const int8_t* xq, const float* xscale, std::size_t n) {
    float sum = 0.0f;
    for (std::size_t b = 0; b < n / 32; ++b) {
        const std::byte* block = row + b * 34;
        uint16_t scale_bits;
        std::memcpy(&scale_bits, block, 2);
        const auto* w = reinterpret_cast<const int8_t*>(block + 2);
        const int8_t* x = xq + b * 32;
        int32_t acc = 0;
        for (std::size_t i = 0; i < 32; ++i) acc += static_cast<int32_t>(w[i]) * x[i];
        sum += f16_to_f32(scale_bits) * xscale[b] * static_cast<float>(acc);
    }
    return sum;
}

float dot_q4_0_q8(const std::byte* row, const int8_t* xq, const float* xscale, std::size_t n) {
    float sum = 0.0f;
    for (std::size_t b = 0; b < n / 32; ++b) {
        const std::byte* block = row + b * 18;
        uint16_t scale_bits;
        std::memcpy(&scale_bits, block, 2);
        const auto* w = reinterpret_cast<const uint8_t*>(block + 2);
        const int8_t* x = xq + b * 32;
        int32_t acc = 0;
        for (std::size_t j = 0; j < 16; ++j) {
            acc += (static_cast<int32_t>(w[j] & 0x0F) - 8) * x[j];
            acc += (static_cast<int32_t>(w[j] >> 4) - 8) * x[j + 16];
        }
        sum += f16_to_f32(scale_bits) * xscale[b] * static_cast<float>(acc);
    }
    return sum;
}

void scan_q4_0_q8(const std::byte* rows, std::size_t n_rows, std::size_t n, const int8_t* xq, const float* xscale,
                  float* out) {
    const std::size_t rb = n / 32 * 18;
    for (std::size_t r = 0; r < n_rows; ++r) out[r] = dot_q4_0_q8(rows + r * rb, xq, xscale, n);
}

} // namespace lfm::math::ref
