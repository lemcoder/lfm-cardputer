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
  assert(weight.size() == d);
  assert(out.size() >= d);

  // 1. Calculate the size of the input vector
  std::size_t d = x.size();

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
  assert(acc.size() == a.size() && a.size() == b.size());

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

} // namespace lfm::math::ref
