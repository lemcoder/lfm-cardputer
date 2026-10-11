#pragma once

// Reference (scalar, portable) implementations. Correctness first: these are
// the ground truth every SIMD backend is tested against. Keep them simple.

#include <cstddef>
#include <cstdint>
#include <span>

#include "lfm/math/types.hpp"

namespace lfm::math::ref {

// sum_i a[i] * b[i]. a and b have the same size.
float dot(std::span<const float> a, std::span<const float> b);

// out[r] = dot(w.row(r), x).  out.size() == w.rows, x.size() == w.cols.
void matvec(std::span<float> out, MatRef w, std::span<const float> x);

// out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i].  out may alias x.
void rmsnorm(std::span<float> out, std::span<const float> x, std::span<const float> weight, float eps);

// acc[i] += x[i]  (residual connection)
void add(std::span<float> acc, std::span<const float> x);

// out[i] = a[i] * b[i].  out may alias a or b.
void mul(std::span<float> out, std::span<const float> a, std::span<const float> b);

// x[i] = x[i] * sigmoid(x[i]), in place.
void silu(std::span<float> x);

// out[i] = silu(gate[i]) * up[i].  The MLP activation. out may alias gate or up.
void swiglu(std::span<float> out, std::span<const float> gate, std::span<const float> up);

// In-place softmax. Must not overflow for large inputs (subtract the max first).
void softmax(std::span<float> x);

// Rotary position embedding, "rotate_half" form, in place.
// x holds x.size() / head_dim heads back to back. Within each head, element i
// (i < head_dim/2) is paired with element i + head_dim/2 and the pair is
// rotated by angle pos * theta^(-2i / head_dim).
void rope(std::span<float> x, std::size_t head_dim, std::size_t pos, float theta);

// bf16 -> fp32: the bf16 bits are the top 16 bits of the fp32.
void bf16_to_f32(std::span<float> out, std::span<const uint16_t> in);

// Index of the largest element (first one on ties). Greedy sampling.
std::size_t argmax(std::span<const float> x);

}  // namespace lfm::math::ref
