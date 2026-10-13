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

// ---- scalar conversions

// IEEE half -> float. Handles zeros, subnormals, inf/NaN.
float f16_to_f32(uint16_t h);

// float -> bf16 with round-to-nearest-even (NaN stays NaN).
uint16_t f32_to_bf16(float f);

// ---- dot product of one stored weight row with a float vector
// `row` points at raw file bytes (no alignment assumed); x.size() == cols.

// Row of bf16 values.
float dot_bf16(const std::byte* row, std::span<const float> x);

// Row of Q8_0 blocks: x.size() / 32 blocks of {f16 scale, int8[32]}.
float dot_q8_0(const std::byte* row, std::span<const float> x);

// Row of Q4_0 blocks: x.size() / 32 blocks of {f16 scale, uint8[16]}.
float dot_q4_0(const std::byte* row, std::span<const float> x);

// ---- integer path: activations quantized to int8 too (Q8 x Q8)
// A vector of n floats becomes n / 32 blocks: int8 q[n] and a float scale
// per block, x ~= scale[b] * q[b*32 + i]. Then a weight row dot product is
// sum over blocks of (w_scale * x_scale * integer dot of 32 int8 pairs):
// the inner loop is pure integer math, which SIMD units do 16 at a time.

// q[i] = round(x[i] / scale[b]), scale[b] = max|x in block b| / 127.
void quantize_q8(std::span<const float> x, int8_t* q, float* scale);

// Q8_0 weight row (34-byte blocks) . quantized activation.
float dot_q8_0_q8(const std::byte* row, const int8_t* xq, const float* xscale, std::size_t n);

// Q4_0 weight row (18-byte blocks) . quantized activation.
float dot_q4_0_q8(const std::byte* row, const int8_t* xq, const float* xscale, std::size_t n);

// rows Q4_0 rows of n columns, stored back to back, each dotted with the
// same quantized activation: out[r]. (An LM head table scan.)
void scan_q4_0_q8(const std::byte* rows, std::size_t n_rows, std::size_t n, const int8_t* xq, const float* xscale,
                  float* out);

}  // namespace lfm::math::ref
