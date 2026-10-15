#pragma once

// ESP32-S3 backend. The S3's PIE SIMD unit works on 128-bit registers of
// int8/int16 lanes (EE.VMULAS.S8.ACCX etc.), not floats, so it accelerates
// the integer (int8 activation) dot products. Kernels: esp32s3_dot.S.
// Reference: ESP32-S3 TRM, "Processor Instruction Extensions"; esp-dsp.
//
// Same pattern as neon.hpp: declare, implement in lfm/src/math/esp32s3.cpp,
// then #define LFM_ESP32S3_HAS_<OP> 1.

#include <cstddef>
#include <cstdint>

namespace lfm::math::esp32s3 {

// Same contract as ref::dot_q8_0_q8 / ref::dot_q4_0_q8, plus: xq is 16-byte
// aligned and up to 16 bytes past the row may be read.
float dot_q8_0_q8(const std::byte* row, const int8_t* xq, const float* xscale, std::size_t n);
float dot_q4_0_q8(const std::byte* row, const int8_t* xq, const float* xscale, std::size_t n);
// n_tok <= 8; xq_stride a multiple of 16.
void dot_q4_0_q8_multi(const std::byte* row, const int8_t* xq, std::size_t xq_stride, const float* xscale,
                       std::size_t xs_stride, int n_tok, std::size_t n, float* out);
void scan_q4_0_q8(const std::byte* rows, std::size_t n_rows, std::size_t n, const int8_t* xq, const float* xscale,
                  float* out);

}  // namespace lfm::math::esp32s3

#define LFM_ESP32S3_HAS_DOT_Q8_Q8 1
#define LFM_ESP32S3_HAS_DOT_Q4_Q8 1
#define LFM_ESP32S3_HAS_SCAN_Q4_Q8 1
#define LFM_ESP32S3_HAS_DOT_Q4_Q8_MULTI 1
