#pragma once

// ESP32-S3 backend. The S3's PIE SIMD unit works on 128-bit registers of
// int8/int16 lanes (EE.VMULAS.S8.ACCX etc.), not floats, so the fast path here
// will be the quantized matvec. Reference: ESP32-S3 TRM, "Processor Instruction
// Extensions", and espressif/esp-dsp / esp-nn sources for examples.
//
// Same pattern as neon.hpp: declare, implement in lfm/src/math/esp32s3.cpp,
// then #define LFM_ESP32S3_HAS_<OP> 1.

#include <span>

#include "lfm/math/types.hpp"

namespace lfm::math::esp32s3 {

}  // namespace lfm::math::esp32s3
