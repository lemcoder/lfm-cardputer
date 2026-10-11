#pragma once

// ARM NEON backend (Apple Silicon host). Practice ground for SIMD before the
// ESP32-S3: same ideas (lanes, fused multiply-add, horizontal sum), friendlier
// tooling. Intrinsics reference: <arm_neon.h>, e.g. vld1q_f32, vfmaq_f32, vaddvq_f32.
//
// To add an op:
//   1. declare it below and implement it in lfm/src/math/neon.cpp
//   2. #define LFM_NEON_HAS_<OP> 1 here
//   3. the tests automatically compare it against ref::

#include <span>

#include "lfm/math/types.hpp"

namespace lfm::math::neon {

// float dot(std::span<const float> a, std::span<const float> b);
// #define LFM_NEON_HAS_DOT 1

}  // namespace lfm::math::neon
