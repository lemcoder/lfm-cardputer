// Tests for lfm::math. Two kinds:
//   MathRef.*      hand-computed values: is the reference implementation right?
//   MathBackend.*  random inputs: does the dispatched (SIMD) op agree with ref::?
//
// Run one group:  ./host/build/lfm_tests --gtest_filter='MathRef.Rope'

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

#include "lfm/math/ops.hpp"

namespace m = lfm::math;
using Vec = std::vector<float>;

namespace {

// Relative tolerance: |got - want| <= tol * (1 + |want|).
// Returns an AssertionResult so failures name the first bad index.
::testing::AssertionResult SpanNear(std::span<const float> got, std::span<const float> want, float tol) {
    if (got.size() != want.size()) {
        return ::testing::AssertionFailure() << "size " << got.size() << " != " << want.size();
    }
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (!(std::fabs(got[i] - want[i]) <= tol * (1.0f + std::fabs(want[i])))) {
            return ::testing::AssertionFailure() << "[" << i << "] got " << got[i] << ", want " << want[i];
        }
    }
    return ::testing::AssertionSuccess();
}

Vec RandomVec(std::size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    Vec v(n);
    for (float& f : v) {
        f = dist(rng);
    }
    return v;
}

}  // namespace

// ---------------------------------------------------------------- reference

TEST(MathRef, Dot) {
    EXPECT_FLOAT_EQ(m::ref::dot(Vec{1, 2, 3}, Vec{4, 5, 6}), 32.0f);  // 4 + 10 + 18
    EXPECT_FLOAT_EQ(m::ref::dot(Vec{}, Vec{}), 0.0f);
}

TEST(MathRef, Matvec) {
    // W = [[1 2 3],
    //      [4 5 6]]   x = [1 0 -1]   ->   [1-3, 4-6] = [-2, -2]
    Vec w{1, 2, 3, 4, 5, 6}, x{1, 0, -1}, out(2, 99.0f);
    m::ref::matvec(out, {w.data(), 2, 3}, x);
    EXPECT_TRUE(SpanNear(out, Vec{-2, -2}, 1e-6f));
}

TEST(MathRef, Rmsnorm) {
    // mean(x^2) = (1+4+9+16)/4 = 7.5; 1/sqrt(7.5) = 0.3651484
    const Vec want{0.3651484f, 0.7302967f, 1.0954451f, 2.9211869f};
    Vec x{1, 2, 3, 4}, w{1, 1, 1, 2}, out(4);
    m::ref::rmsnorm(out, x, w, 0.0f);
    EXPECT_TRUE(SpanNear(out, want, 1e-6f));

    // In place (out aliases x).
    m::ref::rmsnorm(x, x, w, 0.0f);
    EXPECT_TRUE(SpanNear(x, want, 1e-6f));
}

TEST(MathRef, RmsnormZeroInputUsesEps) {
    // x = 0 must give 0, not NaN: that's what eps is for.
    Vec z{0, 0}, ones{1, 1}, out(2);
    m::ref::rmsnorm(out, z, ones, 1e-5f);
    EXPECT_TRUE(SpanNear(out, Vec{0, 0}, 0.0f));
}

TEST(MathRef, Add) {
    Vec acc{1, 2, 3};
    m::ref::add(acc, Vec{10, 20, 30});
    EXPECT_TRUE(SpanNear(acc, Vec{11, 22, 33}, 0.0f));
}

TEST(MathRef, Mul) {
    Vec out(3);
    m::ref::mul(out, Vec{1, 2, 3}, Vec{-1, 0.5f, 2});
    EXPECT_TRUE(SpanNear(out, Vec{-1, 1, 6}, 0.0f));
}

TEST(MathRef, Silu) {
    // silu(x) = x / (1 + e^-x)
    Vec x{0, 1, -1, 10};
    m::ref::silu(x);
    EXPECT_TRUE(SpanNear(x, Vec{0, 0.7310586f, -0.2689414f, 9.9995460f}, 1e-6f));
}

TEST(MathRef, Swiglu) {
    // silu(1)*2 = 1.4621172, silu(-1)*3 = -0.8068243
    Vec out(2);
    m::ref::swiglu(out, Vec{1, -1}, Vec{2, 3});
    EXPECT_TRUE(SpanNear(out, Vec{1.4621172f, -0.8068243f}, 1e-6f));
}

TEST(MathRef, Softmax) {
    Vec x{1, 2, 3};
    m::ref::softmax(x);
    EXPECT_TRUE(SpanNear(x, Vec{0.0900306f, 0.2447285f, 0.6652410f}, 1e-6f));
}

TEST(MathRef, SoftmaxLargeInputsDontOverflow) {
    // Same distribution as above: softmax is shift-invariant. Naive exp(1000) = inf.
    Vec x{1000, 1001, 1002};
    m::ref::softmax(x);
    EXPECT_TRUE(SpanNear(x, Vec{0.0900306f, 0.2447285f, 0.6652410f}, 1e-6f));
}

TEST(MathRef, SoftmaxMaskedEntriesAreZero) {
    Vec x{0, -INFINITY};
    m::ref::softmax(x);
    EXPECT_TRUE(SpanNear(x, Vec{1, 0}, 0.0f));
}

TEST(MathRef, Rope) {
    // head_dim 4 -> pairs (0,2) and (1,3), frequencies theta^0 = 1 and
    // theta^(-2/4) = 0.01 for theta = 10000. At pos 1 the angles are 1 and 0.01.
    //   out[i]       = x[i] * cos(a) - x[i + d/2] * sin(a)
    //   out[i + d/2] = x[i + d/2] * cos(a) + x[i] * sin(a)
    Vec x{1, 0, 0, 0,   // head 0: rotates the (0,2) pair by 1 rad
          0, 1, 0, 0};  // head 1: rotates the (1,3) pair by 0.01 rad
    m::ref::rope(x, 4, 1, 10000.0f);
    EXPECT_TRUE(SpanNear(x, Vec{0.5403023f, 0, 0.8414710f, 0,
                                0, 0.9999500f, 0, 0.0099998f}, 1e-6f));
}

TEST(MathRef, RopeAtPositionZeroIsIdentity) {
    Vec x{1, 2, 3, 4};
    m::ref::rope(x, 4, 0, 10000.0f);
    EXPECT_TRUE(SpanNear(x, Vec{1, 2, 3, 4}, 0.0f));
}

TEST(MathRef, Bf16ToF32) {
    std::vector<uint16_t> in{0x3F80, 0xC000, 0x4049, 0x0000};
    Vec out(4);
    m::ref::bf16_to_f32(out, in);
    EXPECT_TRUE(SpanNear(out, Vec{1.0f, -2.0f, 3.140625f, 0.0f}, 0.0f));
}

TEST(MathRef, Argmax) {
    EXPECT_EQ(m::ref::argmax(Vec{1, 5, 3}), 1u);
    EXPECT_EQ(m::ref::argmax(Vec{-3, -1, -2}), 1u);
    EXPECT_EQ(m::ref::argmax(Vec{2, 7, 7}), 1u) << "first index on ties";
}

// ------------------------------------------------- backend vs reference
//
// Parameterized over sizes. Odd sizes on purpose: SIMD code must handle the
// leftover elements that don't fill a whole vector register.

class MathBackend : public ::testing::TestWithParam<std::size_t> {};

TEST_P(MathBackend, DotMatchesRef) {
    const std::size_t n = GetParam();
    Vec a = RandomVec(n, 1), b = RandomVec(n, 2);
    EXPECT_NEAR(m::dot(a, b), m::ref::dot(a, b), 1e-4f * (1.0f + std::sqrt(static_cast<float>(n))));
}

TEST_P(MathBackend, MatvecMatchesRef) {
    const std::size_t rows = 37, cols = GetParam();
    Vec w = RandomVec(rows * cols, 3), x = RandomVec(cols, 4);
    Vec got(rows), want(rows);
    m::matvec(got, {w.data(), rows, cols}, x);
    m::ref::matvec(want, {w.data(), rows, cols}, x);
    EXPECT_TRUE(SpanNear(got, want, 1e-4f));
}

INSTANTIATE_TEST_SUITE_P(Sizes, MathBackend, ::testing::Values(1, 3, 4, 17, 1024, 1027, 2560));

// ------------------------------------------------- storage formats

TEST(MathRef, F16ToF32) {
    EXPECT_EQ(m::ref::f16_to_f32(0x3C00), 1.0f);
    EXPECT_EQ(m::ref::f16_to_f32(0xC000), -2.0f);
    EXPECT_EQ(m::ref::f16_to_f32(0x3555), 0.333251953125f);
    EXPECT_EQ(m::ref::f16_to_f32(0x7BFF), 65504.0f);              // largest normal
    EXPECT_EQ(m::ref::f16_to_f32(0x0001), 5.9604644775390625e-8f);  // smallest subnormal
    EXPECT_EQ(m::ref::f16_to_f32(0x8000), -0.0f);
    EXPECT_TRUE(std::isinf(m::ref::f16_to_f32(0x7C00)));
    EXPECT_TRUE(std::isnan(m::ref::f16_to_f32(0x7E00)));
}

TEST(MathRef, F32ToBf16RoundsToNearestEven) {
    EXPECT_EQ(m::ref::f32_to_bf16(1.0f), 0x3F80);
    EXPECT_EQ(m::ref::f32_to_bf16(-2.0f), 0xC000);
    EXPECT_EQ(m::ref::f32_to_bf16(3.14159265f), 0x4049);         // 3.140625 is nearest
    EXPECT_EQ(m::ref::f32_to_bf16(std::bit_cast<float>(0x3F808000u)), 0x3F80);  // tie -> even
    EXPECT_EQ(m::ref::f32_to_bf16(std::bit_cast<float>(0x3F818000u)), 0x3F82);  // tie -> even (up)
    EXPECT_TRUE(std::isnan(std::bit_cast<float>(uint32_t{m::ref::f32_to_bf16(NAN)} << 16)));
}

TEST(MathRef, DotBf16) {
    const uint16_t row[3] = {0x3F80, 0xC000, 0x4040};  // 1, -2, 3
    EXPECT_FLOAT_EQ(m::ref::dot_bf16(reinterpret_cast<const std::byte*>(row), Vec{1, 1, 2}), 1 - 2 + 6);
}

TEST(MathRef, DotQ8_0) {
    // Two blocks: scale 0.5 with q = i, scale 2 with q = -1.
    std::vector<std::byte> row(2 * 34);
    const uint16_t half = 0x3800, two = 0x4000;
    std::memcpy(row.data(), &half, 2);
    std::memcpy(row.data() + 34, &two, 2);
    for (int i = 0; i < 32; ++i) {
        row[2 + i] = static_cast<std::byte>(static_cast<int8_t>(i));
        row[36 + i] = static_cast<std::byte>(static_cast<int8_t>(-1));
    }
    const Vec x(64, 1.0f);
    // 0.5 * (0 + 1 + ... + 31) + 2 * (-32) = 248 - 64
    EXPECT_FLOAT_EQ(m::ref::dot_q8_0(row.data(), x), 184.0f);
}
