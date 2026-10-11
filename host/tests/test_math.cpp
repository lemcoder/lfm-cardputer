// Tests for lfm::math. Two kinds:
//   ref_*      hand-computed values: is the reference implementation right?
//   backend_*  random inputs: does the dispatched (SIMD) op agree with ref::?

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "check.hpp"
#include "lfm/math/ops.hpp"

namespace m = lfm::math;
using Vec = std::vector<float>;

// ---------------------------------------------------------------- reference

TEST(ref_dot) {
    Vec a{1, 2, 3}, b{4, 5, 6};
    CHECK_NEAR(m::ref::dot(a, b), 32.0f, 1e-6f);  // 4 + 10 + 18
    CHECK_NEAR(m::ref::dot(Vec{}, Vec{}), 0.0f, 0.0f);
}

TEST(ref_matvec) {
    // W = [[1 2 3],
    //      [4 5 6]]   x = [1 0 -1]   ->   [1-3, 4-6] = [-2, -2]
    Vec w{1, 2, 3, 4, 5, 6}, x{1, 0, -1}, out(2, 99.0f);
    m::ref::matvec(out, {w.data(), 2, 3}, x);
    CHECK_SPAN_NEAR(out, (Vec{-2, -2}), 1e-6f);
}

TEST(ref_rmsnorm) {
    // mean(x^2) = (1+4+9+16)/4 = 7.5; 1/sqrt(7.5) = 0.3651484
    Vec x{1, 2, 3, 4}, w{1, 1, 1, 2}, out(4);
    m::ref::rmsnorm(out, x, w, 0.0f);
    CHECK_SPAN_NEAR(out, (Vec{0.3651484f, 0.7302967f, 1.0954451f, 2.9211869f}), 1e-6f);

    // eps matters when x is tiny: x = 0 must give 0, not NaN.
    Vec z{0, 0}, ones{1, 1}, zout(2);
    m::ref::rmsnorm(zout, z, ones, 1e-5f);
    CHECK_SPAN_NEAR(zout, (Vec{0, 0}), 0.0f);

    // in place (out aliases x)
    m::ref::rmsnorm(x, x, w, 0.0f);
    CHECK_SPAN_NEAR(x, (Vec{0.3651484f, 0.7302967f, 1.0954451f, 2.9211869f}), 1e-6f);
}

TEST(ref_add_mul) {
    Vec acc{1, 2, 3};
    m::ref::add(acc, Vec{10, 20, 30});
    CHECK_SPAN_NEAR(acc, (Vec{11, 22, 33}), 0.0f);

    Vec out(3);
    m::ref::mul(out, Vec{1, 2, 3}, Vec{-1, 0.5f, 2});
    CHECK_SPAN_NEAR(out, (Vec{-1, 1, 6}), 0.0f);
}

TEST(ref_silu) {
    // silu(x) = x / (1 + e^-x)
    Vec x{0, 1, -1, 10};
    m::ref::silu(x);
    CHECK_SPAN_NEAR(x, (Vec{0, 0.7310586f, -0.2689414f, 9.9995460f}), 1e-6f);
}

TEST(ref_swiglu) {
    // silu(1)*2 = 1.4621172, silu(-1)*3 = -0.8068243
    Vec out(2);
    m::ref::swiglu(out, Vec{1, -1}, Vec{2, 3});
    CHECK_SPAN_NEAR(out, (Vec{1.4621172f, -0.8068243f}), 1e-6f);
}

TEST(ref_softmax) {
    Vec x{1, 2, 3};
    m::ref::softmax(x);
    CHECK_SPAN_NEAR(x, (Vec{0.0900306f, 0.2447285f, 0.6652410f}), 1e-6f);

    // Same distribution: softmax is shift-invariant. Naive exp(1000) overflows.
    Vec big{1000, 1001, 1002};
    m::ref::softmax(big);
    CHECK_SPAN_NEAR(big, (Vec{0.0900306f, 0.2447285f, 0.6652410f}), 1e-6f);

    // Masked-out (-inf) entries get exactly 0.
    Vec masked{0, -INFINITY};
    m::ref::softmax(masked);
    CHECK_SPAN_NEAR(masked, (Vec{1, 0}), 0.0f);
}

TEST(ref_rope) {
    // head_dim 4 -> pairs (0,2) and (1,3), frequencies theta^0 = 1 and
    // theta^(-2/4) = 0.01 for theta = 10000. At pos 1 the angles are 1 and 0.01.
    //   out[i]       = x[i] * cos(a) - x[i + d/2] * sin(a)
    //   out[i + d/2] = x[i + d/2] * cos(a) + x[i] * sin(a)
    Vec x{1, 0, 0, 0,   // head 0: rotates the (0,2) pair by 1 rad
          0, 1, 0, 0};  // head 1: rotates the (1,3) pair by 0.01 rad
    m::ref::rope(x, 4, 1, 10000.0f);
    CHECK_SPAN_NEAR(x, (Vec{0.5403023f, 0, 0.8414710f, 0,
                            0, 0.9999500f, 0, 0.0099998f}), 1e-6f);

    // pos 0 is the identity.
    Vec y{1, 2, 3, 4};
    m::ref::rope(y, 4, 0, 10000.0f);
    CHECK_SPAN_NEAR(y, (Vec{1, 2, 3, 4}), 0.0f);
}

TEST(ref_bf16_to_f32) {
    std::vector<uint16_t> in{0x3F80, 0xC000, 0x4049, 0x0000};
    Vec out(4);
    m::ref::bf16_to_f32(out, in);
    CHECK_SPAN_NEAR(out, (Vec{1.0f, -2.0f, 3.140625f, 0.0f}), 0.0f);
}

TEST(ref_argmax) {
    CHECK(m::ref::argmax(Vec{1, 5, 3}) == 1);
    CHECK(m::ref::argmax(Vec{-3, -1, -2}) == 1);
    CHECK(m::ref::argmax(Vec{2, 7, 7}) == 1);  // first on ties
}

// ------------------------------------------------- backend vs reference

namespace {
Vec random_vec(std::size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    Vec v(n);
    for (float& f : v) {
        f = dist(rng);
    }
    return v;
}
}  // namespace

// Odd sizes on purpose: SIMD code must handle the leftover elements that don't
// fill a whole vector register.
TEST(backend_dot) {
    for (std::size_t n : {1u, 3u, 4u, 17u, 1024u, 2560u}) {
        Vec a = random_vec(n, 1), b = random_vec(n, 2);
        CHECK_NEAR(m::dot(a, b), m::ref::dot(a, b), 1e-4f);
    }
}

TEST(backend_matvec) {
    const std::size_t rows = 37, cols = 1027;
    Vec w = random_vec(rows * cols, 3), x = random_vec(cols, 4);
    Vec got(rows), want(rows);
    m::matvec(got, {w.data(), rows, cols}, x);
    m::ref::matvec(want, {w.data(), rows, cols}, x);
    CHECK_SPAN_NEAR(got, want, 1e-4f);
}
