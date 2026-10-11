#include "lfm/math/ref.hpp"

#include <cmath>
#include <cstring>

namespace lfm::math::ref {

float dot(std::span<const float> a, std::span<const float> b) {
    // TODO
    return 0.0f;
}

void matvec(std::span<float> out, MatRef w, std::span<const float> x) {
    // TODO
}

void rmsnorm(std::span<float> out, std::span<const float> x, std::span<const float> weight, float eps) {
    // TODO
}

void add(std::span<float> acc, std::span<const float> x) {
    // TODO
}

void mul(std::span<float> out, std::span<const float> a, std::span<const float> b) {
    // TODO
}

void silu(std::span<float> x) {
    // TODO
}

void swiglu(std::span<float> out, std::span<const float> gate, std::span<const float> up) {
    // TODO
}

void softmax(std::span<float> x) {
    // TODO
}

void rope(std::span<float> x, std::size_t head_dim, std::size_t pos, float theta) {
    // TODO
}

void bf16_to_f32(std::span<float> out, std::span<const uint16_t> in) {
    // TODO
}

std::size_t argmax(std::span<const float> x) {
    // TODO
    return 0;
}

}  // namespace lfm::math::ref
