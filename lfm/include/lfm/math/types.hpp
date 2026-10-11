#pragma once

#include <cstddef>
#include <span>

namespace lfm::math {

// Non-owning view of a row-major matrix: element (r, c) is data[r * cols + c].
// Same layout as a PyTorch nn.Linear weight [out_features, in_features],
// so y = W x means out[r] = dot(row r, x).
struct MatRef {
    const float* data;
    std::size_t rows;
    std::size_t cols;

    std::span<const float> row(std::size_t r) const { return {data + r * cols, cols}; }
};

}  // namespace lfm::math
