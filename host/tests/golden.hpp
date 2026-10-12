#pragma once
// Shared test helpers: the golden reference (tools/golden.py output) and the
// model weights, loaded once per test binary.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "lfm/io/safetensors.hpp"

namespace golden {

inline const std::string kDir = std::string(LFM_REPO_ROOT) + "/golden/default";

// One golden activation: row-major [rows, cols] float32 (rows = tokens).
struct Activation {
    std::vector<float> data;
    std::size_t rows = 0;
    std::size_t cols = 0;

    std::span<const float> row(std::size_t r) const { return {data.data() + r * cols, cols}; }
};

inline const nlohmann::json& Manifest() {
    static const nlohmann::json m = [] {
        std::ifstream f(kDir + "/manifest.json");
        return f ? nlohmann::json::parse(f) : nlohmann::json{};
    }();
    return m;
}

inline bool Available() { return Manifest().contains("weights"); }

// Load golden/default/<name>.f32 using the shape from the manifest.
inline Activation Load(const std::string& name) {
    const auto& entry = Manifest()["tensors"][name];
    Activation a;
    const auto& shape = entry["shape"];
    a.rows = shape.size() == 2 ? shape[0].get<std::size_t>() : 1;
    a.cols = shape.back().get<std::size_t>();
    a.data.resize(a.rows * a.cols);

    std::ifstream f(kDir + "/" + entry["file"].get<std::string>(), std::ios::binary);
    f.read(reinterpret_cast<char*>(a.data.data()), static_cast<std::streamsize>(a.data.size() * sizeof(float)));
    return a;
}

// The model file, opened once and shared by every test.
inline const lfm::io::Safetensors& Model() {
    static const std::unique_ptr<lfm::io::Safetensors> st = [] {
        std::string path = Available() ? Manifest()["weights"].get<std::string>() : "";
        return std::make_unique<lfm::io::Safetensors>(path.c_str());
    }();
    return *st;
}

inline std::vector<int> Tokens() { return Manifest()["tokens"].get<std::vector<int>>(); }

// Compare with a tolerance scaled by the vector's size: max|got-want| must be
// <= tol * max|want|. Per-element relative error is meaningless near zero.
inline ::testing::AssertionResult Near(std::span<const float> got, std::span<const float> want, float tol) {
    if (got.size() != want.size()) {
        return ::testing::AssertionFailure() << "size " << got.size() << " != " << want.size();
    }
    float scale = 0.0f, worst = 0.0f;
    std::size_t worst_i = 0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        if (!std::isfinite(got[i])) {
            return ::testing::AssertionFailure() << "[" << i << "] got " << got[i];
        }
        scale = std::fmax(scale, std::fabs(want[i]));
        const float d = std::fabs(got[i] - want[i]);
        if (d > worst) {
            worst = d;
            worst_i = i;
        }
    }
    if (worst <= tol * scale) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << "max diff " << worst << " at [" << worst_i << "] got " << got[worst_i]
                                         << " want " << want[worst_i] << " (allowed " << tol * scale << ")";
}

}  // namespace golden

// Skip a test when the golden dump isn't there (e.g. fresh clone).
#define GOLDEN_REQUIRED()                                                     \
    do {                                                                      \
        if (!golden::Available()) GTEST_SKIP() << "run tools/golden.py first"; \
    } while (0)
