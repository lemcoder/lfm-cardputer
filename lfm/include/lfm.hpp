#pragma once

#include <array>
#include <cstdint>

namespace lfm {

// LFM2.5-230M hyperparameters, from LiquidAI/LFM2.5-230M config.json.
inline constexpr int kDim = 1024;
inline constexpr int kFfDim = 2560;  // SwiGLU, w1/w3 stored merged ("parallel_mlp_merged")
inline constexpr int kNumLayers = 14;
inline constexpr int kNumHeads = 16;
inline constexpr int kNumKvHeads = 8;
inline constexpr int kHeadDim = kDim / kNumHeads;  // 64
inline constexpr int kVocab = 65536;
inline constexpr int kConvLCache = 3;
inline constexpr float kRopeTheta = 1000000.0f;
inline constexpr float kNormEps = 1e-5f;
inline constexpr int32_t kBosId = 1;
inline constexpr int32_t kEosId = 7;

enum class LayerType : uint8_t { Conv, Attention };

inline constexpr std::array<LayerType, kNumLayers> kLayerTypes = {
    LayerType::Conv,      LayerType::Conv, LayerType::Attention, LayerType::Conv,
    LayerType::Attention, LayerType::Conv, LayerType::Attention, LayerType::Conv,
    LayerType::Attention, LayerType::Conv, LayerType::Attention, LayerType::Conv,
    LayerType::Attention, LayerType::Conv,
};

}  // namespace lfm
