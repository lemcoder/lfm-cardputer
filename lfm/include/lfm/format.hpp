#pragma once
// On-disk layout of a .lfm model file, written by tools/export.py.
// Little-endian, read with a single memcpy into FileHeader. Keep in sync with
// the HEADER_HEAD / LAYER_FIELDS definitions in export.py.

#include <cstddef>
#include <cstdint>

namespace lfm {

inline constexpr char kMagic[8] = {'L', 'F', 'M', '2', 'E', 'N', 'G', '\0'};
inline constexpr uint32_t kFormatVersion = 2;
inline constexpr int kMaxLayers = 16;

enum class Dtype : uint32_t {
    F32 = 0,   // float32
    BF16 = 1,  // bfloat16
    Q8_0 = 2,  // blocks of 32: {f16 scale, int8[32]}, w = scale * q
    Q4_0 = 3,  // blocks of 32: {f16 scale, uint8[16]}: byte j holds elements j (low
               // nibble) and j+16 (high nibble), w = scale * (nibble - 8)
};

inline constexpr std::size_t kQ8BlockSize = 32;
inline constexpr std::size_t kQ8BlockBytes = 2 + kQ8BlockSize;
inline constexpr std::size_t kQ4BlockSize = 32;
inline constexpr std::size_t kQ4BlockBytes = 2 + kQ4BlockSize / 2;

// Bytes per matrix row of `cols` elements.
constexpr std::size_t row_bytes(Dtype dt, std::size_t cols) {
    switch (dt) {
        case Dtype::F32: return cols * 4;
        case Dtype::BF16: return cols * 2;
        case Dtype::Q8_0: return cols / kQ8BlockSize * kQ8BlockBytes;
        case Dtype::Q4_0: return cols / kQ4BlockSize * kQ4BlockBytes;
    }
    return 0;
}

enum class LayerKind : uint32_t { Conv = 0, Attention = 1 };

// Absolute file offsets of one layer's tensors. Fields that don't apply to
// the layer's kind are 0. Norms and the conv kernel are F32; each matrix's
// encoding is in FileHeader::layer_dtype[layer][field index].
struct LayerOffsets {
    uint64_t operator_norm;  // [dim]
    uint64_t ffn_norm;       // [dim]
    uint64_t w1;             // [ff, dim]   gate
    uint64_t w2;             // [dim, ff]   down
    uint64_t w3;             // [ff, dim]   up
    uint64_t conv_in;        // [3*dim, dim] -> [B | C | x]
    uint64_t conv_kernel;    // [dim, conv_l] F32, oldest tap first
    uint64_t conv_out;       // [dim, dim]
    uint64_t q;              // [n_heads*head_dim, dim]
    uint64_t k;              // [n_kv*head_dim, dim]
    uint64_t v;              // [n_kv*head_dim, dim]
    uint64_t o;              // [dim, n_heads*head_dim]
    uint64_t q_norm;         // [head_dim]
    uint64_t k_norm;         // [head_dim]
};

// Field indices into LayerOffsets, for FileHeader::layer_dtype.
enum LayerField : int {
    kOperatorNorm, kFfnNorm, kW1, kW2, kW3, kConvIn, kConvKernel, kConvOut, kQ, kK, kV, kO, kQNorm, kKNorm,
    kNumLayerFields
};

struct FileHeader {
    char magic[8];
    uint32_t version;
    Dtype embed_dtype;  // embedding table = tied LM head
    uint32_t dim;
    uint32_t ff_dim;
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t vocab;
    uint32_t conv_l;
    uint32_t reserved[2];
    float norm_eps;
    float rope_theta;
    LayerKind layer_kind[kMaxLayers];
    uint64_t embed;        // [vocab, dim] embed_dtype; also the (tied) LM head
    uint64_t final_norm;   // [dim] F32
    uint64_t vocab_index;  // uint32[vocab + 1], offsets into vocab_bytes
    uint64_t vocab_bytes;
    LayerOffsets layers[kMaxLayers];
    uint8_t layer_dtype[kMaxLayers][16];  // Dtype of each LayerField (first kNumLayerFields used)
    // Optional low-precision copy of the embedding, used only to shortlist
    // the LM head's top candidates, which are then re-scored with `embed`.
    // 0 = none (scan `embed` directly).
    uint64_t lm_head;
    Dtype lm_head_dtype;
    uint32_t lm_head_candidates;

    Dtype dtype(int layer, LayerField f) const { return static_cast<Dtype>(layer_dtype[layer][f]); }
};

static_assert(sizeof(LayerOffsets) == kNumLayerFields * 8);
static_assert(offsetof(FileHeader, embed) == 128, "must match export.py HEADER_HEAD");
static_assert(sizeof(FileHeader) == 160 + kMaxLayers * (sizeof(LayerOffsets) + 16) + 16, "must match export.py HEADER_SIZE");

}  // namespace lfm
