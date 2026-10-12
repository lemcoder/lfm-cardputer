#pragma once
// On-disk layout of a .lfm model file, written by tools/export.py.
// Little-endian, read with a single memcpy into FileHeader. Keep in sync with
// the HEADER_HEAD / LAYER_FIELDS definitions in export.py.

#include <cstddef>
#include <cstdint>

namespace lfm {

inline constexpr char kMagic[8] = {'L', 'F', 'M', '2', 'E', 'N', 'G', '\0'};
inline constexpr uint32_t kFormatVersion = 1;
inline constexpr int kMaxLayers = 16;

enum class Dtype : uint32_t {
    F32 = 0,   // float32
    BF16 = 1,  // bfloat16
    Q8_0 = 2,  // blocks of 32: {f16 scale, int8[32]}, w = scale * q
};

inline constexpr std::size_t kQ8BlockSize = 32;
inline constexpr std::size_t kQ8BlockBytes = 2 + kQ8BlockSize;

// Bytes per matrix row of `cols` elements.
constexpr std::size_t row_bytes(Dtype dt, std::size_t cols) {
    switch (dt) {
        case Dtype::F32: return cols * 4;
        case Dtype::BF16: return cols * 2;
        case Dtype::Q8_0: return cols / kQ8BlockSize * kQ8BlockBytes;
    }
    return 0;
}

enum class LayerKind : uint32_t { Conv = 0, Attention = 1 };

// Absolute file offsets of one layer's tensors. Fields that don't apply to
// the layer's kind are 0. Norms and the conv kernel are F32; the rest use
// FileHeader::matrix_dtype.
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

struct FileHeader {
    char magic[8];
    uint32_t version;
    Dtype matrix_dtype;
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
    uint64_t embed;        // [vocab, dim] matrix_dtype; also the (tied) LM head
    uint64_t final_norm;   // [dim] F32
    uint64_t vocab_index;  // uint32[vocab + 1], offsets into vocab_bytes
    uint64_t vocab_bytes;
    LayerOffsets layers[kMaxLayers];
};

static_assert(offsetof(FileHeader, embed) == 128, "must match export.py HEADER_HEAD");
static_assert(sizeof(FileHeader) == 160 + kMaxLayers * sizeof(LayerOffsets), "must match export.py HEADER_SIZE");

}  // namespace lfm
