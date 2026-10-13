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
    // Optional cheap copy of the LM head, used only to shortlist the top
    // `lm_head_candidates` tokens, which are then re-scored with `embed`.
    // 0 = none (scan `embed` directly). Two kinds:
    //   lm_rank == 0: [vocab, dim] low-precision copy of the embedding
    //   lm_rank  > 0: low rank: z = lm_proj . h ([rank, dim] matrix), then
    //                 approx logit[v] = lm_head row v . z ([vocab, rank])
    uint64_t lm_head;
    Dtype lm_head_dtype;
    uint32_t lm_head_candidates;
    uint64_t lm_proj;
    uint32_t lm_rank;
    uint32_t reserved_tail;

    Dtype dtype(int layer, LayerField f) const { return static_cast<Dtype>(layer_dtype[layer][f]); }
};

static_assert(sizeof(LayerOffsets) == kNumLayerFields * 8);
static_assert(offsetof(FileHeader, embed) == 128, "must match export.py HEADER_HEAD");
static_assert(sizeof(FileHeader) == 160 + kMaxLayers * (sizeof(LayerOffsets) + 16) + 32, "must match export.py HEADER_SIZE");

// LM head shortlist table for memory-mapped storage (the ESP32's internal
// flash), written by tools/flash_head.py. The engine scores every vocab row
// approximately from it (no SD reads) and re-scores the top `candidates`
// exactly from the .lfm embedding:
//   z = proj . h                (proj: [rank, dim], proj_dtype)
//   approx logit[v] = table row v . z   (table: [vocab, rank], table_dtype)
struct FlashHeadHeader {
    char magic[8];          // "LFMHEAD\0"
    uint32_t version;       // 1
    uint32_t vocab;
    uint32_t dim;
    uint32_t rank;
    uint32_t candidates;
    Dtype proj_dtype;
    Dtype table_dtype;
    uint32_t reserved;
    uint64_t proj_offset;   // from the start of the header
    uint64_t table_offset;
    // v2: exact embedding rows of tokens 0 .. cache_rows-1 (the most frequent:
    // the vocab is frequency-ordered), byte-identical to the .lfm embedding,
    // so the re-score can skip the SD card for them.
    uint64_t cache_offset;
    uint32_t cache_rows;
    Dtype cache_dtype;
};
static_assert(sizeof(FlashHeadHeader) == 72);
inline constexpr char kFlashHeadMagic[8] = {'L', 'F', 'M', 'H', 'E', 'A', 'D', '\0'};

}  // namespace lfm
