#include "lfm/engine.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "lfm/math/ops.hpp"

namespace lfm {

namespace m = math;

const char* to_string(Status s) {
    switch (s) {
        case Status::Ok: return "ok";
        case Status::IoError: return "I/O error";
        case Status::BadMagic: return "not an .lfm file";
        case Status::BadVersion: return "unsupported .lfm version";
        case Status::Unsupported: return "unsupported model config";
        case Status::OutOfMemory: return "out of memory";
        case Status::KvFileError: return "can't create KV cache / history file";
    }
    return "?";
}

namespace {

// Sizes derived from the header, in floats unless noted.
struct Dims {
    std::size_t dim, ff, heads, kv_heads, head_dim, q_dim, kv_dim, conv_l, vocab;

    explicit Dims(const FileHeader& h)
        : dim(h.dim), ff(h.ff_dim), heads(h.n_heads), kv_heads(h.n_kv_heads), head_dim(h.head_dim),
          q_dim(h.n_heads * h.head_dim), kv_dim(h.n_kv_heads * h.head_dim), conv_l(h.conv_l), vocab(h.vocab) {}

    // Per-token scratch (one slot per batch entry; only one op runs at a
    // time). Independent of the context length: attention streams the KV
    // cache with an online softmax.
    std::size_t slot() const {
        const std::size_t conv = 2 * dim;              // in_proj's B and C (X goes to tmp_)
        const std::size_t mlp = ff;                    // gate, then silu(gate) * up in place
        const std::size_t attn = q_dim + 2 * kv_dim;   // q, k, v (ctx goes to xn_)
        return std::max({conv, mlp, attn});
    }
    // Scratch shared by the whole batch.
    std::size_t shared() const {
        return std::max(dim * conv_l,                // conv kernel
                        2 * heads + 4 * head_dim);   // running max/sum, q/k norm, one K and V row
    }
    // All activation + scratch floats for a batch of `batch` tokens.
    std::size_t floats(int batch) const {
        return static_cast<std::size_t>(batch) * (3 * dim + slot()) + dim + shared();
    }
};

bool is_matrix_dtype(Dtype dt) {
    return dt == Dtype::BF16 || dt == Dtype::Q8_0 || dt == Dtype::Q4_0;
}

float dot_row(Dtype dt, const std::byte* row, std::span<const float> x) {
    switch (dt) {
        case Dtype::Q8_0: return m::dot_q8_0(row, x);
        case Dtype::Q4_0: return m::dot_q4_0(row, x);
        default: return m::dot_bf16(row, x);
    }
}

void dequant_row(Dtype dt, const std::byte* row, std::span<float> out) {
    if (dt == Dtype::BF16) {
        for (std::size_t i = 0; i < out.size(); ++i) {
            uint16_t h;
            std::memcpy(&h, row + 2 * i, 2);
            out[i] = std::bit_cast<float>(static_cast<uint32_t>(h) << 16);
        }
        return;
    }
    if (dt == Dtype::Q4_0) {
        for (std::size_t b = 0; b < out.size() / kQ4BlockSize; ++b) {
            const std::byte* block = row + b * kQ4BlockBytes;
            uint16_t scale_bits;
            std::memcpy(&scale_bits, block, 2);
            const float scale = m::f16_to_f32(scale_bits);
            const auto* q = reinterpret_cast<const uint8_t*>(block + 2);
            float* o = out.data() + b * kQ4BlockSize;
            for (std::size_t j = 0; j < kQ4BlockSize / 2; ++j) {
                o[j] = scale * static_cast<float>(static_cast<int>(q[j] & 0x0F) - 8);
                o[j + kQ4BlockSize / 2] = scale * static_cast<float>(static_cast<int>(q[j] >> 4) - 8);
            }
        }
        return;
    }
    // Q8_0
    for (std::size_t b = 0; b < out.size() / kQ8BlockSize; ++b) {
        const std::byte* block = row + b * kQ8BlockBytes;
        uint16_t scale_bits;
        std::memcpy(&scale_bits, block, 2);
        const float scale = m::f16_to_f32(scale_bits);
        const auto* q = reinterpret_cast<const int8_t*>(block + 2);
        for (std::size_t i = 0; i < kQ8BlockSize; ++i) {
            out[b * kQ8BlockSize + i] = scale * static_cast<float>(q[i]);
        }
    }
}

}  // namespace

Engine::~Engine() {
    for (int i = 0; i < n_blocks_; ++i) std::free(blocks_[i]);
    for (float* p : conv_state_) std::free(p);
    for (std::byte* p : kv_ram_) std::free(p);
    delete kv_file_;
    std::free(hist_ram_);
    delete hist_file_;
    delete state_file_;
}

// 16-byte aligned (SIMD loads), and 16 bytes of slack at the end: the SIMD
// kernels may read up to 16 bytes past the last weight row in a buffer.
void* Engine::alloc(std::size_t bytes) {
    assert(n_blocks_ < kMaxBlocks);
    void* p = nullptr;
    if (posix_memalign(&p, 16, bytes + 16) != 0) p = nullptr;  // std::aligned_alloc isn't in ESP-IDF's libstdc++
    if (p != nullptr) blocks_[n_blocks_++] = p;
    return p;
}

std::size_t Engine::ram_bytes(const EngineConfig& cfg) const {
    const Dims d(h_);
    const std::size_t kv = cfg.kv_path != nullptr ? 0
        : static_cast<std::size_t>(n_attn_) * static_cast<std::size_t>(cfg.max_ctx) * 2 * d.kv_dim * (cfg.kv_bf16 ? 2 : 4);
    const std::size_t xq = cfg.int8_activations
        ? static_cast<std::size_t>(cfg.max_batch) * std::max({d.dim, d.ff, d.q_dim}) * (1 + sizeof(float) / 32.0)
        : 0;
    return cfg.io_bytes * (cfg.double_buffer ? 2 : 1) + row_bytes(Dtype::BF16, d.ff) + xq + d.floats(cfg.max_batch) * sizeof(float)
           + (cfg.conv_state_path != nullptr ? 0 : static_cast<std::size_t>(n_conv_) * (d.conv_l - 1) * d.dim * sizeof(float))
           + kv;
}

Status Engine::open(Source& src, const EngineConfig& cfg) {
    assert(n_blocks_ == 0 && "open() called twice");
    src_ = &src;
    cfg_ = cfg;

    if (!src.read_at(0, &h_, sizeof(h_))) return Status::IoError;
    if (std::memcmp(h_.magic, kMagic, sizeof(kMagic)) != 0) return Status::BadMagic;
    if (h_.version == 1) {
        // v1: one encoding for every matrix (in the field v2 calls embed_dtype);
        // the per-matrix table and LM head shortlist didn't exist (padding).
        for (auto& row : h_.layer_dtype) {
            for (auto& dt : row) dt = static_cast<uint8_t>(h_.embed_dtype);
        }
        h_.lm_head = 0;
        h_.lm_proj = 0;
        h_.lm_rank = 0;
    } else if (h_.version != kFormatVersion) {
        return Status::BadVersion;
    }

    const Dims d(h_);
    bool ok = is_matrix_dtype(h_.embed_dtype) &&
              (h_.lm_head == 0 || (is_matrix_dtype(h_.lm_head_dtype) && h_.lm_head_candidates >= 1 &&
                                   h_.lm_head_candidates <= kMaxCandidates)) &&
              (h_.lm_rank == 0 || (h_.lm_proj != 0 && h_.lm_rank % 32 == 0 && h_.lm_rank <= kTopKOffset)) &&
              h_.n_layers >= 1 && h_.n_layers <= static_cast<uint32_t>(kMaxLayers) &&
              d.dim % kQ8BlockSize == 0 && d.ff % kQ8BlockSize == 0 && d.q_dim % kQ8BlockSize == 0 &&
              d.heads % d.kv_heads == 0 && d.head_dim % 2 == 0 && d.conv_l >= 2 &&
              row_bytes(Dtype::BF16, d.ff) <= cfg.io_bytes &&  // widest row we stream
              2 * d.kv_dim * sizeof(float) <= cfg.io_bytes &&
              (d.conv_l - 1) * d.dim * sizeof(float) <= cfg.io_bytes &&  // conv state staged in io_
              d.q_dim <= d.dim &&                                         // attention ctx reuses xn_ cfg.max_ctx >= 1 && cfg.max_batch >= 1 &&
              cfg.max_batch <= kMaxBatch;
    for (uint32_t i = 0; ok && i < h_.n_layers; ++i) {
        const bool conv = h_.layer_kind[i] == LayerKind::Conv;
        for (LayerField f : {kW1, kW2, kW3}) ok = ok && is_matrix_dtype(h_.dtype(i, f));
        if (conv) {
            for (LayerField f : {kConvIn, kConvOut}) ok = ok && is_matrix_dtype(h_.dtype(i, f));
        } else {
            for (LayerField f : {kQ, kK, kV, kO}) ok = ok && is_matrix_dtype(h_.dtype(i, f));
        }
    }
    if (!ok) return Status::Unsupported;

    n_conv_ = n_attn_ = 0;
    for (uint32_t i = 0; i < h_.n_layers; ++i) {
        layer_slot_[i] = h_.layer_kind[i] == LayerKind::Conv ? n_conv_++ : n_attn_++;
    }

    // Biggest blocks first.
    io_ = static_cast<std::byte*>(alloc(cfg.io_bytes));
    io2_ = cfg.double_buffer ? static_cast<std::byte*>(alloc(cfg.io_bytes)) : io_;
    const std::size_t B = static_cast<std::size_t>(cfg.max_batch);
    slot_floats_ = std::max(d.slot(), kTopKOffset + 2 * kMaxCandidates);  // LM head top-k lives here too
    for (std::size_t b = 0; b < B; ++b) {
        slots_[b] = static_cast<float*>(alloc(slot_floats_ * sizeof(float)));
        if (slots_[b] == nullptr) return Status::OutOfMemory;
    }
    for (std::size_t b = 0; b < B; ++b) {
        x_[b] = static_cast<float*>(alloc(d.dim * sizeof(float)));
        xn_[b] = static_cast<float*>(alloc(d.dim * sizeof(float)));
        tmp_[b] = static_cast<float*>(alloc(d.dim * sizeof(float)));
        if (x_[b] == nullptr || xn_[b] == nullptr || tmp_[b] == nullptr) return Status::OutOfMemory;
    }
    norm_w_ = static_cast<float*>(alloc((d.dim + d.shared()) * sizeof(float)));
    carry_ = static_cast<std::byte*>(alloc(row_bytes(Dtype::BF16, d.ff)));  // widest row we stream
    if (cfg.int8_activations) {
        xq_cols_ = std::max({d.dim, d.ff, d.q_dim});
        xq_ = static_cast<int8_t*>(alloc(B * xq_cols_));
        xs_ = static_cast<float*>(alloc(B * (xq_cols_ / 32) * sizeof(float)));
        if (xq_ == nullptr || xs_ == nullptr) return Status::OutOfMemory;
    }
    if (io_ == nullptr || io2_ == nullptr || norm_w_ == nullptr || carry_ == nullptr) {
        return Status::OutOfMemory;
    }
    shared_ = norm_w_ + d.dim;

    if (n_conv_ > 0) {
        if (cfg.conv_state_path != nullptr) {
            state_file_ = new FileSource(cfg.conv_state_path, FileSource::Mode::ReadWrite);
            if (!state_file_->ok()) return Status::KvFileError;
        }
        // One block per conv layer: small blocks fit a fragmented heap.
        for (int i = 0; state_file_ == nullptr && i < n_conv_; ++i) {
            conv_state_[i] = static_cast<float*>(std::malloc((d.conv_l - 1) * d.dim * sizeof(float)));
            if (conv_state_[i] == nullptr) return Status::OutOfMemory;
        }
    }
    if (cfg.speculative && n_conv_ > 0) {
        if (cfg.conv_history_path != nullptr) {
            hist_file_ = new FileSource(cfg.conv_history_path, FileSource::Mode::ReadWrite);
            if (!hist_file_->ok()) return Status::KvFileError;
        } else {
            hist_ram_ = static_cast<float*>(std::malloc(static_cast<std::size_t>(n_conv_) * hist_rows() * d.dim * sizeof(float)));
            if (hist_ram_ == nullptr) return Status::OutOfMemory;
        }
    }
    if (cfg.kv_path != nullptr) {
        kv_file_ = new FileSource(cfg.kv_path, FileSource::Mode::ReadWrite);
        if (!kv_file_->ok()) return Status::KvFileError;
    } else {
        // One block per attention layer: smaller allocations fit a fragmented heap.
        for (int i = 0; i < n_attn_; ++i) {
            kv_ram_[i] = static_cast<std::byte*>(std::malloc(static_cast<std::size_t>(cfg.max_ctx) * kv_row_bytes()));
            if (kv_ram_[i] == nullptr) return Status::OutOfMemory;
        }
    }

    reset();
    return Status::Ok;
}

void Engine::reset() {
    pos_ = 0;
    pending_n_ = 0;
    const std::size_t state_floats = (h_.conv_l - 1) * h_.dim;
    for (int i = 0; i < n_conv_; ++i) {
        if (conv_state_[i] != nullptr) std::memset(conv_state_[i], 0, state_floats * sizeof(float));
    }
    if (state_file_ != nullptr) {
        std::memset(io_, 0, state_floats * sizeof(float));
        for (int i = 0; i < n_conv_; ++i) state_store(i, reinterpret_cast<const float*>(io_));
    }
}

void Engine::progress(int done) {
    if (progress_hook_ != nullptr) progress_hook_(progress_ctx_, done, static_cast<int>(h_.n_layers) + 1);
}

// ---------------------------------------------------------------- weight I/O

bool Engine::read_f32(uint64_t offset, std::span<float> out) {
    return src_->read_at(offset, out.data(), out.size_bytes());
}

// Reads fixed io_bytes blocks (tensors start 4 KB-aligned in the file, so
// every block starts on an SD sector), regardless of where rows end: a read
// that starts mid-sector costs the card an extra command. A row split across
// two blocks is stitched together in carry_. With two buffers, the next
// block loads while fn() runs on this one.
template <typename Fn>
bool Engine::stream_rows(uint64_t offset, std::size_t rows, std::size_t rb, Fn&& fn) {
    const std::size_t total = rows * rb;
    const std::size_t block = cfg_.io_bytes;
    std::byte* buf[2] = {io_, io2_};
    const bool two = io2_ != io_;
    int cur = 0;
    std::size_t row = 0, carry_len = 0;

    std::size_t len = std::min(block, total);
    src_->start_read(offset, buf[cur], len);
    for (std::size_t done = 0; done < total;) {
        if (!src_->wait()) return false;
        const std::size_t next = done + len;
        const std::size_t next_len = std::min(block, total - next);
        if (two && next_len > 0) src_->start_read(offset + next, buf[cur ^ 1], next_len);

        const std::byte* p = buf[cur];
        std::size_t avail = len;
        if (carry_len > 0) {  // finish the row that started in the previous block
            const std::size_t need = rb - carry_len;
            std::memcpy(carry_ + carry_len, p, need);
            fn(row++, carry_);
            p += need;
            avail -= need;
            carry_len = 0;
        }
        for (; avail >= rb; p += rb, avail -= rb) fn(row++, p);
        if (avail > 0) {  // start of a row that continues in the next block
            std::memcpy(carry_, p, avail);
            carry_len = avail;
        }

        if (!two && next_len > 0) {
            src_->start_read(offset + next, buf[cur], next_len);
        } else if (two) {
            cur ^= 1;
        }
        done = next;
        len = next_len;
    }
    assert(row == rows && carry_len == 0);
    return true;
}

// Each streamed row is used by all n tokens before the next one: one read of
// the matrix per batch.
Engine::Ptrs Engine::slots(std::size_t offset) const {
    Ptrs r{};
    for (int b = 0; b < cfg_.max_batch; ++b) r.p[b] = slots_[b] + offset;
    return r;
}

Engine::Ptrs Engine::acts(float* const* base) const {
    Ptrs r{};
    for (int b = 0; b < cfg_.max_batch; ++b) r.p[b] = base[b];
    return r;
}

bool Engine::matvec(const Ptrs& out, std::size_t rows, uint64_t offset, Dtype dt, std::size_t cols, const Ptrs& x,
                    int n) {
    if (xq_ != nullptr && (dt == Dtype::Q8_0 || dt == Dtype::Q4_0)) {
        // Integer path: quantize each token's input once, then int8 x int8 dots.
        for (int b = 0; b < n; ++b) m::quantize_q8({x.p[b], cols}, xq_ + b * xq_cols_, xs_ + b * (xq_cols_ / 32));
        return stream_rows(offset, rows, row_bytes(dt, cols), [&](std::size_t r, const std::byte* row) {
            for (int b = 0; b < n; ++b) {
                const int8_t* q = xq_ + b * xq_cols_;
                const float* sc = xs_ + b * (xq_cols_ / 32);
                out.p[b][r] = dt == Dtype::Q8_0 ? m::dot_q8_0_q8(row, q, sc, cols) : m::dot_q4_0_q8(row, q, sc, cols);
            }
        });
    }
    return stream_rows(offset, rows, row_bytes(dt, cols), [&](std::size_t r, const std::byte* row) {
        for (int b = 0; b < n; ++b) out.p[b][r] = dot_row(dt, row, {x.p[b], cols});
    });
}

// Tied LM head: logit[v] = embed row v · x, over the whole vocab, streamed.
// Without `logits`, only the argmax is kept (all logits would be 256 KB).
//
// If the file has a low-precision copy (h_.lm_head), it is scanned instead
// and only its top `lm_head_candidates` are re-scored with the exact rows:
// same top-1 (unless the true winner falls outside the shortlist), about half
// the bytes. With `logits` requested, the exact full scan is used.
bool Engine::set_flash_head(const void* data, std::size_t size) {
    fh_proj_ = fh_table_ = nullptr;
    if (data == nullptr || size < sizeof(FlashHeadHeader)) return false;
    std::memcpy(&fh_, data, sizeof(fh_));
    const bool ok = std::memcmp(fh_.magic, kFlashHeadMagic, sizeof(kFlashHeadMagic)) == 0 && fh_.version == 2 &&
                    fh_.vocab == h_.vocab && fh_.dim == h_.dim && fh_.rank % 32 == 0 &&
                    fh_.rank <= kTopKOffset && fh_.candidates >= 1 && fh_.candidates <= kMaxCandidates &&
                    is_matrix_dtype(fh_.proj_dtype) && is_matrix_dtype(fh_.table_dtype) &&
                    fh_.proj_offset + fh_.rank * row_bytes(fh_.proj_dtype, fh_.dim) <= size &&
                    fh_.table_offset + static_cast<uint64_t>(fh_.vocab) * row_bytes(fh_.table_dtype, fh_.rank) <= size &&
                    (fh_.cache_rows == 0 ||
                     (fh_.cache_dtype == h_.embed_dtype && fh_.cache_rows <= fh_.vocab &&
                      fh_.cache_offset + static_cast<uint64_t>(fh_.cache_rows) * row_bytes(fh_.cache_dtype, fh_.dim) <= size));
    if (!ok) return false;
    fh_cache_ = fh_.cache_rows > 0 ? static_cast<const std::byte*>(data) + fh_.cache_offset : nullptr;
    fh_proj_ = static_cast<const std::byte*>(data) + fh_.proj_offset;
    fh_table_ = static_cast<const std::byte*>(data) + fh_.table_offset;
    return true;
}

// Integer path helpers shared by the LM head variants.
void Engine::quantize_inputs(const Ptrs& in, int n, std::size_t cols) {
    for (int b = 0; b < n; ++b) m::quantize_q8({in.p[b], cols}, xq_ + b * xq_cols_, xs_ + b * (xq_cols_ / 32));
}

float Engine::score_row(int b, Dtype dt, const std::byte* row, const float* in, std::size_t cols) const {
    const int8_t* q = xq_ + b * xq_cols_;
    const float* sc = xs_ + b * (xq_cols_ / 32);
    if (xq_ != nullptr && dt == Dtype::Q8_0) return m::dot_q8_0_q8(row, q, sc, cols);
    if (xq_ != nullptr && dt == Dtype::Q4_0) return m::dot_q4_0_q8(row, q, sc, cols);
    return dot_row(dt, row, {in, cols});
}

// Exact re-score of the shortlists ids[b][0..count[b]) from the embedding.
// The lists are merged into one ascending, de-duplicated list (in io2_, idle
// during the LM head); every row read is scored for every token, so each
// token takes the best over the union. Nearby rows are fetched together in
// one sector-aligned read: a separate small read costs ~0.5 ms (rows straddle
// 512-byte sectors), about the same as reading 4 KB more. xs must be quantized.
bool Engine::rescore(const Ptrs& xs, int n, int32_t* const* ids, const int* count, int32_t* out) {
    const std::size_t erb = row_bytes(h_.embed_dtype, h_.dim);
    constexpr uint64_t kSector = 512;
    constexpr uint64_t kMaxGap = 4096;  // bytes worth reading through instead of a new command (16 KB measured worse)

    // Sort each list, then merge into the union.
    int head[kMaxBatch] = {};
    for (int b = 0; b < n; ++b) {
        int32_t* id = ids[b];
        for (int i = 1; i < count[b]; ++i) {
            const int32_t v = id[i];
            int j = i;
            for (; j > 0 && id[j - 1] > v; --j) id[j] = id[j - 1];
            id[j] = v;
        }
    }
    auto* uni = reinterpret_cast<int32_t*>(io2_ != io_ ? io2_ : carry_);
    const int cap = static_cast<int>((io2_ != io_ ? cfg_.io_bytes : row_bytes(Dtype::BF16, h_.ff_dim)) / sizeof(int32_t));
    int n_uni = 0;
    while (n_uni < cap) {
        int32_t next = INT32_MAX;
        for (int b = 0; b < n; ++b) {
            if (head[b] < count[b] && ids[b][head[b]] < next) next = ids[b][head[b]];
        }
        if (next == INT32_MAX) break;
        for (int b = 0; b < n; ++b) {
            if (head[b] < count[b] && ids[b][head[b]] == next) ++head[b];
        }
        uni[n_uni++] = next;
    }

    float best[kMaxBatch];
    for (int b = 0; b < n; ++b) {
        best[b] = -INFINITY;
        out[b] = n_uni > 0 ? uni[0] : 0;
    }
    auto row_off = [&](int32_t id) { return h_.embed + static_cast<uint64_t>(id) * erb; };
    auto score_all = [&](const std::byte* row, int32_t id) {
        for (int b = 0; b < n; ++b) {
            const float v = score_row(b, h_.embed_dtype, row, xs.p[b], h_.dim);
            if (v > best[b]) {  // ascending ids: strict > keeps the lowest id on ties
                best[b] = v;
                out[b] = id;
            }
        }
    };
    int i = 0;
    // Frequent tokens: exact rows cached in the flash head, no SD read.
    for (; i < n_uni && fh_cache_ != nullptr && static_cast<uint32_t>(uni[i]) < fh_.cache_rows; ++i) {
        score_all(fh_cache_ + static_cast<std::size_t>(uni[i]) * erb, uni[i]);
    }
    while (i < n_uni) {
        // Span [start, end): sector-aligned, as many following rows as fit
        // in io_ without a gap over kMaxGap.
        const uint64_t start = row_off(uni[i]) / kSector * kSector;
        uint64_t end = (row_off(uni[i]) + erb + kSector - 1) / kSector * kSector;
        int j = i + 1;
        for (; j < n_uni; ++j) {
            const uint64_t e = (row_off(uni[j]) + erb + kSector - 1) / kSector * kSector;
            if (e - start > cfg_.io_bytes || row_off(uni[j]) > end + kMaxGap) break;
            end = e;
        }
        if (!src_->read_at(start, io_, static_cast<std::size_t>(end - start))) return false;
        ++stats_.rescore_reads;
        for (; i < j; ++i) score_all(io_ + (row_off(uni[i]) - start), uni[i]);
    }
    return true;
}

// Top-k of (score, id) as a min-heap: the root is the worst kept entry, so a
// new candidate costs one compare, and an insert O(log k). (A sorted array
// cost ~0.13 s per 65536-row scan at k = 1024.) Lower score is worse; on equal
// scores the higher id is worse, so ties keep the lower id, like argmax.
// The result is unordered: callers sort ids or take the single best.
static inline bool topk_worse(float a, int32_t ai, float b, int32_t bi) { return a < b || (a == b && ai > bi); }

static inline void topk_insert(float* hs, int32_t* hi, int& size, int k, float v, int32_t id) {
    int i;
    if (size < k) {
        i = size++;
        while (i > 0) {  // sift up
            const int p = (i - 1) / 2;
            if (!topk_worse(v, id, hs[p], hi[p])) break;
            hs[i] = hs[p];
            hi[i] = hi[p];
            i = p;
        }
    } else {
        if (!topk_worse(hs[0], hi[0], v, id)) return;  // not better than the worst kept
        i = 0;
        while (true) {  // sift down from the root
            int c = 2 * i + 1;
            if (c >= size) break;
            if (c + 1 < size && topk_worse(hs[c + 1], hi[c + 1], hs[c], hi[c])) ++c;
            if (!topk_worse(hs[c], hi[c], v, id)) break;
            hs[i] = hs[c];
            hi[i] = hi[c];
            i = c;
        }
    }
    hs[i] = v;
    hi[i] = id;
}

// LM head from the memory-mapped flash table: no SD reads except the
// exact re-score of the shortlist.
bool Engine::lm_head_flash(const Ptrs& xs, int n, int32_t* out) {
    const std::size_t rank = fh_.rank;
    const std::size_t prb = row_bytes(fh_.proj_dtype, h_.dim);
    const std::size_t trb = row_bytes(fh_.table_dtype, rank);
    const int k = static_cast<int>(fh_.candidates);

    const auto t_scan = std::chrono::steady_clock::now();
    Ptrs z = slots(0);  // z at the start of each token's slot, top-k after kTopKOffset
    if (xq_ != nullptr) quantize_inputs(xs, n, h_.dim);
    for (std::size_t r = 0; r < rank; ++r) {
        const std::byte* row = fh_proj_ + r * prb;
        for (int b = 0; b < n; ++b) z.p[b][r] = score_row(b, fh_.proj_dtype, row, xs.p[b], h_.dim);
    }

    float* ts[kMaxBatch];
    int32_t* ti[kMaxBatch];
    int n_top[kMaxBatch] = {};
    for (int b = 0; b < n; ++b) {
        ts[b] = slots_[b] + kTopKOffset;
        ti[b] = reinterpret_cast<int32_t*>(slots_[b] + kTopKOffset + kMaxCandidates);
    }
    const auto t_proj_done = std::chrono::steady_clock::now();
    stats_.flash_proj_s += std::chrono::duration<double>(t_proj_done - t_scan).count();
    double topk_s = 0;
    if (xq_ != nullptr) quantize_inputs(z, n, rank);
    if (xq_ != nullptr && fh_.table_dtype == Dtype::Q4_0) {
        // Chunks of rows through the scan kernel (one call per chunk, not per row).
        constexpr std::size_t kChunk = 256;
        float scores[kChunk];
        for (uint32_t v0 = 0; v0 < h_.vocab; v0 += kChunk) {
            const std::size_t cnt = std::min<std::size_t>(kChunk, h_.vocab - v0);
            for (int b = 0; b < n; ++b) {
                m::scan_q4_0_q8(fh_table_ + static_cast<std::size_t>(v0) * trb, cnt, rank, xq_ + b * xq_cols_,
                                xs_ + b * (xq_cols_ / 32), scores);
                const auto tk = std::chrono::steady_clock::now();
                for (std::size_t r = 0; r < cnt; ++r) {
                    topk_insert(ts[b], ti[b], n_top[b], k, scores[r], static_cast<int32_t>(v0 + r));
                }
                topk_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - tk).count();
            }
        }
    } else {
        for (uint32_t v = 0; v < h_.vocab; ++v) {
            const std::byte* row = fh_table_ + static_cast<std::size_t>(v) * trb;
            for (int b = 0; b < n; ++b) {
                topk_insert(ts[b], ti[b], n_top[b], k, score_row(b, fh_.table_dtype, row, z.p[b], rank),
                            static_cast<int32_t>(v));
            }
        }
    }
    stats_.flash_scan_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_scan).count();
    stats_.topk_s += topk_s;
    if (xq_ != nullptr) quantize_inputs(xs, n, h_.dim);
    return rescore(xs, n, ti, n_top, out);
}

bool Engine::lm_head(const Ptrs& xs, int n, int32_t* out, float* logits) {
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = lm_head_impl(xs, n, out, logits);
    stats_.lm_head_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return ok;
}

bool Engine::lm_head_impl(const Ptrs& xs, int n, int32_t* out, float* logits) {
    assert(n >= 1 && n <= cfg_.max_batch && (logits == nullptr || n == 1));
    if (fh_table_ != nullptr && logits == nullptr) return lm_head_flash(xs, n, out);
    const bool shortlist = h_.lm_head != 0 && logits == nullptr;
    const bool lowrank = shortlist && h_.lm_rank > 0;
    const Dtype scan_dt = shortlist ? h_.lm_head_dtype : h_.embed_dtype;
    const uint64_t scan_off = shortlist ? h_.lm_head : h_.embed;
    const std::size_t scan_cols = lowrank ? h_.lm_rank : h_.dim;
    const std::size_t rb = row_bytes(scan_dt, scan_cols);

    // What the scan dots each row with: x itself, or (low rank) z = proj . x.
    Ptrs scan_in = xs;
    if (lowrank) {
        scan_in = slots(0);  // z lives at the start of each token's scratch slot
        if (!matvec(scan_in, h_.lm_rank, h_.lm_proj, h_.lm_head_dtype, h_.dim, xs, n)) return false;
    }

    // Integer path (see matvec): quantize the inputs once per stage.
    const bool int8 = xq_ != nullptr && logits == nullptr;
    auto quantize = [&](const Ptrs& in, std::size_t cols) {
        if (!int8) return;
        for (int b = 0; b < n; ++b) m::quantize_q8({in.p[b], cols}, xq_ + b * xq_cols_, xs_ + b * (xq_cols_ / 32));
    };
    auto score = [&](int b, Dtype dt, const std::byte* row, const Ptrs& in, std::size_t cols) {
        const int8_t* q = xq_ + b * xq_cols_;
        const float* sc = xs_ + b * (xq_cols_ / 32);
        if (int8 && dt == Dtype::Q8_0) return m::dot_q8_0_q8(row, q, sc, cols);
        if (int8 && dt == Dtype::Q4_0) return m::dot_q4_0_q8(row, q, sc, cols);
        return dot_row(dt, row, {in.p[b], cols});
    };

    // Per token: top-k by (approximate) score, sorted descending; ties keep
    // the lower id first, like argmax. Kept in the token's scratch slot,
    // after z (open() checks it fits).
    const int k = shortlist ? static_cast<int>(h_.lm_head_candidates) : 1;
    float* top_score[kMaxBatch];
    int32_t* top_id[kMaxBatch];
    int n_top[kMaxBatch] = {};
    for (int b = 0; b < n; ++b) {
        top_score[b] = slots_[b] + kTopKOffset;
        top_id[b] = reinterpret_cast<int32_t*>(slots_[b] + kTopKOffset + kMaxCandidates);
    }

    quantize(scan_in, scan_cols);
    const bool ok = stream_rows(scan_off, h_.vocab, rb, [&](std::size_t r, const std::byte* row) {
        for (int b = 0; b < n; ++b) {
            const float v = score(b, scan_dt, row, scan_in, scan_cols);
            if (logits != nullptr) logits[r] = v;
            float* ts = top_score[b];
            int32_t* ti = top_id[b];
            if (n_top[b] == k && v <= ts[k - 1]) continue;
            int pos = n_top[b] < k ? n_top[b]++ : k - 1;
            while (pos > 0 && ts[pos - 1] < v) {
                ts[pos] = ts[pos - 1];
                ti[pos] = ti[pos - 1];
                --pos;
            }
            ts[pos] = v;
            ti[pos] = static_cast<int32_t>(r);
        }
    });
    if (!ok) return false;
    if (!shortlist) {
        for (int b = 0; b < n; ++b) out[b] = top_id[b][0];
        return true;
    }

    // Re-score each shortlist with the exact rows, in file order (the SD
    // only seeks forward).
    quantize(xs, h_.dim);
    const std::size_t erb = row_bytes(h_.embed_dtype, h_.dim);
    for (int b = 0; b < n; ++b) {
        int32_t* ids = top_id[b];
        for (int i = 1; i < n_top[b]; ++i) {  // insertion sort ascending
            const int32_t id = ids[i];
            int j = i;
            for (; j > 0 && ids[j - 1] > id; --j) ids[j] = ids[j - 1];
            ids[j] = id;
        }
        float best = -INFINITY;
        int32_t best_id = ids[0];
        for (int i = 0; i < n_top[b]; ++i) {
            if (!src_->read_at(h_.embed + static_cast<uint64_t>(ids[i]) * erb, io_, erb)) return false;
            const float v = score(b, h_.embed_dtype, io_, xs, h_.dim);
            if (v > best) {  // ascending ids: strict > keeps the lowest id on ties
                best = v;
                best_id = ids[i];
            }
        }
        out[b] = best_id;
    }
    return true;
}

// ---------------------------------------------------------------- layers

// LFM2 short convolution (double-gated, causal, depthwise, kernel conv_l):
//   [B | C | X] = in_proj · in
//   bx = B * X
//   y  = sum_k kernel[c][k] * bx[t - (conv_l-1) + k]     (zeros before t = 0)
//   out = out_proj · (C * y)
// The batch's tokens go through the conv state in order.
bool Engine::conv_op(int layer, int n) {
    const Dims d(h_);
    const LayerOffsets& L = h_.layers[layer];
    float* kernel = shared_;  // [dim * conv_l]

    // in_proj = [B | C | X]: B and C into the slot, X into tmp_ (free until
    // out_proj writes it), as two streams over the matrix.
    const Dtype dt_in = h_.dtype(layer, kConvIn);
    if (!matvec(slots(0), 2 * d.dim, L.conv_in, dt_in, d.dim, acts(xn_), n)) return false;
    if (!matvec(acts(tmp_), d.dim, L.conv_in + 2 * d.dim * row_bytes(dt_in, d.dim), dt_in, d.dim, acts(xn_), n)) {
        return false;
    }
    if (!read_f32(L.conv_kernel, {kernel, d.dim * d.conv_l})) return false;

    const int cs = layer_slot_[layer];
    float* state = state_load(cs);
    if (state == nullptr) return false;
    const std::size_t hist = d.conv_l - 1;  // previous tokens kept
    if (recording_) {  // speculative verify: keep what accept() may need to roll back to
        for (std::size_t j = 0; j < hist; ++j) {
            if (!hist_put(cs, static_cast<int>(j), state + j * d.dim)) return false;
        }
    }
    for (int b = 0; b < n; ++b) {
        float* B = slot(b);
        const float* C = B + d.dim;
        const float* X = tmp_[b];
        for (std::size_t c = 0; c < d.dim; ++c) {
            const float* k = kernel + c * d.conv_l;
            const float bx = B[c] * X[c];
            float y = k[hist] * bx;
            for (std::size_t j = 0; j < hist; ++j) {
                y += k[j] * state[j * d.dim + c];
            }
            // Shift history: drop the oldest, append this token's bx.
            for (std::size_t j = 0; j + 1 < hist; ++j) {
                state[j * d.dim + c] = state[(j + 1) * d.dim + c];
            }
            state[(hist - 1) * d.dim + c] = bx;
            B[c] = C[c] * y;  // B is no longer needed: reuse it for out_proj's input
        }
        if (recording_ && !hist_put(cs, static_cast<int>(hist) + b, state + (hist - 1) * d.dim)) return false;
    }
    if (!state_store(cs, state)) return false;  // before out_proj reuses io_
    return matvec(acts(tmp_), d.dim, L.conv_out, h_.dtype(layer, kConvOut), d.dim, slots(0), n);
}

std::size_t Engine::kv_row_bytes() const {
    return 2 * static_cast<std::size_t>(h_.n_kv_heads) * h_.head_dim * (cfg_.kv_bf16 ? 2 : 4);
}

// Store position `pos`'s K and V for one attention layer. Staged in io_.
bool Engine::kv_put(int attn_idx, int pos, std::span<const float> k, std::span<const float> v) {
    const std::size_t rb = kv_row_bytes();
    if (cfg_.kv_bf16) {
        auto* dst = reinterpret_cast<uint16_t*>(io_);
        for (std::size_t i = 0; i < k.size(); ++i) dst[i] = m::f32_to_bf16(k[i]);
        for (std::size_t i = 0; i < v.size(); ++i) dst[k.size() + i] = m::f32_to_bf16(v[i]);
    } else {
        std::memcpy(io_, k.data(), k.size_bytes());
        std::memcpy(io_ + k.size_bytes(), v.data(), v.size_bytes());
    }
    if (kv_file_ != nullptr) {
        const uint64_t row = static_cast<uint64_t>(pos) * static_cast<uint64_t>(n_attn_) + static_cast<uint64_t>(attn_idx);
        return kv_file_->write_at(row * rb, io_, rb);
    }
    std::memcpy(kv_ram_[attn_idx] + static_cast<std::size_t>(pos) * rb, io_, rb);
    return true;
}

// Rows j0 .. j0+count-1 of one attention layer into dst.
bool Engine::kv_load(int attn_idx, int j0, int count, std::byte* dst) {
    const std::size_t rb = kv_row_bytes();
    if (kv_file_ == nullptr) {
        std::memcpy(dst, kv_ram_[attn_idx] + static_cast<std::size_t>(j0) * rb, static_cast<std::size_t>(count) * rb);
        return true;
    }
    for (int j = 0; j < count; ++j) {
        const uint64_t row = static_cast<uint64_t>(j0 + j) * static_cast<uint64_t>(n_attn_) + static_cast<uint64_t>(attn_idx);
        if (!kv_file_->read_at(row * rb, dst + static_cast<std::size_t>(j) * rb, rb)) return false;
    }
    return true;
}

// Grouped-query attention with per-head RMSNorm on q and k, then RoPE:
//   q, k, v = proj(in);  q, k = rope(norm(q)), rope(norm(k))
//   ctx_h = softmax(q_h · K_g^T / sqrt(head_dim)) · V_g      (g = kv head of h)
//   out = o_proj · ctx
//
// K/V rows are streamed through io_ a chunk of positions at a time, so the
// softmax is computed online: per head keep the running max `mx`, the running
// sum of exp(score - mx) `sum`, and ctx accumulated with the same weights;
// when a new max appears, rescale what's accumulated so far.
//
// Batch: every token's K/V is stored first; token b then attends to
// positions 0 .. pos_+b (causal).
bool Engine::attn_op(int layer, int n) {
    const Dims d(h_);
    const LayerOffsets& L = h_.layers[layer];
    const int ai = layer_slot_[layer];
    const std::size_t k_off = d.q_dim, v_off = d.q_dim + d.kv_dim;

    float* mx = shared_;
    float* sum = mx + d.heads;
    float* q_norm = sum + d.heads;
    float* k_norm = q_norm + d.head_dim;
    float* k_row = k_norm + d.head_dim;
    float* v_row = k_row + d.head_dim;

    if (!matvec(slots(0), d.q_dim, L.q, h_.dtype(layer, kQ), d.dim, acts(xn_), n)) return false;
    if (!matvec(slots(k_off), d.kv_dim, L.k, h_.dtype(layer, kK), d.dim, acts(xn_), n)) return false;
    if (!matvec(slots(v_off), d.kv_dim, L.v, h_.dtype(layer, kV), d.dim, acts(xn_), n)) return false;
    if (!read_f32(L.q_norm, {q_norm, d.head_dim})) return false;
    if (!read_f32(L.k_norm, {k_norm, d.head_dim})) return false;

    for (int b = 0; b < n; ++b) {
        float* q = slot(b);
        float* k = q + k_off;
        for (std::size_t hh = 0; hh < d.heads; ++hh) {
            std::span<float> qh{q + hh * d.head_dim, d.head_dim};
            m::rmsnorm(qh, qh, {q_norm, d.head_dim}, h_.norm_eps);
        }
        for (std::size_t g = 0; g < d.kv_heads; ++g) {
            std::span<float> kh{k + g * d.head_dim, d.head_dim};
            m::rmsnorm(kh, kh, {k_norm, d.head_dim}, h_.norm_eps);
        }
        const std::size_t p = static_cast<std::size_t>(pos_ + b);
        m::rope({q, d.q_dim}, d.head_dim, p, h_.rope_theta);
        m::rope({k, d.kv_dim}, d.head_dim, p, h_.rope_theta);
        if (!kv_put(ai, pos_ + b, {k, d.kv_dim}, {q + v_off, d.kv_dim})) return false;
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(d.head_dim));
    const std::size_t group = d.heads / d.kv_heads;
    const std::size_t rb = kv_row_bytes();
    const std::size_t elem = cfg_.kv_bf16 ? 2 : 4;
    const int chunk = static_cast<int>(cfg_.io_bytes / rb);

    // One K or V head slice of a loaded row, as floats.
    auto slice = [&](const std::byte* row, bool value, std::size_t g, float* tmp) -> const float* {
        const std::byte* p = row + ((value ? d.kv_dim : 0) + g * d.head_dim) * elem;
        if (!cfg_.kv_bf16) return reinterpret_cast<const float*>(p);
        m::bf16_to_f32({tmp, d.head_dim}, {reinterpret_cast<const uint16_t*>(p), d.head_dim});
        return tmp;
    };

    for (int b = 0; b < n; ++b) {
        const float* q = slot(b);
        float* ctx = xn_[b];  // xn_ is no longer needed once q, k, v exist
        const int n_pos = pos_ + b + 1;
        std::memset(ctx, 0, d.q_dim * sizeof(float));
        for (std::size_t hh = 0; hh < d.heads; ++hh) {
            mx[hh] = -INFINITY;
            sum[hh] = 0.0f;
        }
        for (int j0 = 0; j0 < n_pos; j0 += chunk) {
            const int count = std::min(chunk, n_pos - j0);
            if (!kv_load(ai, j0, count, io_)) return false;
            for (int jj = 0; jj < count; ++jj) {
                const std::byte* row = io_ + static_cast<std::size_t>(jj) * rb;
                for (std::size_t g = 0; g < d.kv_heads; ++g) {
                    const float* kr = slice(row, false, g, k_row);
                    const float* vr = slice(row, true, g, v_row);
                    for (std::size_t hh = g * group; hh < (g + 1) * group; ++hh) {
                        const float s = m::dot({q + hh * d.head_dim, d.head_dim}, {kr, d.head_dim}) * scale;
                        float* ch = ctx + hh * d.head_dim;
                        if (s > mx[hh]) {
                            const float c = std::exp(mx[hh] - s);  // 0 on the first position
                            sum[hh] *= c;
                            for (std::size_t i = 0; i < d.head_dim; ++i) ch[i] *= c;
                            mx[hh] = s;
                        }
                        const float pw = std::exp(s - mx[hh]);
                        sum[hh] += pw;
                        for (std::size_t i = 0; i < d.head_dim; ++i) ch[i] += pw * vr[i];
                    }
                }
            }
        }
        for (std::size_t hh = 0; hh < d.heads; ++hh) {
            const float inv = 1.0f / sum[hh];
            for (std::size_t i = 0; i < d.head_dim; ++i) ctx[hh * d.head_dim + i] *= inv;
        }
    }
    return matvec(acts(tmp_), d.dim, L.o, h_.dtype(layer, kO), d.q_dim, acts(xn_), n);
}

// SwiGLU: out = w2 · (silu(w1 · in) * (w3 · in)). The gate is computed first;
// w3's rows then turn it into the activation in place, so only ff floats per
// token are needed.
bool Engine::mlp(int layer, int n) {
    const Dims d(h_);
    const LayerOffsets& L = h_.layers[layer];
    if (!matvec(slots(0), d.ff, L.w1, h_.dtype(layer, kW1), d.dim, acts(xn_), n)) return false;

    const Dtype dt3 = h_.dtype(layer, kW3);
    // Integer path: w1's matvec just quantized these same inputs into xq_.
    const bool int8 = xq_ != nullptr && h_.dtype(layer, kW1) == dt3 && (dt3 == Dtype::Q8_0 || dt3 == Dtype::Q4_0);
    const bool ok = stream_rows(L.w3, d.ff, row_bytes(dt3, d.dim), [&](std::size_t r, const std::byte* row) {
        for (int b = 0; b < n; ++b) {
            float up;
            if (int8) {
                const int8_t* q = xq_ + b * xq_cols_;
                const float* sc = xs_ + b * (xq_cols_ / 32);
                up = dt3 == Dtype::Q8_0 ? m::dot_q8_0_q8(row, q, sc, d.dim) : m::dot_q4_0_q8(row, q, sc, d.dim);
            } else {
                up = dot_row(dt3, row, {act(xn_, b), d.dim});
            }
            float& g = slot(b)[r];
            g = g * (1.0f / (1.0f + std::exp(-g))) * up;  // silu(gate) * up
        }
    });
    if (!ok) return false;
    return matvec(acts(tmp_), d.dim, L.w2, h_.dtype(layer, kW2), d.ff, slots(0), n);
}

// ---------------------------------------------------------------- forward

int32_t Engine::run_batch(const int32_t* tokens, int n, bool want_next, float* logits, int32_t* preds) {
    if (pos_ + n > cfg_.max_ctx) return kContextFull;
    const Dims d(h_);
    std::span<float> norm_w{norm_w_, d.dim};

    // Embedding lookup: one row of the (quantized) table per token.
    const std::size_t rb = row_bytes(h_.embed_dtype, d.dim);
    for (int b = 0; b < n; ++b) {
        int32_t token = tokens[b];
        if (token < 0 || static_cast<std::size_t>(token) >= d.vocab) token = 0;
        if (!src_->read_at(h_.embed + static_cast<uint64_t>(token) * rb, io_, rb)) return kIoError;
        dequant_row(h_.embed_dtype, io_, {act(x_, b), d.dim});
    }

    auto norm_all = [&](uint64_t weight) {
        if (!read_f32(weight, norm_w)) return false;
        for (int b = 0; b < n; ++b) m::rmsnorm({act(xn_, b), d.dim}, {act(x_, b), d.dim}, norm_w, h_.norm_eps);
        return true;
    };
    auto add_all = [&] {
        for (int b = 0; b < n; ++b) m::add({act(x_, b), d.dim}, {act(tmp_, b), d.dim});
    };

    for (int i = 0; i < static_cast<int>(h_.n_layers); ++i) {
        const LayerOffsets& L = h_.layers[i];
        if (cfg_.skip_layers & (1u << i)) {
            progress(i + 1);
            continue;
        }

        // h = x + op(operator_norm(x))
        if (!norm_all(L.operator_norm)) return kIoError;
        const bool ok = h_.layer_kind[i] == LayerKind::Conv ? conv_op(i, n) : attn_op(i, n);
        if (!ok) return kIoError;
        add_all();

        // out = h + mlp(ffn_norm(h))
        if (!norm_all(L.ffn_norm)) return kIoError;
        if (!mlp(i, n)) return kIoError;
        add_all();

        if (layer_hook_ != nullptr) {
            for (int b = 0; b < n; ++b) layer_hook_(layer_ctx_, i, pos_ + b, {act(x_, b), d.dim});
        }
        progress(i + 1);
    }
    pos_ += n;

    if (!want_next && preds == nullptr) {
        progress(static_cast<int>(h_.n_layers) + 1);
        return kNoToken;
    }
    if (!read_f32(h_.final_norm, norm_w)) return kIoError;
    int32_t next = kNoToken;
    if (preds != nullptr) {
        // Every position's next token (speculative verification): one scan
        // of the LM head for all of them.
        for (int b = 0; b < n; ++b) m::rmsnorm({act(xn_, b), d.dim}, {act(x_, b), d.dim}, norm_w, h_.norm_eps);
        if (!lm_head(acts(xn_), n, preds, nullptr)) return kIoError;
        next = preds[n - 1];
    } else {
        // Only the last token's prediction is needed.
        m::rmsnorm({act(xn_, n - 1), d.dim}, {act(x_, n - 1), d.dim}, norm_w, h_.norm_eps);
        Ptrs last{};
        last.p[0] = act(xn_, n - 1);
        if (!lm_head(last, 1, &next, logits)) return kIoError;
    }
    progress(static_cast<int>(h_.n_layers) + 1);
    return next;
}

int32_t Engine::forward(int32_t token, bool want_next, float* logits) {
    assert(pending_n_ == 0 && "verify() must be followed by accept()");
    return run_batch(&token, 1, want_next, logits, nullptr);
}

int32_t Engine::verify(std::span<const int32_t> tokens, int32_t* preds) {
    assert(pending_n_ == 0 && "verify() must be followed by accept()");
    const int n = static_cast<int>(tokens.size());
    if (n < 1 || n > cfg_.max_batch || !cfg_.speculative) return kInvalid;
    const int start = pos_;
    recording_ = true;
    const int32_t r = run_batch(tokens.data(), n, true, nullptr, preds);
    recording_ = false;
    if (r < 0) return r;  // nothing pending: pos_ only moves on success
    pending_start_ = start;
    pending_n_ = n;
    return r;
}

bool Engine::accept(int keep) {
    if (pending_n_ == 0 || keep < 1 || keep > pending_n_) return false;
    if (keep < pending_n_) {
        // Conv state after `keep` tokens = history rows keep .. keep+hist-1
        // (history = the state before verify(), then each token's bx).
        const std::size_t hist = h_.conv_l - 1;
        for (int s = 0; s < n_conv_; ++s) {
            float* state = state_file_ != nullptr ? reinterpret_cast<float*>(io_) : conv_state_[s];
            for (std::size_t j = 0; j < hist; ++j) {
                if (!hist_get(s, static_cast<int>(static_cast<std::size_t>(keep) + j), state + j * h_.dim)) return false;
            }
            if (!state_store(s, state)) return false;
        }
        // Rejected K/V rows are simply overwritten by the next positions.
    }
    pos_ = pending_start_ + keep;
    pending_n_ = 0;
    return true;
}

// Conv state: in RAM, or (conv_state_path) in a file, loaded into io_ while
// a conv layer runs (io_ is free between that layer's two matvecs).
float* Engine::state_load(int slot) {
    if (state_file_ == nullptr) return conv_state_[slot];
    const std::size_t bytes = (h_.conv_l - 1) * h_.dim * sizeof(float);
    auto* buf = reinterpret_cast<float*>(io_);
    return state_file_->read_at(static_cast<uint64_t>(slot) * bytes, buf, bytes) ? buf : nullptr;
}

bool Engine::state_store(int slot, const float* state) {
    if (state_file_ == nullptr) return true;  // RAM state was updated in place
    const std::size_t bytes = (h_.conv_l - 1) * h_.dim * sizeof(float);
    return state_file_->write_at(static_cast<uint64_t>(slot) * bytes, state, bytes);
}

// Conv history for speculative rollback: per conv layer, rows of dim floats:
// the conv_l-1 state rows before verify(), then one row (bx) per token.
std::size_t Engine::hist_rows() const { return h_.conv_l - 1 + static_cast<std::size_t>(cfg_.max_batch); }

bool Engine::hist_put(int slot, int row, const float* src) {
    const std::size_t bytes = h_.dim * sizeof(float);
    const std::size_t idx = static_cast<std::size_t>(slot) * hist_rows() + static_cast<std::size_t>(row);
    if (hist_file_ != nullptr) return hist_file_->write_at(idx * bytes, src, bytes);
    std::memcpy(hist_ram_ + idx * h_.dim, src, bytes);
    return true;
}

bool Engine::hist_get(int slot, int row, float* dst) {
    const std::size_t bytes = h_.dim * sizeof(float);
    const std::size_t idx = static_cast<std::size_t>(slot) * hist_rows() + static_cast<std::size_t>(row);
    if (hist_file_ != nullptr) return hist_file_->read_at(idx * bytes, dst, bytes);
    std::memcpy(dst, hist_ram_ + idx * h_.dim, bytes);
    return true;
}

int32_t Engine::prefill(std::span<const int32_t> tokens, bool want_next, float* logits) {
    assert(pending_n_ == 0 && "verify() must be followed by accept()");
    int32_t next = kNoToken;
    for (std::size_t i = 0; i < tokens.size();) {
        const int n = static_cast<int>(std::min<std::size_t>(static_cast<std::size_t>(cfg_.max_batch), tokens.size() - i));
        const bool last = i + static_cast<std::size_t>(n) == tokens.size();
        next = run_batch(tokens.data() + i, n, want_next && last, last ? logits : nullptr, nullptr);
        if (next < kNoToken) return next;  // error / context full
        i += static_cast<std::size_t>(n);
    }
    return next;
}

std::size_t Engine::token_text(int32_t id, char* buf, std::size_t cap) {
    if (id < 0 || static_cast<uint32_t>(id) >= h_.vocab) return 0;
    uint32_t range[2];
    if (!src_->read_at(h_.vocab_index + static_cast<uint64_t>(id) * 4, range, sizeof(range))) return 0;
    const std::size_t len = std::min<std::size_t>(range[1] - range[0], cap);
    if (len > 0 && !src_->read_at(h_.vocab_bytes + range[0], buf, len)) return 0;
    return len;
}

}  // namespace lfm
