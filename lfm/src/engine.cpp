#include "lfm/engine.hpp"

#include <algorithm>
#include <bit>
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
        case Status::KvFileError: return "can't create KV cache file";
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
        const std::size_t conv = 3 * dim;                    // in_proj output [B | C | X]
        const std::size_t mlp = ff;                          // gate, then silu(gate) * up in place
        const std::size_t attn = q_dim + 2 * kv_dim + q_dim;  // q, k, v, ctx
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
           + static_cast<std::size_t>(n_conv_) * (d.conv_l - 1) * d.dim * sizeof(float) + kv;
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
    } else if (h_.version != kFormatVersion) {
        return Status::BadVersion;
    }

    const Dims d(h_);
    bool ok = is_matrix_dtype(h_.embed_dtype) &&
              (h_.lm_head == 0 || (is_matrix_dtype(h_.lm_head_dtype) && h_.lm_head_candidates >= 1 &&
                                   h_.lm_head_candidates <= kMaxCandidates)) &&
              h_.n_layers >= 1 && h_.n_layers <= static_cast<uint32_t>(kMaxLayers) &&
              d.dim % kQ8BlockSize == 0 && d.ff % kQ8BlockSize == 0 && d.q_dim % kQ8BlockSize == 0 &&
              d.heads % d.kv_heads == 0 && d.head_dim % 2 == 0 && d.conv_l >= 2 &&
              row_bytes(Dtype::BF16, d.ff) <= cfg.io_bytes &&  // widest row we stream
              2 * d.kv_dim * sizeof(float) <= cfg.io_bytes && cfg.max_ctx >= 1 && cfg.max_batch >= 1 &&
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
    slot_floats_ = d.slot();
    for (std::size_t b = 0; b < B; ++b) {
        slots_[b] = static_cast<float*>(alloc(slot_floats_ * sizeof(float)));
        if (slots_[b] == nullptr) return Status::OutOfMemory;
    }
    x_ = static_cast<float*>(alloc(B * d.dim * sizeof(float)));
    xn_ = static_cast<float*>(alloc(B * d.dim * sizeof(float)));
    tmp_ = static_cast<float*>(alloc(B * d.dim * sizeof(float)));
    norm_w_ = static_cast<float*>(alloc((d.dim + d.shared()) * sizeof(float)));
    carry_ = static_cast<std::byte*>(alloc(row_bytes(Dtype::BF16, d.ff)));  // widest row we stream
    if (cfg.int8_activations) {
        xq_cols_ = std::max({d.dim, d.ff, d.q_dim});
        xq_ = static_cast<int8_t*>(alloc(B * xq_cols_));
        xs_ = static_cast<float*>(alloc(B * (xq_cols_ / 32) * sizeof(float)));
        if (xq_ == nullptr || xs_ == nullptr) return Status::OutOfMemory;
    }
    if (io_ == nullptr || io2_ == nullptr || x_ == nullptr || xn_ == nullptr ||
        tmp_ == nullptr || norm_w_ == nullptr || carry_ == nullptr) {
        return Status::OutOfMemory;
    }
    shared_ = norm_w_ + d.dim;

    if (n_conv_ > 0) {
        // One block per conv layer: small blocks fit a fragmented heap.
        for (int i = 0; i < n_conv_; ++i) {
            conv_state_[i] = static_cast<float*>(std::malloc((d.conv_l - 1) * d.dim * sizeof(float)));
            if (conv_state_[i] == nullptr) return Status::OutOfMemory;
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
    for (int i = 0; i < n_conv_; ++i) {
        if (conv_state_[i] != nullptr) std::memset(conv_state_[i], 0, (h_.conv_l - 1) * h_.dim * sizeof(float));
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

Engine::Ptrs Engine::acts(float* base) const {
    Ptrs r{};
    for (int b = 0; b < cfg_.max_batch; ++b) r.p[b] = base + static_cast<std::size_t>(b) * h_.dim;
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
int32_t Engine::lm_head(std::span<const float> x, float* logits) {
    const bool shortlist = h_.lm_head != 0 && logits == nullptr;
    const Dtype scan_dt = shortlist ? h_.lm_head_dtype : h_.embed_dtype;
    const uint64_t scan_off = shortlist ? h_.lm_head : h_.embed;
    const std::size_t rb = row_bytes(scan_dt, h_.dim);

    // Integer path (see matvec): quantize x once for the scan and the re-score.
    const bool int8 = xq_ != nullptr && logits == nullptr;
    if (int8) m::quantize_q8(x, xq_, xs_);
    auto score = [&](Dtype dt, const std::byte* row) {
        if (int8 && dt == Dtype::Q8_0) return m::dot_q8_0_q8(row, xq_, xs_, h_.dim);
        if (int8 && dt == Dtype::Q4_0) return m::dot_q4_0_q8(row, xq_, xs_, h_.dim);
        return dot_row(dt, row, x);
    };

    // Top-k by approximate score, kept sorted descending (k is small).
    const int k = shortlist ? static_cast<int>(h_.lm_head_candidates) : 1;
    float top_score[kMaxCandidates];
    int32_t top_id[kMaxCandidates];
    int n_top = 0;

    const bool ok = stream_rows(scan_off, h_.vocab, rb, [&](std::size_t r, const std::byte* row) {
        const float v = score(scan_dt, row);
        if (logits != nullptr) logits[r] = v;
        if (n_top == k && v <= top_score[k - 1]) return;
        int pos = n_top < k ? n_top++ : k - 1;
        while (pos > 0 && top_score[pos - 1] < v) {
            top_score[pos] = top_score[pos - 1];
            top_id[pos] = top_id[pos - 1];
            --pos;
        }
        top_score[pos] = v;
        top_id[pos] = static_cast<int32_t>(r);
    });
    if (!ok) return kIoError;
    if (!shortlist) return top_id[0];

    // Re-score the shortlist with the exact rows.
    const std::size_t erb = row_bytes(h_.embed_dtype, h_.dim);
    float best = -INFINITY;
    int32_t best_id = top_id[0];
    for (int i = 0; i < n_top; ++i) {
        if (!src_->read_at(h_.embed + static_cast<uint64_t>(top_id[i]) * erb, io_, erb)) return kIoError;
        const float v = score(h_.embed_dtype, io_);
        if (v > best || (v == best && top_id[i] < best_id)) {
            best = v;
            best_id = top_id[i];
        }
    }
    return best_id;
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

    if (!matvec(slots(0), 3 * d.dim, L.conv_in, h_.dtype(layer, kConvIn), d.dim, acts(xn_), n)) return false;
    if (!read_f32(L.conv_kernel, {kernel, d.dim * d.conv_l})) return false;

    float* state = conv_state_[layer_slot_[layer]];
    const std::size_t hist = d.conv_l - 1;  // previous tokens kept
    for (int b = 0; b < n; ++b) {
        float* B = slot(b);
        const float* C = B + d.dim;
        const float* X = B + 2 * d.dim;
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
    }
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
    const std::size_t k_off = d.q_dim, v_off = d.q_dim + d.kv_dim, ctx_off = d.q_dim + 2 * d.kv_dim;

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
        float* ctx = slot(b) + ctx_off;
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
    return matvec(acts(tmp_), d.dim, L.o, h_.dtype(layer, kO), d.q_dim, slots(ctx_off), n);
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

int32_t Engine::run_batch(const int32_t* tokens, int n, bool want_next, float* logits) {
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

    if (!want_next) {
        progress(static_cast<int>(h_.n_layers) + 1);
        return kNoToken;
    }
    // Only the last token's next-token prediction is needed.
    std::span<float> xn{act(xn_, n - 1), d.dim};
    if (!read_f32(h_.final_norm, norm_w)) return kIoError;
    m::rmsnorm(xn, {act(x_, n - 1), d.dim}, norm_w, h_.norm_eps);
    const int32_t next = lm_head(xn, logits);
    progress(static_cast<int>(h_.n_layers) + 1);
    return next;
}

int32_t Engine::forward(int32_t token, bool want_next, float* logits) {
    return run_batch(&token, 1, want_next, logits);
}

int32_t Engine::prefill(std::span<const int32_t> tokens, bool want_next, float* logits) {
    int32_t next = kNoToken;
    for (std::size_t i = 0; i < tokens.size();) {
        const int n = static_cast<int>(std::min<std::size_t>(static_cast<std::size_t>(cfg_.max_batch), tokens.size() - i));
        const bool last = i + static_cast<std::size_t>(n) == tokens.size();
        next = run_batch(tokens.data() + i, n, want_next && last, last ? logits : nullptr);
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
