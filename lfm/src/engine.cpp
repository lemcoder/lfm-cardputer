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

    // Scratch shared by the ops (only one runs at a time). Independent of
    // the context length: attention streams the KV cache (online softmax).
    std::size_t arena() const {
        const std::size_t conv = 3 * dim + dim * conv_l;  // proj + kernel
        const std::size_t mlp = 2 * ff;                   // gate + up
        const std::size_t attn = q_dim + 2 * kv_dim       // q, k, v
                                 + q_dim + 2 * heads      // ctx accumulators, running max and sum
                                 + 4 * head_dim;          // q_norm, k_norm, one K and one V row
        std::size_t a = conv > mlp ? conv : mlp;
        return a > attn ? a : attn;
    }
};

void dequant_row(Dtype dt, const std::byte* row, std::span<float> out) {
    if (dt == Dtype::BF16) {
        for (std::size_t i = 0; i < out.size(); ++i) {
            uint16_t h;
            std::memcpy(&h, row + 2 * i, 2);
            out[i] = std::bit_cast<float>(static_cast<uint32_t>(h) << 16);
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
    std::free(mem_);
    std::free(conv_state_);
    for (std::byte* p : kv_ram_) std::free(p);
    delete kv_file_;
}

std::size_t Engine::ram_bytes(const EngineConfig& cfg) const {
    const Dims d(h_);
    const std::size_t kv = cfg.kv_path != nullptr ? 0
        : static_cast<std::size_t>(n_attn_) * static_cast<std::size_t>(cfg.max_ctx) * 2 * d.kv_dim * (cfg.kv_bf16 ? 2 : 4);
    return cfg.io_bytes + (4 * d.dim + d.arena()) * sizeof(float)
           + static_cast<std::size_t>(n_conv_) * (d.conv_l - 1) * d.dim * sizeof(float) + kv;
}

Status Engine::open(Source& src, const EngineConfig& cfg) {
    assert(mem_ == nullptr && "open() called twice");
    src_ = &src;
    cfg_ = cfg;

    if (!src.read_at(0, &h_, sizeof(h_))) return Status::IoError;
    if (std::memcmp(h_.magic, kMagic, sizeof(kMagic)) != 0) return Status::BadMagic;
    if (h_.version != kFormatVersion) return Status::BadVersion;

    const Dims d(h_);
    const bool ok = (h_.matrix_dtype == Dtype::BF16 || h_.matrix_dtype == Dtype::Q8_0) &&
                    h_.n_layers >= 1 && h_.n_layers <= static_cast<uint32_t>(kMaxLayers) &&
                    d.dim % kQ8BlockSize == 0 && d.ff % kQ8BlockSize == 0 && d.q_dim % kQ8BlockSize == 0 &&
                    d.heads % d.kv_heads == 0 && d.head_dim % 2 == 0 && d.conv_l >= 2 &&
                    row_bytes(h_.matrix_dtype, d.ff) <= cfg.io_bytes &&
                    2 * d.kv_dim * sizeof(float) <= cfg.io_bytes && cfg.max_ctx >= 1;
    if (!ok) return Status::Unsupported;

    n_conv_ = n_attn_ = 0;
    for (uint32_t i = 0; i < h_.n_layers; ++i) {
        layer_slot_[i] = h_.layer_kind[i] == LayerKind::Conv ? n_conv_++ : n_attn_++;
    }

    // Main block: io buffer, then x, xn, tmp, norm_w, arena.
    const std::size_t floats = 4 * d.dim + d.arena();
    mem_ = std::malloc(cfg.io_bytes + floats * sizeof(float));
    if (mem_ == nullptr) return Status::OutOfMemory;
    io_ = static_cast<std::byte*>(mem_);
    x_ = reinterpret_cast<float*>(io_ + cfg.io_bytes);
    xn_ = x_ + d.dim;
    tmp_ = xn_ + d.dim;
    norm_w_ = tmp_ + d.dim;
    arena_ = norm_w_ + d.dim;

    if (n_conv_ > 0) {
        conv_state_ = static_cast<float*>(std::malloc(n_conv_ * (d.conv_l - 1) * d.dim * sizeof(float)));
        if (conv_state_ == nullptr) return Status::OutOfMemory;
    }
    if (cfg.kv_path != nullptr) {
        kv_file_ = new FileSource(cfg.kv_path, FileSource::Mode::ReadWrite);
        if (!kv_file_->ok()) return Status::IoError;
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
    if (conv_state_ != nullptr) {
        std::memset(conv_state_, 0, n_conv_ * (h_.conv_l - 1) * h_.dim * sizeof(float));
    }
}

void Engine::progress(int done) {
    if (progress_hook_ != nullptr) progress_hook_(progress_ctx_, done, static_cast<int>(h_.n_layers) + 1);
}

// ---------------------------------------------------------------- weight I/O

bool Engine::read_f32(uint64_t offset, std::span<float> out) {
    return src_->read_at(offset, out.data(), out.size_bytes());
}

float Engine::dot_row(const std::byte* row, std::span<const float> x) const {
    return h_.matrix_dtype == Dtype::Q8_0 ? m::dot_q8_0(row, x) : m::dot_bf16(row, x);
}

// out[r] = row r of the matrix at `offset` (out.size() rows, `cols` columns) · x.
// Streams the matrix through io_ a whole number of rows at a time.
bool Engine::matvec(std::span<float> out, uint64_t offset, std::size_t cols, std::span<const float> x) {
    assert(x.size() == cols);
    const std::size_t rb = row_bytes(h_.matrix_dtype, cols);
    const std::size_t rows_per_chunk = cfg_.io_bytes / rb;
    for (std::size_t r0 = 0; r0 < out.size(); r0 += rows_per_chunk) {
        const std::size_t n = std::min(rows_per_chunk, out.size() - r0);
        if (!src_->read_at(offset + r0 * rb, io_, n * rb)) return false;
        for (std::size_t i = 0; i < n; ++i) {
            out[r0 + i] = dot_row(io_ + i * rb, x);
        }
    }
    return true;
}

// Tied LM head: logit[v] = embed row v · x. Streams all vocab rows, keeps
// only the running argmax (the logits array would be 256 KB).
int32_t Engine::lm_head(std::span<const float> x, float* logits) {
    const std::size_t rb = row_bytes(h_.matrix_dtype, h_.dim);
    const std::size_t rows_per_chunk = cfg_.io_bytes / rb;
    float best = -INFINITY;
    int32_t best_id = 0;
    for (std::size_t r0 = 0; r0 < h_.vocab; r0 += rows_per_chunk) {
        const std::size_t n = std::min<std::size_t>(rows_per_chunk, h_.vocab - r0);
        if (!src_->read_at(h_.embed + r0 * rb, io_, n * rb)) return kIoError;
        for (std::size_t i = 0; i < n; ++i) {
            const float v = dot_row(io_ + i * rb, x);
            if (logits != nullptr) logits[r0 + i] = v;
            if (v > best) {
                best = v;
                best_id = static_cast<int32_t>(r0 + i);
            }
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
bool Engine::conv_op(int layer, std::span<float> out, std::span<const float> in) {
    const Dims d(h_);
    const LayerOffsets& L = h_.layers[layer];
    float* proj = arena_;                 // [3*dim]
    float* kernel = arena_ + 3 * d.dim;   // [dim * conv_l]

    if (!matvec({proj, 3 * d.dim}, L.conv_in, d.dim, in)) return false;
    if (!read_f32(L.conv_kernel, {kernel, d.dim * d.conv_l})) return false;

    float* B = proj;
    const float* C = proj + d.dim;
    const float* X = proj + 2 * d.dim;
    float* state = conv_state_ + layer_slot_[layer] * (d.conv_l - 1) * d.dim;
    const std::size_t hist = d.conv_l - 1;  // previous tokens kept

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
    return matvec(out, L.conv_out, d.dim, {B, d.dim});
}

std::size_t Engine::kv_row_bytes() const {
    return 2 * static_cast<std::size_t>(h_.n_kv_heads) * h_.head_dim * (cfg_.kv_bf16 ? 2 : 4);
}

// Append this position's K and V for one attention layer. Staged in io_.
bool Engine::kv_put(int attn_idx, std::span<const float> k, std::span<const float> v) {
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
        const uint64_t row = static_cast<uint64_t>(pos_) * static_cast<uint64_t>(n_attn_) + static_cast<uint64_t>(attn_idx);
        return kv_file_->write_at(row * rb, io_, rb);
    }
    std::memcpy(kv_ram_[attn_idx] + static_cast<std::size_t>(pos_) * rb, io_, rb);
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
bool Engine::attn_op(int layer, std::span<float> out, std::span<const float> in) {
    const Dims d(h_);
    const LayerOffsets& L = h_.layers[layer];
    const int ai = layer_slot_[layer];
    const int n = pos_ + 1;  // positions to attend over, this one included

    float* q = arena_;
    float* k = q + d.q_dim;
    float* v = k + d.kv_dim;
    float* ctx = v + d.kv_dim;
    float* mx = ctx + d.q_dim;
    float* sum = mx + d.heads;
    float* q_norm = sum + d.heads;
    float* k_norm = q_norm + d.head_dim;
    float* k_row = k_norm + d.head_dim;
    float* v_row = k_row + d.head_dim;

    if (!matvec({q, d.q_dim}, L.q, d.dim, in)) return false;
    if (!matvec({k, d.kv_dim}, L.k, d.dim, in)) return false;
    if (!matvec({v, d.kv_dim}, L.v, d.dim, in)) return false;
    if (!read_f32(L.q_norm, {q_norm, d.head_dim})) return false;
    if (!read_f32(L.k_norm, {k_norm, d.head_dim})) return false;

    for (std::size_t hh = 0; hh < d.heads; ++hh) {
        std::span<float> qh{q + hh * d.head_dim, d.head_dim};
        m::rmsnorm(qh, qh, {q_norm, d.head_dim}, h_.norm_eps);
    }
    for (std::size_t g = 0; g < d.kv_heads; ++g) {
        std::span<float> kh{k + g * d.head_dim, d.head_dim};
        m::rmsnorm(kh, kh, {k_norm, d.head_dim}, h_.norm_eps);
    }
    m::rope({q, d.q_dim}, d.head_dim, static_cast<std::size_t>(pos_), h_.rope_theta);
    m::rope({k, d.kv_dim}, d.head_dim, static_cast<std::size_t>(pos_), h_.rope_theta);
    if (!kv_put(ai, {k, d.kv_dim}, {v, d.kv_dim})) return false;

    const float scale = 1.0f / std::sqrt(static_cast<float>(d.head_dim));
    const std::size_t group = d.heads / d.kv_heads;
    const std::size_t rb = kv_row_bytes();
    const std::size_t elem = cfg_.kv_bf16 ? 2 : 4;
    const int chunk = static_cast<int>(cfg_.io_bytes / rb);

    std::memset(ctx, 0, d.q_dim * sizeof(float));
    for (std::size_t hh = 0; hh < d.heads; ++hh) {
        mx[hh] = -INFINITY;
        sum[hh] = 0.0f;
    }

    // One K or V head slice of a loaded row, as floats.
    auto slice = [&](const std::byte* row, bool value, std::size_t g, float* tmp) -> const float* {
        const std::byte* p = row + ((value ? d.kv_dim : 0) + g * d.head_dim) * elem;
        if (!cfg_.kv_bf16) return reinterpret_cast<const float*>(p);
        m::bf16_to_f32({tmp, d.head_dim}, {reinterpret_cast<const uint16_t*>(p), d.head_dim});
        return tmp;
    };

    for (int j0 = 0; j0 < n; j0 += chunk) {
        const int count = std::min(chunk, n - j0);
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
                    const float p = std::exp(s - mx[hh]);
                    sum[hh] += p;
                    for (std::size_t i = 0; i < d.head_dim; ++i) ch[i] += p * vr[i];
                }
            }
        }
    }
    for (std::size_t hh = 0; hh < d.heads; ++hh) {
        const float inv = 1.0f / sum[hh];
        for (std::size_t i = 0; i < d.head_dim; ++i) ctx[hh * d.head_dim + i] *= inv;
    }
    return matvec(out, L.o, d.q_dim, {ctx, d.q_dim});
}

// SwiGLU: out = w2 · (silu(w1 · in) * (w3 · in))
bool Engine::mlp(int layer, std::span<float> out, std::span<const float> in) {
    const Dims d(h_);
    const LayerOffsets& L = h_.layers[layer];
    std::span<float> gate{arena_, d.ff};
    std::span<float> up{arena_ + d.ff, d.ff};
    if (!matvec(gate, L.w1, d.dim, in)) return false;
    if (!matvec(up, L.w3, d.dim, in)) return false;
    m::swiglu(gate, gate, up);
    return matvec(out, L.w2, d.ff, gate);
}

// ---------------------------------------------------------------- forward

int32_t Engine::forward(int32_t token, bool want_next, float* logits) {
    if (pos_ >= cfg_.max_ctx) return kContextFull;
    const Dims d(h_);
    std::span<float> x{x_, d.dim}, xn{xn_, d.dim}, tmp{tmp_, d.dim}, norm_w{norm_w_, d.dim};

    // Embedding lookup: one row of the (quantized) table.
    const std::size_t rb = row_bytes(h_.matrix_dtype, d.dim);
    if (token < 0 || static_cast<std::size_t>(token) >= d.vocab) token = 0;
    if (!src_->read_at(h_.embed + static_cast<uint64_t>(token) * rb, io_, rb)) return kIoError;
    dequant_row(h_.matrix_dtype, io_, x);

    for (int i = 0; i < static_cast<int>(h_.n_layers); ++i) {
        const LayerOffsets& L = h_.layers[i];

        // h = x + op(operator_norm(x))
        if (!read_f32(L.operator_norm, norm_w)) return kIoError;
        m::rmsnorm(xn, x, norm_w, h_.norm_eps);
        const bool ok = h_.layer_kind[i] == LayerKind::Conv ? conv_op(i, tmp, xn) : attn_op(i, tmp, xn);
        if (!ok) return kIoError;
        m::add(x, tmp);

        // out = h + mlp(ffn_norm(h))
        if (!read_f32(L.ffn_norm, norm_w)) return kIoError;
        m::rmsnorm(xn, x, norm_w, h_.norm_eps);
        if (!mlp(i, tmp, xn)) return kIoError;
        m::add(x, tmp);

        if (layer_hook_ != nullptr) layer_hook_(layer_ctx_, i, x);
        progress(i + 1);
    }
    ++pos_;

    if (!want_next) {
        progress(static_cast<int>(h_.n_layers) + 1);
        return kNoToken;
    }
    if (!read_f32(h_.final_norm, norm_w)) return kIoError;
    m::rmsnorm(xn, x, norm_w, h_.norm_eps);
    const int32_t next = lm_head(xn, logits);
    progress(static_cast<int>(h_.n_layers) + 1);
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
