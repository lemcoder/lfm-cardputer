#pragma once
// LFM2 inference engine, one token at a time, streaming weights from a Source.
//
// RAM holds only activations, the conv state and one I/O buffer, plus the
// KV cache unless it's kept in a file (EngineConfig::kv_path). Every weight
// is read from the Source when it's needed. The same code runs on the Mac
// (host tests, file in page cache) and on the ESP32-S3 (files on the SD card).
//
//   lfm::FileSource src("/sdcard/lfm.lfm");
//   lfm::Engine eng;
//   eng.open(src, {.max_ctx = 1024, .kv_bf16 = true, .kv_path = "/sdcard/kv.bin"});
//   int32_t next = eng.prefill(prompt, true);  // whole prompt, max_batch at a time
//   next = eng.forward(next, true);            // then one token at a time

#include <cstddef>
#include <cstdint>
#include <span>

#include "lfm/format.hpp"
#include "lfm/source.hpp"

namespace lfm {

enum class Status : uint8_t {
    Ok,
    IoError,
    BadMagic,
    BadVersion,
    Unsupported,  // header describes a model this engine can't run
    OutOfMemory,
    KvFileError,  // couldn't create / open EngineConfig::kv_path
};

const char* to_string(Status s);

struct EngineConfig {
    int max_ctx = 128;              // KV cache capacity in tokens
    bool kv_bf16 = false;           // store K/V as bf16: half the size, ~3 digits of precision
    std::size_t io_bytes = 32768;   // weight / KV streaming buffer; keep a multiple of 512 (SD sector)
    // Two weight buffers: compute on one while the Source fills the other
    // (only helps with a Source whose start_read() is really asynchronous).
    bool double_buffer = true;
    // Keep the KV cache in this file (created / emptied on open) instead of
    // RAM. RAM use then doesn't depend on max_ctx. Each token appends
    // n_attn_layers * 2 * n_kv*head_dim values and attention reads them back.
    const char* kv_path = nullptr;
    // prefill() runs up to this many tokens through each layer together, so
    // each weight chunk is read once per batch instead of once per token.
    // Costs ~(3*dim + max(3*dim, ff, 3*dim)) floats of RAM per extra token.
    int max_batch = 1;
    // Quantize each matvec input to int8 (blocks of 32 + scale) so Q8/Q4
    // weight rows are dotted with integer math (llama.cpp's Q8 x Q8). Much
    // faster on the ESP32-S3; slightly less exact than float activations.
    bool int8_activations = false;
};

class Engine {
public:
    // Called after every layer, for every token, with that token's residual
    // stream (for tests/debugging). `pos` is the token's position.
    using LayerHook = void (*)(void* ctx, int layer, int pos, std::span<const float> x);
    // Called as work progresses: `done` of `total` steps (layers + LM head).
    using ProgressHook = void (*)(void* ctx, int done, int total);

    Engine() = default;
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // Read and validate the header, allocate all buffers. Call once.
    Status open(Source& src, const EngineConfig& cfg);

    const FileHeader& header() const { return h_; }
    int max_ctx() const { return cfg_.max_ctx; }
    int pos() const { return pos_; }  // tokens consumed so far

    // Forget all tokens (new sequence).
    void reset();

    // forward() results that aren't token ids.
    static constexpr int32_t kNoToken = -1;      // want_next was false
    static constexpr int32_t kIoError = -2;
    static constexpr int32_t kContextFull = -3;  // pos() == max_ctx()

    // Feed `token` at position pos() and advance.
    // With want_next, also runs the LM head and returns the greedy next token
    // (and, if `logits` is non-null, writes all vocab logits there).
    int32_t forward(int32_t token, bool want_next, float* logits = nullptr);

    // Feed several tokens (e.g. the prompt), max_batch at a time; same result
    // as calling forward() on each, but with fewer weight reads. With
    // want_next, returns the greedy token after the last one.
    int32_t prefill(std::span<const int32_t> tokens, bool want_next, float* logits = nullptr);

    // Bytes of token `id` (UTF-8 / raw bytes) into buf. Returns the length,
    // truncated to cap. Not NUL-terminated.
    std::size_t token_text(int32_t id, char* buf, std::size_t cap);

    // Bytes of RAM open() allocates for a given config (header must be read).
    std::size_t ram_bytes(const EngineConfig& cfg) const;

    void set_layer_hook(LayerHook fn, void* ctx) { layer_hook_ = fn; layer_ctx_ = ctx; }
    void set_progress_hook(ProgressHook fn, void* ctx) { progress_hook_ = fn; progress_ctx_ = ctx; }

private:
    bool read_f32(uint64_t offset, std::span<float> out);
    // For each of n tokens b: out[b*out_stride + r] = row r · x[b*x_stride ..][0..cols).
    bool matvec(float* out, std::size_t out_stride, std::size_t rows, uint64_t offset, Dtype dt, std::size_t cols,
                const float* x, std::size_t x_stride, int n);
    // Stream `rows` rows of `rb` bytes from `offset`, calling fn(row index,
    // row bytes) for each, with the next chunk's read overlapping fn's work.
    template <typename Fn>
    bool stream_rows(uint64_t offset, std::size_t rows, std::size_t rb, Fn&& fn);
    int32_t lm_head(std::span<const float> x, float* logits);

    // n tokens at positions pos_ .. pos_+n-1: read xn_[b], write tmp_[b].
    int32_t run_batch(const int32_t* tokens, int n, bool want_next, float* logits);
    bool conv_op(int layer, int n);
    bool attn_op(int layer, int n);
    bool mlp(int layer, int n);
    float* slot(int b) const { return arena_ + static_cast<std::size_t>(b) * slot_floats_; }
    float* act(float* base, int b) const { return base + static_cast<std::size_t>(b) * h_.dim; }

    // KV cache: one row per (position, attention layer) = [K | V], each
    // n_kv*head_dim values of kv_bf16 ? bf16 : f32.
    std::size_t kv_row_bytes() const;
    bool kv_put(int attn_idx, int pos, std::span<const float> k, std::span<const float> v);
    bool kv_load(int attn_idx, int j0, int count, std::byte* dst);

    void progress(int done);

    static constexpr int kMaxCandidates = 64;  // LM head shortlist cap

    Source* src_ = nullptr;
    FileHeader h_{};
    EngineConfig cfg_{};
    int pos_ = 0;
    int n_conv_ = 0;
    int n_attn_ = 0;
    int layer_slot_[kMaxLayers] = {};  // index among conv or attention layers

    // Separate allocations (a fragmented heap has no single big block).
    static constexpr int kMaxBlocks = 12;
    void* blocks_[kMaxBlocks] = {};
    int n_blocks_ = 0;
    void* alloc(std::size_t bytes);

    std::byte* io_ = nullptr;
    std::byte* io2_ = nullptr;   // second weight buffer (double_buffer), else == io_
    float* x_ = nullptr;         // residual stream [max_batch][dim]
    float* xn_ = nullptr;        // normalised input [max_batch][dim]
    float* tmp_ = nullptr;       // operator / mlp output [max_batch][dim]
    float* norm_w_ = nullptr;    // norm weights [dim]
    float* arena_ = nullptr;     // per-token op scratch [max_batch][slot_floats_]
    float* shared_ = nullptr;    // per-op scratch shared by the batch
    std::byte* carry_ = nullptr; // one weight row split across two read blocks
    int8_t* xq_ = nullptr;       // int8_activations: [max_batch][max cols] quantized inputs
    float* xs_ = nullptr;        // ... and their per-32 scales [max_batch][max cols / 32]
    std::size_t xq_cols_ = 0;    // max cols
    std::size_t slot_floats_ = 0;
    float* conv_state_ = nullptr;      // [n_conv][conv_l-1][dim], oldest first
    // KV cache, either in RAM: per attention layer [max_ctx] rows,
    // or in a file: rows appended position-major, (pos * n_attn + layer).
    std::byte* kv_ram_[kMaxLayers] = {};
    FileSource* kv_file_ = nullptr;

    LayerHook layer_hook_ = nullptr;
    void* layer_ctx_ = nullptr;
    ProgressHook progress_hook_ = nullptr;
    void* progress_ctx_ = nullptr;
};

}  // namespace lfm
