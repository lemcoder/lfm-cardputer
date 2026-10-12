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
//   for (prompt tokens except last) eng.forward(t, false);
//   int32_t next = eng.forward(last, true);   // greedy next token

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
    std::size_t io_bytes = 32768;   // weight / KV streaming buffer
    // Keep the KV cache in this file (created / emptied on open) instead of
    // RAM. RAM use then doesn't depend on max_ctx. Each token appends
    // n_attn_layers * 2 * n_kv*head_dim values and attention reads them back.
    const char* kv_path = nullptr;
};

class Engine {
public:
    // Called after every layer with the residual stream (for tests/debugging).
    using LayerHook = void (*)(void* ctx, int layer, std::span<const float> x);
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

    // Bytes of token `id` (UTF-8 / raw bytes) into buf. Returns the length,
    // truncated to cap. Not NUL-terminated.
    std::size_t token_text(int32_t id, char* buf, std::size_t cap);

    // Bytes of RAM open() allocates for a given config (header must be read).
    std::size_t ram_bytes(const EngineConfig& cfg) const;

    void set_layer_hook(LayerHook fn, void* ctx) { layer_hook_ = fn; layer_ctx_ = ctx; }
    void set_progress_hook(ProgressHook fn, void* ctx) { progress_hook_ = fn; progress_ctx_ = ctx; }

private:
    bool read_f32(uint64_t offset, std::span<float> out);
    bool matvec(std::span<float> out, uint64_t offset, Dtype dt, std::size_t cols, std::span<const float> x);
    int32_t lm_head(std::span<const float> x, float* logits);

    bool conv_op(int layer, std::span<float> out, std::span<const float> in);
    bool attn_op(int layer, std::span<float> out, std::span<const float> in);
    bool mlp(int layer, std::span<float> out, std::span<const float> in);

    // KV cache: one row per (position, attention layer) = [K | V], each
    // n_kv*head_dim values of kv_bf16 ? bf16 : f32.
    std::size_t kv_row_bytes() const;
    bool kv_put(int attn_idx, std::span<const float> k, std::span<const float> v);
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

    // One allocation, carved up in open(). Layout in engine.cpp.
    void* mem_ = nullptr;
    std::byte* io_ = nullptr;
    float* x_ = nullptr;         // residual stream [dim]
    float* xn_ = nullptr;        // normalised input [dim]
    float* tmp_ = nullptr;       // operator / mlp output [dim]
    float* norm_w_ = nullptr;    // norm weights [dim]
    float* arena_ = nullptr;     // per-op scratch
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
