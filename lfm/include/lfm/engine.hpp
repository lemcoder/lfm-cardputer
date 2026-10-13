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
    // Allow verify()/accept() (speculative decoding). Rolling back needs each
    // conv layer's state history: (conv_l - 1 + max_batch) rows of dim floats
    // per conv layer, in RAM, or in this file if set (the device: ~160 KB).
    bool speculative = false;
    const char* conv_history_path = nullptr;
    // Keep the conv layers' state (conv_l-1 rows of dim floats each, 64 KB
    // for LFM2.5-230M) in this file instead of RAM. Each conv layer then
    // reads and writes its 8 KB once per pass: ~128 KB of I/O per pass,
    // traded for RAM that a bigger prompt batch can use.
    const char* conv_state_path = nullptr;
    // Prompt cache (snapshot_prefix / restore_prefix): where the conv state
    // snapshot lives when the conv state is in a file. nullptr: RAM.
    const char* prefix_path = nullptr;
};

// Runs a job on another core while the caller works on its own share (the
// device: a task on core 1, idle in the LM head; host tests: a thread).
class Worker {
public:
    virtual ~Worker() = default;
    virtual void start(void (*fn)(void*), void* arg) = 0;  // one job at a time
    virtual void wait() = 0;                               // until it has finished
};

class Engine {
public:
    static constexpr int kMaxBatch = 8;  // EngineConfig::max_batch cap

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
    int max_batch() const { return cfg_.max_batch; }
    int pos() const { return pos_; }  // tokens consumed so far

    // Forget all tokens (new sequence).
    void reset();

    // forward() results that aren't token ids.
    static constexpr int32_t kNoToken = -1;      // want_next was false
    static constexpr int32_t kIoError = -2;
    static constexpr int32_t kContextFull = -3;  // pos() == max_ctx()
    static constexpr int32_t kInvalid = -4;      // bad verify() arguments / not enabled

    // Feed `token` at position pos() and advance.
    // With want_next, also runs the LM head and returns the greedy next token
    // (and, if `logits` is non-null, writes all vocab logits there).
    int32_t forward(int32_t token, bool want_next, float* logits = nullptr);

    // Feed several tokens (e.g. the prompt), max_batch at a time; same result
    // as calling forward() on each, but with fewer weight reads. With
    // want_next, returns the greedy token after the last one.
    int32_t prefill(std::span<const int32_t> tokens, bool want_next, float* logits = nullptr);

    // Speculative decoding (EngineConfig::speculative). verify() feeds
    // tokens[0] (the last real token) and drafts tokens[1..] at positions
    // pos() .. pos()+n-1 in one pass over the weights (n <= max_batch), and
    // writes the greedy next token after each position into preds[0..n).
    // Draft i is right iff tokens[i+1] == preds[i]. Then accept(k) keeps the
    // first k positions (1 <= k <= n) and rolls the rest back; call it before
    // anything else. Returns preds[n-1], or a negative code.
    // first_pred: only positions first_pred.. get a prediction (the earlier
    // ones are known tokens, e.g. the end of the prompt): saves their LM head.
    int32_t verify(std::span<const int32_t> tokens, int32_t* preds, int first_pred = 0);
    bool accept(int keep);

    // Tree verification: the drafts form a tree instead of a chain.
    // tokens[0] is the last real token; parents[b] (b >= 1) is the index of
    // token b's parent, always < b. Token b sits at position pos() + its depth
    // and sees the context plus its own ancestors, so preds[b] is the greedy
    // token after the path root..b. Then accept_path(path, len) keeps one
    // root-to-node path (path[0] == 0, each entry the parent of the next) and
    // rolls the rest back. verify() is the chain case.
    int32_t verify_tree(std::span<const int32_t> tokens, const int8_t* parents, int32_t* preds);
    bool accept_path(const int8_t* path, int len);

    // Bytes of token `id` (UTF-8 / raw bytes) into buf. Returns the length,
    // truncated to cap. Not NUL-terminated.
    std::size_t token_text(int32_t id, char* buf, std::size_t cap);

    // Bytes of RAM open() allocates for a given config (header must be read).
    std::size_t ram_bytes(const EngineConfig& cfg) const;

    // LM head shortlist table in memory (the device: memory-mapped internal
    // flash; tools/flash_head.py). Used instead of the file's LM head for
    // greedy tokens. False (and not used) if it doesn't match the model.
    bool set_flash_head(const void* data, std::size_t size);

    // Draft heads (lfm/draft.hpp) run between passes and borrow the engine's
    // buffers; everything here is valid until the next forward / prefill / verify.
    //
    // Final hidden state (after the final norm) of the newest token whose
    // prediction is known: prefill's last token, or verify's last accepted one.
    const float* last_hidden() const { return hidden_slot_ >= 0 ? xn_[hidden_slot_] : nullptr; }
    // Free per-token scratch: i in [0, max_batch - 1), scratch_floats() each.
    float* scratch(int i) const { return i >= 0 && i + 1 < cfg_.max_batch ? slots_[i + 1] : nullptr; }
    std::size_t scratch_floats() const { return slot_floats_; }
    // out[r] = row r . x, for a matrix already in memory (memory-mapped flash).
    void matvec_mem(const std::byte* rows, Dtype dt, std::size_t n_rows, std::size_t cols, const float* x, float* out);
    // Embedding row of `token`, dequantized (from the flash head's cache if
    // there, else from the Source).
    bool embedding(int32_t token, float* out);
    // Greedy token for a raw (not yet final-normed) hidden state h, for drafts.
    // With the flash head: its low-rank scan, top kDraftCandidates re-scored
    // exactly (close to, not always, the exact argmax). work: dim floats.
    // top (optional): the m <= 4 best of the re-scored candidates, best first.
    int32_t draft_token(const float* h, float* work, int32_t* top = nullptr, int m = 0);
    static constexpr int kDraftCandidates = 16;

    // Counters for experiments (since open()).
    struct Stats {
        double lm_head_s = 0;   // wall time in the LM head (scan + re-score)
        long rescore_reads = 0; // embedding rows read to re-score shortlists
        double flash_scan_s = 0; // part of lm_head_s: projection + flash table scan
        double flash_proj_s = 0; // part of flash_scan_s: the projection
        double topk_s = 0;       // part of flash_scan_s: top-k inserts
        double draft_head_s = 0; // draft heads' LM head (draft_token), not in lm_head_s
    };
    const Stats& stats() const { return stats_; }

    // The engine's own files (KV cache, conv state + history): time and
    // operation counts since open().
    struct FileIo {
        double kv_s = 0, conv_s = 0, conv_write_s = 0;
        long kv_ops = 0, conv_reads = 0, conv_writes = 0;
    };
    FileIo file_io() const;

    // Prompt cache. snapshot_prefix() remembers the state after the tokens
    // fed so far (pos()); restore_prefix() returns to it later, as long as
    // nothing has been written at positions below it since (generation only
    // writes later positions, so "run the same prompt again" qualifies).
    // The K/V rows stay where they are; only the conv state and pos move.
    bool snapshot_prefix();
    bool restore_prefix();
    int prefix_pos() const { return prefix_pos_; }  // -1: none

    // The flash LM head table scan is split with this worker (optional).
    void set_worker(Worker* w) { worker_ = w; }

    void set_layer_hook(LayerHook fn, void* ctx) { layer_hook_ = fn; layer_ctx_ = ctx; }
    void set_progress_hook(ProgressHook fn, void* ctx) { progress_hook_ = fn; progress_ctx_ = ctx; }

private:
    bool read_f32(uint64_t offset, std::span<float> out);
    // One pointer per batch token (scratch slots are separate allocations).
    struct Ptrs {
        float* p[kMaxBatch];
    };
    // For each of n tokens b: out.p[b][r] = row r · x.p[b][0..cols).
    bool matvec(const Ptrs& out, std::size_t rows, uint64_t offset, Dtype dt, std::size_t cols, const Ptrs& x, int n);
    Ptrs slots(std::size_t offset) const;  // slot b + offset, for every b
    Ptrs acts(float* const* base) const;   // base[b], for every b
    // Stream `rows` rows of `rb` bytes from `offset`, calling fn(row index,
    // row bytes) for each, with the next chunk's read overlapping fn's work.
    template <typename Fn>
    bool stream_rows(uint64_t offset, std::size_t rows, std::size_t rb, Fn&& fn);
    // Greedy next token for each of n normalised vectors (one shared scan);
    // logits (n == 1 only) gets every vocab score.
    bool lm_head(const Ptrs& xs, int n, int32_t* out, float* logits);
    bool lm_head_impl(const Ptrs& xs, int n, int32_t* out, float* logits);
    // lazy_tokens (verify): re-score only along the path that can still be
    // accepted (the batch's tokens and parent_); other outputs stay kNoToken.
    bool lm_head_flash(const Ptrs& xs, int n, int32_t* out, int candidates, int32_t* top = nullptr, int m = 0,
                       const int32_t* lazy_tokens = nullptr);
    struct ScanJob {
        const std::byte* table;
        std::size_t trb, rank;
        uint32_t v_begin, v_end;
        int n, k;
        const int8_t* xq;
        std::size_t xq_cols;
        const float* xs;
        float* ts[kMaxBatch];
        int32_t* ti[kMaxBatch];
        int n_top[kMaxBatch];
        double topk_s;
    };
    static void scan_job(void* arg);
    void quantize_inputs(const Ptrs& in, int n, std::size_t cols);
    float score_row(int b, Dtype dt, const std::byte* row, const float* in, std::size_t cols) const;
    bool rescore(const Ptrs& xs, int n, int32_t* const* ids, const int* count, int32_t* out, int32_t* top = nullptr,
                 int m = 0);

    // n tokens at positions pos_ .. pos_+n-1: read xn_[b], write tmp_[b].
    // preds != nullptr: next token after every position (verify).
    int32_t run_batch(const int32_t* tokens, int n, bool want_next, float* logits, int32_t* preds, int first_pred = 0);
    std::size_t hist_rows() const;
    bool hist_put(int slot, int row, const float* src);
    bool hist_get(int slot, int row, float* dst);
    float* state_load(int slot);                       // conv state, in RAM or staged in io_
    bool state_store(int slot, const float* state);
    bool conv_op(int layer, int n);
    bool attn_op(int layer, int n);
    bool mlp(int layer, int n);
    float* slot(int b) const { return slots_[b]; }
    float* act(float* const* base, int b) const { return base[b]; }

    // KV cache: one row per (position, attention layer) = [K | V], each
    // n_kv*head_dim values of kv_bf16 ? bf16 : f32.
    std::size_t kv_row_bytes() const;
    bool kv_copy(int attn_idx, int from, int to);

    // The batch's shape (run_batch): per token its parent in the batch (-1:
    // the context before it), depth, and ancestors-or-self as a bit mask. A
    // chain unless verify_tree() set a tree.
    int8_t parent_[kMaxBatch] = {};
    int8_t depth_[kMaxBatch] = {};
    uint8_t anc_[kMaxBatch] = {};
    bool tree_ = false;
    void set_chain(int n);
    bool set_tree(const int8_t* parents, int n);
    bool kv_put(int attn_idx, int pos, std::span<const float> k, std::span<const float> v);
    bool kv_load(int attn_idx, int j0, int count, std::byte* dst);

    void progress(int done);

    static constexpr int kMaxCandidates = 1024;  // LM head shortlist cap
    // LM head scratch inside each token's slot: z (low-rank projection, up
    // to kTopKOffset floats), then the top-k scores and ids.
    static constexpr std::size_t kTopKOffset = 512;

    Source* src_ = nullptr;
    FileHeader h_{};
    EngineConfig cfg_{};
    int pos_ = 0;
    int n_conv_ = 0;
    int n_attn_ = 0;
    int layer_slot_[kMaxLayers] = {};  // index among conv or attention layers

    // Separate allocations (a fragmented heap has no single big block).
    static constexpr int kMaxBlocks = 12 + 4 * kMaxBatch;
    void* blocks_[kMaxBlocks] = {};
    int n_blocks_ = 0;
    void* alloc(std::size_t bytes);

    std::byte* io_ = nullptr;
    std::byte* io2_ = nullptr;   // second weight buffer (double_buffer), else == io_
    // Per batch token, each its own small block (a fragmented heap has no
    // large ones): residual stream, normalised input, operator / mlp output.
    float* x_[kMaxBatch] = {};
    float* xn_[kMaxBatch] = {};
    float* tmp_[kMaxBatch] = {};
    float* norm_w_ = nullptr;    // norm weights [dim]
    float* slots_[kMaxBatch] = {};  // per-token op scratch, slot_floats_ each
    float* shared_ = nullptr;    // per-op scratch shared by the batch
    std::byte* carry_ = nullptr; // one weight row split across two read blocks
    int8_t* xq_ = nullptr;       // int8_activations: [max_batch][max cols] quantized inputs
    float* xs_ = nullptr;        // ... and their per-32 scales [max_batch][max cols / 32]
    std::size_t xq_cols_ = 0;    // max cols
    std::size_t slot_floats_ = 0;
    float* conv_state_[kMaxLayers] = {};  // per conv layer: [conv_l-1][dim], oldest first
    // KV cache, either in RAM: per attention layer [max_ctx] rows,
    // or in a file: rows appended position-major, (pos * n_attn + layer).
    std::byte* kv_ram_[kMaxLayers] = {};
    FileSource* kv_file_ = nullptr;
    // Speculative decoding: conv history, verify() state.
    float* hist_ram_ = nullptr;
    FileSource* hist_file_ = nullptr;
    FileSource* state_file_ = nullptr;
    FileSource* prefix_file_ = nullptr;   // conv state snapshot (state in a file)
    float* prefix_ram_ = nullptr;         // conv state snapshot (state in RAM)
    int prefix_pos_ = -1;
    // Cross-stream prefetch (stream_rows): the pass's matrix order, and the
    // state of an async read that may be in flight.
    struct Stream {
        uint64_t offset;
        std::size_t bytes;
    };
    static constexpr int kMaxStreams = 8 * kMaxLayers;
    Stream sched_[kMaxStreams] = {};
    int n_sched_ = 0;
    bool pf_pending_ = false;  // an async read is in flight (wait() not yet called)
    bool pf_ok_ = true;        // result of the last wait()
    bool pf_valid_ = false;    // the next stream's first block is in io_[pf_buf_]
    uint64_t pf_offset_ = 0;
    std::size_t pf_len_ = 0;
    int pf_buf_ = 0;
    void drain();
    void cancel_prefetch();
    const Stream* next_stream(uint64_t offset) const;

    // Memory-mapped LM head shortlist table (set_flash_head).
    FlashHeadHeader fh_{};
    const std::byte* fh_proj_ = nullptr;
    const std::byte* fh_table_ = nullptr;
    const std::byte* fh_cache_ = nullptr;
    int hidden_slot_ = -1;  // last_hidden(): xn_ slot, -1 = none
    bool recording_ = false;
    int pending_start_ = 0;
    int pending_n_ = 0;

    Stats stats_;
    Worker* worker_ = nullptr;
    LayerHook layer_hook_ = nullptr;
    void* layer_ctx_ = nullptr;
    ProgressHook progress_hook_ = nullptr;
    void* progress_ctx_ = nullptr;
};

}  // namespace lfm
