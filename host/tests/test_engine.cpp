// lfm::Engine end to end against the PyTorch golden dump.
//
// Needs the exported model files (skipped otherwise):
//   uv run tools/export.py --dtype bf16   -> out/lfm-bf16.lfm  exact weights: must match golden to fp32 noise
//   uv run tools/export.py                -> out/lfm-q8.lfm    quantized: must give the same greedy tokens

#include <gtest/gtest.h>

#include <cmath>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <fstream>
#include <string>
#include <vector>

#include "golden.hpp"
#include "lfm.hpp"
#include "lfm/engine.hpp"
#include "lfm/math/ops.hpp"

namespace {

const std::string kBf16Path = std::string(LFM_REPO_ROOT) + "/out/lfm-bf16.lfm";
const std::string kQ8Path = std::string(LFM_REPO_ROOT) + "/out/lfm-q8.lfm";
const std::string kQ8HeadPath = std::string(LFM_REPO_ROOT) + "/out/lfm-q8-head-q4.lfm";
const std::string kEvalPath = std::string(LFM_REPO_ROOT) + "/out/eval.txt";
const std::string kKvPath = std::string(LFM_REPO_ROOT) + "/out/test-kv.bin";

bool Exists(const std::string& path) { return lfm::FileSource(path.c_str()).ok(); }

#define MODEL_REQUIRED(path)                                                                 \
    do {                                                                                     \
        GOLDEN_REQUIRED();                                                                   \
        if (!Exists(path)) GTEST_SKIP() << path << " missing: run tools/export.py";          \
    } while (0)

// Collects every layer's output for the current token.
struct LayerCapture {
    std::vector<std::vector<float>> out;              // [layer]: last token seen
    std::vector<std::vector<std::vector<float>>> at;  // [pos][layer]
    static void hook(void* ctx, int layer, int pos, std::span<const float> x) {
        auto* self = static_cast<LayerCapture*>(ctx);
        self->out[static_cast<std::size_t>(layer)].assign(x.begin(), x.end());
        if (self->at.size() <= static_cast<std::size_t>(pos)) self->at.resize(static_cast<std::size_t>(pos) + 1);
        auto& row = self->at[static_cast<std::size_t>(pos)];
        if (row.size() <= static_cast<std::size_t>(layer)) row.resize(static_cast<std::size_t>(layer) + 1);
        row[static_cast<std::size_t>(layer)].assign(x.begin(), x.end());
    }
};

// Worst relative error (max|got - want| / max|want|) of one row.
float RelErr(std::span<const float> got, std::span<const float> want) {
    float diff = 0.0f, scale = 0.0f;
    for (std::size_t i = 0; i < want.size(); ++i) {
        diff = std::fmax(diff, std::fabs(got[i] - want[i]));
        scale = std::fmax(scale, std::fabs(want[i]));
    }
    return diff / scale;
}

std::vector<int32_t> Generate(lfm::Engine& eng, const std::vector<int>& prompt, int max_new) {
    int32_t next = lfm::Engine::kNoToken;
    for (std::size_t i = 0; i < prompt.size(); ++i) next = eng.forward(prompt[i], i + 1 == prompt.size());
    std::vector<int32_t> out;
    while (next >= 0 && static_cast<int>(out.size()) < max_new) {
        out.push_back(next);
        if (next == lfm::kEosId) break;
        next = eng.forward(next, true);
    }
    return out;
}

// Exactly n greedy tokens, ignoring EOS: long runs cross several KV chunks.
std::vector<int32_t> GenerateN(lfm::Engine& eng, const std::vector<int>& prompt, int n) {
    int32_t next = lfm::Engine::kNoToken;
    for (std::size_t i = 0; i < prompt.size(); ++i) next = eng.forward(prompt[i], i + 1 == prompt.size());
    std::vector<int32_t> out;
    while (next >= 0 && static_cast<int>(out.size()) < n) {
        out.push_back(next);
        next = eng.forward(next, true);
    }
    return out;
}

// start_read() hands the read to a worker thread, like the firmware's reader
// task does: exercises the engine's double buffering for real.
class ThreadedSource final : public lfm::Source {
public:
    explicit ThreadedSource(const char* path) : file_(path), worker_([this] { run(); }) {}
    ~ThreadedSource() override {
        {
            std::lock_guard lock(mu_);
            quit_ = true;
        }
        cv_.notify_all();
        worker_.join();
    }
    bool ok() const { return file_.ok(); }
    int async_reads() const { return async_reads_; }

    bool read_at(uint64_t offset, void* dst, std::size_t n) override {
        std::lock_guard lock(mu_);
        EXPECT_FALSE(busy_) << "read_at while an async read is in flight";
        return file_.read_at(offset, dst, n);
    }
    void start_read(uint64_t offset, void* dst, std::size_t n) override {
        std::lock_guard lock(mu_);
        EXPECT_FALSE(busy_) << "two async reads in flight";
        job_ = {offset, dst, n};
        busy_ = true;
        ++async_reads_;
        cv_.notify_all();
    }
    bool wait() override {
        std::unique_lock lock(mu_);
        cv_.wait(lock, [this] { return !busy_; });
        return result_;
    }

private:
    struct Job {
        uint64_t offset;
        void* dst;
        std::size_t n;
    };
    void run() {
        std::unique_lock lock(mu_);
        while (true) {
            cv_.wait(lock, [this] { return quit_ || busy_; });
            if (quit_) return;
            const Job job = job_;
            lock.unlock();
            const bool ok = file_.read_at(job.offset, job.dst, job.n);
            lock.lock();
            result_ = ok;
            busy_ = false;
            cv_.notify_all();
        }
    }

    lfm::FileSource file_;
    std::mutex mu_;
    std::condition_variable cv_;
    Job job_{};
    bool busy_ = false, quit_ = false, result_ = true;
    int async_reads_ = 0;
    std::thread worker_;
};

std::vector<int32_t> GoldenGenerated() { return golden::Manifest()["generated"].get<std::vector<int32_t>>(); }

}  // namespace

// ---------------------------------------------------------------- open / errors

TEST(Engine, MissingFileIsIoError) {
    lfm::FileSource src("/nonexistent.lfm");
    lfm::Engine eng;
    EXPECT_EQ(eng.open(src, {}), lfm::Status::IoError);
}

TEST(Engine, WrongFileIsBadMagic) {
    GOLDEN_REQUIRED();  // any big file will do: the golden manifest's weights
    const std::string path = golden::Manifest()["weights"];
    lfm::FileSource src(path.c_str());
    lfm::Engine eng;
    EXPECT_EQ(eng.open(src, {}), lfm::Status::BadMagic);
}

TEST(Engine, HeaderMatchesModelConstants) {
    MODEL_REQUIRED(kQ8Path);
    lfm::FileSource src(kQ8Path.c_str());
    lfm::Engine eng;
    ASSERT_EQ(eng.open(src, {}), lfm::Status::Ok);
    const lfm::FileHeader& h = eng.header();
    EXPECT_EQ(h.embed_dtype, lfm::Dtype::Q8_0);
    EXPECT_EQ(h.dim, static_cast<uint32_t>(lfm::kDim));
    EXPECT_EQ(h.ff_dim, static_cast<uint32_t>(lfm::kFfDim));
    EXPECT_EQ(h.n_layers, static_cast<uint32_t>(lfm::kNumLayers));
    EXPECT_EQ(h.vocab, static_cast<uint32_t>(lfm::kVocab));
    EXPECT_FLOAT_EQ(h.rope_theta, lfm::kRopeTheta);
    for (int i = 0; i < lfm::kNumLayers; ++i) {
        const bool attn = lfm::kLayerTypes[static_cast<std::size_t>(i)] == lfm::LayerType::Attention;
        EXPECT_EQ(h.layer_kind[i], attn ? lfm::LayerKind::Attention : lfm::LayerKind::Conv) << "layer " << i;
    }
}

TEST(Engine, TokenText) {
    MODEL_REQUIRED(kQ8Path);
    lfm::FileSource src(kQ8Path.c_str());
    lfm::Engine eng;
    ASSERT_EQ(eng.open(src, {}), lfm::Status::Ok);
    char buf[64];
    auto text = [&](int32_t id) { return std::string(buf, eng.token_text(id, buf, sizeof(buf))); };
    EXPECT_EQ(text(1), "<|startoftext|>");
    EXPECT_EQ(text(4481), " France");
    EXPECT_EQ(text(5242), " Paris");
    EXPECT_EQ(text(lfm::kEosId), "<|im_end|>");
    EXPECT_EQ(text(-1), "");
}

TEST(Engine, ContextFull) {
    MODEL_REQUIRED(kQ8Path);
    lfm::FileSource src(kQ8Path.c_str());
    lfm::Engine eng;
    ASSERT_EQ(eng.open(src, {.max_ctx = 2}), lfm::Status::Ok);
    EXPECT_EQ(eng.forward(1, false), lfm::Engine::kNoToken);
    EXPECT_EQ(eng.forward(1098, false), lfm::Engine::kNoToken);
    EXPECT_EQ(eng.forward(5706, false), lfm::Engine::kContextFull);
    eng.reset();
    EXPECT_EQ(eng.pos(), 0);
    EXPECT_EQ(eng.forward(1, false), lfm::Engine::kNoToken);
}

// ---------------------------------------------------------------- exact (bf16 weights)

// Every layer of every prompt token matches PyTorch to fp32 noise.
TEST(EngineBf16, EveryLayerMatchesGolden) {
    MODEL_REQUIRED(kBf16Path);
    lfm::FileSource src(kBf16Path.c_str());
    lfm::Engine eng;
    ASSERT_EQ(eng.open(src, {}), lfm::Status::Ok);
    LayerCapture cap;
    cap.out.resize(lfm::kNumLayers);
    eng.set_layer_hook(&LayerCapture::hook, &cap);

    std::vector<golden::Activation> want;
    for (int i = 0; i < lfm::kNumLayers; ++i) want.push_back(golden::Load("L" + std::to_string(i) + ".output"));

    const std::vector<int> tokens = golden::Tokens();
    for (std::size_t t = 0; t < tokens.size(); ++t) {
        eng.forward(tokens[t], false);
        for (int i = 0; i < lfm::kNumLayers; ++i) {
            ASSERT_TRUE(golden::Near(cap.out[i], want[i].row(t), 1e-4f)) << "token " << t << " layer " << i;
        }
    }
}

TEST(EngineBf16, LogitsMatchGolden) {
    MODEL_REQUIRED(kBf16Path);
    lfm::FileSource src(kBf16Path.c_str());
    lfm::Engine eng;
    ASSERT_EQ(eng.open(src, {}), lfm::Status::Ok);
    const golden::Activation want = golden::Load("logits");
    const std::vector<int> tokens = golden::Tokens();

    std::vector<float> logits(lfm::kVocab);
    for (std::size_t t = 0; t < tokens.size(); ++t) {
        const int32_t next = eng.forward(tokens[t], true, logits.data());
        EXPECT_TRUE(golden::Near(logits, want.row(t), 1e-4f)) << "token " << t;
        EXPECT_EQ(static_cast<std::size_t>(next), lfm::math::argmax(want.row(t))) << "token " << t;
    }
}

TEST(EngineBf16, GeneratesGoldenContinuation) {
    MODEL_REQUIRED(kBf16Path);
    lfm::FileSource src(kBf16Path.c_str());
    lfm::Engine eng;
    ASSERT_EQ(eng.open(src, {}), lfm::Status::Ok);
    EXPECT_EQ(Generate(eng, golden::Tokens(), 8), GoldenGenerated());
}

// ---------------------------------------------------------------- quantized (device file)

TEST(EngineQ8, GeneratesGoldenContinuation) {
    MODEL_REQUIRED(kQ8Path);
    lfm::FileSource src(kQ8Path.c_str());
    lfm::Engine eng;
    ASSERT_EQ(eng.open(src, {}), lfm::Status::Ok);
    EXPECT_EQ(Generate(eng, golden::Tokens(), 8), GoldenGenerated());
}

// The exact config the Cardputer runs: bf16 KV cache, small context.
TEST(EngineQ8, DeviceConfigGeneratesGoldenContinuation) {
    MODEL_REQUIRED(kQ8Path);
    lfm::FileSource src(kQ8Path.c_str());
    lfm::Engine eng;
    const lfm::EngineConfig cfg{.max_ctx = 12, .kv_bf16 = true, .io_bytes = 16384};
    ASSERT_EQ(eng.open(src, cfg), lfm::Status::Ok);
    EXPECT_EQ(Generate(eng, golden::Tokens(), 8), GoldenGenerated());
}

// Quantization error stays small through the whole network.
TEST(EngineQ8, LayerErrorIsBounded) {
    MODEL_REQUIRED(kQ8Path);
    lfm::FileSource src(kQ8Path.c_str());
    lfm::Engine eng;
    ASSERT_EQ(eng.open(src, {}), lfm::Status::Ok);
    LayerCapture cap;
    cap.out.resize(lfm::kNumLayers);
    eng.set_layer_hook(&LayerCapture::hook, &cap);

    const std::vector<int> tokens = golden::Tokens();
    float worst = 0.0f;
    for (std::size_t t = 0; t < tokens.size(); ++t) {
        eng.forward(tokens[t], false);
        for (int i = 0; i < lfm::kNumLayers; ++i) {
            const float e = RelErr(cap.out[i], golden::Load("L" + std::to_string(i) + ".output").row(t));
            worst = std::fmax(worst, e);
        }
    }
    RecordProperty("worst_rel_err", std::to_string(worst));
    EXPECT_LT(worst, 0.05f);
}

// ---------------------------------------------------------------- KV cache in a file

TEST(EngineBf16, KvFileLogitsMatchGolden) {
    MODEL_REQUIRED(kBf16Path);
    lfm::FileSource src(kBf16Path.c_str());
    lfm::Engine eng;
    ASSERT_EQ(eng.open(src, {.kv_path = kKvPath.c_str()}), lfm::Status::Ok);
    const golden::Activation want = golden::Load("logits");
    const std::vector<int> tokens = golden::Tokens();

    std::vector<float> logits(lfm::kVocab);
    for (std::size_t t = 0; t < tokens.size(); ++t) {
        eng.forward(tokens[t], true, logits.data());
        EXPECT_TRUE(golden::Near(logits, want.row(t), 1e-4f)) << "token " << t;
    }
}

// Same numbers whether K/V live in RAM or in a file, over a run long enough
// to need several chunks of the KV stream (16 positions per 32 KB at bf16).
class KvFileVsRam : public ::testing::TestWithParam<bool> {};

TEST_P(KvFileVsRam, SameTokens) {
    MODEL_REQUIRED(kQ8Path);
    const bool bf16 = GetParam();
    lfm::FileSource src_ram(kQ8Path.c_str()), src_file(kQ8Path.c_str());
    lfm::Engine ram, file;
    ASSERT_EQ(ram.open(src_ram, {.max_ctx = 64, .kv_bf16 = bf16}), lfm::Status::Ok);
    ASSERT_EQ(file.open(src_file, {.max_ctx = 64, .kv_bf16 = bf16, .kv_path = kKvPath.c_str()}), lfm::Status::Ok);
    const auto want = GenerateN(ram, golden::Tokens(), 40);
    ASSERT_EQ(want.size(), 40u);
    EXPECT_EQ(GenerateN(file, golden::Tokens(), 40), want);
}

INSTANTIATE_TEST_SUITE_P(Kv, KvFileVsRam, ::testing::Values(false, true),
                         [](const auto& info) { return info.param ? "Bf16" : "F32"; });

// Context is no longer bounded by RAM: reset and reuse work with the file too.
TEST(EngineQ8, KvFileResetAndRerun) {
    MODEL_REQUIRED(kQ8Path);
    lfm::FileSource src(kQ8Path.c_str());
    lfm::Engine eng;
    ASSERT_EQ(eng.open(src, {.max_ctx = 512, .kv_bf16 = true, .kv_path = kKvPath.c_str()}), lfm::Status::Ok);
    EXPECT_EQ(Generate(eng, golden::Tokens(), 8), GoldenGenerated());
    eng.reset();
    EXPECT_EQ(Generate(eng, golden::Tokens(), 8), GoldenGenerated());
}

TEST(Engine, KvFileUnwritablePathIsKvFileError) {
    MODEL_REQUIRED(kQ8Path);
    lfm::FileSource src(kQ8Path.c_str());
    lfm::Engine eng;
    EXPECT_EQ(eng.open(src, {.kv_path = "/nonexistent-dir/kv.bin"}), lfm::Status::KvFileError);
}

// ---------------------------------------------------------------- LM head shortlist

// The Q4 shortlist + exact re-score picks the same next token as scanning the
// exact Q8 head, teacher-forced over real text (tools/eval_text.py).
TEST(EngineQ8, LmHeadShortlistSameTokens) {
    MODEL_REQUIRED(kQ8HeadPath);
    MODEL_REQUIRED(kQ8Path);
    std::vector<int32_t> text;
    {
        std::ifstream f(kEvalPath);
        int32_t id;
        while (f >> id && text.size() < 96) text.push_back(id);
    }
    if (text.size() < 2) GTEST_SKIP() << kEvalPath << " missing: uv run tools/eval_text.py";

    lfm::FileSource src_full(kQ8Path.c_str()), src_short(kQ8HeadPath.c_str());
    lfm::Engine full, shortlist;
    ASSERT_EQ(full.open(src_full, {.max_ctx = 128}), lfm::Status::Ok);
    ASSERT_EQ(shortlist.open(src_short, {.max_ctx = 128}), lfm::Status::Ok);
    ASSERT_NE(shortlist.header().lm_head, 0u);
    for (std::size_t t = 0; t + 1 < text.size(); ++t) {
        ASSERT_EQ(shortlist.forward(text[t], true), full.forward(text[t], true)) << "position " << t;
    }
}

TEST(EngineQ8, LmHeadShortlistGeneratesGolden) {
    MODEL_REQUIRED(kQ8HeadPath);
    lfm::FileSource src(kQ8HeadPath.c_str());
    lfm::Engine eng;
    ASSERT_EQ(eng.open(src, {.max_ctx = 64, .kv_bf16 = true, .kv_path = kKvPath.c_str()}), lfm::Status::Ok);
    EXPECT_EQ(Generate(eng, golden::Tokens(), 8), GoldenGenerated());
}

// ---------------------------------------------------------------- overlapped reads

TEST(EngineBf16, AsyncReadsLogitsMatchGolden) {
    MODEL_REQUIRED(kBf16Path);
    ThreadedSource src(kBf16Path.c_str());
    ASSERT_TRUE(src.ok());
    lfm::Engine eng;
    ASSERT_EQ(eng.open(src, {.double_buffer = true}), lfm::Status::Ok);
    const golden::Activation want = golden::Load("logits");
    const std::vector<int> tokens = golden::Tokens();
    std::vector<float> logits(lfm::kVocab);
    for (std::size_t t = 0; t < tokens.size(); ++t) {
        eng.forward(tokens[t], true, logits.data());
        EXPECT_TRUE(golden::Near(logits, want.row(t), 1e-4f)) << "token " << t;
    }
    EXPECT_GT(src.async_reads(), 1000);
}

// Device setup: async reads, double buffer, KV file, LM head shortlist.
TEST(EngineQ8, AsyncReadsDeviceConfigSameTokens) {
    MODEL_REQUIRED(kQ8HeadPath);
    lfm::FileSource sync_src(kQ8HeadPath.c_str());
    ThreadedSource async_src(kQ8HeadPath.c_str());
    lfm::Engine a, b;
    const lfm::EngineConfig cfg{.max_ctx = 64, .kv_bf16 = true, .double_buffer = true, .kv_path = kKvPath.c_str()};
    ASSERT_EQ(a.open(sync_src, {.max_ctx = 64, .kv_bf16 = true, .double_buffer = false}), lfm::Status::Ok);
    ASSERT_EQ(b.open(async_src, cfg), lfm::Status::Ok);
    const auto want = GenerateN(a, golden::Tokens(), 24);
    EXPECT_EQ(GenerateN(b, golden::Tokens(), 24), want);
}

// ---------------------------------------------------------------- batched prefill

// The whole prompt through each layer at once matches PyTorch at every layer
// and position.
TEST(EngineBf16, BatchedPrefillEveryLayerMatchesGolden) {
    MODEL_REQUIRED(kBf16Path);
    lfm::FileSource src(kBf16Path.c_str());
    lfm::Engine eng;
    ASSERT_EQ(eng.open(src, {.max_batch = 6}), lfm::Status::Ok);
    LayerCapture cap;
    cap.out.resize(lfm::kNumLayers);
    eng.set_layer_hook(&LayerCapture::hook, &cap);

    std::vector<int32_t> tokens;
    for (int t : golden::Tokens()) tokens.push_back(t);
    std::vector<float> logits(lfm::kVocab);
    eng.prefill(tokens, true, logits.data());
    ASSERT_EQ(cap.at.size(), tokens.size());
    for (int i = 0; i < lfm::kNumLayers; ++i) {
        const golden::Activation want = golden::Load("L" + std::to_string(i) + ".output");
        for (std::size_t t = 0; t < tokens.size(); ++t) {
            ASSERT_TRUE(golden::Near(cap.at[t][i], want.row(t), 1e-4f)) << "token " << t << " layer " << i;
        }
    }
    EXPECT_TRUE(golden::Near(logits, golden::Load("logits").row(tokens.size() - 1), 1e-4f));
}

// Device setup with uneven batches (3 + 3, then 1 at a time), async reads,
// KV file, LM head shortlist: same tokens as plain one-at-a-time.
TEST(EngineQ8, BatchedPrefillDeviceConfigSameTokens) {
    MODEL_REQUIRED(kQ8HeadPath);
    lfm::FileSource sync_src(kQ8HeadPath.c_str());
    ThreadedSource async_src(kQ8HeadPath.c_str());
    lfm::Engine one, batched;
    ASSERT_EQ(one.open(sync_src, {.max_ctx = 64, .kv_bf16 = true}), lfm::Status::Ok);
    ASSERT_EQ(batched.open(async_src, {.max_ctx = 64, .kv_bf16 = true, .kv_path = kKvPath.c_str(), .max_batch = 4}),
              lfm::Status::Ok);
    const auto want = GenerateN(one, golden::Tokens(), 16);

    std::vector<int32_t> prompt;
    for (int t : golden::Tokens()) prompt.push_back(t);
    std::vector<int32_t> got;
    int32_t next = batched.prefill(prompt, true);
    while (next >= 0 && got.size() < 16) {
        got.push_back(next);
        next = batched.forward(next, true);
    }
    EXPECT_EQ(got, want);
}

TEST(EngineQ8, PrefillContextFull) {
    MODEL_REQUIRED(kQ8Path);
    lfm::FileSource src(kQ8Path.c_str());
    lfm::Engine eng;
    ASSERT_EQ(eng.open(src, {.max_ctx = 4, .max_batch = 3}), lfm::Status::Ok);
    const std::vector<int32_t> prompt = {1, 1098, 5706, 803, 4481};
    EXPECT_EQ(eng.prefill(prompt, true), lfm::Engine::kContextFull);
}

// ---------------------------------------------------------------- read alignment

// Records where streamed (start_read) reads begin.
class RecordingSource final : public lfm::Source {
public:
    explicit RecordingSource(const char* path) : file_(path) {}
    bool read_at(uint64_t offset, void* dst, std::size_t n) override { return file_.read_at(offset, dst, n); }
    void start_read(uint64_t offset, void* dst, std::size_t n) override {
        ++reads_;
        if (offset % 512 != 0) ++misaligned_;
        ok_ = file_.read_at(offset, dst, n);
    }
    bool wait() override { return ok_; }
    int reads() const { return reads_; }
    int misaligned() const { return misaligned_; }

private:
    lfm::FileSource file_;
    bool ok_ = true;
    int reads_ = 0, misaligned_ = 0;
};

// Every streamed weight read starts on an SD sector boundary, whatever the
// row size (Q8 dim rows 1088 B, w2 rows 2720 B, Q4 head rows 576 B).
TEST(EngineQ8, StreamedReadsAreSectorAligned) {
    MODEL_REQUIRED(kQ8HeadPath);
    RecordingSource src(kQ8HeadPath.c_str());
    lfm::Engine eng;
    ASSERT_EQ(eng.open(src, {.max_ctx = 16}), lfm::Status::Ok);
    EXPECT_EQ(Generate(eng, golden::Tokens(), 8), GoldenGenerated());
    EXPECT_GT(src.reads(), 1000);
    EXPECT_EQ(src.misaligned(), 0);
}

// ---------------------------------------------------------------- int8 activations + Q4 (device)

const std::string kQ4Path = std::string(LFM_REPO_ROOT) + "/out/lfm-q4.lfm";

// Integer path on the near-lossless Q8 file still produces the golden answer.
TEST(EngineQ8, Int8ActivationsGenerateGolden) {
    MODEL_REQUIRED(kQ8HeadPath);
    lfm::FileSource src(kQ8HeadPath.c_str());
    lfm::Engine eng;
    ASSERT_EQ(eng.open(src, {.max_ctx = 64, .int8_activations = true}), lfm::Status::Ok);
    EXPECT_EQ(Generate(eng, golden::Tokens(), 8), GoldenGenerated());
}

// The device file and config: Q4 everywhere, int8 activations, async reads,
// KV file, batched prompt. Same tokens with and without batching/async.
TEST(EngineQ4, DeviceConfigConsistent) {
    MODEL_REQUIRED(kQ4Path);
    lfm::FileSource sync_src(kQ4Path.c_str());
    ThreadedSource async_src(kQ4Path.c_str());
    lfm::Engine one, dev;
    ASSERT_EQ(one.open(sync_src, {.max_ctx = 64, .kv_bf16 = true, .int8_activations = true}), lfm::Status::Ok);
    ASSERT_EQ(dev.open(async_src, {.max_ctx = 64, .kv_bf16 = true, .kv_path = kKvPath.c_str(), .max_batch = 3,
                                   .int8_activations = true}),
              lfm::Status::Ok);
    const auto want = GenerateN(one, golden::Tokens(), 16);
    const std::vector<int> tokens = golden::Tokens();  // one vector: begin/end must match
    std::vector<int32_t> prompt(tokens.begin(), tokens.end());
    std::vector<int32_t> got;
    int32_t next = dev.prefill(prompt, true);
    while (next >= 0 && got.size() < 16) {
        got.push_back(next);
        next = dev.forward(next, true);
    }
    EXPECT_EQ(got, want);
    EXPECT_EQ(want.front(), 5242) << "first token should still be ' Paris'";
}

// ---------------------------------------------------------------- low-rank LM head shortlist

const std::string kQ4LowRank = std::string(LFM_REPO_ROOT) + "/out/lfm-q4-head-lr256.lfm";

// Shortlist with the rank-256 table, re-score the top 256 exactly: the next
// token matches the full head scan at nearly every position of real text
// (tools/sparsity_eval.py predicted 99.8%).
TEST(EngineQ4, LowRankHeadAgreesWithFullScan) {
    MODEL_REQUIRED(kQ4LowRank);
    MODEL_REQUIRED(kQ4Path);
    std::vector<int32_t> text;
    {
        std::ifstream f(kEvalPath);
        int32_t id;
        while (f >> id) text.push_back(id);
    }
    if (text.size() < 2) GTEST_SKIP() << kEvalPath << " missing: uv run tools/eval_text.py";

    lfm::FileSource src_full(kQ4Path.c_str()), src_lr(kQ4LowRank.c_str());
    lfm::Engine full, lr;
    const lfm::EngineConfig cfg{.max_ctx = 512, .int8_activations = true};
    ASSERT_EQ(full.open(src_full, cfg), lfm::Status::Ok);
    ASSERT_EQ(lr.open(src_lr, cfg), lfm::Status::Ok);
    ASSERT_GT(lr.header().lm_rank, 0u);
    int agree = 0, total = 0;
    for (std::size_t t = 0; t + 1 < text.size(); ++t) {
        agree += lr.forward(text[t], true) == full.forward(text[t], true);
        ++total;
    }
    RecordProperty("agreement", std::to_string(static_cast<double>(agree) / total));
    EXPECT_GE(agree, total * 98 / 100) << agree << " / " << total;
}

TEST(EngineQ4, LowRankHeadDeviceConfigGeneratesGolden) {
    MODEL_REQUIRED(kQ4LowRank);
    ThreadedSource src(kQ4LowRank.c_str());
    lfm::Engine eng;
    ASSERT_EQ(eng.open(src, {.max_ctx = 64, .kv_bf16 = true, .kv_path = kKvPath.c_str(), .max_batch = 4,
                             .int8_activations = true, .speculative = true}),
              lfm::Status::Ok);
    EXPECT_EQ(Generate(eng, golden::Tokens(), 8), GoldenGenerated());
}
