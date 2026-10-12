// lfm::Engine end to end against the PyTorch golden dump.
//
// Needs the exported model files (skipped otherwise):
//   uv run tools/export.py --dtype bf16   -> out/lfm-bf16.lfm  exact weights: must match golden to fp32 noise
//   uv run tools/export.py                -> out/lfm-q8.lfm    quantized: must give the same greedy tokens

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "golden.hpp"
#include "lfm.hpp"
#include "lfm/engine.hpp"
#include "lfm/math/ops.hpp"

namespace {

const std::string kBf16Path = std::string(LFM_REPO_ROOT) + "/out/lfm-bf16.lfm";
const std::string kQ8Path = std::string(LFM_REPO_ROOT) + "/out/lfm-q8.lfm";
const std::string kKvPath = std::string(LFM_REPO_ROOT) + "/out/test-kv.bin";

bool Exists(const std::string& path) { return lfm::FileSource(path.c_str()).ok(); }

#define MODEL_REQUIRED(path)                                                                 \
    do {                                                                                     \
        GOLDEN_REQUIRED();                                                                   \
        if (!Exists(path)) GTEST_SKIP() << path << " missing: run tools/export.py";          \
    } while (0)

// Collects every layer's output for the current token.
struct LayerCapture {
    std::vector<std::vector<float>> out;
    static void hook(void* ctx, int layer, std::span<const float> x) {
        auto* self = static_cast<LayerCapture*>(ctx);
        self->out[static_cast<std::size_t>(layer)].assign(x.begin(), x.end());
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
    EXPECT_EQ(h.matrix_dtype, lfm::Dtype::Q8_0);
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

TEST(Engine, KvFileUnwritablePathIsIoError) {
    MODEL_REQUIRED(kQ8Path);
    lfm::FileSource src(kQ8Path.c_str());
    lfm::Engine eng;
    EXPECT_EQ(eng.open(src, {.kv_path = "/nonexistent-dir/kv.bin"}), lfm::Status::IoError);
}
