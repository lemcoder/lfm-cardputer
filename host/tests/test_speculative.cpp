// Speculative decoding must produce exactly the plain greedy tokens: drafts
// only change how many tokens one pass yields. Rejected drafts exercise the
// conv-state rollback in Engine::accept().

#include <gtest/gtest.h>

#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "golden.hpp"
#include "lfm.hpp"
#include "lfm/draft.hpp"
#include "lfm/engine.hpp"

namespace {

const std::string kRoot = LFM_REPO_ROOT;
const std::string kQ4 = kRoot + "/out/lfm-q4.lfm";
const std::string kBf16 = kRoot + "/out/lfm-bf16.lfm";
const std::string kNgram = kRoot + "/out/ngram.bin";
const std::string kHistPath = kRoot + "/out/test-conv-hist.bin";
const std::string kKvPath2 = kRoot + "/out/test-kv2.bin";

bool Exists(const std::string& p) { return std::ifstream(p).good(); }

std::vector<int32_t> Prompt() {
    const std::vector<int> t = golden::Tokens();
    return {t.begin(), t.end()};
}

// Plain greedy: prompt, then n tokens one at a time.
std::vector<int32_t> Plain(const std::string& path, lfm::EngineConfig cfg, int n) {
    lfm::FileSource src(path.c_str());
    lfm::Engine eng;
    EXPECT_EQ(eng.open(src, cfg), lfm::Status::Ok);
    const auto prompt = Prompt();
    int32_t next = eng.prefill(prompt, true);
    std::vector<int32_t> out;
    while (next >= 0 && static_cast<int>(out.size()) < n) {
        out.push_back(next);
        next = eng.forward(next, true);
    }
    return out;
}

std::vector<char> ReadFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), {}};
}

// Speculative: same prompt, then speculative_step until n tokens.
std::vector<int32_t> Speculative(const std::string& path, lfm::EngineConfig cfg, int n, int max_draft,
                                 const std::vector<char>* ngram, lfm::SpecStats* stats) {
    lfm::FileSource src(path.c_str());
    lfm::Engine eng;
    cfg.speculative = true;
    EXPECT_EQ(eng.open(src, cfg), lfm::Status::Ok);
    std::vector<int32_t> hist(1024);
    lfm::Drafter drafter(hist.data(), static_cast<int>(hist.size()));
    if (ngram != nullptr) EXPECT_TRUE(drafter.set_ngram(ngram->data(), ngram->size()));

    const auto prompt = Prompt();
    for (int32_t t : prompt) drafter.push(t);
    int32_t fed = eng.prefill(prompt, true);
    std::vector<int32_t> out{fed};
    drafter.push(fed);
    int32_t buf[lfm::Engine::kMaxBatch];
    while (static_cast<int>(out.size()) < n) {
        const int got = lfm::speculative_step(eng, drafter, fed, max_draft, buf, stats);
        EXPECT_GT(got, 0);
        if (got <= 0) break;
        for (int i = 0; i < got; ++i) out.push_back(buf[i]);
        fed = buf[got - 1];
    }
    out.resize(static_cast<std::size_t>(n));
    return out;
}

#define REQUIRE(path)                                                                   \
    do {                                                                                \
        GOLDEN_REQUIRED();                                                              \
        if (!Exists(path)) GTEST_SKIP() << path << " missing";                          \
    } while (0)

}  // namespace

TEST(Drafter, PromptLookupProposesWhatFollowedBefore) {
    int32_t hist[16];
    lfm::Drafter d(hist, 16);
    for (int32_t t : {10, 11, 12, 13, 14, 10, 11}) d.push(t);
    int32_t out[4];
    ASSERT_EQ(d.draft(out, 3), 3);  // "10 11" seen before, followed by 12 13 14
    EXPECT_EQ(out[0], 12);
    EXPECT_EQ(out[1], 13);
    EXPECT_EQ(out[2], 14);
}

TEST(Drafter, RejectsBadTable) {
    int32_t hist[4];
    lfm::Drafter d(hist, 4);
    const char junk[64] = {};
    EXPECT_FALSE(d.set_ngram(junk, sizeof(junk)));
    int32_t out[2];
    d.push(5);
    EXPECT_EQ(d.draft(out, 2), 0);
}

TEST(Drafter, NgramTableLoadsAndPredicts) {
    if (!Exists(kNgram)) GTEST_SKIP() << "uv run tools/spec_eval.py --export out/ngram.bin";
    const auto table = ReadFile(kNgram);
    int32_t hist[8];
    lfm::Drafter d(hist, 8);
    ASSERT_TRUE(d.set_ngram(table.data(), table.size()));
    for (int32_t t : {1098, 5706, 803}) d.push(t);  // "The capital of"
    int32_t out[3];
    EXPECT_EQ(d.draft(out, 3), 3);
}

// Device configuration: Q4, int8 activations, KV + conv history in files,
// prompt batch 4, real drafts (lookup, else n-gram), up to 3 per pass.
TEST(Speculative, DeviceConfigSameTokensAsGreedy) {
    REQUIRE(kQ4);
    const auto table = Exists(kNgram) ? ReadFile(kNgram) : std::vector<char>{};
    lfm::EngineConfig cfg{.max_ctx = 128, .kv_bf16 = true, .kv_path = kKvPath2.c_str(), .max_batch = 4,
                          .int8_activations = true};
    const auto want = Plain(kQ4, cfg, 48);
    cfg.conv_history_path = kHistPath.c_str();
    lfm::SpecStats st;
    EXPECT_EQ(Speculative(kQ4, cfg, 48, 3, table.empty() ? nullptr : &table, &st), want);
    RecordProperty("tokens_per_pass", std::to_string(static_cast<double>(st.tokens) / st.passes));
    EXPECT_GT(st.accepted, 0) << "no draft was ever accepted";
    EXPECT_LT(st.accepted, st.drafted) << "test should also exercise rejections";
}

// Exact float path, conv history in RAM.
TEST(Speculative, Bf16SameTokensAsGreedy) {
    REQUIRE(kBf16);
    const auto table = Exists(kNgram) ? ReadFile(kNgram) : std::vector<char>{};
    lfm::EngineConfig cfg{.max_ctx = 64, .max_batch = 3};
    const auto want = Plain(kBf16, cfg, 24);
    lfm::SpecStats st;
    EXPECT_EQ(Speculative(kBf16, cfg, 24, 2, table.empty() ? nullptr : &table, &st), want);
}

// Every draft wrong: every pass rolls back all but the first position.
TEST(Speculative, AllDraftsRejectedStillExact) {
    REQUIRE(kQ4);
    lfm::EngineConfig cfg{.max_ctx = 64, .max_batch = 4, .int8_activations = true, .speculative = true};
    const auto want = Plain(kQ4, cfg, 12);

    lfm::FileSource src(kQ4.c_str());
    lfm::Engine eng;
    ASSERT_EQ(eng.open(src, cfg), lfm::Status::Ok);
    int32_t fed = eng.prefill(Prompt(), true);
    std::vector<int32_t> got{fed};
    while (got.size() < want.size()) {
        const int32_t tokens[4] = {fed, 3, 3, 3};  // token 3: never the greedy choice here
        int32_t preds[4];
        ASSERT_GE(eng.verify(tokens, preds), 0);
        ASSERT_NE(preds[0], 3);
        ASSERT_TRUE(eng.accept(1));
        fed = preds[0];
        got.push_back(fed);
    }
    EXPECT_EQ(got, want);
}

// Perfect drafts (the true continuation): every pass accepts everything.
TEST(Speculative, PerfectDraftsAllAccepted) {
    REQUIRE(kQ4);
    lfm::EngineConfig cfg{.max_ctx = 64, .max_batch = 4, .int8_activations = true, .speculative = true};
    const auto want = Plain(kQ4, cfg, 13);

    lfm::FileSource src(kQ4.c_str());
    lfm::Engine eng;
    ASSERT_EQ(eng.open(src, cfg), lfm::Status::Ok);
    int32_t fed = eng.prefill(Prompt(), true);
    ASSERT_EQ(fed, want[0]);
    std::size_t i = 0;  // want[i] == fed
    while (i + 4 < want.size()) {
        const int32_t tokens[4] = {want[i], want[i + 1], want[i + 2], want[i + 3]};
        int32_t preds[4];
        ASSERT_GE(eng.verify(tokens, preds), 0);
        for (int j = 0; j < 3; ++j) ASSERT_EQ(preds[j], want[i + 1 + j]) << "pass at " << i << " draft " << j;
        ASSERT_TRUE(eng.accept(4));
        ASSERT_EQ(preds[3], want[i + 4]);
        i += 4;
    }
}

TEST(Speculative, VerifyNeedsSpeculativeConfig) {
    REQUIRE(kQ4);
    lfm::FileSource src(kQ4.c_str());
    lfm::Engine eng;
    ASSERT_EQ(eng.open(src, {.max_batch = 2}), lfm::Status::Ok);
    const int32_t tokens[2] = {1, 1098};
    int32_t preds[2];
    EXPECT_EQ(eng.verify(tokens, preds), lfm::Engine::kInvalid);
    EXPECT_FALSE(eng.accept(1));
}
