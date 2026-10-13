// Speculative decoding must produce exactly the plain greedy tokens: drafts
// only change how many tokens one pass yields. Rejected drafts exercise the
// conv-state rollback in Engine::accept().

#include <gtest/gtest.h>

#include <fstream>
#include <thread>
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
const std::string kDevice = kRoot + "/out/lfm-q4-head-lr256.lfm";  // the device's model file
const std::string kFlashHead = kRoot + "/out/lmhead.bin";          // tools/flash_head.py
const std::string kDraftHead = kRoot + "/out/dhead.bin";           // tools/draft_heads.py export
const std::string kHistPath = kRoot + "/out/test-conv-hist.bin";
const std::string kKvPath2 = kRoot + "/out/test-kv2.bin";
const std::string kStatePath = kRoot + "/out/test-conv-state.bin";

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

// Speculative: same prompt, then speculative_step until n tokens. Drafts:
// prompt lookup, then the draft head if out/dhead.bin exists.
std::vector<int32_t> Speculative(const std::string& path, lfm::EngineConfig cfg, int n, int max_draft,
                                 bool head, lfm::SpecStats* stats) {
    lfm::FileSource src(path.c_str());
    lfm::Engine eng;
    cfg.speculative = true;
    EXPECT_EQ(eng.open(src, cfg), lfm::Status::Ok);
    std::vector<int32_t> hist(1024);
    lfm::Drafter drafter(hist.data(), static_cast<int>(hist.size()));
    const auto head_file = head && Exists(kDraftHead) ? ReadFile(kDraftHead) : std::vector<char>{};
    lfm::DraftHead draft_head;
    if (!head_file.empty() && draft_head.set(head_file.data(), head_file.size(), eng)) drafter.set_head(&draft_head);

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

// The device's second core, as a thread.
class ThreadWorker final : public lfm::Worker {
public:
    void start(void (*fn)(void*), void* arg) override { t_ = std::thread(fn, arg); }
    void wait() override { t_.join(); }

private:
    std::thread t_;
};

// The device setup: flash LM head, draft head after prompt lookup. Greedy
// (with the same flash LM head) vs speculative, on a chat turn.
struct HeadRun {
    std::vector<int32_t> plain, spec;
    lfm::SpecStats st;
};
HeadRun RunWithDraftHead(int n, int max_draft, lfm::Worker* worker = nullptr, const lfm::DraftShape* shape = nullptr) {
    const auto flash = ReadFile(kFlashHead);
    const auto dhead = ReadFile(kDraftHead);
    // "<|startoftext|><|im_start|>user\nTell me about the planets.<|im_end|>\n<|im_start|>assistant\n"
    const std::vector<int32_t> prompt = {1, 6, 6423, 708, 52486, 1123, 1337, 779, 19603, 523, 7, 708, 6, 64015, 708};
    lfm::EngineConfig cfg{.max_ctx = 256, .kv_bf16 = true, .kv_path = kKvPath2.c_str(), .max_batch = 7,
                          .int8_activations = true, .speculative = true, .conv_history_path = kHistPath.c_str(),
                          .conv_state_path = kStatePath.c_str()};
    HeadRun r;
    {
        lfm::FileSource src(kDevice.c_str());
        lfm::Engine eng;
        EXPECT_EQ(eng.open(src, cfg), lfm::Status::Ok);
        EXPECT_TRUE(eng.set_flash_head(flash.data(), flash.size()));
        int32_t next = eng.prefill(prompt, true);  // plain greedy: serial scan
        while (next >= 0 && static_cast<int>(r.plain.size()) < n) {
            r.plain.push_back(next);
            next = eng.forward(next, true);
        }
    }
    lfm::FileSource src(kDevice.c_str());
    lfm::Engine eng;
    EXPECT_EQ(eng.open(src, cfg), lfm::Status::Ok);
    EXPECT_TRUE(eng.set_flash_head(flash.data(), flash.size()));
    eng.set_worker(worker);
    lfm::DraftHead head;
    EXPECT_TRUE(head.set(dhead.data(), dhead.size(), eng));
    std::vector<int32_t> hist(1024);
    lfm::Drafter drafter(hist.data(), static_cast<int>(hist.size()));
    drafter.set_head(&head);
    drafter.set_shape(shape);
    for (int32_t t : prompt) drafter.push(t);
    int32_t fed = eng.prefill(prompt, true);
    r.spec.push_back(fed);
    drafter.push(fed);
    int32_t buf[lfm::Engine::kMaxBatch];
    while (static_cast<int>(r.spec.size()) < n) {
        const int got = lfm::speculative_step(eng, drafter, fed, max_draft, buf, &r.st);
        EXPECT_GT(got, 0);
        if (got <= 0) break;
        for (int i = 0; i < got; ++i) r.spec.push_back(buf[i]);
        fed = buf[got - 1];
    }
    r.spec.resize(r.plain.size());
    return r;
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
    ASSERT_EQ(d.lookup(out, 3), 3);  // "10 11" seen before, followed by 12 13 14
    EXPECT_EQ(out[0], 12);
    EXPECT_EQ(out[1], 13);
    EXPECT_EQ(out[2], 14);
}

// Device configuration: Q4, int8 activations, KV + conv history in files,
// prompt batch 4, real drafts (lookup, else the draft head), up to 3 per pass.
TEST(Speculative, DeviceConfigSameTokensAsGreedy) {
    REQUIRE(kQ4);
    if (!Exists(kDraftHead)) GTEST_SKIP() << kDraftHead << " missing (tools/draft_heads.py export)";
    lfm::EngineConfig cfg{.max_ctx = 128, .kv_bf16 = true, .kv_path = kKvPath2.c_str(), .max_batch = 4,
                          .int8_activations = true};
    const auto want = Plain(kQ4, cfg, 48);
    cfg.conv_history_path = kHistPath.c_str();
    cfg.conv_state_path = kStatePath.c_str();  // device: conv state on the card too
    lfm::SpecStats st;
    EXPECT_EQ(Speculative(kQ4, cfg, 48, 3, true, &st), want);
    RecordProperty("tokens_per_pass", std::to_string(static_cast<double>(st.tokens) / st.passes));
    EXPECT_GT(st.accepted, 0) << "no draft was ever accepted";
    EXPECT_LT(st.accepted, st.drafted) << "test should also exercise rejections";
}

TEST(DraftHead, RejectsBadFile) {
    if (!Exists(kDevice)) GTEST_SKIP() << kDevice << " missing";
    lfm::FileSource src(kDevice.c_str());
    lfm::Engine eng;
    ASSERT_EQ(eng.open(src, {.max_batch = 3}), lfm::Status::Ok);
    lfm::DraftHead head;
    const char junk[64] = {};
    EXPECT_FALSE(head.set(junk, sizeof(junk), eng));
    EXPECT_FALSE(head.ok());
}

TEST(Speculative, DraftHeadSameTokensAsGreedy) {
    for (const auto& p : {kDevice, kFlashHead, kDraftHead}) {
        if (!Exists(p)) GTEST_SKIP() << p << " missing";
    }
    const HeadRun r = RunWithDraftHead(64, 2);
    EXPECT_EQ(r.spec, r.plain);
    const double tpp = static_cast<double>(r.st.tokens) / r.st.passes;
    RecordProperty("tokens_per_pass", std::to_string(tpp));
    std::printf("draft head: %d passes, %d tokens (%.2f per pass), drafts %d/%d accepted\n", r.st.passes,
                r.st.tokens, tpp, r.st.accepted, r.st.drafted);
    EXPECT_GT(r.st.accepted, 0);
}

// Tree drafts (the device's shape): same tokens, more per pass than the chain.
TEST(Speculative, DraftTreeSameTokensAsGreedy) {
    for (const auto& p : {kDevice, kFlashHead, kDraftHead}) {
        if (!Exists(p)) GTEST_SKIP() << p << " missing";
    }
    const HeadRun chain = RunWithDraftHead(64, 2);
    const HeadRun tree = RunWithDraftHead(64, 6, nullptr, &lfm::kTree3x1);
    EXPECT_EQ(tree.spec, tree.plain);
    const double tc = static_cast<double>(chain.st.tokens) / chain.st.passes;
    const double tt = static_cast<double>(tree.st.tokens) / tree.st.passes;
    std::printf("chain: %.2f tokens/pass, tree: %.2f tokens/pass (%d passes, drafts %d/%d)\n", tc, tt,
                tree.st.passes, tree.st.accepted, tree.st.drafted);
    EXPECT_GT(tt, tc);
}

// The LM head scan split over two "cores" finds the same tokens.
TEST(Speculative, DraftHeadWithWorkerSameTokens) {
    for (const auto& p : {kDevice, kFlashHead, kDraftHead}) {
        if (!Exists(p)) GTEST_SKIP() << p << " missing";
    }
    ThreadWorker worker;
    const HeadRun r = RunWithDraftHead(48, 2, &worker);
    EXPECT_EQ(r.spec, r.plain);
    EXPECT_GT(r.st.accepted, 0);
}

// Exact float path, conv history in RAM.
TEST(Speculative, Bf16SameTokensAsGreedy) {
    REQUIRE(kBf16);
    lfm::EngineConfig cfg{.max_ctx = 64, .max_batch = 3};
    const auto want = Plain(kBf16, cfg, 24);
    lfm::SpecStats st;
    EXPECT_EQ(Speculative(kBf16, cfg, 24, 2, true, &st), want);
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

// Tree verification: every node's prediction equals plain greedy decoding of
// the context plus that node's own path; accepting a side branch leaves the
// engine exactly where decoding that path would (conv state, K/V moved).
TEST(Speculative, TreeMatchesPathByPath) {
    REQUIRE(kQ4);
    lfm::EngineConfig cfg{.max_ctx = 128, .kv_bf16 = true, .kv_path = kKvPath2.c_str(), .max_batch = 7,
                          .int8_activations = true, .speculative = true, .conv_history_path = kHistPath.c_str(),
                          .conv_state_path = kStatePath.c_str()};
    const auto prompt = Prompt();
    // Root 0, three children (1, 2, 3), a grandchild under 1 and under 2, and one under 4.
    const int8_t parents[7] = {-1, 0, 0, 0, 1, 2, 4};
    const std::vector<int32_t> tree = {1098, 5706, 803, 779, 4481, 856, 523};
    auto path_of = [&](int b) {
        std::vector<int32_t> p;
        for (int a = b; a >= 0; a = parents[a]) p.insert(p.begin(), tree[static_cast<std::size_t>(a)]);
        return p;
    };
    auto greedy_after = [&](const std::vector<int32_t>& extra) {
        lfm::FileSource src(kQ4.c_str());
        lfm::Engine eng;
        lfm::EngineConfig c = cfg;
        static const std::string kv3 = kRoot + "/out/test-kv3.bin";
        c.kv_path = kv3.c_str();
        c.conv_history_path = nullptr;
        c.conv_state_path = nullptr;
        EXPECT_EQ(eng.open(src, c), lfm::Status::Ok);
        eng.prefill(prompt, false);
        return eng.prefill(extra, true);
    };

    lfm::FileSource src(kQ4.c_str());
    lfm::Engine eng;
    ASSERT_EQ(eng.open(src, cfg), lfm::Status::Ok);
    eng.prefill(prompt, false);
    int32_t preds[7];
    ASSERT_GE(eng.verify_tree(tree, parents, preds), 0);
    for (int b = 0; b < 7; ++b) EXPECT_EQ(preds[b], greedy_after(path_of(b))) << "node " << b;

    // Keep the side branch 0 -> 2 -> 5, then decode on: same as plain greedy.
    const int8_t path[3] = {0, 2, 5};
    ASSERT_TRUE(eng.accept_path(path, 3));
    EXPECT_EQ(eng.pos(), static_cast<int>(prompt.size()) + 3);
    std::vector<int32_t> extra = path_of(5);
    int32_t next = preds[5];
    for (int i = 0; i < 6; ++i) {
        extra.push_back(next);
        const int32_t mine = eng.forward(next, true);
        EXPECT_EQ(mine, greedy_after(extra)) << "step " << i;
        next = mine;
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

// Conv state in a file gives exactly the same tokens as in RAM, with batched
// prompt and speculative rollbacks.
TEST(Speculative, ConvStateFileSameTokens) {
    REQUIRE(kQ4);
    lfm::EngineConfig cfg{.max_ctx = 64, .max_batch = 6, .int8_activations = true};
    const auto want = Plain(kQ4, cfg, 24);
    cfg.conv_state_path = kStatePath.c_str();
    EXPECT_EQ(Plain(kQ4, cfg, 24), want);
    lfm::SpecStats st;
    EXPECT_EQ(Speculative(kQ4, cfg, 24, 3, false, &st), want);
}

// Prompt cache: generate, restore the post-prompt state, generate again:
// same tokens, without recomputing the prompt. Conv state in RAM and in a file.
TEST(PromptCache, RestoreGivesSameContinuation) {
    REQUIRE(kQ4);
    for (bool file : {false, true}) {
        lfm::EngineConfig cfg{.max_ctx = 64, .max_batch = 4, .int8_activations = true};
        const std::string prefix = kRoot + "/out/test-prefix.bin";
        if (file) {
            cfg.conv_state_path = kStatePath.c_str();
            cfg.prefix_path = prefix.c_str();
        }
        lfm::FileSource src(kQ4.c_str());
        lfm::Engine eng;
        ASSERT_EQ(eng.open(src, cfg), lfm::Status::Ok);
        const auto prompt = Prompt();
        const int32_t first = eng.prefill(prompt, true);
        ASSERT_TRUE(eng.snapshot_prefix());
        std::vector<int32_t> a{first}, b{first};
        for (int i = 0; i < 10; ++i) a.push_back(eng.forward(a.back(), true));
        ASSERT_TRUE(eng.restore_prefix());
        ASSERT_EQ(eng.pos(), static_cast<int>(prompt.size()));
        for (int i = 0; i < 10; ++i) b.push_back(eng.forward(b.back(), true));
        EXPECT_EQ(a, b) << (file ? "file" : "RAM");
    }
}
