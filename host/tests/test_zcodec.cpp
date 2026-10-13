// The C decoder against chunks of real Q4_0 matrices encoded by tools/zcodec.py.

#include <gtest/gtest.h>

#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "lfm/engine.hpp"
#include "lfm/source.hpp"
#include "lfm/zcodec.hpp"
#include "lfm/zfile.hpp"

namespace {

const std::string kTest = std::string(LFM_REPO_ROOT) + "/out/ztest.bin";

uint32_t U32(const std::vector<char>& f, std::size_t at) {
    uint32_t v;
    std::memcpy(&v, f.data() + at, 4);
    return v;
}

}  // namespace

TEST(ZCodec, DecodesRealChunks) {
    std::ifstream in(kTest, std::ios::binary);
    if (!in) GTEST_SKIP() << "uv run tools/zcodec.py test";
    const std::vector<char> f{std::istreambuf_iterator<char>(in), {}};
    ASSERT_EQ(std::memcmp(f.data(), "LFMZTST", 7), 0);
    const uint32_t n = U32(f, 8);
    lfm::ZTable q, s;
    std::size_t at = 16;
    std::memcpy(q.e, f.data() + at, sizeof(q.e));
    std::memcpy(s.e, f.data() + at + sizeof(q.e), sizeof(s.e));
    at += 2 * sizeof(q.e);
    std::vector<uint8_t> out;
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t raw_len = U32(f, at), phase = U32(f, at + 4), comp_len = U32(f, at + 8);
        at += 12;
        const auto* raw = reinterpret_cast<const uint8_t*>(f.data() + at);
        at += raw_len;
        std::vector<uint8_t> comp(f.begin() + static_cast<long>(at), f.begin() + static_cast<long>(at + comp_len));
        comp.resize(comp.size() + 4);  // the decoder may look 3 bytes ahead
        at += comp_len + ((4 - comp_len % 4) % 4);
        out.assign(raw_len, 0);
        ASSERT_TRUE(lfm::z_decode(q, s, comp.data(), comp_len, out.data(), raw_len, phase)) << "chunk " << i;
        ASSERT_EQ(std::memcmp(out.data(), raw, raw_len), 0) << "chunk " << i << " phase " << phase;
        // Resumable: input arriving 2 KB at a time, as from the SD card's DMA.
        out.assign(raw_len, 0);
        lfm::ZDecoder dec;
        dec.begin(q, s, comp.data(), out.data(), raw_len, phase);
        for (std::size_t landed = 2048; !dec.done(); landed += 2048) {
            const bool complete = landed >= comp_len;
            dec.step(comp.data() + std::min<std::size_t>(landed, comp_len), complete);
            ASSERT_TRUE(!complete || dec.done()) << "chunk " << i;
        }
        ASSERT_EQ(std::memcmp(out.data(), raw, raw_len), 0) << "resumable, chunk " << i;
    }
}

// The engine reading the device's model through the .lfz gives the same tokens.
TEST(ZFile, EngineSameTokensThroughCompressedMatrices) {
    const std::string lfm = std::string(LFM_REPO_ROOT) + "/out/lfm-q4-head-lr256.lfm";
    const std::string lfz = std::string(LFM_REPO_ROOT) + "/out/lfm.lfz";
    if (!std::ifstream(lfm) || !std::ifstream(lfz)) GTEST_SKIP() << "uv run tools/zcodec.py pack";
    const std::vector<int32_t> prompt = {1, 1098, 5706, 803, 4481, 856};  // "The capital of France is"
    auto run = [&](lfm::Source& src) {
        lfm::Engine eng;
        EXPECT_EQ(eng.open(src, {.max_ctx = 64, .max_batch = 4, .int8_activations = true}), lfm::Status::Ok);
        std::vector<int32_t> out;
        int32_t next = eng.prefill(prompt, true);
        for (int i = 0; i < 8 && next >= 0; ++i) {
            out.push_back(next);
            next = eng.forward(next, true);
        }
        return out;
    };
    lfm::FileSource plain(lfm.c_str());
    lfm::FileSource raw(lfm.c_str()), z(lfz.c_str());
    lfm::ZSource zs(raw, z);
    ASSERT_TRUE(zs.ok());
    // Every byte of every compressed matrix decodes to the .lfm's.
    lfm::ZIndex idx;
    ASSERT_TRUE(idx.open(z));
    std::vector<uint8_t> a(32768), b(32768);
    for (uint64_t at = 0;;) {
        const uint64_t start = idx.next_start(at);
        if (start == UINT64_MAX) break;
        const lfm::ZTensor* t = idx.find(start);
        ASSERT_NE(t, nullptr);
        for (uint64_t o = t->v_off; o < t->v_off + t->v_len; o += a.size()) {
            const std::size_t n = std::min<uint64_t>(a.size(), t->v_off + t->v_len - o);
            ASSERT_TRUE(plain.read_at(o, a.data(), n));
            ASSERT_TRUE(zs.read_at(o, b.data(), n));
            ASSERT_EQ(std::memcmp(a.data(), b.data(), n), 0) << "matrix at " << t->v_off << " offset " << o;
        }
        at = t->v_off;
    }
    EXPECT_EQ(run(zs), run(plain));
}
