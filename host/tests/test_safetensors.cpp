// Tests for lfm::io::Safetensors against the real model file.
// Needs golden/default/manifest.json (run the "tools: golden dump" task once);
// skipped otherwise.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "lfm/io/safetensors.hpp"
#include "lfm/math/ops.hpp"

namespace {

const std::string kGoldenDir = std::string(LFM_REPO_ROOT) + "/golden/default";

nlohmann::json LoadManifest() {
    std::ifstream f(kGoldenDir + "/manifest.json");
    if (!f) return {};
    return nlohmann::json::parse(f);
}

std::vector<float> LoadF32(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    std::vector<float> v(static_cast<std::size_t>(f.tellg()) / sizeof(float));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
    return v;
}

// One file open shared by all tests in the suite: mapping is cheap, but
// parsing the header 4 times is pointless.
class SafetensorsTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        manifest_ = new nlohmann::json(LoadManifest());
        if (manifest_->contains("weights")) {
            const std::string path = (*manifest_)["weights"];
            st_ = new lfm::io::Safetensors(path.c_str());
        }
    }
    static void TearDownTestSuite() {
        delete st_;
        delete manifest_;
        st_ = nullptr;
        manifest_ = nullptr;
    }
    void SetUp() override {
        if (st_ == nullptr) GTEST_SKIP() << "no golden/default/manifest.json; run tools/golden.py";
    }

    static nlohmann::json* manifest_;
    static lfm::io::Safetensors* st_;
};

nlohmann::json* SafetensorsTest::manifest_ = nullptr;
lfm::io::Safetensors* SafetensorsTest::st_ = nullptr;

}  // namespace

TEST(Safetensors, MissingFileIsNotOk) {
    lfm::io::Safetensors st("/nonexistent/model.safetensors");
    EXPECT_FALSE(st.ok());
    EXPECT_EQ(st.find("model.embed_tokens.weight"), nullptr);
}

TEST_F(SafetensorsTest, Opens) {
    EXPECT_TRUE(st_->ok());
}

TEST_F(SafetensorsTest, EmbeddingShape) {
    const lfm::io::Tensor* t = st_->find("model.embed_tokens.weight");
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->dtype, "BF16");
    ASSERT_EQ(t->ndim, 2u);
    EXPECT_EQ(t->shape[0], 65536u);
    EXPECT_EQ(t->shape[1], 1024u);
    EXPECT_EQ(t->data.size(), t->numel() * 2) << "BF16 = 2 bytes per element";
}

TEST_F(SafetensorsTest, ConvWeightIs3D) {
    const lfm::io::Tensor* t = st_->find("model.layers.0.conv.conv.weight");
    ASSERT_NE(t, nullptr);
    ASSERT_EQ(t->ndim, 3u);
    EXPECT_EQ(t->shape[0], 1024u);
    EXPECT_EQ(t->shape[1], 1u);
    EXPECT_EQ(t->shape[2], 3u);
}

TEST_F(SafetensorsTest, UnknownNameIsNull) {
    EXPECT_EQ(st_->find("model.does.not.exist"), nullptr);
}

// The real proof: embedding rows for the prompt tokens, read from the file
// and converted with our bf16_to_f32, equal what PyTorch produced.
TEST_F(SafetensorsTest, EmbeddingRowsMatchGolden) {
    const lfm::io::Tensor* t = st_->find("model.embed_tokens.weight");
    ASSERT_NE(t, nullptr);
    const std::size_t dim = t->shape[1];

    const std::vector<int> tokens = (*manifest_)["tokens"];
    const std::vector<float> want = LoadF32(kGoldenDir + "/embed.f32");
    ASSERT_EQ(want.size(), tokens.size() * dim);

    // BF16 data as uint16_t: reading through a byte pointer is allowed;
    // memcpy keeps it legal regardless of alignment.
    std::vector<uint16_t> row_bf16(dim);
    std::vector<float> row(dim);
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        const std::size_t row_bytes = dim * sizeof(uint16_t);
        std::memcpy(row_bf16.data(), t->data.data() + tokens[i] * row_bytes, row_bytes);
        lfm::math::bf16_to_f32(row, row_bf16);

        for (std::size_t j = 0; j < dim; ++j) {
            ASSERT_EQ(row[j], want[i * dim + j]) << "token " << tokens[i] << " [" << j << "]";
        }
    }
}
