// Quantization quality: run a token sequence through a reference model and
// candidates side by side, teacher-forced, and compare next-token predictions.
//
//   ./host/build/lfm_eval out/eval.txt out/lfm-bf16.lfm out/lfm-q8.lfm out/lfm-q4.lfm ...
//
// Per candidate: perplexity of the text, top-1 agreement with the reference,
// and mean KL(reference || candidate) over the next-token distributions.
// out/eval.txt is whitespace-separated token ids (tools/eval_text.py).

#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "lfm.hpp"
#include "lfm/engine.hpp"
#include "lfm/math/ops.hpp"

namespace {

struct Model {
    std::string path;
    std::unique_ptr<lfm::FileSource> src;
    std::unique_ptr<lfm::Engine> eng;
    std::vector<float> logits;
    double nll = 0, kl = 0;
    int agree = 0;
};

// log-softmax in place.
void log_softmax(std::vector<float>& x) {
    float mx = x[0];
    for (float v : x) mx = std::fmax(mx, v);
    double sum = 0;
    for (float v : x) sum += std::exp(static_cast<double>(v - mx));
    const float lse = mx + static_cast<float>(std::log(sum));
    for (float& v : x) v -= lse;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s tokens.txt reference.lfm candidate.lfm...\n", argv[0]);
        return 2;
    }
    std::vector<int32_t> tokens;
    {
        std::ifstream f(argv[1]);
        int32_t id;
        while (f >> id) tokens.push_back(id);
    }
    const int n = static_cast<int>(tokens.size());
    if (n < 2) {
        std::fprintf(stderr, "need at least 2 tokens in %s\n", argv[1]);
        return 2;
    }

    std::vector<Model> models;
    for (int i = 2; i < argc; ++i) {
        Model m;
        m.path = argv[i];
        m.src = std::make_unique<lfm::FileSource>(argv[i]);
        m.eng = std::make_unique<lfm::Engine>();
        if (lfm::Status s = m.eng->open(*m.src, {.max_ctx = n}); s != lfm::Status::Ok) {
            std::fprintf(stderr, "%s: %s\n", argv[i], lfm::to_string(s));
            return 1;
        }
        m.logits.resize(lfm::kVocab);
        models.push_back(std::move(m));
    }

    for (int t = 0; t + 1 < n; ++t) {
        for (Model& m : models) {
            m.eng->forward(tokens[t], true, m.logits.data());
            log_softmax(m.logits);
            m.nll -= m.logits[tokens[t + 1]];
        }
        const std::vector<float>& ref = models[0].logits;
        const std::size_t ref_top = lfm::math::argmax(ref);
        for (Model& m : models) {
            m.agree += lfm::math::argmax(m.logits) == ref_top;
            double kl = 0;
            for (std::size_t v = 0; v < ref.size(); ++v) {
                const double p = std::exp(static_cast<double>(ref[v]));
                if (p > 1e-12) kl += p * (ref[v] - m.logits[v]);
            }
            m.kl += kl;
        }
        if ((t + 1) % 50 == 0) std::fprintf(stderr, "%d/%d\n", t + 1, n - 1);
    }

    const int steps = n - 1;
    std::printf("%d tokens\n%-40s %10s %9s %10s %9s\n", n, "model", "ppl", "top1=ref", "KL(ref||m)", "MB");
    for (const Model& m : models) {
        std::ifstream f(m.path, std::ios::binary | std::ios::ate);
        std::printf("%-40s %10.3f %8.1f%% %10.5f %9.1f\n", m.path.c_str(), std::exp(m.nll / steps),
                    100.0 * m.agree / steps, m.kl / steps, static_cast<double>(f.tellg()) / 1e6);
    }
    return 0;
}
