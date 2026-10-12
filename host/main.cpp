// Host runner: same engine as the device, reading the .lfm file from disk.
//
//   ./host/build/lfm_host out/lfm-q8.lfm                  # prompt from out/prompt.txt
//   ./host/build/lfm_host out/lfm-q8.lfm 1 1098 5706 ...  # explicit token ids
//   options: --max-new N (default 32), --ctx N (default 128), --kv-bf16

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#include "lfm.hpp"
#include "lfm/engine.hpp"

namespace {

void print_token(lfm::Engine& eng, int32_t id) {
    char buf[256];
    const std::size_t n = eng.token_text(id, buf, sizeof(buf));
    std::fwrite(buf, 1, n, stdout);
    std::fflush(stdout);
}

std::vector<int32_t> read_prompt(const char* path) {
    std::vector<int32_t> ids;
    std::ifstream f(path);
    int32_t id;
    while (f >> id) ids.push_back(id);
    return ids;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s model.lfm [--max-new N] [--ctx N] [--kv-bf16] [token ids...]\n", argv[0]);
        return 2;
    }
    lfm::EngineConfig cfg;
    int max_new = 32;
    std::vector<int32_t> prompt;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--max-new") == 0 && i + 1 < argc) {
            max_new = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) {
            cfg.max_ctx = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--kv-bf16") == 0) {
            cfg.kv_bf16 = true;
        } else {
            prompt.push_back(static_cast<int32_t>(std::atoi(argv[i])));
        }
    }
    if (prompt.empty()) prompt = read_prompt("out/prompt.txt");
    if (prompt.empty()) {
        std::fprintf(stderr, "no prompt: pass token ids or create out/prompt.txt (tools/export.py)\n");
        return 2;
    }

    lfm::FileSource src(argv[1]);
    if (!src.ok()) {
        std::fprintf(stderr, "can't open %s\n", argv[1]);
        return 1;
    }
    lfm::Engine eng;
    if (lfm::Status s = eng.open(src, cfg); s != lfm::Status::Ok) {
        std::fprintf(stderr, "open: %s\n", lfm::to_string(s));
        return 1;
    }
    std::fprintf(stderr, "engine RAM: %zu KB, ctx %d, kv %s\n", eng.ram_bytes(cfg) / 1024, cfg.max_ctx,
                 cfg.kv_bf16 ? "bf16" : "f32");

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    int32_t next = lfm::Engine::kNoToken;
    for (std::size_t i = 0; i < prompt.size(); ++i) {
        print_token(eng, prompt[i]);
        next = eng.forward(prompt[i], i + 1 == prompt.size());
    }
    const auto t1 = clock::now();

    int generated = 0;
    while (next >= 0 && next != lfm::kEosId && generated < max_new) {
        print_token(eng, next);
        ++generated;
        next = eng.forward(next, true);
    }
    const auto t2 = clock::now();
    std::printf("\n");

    const double prefill = std::chrono::duration<double>(t1 - t0).count();
    const double gen = std::chrono::duration<double>(t2 - t1).count();
    std::fprintf(stderr, "prefill %zu tok %.2f s | generated %d tok %.2f s (%.3f s/tok) | read %.0f MB\n",
                 prompt.size(), prefill, generated, gen, generated ? gen / generated : 0.0,
                 static_cast<double>(src.bytes_read()) / 1e6);
    if (next < 0 && next != lfm::Engine::kNoToken) std::fprintf(stderr, "stopped: error %d\n", next);
    return 0;
}
