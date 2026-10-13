// lfm_tok out/tok.bin < samples: one hex-encoded UTF-8 string per line in,
// its token ids (space separated) per line out. Used by tools/check_tokenizer.py.

#include <cstdio>
#include <string>
#include <vector>

#include "lfm/source.hpp"
#include "lfm/tokenizer.hpp"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s tok.bin < hex lines\n", argv[0]);
        return 2;
    }
    lfm::FileSource file(argv[1]);
    lfm::Tokenizer tok;
    if (!file.ok() || !tok.open(file)) {
        std::fprintf(stderr, "can't open %s\n", argv[1]);
        return 1;
    }
    std::vector<int32_t> ids(1 << 16);
    char line[1 << 17];
    while (std::fgets(line, sizeof(line), stdin) != nullptr) {
        std::string text;
        for (char* p = line; p[0] != '\0' && p[0] != '\n' && p[1] != '\0'; p += 2) {
            unsigned v;
            std::sscanf(p, "%2x", &v);
            text.push_back(static_cast<char>(v));
        }
        const int n = tok.encode(text, ids.data(), static_cast<int>(ids.size()));
        for (int i = 0; i < n; ++i) std::printf(i ? " %d" : "%d", ids[i]);
        std::printf(n < 0 ? "ERROR\n" : "\n");
    }
    return 0;
}
