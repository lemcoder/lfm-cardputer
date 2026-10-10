#include <cstdio>

#include "lfm.hpp"

int main()
{
    std::printf("lfm host: dim=%d layers=%d vocab=%d\n", lfm::kDim, lfm::kNumLayers, lfm::kVocab);
    return 0;
}
