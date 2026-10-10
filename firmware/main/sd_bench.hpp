#pragma once

#include <array>
#include <cstddef>

namespace sdbench {

    inline constexpr std::array<size_t, 4> kBufferSizes = {
        4 * 1024,
        16 * 1024,
        32 * 1024,
        64 * 1024,
    };

    inline constexpr std::array<int, 2> kFrequenciesKhz = {
        20'000,
        40'000
    };

    // The size of the test matrix
    inline constexpr std::size_t kNumResults = kBufferSizes.size() * kFrequenciesKhz.size();

    struct BenchResult {
        std::size_t buffer_size;
        int frequency_khz;
        float megabytes_per_sec;
    };

    std::array<BenchResult, kNumResults> run();

}
