#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace sdbench {

    // Reads >= 16 KB all hit the same ceiling at SPI 20 MHz, so test small vs large.
    inline constexpr std::array<size_t, 2> kBufferSizes = {
        4 * 1024,
        32 * 1024,
    };

    // How the card is driven. Same 4 wires either way:
    //   SPI:        SCLK 40, MOSI 14, MISO 39, CS 12
    //   SD 1-bit:   CLK 40,  CMD 14,  DAT0 39, DAT3 12 (held high)
    enum class Bus : uint8_t { Spi, Sdmmc1Bit };

    struct BusConfig {
        Bus bus;
        int freq_khz;
        int delay_phase;  // SDMMC input sampling delay 0..3 (only used >= 40 MHz)
    };

    // SD-mode configs must come first: once a card has seen SPI mode it stays
    // in SPI mode until power-cycled.
    inline constexpr std::array<BusConfig, 8> kConfigs = {{
        {Bus::Sdmmc1Bit, 20'000, 0},
        {Bus::Sdmmc1Bit, 40'000, 0},
        {Bus::Sdmmc1Bit, 40'000, 1},
        {Bus::Sdmmc1Bit, 40'000, 2},
        {Bus::Sdmmc1Bit, 40'000, 3},
        {Bus::Spi, 20'000, 0},
        {Bus::Spi, 26'667, 0},  // SPI clock is 80 MHz / n: 40, 26.67, 20, ...
        {Bus::Spi, 40'000, 0},
    }};

    // The size of the test matrix
    inline constexpr std::size_t kNumResults = kBufferSizes.size() * kConfigs.size();

    // Bytes read per measurement. Created on the card as kTestPath if missing.
    inline constexpr std::size_t kTestFileBytes = 4 * 1024 * 1024;
    inline constexpr const char* kTestPath = "/sdcard/test.bin";

    struct BenchResult {
        std::size_t buffer_size;
        BusConfig config;
        int actual_khz;               // what the driver really set; 0 = mount failed
        float megabytes_per_sec;      // read() through FAT
        float raw_megabytes_per_sec;  // sdmmc_read_sectors(), no filesystem
    };

    std::array<BenchResult, kNumResults> run();

}
