#pragma once
// microSD in SD 1-bit mode on the Cardputer-Adv's SPI wiring (see sd_bench.hpp
// for why: 4.85 MB/s at 40 MHz vs 1.8 MB/s over SPI). Mounted at /sdcard.
//
// Only DAT0 is wired, so the bus is 1 bit wide: bytes/s = clock / 8. The
// ESP32-S3 SDMMC clock is 160 MHz / divider: 40 MHz (div 4) is the fastest
// the IDF driver offers; div 3 = 53.3 MHz (2:1 duty) and div 2 = 80 MHz are
// overclocks past the SD high-speed limit (50 MHz). Data is CRC-checked by
// the card protocol, so a bad clock shows up as errors, not silent damage.

#include "esp_err.h"
#include "sdmmc_cmd.h"

class SdCard {
public:
    struct Config {
        int host_div = 4;         // 4 = 40 MHz, 3 = 53.3 MHz, 2 = 80 MHz (overclocks)
        int delay_phase = 0;      // input sampling delay 0..3
        int duty_high = -1;       // div 3 only: clock-high counter value (0 or 1); -1 = driver default
    };

    SdCard() : SdCard(Config{}) {}
    explicit SdCard(const Config& cfg);
    ~SdCard();
    SdCard(const SdCard&) = delete;
    SdCard& operator=(const SdCard&) = delete;

    bool ok() const { return card_ != nullptr; }
    esp_err_t err() const { return err_; }
    const sdmmc_card_t* card() const { return card_; }
    int freq_khz() const;  // what the hardware is actually set to

    static constexpr const char* kMountPoint = "/sdcard";

private:
    sdmmc_card_t* card_ = nullptr;
    esp_err_t err_ = ESP_OK;
};
