#pragma once
// microSD in SD 1-bit mode on the Cardputer-Adv's SPI wiring (see sd_bench.hpp
// for why: 4.85 MB/s at 40 MHz vs 1.8 MB/s over SPI). Mounted at /sdcard.

#include "esp_err.h"
#include "sdmmc_cmd.h"

class SdCard {
public:
    explicit SdCard(int freq_khz = 40'000);
    ~SdCard();
    SdCard(const SdCard&) = delete;
    SdCard& operator=(const SdCard&) = delete;

    bool ok() const { return card_ != nullptr; }
    esp_err_t err() const { return err_; }
    const sdmmc_card_t* card() const { return card_; }

    static constexpr const char* kMountPoint = "/sdcard";

private:
    sdmmc_card_t* card_ = nullptr;
    esp_err_t err_ = ESP_OK;
};
