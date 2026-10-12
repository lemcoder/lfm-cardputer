#include "sd_card.hpp"

#include "driver/sdmmc_host.h"
#include "esp_log.h"
#include "esp_vfs_fat.h"

namespace {
constexpr const char* TAG = "sd";
}

SdCard::SdCard(int freq_khz) {
    const esp_vfs_fat_sdmmc_mount_config_t mount = {
        .format_if_mount_failed = false,
        .max_files = 3,
        .allocation_unit_size = 512,
        .disk_status_check_enable = false,
        .use_one_fat = false,
    };

    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    host.flags = SDMMC_HOST_FLAG_1BIT | SDMMC_HOST_FLAG_DEINIT_ARG;
    host.max_freq_khz = freq_khz;

    sdmmc_slot_config_t slot = SDMMC_SLOT_CONFIG_DEFAULT();
    slot.width = 1;
    slot.clk = GPIO_NUM_40;
    slot.cmd = GPIO_NUM_14;  // SPI MOSI wire
    slot.d0 = GPIO_NUM_39;   // SPI MISO wire
    slot.d1 = GPIO_NUM_NC;
    slot.d2 = GPIO_NUM_NC;
    slot.d3 = GPIO_NUM_12;   // SPI CS wire; held high in 1-bit mode
    slot.d4 = GPIO_NUM_NC;
    slot.d5 = GPIO_NUM_NC;
    slot.d6 = GPIO_NUM_NC;
    slot.d7 = GPIO_NUM_NC;
    slot.flags = SDMMC_SLOT_FLAG_INTERNAL_PULLUP;

    err_ = esp_vfs_fat_sdmmc_mount(kMountPoint, &host, &slot, &mount, &card_);
    if (err_ != ESP_OK) {
        ESP_LOGE(TAG, "mount failed: %s", esp_err_to_name(err_));
        card_ = nullptr;
    } else {
        ESP_LOGI(TAG, "mounted %s, %d kHz", card_->cid.name, card_->real_freq_khz);
    }
}

SdCard::~SdCard() {
    if (card_ != nullptr) esp_vfs_fat_sdcard_unmount(kMountPoint, card_);
}
