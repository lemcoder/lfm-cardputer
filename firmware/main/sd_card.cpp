#include "sd_card.hpp"

#include "driver/sdmmc_host.h"
#include "esp_log.h"
#include "esp_rom_sys.h"
#include "esp_vfs_fat.h"
#include "hal/sdmmc_ll.h"
#include "soc/sdmmc_struct.h"

namespace {

constexpr const char* TAG = "sd";

SdCard::Config g_cfg;  // read by set_card_clk() during mount

// The driver programs 40 MHz (div 4) for any high-speed request; for an
// overclock, switch the clock generator to the requested divider right after.
// The generator register is outside the CIU, so no clock-update command is
// needed; the card clock stays enabled throughout.
esp_err_t set_card_clk(int slot, uint32_t freq_khz) {
    esp_err_t err = sdmmc_host_set_card_clk(slot, freq_khz);
    if (err == ESP_OK && freq_khz >= SDMMC_FREQ_HIGHSPEED && g_cfg.host_div != 4) {
        const int div = g_cfg.host_div;
        SDMMC.clock.div_factor_n = div - 1;
        SDMMC.clock.div_factor_l = div - 1;
        SDMMC.clock.div_factor_h = g_cfg.duty_high >= 0 ? g_cfg.duty_high : div / 2 - 1;
        esp_rom_delay_us(10);
    }
    return err;
}

}  // namespace

SdCard::SdCard(const Config& cfg) {
    g_cfg = cfg;
    const esp_vfs_fat_sdmmc_mount_config_t mount = {
        .format_if_mount_failed = false,
        .max_files = 3,
        .allocation_unit_size = 512,
        .disk_status_check_enable = false,
        .use_one_fat = false,
    };

    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    host.flags = SDMMC_HOST_FLAG_1BIT | SDMMC_HOST_FLAG_DEINIT_ARG;
    host.max_freq_khz = 160'000 / cfg.host_div;
    host.input_delay_phase = static_cast<sdmmc_delay_phase_t>(cfg.delay_phase);
    host.set_card_clk = &set_card_clk;

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
        ESP_LOGI(TAG, "mounted %s, %d kHz, phase %d", card_->cid.name, freq_khz(), cfg.delay_phase);
    }
}

SdCard::~SdCard() {
    if (card_ != nullptr) esp_vfs_fat_sdcard_unmount(kMountPoint, card_);
}

int SdCard::freq_khz() const {
    int khz = 0;
    sdmmc_host_get_real_freq(SDMMC_HOST_SLOT_1, &khz);
    return khz;
}
