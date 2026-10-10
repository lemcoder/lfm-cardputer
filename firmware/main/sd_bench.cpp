#include "sd_bench.hpp"

#include <cstddef>
#include <cstdio>
#include <memory>

#include "driver/sdspi_host.h"
#include "driver/spi_common.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_vfs_fat.h"
#include "sd_protocol_types.h"
#include "sdmmc_cmd.h"
#include "soc/gpio_num.h"

namespace {
    constexpr gpio_num_t kPinMiso = GPIO_NUM_39;
    constexpr gpio_num_t kPinMosi = GPIO_NUM_14;
    constexpr gpio_num_t kPinSclk = GPIO_NUM_40;
    constexpr gpio_num_t kPinCs = GPIO_NUM_12;

    constexpr const char* kMountPoint = "/sdcard";
    constexpr const char* TAG = "sdbench";

    class SdCard {
        public:
            explicit SdCard(int freq_khz) {
                sdmmc_host_t host = SDSPI_HOST_DEFAULT();
                host.max_freq_khz = freq_khz;
                host_ = static_cast<spi_host_device_t>(host.slot);


                spi_bus_config_t bus = {
                    .mosi_io_num = kPinMosi,
                    .miso_io_num = kPinMiso,
                    .sclk_io_num = kPinSclk,
                    .quadwp_io_num = -1,
                    .quadhd_io_num = -1,
                    .data4_io_num = -1,
                    .data5_io_num = -1,
                    .data6_io_num = -1,
                    .data7_io_num = -1,
                    .max_transfer_sz = 64 * 1024,
                };


                sdspi_device_config_t deviceConfig = SDSPI_DEVICE_CONFIG_DEFAULT();
                deviceConfig.gpio_cs = kPinCs;
                deviceConfig.host_id = host_;

                ESP_ERROR_CHECK(spi_bus_initialize(host_, &bus, SPI_DMA_CH_AUTO));

                esp_vfs_fat_sdmmc_mount_config_t mountConfig = {
                    .format_if_mount_failed = false,
                    .max_files = 2,
                    .allocation_unit_size = 512,
                    .disk_status_check_enable = false,
                    .use_one_fat = false
                };

                ESP_ERROR_CHECK(esp_vfs_fat_sdspi_mount(kMountPoint, &host, &deviceConfig, &mountConfig, &card_));
            }

            ~SdCard() {

            }

            SdCard(const SdCard&) = delete;
            SdCard& operator=(const SdCard&) = delete;

            sdmmc_card_t* card() const { return card_; }

        private:

            sdmmc_card_t* card_ = nullptr;
            spi_host_device_t host_;
    };
}

namespace sdbench {

    std::array<BenchResult, kNumResults> run() {
        return {};
    }

}
