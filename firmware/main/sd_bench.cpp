#include "sd_bench.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>

#include "driver/sdspi_host.h"
#include "driver/spi_common.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_random.h"
#include "esp_timer.h"
#include "esp_vfs_fat.h"
#include "sdmmc_cmd.h"
#include "soc/gpio_num.h"

namespace {
    constexpr gpio_num_t kPinMiso = GPIO_NUM_39;
    constexpr gpio_num_t kPinMosi = GPIO_NUM_14;
    constexpr gpio_num_t kPinSclk = GPIO_NUM_40;
    constexpr gpio_num_t kPinCs = GPIO_NUM_12;

    constexpr const char* kMountPoint = "/sdcard";
    constexpr const char* TAG = "sdbench";

    constexpr std::size_t kSectorSize = 512;

    // Mounts the card on construction, unmounts on destruction.
    // Mount failures are recorded in err() instead of aborting, so a card that
    // can't do 40 MHz shows up as a failed result rather than a reboot loop.
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
                    .data_io_default_level = false,
                    .max_transfer_sz = 64 * 1024,
                    .flags = 0,
                    .isr_cpu_id = ESP_INTR_CPU_AFFINITY_AUTO,
                    .intr_flags = 0,
                };

                sdspi_device_config_t deviceConfig = SDSPI_DEVICE_CONFIG_DEFAULT();
                deviceConfig.gpio_cs = kPinCs;
                deviceConfig.host_id = host_;

                err_ = spi_bus_initialize(host_, &bus, SPI_DMA_CH_AUTO);
                if (err_ != ESP_OK) {
                    ESP_LOGE(TAG, "spi_bus_initialize: %s", esp_err_to_name(err_));
                    return;
                }
                bus_initialized_ = true;

                esp_vfs_fat_sdmmc_mount_config_t mountConfig = {
                    .format_if_mount_failed = false,
                    .max_files = 2,
                    .allocation_unit_size = 512,
                    .disk_status_check_enable = false,
                    .use_one_fat = false
                };

                err_ = esp_vfs_fat_sdspi_mount(kMountPoint, &host, &deviceConfig, &mountConfig, &card_);
                if (err_ != ESP_OK) {
                    ESP_LOGE(TAG, "mount at %d kHz: %s", freq_khz, esp_err_to_name(err_));
                    card_ = nullptr;
                }
            }

            ~SdCard() {
                if (card_ != nullptr) {
                    esp_err_t err = esp_vfs_fat_sdcard_unmount(kMountPoint, card_);
                    if (err != ESP_OK) {
                        ESP_LOGE(TAG, "unmount failed: %s", esp_err_to_name(err));
                    }
                }

                if (bus_initialized_) {
                    esp_err_t err = spi_bus_free(host_);
                    if (err != ESP_OK) {
                        ESP_LOGE(TAG, "spi_bus_free failed: %s", esp_err_to_name(err));
                    }
                }
            }

            SdCard(const SdCard&) = delete;
            SdCard& operator=(const SdCard&) = delete;

            bool ok() const { return card_ != nullptr; }
            esp_err_t err() const { return err_; }
            sdmmc_card_t* card() const { return card_; }

        private:
            sdmmc_card_t* card_ = nullptr;
            spi_host_device_t host_;
            bool bus_initialized_ = false;
            esp_err_t err_ = ESP_OK;
    };

    // unique_ptr deleters: let RAII free DMA buffers and close files.
    struct HeapCapsFree {
        void operator()(uint8_t* p) const { heap_caps_free(p); }
    };
    using DmaBuffer = std::unique_ptr<uint8_t[], HeapCapsFree>;

    struct FileClose {
        void operator()(std::FILE* f) const { std::fclose(f); }
    };
    using File = std::unique_ptr<std::FILE, FileClose>;

    DmaBuffer alloc_dma(std::size_t size) {
        return DmaBuffer{static_cast<uint8_t*>(heap_caps_malloc(size, MALLOC_CAP_DMA))};
    }

    float mb_per_sec(std::size_t bytes, int64_t micros) {
        // bytes per microsecond == megabytes per second (1 MB = 1e6 B here)
        return micros > 0 ? static_cast<float>(bytes) / static_cast<float>(micros) : 0.0f;
    }

    // Writes kTestFileBytes of random data to sdbench::kTestPath unless it already exists
    // with at least that size. Takes a few seconds the first time.
    bool ensure_test_file() {
        struct stat st = {};
        if (stat(sdbench::kTestPath, &st) == 0 && static_cast<std::size_t>(st.st_size) >= sdbench::kTestFileBytes) {
            return true;
        }

        ESP_LOGI(TAG, "creating %s (%u bytes)", sdbench::kTestPath, static_cast<unsigned>(sdbench::kTestFileBytes));
        constexpr std::size_t kChunk = 32 * 1024;
        DmaBuffer buf = alloc_dma(kChunk);
        File f{std::fopen(sdbench::kTestPath, "wb")};
        if (!buf || !f) {
            ESP_LOGE(TAG, "cannot create test file");
            return false;
        }

        esp_fill_random(buf.get(), kChunk);
        int64_t t0 = esp_timer_get_time();
        for (std::size_t written = 0; written < sdbench::kTestFileBytes; written += kChunk) {
            if (std::fwrite(buf.get(), 1, kChunk, f.get()) != kChunk) {
                ESP_LOGE(TAG, "write failed at %u", static_cast<unsigned>(written));
                return false;
            }
        }
        ESP_LOGI(TAG, "write: %.2f MB/s", mb_per_sec(sdbench::kTestFileBytes, esp_timer_get_time() - t0));
        return true;
    }

    // Sequential read of the test file through the VFS + FATFS.
    // Uses POSIX read(), not fread(): newlib's fread() on an unbuffered stream
    // degrades to tiny reads, which measured ~30x slower than the card itself.
    float measure_file(std::size_t buffer_size) {
        DmaBuffer buf = alloc_dma(buffer_size);
        if (!buf) {
            ESP_LOGE(TAG, "alloc %u failed", static_cast<unsigned>(buffer_size));
            return 0.0f;
        }

        int fd = open(sdbench::kTestPath, O_RDONLY);
        if (fd < 0) {
            ESP_LOGE(TAG, "open %s failed", sdbench::kTestPath);
            return 0.0f;
        }

        std::size_t total = 0;
        int64_t t0 = esp_timer_get_time();
        while (total < sdbench::kTestFileBytes) {
            ssize_t n = read(fd, buf.get(), buffer_size);
            if (n <= 0) {
                if (n < 0) {
                    ESP_LOGE(TAG, "read error after %u bytes", static_cast<unsigned>(total));
                }
                break;
            }
            total += static_cast<std::size_t>(n);
        }
        int64_t elapsed = esp_timer_get_time() - t0;
        close(fd);

        return mb_per_sec(total, elapsed);
    }

    // Sequential read of raw sectors, bypassing the filesystem. Shows FAT overhead.
    float measure_raw(sdmmc_card_t* card, std::size_t buffer_size) {
        DmaBuffer buf = alloc_dma(buffer_size);
        if (!buf) {
            return 0.0f;
        }

        const std::size_t sectors_per_read = buffer_size / kSectorSize;
        const std::size_t total_sectors = sdbench::kTestFileBytes / kSectorSize;
        // Start past the partition table / FAT region; any data works for timing.
        constexpr std::size_t kStartSector = 16 * 1024;

        int64_t t0 = esp_timer_get_time();
        for (std::size_t s = 0; s < total_sectors; s += sectors_per_read) {
            esp_err_t err = sdmmc_read_sectors(card, buf.get(), kStartSector + s, sectors_per_read);
            if (err != ESP_OK) {
                ESP_LOGE(TAG, "raw read: %s", esp_err_to_name(err));
                return 0.0f;
            }
        }
        return mb_per_sec(sdbench::kTestFileBytes, esp_timer_get_time() - t0);
    }
}

namespace sdbench {

    std::array<BenchResult, kNumResults> run() {
        std::array<BenchResult, kNumResults> results = {};
        std::size_t i = 0;

        for (int freq : kFrequenciesKhz) {
            SdCard sd{freq};

            if (sd.ok()) {
                sdmmc_card_print_info(stdout, sd.card());
                ESP_LOGI(TAG, "requested %d kHz, actual %d kHz", freq, sd.card()->real_freq_khz);
            }
            const bool have_file = sd.ok() && ensure_test_file();

            for (std::size_t size : kBufferSizes) {
                BenchResult& r = results[i++];
                r.buffer_size = size;
                r.frequency_khz = freq;
                if (!have_file) {
                    continue;  // speeds stay 0 -> reported as failed
                }
                r.megabytes_per_sec = measure_file(size);
                r.raw_megabytes_per_sec = measure_raw(sd.card(), size);
                ESP_LOGI(TAG, "%5d kHz %3u KB: file %.2f MB/s, raw %.2f MB/s", freq,
                         static_cast<unsigned>(size / 1024), r.megabytes_per_sec, r.raw_megabytes_per_sec);
            }
        }

        return results;
    }

}
