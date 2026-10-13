#include "sd_speed.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <initializer_list>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_rom_crc.h"
#include "esp_timer.h"
#include "driver/sdmmc_host.h"
#include "esp_rom_sys.h"
#include "sd_card.hpp"

namespace {

constexpr const char* TAG = "sdspeed";
constexpr std::size_t kChunk = 32 * 1024;
constexpr std::size_t kBytes = 16 * 1024 * 1024;

struct Case {
    const char* name;
    SdCard::Config cfg;
};

constexpr Case kCases[] = {
    {"80M  p2      ", {2, 2, -1}},  // current setting: reference CRC
    {"160M p0      ", {1, 0, 0}},   // divider 1: straight from the 160 MHz PLL, if the generator allows it
    {"160M p1      ", {1, 1, 0}},
    {"160M p2      ", {1, 2, 0}},
    {"160M p3      ", {1, 3, 0}},
    {"80M  p2 again", {2, 2, -1}},  // card still fine afterwards?
};

}  // namespace

void sd_speed_test(PrintFn print) {
    auto* buf = static_cast<uint8_t*>(heap_caps_malloc(kChunk, MALLOC_CAP_DMA | MALLOC_CAP_8BIT));
    char line[96];
    uint32_t ref_crc = 0;
    bool have_ref = false;

    for (const Case& c : kCases) {
        SdCard card(c.cfg);
        if (!card.ok()) {
            std::snprintf(line, sizeof(line), "%s mount fail %s", c.name, esp_err_to_name(card.err()));
            print(line);
            ESP_LOGW(TAG, "%s", line);
            continue;
        }
        const int fd = open("/sdcard/lfm.lfm", O_RDONLY);
        if (fd < 0) {
            print("no /lfm.lfm");
            break;
        }
        uint32_t crc = 0;
        std::size_t done = 0;
        const int64_t t0 = esp_timer_get_time();
        while (done < kBytes) {
            const ssize_t got = read(fd, buf, kChunk);
            if (got <= 0) break;
            crc = esp_rom_crc32_le(crc, buf, static_cast<uint32_t>(got));
            done += static_cast<std::size_t>(got);
        }
        const double s = (esp_timer_get_time() - t0) / 1e6;
        close(fd);

        const char* verdict;
        if (done < kBytes) {
            verdict = "READ ERR";
        } else if (!have_ref) {
            ref_crc = crc;
            have_ref = true;
            verdict = "ref";
        } else {
            verdict = crc == ref_crc ? "crc ok" : "CRC BAD";
        }
        std::snprintf(line, sizeof(line), "%s %5d kHz %5.2f MB/s %s", c.name, card.freq_khz(), done / 1e6 / s, verdict);
        print(line);
        ESP_LOGI(TAG, "%s", line);
    }
    heap_caps_free(buf);
}

// Multi-block read straight through the host driver: no CMD13 status
// round-trip afterwards (sdmmc_read_sectors sends one per call).
static esp_err_t read_no_status(sdmmc_card_t* card, void* dst, std::size_t lba, std::size_t count) {
    sdmmc_command_t cmd = {};
    cmd.opcode = count == 1 ? 17 : 18;  // READ_SINGLE_BLOCK / READ_MULTIPLE_BLOCK
    cmd.flags = SCF_CMD_ADTC | SCF_CMD_READ | SCF_RSP_R1;
    cmd.arg = (card->ocr & (1u << 30)) ? lba : lba * 512;  // SDHC/SDXC: block addressing
    cmd.blklen = 512;
    cmd.data = dst;
    cmd.datalen = count * 512;
    cmd.buflen = count * 512;
    cmd.timeout_ms = 1000;
    esp_err_t err = card->host.do_transaction(card->host.slot, &cmd);
    return err != ESP_OK ? err : cmd.error;
}

// One CMD18 for the whole 8 MB, through two 32 KB buffers fed back as they're
// consumed; `work_us` of busy CPU per block stands in for the engine's compute.
static double stream_test(sdmmc_card_t* card, uint8_t* bufs[2], std::size_t lba, std::size_t total, int work_us,
                          uint32_t* crc, esp_err_t* err) {
    const uint32_t arg = (card->ocr & (1u << 30)) ? lba : lba * 512;
    const int64_t t0 = esp_timer_get_time();
    *crc = 0;
    *err = sdmmc_host_stream_begin(card->host.slot, arg, total);
    std::size_t fed = 0;
    for (int i = 0; i < 2 && *err == ESP_OK && fed < total; ++i, fed += kChunk) *err = sdmmc_host_stream_feed(bufs[i], kChunk);
    for (std::size_t done = 0, i = 0; done < total && *err == ESP_OK; done += kChunk, i ^= 1) {
        *err = sdmmc_host_stream_wait(bufs[i], 1000);
        if (*err != ESP_OK) break;
        *crc = esp_rom_crc32_le(*crc, bufs[i], kChunk);
        if (work_us > 0) esp_rom_delay_us(work_us);
        if (fed < total) {
            *err = sdmmc_host_stream_feed(bufs[i], kChunk);
            fed += kChunk;
        }
    }
    if (*err == ESP_OK) *err = sdmmc_host_stream_end();
    return total / 1e6 / ((esp_timer_get_time() - t0) / 1e6);
}

void sd_raw_test(PrintFn print, const SdCard& sd) {
    auto* card = const_cast<sdmmc_card_t*>(sd.card());
    constexpr std::size_t kMax = 128 * 1024;
    constexpr std::size_t kTotal = 8u << 20;
    auto* buf = static_cast<uint8_t*>(heap_caps_malloc(kMax, MALLOC_CAP_DMA | MALLOC_CAP_8BIT));
    char line[96];
    if (buf == nullptr) {
        print("no 128 KB DMA buffer");
        return;
    }
    // FATFS path, as the engine reads (32 KB, sequential).
    {
        const int fd = open("/sdcard/lfm.lfm", O_RDONLY);
        const int64_t t0 = esp_timer_get_time();
        std::size_t done = 0;
        while (fd >= 0 && done < kTotal && read(fd, buf, kChunk) == static_cast<ssize_t>(kChunk)) done += kChunk;
        const double s = (esp_timer_get_time() - t0) / 1e6;
        if (fd >= 0) close(fd);
        std::snprintf(line, sizeof(line), "fatfs  32 KB: %5.2f MB/s, %4.0f us/read", done / 1e6 / s, s * 1e6 / (done / kChunk));
        print(line);
        ESP_LOGI(TAG, "%s", line);
    }
    // Raw sectors, from 64 MB into the card (inside the model file's area).
    for (std::size_t sz : {4096u, 8192u, 16384u, 32768u, 65536u, 131072u}) {
        const std::size_t sectors = sz / 512;
        std::size_t lba = (64u << 20) / 512;
        const int64_t t0 = esp_timer_get_time();
        esp_err_t err = ESP_OK;
        for (std::size_t done = 0; done < kTotal && err == ESP_OK; done += sz, lba += sectors) {
            err = sdmmc_read_sectors(card, buf, lba, sectors);
        }
        const double s = (esp_timer_get_time() - t0) / 1e6;
        std::snprintf(line, sizeof(line), "raw %4u KB: %5.2f MB/s, %4.0f us/read%s", (unsigned)(sz / 1024), kTotal / 1e6 / s,
                      s * 1e6 / (kTotal / sz), err == ESP_OK ? "" : " ERROR");
        print(line);
        ESP_LOGI(TAG, "%s", line);
    }
    for (std::size_t sz : {16384u, 32768u, 65536u}) {
        const std::size_t sectors = sz / 512;
        std::size_t lba = (64u << 20) / 512;
        uint32_t crc_a = 0, crc_b = 0;
        sdmmc_read_sectors(card, buf, lba, sectors);
        crc_a = esp_rom_crc32_le(0, buf, sz);
        const int64_t t0 = esp_timer_get_time();
        esp_err_t err = ESP_OK;
        for (std::size_t done = 0; done < kTotal && err == ESP_OK; done += sz, lba += sectors) {
            err = read_no_status(card, buf, lba, sectors);
            if (done == 0) crc_b = esp_rom_crc32_le(0, buf, sz);
        }
        const double s = (esp_timer_get_time() - t0) / 1e6;
        std::snprintf(line, sizeof(line), "nostat %3u KB: %5.2f MB/s, %4.0f us/read%s%s", (unsigned)(sz / 1024),
                      kTotal / 1e6 / s, s * 1e6 / (kTotal / sz), err == ESP_OK ? "" : " ERROR",
                      crc_a == crc_b ? "" : " CRC MISMATCH");
        print(line);
        ESP_LOGI(TAG, "%s", line);
    }
    // Streamed reads vs per-block commands over the same 8 MB.
    {
        uint8_t* bufs[2] = {buf, buf + kChunk};
        const std::size_t lba0 = (64u << 20) / 512;
        uint32_t ref = 0;
        std::size_t lba = lba0;
        for (std::size_t done = 0; done < kTotal; done += kChunk, lba += kChunk / 512) {
            read_no_status(card, buf, lba, kChunk / 512);
            ref = esp_rom_crc32_le(ref, buf, kChunk);
        }
        for (int work : {0, 3000, 5000}) {
            uint32_t crc = 0;
            esp_err_t err = ESP_OK;
            const double mbps = stream_test(card, bufs, lba0, kTotal, work, &crc, &err);
            // Same work per block with one command per 32 KB, for comparison.
            const int64_t t0 = esp_timer_get_time();
            lba = lba0;
            for (std::size_t done = 0; done < kTotal; done += kChunk, lba += kChunk / 512) {
                read_no_status(card, buf, lba, kChunk / 512);
                if (work > 0) esp_rom_delay_us(work);
            }
            const double per_cmd = kTotal / 1e6 / ((esp_timer_get_time() - t0) / 1e6);
            std::snprintf(line, sizeof(line), "stream work %4d us: %5.2f MB/s (per-cmd sync %5.2f) %s %s", work, mbps,
                          per_cmd, err == ESP_OK ? "ok" : esp_err_to_name(err), crc == ref ? "crc ok" : "CRC BAD");
            print(line);
            ESP_LOGI(TAG, "%s", line);
        }
    }
    heap_caps_free(buf);
}
