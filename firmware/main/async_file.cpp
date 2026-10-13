#include "async_file.hpp"

#include "driver/sdmmc_host.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "ff.h"

namespace {
constexpr const char* TAG = "async_file";
constexpr std::size_t kSector = 512;
std::size_t round_up(std::size_t n) { return (n + kSector - 1) / kSector * kSector; }
}  // namespace

AsyncFile::AsyncFile(const char* path) : file_(path) {
    done_ = xSemaphoreCreateBinary();
    if (file_.ok() && done_ != nullptr) {
        xTaskCreatePinnedToCore(reader_task, "sd_reader", 6144, this, 5, &task_, 1);
    }
}

AsyncFile::~AsyncFile() {
    if (task_ != nullptr) {
        quit_ = true;
        xTaskNotifyGive(task_);
        xSemaphoreTake(done_, portMAX_DELAY);
    }
    if (done_ != nullptr) vSemaphoreDelete(done_);
}

void AsyncFile::reader_task(void* arg) {
    auto* self = static_cast<AsyncFile*>(arg);
    while (true) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        if (self->quit_) break;
        self->result_ = self->file_.read_at(self->offset_, self->dst_, self->n_);
        xSemaphoreGive(self->done_);
    }
    xSemaphoreGive(self->done_);
    vTaskDelete(nullptr);
}

// Synchronous reads (header, norms, KV rows, token text) run on the caller's
// core; the engine never issues one while an async read is in flight.
bool AsyncFile::enable_streaming(const sdmmc_card_t* card, const char* fatfs_path) {
    FIL f;
    if (f_open(&f, fatfs_path, FA_READ) != FR_OK) return false;
    // Cluster map: [size, (count, first cluster)..., 0]. One fragment = contiguous.
    DWORD map[8];
    map[0] = 8;
    f.cltbl = map;
    const FRESULT r = f_lseek(&f, CREATE_LINKMAP);
    const bool contiguous = r == FR_OK && map[3] == 0;
    if (contiguous) lba0_ = f.obj.fs->database + static_cast<uint64_t>(map[2] - 2) * f.obj.fs->csize;
    ESP_LOGI(TAG, "%s: %s, first sector %llu", fatfs_path, r != FR_OK ? "no cluster map" : contiguous ? "contiguous" : "fragmented",
             (unsigned long long)lba0_);
    f_close(&f);
    if (contiguous) card_ = const_cast<sdmmc_card_t*>(card);
    return contiguous;
}

void AsyncFile::stream_hint(uint64_t offset, std::size_t bytes) {
    if (card_ == nullptr || offset % kSector != 0) return;
    if (active_) stream_drain();
    hint_off_ = offset;
    hint_bytes_ = bytes;
}

// The engine left a stream before its end: read the rest into the last
// buffer and discard it, so the card is free for other commands.
void AsyncFile::stream_drain() {
    ESP_LOGW(TAG, "stream left early at %llu, %u bytes unread", (unsigned long long)next_off_, (unsigned)remaining_);
    void* buf = stream_dst_;
    if (buf != nullptr) stream_ok_ = sdmmc_host_stream_wait(buf, 1000) == ESP_OK && stream_ok_;
    // Without a buffer of our own, reuse the one the engine last got back (its data is consumed).
    while (remaining_ > 0 && buf != nullptr && stream_ok_) {
        const std::size_t len = remaining_ < 32768 ? remaining_ : 32768;
        stream_ok_ = sdmmc_host_stream_feed(buf, len) == ESP_OK && sdmmc_host_stream_wait(buf, 1000) == ESP_OK;
        remaining_ -= len;
    }
    sdmmc_host_stream_end();
    stream_dst_ = nullptr;
    active_ = false;
    stream_us_ += esp_timer_get_time() - stream_t0_;
}

bool AsyncFile::read_at(uint64_t offset, void* dst, std::size_t n) {
    if (active_) stream_drain();
    const int64_t t0 = esp_timer_get_time();
    const bool ok = file_.read_at(offset, dst, n);
    stall_us_ += esp_timer_get_time() - t0;
    return ok;
}

void AsyncFile::start_read(uint64_t offset, void* dst, std::size_t n) {
    if (hint_bytes_ > 0 && !active_ && offset == hint_off_) {
        // Open the stream for the whole hinted range.
        const std::size_t total = round_up(hint_bytes_);
        const uint64_t lba = lba0_ + offset / kSector;
        const uint32_t arg = (card_->ocr & (1u << 30)) ? static_cast<uint32_t>(lba) : static_cast<uint32_t>(lba * kSector);
        hint_bytes_ = 0;
        stream_t0_ = esp_timer_get_time();
        if (sdmmc_host_stream_begin(card_->host.slot, arg, total) == ESP_OK) {
            active_ = true;
            stream_ok_ = true;
            next_off_ = offset;
            remaining_ = total;
            ++streams_;
        } else {
            ESP_LOGW(TAG, "stream begin failed: reading block by block");
        }
    }
    if (active_ && offset != next_off_) stream_drain();
    if (active_) {
        const std::size_t len = round_up(n) < remaining_ ? round_up(n) : remaining_;
        stream_ok_ = sdmmc_host_stream_feed(dst, len) == ESP_OK && stream_ok_;
        stream_dst_ = dst;
        next_off_ += n;
        remaining_ -= len;
        stream_bytes_ += n;
        return;
    }
    offset_ = offset;
    dst_ = dst;
    n_ = n;
    xTaskNotifyGive(task_);
}

bool AsyncFile::wait() {
    const int64_t t0 = esp_timer_get_time();
    if (stream_dst_ != nullptr) {
        bool ok = stream_ok_ && sdmmc_host_stream_wait(stream_dst_, 1000) == ESP_OK;
        stream_dst_ = nullptr;
        if (ok && remaining_ == 0) {
            ok = sdmmc_host_stream_end() == ESP_OK;
            active_ = false;
            stream_us_ += esp_timer_get_time() - stream_t0_;
        }
        if (!ok) {
            ESP_LOGE(TAG, "stream read failed");
            active_ = false;
        }
        stall_us_ += esp_timer_get_time() - t0;
        return ok;
    }
    xSemaphoreTake(done_, portMAX_DELAY);
    stall_us_ += esp_timer_get_time() - t0;
    return result_;
}
