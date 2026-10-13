#include "async_file.hpp"

#include "driver/sdmmc_host.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "ff.h"

#include <cstring>

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
        self->result_ = self->job_kind_ == 1 ? self->z_decode_job()
                                             : self->file_.read_at(self->offset_, self->dst_, self->n_);
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

bool AsyncFile::enable_compression(const char* lfz_path, const char* lfz_fatfs_path) {
    if (card_ == nullptr) return false;  // streaming first
    FIL f;
    const FRESULT fr = f_open(&f, lfz_fatfs_path, FA_READ);
    if (fr != FR_OK) {
        ESP_LOGI(TAG, "%s: not there (%d): layer matrices from the .lfm", lfz_fatfs_path, (int)fr);
        return false;
    }
    DWORD map[8];
    map[0] = 8;
    f.cltbl = map;
    const bool contiguous = f_lseek(&f, CREATE_LINKMAP) == FR_OK && map[3] == 0;
    if (contiguous) zlba0_ = f.obj.fs->database + static_cast<uint64_t>(map[2] - 2) * f.obj.fs->csize;
    f_close(&f);
    if (!contiguous) {
        ESP_LOGW(TAG, "%s is fragmented: not using it", lfz_fatfs_path);
        return false;
    }
    zfile_ = new lfm::FileSource(lfz_path);
    z_ = new ZState();
    if (!zfile_->ok() || !z_->index.open(*zfile_) || z_->index.header().chunk != io_bytes_ ||
        z_->index.header().slack + 2 * (kSector - 4) > kIoExtra) {
        ESP_LOGW(TAG, "%s: unusable (chunk %u, engine block %u)", lfz_path, (unsigned)z_->index.header().chunk,
                 (unsigned)io_bytes_);
        delete zfile_;
        zfile_ = nullptr;
        delete z_;
        z_ = nullptr;
        return false;
    }
    ESP_LOGI(TAG, "%s: %u compressed matrices, first sector %llu", lfz_path, (unsigned)z_->index.header().n_tensors,
             (unsigned long long)zlba0_);
    return true;
}

// Open a stream over chunks [k, k + count) of t and hand the first to the DMA
// (the caller has made sure no stream is open).
bool AsyncFile::z_start(const lfm::ZTensor& t, uint64_t offset, void* dst, std::size_t n) {
    const uint32_t chunk = z_->index.header().chunk;
    const uint64_t rel = offset - t.v_off;
    if (rel % chunk != 0) {
        ESP_LOGE(TAG, "compressed matrix read at %llu: not a chunk boundary", (unsigned long long)offset);
        return false;
    }
    const uint32_t k = static_cast<uint32_t>(rel / chunk);
    uint32_t count = 1;
    if (hint_bytes_ > 0 && hint_off_ == offset) {  // the rest of the matrix, as hinted
        count = static_cast<uint32_t>((hint_bytes_ + chunk - 1) / chunk);
        hint_bytes_ = 0;
    }
    if (count > static_cast<uint32_t>(kMaxZChunks)) count = kMaxZChunks;
    if (k + count > t.n_chunks) count = t.n_chunks - k;
    if (!z_->index.chunks(*zfile_, t, k, count, z_->chunks)) return false;
    const lfm::ZChunk& last = z_->chunks[count - 1];
    zstream_start_ = z_->chunks[0].off & ~static_cast<uint32_t>(kSector - 1);
    const uint32_t end = static_cast<uint32_t>(round_up(last.off + (last.len & ~lfm::ZChunk::kRaw)));
    zstream_total_ = end - zstream_start_;
    const uint64_t lba = zlba0_ + zstream_start_ / kSector;
    const uint32_t arg = (card_->ocr & (1u << 30)) ? static_cast<uint32_t>(lba) : static_cast<uint32_t>(lba * kSector);
    stream_t0_ = esp_timer_get_time();
    if (sdmmc_host_stream_begin(card_->host.slot, arg, zstream_total_) != ESP_OK) return false;
    zt_ = &t;
    zk0_ = k;
    zn_ = count;
    zk_next_ = k;
    zfed_ = 0;
    zactive_ = true;
    ++streams_;
    return true;
}

// Reader task: decode the current block in place, behind the DMA.
bool AsyncFile::z_decode_job() {
    ZJob& j = zjob_;
    if (!j.raw) z_->dec.begin(z_->index.q(), z_->index.s(), j.in, j.out, j.out_len, j.phase);
    const uint8_t* const data_end = j.in + j.in_len;
    int64_t last_progress = esp_timer_get_time();
    std::size_t seen = 0;
    while (true) {
        const std::size_t landed = sdmmc_host_stream_landed(j.first_desc, j.n_desc);
        const uint8_t* avail = j.feed + landed;
        const bool complete = avail >= data_end;
        if (j.raw) {
            if (complete) {
                std::memmove(j.out, j.in, j.out_len);
                return true;
            }
        } else if (z_->dec.step(complete ? data_end : avail, complete)) {
            return true;
        }
        if (landed != seen) {
            seen = landed;
            last_progress = esp_timer_get_time();
        } else if (esp_timer_get_time() - last_progress > 1000000) {
            ESP_LOGE(TAG, "compressed block: no data for 1 s (%u of %u bytes)", (unsigned)landed, (unsigned)j.feed_len);
            return false;
        }
        if (complete) return false;  // complete but not decoded: corrupt data
    }
}

bool AsyncFile::read_at(uint64_t offset, void* dst, std::size_t n) {
    if (z_ != nullptr && z_->index.find(offset) != nullptr) {
        ESP_LOGE(TAG, "synchronous read of a compressed matrix at %llu", (unsigned long long)offset);
        return false;
    }
    if (active_) stream_drain();
    const int64_t t0 = esp_timer_get_time();
    const bool ok = file_.read_at(offset, dst, n);
    stall_us_ += esp_timer_get_time() - t0;
    return ok;
}

void AsyncFile::start_read(uint64_t offset, void* dst, std::size_t n) {
    if (z_ != nullptr) {
        const lfm::ZTensor* t = z_->index.find(offset);
        if (t != nullptr) {
            const uint32_t chunk = z_->index.header().chunk;
            bool ok = true;
            if (!(zactive_ && t == zt_ && offset == t->v_off + static_cast<uint64_t>(zk_next_) * chunk)) {
                ok = !zactive_ && z_start(*t, offset, dst, n);  // an open stream is always finished by wait()
            }
            if (ok && zk_next_ < zk0_ + zn_) {
                const lfm::ZChunk& c = z_->chunks[zk_next_ - zk0_];
                const uint32_t stored = c.len & ~lfm::ZChunk::kRaw;
                const uint32_t lead = zk_next_ == zk0_ ? c.off - zstream_start_ : 0;
                const bool last = zk_next_ + 1 == zk0_ + zn_;
                const uint32_t feed_len = last ? zstream_total_ - zfed_ : lead + stored;
                auto* buf = static_cast<uint8_t*>(dst);
                ZJob& j = zjob_;
                j.feed = buf + io_bytes_ + kIoExtra - feed_len;
                j.feed_len = feed_len;
                j.in = j.feed + lead;
                j.in_len = stored;
                j.raw = (c.len & lfm::ZChunk::kRaw) != 0;
                j.out = buf;
                j.out_len = static_cast<uint32_t>(n);
                j.phase = static_cast<uint32_t>((offset - t->v_off) % 18);
                ok = sdmmc_host_stream_feed(j.feed, feed_len) == ESP_OK;
                sdmmc_host_stream_last_feed(&j.first_desc, &j.n_desc);
                zfed_ += feed_len;
                ++zk_next_;
                stream_bytes_ += n;
                z_bytes_ += stored;
            } else {
                ok = false;
            }
            zjob_pending_ = true;
            job_kind_ = 1;
            if (!ok) {
                result_ = false;
                xSemaphoreGive(done_);
                return;
            }
            xTaskNotifyGive(task_);
            return;
        }
    }
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
    if (zjob_pending_) {
        xSemaphoreTake(done_, portMAX_DELAY);
        zjob_pending_ = false;
        job_kind_ = 0;
        bool ok = result_;
        if (zactive_) sdmmc_host_stream_release(zjob_.feed);
        if (zactive_ && (!ok || zk_next_ == zk0_ + zn_)) {
            // Last block of the stream (or a failure): close it.
            if (ok) {
                ok = sdmmc_host_stream_end() == ESP_OK;
            } else {
                ESP_LOGE(TAG, "compressed stream failed");
                sdmmc_host_stream_end();
            }
            zactive_ = false;
            stream_us_ += esp_timer_get_time() - stream_t0_;
        }
        stall_us_ += esp_timer_get_time() - t0;
        return ok;
    }
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
