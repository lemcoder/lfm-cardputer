#pragma once
// lfm::Source whose start_read() runs on a reader task pinned to core 1, so
// the engine (core 0) computes on one buffer while the SD card fills the next.
//
// Streaming (enable_streaming): after a stream_hint(), the hinted blocks are
// read by one multi-block SD command instead of one per block (each command
// costs ~0.17 ms): the DMA fills each buffer as start_read() hands it over
// (sdmmc_host_stream_*, firmware/components/esp_driver_sdmmc). Needs the file
// in one piece on the card, so its sectors are known.

#include <cstddef>
#include <cstdint>

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "lfm/source.hpp"
#include "sdmmc_cmd.h"

class AsyncFile final : public lfm::Source {
public:
    explicit AsyncFile(const char* path);
    ~AsyncFile() override;
    AsyncFile(const AsyncFile&) = delete;
    AsyncFile& operator=(const AsyncFile&) = delete;

    bool ok() const { return file_.ok() && task_ != nullptr; }

    bool read_at(uint64_t offset, void* dst, std::size_t n) override;
    void start_read(uint64_t offset, void* dst, std::size_t n) override;
    bool wait() override;

    // Stream hinted ranges straight from the card's sectors. False if the
    // file isn't contiguous (then everything goes through FATFS as before).
    bool enable_streaming(const sdmmc_card_t* card, const char* fatfs_path);
    void stream_hint(uint64_t offset, std::size_t bytes) override;

    uint64_t bytes_read() const { return file_.bytes_read() + stream_bytes_; }
    // Card reading (streams: from start to end, including stalls).
    double sd_busy_seconds() const { return file_.io_seconds() + stream_us_ / 1e6; }
    long streams() const { return streams_; }
    double stall_seconds() const { return stall_us_ / 1e6; }      // engine waiting for data

private:
    static void reader_task(void* arg);

    lfm::FileSource file_;
    TaskHandle_t task_ = nullptr;
    SemaphoreHandle_t done_ = nullptr;
    uint64_t offset_ = 0;
    void* dst_ = nullptr;
    std::size_t n_ = 0;
    volatile bool result_ = true;
    bool quit_ = false;
    int64_t stall_us_ = 0;

    // Streaming.
    void stream_drain();  // finish an open stream early (unexpected access pattern)
    sdmmc_card_t* card_ = nullptr;
    uint64_t lba0_ = 0;            // first sector of the file
    uint64_t hint_off_ = 0;
    std::size_t hint_bytes_ = 0;   // 0: no hint
    bool active_ = false;          // a stream command is open
    uint64_t next_off_ = 0;        // file offset of the next block in the stream
    std::size_t remaining_ = 0;    // stream bytes not handed to the DMA yet
    void* stream_dst_ = nullptr;   // the stream block start_read() handed over (wait() pending)
    bool stream_ok_ = true;
    int64_t stream_t0_ = 0;
    int64_t stream_us_ = 0;
    uint64_t stream_bytes_ = 0;
    long streams_ = 0;
};
