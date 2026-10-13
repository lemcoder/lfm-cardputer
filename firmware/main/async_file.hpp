#pragma once
// lfm::Source whose start_read() runs on a reader task pinned to core 1, so
// the engine (core 0) computes on one buffer while the SD card fills the next.
//
// Streaming (enable_streaming): after a stream_hint(), the hinted blocks are
// read by one multi-block SD command instead of one per block (each command
// costs ~0.17 ms): the DMA fills each buffer as start_read() hands it over
// (sdmmc_host_stream_*, firmware/components/esp_driver_sdmmc). Needs the file
// in one piece on the card, so its sectors are known.
//
// Compression (enable_compression): the .lfm's layer matrices are read from
// a .lfz instead (tools/zcodec.py pack, lfm/zfile.hpp), ~8% fewer bytes.
// Every block of a compressed matrix is streamed into the end of the
// engine's buffer (EngineConfig::io_extra = kIoExtra) and decoded in place
// by the reader task on core 1, following the DMA as the data lands.

#include <cstddef>
#include <cstdint>

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "lfm/source.hpp"
#include "lfm/zcodec.hpp"
#include "lfm/zfile.hpp"
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

    // Read the .lfm's compressed matrices from this .lfz (needs streaming and
    // a contiguous .lfz). The engine's buffers must have kIoExtra spare bytes.
    bool enable_compression(const char* lfz_path, const char* lfz_fatfs_path);
    static constexpr std::size_t kIoExtra = 1088;  // lead + tail sector padding (<= 1016) + decode slack (64)
    void set_io_bytes(std::size_t io_bytes) { io_bytes_ = io_bytes; }
    uint64_t compressed_bytes() const { return z_bytes_; }

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

    // Compression.
    struct ZJob {
        uint8_t* feed;       // where the DMA writes (lead bytes, data, padding)
        uint32_t feed_len;
        uint8_t* in;         // the chunk's data inside it
        uint32_t in_len;
        bool raw;
        uint8_t* out;        // the engine's buffer
        uint32_t out_len;
        uint32_t phase;
        int first_desc, n_desc;
    };
    bool z_start(const lfm::ZTensor& t, uint64_t offset, void* dst, std::size_t n);
    bool z_decode_job();  // reader task
    // Allocated only when compression is on (the RAM is the prompt batch's).
    static constexpr int kMaxZChunks = 64;
    struct ZState {
        lfm::ZIndex index;
        lfm::ZChunk chunks[kMaxZChunks];  // the open stream's chunks zk0_ .. zk0_+zn_-1
        lfm::ZDecoder dec;
    };
    ZState* z_ = nullptr;
    lfm::FileSource* zfile_ = nullptr;
    uint64_t zlba0_ = 0;
    std::size_t io_bytes_ = 32768;
    const lfm::ZTensor* zt_ = nullptr;  // matrix of the open stream
    uint32_t zk0_ = 0, zn_ = 0, zk_next_ = 0;
    uint32_t zstream_start_ = 0, zstream_total_ = 0, zfed_ = 0;
    bool zactive_ = false;
    bool zjob_pending_ = false;
    int job_kind_ = 0;  // reader task: 0 = file read, 1 = decode zjob_
    ZJob zjob_{};
    uint64_t z_bytes_ = 0;
};
