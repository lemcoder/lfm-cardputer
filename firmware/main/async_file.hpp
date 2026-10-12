#pragma once
// lfm::Source whose start_read() runs on a reader task pinned to core 1, so
// the engine (core 0) computes on one buffer while the SD card fills the next.

#include <cstddef>
#include <cstdint>

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "lfm/source.hpp"

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

    uint64_t bytes_read() const { return file_.bytes_read(); }
    double sd_busy_seconds() const { return file_.io_seconds(); }  // card actually reading
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
};
