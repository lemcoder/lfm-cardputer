#include "async_file.hpp"

#include "esp_timer.h"

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
bool AsyncFile::read_at(uint64_t offset, void* dst, std::size_t n) {
    const int64_t t0 = esp_timer_get_time();
    const bool ok = file_.read_at(offset, dst, n);
    stall_us_ += esp_timer_get_time() - t0;
    return ok;
}

void AsyncFile::start_read(uint64_t offset, void* dst, std::size_t n) {
    offset_ = offset;
    dst_ = dst;
    n_ = n;
    xTaskNotifyGive(task_);
}

bool AsyncFile::wait() {
    const int64_t t0 = esp_timer_get_time();
    xSemaphoreTake(done_, portMAX_DELAY);
    stall_us_ += esp_timer_get_time() - t0;
    return result_;
}
