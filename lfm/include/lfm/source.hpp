#pragma once
// Where the engine gets model bytes from. The engine only ever asks for
// "n bytes at offset"; it never holds more than one I/O buffer of weights.

#include <cstddef>
#include <cstdint>

namespace lfm {

class Source {
public:
    virtual ~Source() = default;
    // Fill dst with exactly n bytes starting at offset. False on error / EOF.
    virtual bool read_at(uint64_t offset, void* dst, std::size_t n) = 0;

    // Asynchronous read, so the engine can compute on one buffer while the
    // next one loads. At most one in flight; every start_read() is followed
    // by wait() before any other call. The default just reads synchronously;
    // a platform overrides both to read on another thread / core.
    virtual void start_read(uint64_t offset, void* dst, std::size_t n) { pending_ok_ = read_at(offset, dst, n); }
    virtual bool wait() { return pending_ok_; }

private:
    bool pending_ok_ = true;
};

// A file accessed with POSIX open/lseek/read/write. Works on the Mac and on
// ESP-IDF (FATFS on the SD card). Skips the lseek when access is sequential.
class FileSource final : public Source {
public:
    enum class Mode : uint8_t {
        Read,       // existing file, read-only
        ReadWrite,  // created (or emptied) on open; read_at and write_at
    };

    explicit FileSource(const char* path, Mode mode = Mode::Read);
    ~FileSource() override;
    FileSource(const FileSource&) = delete;
    FileSource& operator=(const FileSource&) = delete;

    bool ok() const { return fd_ >= 0; }
    bool read_at(uint64_t offset, void* dst, std::size_t n) override;
    // Mode::ReadWrite only. Writing past the end grows the file.
    bool write_at(uint64_t offset, const void* src, std::size_t n);

    // Totals so far, for throughput stats: bytes read, and wall time spent
    // inside read_at / write_at (waiting on the disk / SD card).
    uint64_t bytes_read() const { return bytes_read_; }
    double io_seconds() const { return io_seconds_; }

private:
    bool seek(uint64_t offset);

    int fd_ = -1;
    uint64_t pos_ = 0;
    uint64_t bytes_read_ = 0;
    double io_seconds_ = 0.0;
};

}  // namespace lfm
