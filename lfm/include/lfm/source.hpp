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
};

// A file read with POSIX open/lseek/read. Works on the Mac and on ESP-IDF
// (FATFS on the SD card). Skips the lseek when reads are sequential.
class FileSource final : public Source {
public:
    explicit FileSource(const char* path);
    ~FileSource() override;
    FileSource(const FileSource&) = delete;
    FileSource& operator=(const FileSource&) = delete;

    bool ok() const { return fd_ >= 0; }
    bool read_at(uint64_t offset, void* dst, std::size_t n) override;

    // Total bytes read so far (for throughput stats).
    uint64_t bytes_read() const { return bytes_read_; }

private:
    int fd_ = -1;
    uint64_t pos_ = 0;
    uint64_t bytes_read_ = 0;
};

}  // namespace lfm
