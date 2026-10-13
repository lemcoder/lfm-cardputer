#include "lfm/source.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <chrono>

namespace lfm {

namespace {

// Adds the lifetime of this object to `total` (seconds).
class IoTimer {
public:
    explicit IoTimer(double& total) : total_(total), t0_(std::chrono::steady_clock::now()) {}
    ~IoTimer() { total_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count(); }

private:
    double& total_;
    std::chrono::steady_clock::time_point t0_;
};

}  // namespace

FileSource::FileSource(const char* path, Mode mode) {
    fd_ = mode == Mode::Read ? open(path, O_RDONLY) : open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
}

FileSource::~FileSource() {
    if (fd_ >= 0) close(fd_);
}

bool FileSource::seek(uint64_t offset) {
    if (offset == pos_) return true;
    if (lseek(fd_, static_cast<off_t>(offset), SEEK_SET) < 0) {
        pos_ = UINT64_MAX;  // unknown: force a seek next time
        return false;
    }
    pos_ = offset;
    return true;
}

bool FileSource::read_at(uint64_t offset, void* dst, std::size_t n) {
    IoTimer timer(io_seconds_);
    ++reads_;
    if (fd_ < 0 || !seek(offset)) return false;
    auto* out = static_cast<unsigned char*>(dst);
    std::size_t done = 0;
    while (done < n) {
        const ssize_t got = read(fd_, out + done, n - done);
        if (got <= 0) {
            pos_ = UINT64_MAX;
            return false;
        }
        done += static_cast<std::size_t>(got);
    }
    pos_ += n;
    bytes_read_ += n;
    return true;
}

bool FileSource::write_at(uint64_t offset, const void* src, std::size_t n) {
    IoTimer timer(io_seconds_);
    IoTimer wtimer(write_seconds_);
    ++writes_;
    if (fd_ < 0 || !seek(offset)) return false;
    const auto* in = static_cast<const unsigned char*>(src);
    std::size_t done = 0;
    while (done < n) {
        const ssize_t put = write(fd_, in + done, n - done);
        if (put <= 0) {
            pos_ = UINT64_MAX;
            return false;
        }
        done += static_cast<std::size_t>(put);
    }
    pos_ += n;
    return true;
}

}  // namespace lfm
