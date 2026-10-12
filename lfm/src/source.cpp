#include "lfm/source.hpp"

#include <fcntl.h>
#include <unistd.h>

namespace lfm {

FileSource::FileSource(const char* path) {
    fd_ = open(path, O_RDONLY);
}

FileSource::~FileSource() {
    if (fd_ >= 0) close(fd_);
}

bool FileSource::read_at(uint64_t offset, void* dst, std::size_t n) {
    if (fd_ < 0) return false;
    if (offset != pos_) {
        if (lseek(fd_, static_cast<off_t>(offset), SEEK_SET) < 0) return false;
        pos_ = offset;
    }
    auto* out = static_cast<unsigned char*>(dst);
    std::size_t done = 0;
    while (done < n) {
        const ssize_t got = read(fd_, out + done, n - done);
        if (got <= 0) {
            pos_ = UINT64_MAX;  // unknown: force a seek next time
            return false;
        }
        done += static_cast<std::size_t>(got);
    }
    pos_ += n;
    bytes_read_ += n;
    return true;
}

}  // namespace lfm
