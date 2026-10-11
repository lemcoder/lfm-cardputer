#include "lfm/io/safetensors.hpp"

#include <fcntl.h>     // open, O_RDONLY
#include <sys/mman.h>  // mmap, munmap
#include <sys/stat.h>  // fstat
#include <unistd.h>    // close

#include <cstdint>
#include <cstring>  // memcpy

#include <nlohmann/json.hpp>

namespace lfm::io {

Safetensors::Safetensors(const char* path) {
    fd_ = open(path, O_RDONLY);
    if (fd_ < 0) {
        return;
    }

    struct stat s = {};

    int result = fstat(fd_, &s);
    if (result != 0) {
        return;
    }

    size_ = s.st_size;

    void* m = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (m == MAP_FAILED) {
        return;
    }
    map_ = m;

    const char* base = static_cast<const char*>(map_);

    std::uint64_t header_len = 0;

    memcpy(&header_len, base, 8);
    if (8 + header_len > size_) {
        return;
    }

    const char* json_begin = base + 8;
    const char* json_end = json_begin + header_len;
    nlohmann::json header = nlohmann::json::parse(json_begin, json_end);

    const std::byte* data_start = reinterpret_cast<const std::byte*>(json_end);

    for (const auto& [key, value] : header.items()) {
        if (key == "__metadata__") continue;

        Tensor t;
        t.name = key;
        t.dtype = value["dtype"];
        const auto& shape = value["shape"];
        t.ndim = shape.size();

        for (std::size_t i = 0; i < t.ndim; ++i) {
            t.shape[i] = shape[i];
        }

        std::size_t begin = value["data_offsets"][0];
        std::size_t end   = value["data_offsets"][1];
        t.data = {data_start + begin, end - begin};

        tensors_.push_back(t);
    }
}

Safetensors::~Safetensors() {
    if (map_ != nullptr) {
        munmap(map_, size_);
    }
    if (fd_ >= 0) close(fd_);
}

bool Safetensors::ok() const {
    return map_ != nullptr && !tensors_.empty();
}

const Tensor* Safetensors::find(std::string_view name) const {
    for (const Tensor& t : tensors_) {
        if (t.name == name) return &t;
    }
    return nullptr;
}

}  // namespace lfm::io
