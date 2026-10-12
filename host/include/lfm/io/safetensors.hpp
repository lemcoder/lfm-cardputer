#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace lfm::io {

// One entry from the header. `data` points into the mmap'd file, so a Tensor
// is valid only while its Safetensors object is alive.
struct Tensor {
  std::string name;   // owned copy: the parsed JSON is gone after the ctor
  std::string dtype;  // "BF16", "F32", ...
  std::array<std::size_t, 4> shape{};  // first `ndim` entries are used
  std::size_t ndim = 0;
  std::span<const std::byte> data;

  // Number of elements: product of the used shape entries.
  std::size_t numel() const {
    std::size_t n = 1;
    for (std::size_t i = 0; i < ndim; ++i) n *= shape[i];
    return n;
  }
};

class Safetensors {
public:
  explicit Safetensors(const char *path);
  ~Safetensors();
  Safetensors(const Safetensors &) = delete;
  Safetensors &operator=(const Safetensors &) = delete;

  bool ok() const;
  const Tensor *find(std::string_view name) const;

private:
  int fd_ = -1;
  void *map_ = nullptr; // MAP_FAILED handled in the ctor
  std::size_t size_ = 0;
  std::vector<Tensor> tensors_;
};

// Whole tensor as float32 (host convenience; weights are loaded once).
// Supports BF16 only for now: that's all LFM2.5 ships.
std::vector<float> to_f32(const Tensor &t);

}  // namespace lfm::io
