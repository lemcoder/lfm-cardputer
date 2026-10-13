#pragma once
// Compressed layer matrices next to the .lfm (tools/zcodec.py pack).
//
// The .lfz file holds every Q4_0 layer matrix of the .lfm, entropy coded
// (lfm/zcodec.hpp) in chunks of `chunk` decoded bytes counted from the
// matrix start (the engine reads matrices in such blocks). Everything else
// is still read from the .lfm. Layout (little-endian):
//   ZFileHeader
//   ZTable q, s                        decode tables
//   ZTensor[n_tensors]                 sorted by v_off
//   ZChunk[n_chunks]                   per matrix, in order
//   chunk data (from a 512-byte boundary)

#include <cstddef>
#include <cstdint>
#include <span>

#include "lfm/source.hpp"
#include "lfm/zcodec.hpp"

namespace lfm {

struct ZFileHeader {
    char magic[8];  // "LFMZ\0\0\0\0"
    uint32_t version;
    uint32_t chunk;  // decoded bytes per chunk (the last one of a matrix may be shorter)
    uint32_t slack;  // spare bytes a buffer needs past `chunk` to decode in place
    uint32_t n_tensors;
    uint32_t n_chunks;
    uint32_t reserved;
};
struct ZTensor {
    uint64_t v_off;  // offset of the matrix in the .lfm
    uint32_t v_len;
    uint16_t first_chunk;
    uint16_t n_chunks;
};
struct ZChunk {
    static constexpr uint32_t kRaw = 1u << 31;  // stored uncompressed
    uint32_t off;  // in the .lfz
    uint32_t len;  // bytes in the .lfz (multiple of 4) | kRaw
};
static_assert(sizeof(ZFileHeader) == 32 && sizeof(ZTensor) == 16 && sizeof(ZChunk) == 8);

class ZIndex {
public:
    ZIndex() = default;
    ~ZIndex();
    ZIndex(const ZIndex&) = delete;
    ZIndex& operator=(const ZIndex&) = delete;

    bool open(Source& z);  // header, tables and the matrix list (~1.3 KB of RAM)
    const ZFileHeader& header() const { return h_; }
    const ZTable& q() const { return tables_[0]; }
    const ZTable& s() const { return tables_[1]; }

    // The compressed matrix containing .lfm offset v, or nullptr.
    const ZTensor* find(uint64_t v) const;
    // Start of the first compressed matrix after v (UINT64_MAX if none).
    uint64_t next_start(uint64_t v) const;
    // Chunks k .. k+count-1 of a matrix (read from the file, 8 bytes each).
    bool chunks(Source& z, const ZTensor& t, uint32_t k, uint32_t count, ZChunk* out) const;
    bool chunk(Source& z, const ZTensor& t, uint32_t k, ZChunk* out) const { return chunks(z, t, k, 1, out); }

private:
    ZFileHeader h_{};
    ZTable tables_[2];
    ZTensor* tensors_ = nullptr;
    uint64_t chunks_at_ = 0;  // file offset of ZChunk[0]
};

// A Source that reads the .lfm through raw, except its compressed matrices,
// which it decodes from the .lfz (any range; it keeps a chunk-sized buffer
// pair). For the host and tests; the device decodes in AsyncFile.
class ZSource final : public Source {
public:
    ZSource(Source& raw, Source& z);
    ~ZSource() override;
    bool ok() const { return ok_; }
    bool read_at(uint64_t offset, void* dst, std::size_t n) override;

private:
    Source& raw_;
    Source& z_;
    ZIndex index_;
    bool ok_ = false;
    uint8_t* comp_ = nullptr;
    uint8_t* plain_ = nullptr;
};

}  // namespace lfm
