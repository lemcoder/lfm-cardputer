#pragma once
// Lossless entropy coding of Q4_0 matrices (encoder: tools/zcodec.py).
//
// Q4 weights compress little (each 4-bit value carries ~3.7 bits of entropy),
// but the f16 block scales are very predictable: overall a pass reads ~10%
// fewer bytes. Byte roles follow the 18-byte Q4_0 block, counted from the
// start of the matrix ("phase" 0..17):
//   phase 0      scale, low byte: stored raw (8 bits)
//   phase 1      scale, high byte: Huffman table S
//   phase 2..17  two 4-bit values: Huffman table Q
// A matrix is cut into chunks of a fixed number of decoded bytes (the
// engine's read block), each its own bitstream, so any chunk decodes alone.
//
// Codes are at most 10 bits, read LSB first: the next 10 bits index the
// table directly (1024 entries, 2 KB per table).

#include <cstddef>
#include <cstdint>

namespace lfm {

struct ZTable {
    static constexpr int kBits = 10;
    uint16_t e[1 << kBits];  // symbol | code length << 8
};

// Decodes out_len bytes; phase = (matrix byte offset of out[0]) % 18.
// False if the input runs out early. Reads up to 3 bytes past in + in_len
// (they must be addressable; their values don't matter). in must be 2-byte aligned.
bool z_decode(const ZTable& q, const ZTable& s, const uint8_t* in, std::size_t in_len, uint8_t* out,
              std::size_t out_len, uint32_t phase);

// The same decoding, resumable: for input that is still arriving (the
// device decodes a block while the SD card's DMA is writing it).
class ZDecoder {
public:
    void begin(const ZTable& q, const ZTable& s, const uint8_t* in, uint8_t* out, std::size_t out_len, uint32_t phase);
    // Decodes as far as the input up to in_end allows (to the end if
    // complete: then up to 3 bytes past in_end are read). True once all
    // out_len bytes are written.
    bool step(const uint8_t* in_end, bool complete);
    bool done() const { return o_ == o_end_; }

private:
    const ZTable* q_ = nullptr;
    const ZTable* s_ = nullptr;
    const uint8_t* p_ = nullptr;
    uint32_t bb_ = 0;
    int nb_ = 0;
    uint8_t* o_ = nullptr;
    uint8_t* o_end_ = nullptr;
    uint32_t phase_ = 0;
};

}  // namespace lfm
