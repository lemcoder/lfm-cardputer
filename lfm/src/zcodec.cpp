#include "lfm/zcodec.hpp"

namespace lfm {

namespace {

constexpr uint32_t kMask = (1u << ZTable::kBits) - 1;

// Bit reader over 16-bit words: at least kBits valid after need().
struct Bits {
    const uint8_t* p;
    uint32_t bb = 0;  // next bit in bit 0
    int nb = 0;       // valid bits, <= 32
    void need() {     // >= 11 bits (one code); adds a word when below
        if (nb < ZTable::kBits) {
            bb |= (static_cast<uint32_t>(p[0]) | static_cast<uint32_t>(p[1]) << 8) << nb;
            p += 2;
            nb += 16;
        }
    }
    uint8_t code(const ZTable& t) {
        need();
        const uint32_t e = t.e[bb & kMask];
        const int len = static_cast<int>(e >> 8);
        bb >>= len;
        nb -= len;
        return static_cast<uint8_t>(e);
    }
    uint8_t raw8() {
        need();
        const uint8_t v = static_cast<uint8_t>(bb);
        bb >>= 8;
        nb -= 8;
        return v;
    }
};

}  // namespace

bool z_decode(const ZTable& q, const ZTable& s, const uint8_t* in, std::size_t in_len, uint8_t* out,
              std::size_t out_len, uint32_t phase) {
    Bits r{in};
    uint8_t* o = out;
    uint8_t* const o_end = out + out_len;
    auto one = [&] {  // a single byte at any phase
        *o++ = phase == 0 ? r.raw8() : r.code(phase == 1 ? s : q);
        phase = phase == 17 ? 0 : phase + 1;
    };
    while (o < o_end && phase != 0) one();
    // Whole blocks: scale (raw low byte, coded high byte), 16 value bytes.
    while (o_end - o >= 18) {
        o[0] = r.raw8();
        o[1] = r.code(s);
        uint8_t* v = o + 2;
        for (int i = 0; i < 16; i += 4) {
            v[i] = r.code(q);
            v[i + 1] = r.code(q);
            v[i + 2] = r.code(q);
            v[i + 3] = r.code(q);
        }
        o += 18;
    }
    while (o < o_end) one();
    // Bits used must not exceed the input (the encoder pads with zeros).
    const std::size_t used = static_cast<std::size_t>(r.p - in) * 8 - static_cast<std::size_t>(r.nb);
    return used <= in_len * 8;
}

void ZDecoder::begin(const ZTable& q, const ZTable& s, const uint8_t* in, uint8_t* out, std::size_t out_len,
                     uint32_t phase) {
    q_ = &q;
    s_ = &s;
    p_ = in;
    bb_ = 0;
    nb_ = 0;
    o_ = out;
    o_end_ = out + out_len;
    phase_ = phase;
}

bool ZDecoder::step(const uint8_t* in_end, bool complete) {
    Bits r{p_, bb_, nb_};
    uint8_t* o = o_;
    // A block takes at most 8 + 17 * 10 = 178 bits (23 bytes); the reader
    // loads ahead by up to 2 bytes. Keep a margin while input still arrives.
    constexpr std::ptrdiff_t kBlockBytes = 28;
    auto have = [&](std::ptrdiff_t bytes) { return complete || in_end - r.p >= bytes; };
    auto one = [&] {
        *o++ = phase_ == 0 ? r.raw8() : r.code(phase_ == 1 ? *s_ : *q_);
        phase_ = phase_ == 17 ? 0 : phase_ + 1;
    };
    while (o < o_end_ && phase_ != 0 && have(4)) one();
    if (phase_ == 0) {
        while (o_end_ - o >= 18 && have(kBlockBytes)) {
            o[0] = r.raw8();
            o[1] = r.code(*s_);
            uint8_t* v = o + 2;
            for (int i = 0; i < 16; i += 4) {
                v[i] = r.code(*q_);
                v[i + 1] = r.code(*q_);
                v[i + 2] = r.code(*q_);
                v[i + 3] = r.code(*q_);
            }
            o += 18;
        }
        while (o < o_end_ && o_end_ - o < 18 && have(4)) one();
    }
    p_ = r.p;
    bb_ = r.bb;
    nb_ = r.nb;
    o_ = o;
    return o_ == o_end_;
}

}  // namespace lfm
