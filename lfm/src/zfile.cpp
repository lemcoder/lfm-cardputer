#include "lfm/zfile.hpp"

#include <cstdlib>
#include <cstring>

namespace lfm {

namespace {
constexpr char kZMagic[8] = {'L', 'F', 'M', 'Z', '\0', '\0', '\0', '\0'};
}  // namespace

ZIndex::~ZIndex() { std::free(tensors_); }

bool ZIndex::open(Source& z) {
    if (!z.read_at(0, &h_, sizeof(h_)) || std::memcmp(h_.magic, kZMagic, sizeof(kZMagic)) != 0 || h_.version != 1 ||
        h_.n_tensors == 0 || h_.chunk == 0) {
        return false;
    }
    if (!z.read_at(sizeof(h_), tables_, sizeof(tables_))) return false;
    std::free(tensors_);
    tensors_ = static_cast<ZTensor*>(std::malloc(h_.n_tensors * sizeof(ZTensor)));
    const uint64_t t_at = sizeof(h_) + sizeof(tables_);
    if (tensors_ == nullptr || !z.read_at(t_at, tensors_, h_.n_tensors * sizeof(ZTensor))) return false;
    chunks_at_ = t_at + h_.n_tensors * sizeof(ZTensor);
    return true;
}

const ZTensor* ZIndex::find(uint64_t v) const {
    uint32_t lo = 0, hi = h_.n_tensors;  // last tensor with v_off <= v
    while (hi - lo > 1) {
        const uint32_t mid = (lo + hi) / 2;
        (tensors_[mid].v_off <= v ? lo : hi) = mid;
    }
    const ZTensor& t = tensors_[lo];
    return v >= t.v_off && v < t.v_off + t.v_len ? &t : nullptr;
}

uint64_t ZIndex::next_start(uint64_t v) const {
    uint32_t lo = 0, hi = h_.n_tensors;  // first tensor with v_off > v
    while (lo < hi) {
        const uint32_t mid = (lo + hi) / 2;
        if (tensors_[mid].v_off <= v) lo = mid + 1;
        else hi = mid;
    }
    return lo < h_.n_tensors ? tensors_[lo].v_off : UINT64_MAX;
}

bool ZIndex::chunks(Source& z, const ZTensor& t, uint32_t k, uint32_t count, ZChunk* out) const {
    if (count == 0 || k + count > t.n_chunks) return false;
    return z.read_at(chunks_at_ + (static_cast<uint64_t>(t.first_chunk) + k) * sizeof(ZChunk), out,
                     count * sizeof(ZChunk));
}

ZSource::ZSource(Source& raw, Source& z) : raw_(raw), z_(z) {
    ok_ = index_.open(z_);
    if (ok_) {
        comp_ = static_cast<uint8_t*>(std::malloc(index_.header().chunk + 8));
        plain_ = static_cast<uint8_t*>(std::malloc(index_.header().chunk));
        ok_ = comp_ != nullptr && plain_ != nullptr;
    }
}

ZSource::~ZSource() {
    std::free(comp_);
    std::free(plain_);
}

bool ZSource::read_at(uint64_t offset, void* dst, std::size_t n) {
    auto* out = static_cast<uint8_t*>(dst);
    while (n > 0) {
        const ZTensor* t = ok_ ? index_.find(offset) : nullptr;
        if (t == nullptr) {
            // Raw up to the next compressed matrix (or all of it).
            const uint64_t next = ok_ ? index_.next_start(offset) : UINT64_MAX;
            const std::size_t take = next - offset < n ? static_cast<std::size_t>(next - offset) : n;
            if (!raw_.read_at(offset, out, take)) return false;
            offset += take;
            out += take;
            n -= take;
            continue;
        }
        const uint32_t chunk = index_.header().chunk;
        const uint64_t rel = offset - t->v_off;
        const uint32_t k = static_cast<uint32_t>(rel / chunk);
        const uint32_t c_begin = k * chunk;
        const uint32_t c_len = std::min<uint32_t>(chunk, t->v_len - c_begin);
        ZChunk c;
        if (!index_.chunk(z_, *t, k, &c)) return false;
        const uint32_t stored = c.len & ~ZChunk::kRaw;
        std::memset(comp_ + stored, 0, 8);
        if (!z_.read_at(c.off, comp_, stored)) return false;
        if (c.len & ZChunk::kRaw) {
            std::memcpy(plain_, comp_, c_len);
        } else if (!z_decode(index_.q(), index_.s(), comp_, stored, plain_, c_len, c_begin % 18)) {
            return false;
        }
        const std::size_t from = static_cast<std::size_t>(rel - c_begin);
        const std::size_t take = std::min<std::size_t>(n, c_len - from);
        std::memcpy(out, plain_ + from, take);
        offset += take;
        out += take;
        n -= take;
    }
    return true;
}

}  // namespace lfm
