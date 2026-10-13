#include "lfm/draft.hpp"

#include <cstring>

#include "lfm/engine.hpp"

namespace lfm {

bool Drafter::set_ngram(const void* table, std::size_t size) {
    bigram_ = nullptr;
    tri_ = nullptr;
    if (table == nullptr || size < sizeof(NgramHeader)) return false;
    NgramHeader h;
    std::memcpy(&h, table, sizeof(h));
    if (std::memcmp(h.magic, kNgramMagic, sizeof(kNgramMagic)) != 0 || h.version != 1) return false;
    if (h.tri_slots == 0 || (h.tri_slots & (h.tri_slots - 1)) != 0) return false;
    const std::size_t need = sizeof(NgramHeader) + h.vocab * sizeof(uint16_t) + h.tri_slots * sizeof(NgramEntry);
    if (size < need) return false;
    const auto* base = static_cast<const std::byte*>(table);
    bigram_ = reinterpret_cast<const uint16_t*>(base + sizeof(NgramHeader));
    tri_ = reinterpret_cast<const NgramEntry*>(base + sizeof(NgramHeader) + h.vocab * sizeof(uint16_t));
    vocab_ = h.vocab;
    tri_slots_ = h.tri_slots;
    return true;
}

void Drafter::push(int32_t token) {
    if (n_ == cap_) {  // full: drop the oldest half (lookup only needs recent context)
        const int keep = cap_ / 2;
        std::memmove(hist_, hist_ + (cap_ - keep), static_cast<std::size_t>(keep) * sizeof(int32_t));
        n_ = keep;
    }
    hist_[n_++] = token;
}

int Drafter::draft(int32_t* out, int k) const {
    if (k <= 0) return 0;
    const int n = lookup(out, k);
    return n > 0 ? n : ngram(out, k);
}

// Most recent earlier occurrence of the last 2 tokens; propose what followed.
int Drafter::lookup(int32_t* out, int k) const {
    constexpr int kMatch = 2;
    if (n_ <= kMatch) return 0;
    const int32_t* tail = hist_ + n_ - kMatch;
    for (int start = n_ - kMatch - 1; start >= 0; --start) {
        if (hist_[start] == tail[0] && hist_[start + 1] == tail[1]) {
            int m = 0;
            for (int i = start + kMatch; i < n_ && m < k; ++i) out[m++] = hist_[i];
            return m;
        }
    }
    return 0;
}

int32_t Drafter::ngram_next(int32_t a, int32_t b) const {
    if (tri_ == nullptr || b < 0 || static_cast<uint32_t>(b) >= vocab_) return -1;
    if (a >= 0 && static_cast<uint32_t>(a) < vocab_) {
        const uint32_t key = (static_cast<uint32_t>(a) << 16) | static_cast<uint32_t>(b);
        for (uint32_t s = ngram_slot(key, tri_slots_), probes = 0; probes < tri_slots_; s = (s + 1) & (tri_slots_ - 1), ++probes) {
            const NgramEntry& e = tri_[s];
            if (e.next == kNgramNone) break;
            if (e.key == key) return e.next;
        }
    }
    const uint16_t bi = bigram_[b];
    return bi == kNgramNone ? -1 : bi;
}

// Chain n-gram predictions, each extending the context with the previous one.
int Drafter::ngram(int32_t* out, int k) const {
    if (tri_ == nullptr || n_ == 0) return 0;
    int32_t a = n_ >= 2 ? hist_[n_ - 2] : -1;
    int32_t b = hist_[n_ - 1];
    int m = 0;
    while (m < k) {
        const int32_t next = ngram_next(a, b);
        if (next < 0) break;
        out[m++] = next;
        a = b;
        b = next;
    }
    return m;
}

int speculative_step(Engine& eng, Drafter& drafter, int32_t fed, int max_draft, int32_t* out, SpecStats* stats) {
    int32_t tokens[Engine::kMaxBatch];
    int32_t preds[Engine::kMaxBatch];
    tokens[0] = fed;
    // Never draft past the context.
    int k = max_draft;
    if (k > Engine::kMaxBatch - 1) k = Engine::kMaxBatch - 1;
    if (k > eng.max_ctx() - eng.pos() - 1) k = eng.max_ctx() - eng.pos() - 1;
    const int n_draft = k > 0 ? drafter.draft(tokens + 1, k) : 0;

    const int32_t r = eng.verify({tokens, static_cast<std::size_t>(1 + n_draft)}, preds);
    if (r < 0) return r;
    int accepted = 0;
    while (accepted < n_draft && tokens[1 + accepted] == preds[accepted]) ++accepted;
    if (!eng.accept(1 + accepted)) return Engine::kIoError;

    int n_out = 0;
    for (int i = 0; i < accepted; ++i) out[n_out++] = tokens[1 + i];
    out[n_out++] = preds[accepted];
    for (int i = 0; i < n_out; ++i) drafter.push(out[i]);

    if (stats != nullptr) {
        ++stats->passes;
        stats->tokens += n_out;
        stats->drafted += n_draft;
        stats->accepted += accepted;
    }
    return n_out;
}

}  // namespace lfm
